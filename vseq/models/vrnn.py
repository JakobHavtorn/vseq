from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.jit as jit

from torchtyping import TensorType
from tqdm import tqdm

from vseq.evaluation import LossMetric, LLMetric, KLMetric, PerplexityMetric, BitsPerDimMetric, LatestMeanMetric
from vseq.models import BaseModel
from vseq.modules.convenience import Permute, View
from vseq.modules.distributions import (
    DiscretizedLogisticDense,
    DiscretizedLogisticMixtureDense,
    GaussianDense,
    BernoulliDense,
    CategoricalDense,
    PolarCoordinatesSpectrogram,
)
from vseq.modules.dropout import WordDropout
from vseq.utils.operations import sequence_mask
from vseq.utils.variational import discount_free_nats, kl_divergence_gaussian, rsample_gaussian
from vseq.utils.dict import list_of_dict_to_dict_of_list


class VRNNCell(jit.ScriptModule):
    def __init__(self, x_dim: int, h_dim: int, z_dim: int, condition_h_on_x: bool = True):
        """Variational Recurrent Neural Network (VRNN) cell from [1].

        Uses unimodal isotropic gaussian distributions for inference, prior, and generative models.

        Args:
            x_dim (int): Input space size
            h_dim (int): Hidden space (GRU) size
            z_dim (int): Stochastic latent variable size
            condition_h_on_x (bool): If True, condition h on x observation in inference and generation.

        [1] https://arxiv.org/abs/1506.02216
        """
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.condition_h_on_x = condition_h_on_x

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            GaussianDense(h_dim, z_dim),
        )

        self.encoder = nn.Sequential(
            nn.Linear(x_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            GaussianDense(h_dim, z_dim),
        )

        gru_in_dim = x_dim + h_dim if self.condition_h_on_x else h_dim
        self.gru_cell = nn.GRUCell(gru_in_dim, h_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.orthogonal_(self.gru_cell.weight_hh)

    def forward(self, x: TensorType["B", "x_dim"], h: TensorType["B", "h_dim"]):
        # prior p(z)
        prior_mu, prior_sd = self.prior(h)

        # encoder q(z|x)
        enc_mu, enc_sd = self.encoder(torch.cat([h, x], -1))

        # sampling and reparameterization
        z = rsample_gaussian(enc_mu, enc_sd)

        # z features
        phi_z = self.phi_z(z)

        # gru cell
        if self.condition_h_on_x:
            h = self.gru_cell(torch.cat([x, phi_z], -1), h)
        else:
            h = self.gru_cell(phi_z, h)

        outputs = SimpleNamespace(z=z, enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=prior_mu, prior_sd=prior_sd)
        return h, phi_z, enc_mu, enc_sd, prior_mu, prior_sd, outputs

    def generate(self, x: TensorType["B", "x_dim"], h: TensorType["B", "h_dim"]):
        # prior p(z)
        prior_mu, prior_sd = self.prior(h)

        # sampling and reparameterization
        z = rsample_gaussian(prior_mu, prior_sd)

        # z features
        phi_z = self.phi_z(z)

        # gru cell
        if self.condition_h_on_x:
            h = self.gru_cell(torch.cat([x, phi_z], -1), h)
        else:
            h = self.gru_cell(phi_z, h)

        return h, phi_z, SimpleNamespace(z=z, prior_mu=prior_mu, prior_sd=prior_sd)


class VRNN(nn.Module):
    def __init__(
        self,
        phi_x: nn.Module,
        likelihood: nn.Module,
        x_dim: int,
        h_dim: int,
        z_dim: int,
        o_dim: int,
        condition_h_on_x: bool = True,
        condition_x_on_h: bool = True,
        word_dropout: float = 0,
        dropout: float = 0,
    ):
        """Variational Recurrent Neural Network (VRNN) from [1].

        Uses unimodal isotropic gaussian distributions for inference, prior, and generative models.

            ┌───┐       ┌───┐       ┌───┐          ┌───┐       ┌───┐       ┌───┐
            │h_1├─┬────►│h_2├─┬────►│h_3│          │h_1├─┬────►│h_2├─┬────►│h_3│
            └─┬─┘ │     └─┬─┘ │     └─┬─┘          └─┬─┘ │     └─┬─┘ │     └─┬─┘
              │   │       │   │       │              │   │       │   │       │
              │   │       │   │       │              │   │   ┌───┤   │   ┌───┤
              │   │       │   │       │              │   │   │   │   │   │   │
              ▼   │       ▼   │       ▼              ▼   │   │   ▼   │   │   ▼
            ┌───┐ │     ┌───┐ │     ┌───┐          ┌───┐ │   │ ┌───┐ │   │ ┌───┐
            │z_1├─┤     │z_2├─┤     │z_3│          │z_1├─┤   │ │z_2├─┤   │ │z_3│
            └───┘ │     └───┘ │     └───┘          └─┬─┘ │   │ └─┬─┘ │   │ └─┬─┘
              ▲   │       ▲   │       ▲              │   │   │   │   │   │   │
              │   │       │   │       │              │   │   │   │   │   │   │
              │   │       │   │       │              │   │   │   │   │   │   │
              │   │       │   │       │              ▼   │   │   ▼   │   │   ▼
            ┌─┴─┐ │     ┌─┴─┐ │     ┌─┴─┐          ┌───┐ │   │ ┌───┐ │   │ ┌───┐
            │x_1├─┘     │x_2├─┘     │x_3│          │x_1├─┘   └►│x_2├─┘   └►│x_3│
            └───┘       └───┘       └───┘          └───┘       └───┘       └───┘

                   INFERENCE MODEL                       GENERATIVE MODEL

        Args:
            phi_x (nn.Module): Input transformation
            x_dim (int): Input space size
            h_dim (int): Hidden space (GRU) size
            z_dim (int): Stochastic latent variable size
            o_dim (int): Output space size
            condition_h_on_x (bool): If True, condition h on x in inference and generation (parameter sharing).
            condition_x_on_h (bool): If True, condition x on h in generation.

        [1] https://arxiv.org/abs/1506.02216
        """
        super().__init__()

        self.phi_x = phi_x
        self.likelihood = likelihood

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.o_dim = o_dim
        self.condition_h_on_x = condition_h_on_x
        self.condition_x_on_h = condition_x_on_h

        self.vrnn_cell = VRNNCell(x_dim=x_dim, h_dim=h_dim, z_dim=z_dim, condition_h_on_x=condition_h_on_x)

        decoder_in_dim = 2 * h_dim if condition_x_on_h else h_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.likelihood = likelihood
        self.word_dropout = WordDropout(word_dropout) if word_dropout else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def compute_elbo(
        self,
        y: TensorType["B", "T"],
        logits: TensorType["B", "T", "D"],
        kld_twise: TensorType["B", "T", "latent_size"],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
    ):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""

        seq_mask = sequence_mask(x_sl, dtype=float, device=y.device)

        log_prob_twise = self.likelihood.log_prob(y, logits) * seq_mask
        log_prob = log_prob_twise.view(y.size(0), -1).sum(1)  # (B,)

        kld = (kld_twise * seq_mask.unsqueeze(-1)).sum((1, 2))  # (B,)
        elbo = log_prob - kld  # (B,)

        kld_twise_fn = discount_free_nats(kld_twise, free_nats, shared_dims=-1)
        kld = (kld_twise_fn * seq_mask.unsqueeze(-1)).sum((1, 2))  # (B,)
        loss = -(log_prob - beta * kld).sum() / x_sl.sum()  # (1,)

        return loss, elbo, log_prob, kld, seq_mask

    def forward(
        self,
        x: TensorType["B", "T", "x_dim"],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h0: TensorType["B", "h_dim"] = None,
    ):
        batch_size, timesteps = x.size(0), x.size(1)

        all_h = []
        all_phi_z = []
        all_enc_mu = []
        all_enc_sd = []
        all_prior_mu = []
        all_prior_sd = []
        all_outputs = []

        # target
        y = x.clone().detach()

        # dropout
        x = self.word_dropout(x) if self.word_dropout else x

        # x features
        phi_x = self.phi_x(x)
        phi_x = phi_x.unbind(1)

        # initial h
        h = torch.zeros(batch_size, self.h_dim, device=x.device) if h0 is None else h0
        all_h.append(h)

        for t in range(timesteps):
            h, phi_z, enc_mu, enc_sd, prior_mu, prior_sd, outputs = self.vrnn_cell(phi_x[t], h)

            all_h.append(h)
            all_phi_z.append(phi_z)
            all_enc_mu.append(enc_mu)
            all_enc_sd.append(enc_sd)
            all_prior_mu.append(prior_mu)
            all_prior_sd.append(prior_sd)
            all_outputs.append(outputs)

        all_h.pop()  # Include initial and not last

        # output distribution
        phi_z = torch.stack(all_phi_z, dim=1)
        if self.condition_x_on_h:
            h = torch.stack(all_h, dim=1)
            dec = self.decoder(torch.cat([phi_z, h], -1))
        else:
            dec = self.decoder(phi_z)

        dec = self.dropout(dec) if self.dropout is not None else dec

        logits = self.likelihood(dec)  # (B, T, D)

        # kl divergence, elbo and loss
        enc_mu = torch.stack(all_enc_mu, dim=1)
        enc_sd = torch.stack(all_enc_sd, dim=1)
        prior_mu = torch.stack(all_prior_mu, dim=1)
        prior_sd = torch.stack(all_prior_sd, dim=1)
        kld = kl_divergence_gaussian(enc_mu, enc_sd, prior_mu, prior_sd)
        # kld = (kld * 0).detach()

        loss, elbo, log_prob, kl, seq_mask = self.compute_elbo(y, logits, kld, x_sl, beta, free_nats)

        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo"),
            LLMetric(log_prob, name="rec"),
            KLMetric(kl),
            BitsPerDimMetric(elbo, reduce_by=x_sl),
            LatestMeanMetric(beta, name="beta"),
            LatestMeanMetric(free_nats, name="free_nats"),
        ]
        outputs = SimpleNamespace(elbo=elbo, log_prob=log_prob, kl=kl, y=y, logits=logits, seq_mask=seq_mask)
        return loss, metrics, outputs

    def generate(
        self,
        x: TensorType["B", "x_dim"],
        h0: TensorType["B", "h_dim"] = None,
        n_samples: int = 1,
        max_timesteps: int = 100,
        stop_value: float = None,
        use_mode: bool = False,
    ):

        if x.size(0) > 1:
            assert x.size(0) == n_samples
        else:
            x = x.repeat(n_samples, *[1] * (x.ndim - 1))  # Repeat along batch

        # TODO If multiple timesteps in x, run a forward pass to get conditional initial h (assert h is None)

        all_outputs = []
        all_x = [x]
        x_sl = torch.ones(n_samples, dtype=torch.int)

        # initial h
        h = torch.zeros(n_samples, self.h_dim, device=x.device) if h0 is None else h0

        seq_active = torch.ones(n_samples, dtype=torch.int)
        all_ended, t = False, 0  # Used to condition while loop

        pbar = tqdm(total=max_timesteps)
        while not all_ended and t < max_timesteps:
            phi_x = self.phi_x(x)

            h, phi_z, outputs = self.vrnn_cell.generate(phi_x, h)

            if self.condition_x_on_h:
                dec = self.decoder(torch.cat([phi_z, h], -1))
            else:
                dec = self.decoder(phi_z)
            logits = self.likelihood(dec)  # (B, T, D)

            if use_mode:
                x = self.likelihood.mode(logits)
            else:
                x = self.likelihood.sample(logits)

            all_x.append(x)
            all_outputs.append(outputs)

            # Update sequence length
            x_sl += seq_active
            seq_ending = x == stop_value  # (,), (B,) or (B, D*)
            if isinstance(seq_ending, torch.Tensor):
                seq_ending = seq_ending.all(*list(range(1, seq_ending.ndim))) if seq_ending.ndim > 1 else seq_ending
                seq_ending = seq_ending.to(int).cpu()
            else:
                seq_ending = int(seq_ending)
            seq_active *= 1 - seq_ending

            # Update loop conditions
            t += 1
            all_ended = torch.all(1 - seq_active).item()
            pbar.update(1)

        pbar.close()
        x = torch.stack(all_x, dim=1)

        outputs = SimpleNamespace(**list_of_dict_to_dict_of_list([vars(ns) for ns in all_outputs]))
        return (x, x_sl), outputs


class VRNNLM(BaseModel):
    def __init__(
        self,
        num_embeddings: int,
        delimiter_token_idx: int,
        embedding_dim: int = 300,
        hidden_size: int = 256,
        latent_size: int = 64,
        condition_h_on_x: bool = True,
        condition_x_on_h: bool = True,
        word_dropout: float = 0,
        dropout: float = 0,
    ):
        """A VRNN for language modelling.

        Notes:
        - WordDropout reduces overfitting but does not improve LL much
        - Removing the start token reduces LL a bit
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.word_dropout = word_dropout
        self.dropout = dropout
        self.delimiter_token_idx = delimiter_token_idx
        self.condition_h_on_x = condition_h_on_x
        self.condition_x_on_h = condition_x_on_h

        embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        categorical = CategoricalDense(hidden_size, num_embeddings)

        self.vrnn = VRNN(
            phi_x=embedding,
            likelihood=categorical,
            x_dim=embedding_dim,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=num_embeddings,
            condition_h_on_x=condition_h_on_x,
            condition_x_on_h=condition_x_on_h,
            word_dropout=word_dropout,
            dropout=dropout,
        )

    def forward(
        self,
        x: TensorType["B", "T", int],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h0: TensorType["B", "h_dim"] = None,
    ):
        return self.vrnn(x, x_sl, beta, free_nats, h0)

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = True,
        x: TensorType["B", "x_dim"] = None,
        h0: TensorType["B", "h_dim"] = None,
    ):
        x = torch.full([n_samples], self.delimiter_token_idx, device=self.device) if x is None else x
        return self.vrnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=self.delimiter_token_idx,
            use_mode=use_mode,
            x=x,
            h0=h0,
        )


class VRNN_MIDI(BaseModel):
    def __init__(
        self,
        input_size: int = 88,
        hidden_size: int = 256,
        latent_size: int = 64,
        condition_h_on_x: bool = True,
        condition_x_on_h: bool = True,
    ):
        """A VRNN for modelling the MIDI music dataset."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.condition_h_on_x = condition_h_on_x
        self.condition_x_on_h = condition_x_on_h

        embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        bernoulli = BernoulliDense(hidden_size, input_size)

        self.vrnn = VRNN(
            phi_x=embedding,
            likelihood=bernoulli,
            x_dim=hidden_size,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=input_size,
            condition_h_on_x=condition_h_on_x,
            condition_x_on_h=condition_x_on_h,
        )

    def forward(
        self,
        x: TensorType["B", "T", "D", float],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h0: TensorType["B", "h_dim"] = None,
    ):
        return self.vrnn(x, x_sl, beta, free_nats, h0)

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: TensorType["B", "x_dim"] = None,
        h0: TensorType["B", "h_dim"] = None,
    ):
        x = torch.zeros(n_samples, self.input_size, device=self.device) if x is None else x
        return self.vrnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=None,
            use_mode=use_mode,
            x=x,
            h0=h0,
        )


class VRNNAudioDML(BaseModel):
    def __init__(
        self,
        input_size: int = 200,
        hidden_size: int = 256,
        latent_size: int = 64,
        condition_h_on_x: bool = True,
        condition_x_on_h: bool = True,
        num_mix: int = 10,
        num_bins: int = 256,
    ):
        """A VRNN for modelling audio waveforms."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.condition_h_on_x = condition_h_on_x
        self.condition_x_on_h = condition_x_on_h
        self.num_mix = num_mix
        self.num_bins = num_bins

        embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        likelihood = DiscretizedLogisticMixtureDense(
            x_dim=hidden_size,
            y_dim=input_size,
            num_mix=num_mix,
            num_bins=num_bins,
            reduce_dim=-1,
        )
        # likelihood = DiscretizedLogisticDense(
        #     x_dim=hidden_size,
        #     y_dim=input_size,
        #     num_bins=num_bins,
        #     reduce_dim=-1,
        # )
        self.vrnn = VRNN(
            phi_x=embedding,
            likelihood=likelihood,
            x_dim=hidden_size,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=input_size,
            condition_h_on_x=condition_h_on_x,
            condition_x_on_h=condition_x_on_h,
        )

    def forward(
        self,
        x: TensorType["B", "T", "D", float],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h0: TensorType["B", "h_dim"] = None,
    ):
        loss, metrics, outputs = self.vrnn(x, x_sl, beta, free_nats, h0)
        outputs.x_hat = self.vrnn.likelihood.sample(outputs.logits)
        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: TensorType["B", "x_dim"] = None,
        h0: TensorType["B", "h_dim"] = None,
    ):
        x = torch.zeros(n_samples, self.input_size, device=self.device) if x is None else x
        return self.vrnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=None,
            use_mode=use_mode,
            x=x,
            h0=h0,
        )


class VRNNAudioGauss(BaseModel):
    def __init__(
        self,
        input_size: int = 200,
        hidden_size: int = 256,
        latent_size: int = 64,
        condition_h_on_x: bool = True,
        condition_x_on_h: bool = True,
    ):
        """A VRNN for modelling audio waveforms."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.condition_h_on_x = condition_h_on_x
        self.condition_x_on_h = condition_x_on_h

        embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        likelihood = GaussianDense(hidden_size, input_size, reduce_dim=-1)

        self.vrnn = VRNN(
            phi_x=embedding,
            likelihood=likelihood,
            x_dim=hidden_size,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=input_size,
            condition_h_on_x=condition_h_on_x,
            condition_x_on_h=condition_x_on_h,
        )

    def forward(
        self,
        x: TensorType["B", "T", float],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h0: TensorType["B", "h_dim"] = None,
    ):
        loss, metrics, outputs = self.vrnn(x, x_sl, beta, free_nats, h0)

        for metric_name in ["BitsPerDimMetric", "PerplexityMetric"]:
            bpd_metric_idx = [i for i, metric in enumerate(metrics) if metric.__class__.__name__ == metric_name][0]
            del metrics[bpd_metric_idx]

        seq_mask = outputs.seq_mask.to(bool)
        mse = ((outputs.y[seq_mask, :] - outputs.logits[0][seq_mask, :]) ** 2).mean().sqrt().item()
        std = outputs.logits[1][seq_mask, :].mean().item()

        metrics.extend(
            [
                LatestMeanMetric(mse, name="mse"),
                LatestMeanMetric(std, name="stddev"),
            ]
        )

        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: TensorType["B", "x_dim"] = None,
        h0: TensorType["B", "h_dim"] = None,
    ):
        x = torch.zeros(n_samples, self.input_size, device=self.device) if x is None else x
        return self.vrnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=None,
            use_mode=use_mode,
            x=x,
            h0=h0,
        )


class VRNNAudioSpec(BaseModel):
    def __init__(
        self,
        n_fft: int = 320,
        hop_length: int = 320,
        win_length: int = 320,
        window: Optional[torch.Tensor] = None,
        hidden_size: int = 256,
        latent_size: int = 64,
        condition_h_on_x: bool = True,
        condition_x_on_h: bool = True,
    ):
        """A VRNN for modelling audio waveforms."""
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.condition_h_on_x = condition_h_on_x
        self.condition_x_on_h = condition_x_on_h

        self.stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

        embedding = nn.Sequential(
            Permute(3, 1, 2),
            View(-1, 2 * self.stft.out_features),
            nn.Linear(2 * self.stft.out_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        likelihood = PolarCoordinatesSpectrogram(hidden_size, self.stft.out_features, num_mix=10, num_bins=256)

        self.vrnn = VRNN(
            phi_x=embedding,
            likelihood=likelihood,
            x_dim=hidden_size,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=n_fft,
            condition_h_on_x=condition_h_on_x,
            condition_x_on_h=condition_x_on_h,
        )

    def forward(
        self,
        x: TensorType["B", "T", float],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h0: TensorType["B", "h_dim"] = None,
        return_audio: bool = False,
    ):
        x = self.stft(x)
        # TODO Rescale stft r part to [-1, 1]
        # TODO Recompute sequence lengths
        loss, metrics, outputs = self.vrnn(x, x_sl, beta, free_nats, h0)

        import IPython

        IPython.embed(using=False)

        if return_audio:
            audio = self.istft(outputs.x_hat)
            outputs.audio = audio

        for metric_name in ["BitsPerDimMetric", "PerplexityMetric"]:
            bpd_metric_idx = [i for i, metric in enumerate(metrics) if metric.__class__.__name__ == metric_name][0]
            del metrics[bpd_metric_idx]

        # seq_mask = outputs.seq_mask.to(bool)
        # mse = ((outputs.y[seq_mask, :] - outputs.logits[0][seq_mask, :]) ** 2).mean().sqrt().item()
        # std = outputs.logits[1][seq_mask, :].mean().item()
        # bpd = outputs.elbo.cpu() - x_sl * self.n_fft * math.log(2 ** 8)

        # metrics.extend(
        #     [
        #         BitsPerDimMetric(bpd, reduce_by=x_sl),
        #         LatestMeanMetric(mse, name="mse"),
        #         LatestMeanMetric(std, name="stddev"),
        #     ]
        # )

        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: TensorType["B", "x_dim"] = None,
        h0: TensorType["B", "h_dim"] = None,
    ):
        x = torch.zeros(n_samples, self.stft.out_features, device=self.device) if x is None else x
        return self.vrnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=None,
            use_mode=use_mode,
            x=x,
            h0=h0,
        )
