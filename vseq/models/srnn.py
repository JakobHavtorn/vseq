from types import SimpleNamespace
from vseq.modules.distributions import DiscretizedLogisticDense

import torch
import torch.nn as nn
import torch.nn.init as init

from torchtyping import TensorType
from tqdm import tqdm

from vseq.evaluation.metrics import BitsPerDimMetric, KLMetric, LLMetric, LatestMeanMetric, LossMetric, PerplexityMetric
from vseq.models import BaseModel
from vseq.modules import CategoricalDense, GaussianDense, WordDropout
from vseq.utils.variational import discount_free_nats, kl_divergence_gaussian, rsample_gaussian
from vseq.utils.operations import sequence_mask, reverse_sequences


class SRNN(nn.Module):
    def __init__(
        self,
        x_embedding,
        likelihood,
        x_dim,
        h_dim,
        z_dim,
        o_dim,
        dropout: float = 0,
        word_dropout: float = 0,
        num_layers: int = 1,
        residual_posterior: bool = False,
    ):
        """Stochastic Recurrent Neural Network from [1]

        [1] https://arxiv.org/abs/1605.07571
        """
        super(SRNN, self).__init__()

        self.x_embedding = x_embedding
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.o_dim = o_dim
        self.dropout = dropout
        self.word_dropout = word_dropout
        self.num_layers = num_layers
        self.residual_posterior = residual_posterior

        self.forward_recurrent = nn.GRU(x_dim, h_dim, num_layers)
        self.backward_recurrent = nn.GRU(x_dim + h_dim, h_dim, num_layers)

        # encoder  x/u to z, input to latent variable, inference model
        self.encoder = nn.Sequential(
            nn.Linear(h_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            GaussianDense(h_dim, z_dim),
        )

        # prior transition of zt-1 to zt
        self.prior = nn.Sequential(
            nn.Linear(h_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            GaussianDense(h_dim, z_dim),
        )

        # decoder from latent variable to output, from z to u
        self.decoder = nn.Sequential(
            nn.Linear(h_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout) if dropout else None
        self.likelihood = likelihood

        self.word_dropout = WordDropout(word_dropout) if word_dropout else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i_layer in range(self.num_layers):
            weight_name = f"weight_hh_l{i_layer}"
            init.orthogonal_(getattr(self.forward_recurrent, weight_name))
            init.orthogonal_(getattr(self.backward_recurrent, weight_name))

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

        kl = (kld_twise * seq_mask.unsqueeze(-1)).sum((1, 2))  # (B,)
        elbo = log_prob - kl  # (B,)

        kld_twise_fn = discount_free_nats(kld_twise, free_nats, shared_dims=-1)
        kl = (kld_twise_fn * seq_mask.unsqueeze(-1)).sum((1, 2))  # (B,)
        loss = -(log_prob - beta * kl).sum() / x_sl.sum()  # (1,)

        return loss, elbo, log_prob, kl, seq_mask

    def forward(
        self,
        x: TensorType["B", "T", "x_dim"],
        x_sl: TensorType["B", int],
        u: TensorType["B", "T", "x_dim"] = None,
        d0: TensorType["num_layers", "B", "h_dim"] = None,
        a0: TensorType["num_layers", "B", "h_dim"] = None,
        z0: TensorType["B", "z_dim"] = None,
        beta: float = 1,
        free_nats: float = 0,
    ):
        batch_size = x.size(0)
        device = x.device

        # target
        y = x.clone().detach()

        # dropout
        x = self.word_dropout(x) if self.word_dropout else x

        # x features
        x_embedding = self.x_embedding(x)
        x_embedding = x_embedding.permute(1, 0, 2)  # (T, B, D)

        # u_features
        u_embedding = torch.cat([torch.zeros_like(x_embedding[0:1]), x_embedding[:-1, ...]], dim=0) if u is None else u

        # u_t to d_t
        d0 = torch.zeros(self.num_layers, batch_size, self.h_dim, device=device) if d0 is None else d0
        d, d_n = self.forward_recurrent(u_embedding, d0)
        d = torch.cat([d0, d[:-1, ...]], dim=0)  # Pop last hidden, prepend initial

        # x_t and d_t to a_t
        concat_h_t_x_t = torch.cat([x_embedding, d], dim=-1)
        concat_h_t_x_t = reverse_sequences(concat_h_t_x_t, x_sl)

        a0 = torch.zeros(self.num_layers, batch_size, self.h_dim, device=device) if a0 is None else a0
        a, a_n = self.backward_recurrent(concat_h_t_x_t, a0)
        a = reverse_sequences(a, x_sl)  # reverse back again (index 0 == time 0)

        # prepare for iteration
        all_enc_mu, all_enc_sd = [], []
        all_prior_mu, all_prior_sd = [], []
        z_t_sampled = []

        d = d.permute(1, 0, 2)
        a = a.permute(1, 0, 2)

        z_t = torch.zeros(batch_size, self.z_dim, device=device) if z0 is None else z0

        for d_t, a_t in zip(d.unbind(1), a.unbind(1)):

            # prior conditioned on d_t and z_{t-1}
            prior_mu_t, prior_sd_t = self.prior(torch.cat([d_t, z_t], dim=-1))

            # encoder conditioned on a_t and z_{t-1}
            enc_mu_t, enc_sd_t = self.encoder(torch.cat([a_t, z_t], dim=-1))

            # residual parameterization of posterior
            if self.residual_posterior:
                enc_mu_t = enc_mu_t + prior_mu_t

            # sampling and reparameterization
            z_t = rsample_gaussian(enc_mu_t, enc_sd_t)

            all_prior_mu.append(prior_mu_t)
            all_prior_sd.append(prior_sd_t)
            all_enc_mu.append(enc_mu_t)
            all_enc_sd.append(enc_sd_t)
            z_t_sampled.append(z_t)

        # decoder emission (generative model)
        z = torch.stack(z_t_sampled, dim=1)
        dec = self.decoder(torch.cat([z, d], dim=-1))

        dec = self.dropout(dec) if self.dropout is not None else dec

        logits = self.likelihood(dec)  # (B, T, D)

        enc_mu = torch.stack(all_enc_mu, dim=1)
        enc_sd = torch.stack(all_enc_sd, dim=1)
        prior_mu = torch.stack(all_prior_mu, dim=1)
        prior_sd = torch.stack(all_prior_sd, dim=1)
        kld = kl_divergence_gaussian(enc_mu, enc_sd, prior_mu, prior_sd)

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
        outputs = SimpleNamespace(
            elbo=elbo,
            log_prob=log_prob,
            kl=kl,
            y=y,
            logits=logits,
            seq_mask=seq_mask,
            d=d_n,
            a=a_n,
            z=z_t_sampled[-1],
        )
        return loss, metrics, outputs

    def generate(
        self,
        x: TensorType["B", "T", "x_dim"],
        u: TensorType["B", "T", "x_dim"] = None,
        d0: TensorType["num_layers", "B", "h_dim"] = None,
        a0: TensorType["num_layers", "B", "h_dim"] = None,
        z0: TensorType["B", "z_dim"] = None,
        n_samples: int = 1,
        max_timesteps: int = 100,
        stop_value: float = None,
        use_mode: bool = False,
    ):
        # TODO Implement SRNN generation
        device = self.forward_recurrent.weight_hh_l0.device

        # Conditional generation
        if x is not None:
            x_sl = torch.full_like(x, fill_value=x.size(1))
            _, _, outputs = self.forward(x, x_sl, u, d0=d0, a0=a0, z0=z0)
            d0 = outputs.d
            a0 = outputs.a
            z_t = outputs.z
        else:
            d0 = torch.zeros(self.num_layers, n_samples, self.h_dim, device=device) if d0 is None else d0
            a0 = torch.zeros(self.num_layers, n_samples, self.h_dim, device=device) if a0 is None else a0
            z_t = torch.zeros(n_samples, self.z_dim, device=device) if z0 is None else z0

        all_prior_mu, all_prior_sd = [], []
        z_t_sampled = []

        seq_active = torch.ones(n_samples, dtype=torch.int)
        all_ended, t = False, 0  # Used to condition while loop

        pbar = tqdm(total=max_timesteps)
        while not all_ended and t < max_timesteps:
            

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


class SRNNLM(BaseModel):
    def __init__(
        self,
        num_embeddings: int,
        delimiter_token_idx: int,
        embedding_dim: int = 300,
        hidden_size: int = 256,
        latent_size: int = 64,
        word_dropout: float = 0,
        dropout: float = 0,
        residual_posterior: bool = False,
    ):
        """An SRNN for language modelling.

        Notes:
        -
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.word_dropout = word_dropout
        self.dropout = dropout
        self.delimiter_token_idx = delimiter_token_idx
        self.residual_posterior = residual_posterior

        embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        categorical = CategoricalDense(hidden_size, num_embeddings)

        self.srnn = SRNN(
            x_embedding=embedding,
            likelihood=categorical,
            x_dim=embedding_dim,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=num_embeddings,
            word_dropout=word_dropout,
            dropout=dropout,
            residual_posterior=residual_posterior,
        )

    def forward(
        self,
        x: TensorType["B", "T", int],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        d0: TensorType["num_layers", "B", "h_dim"] = None,
        a0: TensorType["num_layers", "B", "h_dim"] = None,
        z0: TensorType["B", "z_dim"] = None,
    ):
        return self.srnn(x=x, x_sl=x_sl, d0=d0, a0=a0, z0=z0, beta=beta, free_nats=free_nats)

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = True,
        x: TensorType["B", "T", int] = None,
        d0: TensorType["num_layers", "B", "h_dim"] = None,
        a0: TensorType["num_layers", "B", "h_dim"] = None,
        z0: TensorType["B", "z_dim"] = None,
    ):
        x = torch.full([n_samples], self.delimiter_token_idx, device=self.device) if x is None else x
        return self.srnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=self.delimiter_token_idx,
            use_mode=use_mode,
            x=x,
            d0=d0,
            a0=a0,
            z0=z0,
        )


class SRNNAudioDML(BaseModel):
    def __init__(
        self,
        input_size: int = 200,
        hidden_size: int = 256,
        latent_size: int = 64,
        word_dropout: float = 0,
        dropout: float = 0,
        residual_posterior: bool = False,
        num_mix: int = 10,
        num_bins: int = 256,
    ):
        """An SRNN for modelling audio waveforms."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.word_dropout = word_dropout
        self.dropout = dropout
        self.residual_posterior = residual_posterior
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
        # likelihood = DiscretizedLogisticMixtureDense(
        #     x_dim=hidden_size,
        #     y_dim=input_size,
        #     num_mix=num_mix,
        #     num_bins=num_bins,
        #     reduce_dim=-1,
        # )
        likelihood = DiscretizedLogisticDense(
            x_dim=hidden_size,
            y_dim=input_size,
            num_bins=num_bins,
            reduce_dim=-1,
        )
        self.srnn = SRNN(
            x_embedding=embedding,
            likelihood=likelihood,
            x_dim=hidden_size,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=input_size,
            word_dropout=word_dropout,
            dropout=dropout,
            residual_posterior=residual_posterior,
        )

    def forward(
        self,
        x: TensorType["B", "T", "D", float],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        d0: TensorType["num_layers", "B", "h_dim"] = None,
        a0: TensorType["num_layers", "B", "h_dim"] = None,
        z0: TensorType["B", "h_dim"] = None,
    ):
        loss, metrics, outputs = self.srnn(x=x, x_sl=x_sl, d0=d0, a0=a0, z0=z0, beta=beta, free_nats=free_nats)
        outputs.x_hat = self.srnn.likelihood.sample(outputs.logits)
        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: TensorType["B", "x_dim"] = None,
        h: TensorType["B", "h_dim"] = None,
    ):
        x = torch.zeros(n_samples, self.input_size, device=self.device) if x is None else x
        return self.srnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=None,
            use_mode=use_mode,
            x=x,
            h=h,
        )


class SRNNAudioGauss(BaseModel):
    def __init__(
        self,
        input_size: int = 200,
        hidden_size: int = 256,
        latent_size: int = 64,
        word_dropout: float = 0,
        dropout: float = 0,
        residual_posterior: bool = False,
    ):
        """An SRNN for modelling audio waveforms."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.word_dropout = word_dropout
        self.dropout = dropout
        self.residual_posterior = residual_posterior

        embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        likelihood = GaussianDense(hidden_size, input_size, reduce_dim=-1)

        self.srnn = SRNN(
            x_embedding=embedding,
            likelihood=likelihood,
            x_dim=hidden_size,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=input_size,
            word_dropout=word_dropout,
            dropout=dropout,
            residual_posterior=residual_posterior,
        )

    def forward(
        self,
        x: TensorType["B", "T", float],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        d0: TensorType["num_layers", "B", "h_dim"] = None,
        a0: TensorType["num_layers", "B", "h_dim"] = None,
        z0: TensorType["B", "h_dim"] = None,
    ):
        loss, metrics, outputs = self.srnn(x=x, x_sl=x_sl, d0=d0, a0=a0, z0=z0, beta=beta, free_nats=free_nats)

        for metric_name in ["BitsPerDimMetric"]:
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
        h: TensorType["B", "h_dim"] = None,
    ):
        x = torch.zeros(n_samples, self.input_size, device=self.device) if x is None else x
        return self.srnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=None,
            use_mode=use_mode,
            x=x,
            h=h,
        )
