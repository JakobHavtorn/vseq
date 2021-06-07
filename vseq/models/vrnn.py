import math

from types import SimpleNamespace
from vseq.utils.dict import list_of_dict_to_dict_of_list

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.jit as jit

from torchtyping import TensorType

from vseq.evaluation import LossMetric, LLMetric, KLMetric, PerplexityMetric, BitsPerDimMetric, LatestMeanMetric
from vseq.models import BaseModel
from vseq.modules.convenience import AddConstant
from vseq.modules.distributions import GaussianDense, BernoulliDense, CategoricalDense
from vseq.modules.dropout import WordDropout
from vseq.utils.operations import sequence_mask
from vseq.utils.variational import discount_free_nats


class VRNNCell(jit.ScriptModule):
    def __init__(self, x_dim: int, h_dim: int, z_dim: int, depth: int = 3):
        """Variational Recurrent Neural Network (VRNN) cell from [1].

        Uses unimodal isotropic gaussian distributions for inference, prior, and generative models.

        Args:
            x_dim (int): Input space size
            h_dim (int): Hidden space (GRU) size
            z_dim (int): Stochastic latent variable size

        [1] https://arxiv.org/abs/1506.02216
        """
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.depth = depth

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

        # self.gru_cell = nn.GRUCell(h_dim, h_dim)
        self.gru_cell = nn.GRUCell(x_dim + h_dim, h_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.orthogonal_(self.gru_cell.weight_hh)

    def forward(self, x: TensorType["B", "x_dim"], h: TensorType["B", "h_dim"]):
        # prior p(z)
        prior_mu, prior_sd = self.prior(h)

        # encoder q(z|x)
        enc_mu, enc_sd = self.encoder(torch.cat([x, h], -1))

        # sampling and reparameterization
        z = self._reparameterized_sample(enc_mu, enc_sd)

        # z features
        phi_z = self.phi_z(z)

        # gru cell
        h = self.gru_cell(torch.cat([x, phi_z], -1), h)
        # h = self.gru_cell(phi_z, h)

        # kl divergence
        kld = self._kld_gauss(enc_mu, enc_sd, prior_mu, prior_sd)

        outputs = SimpleNamespace(z=z, enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=prior_mu, prior_sd=prior_sd)
        return h, phi_z, kld, outputs

    def generate(self, x: TensorType["B", "x_dim"], h: TensorType["B", "h_dim"]):
        # prior p(z)
        prior_mu, prior_sd = self.prior(h)

        # sampling and reparameterization
        z = self._reparameterized_sample(prior_mu, prior_sd)

        # z features
        phi_z = self.phi_z(z)

        # gru cell
        # h = self.gru_cell(torch.cat([x, phi_z], 1), h)
        h = self.gru_cell(phi_z, h)

        return h, phi_z, SimpleNamespace(z=z, prior_mu=prior_mu, prior_sd=prior_sd)

    @jit.export
    def _reparameterized_sample(self, mu, sd):
        """using sd to sample"""
        return torch.randn_like(sd).mul(sd).add_(mu)

    @jit.export
    def _kld_gauss(self, mu_1, sd_1, mu_2, sd_2):
        """Compute element-wise KL divergence between two Gaussians"""
        return torch.distributions.kl_divergence(
            torch.distributions.Normal(mu_1, sd_1), torch.distributions.Normal(mu_2, sd_2)
        )
        # return sd_2.log() - sd_1.log() + (sd_1.pow(2) + (mu_1 - mu_2).pow(2)) / (2 * sd_2.pow(2)) - 0.5


class VRNN(nn.Module):
    def __init__(
        self,
        phi_x: nn.Module,
        likelihood: nn.Module,
        x_dim: int,
        h_dim: int,
        z_dim: int,
        o_dim: int,
        word_dropout: float = 0,
    ):
        """Variational Recurrent Neural Network (VRNN) from [1].

        Uses unimodal isotropic gaussian distributions for inference, prior, and generative models.

        Args:
            phi_x (nn.Module): Input transformation
            x_dim (int): Input space size
            h_dim (int): Hidden space (GRU) size
            z_dim (int): Stochastic latent variable size
            o_dim (int): Output space size

        [1] https://arxiv.org/abs/1506.02216
        """
        super().__init__()

        self.phi_x = phi_x
        self.likelihood = likelihood

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.o_dim = o_dim

        self.vrnn_cell = VRNNCell(
            x_dim=x_dim,
            h_dim=h_dim,
            z_dim=z_dim,
        )

        self.decoder = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.likelihood = likelihood
        self.word_dropout = WordDropout(word_dropout) if word_dropout else None

    def compute_elbo(
        self,
        y: TensorType["B", "T"],
        logits: TensorType["B", "T", "D"],
        kl_twise: TensorType["B", "T", "latent_size"],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
    ):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""
        seq_mask = sequence_mask(x_sl, dtype=float, device=y.device)
        log_prob_twise = self.likelihood.log_prob(y, logits) * seq_mask
        log_prob = log_prob_twise.view(y.size(0), -1).sum(1)  # (B,)

        kl = kl_twise.sum((1, 2))  # (B,)
        elbo = log_prob - kl  # (B,)

        kl_twise = discount_free_nats(kl_twise, free_nats, -1)
        kl = kl_twise.sum((1, 2))  # (B,)

        loss = -(log_prob - beta * kl).sum() / x_sl.sum()  # (1,)
        return loss, elbo, log_prob, kl

    def forward(
        self,
        x: TensorType["B", "T", "x_dim"],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h: TensorType["B", "h_dim"] = None,
    ):

        batch_size, timesteps = x.size(0), x.size(1)

        all_h = []
        all_phi_z = []
        all_kld = []
        all_outputs = []

        # target
        y = x.clone().detach()

        # dropout
        x = self.word_dropout(x) if self.word_dropout else x

        # x features
        phi_x = self.phi_x(x)
        phi_x = phi_x.unbind(1)

        # initial h
        h = torch.zeros(batch_size, self.h_dim, device=x.device) if h is None else h
        all_h.append(h)

        for t in range(timesteps):
            h, phi_z, kld, outputs = self.vrnn_cell(phi_x[t], h)

            all_h.append(h)
            all_phi_z.append(phi_z)
            all_kld.append(kld)
            all_outputs.append(outputs)

        all_h.pop()

        # output distribution
        h = torch.stack(all_h, dim=1)
        phi_z = torch.stack(all_phi_z, dim=1)
        logits = self.likelihood(self.decoder(torch.cat([phi_z, h], -1)))  # (B, T, D)

        # loss
        kld = torch.stack(all_kld, dim=1)
        loss, elbo, log_prob, kl = self.compute_elbo(y, logits, kld, x_sl, beta, free_nats)

        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo"),
            LLMetric(log_prob, name="rec"),
            KLMetric(kl),
            BitsPerDimMetric(elbo, reduce_by=x_sl),
            PerplexityMetric(elbo, reduce_by=x_sl),
            LatestMeanMetric(beta, name="beta"),
            LatestMeanMetric(free_nats, name="free_nats"),
        ]
        outputs = SimpleNamespace(elbo=elbo, log_prob=log_prob, kl=kl, y=y, logits=logits)
        return loss, metrics, outputs

    def generate(
        self,
        x: TensorType["B", "T", "x_dim"],
        h: TensorType["B", "h_dim"] = None,
        n_samples: int = 1,
        max_timesteps: int = 100,
        stop_value: float = None,
        use_mode: bool = False,
    ):

        if x.size(0) > 1:
            assert x.size(0) == n_samples
        else:
            x = x.repeat(n_samples, *[1] * (x.ndim - 1))  # Repeat only batch

        # TODO If multiple timesteps in x, run a forward pass to get conditional initial h (assert h is None)

        all_outputs = []
        all_x = [x]
        x_sl = torch.ones(n_samples, dtype=torch.int)

        # initial h
        h = torch.zeros(n_samples, self.h_dim, device=x.device) if h is None else h

        seq_active = torch.ones(n_samples, dtype=torch.int)
        all_ended, t = False, 0  # Used to condition while loop
        while not all_ended and t < max_timesteps:
            phi_x = self.phi_x(x)

            h, phi_z, outputs = self.vrnn_cell.generate(phi_x, h)

            logits = self.likelihood(self.decoder(torch.cat([phi_z, h], -1)))  # (B, D)

            if use_mode:
                x = self.likelihood.mode(logits)
            else:
                x = self.likelihood.get_distribution(logits).sample()

            all_x.append(x)
            all_outputs.append(outputs)

            # Update sequence length
            x_sl += seq_active
            seq_ending = x == stop_value  # (,), (B,) or (B, D)
            if isinstance(seq_ending, torch.Tensor):
                seq_ending = seq_ending.all(*list(range(1, seq_ending.ndim))) if seq_ending.ndim > 1 else seq_ending
                seq_ending = seq_ending.to(int).cpu()
            else:
                seq_ending = int(seq_ending)
            seq_active *= 1 - seq_ending

            # Update loop conditions
            t += 1
            all_ended = torch.all(1 - seq_active).item()

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
        word_dropout: float = 0,
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
        self.delimiter_token_idx = delimiter_token_idx

        embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        categorical = CategoricalDense(hidden_size, num_embeddings)

        self.vrnn = VRNN(
            phi_x=embedding,
            likelihood=categorical,
            x_dim=embedding_dim,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=num_embeddings,
            word_dropout=word_dropout,
        )

    def forward(
        self,
        x: TensorType["B", "T", int],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h: TensorType["B", "h_dim"] = None,
    ):
        return self.vrnn(x, x_sl, beta, free_nats, h)

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = True,
        x: TensorType["B", "x_dim"] = None,
        h: TensorType["B", "h_dim"] = None,
    ):
        x = torch.full([n_samples], self.delimiter_token_idx, device=self.device) if x is None else x
        return self.vrnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=self.delimiter_token_idx,
            use_mode=use_mode,
            x=x,
            h=h,
        )


class VRNN_MIDI(BaseModel):
    def __init__(
        self,
        input_size: int = 88,
        hidden_size: int = 256,
        latent_size: int = 64,
    ):
        """A VRNN for modelling the MIDI music dataset."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

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
        )

    def forward(
        self,
        x: TensorType["B", "T", "D", float],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h: TensorType["B", "h_dim"] = None,
    ):
        return self.vrnn(x, x_sl, beta, free_nats, h)

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: TensorType["B", "x_dim"] = None,
        h: TensorType["B", "h_dim"] = None,
    ):
        x = torch.zeros(n_samples, self.input_size, device=self.device) if x is None else x
        return self.vrnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=None,
            use_mode=use_mode,
            x=x,
            h=h,
        )


class VRNNAudio(BaseModel):
    def __init__(
        self,
        input_size: int = 200,
        hidden_size: int = 256,
        latent_size: int = 64,
    ):
        """A VRNN for modelling audio waveforms."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

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
        )

    def forward(
        self,
        x: TensorType["B", "T", float],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h: TensorType["B", "h_dim"] = None,
    ):
        loss, metrics, outputs = self.vrnn(x, x_sl, beta, free_nats, h)

        for metric_name in ["BitsPerDimMetric", "PerplexityMetric"]:
            bpd_metric_idx = [i for i, metric in enumerate(metrics) if metric.__class__.__name__ == metric_name][0]
            del metrics[bpd_metric_idx]

        metrics.extend([
            BitsPerDimMetric(outputs.elbo.cpu() - x_sl * self.input_size * math.log(2 ** 8), reduce_by=x_sl),
            LatestMeanMetric(((outputs.y - outputs.logits[0]) ** 2).mean().sqrt().item(), name="mse"),
            LatestMeanMetric(outputs.logits[1].mean().item(), name="stddev"),
        ])

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
        return self.vrnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=None,
            use_mode=use_mode,
            x=x,
            h=h,
        )


# ==================================
#  DEPRECATED MODELS BELOW THIS LINE
# ==================================


class VRNNOld(nn.Module):
    def __init__(self, x_dim: int, h_dim: int, z_dim: int, o_dim: int):
        """Variational Recurrent Neural Network (VRNN) from [1].

        Uses unimodal isotropic gaussian distributions for inference, prior, and generative models.

        Args:
            x_dim (int): Input space size
            h_dim (int): Hidden space (GRU) size
            z_dim (int): Stochastic latent variable size
            o_dim (int): Output space size

        [1] https://arxiv.org/abs/1506.02216
        """
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.o_dim = o_dim

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
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

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(h_dim, z_dim)
        self.enc_sd = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus(beta=math.log(2)), AddConstant(1e-3))

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.prior_mu = nn.Linear(h_dim, z_dim)
        self.prior_sd = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus(beta=math.log(2)), AddConstant(1e-3))

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),  # nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.dec_logits = nn.Linear(h_dim, o_dim)

        # recurrence
        self.gru_cell = nn.GRUCell(2 * h_dim, h_dim)
        # self.gru_cell = nn.GRUCell(h_dim, h_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.orthogonal_(self.gru_cell.weight_hh)

    def forward(self, x: TensorType["B", "T", "x_dim"]):

        all_enc_mu, all_enc_sd, all_enc_z = [], [], []
        all_prior_mu, all_prior_sd = [], []
        all_dec_logits = []
        all_kld = []

        batch_size, timesteps = x.size(0), x.size(1)

        # x features
        phi_x = self.phi_x(x)
        phi_x = phi_x.unbind(1)

        h = torch.zeros(batch_size, self.h_dim, device=x.device)
        for t in range(timesteps):

            # prior p(z)
            prior_t = self.prior(h)
            prior_mu_t = self.prior_mu(prior_t)
            prior_sd_t = self.prior_sd(prior_t)

            # encoder q(z|x)
            enc_t = self.enc(torch.cat([phi_x[t], h], 1))
            enc_mu_t = self.enc_mu(enc_t)
            enc_sd_t = self.enc_sd(enc_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mu_t, enc_sd_t)

            # z features
            phi_z_t = self.phi_z(z_t)

            # decoder p(x|z)
            dec_t = self.dec(torch.cat([phi_z_t, h], 1))  # dec_t = self.dec(phi_z_t)
            dec_logits_t = self.dec_logits(dec_t)

            # gru cell (teacher forced by conditioning on phi_x[t])
            # input of shape (batch, input_size)
            # hidden of shape (batch, hidden_size)
            h = self.gru_cell(torch.cat([phi_x[t], phi_z_t], 1), h)  # h = self.gru_cell(phi_z_t, h)

            # computing losses
            kld_t = self._kld_gauss(enc_mu_t, enc_sd_t, prior_mu_t, prior_sd_t)

            all_prior_mu.append(prior_mu_t)
            all_prior_sd.append(prior_sd_t)
            all_kld.append(kld_t)
            all_enc_mu.append(enc_mu_t)
            all_enc_sd.append(enc_sd_t)
            all_enc_z.append(z_t)
            all_dec_logits.append(dec_logits_t)

        o_logits = torch.stack(all_dec_logits, dim=1)
        q_z_x = torch.distributions.Normal(torch.stack(all_enc_mu, dim=1), torch.stack(all_enc_sd, dim=1))
        p_z = torch.distributions.Normal(torch.stack(all_prior_mu, dim=1), torch.stack(all_prior_sd, dim=1))
        kld = torch.stack(all_kld, dim=1)
        z = torch.stack(all_enc_z, dim=1)
        return z, o_logits, q_z_x, p_z, kld

    def generate(self, n_samples: int = 1, t_max: int = 100, use_mode: bool = False):

        sample = torch.zeros(t_max, self.x_dim)

        h = torch.zeros(n_samples, self.h_dim)
        for t in range(t_max):

            # prior
            prior_t = self.prior(h)
            prior_mu_t = self.prior_mu(prior_t)
            prior_sd_t = self.prior_sd(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mu_t, prior_sd_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h], 1))
            dec_logits = self.dec_logits(dec_t)
            p_x_z = torch.distributions.Categorical(dec_logits)
            dec_sample_t = p_x_z.logits.argmax(dim=-1) if use_mode else p_x_z.sample()

            phi_x_t = self.phi_x(dec_sample_t)

            # gru cell
            _, h = self.gru_cell(torch.cat([phi_x_t, phi_z_t], 1), h)

            sample[t] = dec_sample_t

        return sample

    def _reparameterized_sample(self, mu, sd):
        """using sd to sample"""
        return torch.randn_like(sd).mul(sd).add_(mu)

    def _kld_gauss(self, mu_1, sd_1, mu_2, sd_2):
        """Compute element-wise KL divergence between two Gaussians"""
        return sd_2.log() - sd_1.log() + (sd_1.pow(2) + (mu_1 - mu_2).pow(2)) / (2 * sd_2.pow(2)) - 0.5


class VRNNLMOld(BaseModel):
    def __init__(
        self,
        num_embeddings: int,
        delimiter_token_idx: int,
        embedding_dim: int = 300,
        hidden_size: int = 256,
        latent_size: int = 64,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.delimiter_token_idx = delimiter_token_idx
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # NOTE the phi_x feature extraction could be variable, i.e. Embedding here and Sequential NN for VRNN2D

        self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)
        self.mask_token_idx = num_embeddings

        self.vrnn = VRNNOld(
            x_dim=embedding_dim,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=num_embeddings,
        )

    def compute_elbo(
        self,
        log_prob_twise: TensorType["B", "T"],
        kl_twise: TensorType["B", "T", "latent_size"],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
    ):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""
        kl_twise = discount_free_nats(kl_twise, free_nats, -1)
        kl = kl_twise.sum((1, 2))  # (B,)
        log_prob = log_prob_twise.flatten(start_dim=1).sum(1)  # (B,)
        elbo = log_prob - kl  # (B,)
        loss = -(log_prob - beta * kl).sum() / x_sl.sum()  # (1,)
        return loss, elbo, log_prob, kl

    def forward(self, x: TensorType["B", "T", int], x_sl: TensorType["B", int], beta: float = 1, free_nats: float = 0):
        # Prepare inputs (x) and targets (y)
        y = x.clone().detach()  # Form target

        e = self.embedding(x)  # (B, T, D)

        z, o_logits, q_z_x, p_z, kl_twise = self.vrnn(e)

        # Compute loss
        p_x_z = torch.distributions.Categorical(logits=o_logits)
        seq_mask = sequence_mask(x_sl, dtype=float, device=p_x_z.logits.device)
        log_prob_twise = p_x_z.log_prob(y) * seq_mask
        loss, elbo, log_prob, kl = self.compute_elbo(
            log_prob_twise, kl_twise, x_sl=x_sl, beta=beta, free_nats=free_nats
        )

        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo"),
            LLMetric(log_prob, name="rec"),
            KLMetric(kl),
            BitsPerDimMetric(elbo, reduce_by=x_sl),
            PerplexityMetric(elbo, reduce_by=x_sl),
            LatestMeanMetric(beta, name="beta"),
            LatestMeanMetric(free_nats, name="free_nats"),
        ]

        outputs = SimpleNamespace(
            loss=loss,
            elbo=elbo,
            rec=log_prob,
            kl=kl,
            p_x_z=p_x_z,
            q_z_x=q_z_x,
            p_z=p_z,
            z=z,
        )
        return loss, metrics, outputs


class VRNN_MIDIOld(BaseModel):
    def __init__(
        self,
        input_size: int = 88,
        hidden_size: int = 256,
        latent_size: int = 64,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.vrnn = VRNNOld(
            x_dim=input_size,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=input_size,
        )

    def compute_elbo(
        self,
        log_prob_twise: TensorType["B", "T"],
        kl_twise: TensorType["B", "T", "latent_size"],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
    ):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""
        kl_twise = discount_free_nats(kl_twise, free_nats, -1)
        kl = kl_twise.sum((1, 2))  # (B,)
        log_prob = log_prob_twise.flatten(start_dim=1).sum(1)  # (B,)
        elbo = log_prob - kl  # (B,)
        loss = -(log_prob - beta * kl).sum() / x_sl.sum()  # (1,)
        return loss, elbo, log_prob, kl

    def forward(
        self, x: TensorType["B", "T", "input_size"], x_sl: TensorType["B", int], beta: float = 1, free_nats: float = 0
    ):
        # Prepare inputs (x) and targets (y)
        y = x.clone().detach()  # Form target

        z, o_logits, q_z_x, p_z, kl_twise = self.vrnn(x)

        # Compute loss
        p_x_z = torch.distributions.Bernoulli(logits=o_logits)
        seq_mask = sequence_mask(x_sl, dtype=float, device=p_x_z.logits.device)
        log_prob_twise = p_x_z.log_prob(y) * seq_mask.unsqueeze(-1)
        loss, elbo, log_prob, kl = self.compute_elbo(
            log_prob_twise, kl_twise, x_sl=x_sl, beta=beta, free_nats=free_nats
        )

        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo"),
            LLMetric(log_prob, name="rec"),
            KLMetric(kl),
            BitsPerDimMetric(elbo, reduce_by=x_sl),
            PerplexityMetric(elbo, reduce_by=x_sl),
            LatestMeanMetric(beta, name="beta"),
            LatestMeanMetric(free_nats, name="free_nats"),
        ]

        outputs = SimpleNamespace(
            loss=loss,
            elbo=elbo,
            rec=log_prob,
            kl=kl,
            p_x_z=p_x_z,
            q_z_x=q_z_x,
            p_z=p_z,
            z=z,
        )
        return loss, metrics, outputs
