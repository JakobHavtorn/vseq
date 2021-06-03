import math

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.init as init

from torchtyping import TensorType

from vseq.evaluation import LossMetric, LLMetric, KLMetric, PerplexityMetric, BitsPerDimMetric, LatestMeanMetric
from vseq.models import BaseModel
from vseq.modules.convenience import AddConstant
from vseq.utils.operations import sequence_mask
from vseq.utils.variational import discount_free_nats


class VRNN(nn.Module):
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
            # NOTE Should we use the same phi_x[t] in these two places?
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
            # NOTE Should we use the same phi_x[t] in these two places?
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
        # kld = kld.detach()
        # kld = kld * 0
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


class VRNNLM(BaseModel):
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

        self.vrnn = VRNN(
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


class VRNN2D(BaseModel):
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

        self.vrnn = VRNN(
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
