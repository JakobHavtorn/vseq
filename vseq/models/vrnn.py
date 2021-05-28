from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

from torchtyping import TensorType

from vseq.models import BaseModel
from vseq.utils.operations import sequence_mask
from vseq.evaluation import LossMetric, LLMetric, KLMetric, PerplexityMetric, BitsPerDimMetric


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


# TODO Output distribution Likelihood Module
# TODO Latent space Stochastic Module


class VRNN(nn.Module):
    def __init__(self, x_dim: int, h_dim: int, z_dim: int, o_dim: int, num_layers: int = 1, bias: bool = True):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.o_dim = o_dim
        self.num_layers = num_layers

        # feature-extracting transformations
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

        # encoder
        self.enc = nn.Sequential(nn.Linear(2 * h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        # prior
        self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        # decoder
        self.dec = nn.Sequential(nn.Linear(2 * h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU())
        self.dec_logits = nn.Linear(h_dim, o_dim)
        # self.dec_std = nn.Sequential(nn.Linear(h_dim, x_dim), nn.Softplus())
        # self.dec_mean = nn.Linear(h_dim, x_dim)
        # self.dec_mean = nn.Sequential(nn.Linear(h_dim, x_dim), nn.Sigmoid())

        # recurrence
        self.rnn = nn.GRU(2 * h_dim, h_dim, num_layers=num_layers, bias=bias)

    def forward(self, x):

        all_enc_mean, all_enc_std, all_enc_z = [], [], []
        all_prior_mean, all_prior_std = [], []
        all_dec_logits = []
        all_kld = []

        # x features
        phi_x = self.phi_x(x)

        h = torch.zeros(self.num_layers, x.size(1), self.h_dim, device=x.device)
        for t in range(x.size(0)):

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # encoder
            enc_t = self.enc(torch.cat([phi_x[t], h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)

            # z features
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_logits_t = self.dec_logits(dec_t)
            # dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x[t], phi_z_t], 1).unsqueeze(0), h)

            # computing losses
            kld_t = self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_kld.append(kld_t)
            all_enc_mean.append(enc_mean_t)
            all_enc_std.append(enc_std_t)
            all_enc_z.append(z_t)
            all_dec_logits.append(dec_logits_t)

        o_logits = torch.stack(all_dec_logits)
        q_z_x = torch.distributions.Normal(torch.stack(all_enc_mean), torch.stack(all_enc_std))
        p_z = torch.distributions.Normal(torch.stack(all_prior_mean), torch.stack(all_prior_std))
        kld = torch.stack(all_kld)
        z = torch.stack(all_enc_z)
        return z, o_logits, q_z_x, p_z, kld

    def generate(self, n_samples: int = 1, t_max: int = 100, use_mode: bool = False):

        sample = torch.zeros(t_max, self.x_dim)

        h = torch.zeros(self.num_layers, n_samples, self.h_dim)
        for t in range(t_max):

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_logits = self.dec_logits(dec_t)
            p_x_z = torch.distributions.Categorical(dec_logits)
            dec_sample_t = p_x_z.logits.argmax(dim=-1) if use_mode else p_x_z.sample()

            phi_x_t = self.phi_x(dec_sample_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_sample_t

        return sample

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        return torch.randn_like(std).mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (
            2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1
        )
        return 0.5 * kld_element


class VRNNLM(BaseModel):
    def __init__(
        self,
        num_embeddings: int,
        delimiter_token_idx: int,
        embedding_dim: int = 300,
        hidden_size: int = 256,
        latent_size: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.delimiter_token_idx = delimiter_token_idx
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)
        self.mask_token_idx = num_embeddings

        self.vrnn = VRNN(
            x_dim=embedding_dim,
            h_dim=hidden_size,
            z_dim=latent_size,
            o_dim=num_embeddings,
            num_layers=num_layers,
            bias=False,
        )

    def compute_elbo(
        self,
        log_prob_twise: TensorType["B", "T"],
        kl_twise: TensorType["B", "T", "latent_size"],
        x_sl: TensorType["B", int],
        beta: float = 1,
    ):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""
        kl = kl_twise.sum((1, 2))  # (B,)
        log_prob = log_prob_twise.sum(1)  # (B,)
        elbo = log_prob - kl  # (B,)
        loss = -(log_prob - beta * kl).sum() / x_sl.sum()  # (1,)
        return loss, elbo, log_prob, kl

    def forward(self, x: TensorType["B", "T", int], x_sl: TensorType["B", int], beta: float = 1):
        # Prepare inputs (x) and targets (y)
        y = x[:, 1:].clone().detach()  # Remove start token and form target
        x, x_sl = x[:, :-1], x_sl - 1  # Remove end token

        e = self.embedding(x)  # (B, T, D)

        z, o_logits, q_z_x, p_z, kl_twise = self.vrnn(e)

        # Compute loss
        p_x_z = torch.distributions.Categorical(logits=o_logits)
        seq_mask = sequence_mask(x_sl, dtype=float, device=p_x_z.logits.device)
        log_prob_twise = p_x_z.log_prob(y) * seq_mask
        loss, elbo, log_prob, kl = self.compute_elbo(log_prob_twise, kl_twise, x_sl=x_sl, beta=beta)

        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo"),
            LLMetric(log_prob, name="rec"),
            KLMetric(kl),
            BitsPerDimMetric(elbo, reduce_by=x_sl),
            PerplexityMetric(elbo, reduce_by=x_sl),
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
