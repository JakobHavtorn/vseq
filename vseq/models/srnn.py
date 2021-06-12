from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.init as init

from torchtyping import TensorType

from vseq.evaluation.metrics import BitsPerDimMetric, KLMetric, LLMetric, LatestMeanMetric, LossMetric, PerplexityMetric
from vseq.models import BaseModel
from vseq.modules import CategoricalDense, GaussianDense, WordDropout
from vseq.utils.variational import discount_free_nats, kl_divergence_gaussian, rsample_gaussian
from vseq.utils.operations import sequence_mask


# TODO Make same structure as VRNN with an SRNN cell (?)


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
        self.likelihood = likelihood
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.o_dim = o_dim
        self.dropout = dropout
        self.word_dropout = word_dropout
        self.num_layers = num_layers
        self.residual_posterior = residual_posterior

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

        self.forward_recurrent = nn.GRU(x_dim, h_dim, num_layers)
        self.backward_recurrent = nn.GRU(x_dim + h_dim, h_dim, num_layers)

        self.word_dropout = WordDropout(word_dropout) if word_dropout else None
        self.dropout = nn.Dropout(dropout) if dropout else None

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

        return loss, elbo, log_prob, kl

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
        d0 = torch.zeros(self.num_layers, batch_size, self.h_dim, device=x.device) if d0 is None else d0
        d, _ = self.forward_recurrent(u_embedding, d0)

        # x_t and d_t to a_t
        concat_h_t_x_t = torch.cat([x_embedding, d], dim=-1)
        concat_h_t_x_t = concat_h_t_x_t.flip(0)  # reverse (index 0 == time T)

        a0 = torch.zeros(self.num_layers, batch_size, self.h_dim, device=x.device) if a0 is None else a0
        a, _ = self.backward_recurrent(concat_h_t_x_t, a0)
        a = a.flip(0)  # reverse back again (index 0 == time 0)

        # prepare for iteration
        all_enc_mu, all_enc_sd = [], []
        all_prior_mu, all_prior_sd = [], []
        z_t_sampled = []

        d = d.permute(1, 0, 2)
        a = a.permute(1, 0, 2)

        z_t = torch.zeros(batch_size, self.z_dim, device=x.device) if z0 is None else z0

        for h_t, a_t in zip(d.unbind(1), a.unbind(1)):

            # prior conditioned on h_t and z_{t-1}
            prior_mu_t, prior_sd_t = self.prior(torch.cat([h_t, z_t], dim=-1))

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

    def generate(self, x, step, u):
        all_enc_mu, all_enc_sd = [], []
        all_dec_mean, all_dec_std = [], []
        z_t_sampled = []
        z_t = torch.zeros(self.num_layers, x.size(1), self.h_dim)[-1]

        # computing hidden state in list and x_t & u_t in list outside the loop
        d = torch.zeros(self.num_layers, x.size(1), self.h_dim)
        h_list = []
        x_t_list = []
        u_t_list = []
        for t in range(x.size(0)):
            x_t = x[t]
            u_t = u[t]
            _, d = self.forward_recurrent(torch.cat([x_t], 1).unsqueeze(0), d)
            x_t_list.append(x_t)
            u_t_list.append(u_t)
            h_list.append(d[-1])

            # reversing hidden state list
        reversed_h = h_list
        reversed_h.reverse()

        # reversing u_t list
        reversed_u_t = u_t_list
        reversed_u_t.reverse()

        #         #concat reverse d with reverse x_t
        concat_h_t_u_t_list = []
        for t in range(x.size(0)):
            concat_h_t_u_t = torch.cat([reversed_u_t[t], reversed_h[t]], 1).unsqueeze(0)
            concat_h_t_u_t_list.append(concat_h_t_u_t)

        #         #compute reverse a_t
        a_t = torch.zeros(self.num_layers, x.size(1), self.h_dim)
        reversed_a_t_list = []
        for t in range(x.size(0)):
            _, a_t = self.backward_recurrent(concat_h_t_u_t_list[t], a_t)  # RNN new
            reversed_a_t_list.append(a_t[-1])
        reversed_a_t_list.reverse()

        for t in range(x.size(0)):
            x_t = x[t]

            # encoder
            enc_t = self.enc(reversed_a_t_list[t])
            enc_mu_t = self.enc_mean(enc_t)
            enc_sd_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h_list[t])
            prior_mu_t = self.prior_mean(prior_t)
            prior_sd_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mu_t, enc_sd_t)
            z_t_sampled.append(z_t)

            # decoder
            dec_t = self.decoder(torch.cat([z_t, h_list[t]], 1))
            dec_mean_t = self.decoder_mean(dec_t)
            dec_std_t = self.decoder_std(dec_t)

            all_enc_sd.append(enc_sd_t)
            all_enc_mu.append(enc_mu_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        x_predict = []
        for i in range(step):
            # prior
            prior_t = self.prior(h_list[t])
            prior_mu_t = self.prior_mean(prior_t)
            prior_sd_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mu_t, enc_sd_t)
            z_t_sampled.append(z_t)

            # decoder
            dec_t = self.decoder(torch.cat([z_t, h_list[i]], 1))
            dec_mean_t = self.decoder_mean(dec_t)
            dec_std_t = self.decoder_std(dec_t)

            x_t = dec_mean_t
            x_predict.append(dec_mean_t)

        return x_predict, z_t_sampled


class SRNNLM(BaseModel):
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
        residual_posterior: bool = False
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
        self.condition_h_on_x = condition_h_on_x
        self.condition_x_on_h = condition_x_on_h
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
            # condition_h_on_x=condition_h_on_x,
            # condition_x_on_h=condition_x_on_h,
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
        d0: TensorType["B", "h_dim"] = None,
        z0: TensorType["B", "h_dim"] = None,
    ):
        return self.srnn(x=x, x_sl=x_sl, d0=d0, z0=z0, beta=beta, free_nats=free_nats)

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = True,
        x: TensorType["B", "x_dim"] = None,
        d: TensorType["B", "h_dim"] = None,
    ):
        x = torch.full([n_samples], self.delimiter_token_idx, device=self.device) if x is None else x
        return self.srnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=self.delimiter_token_idx,
            use_mode=use_mode,
            x=x,
            d=d,
        )