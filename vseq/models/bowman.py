from collections import namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from torch.nn.modules import dropout
from torch.nn.modules.sparse import Embedding

import vseq.modules
import vseq.modules.activations
from vseq.utils.operations import sequence_mask
from vseq.data.tokens import START_TOKEN
from vseq.utils.variational import kl_divergence_mc

from .base_module import BaseModule


Output = namedtuple("Output", ["loss", "elbo", "rec", "kl", "rec_twise", "kl_dwise", "p_x", "q_z", "p_z", "z"])


class Bowman(BaseModule):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_size: int, latent_dim: int, delimiter_token_idx: int):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.std_activation = nn.Softplus(beta=np.log(2))
        self.std_activation_inverse = vseq.modules.activations.InverseSoftplus(beta=np.log(2))

        self.delimiter_token_idx = delimiter_token_idx

        # The input embedding for x. We use one embedding shared between encoder and decoder. This may be inappropriate.
        self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)
        self.mask_token_idx = num_embeddings

        self.lstm_encode = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
        )

        self.h_to_z = nn.Linear(hidden_size, 2 * latent_dim)
        self.z_to_h = nn.Linear(latent_dim, hidden_size)

        self.lstm_decode = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
        )

        self.output = nn.Linear(hidden_size, num_embeddings)

        prior_logits = torch.cat([torch.zeros(latent_dim), torch.ones(latent_dim)])
        self.register_buffer("prior_logits", prior_logits)

        # TODO WordDropout as module
        # TODO Likelihood as module
        # TODO Stochastic layer as module (incl. reparameterization (and KL?))

    def prior(self):
        """Return the prior distribution without a batch dimension"""
        mu, sigma = self.prior_logits.chunk(2, dim=0)
        return D.Normal(mu, sigma)

    def compute_elbo(self, log_prob_twise, kl_dwise, x_sl, beta: float = 1):
        kl = kl_dwise.sum(2).squeeze()  # (B,)
        log_prob = log_prob_twise.sum(1)  # (B,)
        elbo = log_prob - kl  # (B,)
        loss = -(log_prob - beta * kl).mean()  # (1,)
        # loss = - log_prob.sum() / x_sl.sum() + beta * kl.mean()  # (1,)
        return loss, elbo, log_prob, kl

    def forward(self, x, x_sl, word_dropout_rate: float = 0.75, beta: float = 1) -> Tuple[torch.Tensor, Output]:
        """Perform inference and generative passes on input x of shape (B, T)"""
        z, q_z = self.infer(x, x_sl)
        p_z = self.prior()
        # kl_dwise, _, _ = kl_divergence_mc(q_z, p_z, z)
        kl_dwise = torch.distributions.kl_divergence(q_z, p_z)

        log_prob_twise, p_x = self.reconstruct(z=z, x=x, x_sl=x_sl, word_dropout_rate=word_dropout_rate)

        loss, elbo, log_prob, kl = self.compute_elbo(log_prob_twise, kl_dwise, x_sl=x_sl)

        output = Output(
            loss=loss,
            elbo=elbo,
            rec=log_prob,
            kl=kl,
            rec_twise=log_prob_twise,
            kl_dwise=kl_dwise,
            p_x=p_x,
            q_z=q_z,
            p_z=p_z,
            z=z,
        )
        return loss, output

    def infer(self, x: torch.Tensor, x_sl: torch.Tensor):
        # Encode input sequence
        x, x_sl = x[:, 1:], x_sl - 1  # Remove start token
        e = self.embedding(x)
        e = torch.nn.utils.rnn.pack_padded_sequence(e, x_sl, batch_first=True)
        h, (h_t, c_t) = self.lstm_encode(e)
        h_t = h_t.transpose(0, 1)  # (T=1, B, D) --> (B, T=1, D)

        # Compute and sample from q(z|x)
        q_z_logits = self.h_to_z(h_t)
        mu, log_sigma = q_z_logits.chunk(2, dim=2)
        sigma = self.std_activation(log_sigma)
        q_z = D.Normal(mu, sigma)
        z = q_z.rsample()
        return z, q_z  # (B, T=1, D)

    def reconstruct(self, z: torch.Tensor, x: torch.Tensor, x_sl: torch.Tensor, word_dropout_rate: float = 0.75):
        """
        Computes log-likelihood for x under p(x|z).
        """
        # Prepare decoder initial states
        h_0 = self.z_to_h(z.transpose(0, 1))  # (B, T, D) -> (T, B, D)
        c_0 = torch.zeros_like(h_0)

        # Prepare inputs (x) and targets (y)
        # import IPython; IPython.embed()
        y = x[:, 1:].clone().detach()  # Remove start token, batch_first=False and prevent from being masked
        if self.training:
            mask = torch.bernoulli(torch.full(x.shape, word_dropout_rate)).to(bool)
            mask[:, 0] = False  # We never mask the start token - or do we?
            x = x.clone()  # We can't modify x in-place
            x[mask] = self.mask_token_idx
        e = self.embedding(x)

        # Compute log probs for p(x|z)
        e = torch.nn.utils.rnn.pack_padded_sequence(e, x_sl - 1, batch_first=True)  # x_sl - 1 --> remove end token
        h, (h_n, c_n) = self.lstm_decode(e, (h_0, c_0))

        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        p_logits = self.output(h)  # labo: we could use our embedding matrix here
        p = D.Categorical(logits=p_logits)
        seq_mask = sequence_mask(x_sl - 1, dtype=float).to(self.device)
        log_prob = p.log_prob(y) * seq_mask

        return log_prob, p

    def generate(self, z: torch.Tensor = None, n_samples: int = 1, t_max: int = 100):
        """
        Generates a sequence by autoregressively sampling from p(x_t|x_<t, z).
        """
        # Setup initial loop conditions
        x_t = torch.full([1, n_samples], self.delimiter_token_idx, device=self.device)
        h_t = z if z is not None else self.prior().sample(torch.Size([1, n_samples]))
        c_t = torch.zeros_like(h_t)
        log_prob, x, x_sl = [], [], torch.zeros(n_samples, dtype=torch.int)
        seq_active = torch.ones(n_samples, dtype=torch.int)
        all_ended, t = False, 0  # Used to condition while loop

        # Sample x from p(x|z)
        while not all_ended and t < t_max:

            # Sample x_t from p(x_t|z, x_<t)
            e_t = self.embedding(x_t)
            _, (h_t, c_t) = self.lstm_decode(e_t, (h_t, c_t))
            p_logits = self.output(h_t)  # labo: again, we could use our embedding matrix here
            p = D.Categorical(logits=p_logits)
            x_t = p.sample()
            log_prob_t = p.log_prob(x_t)

            # Update outputs
            x.append(x_t)
            log_prob.append(log_prob_t)

            # Update sequence length
            x_sl += seq_active
            seq_ending = (x_t[0].cpu() == self.delimiter_token_idx).to(int)  # TODO move to cpu once at end instead
            seq_active *= 1 - seq_ending

            # Update loop conditions
            t += 1
            all_ended = torch.all(1 - seq_active).item()

        seq_mask = sequence_mask(x_sl, dtype=int).T.to(self.device)
        x = torch.cat(x) * seq_mask
        log_prob = torch.cat(log_prob) * seq_mask.to(float)

        return x, log_prob


class WordDropout(nn.Module):
    def __init__(self, unknown_idx, dropout_rate=0.75):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.keep_rate = 1 - self.dropout_rate

    def forward(self, x: torch.Tensor):
        """Dropout entire timesteps in x of shape (T, B, 1)"""
        mask = torch.bernoulli(torch.ones(x.shape[0]) * self.keep_rate)
        x *= mask
        return x
