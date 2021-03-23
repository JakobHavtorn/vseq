import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from torch.nn.modules import dropout
from torch.nn.modules.sparse import Embedding

import vseq.modules
import vseq.modules.activations
from vseq.data.tokens import START_TOKEN

from .base_module import BaseModule


class Bowman(BaseModule):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_size: int, delimiter_token_idx: int):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.std_activation = nn.Softplus(beta=np.log(2))
        self.std_activation_inverse = vseq.modules.activations.InverseSoftplus(beta=np.log(2))

        self.delimiter_token_idx = delimiter_token_idx

        # The input embedding for x. We use one embedding shared between encoder and decoder. This may be inappropriate.
        self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)
        self.unknown_token_idx = num_embeddings

        self.lstm_encode = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)

        self.h_to_q = nn.Linear(hidden_size, 2 * hidden_size)

        self.lstm_decode = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)

        self.output = nn.Linear(hidden_size, num_embeddings)

        prior_logits = torch.cat([
            torch.zeros(hidden_size),
            torch.ones(hidden_size)
        ])
        self.register_buffer("prior_logits", prior_logits)

        # TODO WordDropout as module
        # TODO Likelihood as module
        # TODO Stochastic layer as module (incl. reparameterization (and KL?))

    @property
    def prior(self):
        """Return the prior distribution without a batch dimension"""
        mu, sigma = self.prior_logits.chunk(2, dim=0)
        return D.Normal(mu, sigma)

    def compute_kl(self, z, q_z):
        """Compute KL divergence from approximate posterior to prior by MC sampling"""
        return q_z.log_prob(z) - self.prior.log_prob(z)

    def infer(self, x, x_sl):
        # import IPython; IPython.embed(using=False)

        x, x_sl = x[:, 1:], x_sl - 1  # Remove start token
        x = self.embedding(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_sl, batch_first=True)
        h, (h_n, c_n) = self.lstm_encode(x)

        # q(z|x)
        q_logits = self.h_to_q(h_n)
        mu, log_sigma = q_logits.chunk(2, dim=2)
        sigma = self.std_activation(log_sigma)
        q_z = D.Normal(mu, sigma)
        z = q_z.rsample()
        return z, q_z

    def generate(self, z=None, x=None, x_sl=None, num_samples=None, word_dropout_rate: float = 0.75, batch_size: int = 1):
        """Sample from prior if z is None, evaluate likelihood if x is not None"""

        # if z is not None:
        #     h_0 = z
        # else:
        #     if x is None:
        #         h_0 = self.prior.rsample()
        #     else:
        #         h_0 = self.prior.rsample(x.size)
        h_0 = z if z is not None else self.prior.rsample() # labo: is rsample necessary here?
        c_0 = torch.zeros_like(h_0)

        if x is not None:
            targets = x[:, 1:].T.clone().detach() # Remove start token, batch_first=False and prevent from being masked
            mask = torch.bernoulli(torch.full(x.shape, word_dropout_rate)).to(bool)
            mask[:, 0] = False # We never mask the start token - or do we?
            x[mask] = self.unknown_token_idx
            x = self.embedding(x)

            x = torch.nn.utils.rnn.pack_padded_sequence(x, x_sl - 1, batch_first=True)  # Remove end token
            h, (h_n, c_n) = self.lstm_decode(x, (h_0, c_0))

            # p(x|z)
            h, _ = torch.nn.utils.rnn.pad_packed_sequence(h)
            p_logits = self.output(h) # labo: we could use our word embedding matrix here
            p = D.Categorical(logits=p_logits)
            log_prob = p.log_prob(targets) # labo: maybe we should mask this tensor?
        else:
            batch_size = z.shape[0] if z is not None else batch_size
            next_idx = torch.full(batch_size, self.delimiter_token_idx)
            for x_t in x:
                x_t = self.embedding(next_idx)
                (h_n, c_n) = self.lstm_decode(x_t)
                p_logits = self.output(h_n)
                p = D.Categorical(logits=p_logits)
                next_idx = p.sample()
            pass

        return p, log_prob

    def forward(self, x, x_sl, word_dropout_rate: float = 0.75):
        """Perform inference and generative passes on input x of shape (B, T)"""

        z, q_z = self.infer(x, x_sl)
        kl_divergence = self.compute_kl(z, q_z)
        #import IPython; IPython.embed(using=False)
        p, log_prob = self.generate(z=z, x=x, x_sl=x_sl, word_dropout_rate=word_dropout_rate)
        return p, log_prob, kl_divergence


class WordDropout(nn.Module):
    def __init__(self, unknown_idx, dropout_rate=0.75):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.keep_rate = 1 - self.dropout_rate

    def forward(self, x: torch.Tensor):
        """Dropout entire timesteps in x of shape (T, N, 1)"""
        mask = torch.bernoulli(torch.ones(x.shape[0]) * self.keep_rate)
        x *= mask
        return x
