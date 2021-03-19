import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from torch.nn.modules import dropout
from torch.nn.modules.sparse import Embedding

import vseq.modules
from vseq.data.tokens import START_TOKEN

from .base_module import BaseModule


class Bowman(BaseModule):
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_size: int):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.std_activation = nn.Softplus(beta=np.log(2))
        self.std_activation_inverse = vseq.modules.activations.InverseSoftplus(beta=np.log(2))

        self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim, padding_idx=num_embeddings)

        self.lstm_encode = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)

        self.h_to_q = nn.Linear(hidden_size, 2 * hidden_size)

        self.lstm_decode = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False)

        self.output = nn.Linear(hidden_size, num_embeddings)

        self.define_prior(hidden_size)

        # TODO WordDropout as module
        # TODO Likelihood as module
        # TODO Stochastic layer as module (incl. reparameterization (and KL?))

    def define_prior(self, *shape):
        """Define the prior as standard normal

        Since we apply Softplus to the scale paramteer we must do the inverse for the initial value.
        """
        mu = torch.zeros(*shape)
        sigma = torch.ones(*shape)
        log_scale = self.std_activation_inverse(sigma)

        prior_logits = torch.cat([mu, log_scale])
        if False:  # self.learn_prior:
            self.prior_logits = nn.Parameter(prior_logits)
        else:
            self.register_buffer("prior_logits", prior_logits)

    @property
    def prior(self):
        """Return the prior distribution without a batch dimension"""
        mu, sigma = self.logits_to_mu_and_sigma(self.prior_logits)
        return D.Normal(mu, sigma)

    def logits_to_mu_and_sigma(self, logits):
        """Convert logits to parameters for the Normal. We chunk on axis 0 or 1 depending on batching"""
        mu, log_scale = logits.chunk(2, dim=2)
        sigma = self.std_activation(log_scale)
        return mu, sigma

    def compute_kl(self, z, q_z):
        """Compute KL divergence from approximate posterior to prior by MC sampling"""
        return q_z.log_prob(z) - self.prior.log_prob(z)

    def infer(self, x):
        import IPython; IPython.embed(using=False)

        x[x == -1] = self.num_embeddings
        x = self.embedding(x)
        h, (h_n, c_n) = self.lstm_encode(x)

        # q(z|x)
        q_logits = self.h_to_q(h_n)
        mu, sigma = self.logits_to_mu_and_sigma(q_logits)
        q_z = D.Normal(mu, sigma)
        z = q_z.rsample()
        return z, q_z

    def generate(self, z=None, x=None, word_dropout_rate: float = 0.75):
        """Sample from prior if z is None, evaluate likelihood if x is not None"""
        import IPython; IPython.embed(using=False)

        if z is not None:
            h_0 = z 
        else:
            if x is None:
                h_0 = self.prior.rsample()
            else:
                h_0 = self.prior.rsample(x.size)
        c_0 = torch.zeros_like(h_0)

        x = x.prepend(START_TOKEN)
        mask = torch.bernoulli(torch.ones(x.shape[0]) * (1 - word_dropout_rate))
        h, (h_n, c_n) = self.lstm_decode(x, (h_0, c_0))

        # p(x|z)
        p_logits = self.output(h)
        p = D.Categorical(logits=p_logits)
        log_prob = p.log_prob(x) if x is not None else None
        return p, log_prob

    def forward(self, x, word_dropout_rate: float = 0.75):
        """Perform inference and generative passes on input x of shape (T, B, 1)"""
        import IPython; IPython.embed(using=False)

        z, q_z = self.infer(x)
        kl_divergence = self.compute_kl(z, q_z)
        p, log_prob = self.generate(x=x, z=z, word_dropout_rate=word_dropout_rate)
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
