import math

from typing import Optional

import torch
import torch.nn as nn

from vseq.utils.log_likelihoods import gaussian_ll, categorical_ll, bernoulli_ll

from .convenience import AddConstant


class Distribution(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_distribution(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def mode(logits):
        raise NotImplementedError()

    def log_prob(self, x):
        raise NotImplementedError()


class GaussianDense(Distribution):
    def __init__(self, x_dim, y_dim, initial_sd: float = 1, epsilon: float = 1e-6, reduce_dim: Optional[int] = None):
        """Parameterizes a Gaussian distribution with diagonal covariance"""
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.initial_sd = initial_sd
        self.epsilon = epsilon
        self.reduce_dim = reduce_dim

        self.logits = nn.Linear(x_dim, 2 * y_dim)

        if epsilon > 0:
            self.sd_activation = nn.Sequential(nn.Softplus(beta=math.log(2) / initial_sd), AddConstant(epsilon))
        else:
            self.sd_activation = nn.Sequential(nn.Softplus(beta=math.log(2) / initial_sd))

        self.reset_parameters()

    def reset_parameters(self):
        pass

    @staticmethod
    def get_distribution(logits):
        return torch.distributions.Normal(loc=logits[0], scale=logits[1])

    @staticmethod
    def mode(logits):
        return logits[0]

    def log_prob(self, y, logits):
        if self.reduce_dim is not None:
            return gaussian_ll(y, logits[0], logits[1] ** 2, epsilon=0).sum(self.reduce_dim)
        return gaussian_ll(y, logits[0], logits[1] ** 2, epsilon=0)

    def forward(self, x):
        logits = self.logits(x)
        mu, log_sd = logits.chunk(2, dim=-1)
        sd = self.sd_activation(log_sd)
        return mu, sd


class CategoricalDense(Distribution):
    def __init__(self, x_dim, y_dim):
        """Parameterizes a Gaussian distribution with diagonal covariance"""
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.logits = nn.Linear(x_dim, y_dim)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    @staticmethod
    def get_distribution(logits):
        return torch.distributions.Categorical(logits=logits)

    @staticmethod
    def mode(logits, dim: int = -1):
        return torch.argmax(logits, dim=dim)

    def log_prob(self, y, logits):
        return categorical_ll(y, logits)

    def forward(self, x):
        return self.logits(x)


class BernoulliDense(Distribution):
    def __init__(self, x_dim, y_dim, reduce_dim: int = -1):
        """Parameterizes a Gaussian distribution with diagonal covariance"""
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.reduce_dim = reduce_dim

        self.logits = nn.Linear(x_dim, y_dim)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    @staticmethod
    def get_distribution(logits):
        return torch.distributions.Bernoulli(logits=logits)

    def mode(self, logits):
        return torch.argmax(logits, dim=self.reduce_dim)

    def log_prob(self, y, logits):
        return bernoulli_ll(y, logits).sum(self.reduce_dim)

    def forward(self, x):
        return self.logits(x)
