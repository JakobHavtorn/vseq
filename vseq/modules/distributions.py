import math

from typing import Optional

import torch
import torch.nn as nn

from torchtyping import TensorType

from vseq.utils.log_likelihoods import gaussian_ll, categorical_ll, bernoulli_ll, discretized_logistic_mixture_ll
from vseq.utils.variational import rsample_discretized_logistic_mixture

from .convenience import AddConstant


# TODO Add reduce method that can reduce a log-likelihood along a single or multiple dimensions with a given operation (sum, mean etc.)


class ConditionalDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_distribution(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def sample(logits):
        raise NotImplementedError()

    @staticmethod
    def rsample(logits):
        raise NotImplementedError()

    @staticmethod
    def mode(logits):
        raise NotImplementedError()

    def log_prob(self, x):
        raise NotImplementedError()


class GaussianDense(ConditionalDistribution):
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
    def sample(logits):
        return torch.distributions.Normal(loc=logits[0], scale=logits[1]).sample()

    @staticmethod
    def rsample(logits):
        return torch.distributions.Normal(loc=logits[0], scale=logits[1]).rsample()

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


class CategoricalDense(ConditionalDistribution):
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
    def sample(logits):
        return torch.distributions.Categorical(logits=logits).sample()

    @staticmethod
    def mode(logits, dim: int = -1):
        return torch.argmax(logits, dim=dim)

    def log_prob(self, y, logits):
        return categorical_ll(y, logits)

    def forward(self, x):
        return self.logits(x)


class BernoulliDense(ConditionalDistribution):
    def __init__(self, x_dim, y_dim, reduce_dim: int = -1):
        """Parameterizes a Bernoulli distribution"""
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

    @staticmethod
    def sample(logits):
        return torch.distributions.Bernoulli(logits=logits).sample()

    def mode(self, logits):
        return torch.argmax(logits, dim=self.reduce_dim)

    def log_prob(self, y, logits):
        return bernoulli_ll(y, logits).sum(self.reduce_dim)

    def forward(self, x):
        return self.logits(x)


class PolarCoordinatesSpectrogram(ConditionalDistribution):
    def __init__(self, x_dim: int, y_dim: int, num_mix: int = 10, num_bins: int = 256, initial_concentration: float = 1):
        super().__init__()
        self.von_mises = VonMisesDense(
            x_dim=x_dim,
            y_dim=y_dim,
            initial_concentration=initial_concentration,
            reduce_dim=None,
        )
        self.discretized_logistic_mixture = DiscretizedLogisticMixtureDense(
            x_dim=x_dim,
            y_dim=y_dim,
            num_mix=num_mix,
            num_bins=num_bins,
        )

    def reset_parameters(self):
        pass

    @staticmethod
    def get_distribution(logits):
        raise NotImplementedError()

    def sample(self, logits):
        sample_dlm = self.discretized_logistic_mixture.sample(logits[0])
        sample_vm = self.von_mises.sample(logits[1])
        return torch.stack([sample_dlm, sample_vm], dim=1)

    def mode(self, logits):
        mode_dlm = self.discretized_logistic_mixture.mode(logits[0])
        mode_vm = self.von_mises.mode(logits[1])
        return torch.stack([mode_dlm, mode_vm], dim=1)

    def log_prob(self, stft: TensorType["B", 2, "N", "T"], logits: tuple):
        stft_r, stft_phi = stft.chunk(2, dim=1)
        stft_r, stft_phi = stft_r.squeeze(1), stft_phi.squeeze(1)
        log_prob_dlm = self.discretized_logistic_mixture.log_prob(stft_r, logits[0])
        log_prob_vm = self.von_mises.log_prob(stft_phi, logits[1])
        return log_prob_dlm + log_prob_vm

    def forward(self, x):
        logits_dlm = self.discretized_logistic_mixture(x)
        logits_vm = self.von_mises(x)
        return (logits_dlm, logits_vm)


class VonMisesDense(ConditionalDistribution):
    def __init__(self, x_dim, y_dim, initial_concentration: float = 1, reduce_dim: Optional[int] = None):
        """Parameterizes a Von Mises distribution"""
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.initial_concentration = initial_concentration
        self.reduce_dim = reduce_dim

        self.logits = nn.Linear(x_dim, 2 * y_dim)

        self.sd_activation = nn.Sequential(nn.Softplus(beta=math.log(2) / initial_concentration))

        self.reset_parameters()

    def reset_parameters(self):
        pass

    @staticmethod
    def get_distribution(logits):
        return torch.distributions.VonMises(loc=logits[0], concentration=logits[1])

    @staticmethod
    def sample(logits):
        return torch.distributions.VonMises(loc=logits[0], concentration=logits[1]).sample()

    @staticmethod
    def mode(logits):
        return logits[0]

    def log_prob(self, y, logits):
        if self.reduce_dim is not None:
            return self.get_distribution(logits).sum(self.reduce_dim)
        return self.get_distribution(logits)

    def forward(self, x):
        logits = self.logits(x)
        mu, log_concentration = logits.chunk(2, dim=-1)
        concentration = self.sd_activation(log_concentration)
        return mu, concentration


class DiscretizedLogisticMixtureDense(ConditionalDistribution):
    def __init__(self, x_dim: int, y_dim: int, num_mix: int = 10, num_bins: int = 256, reduce_dim: Optional[int] = None):
        """Discretized Logistic Mixture distribution.

        The distribution has the following parameters:

        - Mean value per mixture: `num_mix`.
        - Log-scale per mixture: `num_mix`.
        - Mixture coefficient per mixture: `num_mix`.

        This yields a total of `3 * num_mix` parameters. 
        This is different to the Discretized Mixture of Logistics used the PixelCNN++ paper which is tailored
        for RGB images and treats the channel dimension in a speciail way. There are no such special dimensions here.

        Assumes input data to be originally uint8 (0, ..., num_bins) and then rescaled
        by 1/num_bins: discrete values in {0, 1/num_bins, ..., num_bins/num_bins}.

        When using the original discretized logistic mixture logprob implementation,
        this data should be rescaled to be in the interval [-1, 1].

        Mean and mode are not implemented for now.

        Args:
            x_dim (int): Number of channels in the input
            y_dim (int): Number of channels in the output
            num_mix (int, optional): Number of components. Defaults to 10.
            num_bins (int, optional): Number of quantization bins. Defaults to 256 (8 bit).
        """
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_mix = num_mix
        self.num_bins = num_bins
        self.reduce_dim = reduce_dim

        self.out_features = (y_dim * 3) * num_mix

        self.logits = nn.Linear(x_dim, self.out_features)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    @staticmethod
    def get_distribution(logits):
        raise NotImplementedError("Discretized mixture of logistics does not have a Distribution object (yet)")

    def rsample(self, logits):
        return rsample_discretized_logistic_mixture(logits, num_mix=self.num_mix)

    @torch.no_grad()
    def sample(self, logits):
        return rsample_discretized_logistic_mixture(logits, num_mix=self.num_mix)

    def mode(self, logits):
        raise NotImplementedError()

    def log_prob(self, y, logits):
        """Compute log-likelihood. Inputs are assumed to be [-1, 1]"""
        log_prob = discretized_logistic_mixture_ll(y, logits, num_bins=self.num_bins)
        if self.reduce_dim is not None:
            return log_prob.sum(self.reduce_dim)
        return log_prob

    def forward(self, x):
        return self.logits(x)
