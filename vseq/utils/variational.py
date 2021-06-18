import math

from typing import Union, Tuple

import torch

from torchtyping import TensorType


def kl_divergence_mc(
    q_distrib: torch.distributions.Distribution, p_distrib: torch.distributions.Distribution, z: torch.Tensor
):
    """Elementwise Monte-Carlo estimation of KL between two distributions KL(q||p) (no reduction applied).

    Any number of dimensions works via broadcasting and correctly set `event_shape` (be careful).

    Args:
        z: Sample or samples from the variational distribution `q_distrib`.
        q_distrib: Variational distribution.
        p_distrib: Target distribution.

    Returns:
        tuple: Spatial KL divergence and log-likelihood of samples under q and under p (torch.Tensor)
    """
    q_logprob = q_distrib.log_prob(z)
    p_logprob = p_distrib.log_prob(z)
    kl_dwise = q_logprob - p_logprob
    return kl_dwise, q_logprob, p_logprob


@torch.jit.script
def kl_divergence_gaussian(mu_q: torch.Tensor, sd_q: torch.Tensor, mu_p: torch.Tensor, sd_p: torch.Tensor):
    """Elementwise analytical KL divergence between two Gaussian distributions KL(q||p) (no reduction applied)."""
    return sd_p.log() - sd_q.log() + (sd_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sd_p.pow(2)) - 0.5


@torch.jit.script
def rsample_gaussian(mu: torch.Tensor, sd: torch.Tensor):
    """Return a reparameterized sample from a given Gaussian distribution.

    Args:
        mu (torch.Tensor): Gaussian mean
        sd (torch.Tensor): Gaussian standard deviation

    Returns:
        torch.Tensor: Reparameterized sample
    """
    return torch.randn_like(sd).mul(sd).add(mu)


@torch.jit.script
def rsample_logistic(mu: torch.Tensor, log_scale: torch.Tensor, eps: float = 1e-8):
    """
    Returns a sample from Logistic with specified mean and log scale.

    :param mu: a tensor containing the mean.
    :param log_scale: a tensor containing the log scale.
    :return: a reparameterized sample with the same size as the input mean and log scale.
    """
    u = torch.zeros_like(mu).uniform_(eps, 1 - eps)  # uniform sample in the interval (eps, 1 - eps)
    sample = mu + torch.exp(log_scale) * (torch.log(u) - torch.log(1 - u))  # transform to logistic
    return sample


# @torch.jit.script
def rsample_discretized_logistic(mu: torch.Tensor, log_scale: torch.Tensor, eps: float = 1e-8):
    """Return a sample from a discretized logistic with values standardized to be in [-1, 1]
    
    This is done by sampling the corresponding continuous logistic and clamping values outside\
    the interval to the endpoints.
    """
    return rsample_logistic(mu, log_scale, eps).clamp(-1, 1)


def rsample_discretized_logistic_mixture(
    logit_probs: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    num_mix: int,
    eps: float = 1e-5,
    t: float = 1.0,
):
    """Return a reparameterized sample from a given Discretized Logistic Mixture distribution.

    Code taken from PyTorch adaptation of original PixelCNN++ TensorFlow implementation:
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py but does not include the channel specific conditional modelling.

    Args:
        logit_probs (torch.Tensor): (*, D, num_mix)
        means (torch.Tensor): (*, D, num_mix)
        log_scales (torch.Tensor): (*, D, num_mix)
        num_mix (int): Number of mixture components
        eps (float): Bounds [eps, 1-eps] on the uniform rv used to sample the mixture coefficients and the logistic.
        t (float): Temperature for Gumbel sampling

    Returns:
        torch.Tensor: Sample from the DLM `(*, D)`
    """
    # sample mixture indicator from softmax
    gumbel = -torch.log(-torch.log(torch.empty_like(means).uniform_(eps, 1.0 - eps)))
    argmax = torch.argmax(logit_probs / t + gumbel, dim=-1)
    one_hot = torch.nn.functional.one_hot(argmax, num_mix)

    # select logistic parameters
    means = torch.sum(means * one_hot, dim=-1)
    log_scales = torch.sum(log_scales * one_hot, dim=-1)

    # sample from logistic (we don't actually round to the nearest 8bit value)
    u = torch.empty_like(means).uniform_(eps, 1.0 - eps)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))

    # Enforce standardization
    x = x.clamp(min=-1, max=1)
    return x


def rsample_discretized_logistic_mixture_rgb(parameters: torch.Tensor, num_mix: int, eps: float = 1e-5, t: float = 1.0):
    """Return a reparameterized sample from a given Discretized Logistic Mixture distribution for RGB images.

    Code taken from PyTorch adaptation of original PixelCNN++ TensorFlow implementation:
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py.

    Args:
        parameters (torch.Tensor): Mixture coefficients, means and log-scales for logistic mixtures `(*, D * 3 * num_mix)`.
        num_mix (int): Number of mixture components in the the DLM.
        eps (float): Bounds [eps, 1-eps] on the uniform rv used to sample the mixture coefficients and the logistic.

    Returns:
        torch.Tensor: Sample from the DLM `(*, D)`
    """
    # TODO Streamline
    B, C, H, W = parameters.size()

    # unpack parameters
    logit_probs = parameters[:, :num_mix, :, :]  # B, M, H, W
    l = parameters[:, num_mix:, :, :].view(B, 3, 3 * num_mix, H, W)  # B, 3, 3 * M, H, W
    means = l[:, :, :num_mix, :, :]  # B, 3, M, H, W
    log_scales = torch.clamp(l[:, :, num_mix : 2 * num_mix, :, :], min=-7.0)  # B, 3, M, H, W
    coeffs = torch.tanh(l[:, :, 2 * num_mix : 3 * num_mix, :, :])  # B, 3, M, H, W

    gumbel = -torch.log(-torch.log(torch.empty_like(logit_probs).uniform_(eps, 1.0 - eps)))  # B, M, H, W
    argmax = torch.argmax(logit_probs / t + gumbel, 1)
    one_hot = torch.nn.functional.one_hot(argmax, num_mix, dim=1)  # B, M, H, W
    one_hot = one_hot.unsqueeze(1)  # B, 1, M, H, W

    # select logistic parameters
    means = torch.sum(means * one_hot, dim=2)  # B, 3, H, W
    log_scales = torch.sum(log_scales * one_hot, dim=2)  # B, 3, H, W
    coeffs = torch.sum(coeffs * one_hot, dim=2)  # B, 3, H, W

    # cells from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.Tensor(means.size()).uniform_(eps, 1.0 - eps).cuda()  # B, 3, H, W
    x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1.0 - u))  # B, 3, H, W

    x0 = torch.clamp(x[:, 0, :, :], -1, 1.0)  # B, H, W
    x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)  # B, H, W
    x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

    x = torch.stack([x0, x1, x2], dim=1)
    return x


def discount_free_nats(
    kl: TensorType["B":..., "shared":...],
    free_nats: float = None,
    shared_dims: Union[Tuple[int], int] = None,
) -> torch.Tensor:
    """Free bits as introduced in [1] but renamed to free nats because that's what it really is with log_e.

    In the paper they divide all latents Z into K groups. This implementation assumes a that each KL tensor passed
    to __call__ is one such group.

    By default, this method discounts `free_nats` units of nats elementwise in the KL regardless of its shape.

    If the KL tensor has more dimensions than the batch dimension, the free_nats budget can be optionally
    shared across those dimensions by setting `shared_dims`. E.g. if `kl.shape` is (32, 10) and `shared_dims` is -1,
    each of the 10 elements in the last dimension will get 1/10th of the free nats budget. If `kl.shape` is (32, 10, 10)
    and `shared_dims` is (-2, -1) each of the 10*10=100 elements will get 1 / 100th.

    The returned KL with `free_nats` discounted is equal to max(kl, freebits_per_dim)

    [1] https://arxiv.org/pdf/1606.04934
    """
    if free_nats is None or free_nats == 0:
        return kl

    if isinstance(shared_dims, int):
        shared_dims = (shared_dims,)

    # equally divide free nats budget over the elements in shared_dims
    if shared_dims is not None:
        n_elements = math.prod([kl.shape[d] for d in shared_dims])
        min_kl_per_dim = free_nats / n_elements
    else:
        min_kl_per_dim = free_nats

    min_kl_per_dim = torch.tensor(min_kl_per_dim, dtype=kl.dtype, device=kl.device)
    freenats_kl = torch.maximum(kl, min_kl_per_dim)
    return freenats_kl
