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


def logistic_rsample(mu: torch.Tensor, log_scale: torch.Tensor, eps: float = 1e-8):
    """
    Returns a sample from Logistic with specified mean and log scale.

    :param mu_ls: a tensor containing mean and log scale along dim=1,
            or a tuple (mean, log scale)
    :return: a reparameterized sample with the same size as the input
            mean and log scale
    """
    # Get parameters
    scale = log_scale.exp()

    # Get uniform sample in open interval (0, 1)
    u = torch.zeros_like(mu)
    u.uniform_(eps, 1 - eps)

    # Transform into logistic sample
    sample = mu + scale * (torch.log(u) - torch.log(1 - u))

    return sample


def rsample_discretized_logistic_mixture(logits, num_mix: int, eps: float = 1e-8):
    """Return a reparameterized sample from a given Discretized Logistic Mixture distribution.

    Code taken from PyTorch adaptation of original PixelCNN++ TensorFlow implementation:
    https://github.com/pclucas14/pixel-cnn-pp

    Args:
        logits (torch.Tensor): Mixture coefficients, means and log-scales for logistic mixtures `(*, D * 3 * num_mix)`.
        num_mix (int): Number of mixture components in the the DLM.
        eps (float): Bounds [eps, 1-eps] on the uniform rv used to sample the mixture coefficients and the logistic.

    Returns:
        torch.Tensor: Sample from the DLM `(*, D)`
    """
    ls = [int(y) for y in logits.size()]
    x_dim = int(ls[-1] / (3 * num_mix))
    xs = ls[:-1] + [x_dim]

    # unpack parameters
    logits = logits.view(xs + [num_mix * 3])  # 3 for mean, scale, coefficients (D, 3 x num_mix)
    coeffients = logits[..., :num_mix]
    means = logits[..., :num_mix]
    log_scales = logits[..., num_mix : 2 * num_mix].clamp(min=-7.0)

    # sample mixture indicator from softmax
    temp = torch.empty_like(coeffients)
    temp.uniform_(eps, 1.0 - eps)
    temp = coeffients.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=-1)
    one_hot = torch.nn.functional.one_hot(argmax, num_mix)

    # select logistic parameters
    means = torch.sum(logits[..., :num_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(torch.sum(logits[..., num_mix : 2 * num_mix] * one_hot, dim=-1), min=-7.0)

    # sample from logistc
    u = torch.empty_like(means)
    u.uniform_(eps, 1.0 - eps)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))

    # Enforce normalization
    x = x.clamp(min=-1, max=1)
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
