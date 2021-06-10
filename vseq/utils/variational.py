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
def kl_divergence_gaussian(mu_q, sd_q, mu_p, sd_p):
    """Elementwise analytical KL divergence between two Gaussian distributions KL(q||p) (no reduction applied)."""
    return sd_p.log() - sd_q.log() + (sd_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sd_p.pow(2)) - 0.5


@torch.jit.script
def rsample_gaussian(mu, sd):
    """Return a reparameterized sample from a given Gaussian distribution.

    Args:
        mu (torch.Tensor): Gaussian mean
        sd (torch.Tensor): Gaussian standard deviation

    Returns:
        torch.Tensor: Reparameterized sample
    """
    return torch.randn_like(sd).mul(sd).add(mu)


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
