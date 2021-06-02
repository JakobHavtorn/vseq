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


def discount_free_nats(
    kl: TensorType["B":..., "shared":...],
    free_bits: float = None,
    shared_dims: Union[Tuple[int], int] = None,
) -> torch.Tensor:
    """Free bits as introduced in [1] but renamed to free nats because that's what it really is with log_e.

    In the paper they divide all latents Z into K groups. This implementation assumes a that each KL tensor passed
    to __call__ is one such group.

    The KL tensor may have more dimensions than the batch dimension, in which case the free_bits budget can be
    distributed across those dimensions by setting `shared_dims`. E.g. if `kl.shape` is (32, 10) and `shared_dims` is -1
    each of the 10 elements in the last dimension will get 1/10 of the free nats budget.
    If `kl.shape` is (32, 10, 10) and `shared_dims` is (-2, -1) each of the 10*10=100 elements will get 1 / 100th.

    The returned free nats KL is equal to max(kl, freebits_per_dim, dim=shared_dims)

    [1] https://arxiv.org/pdf/1606.04934
    """
    if free_bits == 0 or free_bits is None:
        return kl

    if isinstance(shared_dims, int):
        shared_dims = (shared_dims,)

    # equally divide free nats budget over the elements
    n_elements = math.prod(shared_dims) if shared_dims is not None else 1
    min_kl_per_dim = free_bits / n_elements

    min_kl_per_dim = torch.tensor(min_kl_per_dim, dtype=kl.dtype, device=kl.device)
    freenats_kl = torch.maximum(kl, min_kl_per_dim)
    return freenats_kl
