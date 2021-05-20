from collections.abc import Iterable
from typing import Any, Union, List

import torch

from torch import Tensor

from .shape import elevate_sample_dim


def reduce_batch(tensor, batch_dim=0, reduction=torch.sum):
    """Reduce the batch dimension.

    Given (D*, B, D*) returns (D*, D*)
    """
    return reduction(tensor, axis=batch_dim)


def reduce_samples(tensor, batch_dim=0, sample_dim=0, n_samples=None, reduction=torch.sum):
    """Reduce the posterior or prior samples whether they are in a separate dimension or in the batch_dimension.

    Given shape (B, S, D*) returns (B, D*).
    Given shape (B * S, D*) returns (B, D*).

    Typically, the D dimensions will not exist.
    """
    if batch_dim == sample_dim:
        if n_samples is None:
            raise ValueError(f"When 'batch_dim'=='sample_dim' ({batch_dim}), the number of samples must be supplied")
        if batch_dim != 0:
            raise ValueError(f"Cannot reduce samples for 'batch_dim' other than 0 but got '{batch_dim=}'")

        tensor = elevate_sample_dim(tensor, n_samples)
        batch_dim = batch_dim + 1
        sample_dim = 0

    return reduction(tensor, dim=sample_dim)


def reduce_to_batch(tensor, batch_dim=0, reduction=torch.sum):
    """Assuming that the batch dimension is the left-most dimension, reduce all others by summation.

    Given shape (B, D*) returns (B,).
    """
    reduce_dims = list(range(tensor.ndim))
    if not reduce_dims:
        return tensor

    reduce_dims.remove(batch_dim)

    if not reduce_dims:
        return tensor
    return reduction(tensor, dim=reduce_dims)


def reduce_to_latent(tensor, batch_dim=0, latent_dim=1, reduction=torch.sum):
    """Assuming that the batch and latent dimensions are 0 and 1, respectively, reduce all others by summation.

    Given shape (B, L, D*) returns (B, L).
    """
    reduce_dims = list(range(tensor.ndim))
    if not reduce_dims:
        return tensor

    reduce_dims.remove(batch_dim)
    reduce_dims.remove(latent_dim)

    if not reduce_dims:
        return tensor
    return reduction(tensor, dim=reduce_dims)


def log_sum_exp(tensor, axis=-1, dim=None, sum_op=torch.mean):
    """Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.

    :param tensor: Tensor to compute LSE over
    :param axis: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    axis = dim if dim is not None else axis
    maximum, _ = torch.max(tensor, axis=axis, keepdim=False)
    return torch.log(sum_op(torch.exp(tensor - maximum), axis=axis, keepdim=False) + 1e-8) + maximum


def hard_sigmoid(x, a: Union[float, Tensor] = 1/3):
    """Hard sigmoid function with variable slope.

    The variable slope is useful for annealing towards step function when estimating gradients via. a straight
    through estimator.

    Otherwise nn.Hardsigmoid() or F.hardsigmoid() may be more efficient.
    """
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output


def sequence_mask(seq_lens: Union[list, torch.Tensor], max_len=None, dtype=torch.bool, device: torch.device = None):
    """
    Creates a binary sequence mask where all entries up to seq_lens are 1 and the remaining are 0.

    Args:
        seq_lens (Tensor): The sequence lengths from which to construct the mask. Should be shape N with dtype == int64.
        max_len (int): The temporal dimension of the sequence mask. If None, will use max of seq_lens.
        dtype (torch.dtype): The type of the mask. Default is torch.bool.

    Returns:
        Tensor: The sequence mask of shape NT.
    """
    if isinstance(seq_lens, torch.Tensor):
        device = seq_lens.device if device is None else device
        if device != seq_lens.device:
            seq_lens = seq_lens.to(device)
    else:
        seq_lens = torch.tensor(seq_lens, device=device, dtype=int)

    N = seq_lens.size(0)
    T = max_len or seq_lens.max()
    seq_mask = torch.arange(T, device=device).unsqueeze(0).repeat((N, 1)) < seq_lens.unsqueeze(1)
    return seq_mask.to(dtype)


def detach(x: Union[torch.Tensor, Any]):
    """Detach a tensor from the computational graph"""
    if isinstance(x, torch.Tensor):
        return x.detach()

    return x


def detach_to_device(x: Union[torch.Tensor, float, List[float], None], device: torch.device):
    """Detach a tensor from the computational graph, clone and place it on the given device"""
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(device)

    return torch.tensor(x, device=device, dtype=torch.float)


def infer_device(x: Any):
    """Infer the device of any object (CPU for any non-torch object)"""
    if isinstance(x, torch.Tensor):
        return x.device
    return torch.device('cpu')
