from collections.abc import Iterable
from typing import Any, Union, List

import torch

from torch import Tensor


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


def hard_sigmoid(x, a: Union[float, Tensor] = 1 / 3):
    """Hard sigmoid function with variable slope.

    The variable slope is useful for annealing towards step function when estimating gradients via. a straight
    through estimator.

    Otherwise nn.Hardsigmoid() or F.hardsigmoid() may be more efficient.
    """
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output


@torch.jit.script
def reverse_sequences(x, x_sl):
    """Reverse a sequence keeping right padding untouched and in position.

    Note: This method only works with right padding (not left padding or a combination).

    Args:
        x (torch.Tensor): Padded sequences to reverse (B, T, *)
        x_sl (torch.Tensor): Sequence lengths

    Returns:
        torch.Tensor: Sequences reversed along time axis but with same padding as before
    """
    max_len = x_sl.max()
    padding = (max_len - x_sl).unsqueeze(1).to(x.device)
    reverse_ids = torch.arange(start=max_len - 1, end=-1, step=-1, device=x.device).expand(x.size(0), -1)
    indices = reverse_ids - padding
    indices[indices < 0] = indices[indices < 0] + max_len
    return torch.gather(x, 1, indices)


def sequence_mask(seq_lens: Union[list, torch.Tensor], max_len=None, dtype=torch.bool, device: torch.device = None):
    """
    Creates a binary sequence mask where all entries up to seq_lens are 1 and the remaining are 0.

    Args:
        seq_lens (Tensor): The sequence lengths from which to construct the mask. Should be shape N with dtype == int64.
        max_len (int): The temporal dimension of the sequence mask. If None, will use max of seq_lens.
        dtype (torch.dtype): The type of the mask. Default is torch.bool.

    Returns:
        Tensor: The sequence mask of shape (N, T).
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
    return torch.device("cpu")
