import math

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
    return (
        torch.log(sum_op(torch.exp(tensor - maximum), axis=axis, keepdim=False) + 1e-8)
        + maximum
    )


def hard_sigmoid(x, a: Union[float, Tensor] = 1 / 3):
    """Hard sigmoid function with variable slope.

    The variable slope is useful for annealing towards step function when estimating gradients via. a straight
    through estimator.

    Otherwise nn.Hardsigmoid() or F.hardsigmoid() may be more efficient.
    """
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output


# @torch.jit.script 
def reverse_sequences(x, x_sl, batch_first: bool = False):
    """Reverse a sequence keeping right padding untouched and in position.

    Note: This method only works with right padding (not left padding or a combination).

    Args:
        x (torch.Tensor): Padded sequences to reverse (T, B, *) (or (B, T, *) if `batch_first == True`)
        x_sl (torch.Tensor): Sequence lengths

    Returns:
        torch.Tensor: Sequences reversed along time axis but with same padding as before
    """
    if batch_first:
        x = x.permute(1, 0)

    max_len = x_sl.max()
    padding = (max_len - x_sl).unsqueeze(0).to(x.device)
    forward_ids = (
        torch.arange(0, max_len, 1, device=x.device).expand(x.size(1), -1).permute(1, 0)
    )
    reverse_ids = (
        torch.arange(max_len - 1, -1, -1, device=x.device)
        .expand(x.size(1), -1)
        .permute(1, 0)
        - padding
    )

    mask = reverse_ids < 0
    reverse_ids[mask] = forward_ids[mask]  # Do not reverse padding

    # Match shape with x as a view
    x_shape_singular_dims = reverse_ids.shape[:2] + (1,) * (
        x.ndim - 2
    )  # (T, B, 1, 1, ...)
    reverse_ids = reverse_ids.view(x_shape_singular_dims).expand(
        -1, -1, *x.size()[2:]
    )  # (T, B, *x.shape[2:])
    out = torch.gather(x, 0, reverse_ids)
    if batch_first:
        return out.permute(1, 0)
    return out


<<<<<<< HEAD
def sequence_mask(
    seq_lens: Union[list, torch.Tensor],
    max_len=None,
    dtype=torch.bool,
    device: torch.device = None,
):
=======
def sequence_mask(seq_lens: Union[list, torch.Tensor], stride: int = 1, max_len: int = None, dtype: torch.dtype = torch.bool, device: torch.device = None):
>>>>>>> clockwork-vae
    """
    Creates a binary sequence mask where all entries up to seq_lens are 1 and the remaining are 0.
    Args:
        seq_lens (Tensor): The sequence lengths from which to construct the mask. Should be shape N with dtype == int64.
        stride (int): 
        max_len (int): The temporal dimension of the sequence mask. If None, will use max of seq_lens.
        dtype (torch.dtype): The type of the mask. Default is torch.bool.
    Returns:
        Tensor: The sequence mask of shape (N, T).
    """
    if not isinstance(seq_lens, torch.Tensor):
        seq_lens = (
            torch.LongTensor(seq_lens)
            if device == torch.device("cpu")
            else torch.cuda.LongTensor(seq_lens)
        )
    device = seq_lens.device if device is None else device
    if device != seq_lens.device:
        seq_lens = seq_lens.to(device)
    N = seq_lens.size(0)
    T = max_len or math.ceil(seq_lens.max() / stride)
    seq_mask = torch.arange(T, device=device).unsqueeze(0).repeat((N, 1)) < seq_lens.unsqueeze(1)
    return seq_mask.to(dtype)


def update_running_variance(avg_a: Union[torch.Tensor, float], avg_b, w_a, w_b, M2_a, M2_b):
    """Online variance update c.f. parallel variance algorithm at [1].

    [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    w = w_a + w_b
    delta = avg_b - avg_a
    M2 = M2_a + M2_b + delta ** 2 * w_a * w_b / w
    var = M2 / (w - 1)
    avg = (w_a * avg_a + w_b * avg_b) / w
    return var, avg, w, M2


def detach(x: Union[torch.Tensor, Any]):
    """Detach a tensor from the computational graph"""
    if isinstance(x, torch.Tensor):
        return x.detach()

    return x


def detach_to_device(
    x: Union[torch.Tensor, float, List[float], None], device: torch.device
):
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
