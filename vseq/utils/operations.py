import torch

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
    """Assuming that the batch and latent dimensions are 0 and 1, respectively, reduce all others by sumnmation.

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


def detach(x):
    """Detach a tensor from the computational graph"""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach()

    return x


def detach_to_device(x, device):
    """Detach a tensor from the computational graph, clone and place it on the given device"""
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(device)

    return torch.tensor(x, device=device, dtype=torch.float)
