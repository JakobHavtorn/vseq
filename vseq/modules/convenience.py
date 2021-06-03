import torch
import torch.nn as nn
import torch.nn.functional as F


class Permute(nn.Module):
    """nn.Module wrapper of Tensor.permute"""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def __repr__(self):
        return f"Permute({self.dims})"


class View(nn.Module):
    """nn.Module wrapper of Tensor.view"""

    def __init__(self, shape, n_batch_dims=1):
        super().__init__()
        self.shape = shape
        self.n_batch_dims = n_batch_dims

    def forward(self, x):
        return x.view(*x.shape[: self.n_batch_dims], *self.shape)

    def extra_repr(self):
        return f"n_batch_dims={self.n_batch_dims}, shape={self.shape}"


class Clamp(nn.Module):
    """nn.Module wrapper of Tensor.clamp"""

    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min if min is not None else -float("inf")
        self.max = max if max is not None else float("inf")

    def forward(self, tensor):
        return tensor.clamp(min=self.min, max=self.max)

    def __repr__(self):
        return f"Clamp({self.min, self.max})"


class Chunk(nn.Module):
    """nn.Module wrapper of torch.chunk"""
    def __init__(self, chunks: int, dim: int = -1):
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return torch.chunk(x, self.chunks, dim=self.dim)


class AddConstant(nn.Module):
    """Adds a `constant` to the input"""

    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, tensor1):
        return tensor1 + self.constant

    def __repr__(self):
        return f"AddConstant({self.constant})"
