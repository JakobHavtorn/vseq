import torch
import torch.nn as nn
import torch.nn.functional as F


class Permute(nn.Module):
    def __init__(self, *dims):
        """nn.Module wrapper of Tensor.permute"""
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def __repr__(self):
        return f"Permute({self.dims})"


class View(nn.Module):
    def __init__(self, shape, n_batch_dims=1):
        """nn.Module wrapper of Tensor.view"""
        super().__init__()
        self.shape = shape
        self.n_batch_dims = n_batch_dims

    def forward(self, x):
        return x.view(*x.shape[: self.n_batch_dims], *self.shape)

    def extra_repr(self):
        return f"n_batch_dims={self.n_batch_dims}, shape={self.shape}"


class Clamp(nn.Module):
    def __init__(self, min=None, max=None):
        """nn.Module wrapper of Tensor.clamp"""
        super().__init__()
        self.min = min if min is not None else -float("inf")
        self.max = max if max is not None else float("inf")

    def forward(self, tensor):
        return tensor.clamp(min=self.min, max=self.max)

    def __repr__(self):
        return f"Clamp({self.min, self.max})"


class Chunk(nn.Module):
    def __init__(self, chunks: int, dim: int = -1):
        """nn.Module wrapper of torch.chunk"""
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return torch.chunk(x, self.chunks, dim=self.dim)


class AddConstant(nn.Module):
    def __init__(self, constant):
        """Adds a `constant` to the input"""
        super().__init__()
        self.constant = constant

    def forward(self, tensor1):
        return tensor1 + self.constant

    def __repr__(self):
        return f"AddConstant({self.constant})"
