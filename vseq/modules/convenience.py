import torch.nn as nn

from torch import nn
from torch.nn import functional as F


class Interpolate(nn.Module):
    """Wrapper for torch.nn.functional.interpolate.

    Down/up samples the input to either the given `size` or the given `scale_factor`.

    The algorithm used for interpolation is determined by :attr:mode.

    Currently temporal, spatial and volumetric sampling are supported, i.e. expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form: `mini-batch x channels x [optional depth] x [optional height] x width`

    TODO This layer is depracted in favour of nn.Upsample
    """

    def __init__(self, size=None, scale=None, mode="bilinear", align_corners=False):
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = F.interpolate(
            x, size=self.size, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners
        )
        return out


class Permute(nn.Module):
    """Wrapper around torch.permute but ignoring batch dimension"""

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(0, *self.dims)

    def __repr__(self):
        return f"Permute({self.dims})"


class View(nn.Module):
    """Module that returns a view of an input"""

    def __init__(self, shape, n_batch_dims=1):
        super().__init__()
        self.shape = shape
        self.n_batch_dims = n_batch_dims

    def forward(self, x):
        return x.view(*x.shape[: self.n_batch_dims], *self.shape)

    def extra_repr(self):
        return f"n_batch_dims={self.n_batch_dims}, shape={self.shape}"


class AddConstant(nn.Module):
    """Adds a `constant` to the input"""

    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, tensor1):
        return tensor1 + self.constant

    def __repr__(self):
        return f"AddConstant({self.constant})"


class Clamp(nn.Module):
    """Clamps the input to be between min and max"""

    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min if min is not None else -float("inf")
        self.max = max if max is not None else float("inf")

    def forward(self, tensor):
        return tensor.clamp(min=self.min, max=self.max)

    def __repr__(self):
        return f"Clamp({self.min, self.max})"
