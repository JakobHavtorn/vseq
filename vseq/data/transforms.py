import math

from typing import Callable, Optional

import torch
import torch.nn as nn


class Transform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Reshape(Transform):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, *self.shape)


class TextCleaner(Transform):
    def __init__(self, cleaner_fcn: Callable):
        super().__init__()
        self.cleaner_fcn = cleaner_fcn

    def forward(self, x):
        return self.cleaner_fcn(x)


class EncodeInteger(Transform):
    def __init__(self, tokenizer, token_map):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_map = token_map

    def forward(self, x: str):
        x = self.tokenizer(x)
        x = self.token_map.encode(x)
        return x


class DecodeInteger(Transform):
    def __init__(self):
        super().__init__()
        raise NotImplementedError()


class RandomSegment(Transform):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def forward(self, x):
        high = max(x.size(0) - self.length, 1)
        start_idx = torch.randint(low=0, high=high, size=(1,))
        return x[start_idx : start_idx + self.length]


class Scale(Transform):
    def __init__(self, low=-1, high=1, min_val=None, max_val=None):
        """Scale an input to be in [low, high] by normalizing with data min and max values"""
        super().__init__()
        assert (low is not None) == (high is not None), "must set both low and high or neither"
        self.low = low
        self.high = high
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x_min = self.min_val if self.min_val is not None else x.min()
        x_max = self.max_val if self.max_val is not None else x.max()

        x_scaled = (x - x_min) / (x_max - x_min)

        if self.low == 0 and self.high == 1:
            return x_scaled

        return self.low + x_scaled * (self.high - self.low)


class MuLawEncode(Transform):
    def __init__(self, bits: int = 8):
        """Encode PCM audio in [-1, 1] via µ-law companding to some number of bits (8 by default)"""
        super().__init__()
        self.bits = bits
        self.mu = 2 ** bits - 1
        self._divisor = math.log(self.mu + 1)

    def forward(self, x: torch.Tensor):
        return x.sign() * torch.log(1 + self.mu * x.abs()) / self._divisor


class MuLawDecode(Transform):
    def __init__(self, bits: int = 8):
        """Decode PCM audio via µ-law companding from some number of bits (8 by default)"""
        super().__init__()
        self.bits = bits
        self.mu = 2 ** bits - 1

    def forward(self, x: torch.Tensor):
        return x.sign() * (torch.exp(x.abs() * self._divisor) - 1) / self.mu


class Quantize(Transform):
    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        bits: int = 8,
        bins: Optional[int] = None,
        force_out_int64: bool = True,
    ):
        """Quantize a tensor of values between `low` and `high` using a number of `bits`.

        The return value is an integer tensor with values in [0, 2**bits - 1].

        If `bits` is 32 or smaller, the integer tensor is of type `IntTensor` (32 bits).
        If `bits` is 33 or larger, the integer tensor is of type `LongTensor` (64 bits).

        We can force `LongTensor` (64 bit) output if `force_out_int64` is `True`.

        Args:
            low (float, optional): [description]. Defaults to -1.0.
            high (float, optional): [description]. Defaults to 1.0.
            bits (int, optional): [description]. Defaults to 8.
            bins (Optional[int], optional): [description]. Defaults to None.
        """
        super().__init__()
        assert (bits is None) != (bins is None), "Must set one and only one of `bits` and `bins`"
        self.low = low
        self.high = high
        self.bits = bins // 8 if bits is None else bits
        self.bins = 2 ** bits if bins is None else bins
        self.boundaries = torch.linspace(start=-1, end=1, steps=bins)
        self.out_int32 = (self.bits <= 32) and (not force_out_int64)

    def forward(self, x: torch.Tensor):
        return torch.bucketize(x, self.boundaries, out_int32=self.out_int32, right=False)


class Binarize(Transform):
    def __init__(self, resample: bool = False, threshold: float = None):
        super().__init__()
        assert bool(threshold) != bool(resample), "Must set exactly one of threshold and resample"
        self.resample = resample
        self.threshold = threshold

    def forward(self, x):
        if self.resample:
            return torch.bernoulli(x)

        return x > self.threshold


class Dequantize(Transform):
    """Dequantize a quantized data point by adding uniform noise.

    Sppecifically, assume the quantized data is x in {0, 1, 2, ..., D} for some D e.g. 255 for int8 data.
    Then, the transformation is given by definition of the dequantized data z as

        z = x + u
        u ~ U(0, 1)

    where u is sampled uniform noise of same shape as x.

    The dequantized data is in the continuous interval [0, D + 1]

    If the value is to scaled subsequently, the maximum value attainable is hence D + 1 due to the uniform noise.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + torch.rand_like(x)
