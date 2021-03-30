from typing import Callable, List, Any
from vseq.data.collate import collate_spectrogram

import numpy as np
import torch
import torch.nn as nn
import torchaudio, torchvision


# TODO Where do we place the `collate` function?
# TODO Data types can define extensions and collate functions
# TODO In datasets, for inputs we care only about extensions and for outputs we care only about collate functions.
# TODO So is the dataset abstraction artificial?


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


class Scale(nn.Module):
    def __init__(self, low=0, high=1, min_val=None, max_val=None):
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


class Binarize(nn.Module):
    def __init__(self, resample: bool = False, threshold: float = None):
        super().__init__()
        assert bool(threshold) != bool(resample), "Must set exactly one of threshold and resample"
        self.resample = resample
        self.threshold = threshold

    def forward(self, x):
        if self.resample:
            return torch.bernoulli(x)

        return x > self.threshold


class Dequantize(nn.Module):
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
