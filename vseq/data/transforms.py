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
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class TextCleaner(Transform):
    def __init__(self, cleaner_fcn: Callable):
        super().__init__()
        self.cleaner_fcn = cleaner_fcn

    def forward(self, x):
        return self.cleaner_fcn(x)


class EncodeInteger(Transform):
    def __init__(self, tokenizer, token_map, prefix, suffix):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_map = token_map
        self.prefix = prefix
        self.suffix = suffix
 
    def forward(self, x: str):
        x = self.tokenizer(x)
        x = self.token_map.encode(x, prefix=self.prefix, suffix=self.suffix)
        return x

import torchvision
torchvision.transforms.Resize


class DecodeInteger(Transform):
    def __init__(self):
        super().__init__()
        raise NotImplementedError()
