from typing import List, Any
from vseq.data.collate import collate_spectrogram

import numpy as np
import torch
import torch.nn as nn
import torchaudio


# TODO Where do we place the `collate` function?
# TODO Data types can define extensions and collate functions
# TODO In datasets, for inputs we care only about extensions and for outputs we care only about collate functions.
# TODO So is the dataset abstraction artificial?


class Transform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()


# class Compose(Transform):
#     def __init__(self, *transforms):
#         self.transforms = transforms
#         self.collate = self.get_collate(transforms)

#     def forward(self, x):
#         for transform in self.transforms:
#             x = transform(x)
#         return x

#     def get_collate(self, transforms):
#         for transform in reversed(transforms):
#             if hasattr(transform, 'collate'):
#                 return transform.collate

#         raise AttributeError('No given transforms have the required `collate` function.')


# class MelSpectrogram(Transform, torchaudio.transforms.MelSpectrogram):
#     def collate(self, batch):
#         return collate_spectrogram



class EncodeInteger(Transform):
    def __init__(self):
        pass

    def forward(self, x):
        pass
