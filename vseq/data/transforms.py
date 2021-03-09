from typing import List, Any

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

    # def in_datatype(self):
    #     pass

    # def out_datatype(self):

    @property
    def extension(self):
        return self._extension

    def forward(self, x):
        raise NotImplementedError()

    def collate(self, batch: List[Any]):
        raise NotImplementedError()


class Compose(Transform):
    def __init__(self, *transforms):
        self.transforms = transforms

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def collate(self, batch):
        return self.transforms[-1].collate(batch)


class MelSpectrogram(torchaudio.transforms.MelSpectrogram, Transform):
    def __init__(self, extension: str):
        """Transform audio to a mel spectrogram"""
        self._extension = extension

        self.in_datatype = Audio()
        self.out_datatype = Spectrogram()

    # def collate(self, batch: List[torch.Tensor]):
    #     """Zero pad batch of spectrograms to maximum temporal length and concatenate"""
    #     sequence_lengths = [spectrogram.shape[1] for spectrogram in batch]

    #     T_max = max(sequence_lengths)
    #     N, F = len(batch), batch[0].shape[0]

    #     padded_batch = torch.zeros((N, F, T_max), dtype=torch.float32)
    #     for i, seq_len in enumerate(sequence_lengths):
    #         padded_batch[i, :, :seq_len] = batch[i]

    #     return padded_batch, sequence_lengths

