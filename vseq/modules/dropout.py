from typing import Union

import torch
import torch.nn as nn

from torchtyping import TensorType


class WordDropout(nn.Module):
    def __init__(self, dropout_rate: float = 0.0, mask_value: Union[float, int] = 0, mask_first_timestep: bool = False):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask_value = mask_value
        self.mask_first_timestep = mask_first_timestep

    def forward(self, x: TensorType["B", "T", -1]):
        """Dropout entire timesteps in x of shape (B, T, *D)"""
        if self.training and self.dropout_rate > 0:
            mask = torch.bernoulli(torch.full((x.size(0), x.size(1)), self.dropout_rate, device=x.device)).to(bool)
            mask[:, 0] = self.mask_first_timestep
            x = x.clone()  # We can't modify x in-place
            x[mask, ...] = self.mask_value

        return x

    def __repr__(self):
        return f"WordDropout(dropout_rate={self.dropout_rate}, mask_value={self.mask_value}, mask_first_timestep={self.mask_first_timestep})"
