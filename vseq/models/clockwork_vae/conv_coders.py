from typing import List, Optional, Tuple, Union

from torch.tensor import Tensor

import torch.nn as nn
import torch.nn.functional as F

from torchtyping import TensorType

from vseq.modules.convenience import Pad, Permute
from vseq.utils.convolutions import compute_conv_attributes, compute_conv_attributes_single, get_same_padding


def get_level(
    in_channels: int,
    channel_increment: int,
    strides: List[int],
    kernel_overlap_factor: float,
    activation: nn.Module = nn.ReLU,
    normalize: bool = True,
    transpose: bool = False,
):
    n_layers = len(strides)

    # if not transpose:
    c_is = [in_channels + l * channel_increment for l in range(n_layers)]
    c_os = [in_channels + (l + 1) * channel_increment for l in range(n_layers)]
    iterator = list(enumerate(zip(c_is, c_os, strides)))
    # else:
        # c_is = [in_channels - l * channel_increment for l in range(n_layers)]
        # c_os = [in_channels - (l + 1) * channel_increment for l in range(n_layers)]
        # strides = list(reversed(strides))
        # iterator = list(enumerate(zip(c_is, c_os, strides)))

    layers = []
    # jump, receptive_field = 1, 1
    for l, (c_i, c_o, stride) in iterator:
        kernel_size = max(round(stride * kernel_overlap_factor) - 1, stride)

        if transpose and stride > 1:
            layers.append(nn.ConvTranspose1d(c_i, c_o, kernel_size=kernel_size, stride=stride))
            layers.append(Pad(get_same_padding(layers[-1])[1]))
        else:
            padding = 0 if kernel_overlap_factor == 0 else kernel_size // 2  # same padding
            layers.append(nn.Conv1d(c_i, c_o, kernel_size=kernel_size, stride=stride, padding=padding))

        layers.append(activation())

        if normalize:
            layers.append(nn.GroupNorm(num_channels=c_o, num_groups=c_o))

        # o_out, jump, receptive_field, start_out = compute_conv_attributes_single(
        #     in_shape=1,
        #     kernels=kernel_size,
        #     paddings=padding,
        #     strides=stride,
        #     jump_in=jump,
        #     receptive_field_in=receptive_field,
        # )

    return layers


class AudioEncoderConv1d(nn.Module):
    def __init__(
        self,
        # h_size: int,
        # time_factors: List[int],
        # # proj_size: Union[int, List[int]] = None,
        # num_level_layers: int = 3,
        num_levels: int = 3,
        # activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        level_1 = [
            *get_level(1, 15, [8], kernel_overlap_factor=2),  # (B, 1, T) -> (B, 16, T/8)
            *get_level(16, 16, [4, 2, 1], kernel_overlap_factor=2),  # (B, 16, T) -> (B, 64, T/64)
        ]
        level_2 = get_level(64, 32, [2, 2, 2, 1], kernel_overlap_factor=2)  # (B, 64, T) -> (B, 192, T/512)
        level_3 = get_level(192, 64, [2, 2, 2, 1], kernel_overlap_factor=2)  # (B, 192, T) -> (B, 448, T/4096)

        self.levels = nn.ModuleList([nn.Sequential(*level_1), nn.Sequential(*level_2), nn.Sequential(*level_3)])
        self.levels = self.levels[:num_levels]

        self.e_size = [64, 192, 448][:num_levels]
        self.receptive_field = 400
        self.overall_stride = (8 * 4 * 2) * (2 * 2 * 2) * (2 * 2 * 2)

    def forward(self, x: TensorType["B", "T"]) -> List[TensorType["B", "T", "D"]]:
        hidden = x.unsqueeze(1)
        encodings = []
        for level in self.levels:
            hidden = level(hidden)
            encodings.append(hidden.permute(0, 2, 1))
        return encodings


class ContextDecoderConv1d(nn.Module):
    def __init__(
        self,
        # h_size: List[int],
        # z_size: List[int],
        # time_factors: List[int],
        # kernel_overlap_factor: int = 2,
        num_levels: int = 2,
        # # proj_size: Union[int, List[int]] = None,
        # activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        # in_channels per level is equal to hidden size plus latent size (i.e. the top-down contex)

        level_3 = nn.Sequential(
            Permute(0, 2, 1),
            *get_level(640, -(320-64), [1], kernel_overlap_factor=2, transpose=True),  # (B, 640, T) -> (B, 320, T)
            *get_level(384, -64, [2, 2, 2], kernel_overlap_factor=2, transpose=True),  # (B, 320, T) -> (B, 192, 8*T)
            Permute(0, 2, 1),
        )

        level_2 = nn.Sequential(
            Permute(0, 2, 1),
            *get_level(320, -160, [1], kernel_overlap_factor=2, transpose=True),  # (B, 320, T*8) -> (B, 160, T*8)
            *get_level(160, -32, [2, 2, 2], kernel_overlap_factor=2, transpose=True),  # (B, 160, T*8) -> (B, 64, T*64)
            Permute(0, 2, 1),
        )

        self.levels = nn.ModuleList([level_2, level_3])[:num_levels]

    def forward(self, x: TensorType["B", "T", "D"]) -> List[TensorType["B", "T", "D"]]:
        raise NotImplementedError()


class AudioDecoderConv1d(nn.Module):
    def __init__(
        self,
        # h_size: int,
        # time_factors: List[int],
        # # proj_size: Union[int, List[int]] = None,
        num_levels: int = 3,
        # activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        level_1 = nn.Sequential(
            Permute(0, 2, 1),
            *get_level(128, -64, [1], kernel_overlap_factor=2, transpose=True),  # (B, 128, T*64) -> (B, 64, T*64)
            *get_level(64, -16, [2, 4, 8], kernel_overlap_factor=2, transpose=True),  # (B, 64, T*64) -> (B, 16, T*4096)
            Permute(0, 2, 1),
        )

        self.level = nn.Sequential(*level_1)

    def forward(self, x: TensorType["B", "T", "D"]) -> List[TensorType["B", "T", "D"]]:
        hidden = self.level(x)
        return hidden
