import math

from typing import List, Union
from vseq.modules.convenience import Pad, Permute

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtyping import TensorType

from vseq.utils.convolutions import compute_conv_attributes, get_same_padding, _single


# TODO Share more code between encoder and decoders. Do this via `transposed` argument to some of the modules.
#      Which module "level" to do this at? TemporalBlock or Conv(Transpose)DepthwiseSeparable1d?
# TODO Implement a context decoder.
# TODO Make the decoder be layered as the encoder
# TODO Create a single TasNetCoder module that can act as decoder and encoder (e.g. share blocks but have in and out transforms be unique to encoder and decoder)
# TODO Add a small autoregressive model to the output space


class TasNetEncoder(nn.Module):
    def __init__(
        self,
        strides: List[int],
        channels_in: int = 1,
        channels_bottleneck: int = 128,
        channels_block: int = 512,
        kernel_size: int = 5,
        num_blocks: int = 8,
        num_levels: int = 3,
        norm_type: str = "GlobalLayerNorm",
    ):
        """TasNet-based encoder similar to the paper https://arxiv.org/pdf/1809.07454.pdf.

        Default arguments above correspond to the highest performing model in the paper (Table II)

        Args:
            channels_bottleneck: Number of channels in bottleneck 1 × 1-conv block
            channels_block: Number of channels in convolutional blocks
            kernel_size: Kernel size in convolutional blocks
            num_blocks: Number of convolutional blocks in each repeat
            num_levels: int: Number of repeats
            num_speakers: Number of speakers
            norm_type: GlobalLayerNorm, ChannelwiseLayerNorm, TemporalLayerNorm
        """
        super().__init__()

        self.strides = strides
        self.channels_bottleneck = channels_bottleneck
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.num_levels = num_levels
        self.norm_type = norm_type

        self.e_size = [self.channels_bottleneck] * num_levels
        self.overall_stride = strides[-1]

        assert strides[0] >= 2, "First level stride must be larger 2"
        assert all(tf % 2 == 0 for tf in strides), "All time factors must be wholly divisible by 2 (power of 2)"

        self.stride_per_level = [self.strides[0]]
        self.stride_per_level += [self.strides[l] // self.strides[l - 1] for l in range(1, self.num_levels)]

        assert all(
            2 ** num_blocks >= s for s in self.stride_per_level
        ), f"Not enough blocks per level to get strides of {strides=}"

        self.in_transform = nn.Sequential(
            # [B, 1, T] -> [B, channels_block, T]
            nn.Conv1d(
                channels_in,
                channels_block,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.PReLU(),
            nn.GroupNorm(num_channels=channels_block, num_groups=channels_block),
            # [B, channels_block, T] -> [B, channels_bottleneck, T]
            nn.Conv1d(channels_block, channels_bottleneck, 1, bias=False),
        )

        # import IPython; IPython.embed()
        # Blocks
        self.receptive_fields = []
        self.levels = nn.ModuleList()

        jump_in, receptive_field_in = 1, 1
        for l in range(num_levels):
            remaining_stride = self.stride_per_level[l] if l > 0 else self.stride_per_level[0] // 2
            blocks = []
            for x in range(num_blocks):
                dilation = 1
                stride = 2 if remaining_stride >= 2 else 1
                padding = kernel_size // 2  # same padding
                blocks += [
                    TemporalBlock(
                        channels_bottleneck,
                        channels_block,
                        kernel_size,
                        stride=stride,
                        padding="same",
                        dilation=dilation,
                        norm_type=norm_type,
                        transposed=False,
                    )
                ]
                remaining_stride = remaining_stride // 2 if remaining_stride >= 2 else remaining_stride

                # keep track of receptive field
                _, jump_in, receptive_field_in, _ = compute_conv_attributes(
                    in_shape=1,
                    kernels=[1, kernel_size],
                    paddings=[0, padding],
                    strides=[1, stride],
                    jump_in=jump_in,
                    receptive_field_in=receptive_field_in,
                )

            assert remaining_stride == 1

            self.receptive_fields.append(receptive_field_in[0])
            self.levels.append(nn.Sequential(*blocks))

        self.receptive_field = self.receptive_fields[-1]

    @property
    def device(self):
        return self.in_transform[0].weight.device

    def forward(self, x: TensorType["B", "T"]) -> List[TensorType["B", "T", "D"]]:
        """
        Args:
            x: [B, T], B is batch size
        returns:
            est_mask: [B, num_speakers, channels_block, T]
        """
        x = x.unsqueeze(1)
        hidden = self.in_transform(x)
        encodings = []
        for l in range(self.num_levels):
            hidden = self.levels[l](hidden)
            encodings.append(hidden.permute(0, 2, 1))
        return encodings


class TasNetDecoder(nn.Module):
    def __init__(
        self,
        strides: int,
        channels_in: int,
        channels_out: int,
        channels_bottleneck: int = 128,
        channels_block: int = 512,
        kernel_size: int = 5,
        num_blocks: int = 8,
        norm_type: str = "GlobalLayerNorm",
    ):
        super().__init__()

        # [B, channels_in, T] -> [B, channels_block, T] -> [B, channels_bottleneck, T]
        self.in_transform = nn.Sequential(
            nn.Conv1d(channels_in, channels_block, kernel_size, stride=1, padding=kernel_size // 2, bias=True),
            nn.PReLU(),
            nn.GroupNorm(num_channels=channels_block, num_groups=channels_block),
            nn.Conv1d(channels_block, channels_bottleneck, 1, bias=False),
        )

        remaining_stride = strides
        blocks = []
        for x in range(num_blocks):
            dilation = 1
            stride = 2 if remaining_stride >= 2 else 1
            temporal_block = TemporalBlock(
                channels_bottleneck,
                channels_block,
                kernel_size,
                stride=stride,
                padding="same",
                dilation=dilation,
                norm_type=norm_type,
                transposed=stride > 1,
            )
            blocks += [temporal_block]

            remaining_stride = remaining_stride // 2 if remaining_stride >= 2 else remaining_stride

        assert remaining_stride == 1

        self.blocks = nn.Sequential(*blocks)

        self.out_transform = nn.Sequential(
            # [B, channels_in, T] -> [B, channels_block, T]
            nn.Conv1d(
                channels_bottleneck,
                channels_out,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.PReLU(),
            nn.GroupNorm(num_channels=channels_out, num_groups=channels_out),
        )

    def forward(self, x: TensorType["B", "T", "D"]) -> TensorType["B", "T", "D"]:
        h = x.permute(0, 2, 1)
        h = self.in_transform(h)
        h = self.blocks(h)
        h = self.out_transform(h)
        h = h.permute(0, 2, 1)
        return h


# TODO Make me into a general TasNetCoder
class TasNetCoder(nn.Module):
    def __init__(
        self,
        channels_in: int,
        strides: List[int],
        channels_out: Union[Union[None, int], List[Union[None, int]]] = None,
        channels_bottleneck: int = 128,
        channels_block: int = 512,
        kernel_size: int = 5,
        dilation: int = 1,
        num_blocks: int = 8,
        num_levels: int = 3,
        norm_type: str = "ChannelwiseLayerNorm",
        transposed: bool = False,
        stride_per_block: int = 2,
    ):
        """TasNet-based coder similar to the paper https://arxiv.org/pdf/1809.07454.pdf.

        Default arguments above correspond to the highest performing model in the paper (Table II)
        
        The dimension of the embedding at layer `l` is:
        - `channels_bottleneck` if `channels_out[l] is None`
        - `channels_out[l]` if `channels_out[l] is not None`

        Args:
            channels_in: 
            channels_out: 
            channels_bottleneck: Number of channels in bottleneck 1 × 1-conv block
            channels_block: Number of channels in convolutional blocks
            kernel_size: Kernel size in convolutional blocks
            num_blocks: Number of convolutional blocks in each level
            num_levels: int: Number of levels
            norm_type: GlobalLayerNorm, ChannelwiseLayerNorm
        """
        super().__init__()

        self.strides = strides
        self.channels_in = channels_in
        self.channels_out = [channels_out] * num_levels if isinstance(channels_out, int) else channels_out
        self.channels_bottleneck = channels_bottleneck
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.num_levels = num_levels
        self.norm_type = norm_type
        self.transposed = transposed
        self.stride_per_block = stride_per_block

        self.e_size = [self.channels_bottleneck] * num_levels
        self.overall_stride = strides[-1]

        assert strides[0] >= 2, "First level stride must be larger 2"
        assert all(tf % 2 == 0 for tf in strides), "All time factors must be wholly divisible by 2 (power of 2)"

        self.strides = [self.strides[0]]
        self.strides += [self.strides[l] // self.strides[l - 1] for l in range(1, self.num_levels)]

        assert all(
            2 ** num_blocks >= s for s in self.strides
        ), f"Not enough blocks per level to get strides of {self.strides=}"

        self.in_transform = nn.Sequential(
            # [B, 1, T] -> [B, channels_block, T]
            nn.Conv1d(
                channels_in,
                channels_block,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.PReLU(),
            nn.GroupNorm(num_channels=channels_block, num_groups=channels_block),
            # [B, channels_block, T] -> [B, channels_bottleneck, T]
            nn.Conv1d(channels_block, channels_bottleneck, 1, bias=False),
        )

        # Blocks
        self.levels = nn.ModuleList()
        self.out_projs = nn.ModuleList()

        for l in range(num_levels):
            # build blocks
            remaining_stride = self.strides[l] if l > 0 else self.strides[0]
            blocks = []
            for x in range(num_blocks):
                # dilation = 2 ** x
                # padding = (kernel_size - 1) * dilation if causal else (kernel_size - 1) * dilation // 2
                if remaining_stride >= self.stride_per_block:
                    stride = self.stride_per_block
                    remaining_stride = remaining_stride // self.stride_per_block
                else:
                    stride = 1
                temporal_block = TemporalBlock(
                    channels_bottleneck,
                    channels_block,
                    kernel_size,
                    stride=stride,
                    padding="same",
                    dilation=dilation,
                    norm_type=norm_type,
                    transposed=transposed,
                )
                blocks += [temporal_block]

            assert remaining_stride == 1

            self.levels.append(nn.Sequential(*blocks))

            # build out projection
            out_proj_l = [Permute(0, 2, 1)]  # [B, channels_bottleneck, T] to [B, T, channels_bottleneck]
            if self.channels_out[l] is not None:
                out_proj_l.extend(
                    [
                        # [B, T, channels_bottleneck] -> [B, T, self.channels_out[l]] (1x1 convolution)
                        nn.Linear(channels_bottleneck, self.channels_out[l]),
                        nn.PReLU(),
                    ]
                )
            self.out_projs.append(nn.Sequential(*out_proj_l))

    @property
    def device(self):
        return self.in_transform[0].weight.device

    def forward_level(self, hidden: TensorType["B", "T", "C"], level: int) -> List[TensorType["B", "T", "D"]]:
        if level == 0:
            hidden = hidden.permute(0, 2, 1)
            hidden = self.in_transform(hidden)

        hidden = self.levels[level](hidden)
        encoding = self.out_proj[level](hidden)
        return hidden, encoding

    def forward(self, hidden: TensorType["B", "T", "C"]) -> List[TensorType["B", "T", "D"]]:
        """
        Args:
            hidden: [B, T], B is batch size
        returns:
            est_mask: [B, num_speakers, channels_block, T]
        """
        encodings = []
        for l in range(self.num_levels):
            hidden, encoding = self.forward_level(hidden, level=l)
            encodings.append(encoding)
        return encodings


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[int, str],
        dilation: int,
        norm_type: str,
        transposed: bool = False,
    ):
        """Temporal Convolutional Network block.

        Args:
            in_out_channels (int): Number of channels in input and output
            hidden_channels (int): Number of channels in intermediary hidden representations. Usually > in_out_channels.
            kernel_size (int): Kernel size for depth-wise separable convolution
            stride (int): Stride for depth-wise separable convolution
            padding (str or int): Integer padding for depth-wise separable convolution or str "same" for same padding.
            dilation (int): Dilation for depth-wise separable convolution
            norm_type (str, optional): Name of the normalization to use. Defaults to "GlobalLayerNorm".
        """
        super().__init__()

        self.in_out_channels = in_out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.norm_type = norm_type
        self.transposed = transposed

        # [B, channels_bottleneck, T] -> [B, channels_block, T]
        self.conv1x1 = nn.Conv1d(in_out_channels, hidden_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.norm = get_normalization(norm_type, num_channels=hidden_channels)
        # [B, channels_block, T] -> [B, channels_bottleneck, T]

        if padding == "same":
            sym_pad, unsym_pad = get_same_padding(kernel_size=kernel_size, stride=stride, dilation=dilation, transposed=transposed)
            padding = unsym_pad if transposed and stride > 1 else sym_pad

        kwargs = dict(
            in_channels=hidden_channels,
            out_channels=in_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            norm_type=norm_type,
        )
        if transposed and stride > 1:
            self.pad = Pad(padding)
            self.dsconv = ConvTransposeDepthwiseSeparable1d(**kwargs)
        else:
            self.pad = None
            self.dsconv = ConvDepthwiseSeparable1d(**kwargs, padding=padding)
        
    def forward(self, x):
        """
        Args:
            x: [B, channels_bottleneck, T]
        Returns:
            [B, channels_bottleneck, T]
        """
        residual = x

        x = self.conv1x1(x)
        x = self.prelu(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.dsconv(x)
        x = self.pad(x) if self.pad is not None else x

        if self.transposed and self.stride > 1:
            # residual connection via copies of residual
            out_length = residual.shape[2] * self.stride
            c_start = math.floor((x.shape[2] - out_length) / 2)
            c_final = math.ceil((x.shape[2] - out_length) / 2) + out_length
            x[:, :, c_start:c_final] = x[:, :, c_start:c_final] + residual.repeat(1, 1, self.stride)
        else:
            # residual connection via striding
            x = x + residual[:, :, :: self.stride]
            # TODO residual connection via average over stride steps
            # TODO Maybe use nn.Fold to create windows 
            # https://pytorch.org/docs/stable/generated/torch.nn.Fold.html#torch.nn.Fold
            # out_length = residual.shape[2]
            # x = x + residual.view(residual.size(0), residual.size(1), -1, self.stride).mean(-1)

        return x


class ConvDepthwiseSeparable1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        norm_type: str = "ChannelwiseLayerNorm",
    ):
        super().__init__()

        self.stride = _single(stride)
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)
        self.dilation = _single(dilation)

        # [B, channels_block, T] -> [B, channels_block, T]
        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=True,
        )
        self.prelu = nn.PReLU()
        self.norm = get_normalization(norm_type, num_channels=in_channels)
        # [B, channels_block, T] -> [B, channels_bottleneck, T]
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, channels_block, T]
        Returns:
            result: [B, channels_bottleneck, T]
        """
        x = self.depthwise_conv(x)
        x = self.prelu(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.pointwise_conv(x)
        return x


class ConvTransposeDepthwiseSeparable1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        norm_type: str = "ChannelwiseLayerNorm",
    ):
        super().__init__()

        self.stride = _single(stride)
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)
        self.dilation = _single(dilation)

        # [B, channels_block, T] -> [B, channels_block, T]
        self.depthwise_conv = nn.ConvTranspose1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=True,
        )
        self.prelu = nn.PReLU()
        self.norm = get_normalization(norm_type, num_channels=in_channels)
        # [B, channels_block, T] -> [B, channels_bottleneck, T]
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, channels_block, T]
        Returns:
            result: [B, channels_bottleneck, T]
        """
        x = self.depthwise_conv(x)
        x = self.prelu(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.pointwise_conv(x)
        return x


def get_normalization(norm_type: str, num_channels: int):
    """The input of normlization will be (B, num_speakers, T), where B is batch size,
    num_speakers is channel size and T is sequence length.
    """
    if norm_type == "GlobalLayerNorm":
        return GlobalLayerNorm(num_channels)
    elif norm_type == "ChannelwiseLayerNorm":
        return nn.GroupNorm(num_channels=num_channels, num_groups=num_channels)
    elif norm_type is None:
        return None
    raise NotImplementedError(f"Unknown {norm_type=}")


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (GlobalLayerNorm) normalizes over channels and time"""

    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, num_channels, 1))  # [1, channels_block, 1]
        self.beta = nn.Parameter(torch.Tensor(1, num_channels, 1))  # [1, channels_block, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, x: torch.Tensor, eps: float = 1e-8):
        """
        Args:
            x: [B, channels_block, T], B is batch size, channels_block is channel size, T is length
        Returns:
            GlobalLayerNorm_y: [B, channels_block, T]
        """
        mean = x.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        x_normed = self.gamma * (x - mean) / torch.pow(var + eps, 0.5) + self.beta
        return x_normed


if __name__ == "__main__":
    import IPython

    IPython.embed(using=False)
