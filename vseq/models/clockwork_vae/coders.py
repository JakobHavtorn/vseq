from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from torchtyping import TensorType

from vseq.modules.convenience import Permute


class StackWaveform(nn.Module):
    def __init__(self, stack_size: int, pad_value: float = 0.0):
        super().__init__()
        self.stack_size = stack_size
        self.pad_value = pad_value

    def forward(self, x: TensorType["B":..., "T"], x_sl: TensorType["B", int] = None) -> TensorType["B":..., "T", "D"]:
        padding = (self.stack_size - x.size(-1) % self.stack_size) % self.stack_size
        x = torch.cat([x, torch.full((*x.shape[:-1], padding), fill_value=self.pad_value, device=x.device)], dim=-1)
        x = x.view(*x.shape[:-1], -1, self.stack_size)  # (B, ..., T / stack_size, stack_size)
        if x_sl is None:
            return x, padding
        x_sl = (x_sl + padding) // self.stack_size
        return x, x_sl, padding

    def reverse(self, x: TensorType["B":..., "T"], padding: Optional[int] = None, x_sl: TensorType["B", int] = None):
        x = x.view(*x.shape[:-2], x.shape[-2] * self.stack_size)
        if padding is None:
            return x

        x = x[..., :-padding]
        if x_sl is None:
            return x

        x_sl = x_sl + self.stack_size - padding
        return x, x_sl


class MultiLevelEncoderAudioDense(nn.Module):
    def __init__(
        self,
        h_size: Union[int, List[int]],
        time_factors: List[int],
        proj_size: Union[int, List[int]] = None,
        num_level_layers: int = 3,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.time_factors = time_factors
        self.proj_size = proj_size
        self.num_level_layers = num_level_layers
        self.activation = activation

        num_levels = len(time_factors)

        h_size = [h_size] * num_levels if isinstance(h_size, int) else h_size
        proj_size = [proj_size] * num_levels if isinstance(proj_size, int) else proj_size

        project_out = (proj_size is not None) and (proj_size > 0)

        self.stack_waveform = StackWaveform(time_factors[0], pad_value=0)  # float('nan'))

        self.levels = nn.ModuleList()
        self.levels.extend([self.get_level(time_factors[0], h_size[0], num_level_layers, activation)])
        self.levels.extend(
            [self.get_level(h_size[l - 1], h_size[l], num_level_layers, activation) for l in range(1, num_levels)]
        )

        if project_out:
            self.out_proj = nn.ModuleList(
                [self.get_level(h_size[l], proj_size, 1, activation) for l in range(num_levels)]
            )

        self.num_levels = num_levels
        self.h_size = h_size
        self.project_out = project_out
        self.out_size = proj_size if project_out else h_size

    @staticmethod
    def get_level(in_dim, h_dim, num_level_layers, activation, o_dim: int = None):
        o_dim = h_dim if o_dim is None else o_dim

        level = [nn.Linear(in_dim, h_dim), activation()]
        for _ in range(1, num_level_layers - 1):
            level.extend([nn.Linear(h_dim, h_dim), activation()])

        level.extend([nn.Linear(h_dim, o_dim), activation()])
        return nn.Sequential(*level)

    def compute_encoding(self, level: int, pre_enc: TensorType["B", "T", "D"]) -> TensorType["B", "T//factor", "D"]:
        B, T, D = pre_enc.shape
        n_merge_steps = int(self.time_factors[level] / self.time_factors[0])  # l=0 is merged by stacking waveform
        n_pad_steps = (n_merge_steps - T % n_merge_steps) % n_merge_steps
        padding = (0, 0, 0, n_pad_steps, 0, 0)  # pad D-dim by (0, 0) and T-dim by (0, N) and B-dim by (0, 0)
        pre_enc = torch.nn.functional.pad(pre_enc, padding, mode="constant", value=0)
        pre_enc = pre_enc.view(B, -1, n_merge_steps, D)
        enc = pre_enc.sum(2)
        return enc

    def forward(self, x: TensorType["B", "T", float]) -> List[TensorType["B", "T", "D", float]]:
        """Encode a sequence of inputs to multiple representations at different timescales.

        The representation at level `l` will be of length `T // time_factors[l]`.

        Args:
            x (torch.Tensor): Input sequence

        Returns:
            List[torch.Tensor]: List of encodings per level
        """
        encodings = []
        hidden, padding = self.stack_waveform(x)
        for l in range(self.num_levels):
            hidden = self.levels[l](hidden)
            pre_enc = self.out_proj[l](hidden) if self.project_out else hidden
            encoding = self.compute_encoding(l, pre_enc) if self.time_factors[l] != 1 else pre_enc
            encodings.append(encoding)
        return encodings


class MultiLevelEncoderConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        h_size: Union[int, List[int]],
        time_factors: List[int],
        proj_size: Union[int, List[int]] = None,
        num_level_layers: int = 3,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.time_factors = time_factors
        self.proj_size = proj_size
        self.num_level_layers = num_level_layers
        self.activation = activation

        num_levels = len(time_factors)

        h_size = [h_size] * num_levels if isinstance(h_size, int) else h_size
        proj_size = [proj_size] * num_levels if isinstance(proj_size, int) else proj_size

        project_out = (proj_size is not None) and (proj_size > 0)

        # six 1D-convolutions with filters set to (64, 128, 192, 256, 512, 512), kernel sizes (10, 8, 4, 4, 4, 1), and strides (5, 4, 2, 2, 2, 1).

        import IPython; IPython.embed(using=False)
        channels = 64
        level_1 = nn.Sequential(
            nn.Conv1d(channels, 1 * channels, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.GroupNorm(num_grous=32, num_channels=1 * channels),
            nn.Conv1d(1 * channels, 2 * channels, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.GroupNorm(num_grous=32, num_channels=2 * channels),
            nn.Conv1d(2 * channels, 3 * channels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.GroupNorm(num_grous=32, num_channels=3 * channels),
            nn.Conv1d(3 * channels, 4 * channels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.GroupNorm(num_grous=32, num_channels=4 * channels),
            nn.Conv1d(4 * channels, 8 * channels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.GroupNorm(num_grous=32, num_channels=8 * channels),
            nn.Conv1d(8 * channels, 8 * channels, kernel_size=1, stride=1),  # dense
        )  # (B, C, T) -> (B, 8xC, T/160)

        strides = [self.time_factors[0]] + [self.time_factors[l + 1] // self.time_factors[l] for l in range(num_levels - 1)]
        kernel_sizes = [2 * stride for stride in strides]

        self.levels = nn.ModuleList()
        self.levels.extend(
            [self.get_level(in_channels, h_size[0], num_level_layers, activation, kernel_sizes[0], strides[0])]
        )
        self.levels.extend(
            [
                self.get_level(h_size[l - 1], h_size[l], num_level_layers, activation, kernel_sizes[l], strides[l])
                for l in range(1, num_levels)
            ]
        )

        if project_out:
            self.out_proj = nn.ModuleList(
                [self.get_level(h_size[l], proj_size, 1, activation, 1, 1) for l in range(num_levels)]
            )

        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.num_levels = num_levels
        self.h_size = h_size
        self.project_out = project_out
        self.out_size = proj_size if project_out else h_size

    @staticmethod
    def get_level(in_dim, h_dim, num_level_layers, activation, kernel_size, stride, o_dim: int = None):
        o_dim = h_dim if o_dim is None else o_dim

        level = [nn.Conv1d(in_dim, h_dim, kernel_size, stride), activation(), Permute(1, 0)]  # (B, C, T) to (B, T, C)
        for _ in range(1, num_level_layers - 1):
            level.extend([nn.Linear(h_dim, h_dim), activation()])

        level.extend([nn.Linear(h_dim, o_dim), activation()])
        return nn.Sequential(*level)

    def forward(self, x: TensorType["B", "T", float]) -> List[TensorType["B", "T", "D", float]]:
        """Encode a sequence of inputs to multiple representations at different timescales.

        The representation at level `l` will be of length `T // time_factors[l]`.

        Args:
            x (torch.Tensor): Input sequence

        Returns:
            List[torch.Tensor]: List of encodings per level
        """
        # import IPython; IPython.embed(using=False)
        # TODO Figure out padding
        encodings = []
        hidden = x.unsqueeze(-1)
        for l in range(self.num_levels):
            hidden = self.levels[l](hidden)
            pre_enc = self.out_proj[l](hidden) if self.project_out else hidden
            encoding = self.compute_encoding(l, pre_enc) if self.time_factors[l] != 1 else pre_enc
            encodings.append(encoding)
        return encodings


class DecoderAudioDense(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        o_dim: int,
        time_factors: List[int],
        num_level_layers: int = 3,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.time_factors = time_factors
        self.num_level_layers = num_level_layers
        self.activation = activation
        self.decoder = MultiLevelEncoderAudioDense.get_level(
            in_dim, h_dim, num_level_layers, activation, o_dim=o_dim * time_factors[0]
        )

    def forward(self, x):
        hidden = self.decoder(x)
        hidden = hidden.view(hidden.size(0), -1, self.o_dim)
        return hidden


class DecoderAudioConv1d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        o_dim: int,
        time_factors: List[int],
        num_level_layers: int = 3,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.time_factors = time_factors
        self.num_level_layers = num_level_layers
        self.activation = activation
        self.decoder = MultiLevelEncoderConv1d.get_level(
            in_dim,
            h_dim,
            num_level_layers,
            activation,
            2 * time_factors[0],
            time_factors[0],
            o_dim=o_dim,
        )

    def forward(self, x):
        hidden = self.decoder(x)
        hidden = hidden.view(hidden.size(0), -1, self.o_dim)
        return hidden
