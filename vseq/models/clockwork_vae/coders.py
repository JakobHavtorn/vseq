from typing import List, Optional, Tuple, Union
from vseq.utils.convolutions import get_same_padding

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

from torchtyping import TensorType

from vseq.modules.convenience import Pad, Permute


class MultiLevelEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError()


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


class DenseAudioEncoder(MultiLevelEncoder):
    def __init__(
        self,
        h_size: Union[int, List[int]],
        time_factors: List[int],
        proj_size: Union[int, List[int]] = None,
        num_level_layers: int = 3,
        activation: nn.Module = nn.ReLU,
    ):
        """Encode a waveform by stacking it into feature vectors of size time_factors[0] and densely transforming.

        The encoder has len(time_factors) outputs and is hence "MultiLevel".
        The first level downsamples with a factor of time_factor[0] by initially stacking the waveform (stack_waveform).
        The subsequent levels downsample by summing time_factor[l] consecutive hidden representations (compute_encoding)

        The first level of this encoder is therefore equivalent to a 1D convolution with kernel size and stride equal
        to time_factor[0] and h_size[0] output channels followed by num_level_layers 1D 1x1 convolutions.

        The subsequent levels are densely connected and are therefore equivalent to 1D 1x1 convolutions.

        Ignoring activations, the first level:
        ```
        Conv1d(in_channels=1,           out_channels=hidden_size, kernel_size=stack_size, stride=stack_size)
        Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1)
        Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1)
        ```

        Levels l>0:
        ```
        Conv1d(in_channels=h_size[l-1], out_channels=h_size[l], kernel_size=1, stride=1)
        Conv1d(in_channels=h_size[l],   out_channels=h_size[l], kernel_size=1, stride=1)
        Conv1d(in_channels=h_size[l],   out_channels=h_size[l], kernel_size=1, stride=1)
        ```

        Args:
            h_size (Union[int, List[int]]): List of hidden sizes for the levels or a single size for all levels.
            time_factors (List[int]): Time factors for each level.
            proj_size (Union[int, List[int]], optional): Optional projection size for output per level
                                                         (different from hidden_size[l]). Defaults to None.
            num_level_layers (int, optional): Number of dense transforms within each layer. Defaults to 3.
            activation (nn.Module, optional): The activation function. Defaults to nn.ReLU.
        """
        super().__init__()

        self.time_factors = time_factors
        self.proj_size = proj_size
        self.num_level_layers = num_level_layers
        self.activation = activation

        num_levels = len(time_factors)

        h_size = [h_size] * num_levels if isinstance(h_size, int) else h_size
        proj_size = [proj_size] * num_levels if isinstance(proj_size, int) else proj_size

        project_out = (proj_size is not None) and (proj_size > 0)

        self.h_size = h_size
        self.e_size = proj_size if project_out else h_size
        self.num_levels = num_levels
        self.project_out = project_out
        self.receptive_field = time_factors[-1]
        self.overall_stride = time_factors[-1]

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

    @staticmethod
    def get_level(in_dim, h_dim, num_level_layers, activation, o_dim: int = None):
        o_dim = h_dim if o_dim is None else o_dim
        dims = [in_dim] + [h_dim] * (num_level_layers - 1) + [o_dim]
        layers = []
        for l in range(num_level_layers):
            layers.extend([nn.Linear(dims[l], dims[l + 1]), activation()])

        level = nn.Sequential(*layers)
        return level

    def compute_encoding(self, level: int, pre_enc: TensorType["B", "T", "D"]) -> TensorType["B", "T//factor", "D"]:
        B, T, D = pre_enc.shape
        n_merge_steps = int(self.time_factors[level] / self.time_factors[0])  # l=0 is merged by stacking waveform
        n_pad_steps = (n_merge_steps - T % n_merge_steps) % n_merge_steps
        padding = (0, 0, 0, n_pad_steps, 0, 0)  # pad D-dim by (0, 0) and T-dim by (0, N) and B-dim by (0, 0)
        pre_enc = torch.nn.functional.pad(pre_enc, padding, mode="constant", value=0)
        pre_enc = pre_enc.view(B, -1, n_merge_steps, D)
        enc = pre_enc.sum(2)
        return enc

    def forward(self, x: TensorType["B", "T"]) -> List[TensorType["B", "T", "D"]]:
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


class DenseAudioDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        o_dim: int,
        time_factor: int,
        num_level_layers: int = 3,
        activation: nn.Module = nn.ReLU,
    ):
        """This decoder matches the MultiLevelEncoderAudioDense by doing the corresponding time_factor[0] upsampling.

        Args:
            in_dim (int): Input size
            h_dim (int): Hidden size
            o_dim (int): Output size
            time_factor (List[int]): Time factors for the first level.
            num_level_layers (int, optional): Number of dense layers per level. Defaults to 3.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.time_factor = time_factor
        self.num_level_layers = num_level_layers
        self.activation = activation
        self.decoder = DenseAudioEncoder.get_level(
            in_dim, h_dim, num_level_layers, activation, o_dim=o_dim * time_factor
        )

    def forward(self, x: TensorType["B", "T", "D"]):
        hidden = self.decoder(x)
        hidden = hidden.view(hidden.size(0), -1, self.o_dim)
        return hidden


class PretrainedCPCEncoder(MultiLevelEncoder):
    checkpoint_url = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt"
    github_repository = "facebookresearch/CPC_audio"

    def __init__(self, num_levels: int = 3, freeze_parameters: bool = True):
        """Wrapper around the pretrained CPC encoder supplied at https://github.com/facebookresearch/CPC_audio.

        Receptive field and time scales:
        ```
          layer recep.   kernel   stride  time-scale
            1   10       10       5       5
            2   44       8        4       20
            3   90       4        2       40
            4   182      4        2       80
            5   366      4        2       160
        ```
        Model:
        ```
        CPCModel(
            (gEncoder): CPCEncoder(
                (conv0): Conv1d(1, 256, kernel_size=(10,), stride=(5,), padding=(3,))
                (batchNorm0): ChannelNorm()
                (conv1): Conv1d(256, 256, kernel_size=(8,), stride=(4,), padding=(2,))
                (batchNorm1): ChannelNorm()
                (conv2): Conv1d(256, 256, kernel_size=(4,), stride=(2,), padding=(1,))
                (batchNorm2): ChannelNorm()
                (conv3): Conv1d(256, 256, kernel_size=(4,), stride=(2,), padding=(1,))
                (batchNorm3): ChannelNorm()
                (conv4): Conv1d(256, 256, kernel_size=(4,), stride=(2,), padding=(1,))
                (batchNorm4): ChannelNorm()
            )
            (gAR): CPCAR(
                (baseNet): LSTM(256, 256, batch_first=True)
            )
        )
        ```
        Summary:
        ```
        ==========================================================================================
        Layer (type:depth-idx)                   Output Shape              Param #
        ==========================================================================================
        PretrainedCPCEncoder                     --                        --
        ├─CPCEncoder: 1                          --                        --
        │    └─Conv1d: 2-1                       [32, 256, 32]             (2,816)
        │    └─ChannelNorm: 2-2                  [32, 256, 32]             (512)
        │    └─Conv1d: 2-3                       [32, 256, 8]              (524,544)
        │    └─ChannelNorm: 2-4                  [32, 256, 8]              (512)
        │    └─Conv1d: 2-5                       [32, 256, 4]              (262,400)
        │    └─ChannelNorm: 2-6                  [32, 256, 4]              (512)
        │    └─Conv1d: 2-7                       [32, 256, 2]              (262,400)
        │    └─ChannelNorm: 2-8                  [32, 256, 2]              (512)
        │    └─Conv1d: 2-9                       [32, 256, 1]              (262,400)
        │    └─ChannelNorm: 2-10                 [32, 256, 1]              (512)
        ├─CPCAR: 1-1                             [32, 1, 256]              --
        │    └─LSTM: 2-11                        [32, 1, 256]              (526,336)
        ==========================================================================================
        Total params: 1,843,456
        Trainable params: 0
        Non-trainable params: 1,843,456
        Total mult-adds (M): 212.87
        ==========================================================================================
        Input size (MB): 0.02
        Forward/backward pass size (MB): 6.23
        Params size (MB): 7.37
        Estimated Total Size (MB): 13.62
        ==========================================================================================
        ```

        Args:
            num_levels (int, optional): [description]. Defaults to 3.
            freeze_parameters (bool, optional): [description]. Defaults to True.
        """
        super().__init__()
        assert 1 <= num_levels <= 5, "Pretrained model has between 1 and 5 levels"

        self.num_levels = num_levels
        self.freeze_parameters = freeze_parameters

        self.receptive_fields = [10, 44, 90, 182, 366][-self.num_levels :]
        self.receptive_field = 366
        self.time_factors = [5, 20, 40, 80, 160][-self.num_levels :]
        self.overall_stride = 160
        self.strides = [5, 4, 2, 2, 2][-self.num_levels :]
        self.h_size = [256, 256, 256, 256, 256][-self.num_levels :]
        self.e_size = self.h_size[-self.num_levels :]

        self.load_pretrained_checkpoint()

        if freeze_parameters:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.ar_net.parameters():
                p.requires_grad_(False)

    def load_pretrained_checkpoint(self):
        # import argparse
        checkpoint = torch.hub.load_state_dict_from_url(self.checkpoint_url, progress=False, map_location="cpu")
        config = torch.hub.load(self.github_repository, "get_default_cpc_config")
        # config = torch.hub.load(self.github_repository, "loadArgs", config, argparse.Namespace(**checkpoint["config"]))
        encoder = torch.hub.load(self.github_repository, "getEncoder", config)
        ar_net = torch.hub.load(self.github_repository, "getAR", config)
        model = torch.hub.load(self.github_repository, "cpcmodel", encoder, ar_net)
        model.load_state_dict(checkpoint["weights"], strict=False)

        # self.model = model  # torch.hub.load("facebookresearch/CPC_audio", "CPC_audio", pretrained=True)
        self.encoder = model.gEncoder
        self.ar_net = model.gAR

    def forward_encoder(self, x: TensorType["B", "C", "T", float]) -> TensorType["B", "C", "T", float]:
        x1 = F.relu(self.encoder.batchNorm0(self.encoder.conv0(x)))
        x2 = F.relu(self.encoder.batchNorm1(self.encoder.conv1(x1)))
        x3 = F.relu(self.encoder.batchNorm2(self.encoder.conv2(x2)))
        x4 = F.relu(self.encoder.batchNorm3(self.encoder.conv3(x3)))
        x5 = F.relu(self.encoder.batchNorm4(self.encoder.conv4(x4)))

        encodings = [x1, x2, x3, x4]
        return x5, encodings

    def forward(self, x: TensorType["B", "T"]) -> List[TensorType["B", "T", "D"]]:
        """Compute encodings.

        Forward pass customized using https://github.com/facebookresearch/CPC_audio/blob/master/cpc/model.py
        """
        hidden = x.unsqueeze(1)

        hidden, encodings = self.forward_encoder(hidden)
        hidden = hidden.permute(0, 2, 1)
        encodings = [enc.permute(0, 2, 1) for enc in encodings]

        ar_features = self.ar_net(hidden)
        encodings.append(ar_features)
        return encodings[-self.num_levels :]

    def summary(self):
        return torchinfo.summary(self, input_size=(32, 160))


class PretrainedCPCDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        o_dim: int,
    ):
        """A (not pretrained) decoder for the PretrainedCPCEncoder architecture.s"""
        super().__init__()

        h_dim = 256

        self.in_dim = in_dim
        self.o_dim = o_dim

        self.h_dim = h_dim

        kernels = [5, 5, 5, 9, 11]
        strides = [2, 2, 2, 4, 5]
        i_chan = [in_dim, h_dim, h_dim, h_dim, h_dim]
        o_chan = [h_dim, h_dim, h_dim, h_dim, o_dim]

        layers = []
        for k, s, i, o in zip(kernels, strides, i_chan, o_chan):
            conv = nn.ConvTranspose1d(i, o, kernel_size=k, stride=s)
            _, unsym_pad = get_same_padding(conv)
            layers.extend([nn.GroupNorm(num_channels=i, num_groups=i), conv, Pad(unsym_pad), nn.ReLU()])
        # padding = [2, 2, 2, 4, 5]
        # layers = []
        # for k, s, i, o, p in zip(kernels, strides, i_chan, o_chan, padding):
        #     upsample = nn.Upsample(scale_factor=s, mode="nearest")
        #     conv = nn.Conv1d(i, o, kernel_size=k, padding=p)
        #     layers.extend([nn.GroupNorm(num_channels=i, num_groups=i), upsample, conv, nn.ReLU()])
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: TensorType["B", "T", "D"]):
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x

    def summary(self):
        return torchinfo.summary(self, input_size=(32, 1, self.in_dim))


class DecoderAudioConv1d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        o_dim: int,
        time_factors: List[int],
        num_level_layers: int = 3,
        activation: nn.Module = nn.ReLU,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.time_factors = time_factors
        self.num_level_layers = num_level_layers
        self.activation = activation

        assert h_dim / (2 * 2 * 2 * 4 * 5) == h_dim // (2 * 2 * 2 * 4 * 5)

        nn.Sequential(
            nn.GroupNorm(num_channels=in_dim, num_groups=num_groups),
            nn.ConvTranspose1d(in_dim, h_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.GroupNorm(num_channels=h_dim, num_groups=num_groups),
            nn.ConvTranspose1d(h_dim, h_dim // 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.GroupNorm(num_channels=h_dim // 2, num_groups=num_groups),
            nn.ConvTranspose1d(h_dim // 2, h_dim // 4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.GroupNorm(num_channels=h_dim // 4, num_groups=num_groups),
            nn.ConvTranspose1d(h_dim // 4, h_dim // 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.GroupNorm(num_channels=h_dim // 8, num_groups=num_groups),
            nn.ConvTranspose1d(h_dim // 8, h_dim, kernel_size=4, stride=2),
            nn.ReLU(),
        )

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

    def forward(self, x):
        hidden = self.decoder(x)
        hidden = hidden.view(hidden.size(0), -1, self.o_dim)
        return hidden


class Conv1dAudioEncoder(nn.Module):
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

        import IPython

        IPython.embed(using=False)
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

        strides = [self.time_factors[0]] + [
            self.time_factors[l + 1] // self.time_factors[l] for l in range(num_levels - 1)
        ]
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
        self.e_size = proj_size if project_out else h_size

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
        self.decoder = Conv1dAudioEncoder.get_level(
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
