from typing import List

from vseq.utils.convolutions import get_same_padding

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

from torchtyping import TensorType

from vseq.modules.convenience import Pad


class CPCEncoder(nn.Module):
    checkpoint_url = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt"
    github_repository = "facebookresearch/CPC_audio"

    def __init__(self, num_levels: int = 3, pretrained: bool = True, freeze_parameters: bool = True):
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
        self.pretrained = pretrained
        self.freeze_parameters = freeze_parameters

        self.receptive_fields = [10, 44, 90, 182, 366][-self.num_levels :]
        self.receptive_field = 366
        self.time_factors = [5, 20, 40, 80, 160][-self.num_levels :]
        self.overall_stride = 160
        self.strides = [5, 4, 2, 2, 2][-self.num_levels :]
        self.h_size = [256, 256, 256, 256, 256][-self.num_levels :]
        self.e_size = self.h_size[-self.num_levels :]

        self.load_pretrained_checkpoint()

        if not pretrained:
            for m in self.encoder.children():
                m.reset_parameters()

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


class CPCDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        o_dim: int,
    ):
        """A (not pretrained) decoder for the PretrainedCPCEncoder architecture."""
        super().__init__()

        self.in_dim = in_dim
        self.o_dim = o_dim
        self.h_dim = 256

        kernels = [5, 5, 5, 9, 11]
        strides = [2, 2, 2, 4, 5]
        i_chan = [in_dim, self.h_dim, self.h_dim, self.h_dim, self.h_dim]
        o_chan = [self.h_dim, self.h_dim, self.h_dim, self.h_dim, o_dim]
        layers = []
        for k, s, i, o in zip(kernels, strides, i_chan, o_chan):
            conv = nn.ConvTranspose1d(i, o, kernel_size=k, stride=s)
            _, unsym_pad = get_same_padding(conv)
            layers.extend([nn.GroupNorm(num_channels=i, num_groups=i), conv, Pad(unsym_pad), nn.ReLU()])

        # kernels = [5, 5, 5, 9, 11]
        # strides = [2, 2, 2, 4, 5]
        # i_chan = [in_dim, self.h_dim, self.h_dim, self.h_dim, self.h_dim]
        # o_chan = [self.h_dim, self.h_dim, self.h_dim, self.h_dim, o_dim]
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


