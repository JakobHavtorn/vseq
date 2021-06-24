"""WaveNet main model"""

from types import SimpleNamespace
from typing import List
from torch.tensor import Tensor

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from torchtyping import TensorType

from vseq.evaluation.metrics import BitsPerDimMetric, LLMetric, LossMetric
from vseq.utils.operations import sequence_mask

from .modules import CausalConv1d, ResidualStack, OutConv1d
from ..base_model import BaseModel


class InputSizeError(Exception):
    def __init__(self, input_size, receptive_field):
        message = "Input size has to be larger than receptive_field\n"
        message += f"Input size: {input_size}, Receptive fields size: {receptive_field}"
        super().__init__(message)


class WaveNet(BaseModel):
    def __init__(
        self,
        in_channels: int = 256,
        out_classes: int = 256,
        n_layers: int = 10,
        n_stacks: int = 5,
        res_channels: int = 512,
    ):
        """Stochastic autoregressive modelling of audio waveform frames with conditional dilated causal convolutions.

        The total number of residual blocks (layers) used is equal to the n_layers times the n_stacks.
        This is `k` in Figure 4 in the paper.

        An illustration of the model:

                             |---------------------------------------------------| *residual*
                             |                                                   |
                             |             |-- tanh --|                          |
                 -> *input* ---> dilate ---|          * ---> 1x1 ---|----------- + ---> *input*
                                           |-- sigm --|             |
                                                                    |
                                                                    |
                 -> *skip* ---------------------------------------- + ----------------> *skip*

        Args:
            in_channels (int): Number of channels in the input data.
            out_classes (int): Number of classes for output (i.e. number of quantized values in target audio values)
            n_layers (int): Number of stacked residual blocks. Dilations chosen as 2, 4, 8, 16, 32, 64...
            n_stacks (int): Number of stacks of residual blocks with skip connections to the output.
            res_channels (int): Number of channels in residual blocks (and embedding if in_channels > 1).

        Reference:
            [1] WaveNet: A Generative Model for Raw Audio (https://arxiv.org/abs/1609.03499)
        """
        super().__init__()

        self.n_layers = n_layers
        self.n_stacks = n_stacks
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.out_classes = out_classes

        self.receptive_field = self.compute_receptive_field(n_layers, n_stacks)

        if in_channels > 1:
            self.embedding = nn.Embedding(num_embeddings=in_channels, embedding_dim=res_channels)
            # TODO Could be depth-wise separable (i.e. one kernel shared for all channels)
            self.causal = CausalConv1d(res_channels, res_channels, receptive_field=self.receptive_field)
        else:
            self.embedding = None
            self.causal = CausalConv1d(in_channels, res_channels, receptive_field=self.receptive_field)

        self.res_stack = ResidualStack(n_layers=n_layers, n_stacks=n_stacks, res_channels=res_channels)

        self.out_convs = OutConv1d(res_channels, out_classes)

        self.nll_criterion = torch.nn.NLLLoss(reduction="none")

    @staticmethod
    def compute_receptive_field(n_layers: int, n_stacks: int):
        """Compute and return the receptive field of a WaveNet model"""
        layers = [2 ** i for i in range(0, n_layers)] * n_stacks
        receptive_field = np.sum(layers)
        receptive_field = receptive_field + 2  # Plus two for causal conv
        return int(receptive_field)

    def check_input_size(self, x: TensorType["B", "C", "T"]):
        """Require input to be longer than the model's receptive field"""
        if x.size(2) <= self.receptive_field:
            raise InputSizeError(int(x.size(2)), self.receptive_field)

    def _get_target(self, x: TensorType["B", "T", 1]):
        """Convert a PCM audio waveform to quantized integer targets for NLL loss"""
        target = x.squeeze(-1)  # (B, T, C) to (B, T)
        target = (target + 1) / 2  # Transform [-1, 1] to [0, 1]
        target = target * (self.out_classes - 1)  # Transform [0, 1] to [0, 255]
        target = target.floor().to(torch.int64)  # To integer (floor because of added noise for dequantization)
        return target

    def compute_loss(
        self,
        target: TensorType["B", "T", int],
        x_sl: TensorType["B", int],
        logits: TensorType["B", "C", "T", float],
    ):
        """Compute the loss as negative log-likelihood per frame, masked and mormalized according to sequence lengths.

        Args:
            target (torch.LongTensor): Input audio waveform, i.e. the target (B, T) of quantized integers.
            x_sl (torch.LongTensor): Sequence lengths of examples in the batch.
            logits (torch.FloatTensor): Model reconstruction with log softmax scores per possible frame value (B, C, T).
        """
        nll = self.nll_criterion(logits, target)
        mask = sequence_mask(x_sl, device=nll.device)
        nll *= mask
        nll = nll.sum(1)  # sum T
        loss = nll.nansum() / x_sl.nansum()  # sum B, normalize by sequence lengths
        return loss, -nll

    def forward(self, x: TensorType["B", "T", "C", float], x_sl: TensorType["B", int]):
        """Reconstruct an input and compute the log-likelihood loss.

        The duration of x has to be longer than the receptive field.

        Args:
            x (torch.Tensor): Audio waveform (batch, timestep, channels) with values in [-1, 1] (optinally dequantized)
            x_sl (torch.LongTensor): Sequence lengths of each example in the batch.
        """
        if self.in_channels == 1:
            x = x.unsqueeze(-1) if x.ndim == 2 else x  # (B, T, C)
            target = self._get_target(x)
        else:
            target = x.clone()
            x = self.embedding(x)  # (B, T, C)

        x = x.transpose(1, 2)  # (B, C, T)

        self.check_input_size(x)

        output = self.causal(x)
        skip_connections = self.res_stack(output, skip_size=x.size(2))
        output = torch.sum(skip_connections, dim=0)
        logits = self.out_convs(output)

        loss, ll = self.compute_loss(target, x_sl, logits)

        x_hat = logits.argmax(1) / (self.out_classes - 1)
        x_hat = (2 * x_hat) - 1

        metrics = [LossMetric(loss, weight_by=ll.numel()), LLMetric(ll), BitsPerDimMetric(ll, reduce_by=x_sl)]
        output = SimpleNamespace(loss=loss, ll=ll, logits=logits, target=target, x_hat=x_hat)
        return loss, metrics, output

    def generate(self, n_samples: int, n_frames: int = 48000):
        """Generate samples from the WaveNet starting from a zero vector"""
        if self.in_channels == 1:
            # start with floats of zeros
            x = torch.zeros(n_samples, self.receptive_field, 1, device=self.device)  # (B, T, C)
        else:
            # start with embeddings of the zeros
            x = torch.zeros(n_samples, self.receptive_field, device=self.device, dtype=torch.int64)  # (B, T)
            x = self.embedding(x)  # (B, T, C)

        x = x.transpose(1, 2)  # (B, C, T)

        x_hat = []
        for _ in tqdm.tqdm(range(n_frames)):

            output = self.causal(x, pad=False)
            skip_connections = self.res_stack(output, skip_size=1)
            output = torch.sum(skip_connections, dim=0)
            output = self.out_convs(output)

            categorical = D.Categorical(logits=output.transpose(1, 2))
            x_new = categorical.sample()  # Value in {0, ..., 255}
            x_hat.append(x_new)

            # prepare prediction as next input
            if self.in_channels == 1:
                x_new = x_new.unsqueeze(-1)  # (B, T, C) (1, 1, 1)
                x_new = x_new / (self.out_classes - 1)  # To [0, 1]
                x_new = x_new * 2 - 1  # To [-1, 1]
            else:
                x_new = self.embedding(x_new)  # (B, T, C) (1, 1, C)

            x_new = x_new.transpose(1, 2)  # (B, C, T)

            x = torch.cat([x[:, :, 1:], x_new], dim=2)  # FIFO along T

        x_hat = torch.hstack(x_hat)
        x_hat = x_hat / (self.out_classes - 1)  # To [0, 1]
        x_hat = x_hat * 2 - 1  # To [-1, 1]
        return x_hat
