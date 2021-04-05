"""WaveNet main model"""

from torch._C import import_ir_module_from_buffer
import tqdm

from collections import namedtuple

import numpy as np
import torch
import torch.distributions as D

from vseq.utils.operations import sequence_mask

from .modules import CausalConv1d, ResidualStack, DenseNet
from ..base_module import BaseModule


Output = namedtuple("Output", ["loss", "likelihood", "logits", "categorical"])


class InputSizeError(Exception):
    def __init__(self, input_size, receptive_field, output_size):
        message = "Input size has to be larger than receptive_field\n"
        message += f"Input size: {input_size}, Receptive fields size: {receptive_field}, Output size: {output_size}"
        super().__init__(message)


class WaveNet(BaseModule):
    def __init__(
        self, layer_size: int, stack_size: int, res_channels: int, in_channels: int = 1, out_classes: int = 256
    ):
        """Stochastic autoregressive modelling of audio waveform frames with conditional dilated causal convolutions.

        The total number of residual blocks (layers) used is equal to the layer_size times the stack_size.
        This is `k` in Figure 4 in the paper.

                       |----------------------------------------|     *residual*
                       |                                        |
                       |    |-- conv -- tanh --|                |
            -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
                            |-- conv -- sigm --|     |
                                                    1x1
                                                     |
            ---------------------------------------> + ------------->	*skip*

        Args:
            layer_size (int): Number of stacked residual blocks. Dilations chosen as 2, 4, 8, 16, 32, 64...
            stack_size (int): Number of stacks of residual blocks with skip connections to the output.
            in_channels (int): Number of channels in the input data. Skip channel is the same as the input channel.
            res_channels (int): Number of channels in residual blocks.

        Reference:
            [1] WaveNet: A Generative Model for Raw Audio (https://arxiv.org/abs/1609.03499)
        """
        super().__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.out_classes = out_classes

        self.receptive_field = self.calc_receptive_field(layer_size, stack_size)

        self.causal = CausalConv1d(in_channels, res_channels)

        self.res_stack = ResidualStack(
            layer_size=layer_size, stack_size=stack_size, res_channels=res_channels, skip_channels=res_channels
        )

        self.densenet = DenseNet(res_channels, out_classes)

        self.nll_criterion = torch.nn.NLLLoss(reduction="none")

    @staticmethod
    def calc_receptive_field(layer_size, stack_size):
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        receptive_field = np.sum(layers)
        return int(receptive_field)

    def calc_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_field
        self.check_input_size(x, output_size)
        return output_size

    def check_input_size(self, x, output_size):
        if output_size < 1:
            raise InputSizeError(int(x.size(2)), self.receptive_field, output_size)

    def compute_loss(self, x: torch.FloatTensor, x_sl: list[int], output: torch.FloatTensor):
        """To compute the loss, the inputs must be compared to outputs shifted by the receptive field.

        Args:
            x (torch.FloatTensor): Input audio waveform, i.e. the target (B, T) (may be dequantized with [0, 1) noise)
            x_sl (list[int]): Sequence lengths in the batch
            output (torch.FloatTensor): Model reconstruction with log softmax scores per possible frame value (B, C, T)
        """
        # Convert x to targets
        assert x.shape[1] == 1, "Multiple input channels not supported"
        target = x.squeeze(1)  # (B, C, T) to (B, T)  NOTE Only works for one input channel
        target = (target + 1) / 2  # Transform [-1, 1] to [0, 1]
        target = target * (self.out_classes - 1)  # Transform [0, 1] to [0, 255]
        target = target[:, self.receptive_field :]  # Causal
        target = target.floor().to(torch.int64)  # To integer (floor because of added noise for dequantization)

        likelihood = - self.nll_criterion(output, target)

        # Mask and reduce likelihood
        x_sl = torch.cuda.LongTensor(x_sl) if likelihood.is_cuda else torch.LongTensor(x_sl)
        x_sl -= self.receptive_field
        mask = sequence_mask(x_sl, device=likelihood.device)
        likelihood *= mask
        loss = - likelihood.sum(1).mean()  # Sum T, mean B
        return loss, likelihood

    def forward(self, x, x_sl):
        """Reconstruct an input and compute the log-likelihood loss.

        The duration of x has to be longer than the receptive field.

        Args:
            x (torch.Tensor): Audio waveform (batch, timestep, channels) with values in [-1, 1] (optinally dequantized)
            x_sl (list): Sequence lengths of each example in the batch.
        """
        x = x.unsqueeze(-1) if x.ndim == 2 else x
        x = x.transpose(1, 2)  # (batch, channels, timestep)

        output_size = self.calc_output_size(x)

        output = self.causal(x)

        skip_connections = self.res_stack(output, skip_size=output_size)

        output = torch.sum(skip_connections, dim=0)

        output = self.densenet(output)

        loss, likelihood = self.compute_loss(x, x_sl, output)

        categorical = D.Categorical(logits=output.transpose(1, 2))

        output = Output(loss=loss, likelihood=likelihood, logits=output, categorical=categorical)

        return loss, output

    def generate(self, n_samples: int, n_frames: int = 48000):

        x = torch.zeros(n_samples, 1, self.receptive_field + 1, device=self.device)  # (bath, channel, timestep)
        output_size = self.calc_output_size(x)

        for _ in tqdm.tqdm(range(n_frames)):
            output = self.causal(x[..., -self.receptive_field - 1:])

            skip_connections = self.res_stack(output, skip_size=output_size)

            output = torch.sum(skip_connections, dim=0)

            output = self.densenet(output)

            categorical = D.Categorical(logits=output.transpose(1, 2))

            x_new = categorical.sample()  # Value in {0, ..., 255}
            x_new = x_new / (self.out_classes - 1)  # To [0, 1]
            x_new = x_new * 2 - 1  # To [-1, 1]

            x = torch.cat([x, x_new.unsqueeze(1)], dim=2)

        return x
