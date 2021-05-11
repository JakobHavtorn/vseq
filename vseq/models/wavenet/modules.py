"""Mdules for WaveNet

References:
    https://arxiv.org/pdf/1609.03499.pdf
    https://github.com/ibab/tensorflow-wavenet
    https://qiita.com/MasaEguchi/items/cd5f7e9735a120f27e2a
    https://github.com/musyoku/wavenet/issues/4
"""

import torch

from torchtyping import TensorType


class CausalConv1d(torch.nn.Module):
    """Causal Convolution for WaveNet. Causality imposed by removing last timestep of output (and left same padding)
    
    """

    def __init__(self, in_channels: int, out_channels: int, receptive_field: int):
        super().__init__()
        self.receptive_field = receptive_field
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=2, bias=False)
        # TODO Add activation function and bias?

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)
                # m.bias.data.fill_(0)

    def forward(self, x: TensorType["B", "C", "T", float], pad: bool = True):
        if pad:
            # Pad with receptive field and remove last input (causal convolution)
            x = torch.nn.functional.pad(x, (self.receptive_field, -1))
        output = self.conv(x)
        return output


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        """Residual block

        Args:
            res_channels (int): number of residual channel for input, output
            skip_channels (int): number of skip channel for output
            dilation (int): amount of dilation

        TODO Should we have a single 1x1 convolution from which we form the "output" and the "skip connection", or
             should of these have their own?
        """
        super().__init__()

        self.dilated = torch.nn.Conv1d(res_channels, res_channels, kernel_size=2, dilation=dilation)
        self.conv1x1_res = torch.nn.Conv1d(res_channels, res_channels, kernel_size=1)
        # self.conv1x1_skip = torch.nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        gate_in = self.dilated(x)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(gate_in)
        gated_sigmoid = self.gate_sigmoid(gate_in)
        gate_out = gated_tanh * gated_sigmoid

        # Residual network
        # import IPython; IPython.embed()
        conv1x1_out = self.conv1x1_res(gate_out)
        residual = x[:, :, -conv1x1_out.size(2) :]  # Remove superfluous timesteps
        output = conv1x1_out + residual

        # Skip connection
        skip = conv1x1_out[:, :, -skip_size:]
        # skip = self.conv1x1_skip(gate_out)
        # skip = skip[:, :, -skip_size:]

        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels):
        """Stack residual blocks by layer and stack size

        Args:
            layer_size (int): Number of stacked residual blocks (k). Dilations chosen as 2, 4, 8, 16, 32, 64...
            stack_size (int): Number of stacks of residual blocks with skip connections to the output.
            res_channels (int): Number of channels in the residual connections.
            skip_channels (int): Number of channels in the skip connections (to sum to give output).
        """
        super().__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size

        self.dilations = self.build_dilations()

        res_blocks = torch.nn.ModuleList()
        for dilation in self.dilations:
            block = ResidualBlock(res_channels, skip_channels, dilation)
            res_blocks.append(block)

        self.res_blocks = res_blocks

    def build_dilations(self):
        """Return a list of dilations {2, 4, 8, 16, ...} with a dilation for each of the residual blocks"""
        dilations = []

        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        for s in range(0, self.stack_size):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l in range(0, self.layer_size):
                dilations.append(2 ** l)

        return dilations

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = x
        skip_connections = []

        for i, res_block in enumerate(self.res_blocks):
            # output is the next input
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)

        return torch.stack(skip_connections)


class DenseNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """The last network of WaveNet. Outputs log probabilities of frame classes.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output classes
        """        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = torch.nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = torch.nn.Conv1d(in_channels, out_channels, 1)

        self.relu = torch.nn.ReLU()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)

        output = self.log_softmax(output)

        return output
