"""Mdules for WaveNet"""

import torch

from torchtyping import TensorType


class CausalConv1d(torch.nn.Module):
    """Causal Convolution for WaveNet. Causality imposed by removing last timestep of output (and left same padding)"""

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

    def causal_padding(self, x):
        """Pad with receptive field and remove last input (causal convolution)"""
        return torch.nn.functional.pad(x, (self.receptive_field, -1))

    def forward(self, x: TensorType["B", "C", "T", float], pad: bool = True):
        if pad:
            x = self.causal_padding(x)
        output = self.conv(x)
        return output


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, dilation):
        """Residual block

        Args:
            res_channels (int): number of residual channel for input, output
            dilation (int): amount of dilation
        """
        super().__init__()

        self.dilated = torch.nn.Conv1d(res_channels, res_channels, kernel_size=2, dilation=dilation)
        self.conv1x1 = torch.nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size):
        """
        Args:
            x (torch.Tensor): Input
            skip_size (int): The last output size for loss and prediction
        """
        gate_in = self.dilated(x)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(gate_in)
        gated_sigmoid = self.gate_sigmoid(gate_in)
        gate_out = gated_tanh * gated_sigmoid

        # Residual network
        conv1x1_out = self.conv1x1(gate_out)
        residual = x[:, :, -conv1x1_out.size(2) :]  # Remove superfluous timesteps
        output = conv1x1_out + residual

        # Skip connection
        skip = conv1x1_out[:, :, -skip_size:]

        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(self, n_layers, n_stacks, res_channels):
        """Stack residual blocks by layer and stack size

        Args:
            n_layers (int): Number of stacked residual blocks (k). Dilations chosen as 2, 4, 8, 16, 32, 64...
            n_stacks (int): Number of stacks of residual blocks with skip connections to the output.
            res_channels (int): Number of channels in the residual connections.
        """
        super().__init__()

        self.n_layers = n_layers
        self.n_stacks = n_stacks

        self.dilations = self.build_dilations()

        res_blocks = torch.nn.ModuleList()
        for dilation in self.dilations:
            block = ResidualBlock(res_channels, dilation)
            res_blocks.append(block)

        self.res_blocks = res_blocks

    def build_dilations(self):
        """Return a list of dilations {2, 4, 8, 16, ...} with a dilation for each of the residual blocks"""
        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        dilations = []
        for _ in range(self.n_stacks):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l in range(self.n_layers):
                dilations.append(2 ** l)

        return dilations

    def forward(self, x, skip_size):
        """
        Args:
            x (torch.Tensor): Input
            skip_size (int): The last output size for loss and prediction
        """
        output = x
        skip_connections = []
        
        for i, res_block in enumerate(self.res_blocks):
            # output is the next input
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)

        return torch.stack(skip_connections)


class OutConv1d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """The last network of WaveNet. Outputs log probabilities of frame classes.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output classes
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.relu2 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.relu1(x)
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)

        output = self.log_softmax(output)

        return output
