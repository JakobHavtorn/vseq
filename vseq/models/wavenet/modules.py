"""Mdules for WaveNet

References:
    https://arxiv.org/pdf/1609.03499.pdf
    https://github.com/ibab/tensorflow-wavenet
    https://qiita.com/MasaEguchi/items/cd5f7e9735a120f27e2a
    https://github.com/musyoku/wavenet/issues/4
"""

import torch


class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet"""

    def __init__(self, channels, dilation=1):
        super().__init__()

        self.conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=2,
            stride=1,
            dilation=dilation,
            padding=0,
            bias=False,
        )

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        return output


class CausalConv1d(torch.nn.Module):
    """Causal Convolution for WaveNet"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # padding=1 for same size(length) between input and output for causal convolution
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=2, stride=1, padding=1, bias=False)

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        # remove last timestep for causal convolution
        return output[:, :, :-1]


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        """
        Residual block
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :param dilation:
        """
        super().__init__()

        self.dilated = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = self.dilated(x)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid

        # Residual network
        output = self.conv_res(gated)
        input_cut = x[:, :, -output.size(2) :]
        output += input_cut

        # Skip connection
        skip = self.conv_skip(gated)
        skip = skip[:, :, -skip_size:]

        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels):
        """
        Stack residual blocks by layer and stack size

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

        for res_block in self.res_blocks:
            # output is the next input
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)

        return torch.stack(skip_connections)


class DenseNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """The last network of WaveNet. Outputs log probabilities of each frame value class.

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
