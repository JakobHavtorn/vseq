import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class TasNetEncoder(nn.Module):
    def __init__(
        self,
        channels_autoencoder: int = 512,  # TODO Not super useful for our case?
        channels_bottleneck: int= 128,
        channels_block: int = 512,
        kernel_size: int = 3,
        num_blocks: int = 8,
        num_levels: int = 3,
        num_speakers: int = None,  # TODO What does this correspond to in our case?
        norm_type: str = "GlobalLayerNorm",
        causal: bool = False,
        mask_nonlinearity = "relu",
    ):
        """TasNet-based encoder similar to the paper https://arxiv.org/pdf/1809.07454.pdf.

        Default arguments above correspond to the highest performing model in the paper (Table II)

        Args:
            channels_autoencoder: Number of filters in autoencoder
            channels_bottleneck: Number of channels in bottleneck 1 × 1-conv block
            channels_block: Number of channels in convolutional blocks
            kernel_size: Kernel size in convolutional blocks
            num_blocks: Number of convolutional blocks in each repeat
            num_levels: int: Number of repeats
            num_speakers: Number of speakers
            norm_type: GlobalLayerNorm, ChannelwiseLayerNorm, TemporalLayerNorm
            causal: causal or non-causal
            mask_nonlinearity: use which non-linear function to generate mask
        """
        super().__init__()
        # Hyper-parameter
        self.num_speakers = num_speakers
        self.mask_nonlinearity = mask_nonlinearity
        # Components
        # [B, channels_autoencoder, T] -> [B, channels_autoencoder, T]
        layer_norm = nn.GroupNorm(num_channels=channels_autoencoder, num_groups=channels_autoencoder)
        # [B, channels_autoencoder, T] -> [B, channels_bottleneck, T]
        bottleneck_conv1x1 = nn.Conv1d(channels_autoencoder, channels_bottleneck, 1, bias=False)
        # [B, channels_bottleneck, T] -> [B, channels_bottleneck, T]
        repeats = []
        for r in range(num_levels):
            blocks = []
            for x in range(num_blocks):
                dilation = 2 ** x
                padding = (kernel_size - 1) * dilation if causal else (kernel_size - 1) * dilation // 2
                blocks += [
                    TemporalBlock(
                        channels_bottleneck,
                        channels_block,
                        kernel_size,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    )
                ]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # [B, channels_bottleneck, T] -> [B, num_speakers*channels_autoencoder, T]
        mask_conv1x1 = nn.Conv1d(channels_bottleneck, num_speakers * channels_autoencoder, 1, bias=False)
        # Put together
        self.network = nn.Sequential(layer_norm, bottleneck_conv1x1, temporal_conv_net, mask_conv1x1)

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [B, channels_autoencoder, T], B is batch size
        returns:
            est_mask: [B, num_speakers, channels_autoencoder, T]
        """
        B, channels_autoencoder, T = mixture_w.size()
        score = self.network(mixture_w)  # [B, channels_autoencoder, T] -> [B, num_speakers*channels_autoencoder, T]
        score = score.view(
            B, self.num_speakers, channels_autoencoder, T
        )  # [B, num_speakers*channels_autoencoder, T] -> [B, num_speakers, channels_autoencoder, T]
        if self.mask_nonlinearity == "softmax":
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinearity == "relu":
            est_mask = F.relu(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_out_channels,
        hidden_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="GlobalLayerNorm",
        causal=False,
    ):
        super().__init__()

        if causal:
            raise NotImplementedError("This used to add a 'Chomp' layer")

        # [B, channels_bottleneck, T] -> [B, channels_block, T]
        conv1x1 = nn.Conv1d(in_out_channels, hidden_channels, 1, bias=False)
        prelu = nn.PReLU()
        # norm = nn.GroupNorm(num_channels=in_out_channels, num_groups=in_out_channels)
        norm = GlobalLayerNorm(num_channels=in_out_channels)
        # [B, channels_block, T] -> [B, channels_bottleneck, T]
        dsconv = ConvDepthwiseSeparable1d(hidden_channels, in_out_channels, kernel_size, stride, padding, dilation, norm_type)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [B, channels_bottleneck, T]
        Returns:
            [B, channels_bottleneck, T]
        """
        residual = x
        out = self.net(x)
        # TODO: when kernel_size = 3 here works fine, but when kernel_size = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)


class ConvDepthwiseSeparable1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, norm_type):
        super().__init__()
        # Use `groups` option to implement depthwise convolution
        # [B, channels_block, T] -> [B, channels_block, T]
        depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        prelu = nn.PReLU()
        norm = get_normalization(norm_type, num_channels=in_channels)
        # [B, channels_block, T] -> [B, channels_bottleneck, T]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [B, channels_block, T]
        Returns:
            result: [B, channels_bottleneck, T]
        """
        return self.net(x)


def get_normalization(norm_type: str, num_channels: int):
    """The input of normlization will be (B, num_speakers, T), where B is batch size,
    num_speakers is channel size and T is sequence length.
    """
    if norm_type == "GlobalLayerNorm":
        return GlobalLayerNorm(num_channels)
    elif norm_type == "ChannelwiseLayerNorm":
        return nn.GroupNorm(num_channels=num_channels, num_groups=num_channels)
    raise NotImplementedError(f"Unknown {norm_type=}")


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (GlobalLayerNorm)"""

    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, num_channels, 1))  # [1, channels_autoencoder, 1]
        self.beta = nn.Parameter(torch.Tensor(1, num_channels, 1))  # [1, channels_autoencoder, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y, eps: float = 1e-8):
        """
        Args:
            y: [B, channels_autoencoder, T], B is batch size, channels_autoencoder is channel size, T is length
        Returns:
            GlobalLayerNorm_y: [B, channels_autoencoder, T]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [B, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        GlobalLayerNorm_y = self.gamma * (y - mean) / torch.pow(var + eps, 0.5) + self.beta
        return GlobalLayerNorm_y


if __name__ == "__main__":
    import IPython

    IPython.embed(using=False)
