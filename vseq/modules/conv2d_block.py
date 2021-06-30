from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

DEEP_LSTM_2D_CONV_FRONTEND = ((32, (3, 3), (2, 2)),
                              (32, (3, 3), (2, 1)),
                              (32, (3, 3), (2, 1)))

def same_pad(input, layer):
    padding = []
    pdim = 2
    for s, k in zip(layer.stride, layer.kernel_size):
        t = input.shape[pdim]
        out_dim = ceil(t / s)
        pad_total = max((out_dim - 1) * s + k - t, 0)
        pad_start = int(pad_total // 2)
        pad_end = int(pad_total - pad_start)
        padding = [pad_start, pad_end] + padding
        pdim += 1
    
    return F.pad(input, padding, mode='constant', value=0)

class Conv2dBlock(nn.Module):

    def __init__(self, config=DEEP_LSTM_2D_CONV_FRONTEND, input_channels=1, activation=nn.ReLU6, dropout_prob=0.4):
        """
        Block of 2D-convolutional layers.

        Args:
            config (list): List of lists, such that len(config) == num_layers and config[idx] == [out_channels, 
                kernel_size, stride].
            input_channels (int): The number of input channels.
            activation (func): Non-linear function to use after each convolutional layer.
        """
        super(Conv2dBlock, self).__init__()
        self.config = config
        self.input_channels = input_channels
        self.activation = activation
        self.p = dropout_prob
        self.conv2d_layers = []
        self.activations = []
        self.dropout_layers = []

        in_channels = input_channels
        for idx, (out_channels, kernel_size, stride) in enumerate(config):
            conv2d_layer = nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride)
            act = activation()
            dropout_layer = nn.Dropout2d(p=dropout_prob)
            setattr(self, f"conv2d_layer_{idx}", conv2d_layer)  
            setattr(self, f"act_{idx}", act)
            setattr(self, f"dropout_layer_{idx}", dropout_layer) 
            self.conv2d_layers.append(conv2d_layer)
            self.activations.append(act)
            self.dropout_layers.append(dropout_layer)
            in_channels = out_channels

    def forward(self, input, seq_lens=None):
        """
        Args:
            input (Tensor): Input with shape (batch x frq_dim x time) and dtype == float32.
            seq_lens (Tensor): The sequence lengths of the input of size (N) with dtype == int64.
        
        Returns:
            Tensor: Output of shape (batch x filters x frq_dim x time) and dtype == float32.
            Tensor: 'seq_lens' reduced according to temporal stride. Only if 'seq_lens' was given.
        """
        
        x = input
        for conv2d_layer, act, dropout_layer in zip(self.conv2d_layers, self.activations, self.dropout_layers):
            x = same_pad(x, conv2d_layer)
            x = conv2d_layer(x)
            x = act(x)
            x = dropout_layer(x)
            if seq_lens is not None:
                temp_stride = conv2d_layer.stride[1]
                seq_lens = torch.true_divide(seq_lens, temp_stride).ceil().to(torch.long)

        return x if seq_lens is None else (x, seq_lens)
        