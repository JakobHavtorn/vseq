from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from vseq.modules import Dropout1d

LSTM_AR_1D_CONV_FRONTEND = ((32, 2, 1), # num_filters, kernel_size, dilation
                            (64, 2, 2),
                            (96, 2, 4),
                            (128, 2, 8),
                            (256, 2, 16))

def causal_pad_1d(inputs, layer, skip_last=False):
    D = layer.dilation[0]
    K = layer.kernel_size[0]
    padding = ((D - 1) * (K - 1) + K) - int(not skip_last)
    inputs  = inputs[:, :, :-1] if skip_last else inputs
    return F.pad(inputs, [padding, 0], mode='constant', value=0)

class CausalConv1dBlock(nn.Module):

    def __init__(self, config=LSTM_AR_1D_CONV_FRONTEND, input_channels=1, activation=nn.ReLU6, dropout_prob=0.4,
                 bottleneck_size=256):
        """
        Block of 1D-convolutional layers with group normalization.

        Args:
            config (list): List of lists, such that len(config) == num_layers and config[idx] == [out_channels, 
                kernel_size, stride].
            input_channels (qint): The number of input channels.
            activation (func): Non-linear function to use after each convolutional layer.
        """
        super(CausalConv1dBlock, self).__init__()
        self.config = config
        self.input_channels = input_channels
        self.activation = activation
        self.p = dropout_prob
        self.bottleneck_size = bottleneck_size
        self.conv1d_layers = []
        self.activations = []
        self.dropout_layers = []
        self.group_norm_layers = []

        in_channels = input_channels
        for idx, (out_channels, kernel_size, dilation) in enumerate(config):
            conv1d_layer = nn.Conv1d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=1, # assumed to keep temporal resolution
                                     dilation=dilation)
            num_groups = min(32, out_channels // 2)
            group_norm_layer = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            act = activation()
            dropout_layer = Dropout1d(p=dropout_prob)
            setattr(self, f"conv1d_layer_{idx}", conv1d_layer)
            setattr(self, f"group_norm_layer_{idx}", group_norm_layer)
            setattr(self, f"act_{idx}", act)
            setattr(self, f"dropout_layer_{idx}", dropout_layer)  
            self.conv1d_layers.append(conv1d_layer)
            self.activations.append(act)
            self.dropout_layers.append(dropout_layer)
            self.group_norm_layers.append(group_norm_layer)
            in_channels = out_channels
        
        if bottleneck_size:
            num_inputs = sum([n for n, _, _ in config])
            self.bottleneck_layer = nn.Linear(num_inputs, bottleneck_size)
            self.bottleneck_act = activation()

    def forward(self, input, seq_lens=None):
        """
        Args:
            input (Tensor): Input with shape (batch x frq_dim x time) and dtype == float32.
            seq_lens (Tensor): The sequence lengths of the input of size (N) with dtype == int64.
        
        Returns:
            Tensor: Output of shape (batch x filters x frq_dim x time) and dtype == float32.
            Tensor: 'seq_lens' reduced according to temporal stride. Only if 'seq_lens' was given.
        """
        
        if input.ndim == 2:
            x = input.unsqueeze(1)
        else:
            x = input
            
        layers = zip(self.conv1d_layers, self.activations, self.dropout_layers, self.group_norm_layers)
        skip_last = True
        outputs = []
        for conv1d_layer, act, dropout_layer, group_norm_layer in layers:
            x = causal_pad_1d(x, conv1d_layer, skip_last=skip_last)
            x = conv1d_layer(x)
            x = group_norm_layer(x)
            x = act(x)
            x = dropout_layer(x)
            skip_last = False
            outputs.append(x.transpose(1, 2))

        if self.bottleneck_size:
            x = torch.cat(outputs, 2)
            x = self.bottleneck_layer(x)
            x = self.bottleneck_act(x)
        
        return x if seq_lens is None else (x, seq_lens)
    
    # def generate(self, input):
    
    #     if input.ndim == 2:
    #         x = input.unsqueeze(1)
    #     else:
    #         x = input
            
    #     layers = zip(self.conv1d_layers, self.activations, self.dropout_layers, self.group_norm_layers)
    #     skip_last = True
    #     outputs = []
    #     for conv1d_layer, act, dropout_layer, group_norm_layer in layers:
    #         # x = causal_pad_1d(x, conv1d_layer, skip_last=skip_last)
            
    #         x = conv1d_layer(x)
    #         x = group_norm_layer(x)
    #         x = act(x)
    #         x = dropout_layer(x)
    #         skip_last = False
    #         outputs.append(x.transpose(1, 2))

    #     if self.bottleneck_size:
    #         x = torch.cat(outputs, 2)
    #         x = self.bottleneck_layer(x)
    #         x = self.bottleneck_act(x)
        
    #     return x if seq_lens is None else (x, seq_lens)
        