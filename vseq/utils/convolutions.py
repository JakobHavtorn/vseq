import numpy as np
import torch


def get_same_padding(convolution, in_shape=None):
    """
    Return the padding to apply to a given convolution such as it reproduces the 'same' behavior from Tensorflow

    This also works for pooling layers.

    For transposed convolutions, the symmetric padding is always returned as None.

    Args:
        in_shape (tuple): Input tensor shape excluding batch (D1, D2, ...)
        convolution (nn.Module): Convolution module object
    returns:
        sym_padding, unsym_padding: Symmetric and unsymmetric padding to apply. We split in two because nn.Conv only
                                    allows setting symmetric padding so unsymmetric has to be done manually.
    """
    kernel_size = np.asarray(convolution.kernel_size)
    pad = np.asarray(convolution.padding)
    dilation = np.asarray(convolution.dilation) if hasattr(convolution, "dilation") else 1
    stride = np.asarray(convolution.stride)

    # handle pooling layers
    if not hasattr(convolution, "transposed"):
        convolution.transposed = False

    if not convolution.transposed:
        in_shape = np.asarray(in_shape)
        assert len(in_shape) == len(kernel_size), "tensor is not the same dimension as the kernel"
        # effective_filter_size = (kernel_size - 1) * dilation + 1
        output_size = (in_shape - 1) // stride + 1
        padding_input = np.maximum(0, (output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - in_shape)
        odd_padding = padding_input % 2 != 0
        sym_padding = tuple(padding_input // 2)
        unsym_padding = [y for x in odd_padding for y in [0, int(x)]]
    else:
        padding_input = kernel_size - stride
        sym_padding = None
        unsym_padding = [
            y for x in padding_input for y in [-int(np.floor(int(x) / 2)), -int(np.floor(int(x) / 2) + int(x) % 2)]
        ]

    return sym_padding, unsym_padding
