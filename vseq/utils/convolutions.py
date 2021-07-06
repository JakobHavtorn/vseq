from itertools import repeat
from typing import List, Union

import numpy as np

from torch._six import container_abcs


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


def _ntuple(n):
    """Given an integer, return that integer in an n-tuple. Given an Iterable, return that directly instead"""

    def parse(x):
        """Given an integer, return that integer in an n-tuple. Given an Iterable, return that directly instead"""
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def compute_conv_attributes_single(i=np.nan, k=np.nan, p=np.nan, s=np.nan, j_in=1, r_in=1, start_in=0):
    """Computes the output channels and receptive field of a (potentially N-dimensional) convolution.

    To calculate the receptive field in each layer, besides the number of features n in each dimension, we need to
    keep track of some extra information for each layer. These include the current receptive field size r, the
    distance between two adjacent features ("jump" or "effective stride") j, and the center coordinate of the upper left
    feature (the first feature) start.

    Note that the center coordinate of a feature is defined to be the center coordinate of its receptive field.

    All the arguments default to nan since, in order to get any of them at the output layer, for instance the receptive
    field, not all are required and nan allows running all calculations regardless.

    The requirements are:
        o_out:      i, k, p, s
        j_out:      j_in, s
        r_out:      r_in, j_in, k, s
        start_out:  start_in, i, j_in, k, p, s

    [1] https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807

    Args:
        i (int or np.ndarray of int): Number of features
        k (int or np.ndarray of int): Kernel size
        p (int or np.ndarray of int): Amount of padding applied
        s (int or np.ndarray of int): Size of the stride used
        j_in (int): Distance between the centers of two adjacent features
        r_in (int): Receptive field of a feature.
        start_in (float): Position of the first feature's receptive field in layer i. Defaults to 0.
                          (index starts from 0, negative means center is in the padding)

    Returns:
        tuple: (o_out, j_out, r_out, start_out) corresponding to the above but mapped through this conv layer.
    """
    o_out = ((i - k + 2 * p) // s) + 1
    actual_padding = (o_out - 1) * s - i + k
    pad_left = actual_padding // 2
    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pad_left) * j_in
    return o_out, j_out, r_out, start_out


def compute_conv_attributes(
    in_shape: Union[tuple, int],
    kernels: List[Union[tuple, int]],
    paddings: List[Union[tuple, int]],
    strides: List[Union[tuple, int]],
    jump_in: Union[tuple, int] = 1,
    receptive_field_in: Union[tuple, int] = 1,
    start_in: Union[tuple, int] = 0,
):
    """Repeatedly applies `compute_conv_attributes_single` to multiple consecutive N-dimensional convolutions.

    Args:
        in_shape (tuple or int): Input length
        kernels (list of tuple or int): Kernel sizes
        paddings (list of tuple or int): Padding sizes
        strides (list of tuple or int): Stride sizes
        jump_in (tuple or int, optional): Distance between the centers of two adjacent features. Defaults to 1.
        receptive_field_in (tuple or int, optional): Receptive field of a feature. Defaults to 1.
        start_in (tuple or int, optional): Position of the first feature's receptive field in layer i. Defaults to 0.
                                           (index starts from zero, negative means center is in the padding).

    Returns:
        tuple: (out_shape, jump_out, receptive_field_out, start_out) corresponding to the above but mapped through this conv layer.
    """
    # Check inputs
    assert len(kernels) == len(paddings) == len(strides), "Number of layers in each of the parameters must be equal"

    all_n_dims = {len(in_shape)} if isinstance(in_shape, tuple) else {1}
    for k, p, s in zip(kernels, paddings, strides):
        all_n_dims.add(len(k) if isinstance(k, tuple) else 1)
        all_n_dims.add(len(p) if isinstance(p, tuple) else 1)
        all_n_dims.add(len(s) if isinstance(s, tuple) else 1)
    assert len(all_n_dims) == 1 or (
        len(all_n_dims) == 2 and min(all_n_dims) == 1
    ), f"Must give only tuples (or ints) of same dimensionality but got {len(all_n_dims)} different dims: {all_n_dims}"

    # Chosen dimensionality is the maximum of the two encountered (i.e. 1 or N)
    n_dims = max(all_n_dims)

    # Convert all parameters from a mix of ints and tuples to tuples (arrays) of the same dimensionality
    in_shape = np.array(_ntuple(n_dims)(in_shape))

    jump_in = np.array(_ntuple(n_dims)(jump_in))
    receptive_field_in = np.array(_ntuple(n_dims)(receptive_field_in))
    start_in = np.array(_ntuple(n_dims)(start_in))

    kernels = [np.array(_ntuple(n_dims)(k)) for k in kernels]
    paddings = [np.array(_ntuple(n_dims)(k)) for k in paddings]
    strides = [np.array(_ntuple(n_dims)(k)) for k in strides]

    out_shape = in_shape
    for k, p, s in zip(kernels, paddings, strides):
        out_shape, jump_in, receptive_field_in, start_in = compute_conv_attributes_single(
            out_shape, k, p, s, jump_in, receptive_field_in, start_in
        )

    return (
        tuple(out_shape.tolist()),
        tuple(jump_in.tolist()),
        tuple(receptive_field_in.tolist()),
        tuple(start_in.tolist()),
    )
