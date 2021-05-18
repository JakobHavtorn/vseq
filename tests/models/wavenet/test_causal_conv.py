"""
Test Dilated Causal Convolution
"""
import logging

import torch
import pytest
import numpy as np

from vseq.models.wavenet import CausalConv1d


LOGGER = logging.getLogger(name=__file__)


def causal_conv(data, in_channels, out_channels, receptive_field, print_result=True):
    conv = CausalConv1d(in_channels, out_channels, receptive_field)
    conv.init_weights_for_test()

    output = conv(data)

    LOGGER.debug("Causal convolution")
    if print_result:
        LOGGER.debug("    {0}".format(output.data.numpy().astype(int)))

    return output


@pytest.fixture
def input_data():
    """Input data [1, 2, 3, 4, 5, ..., 30, 31, 32]"""
    x = np.arange(1, 33, dtype=np.float32)
    x = np.reshape(x, [1, 2, 16])  # (B, C, T)
    x = torch.from_numpy(x)

    LOGGER.debug("\nInput size={0}".format(x.shape))
    LOGGER.debug(x.data.numpy().astype(int))
    return x


def test_causal_conv(input_data):
    """Test normal convolution 1d"""
    CAUSAL_RESULT = [[[38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90]]]

    result = causal_conv(input_data, 2, 1, receptive_field=0)

    np.testing.assert_array_equal(result.data.numpy().astype(int), CAUSAL_RESULT)

    return result


def test_causal_conv_nonzero_receptive_field(input_data):
    """Test normal convolution 1d"""
    CAUSAL_RESULT = [[[0, 0, 0, 0, 18, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90]]]

    result = causal_conv(input_data, 2, 1, receptive_field=5)

    print(input_data)

    np.testing.assert_array_equal(result.data.numpy().astype(int), CAUSAL_RESULT)

    return result
