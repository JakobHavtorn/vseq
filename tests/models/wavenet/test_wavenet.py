"""
Test WaveNet model
"""

import numpy as np
import torch

import pytest

from vseq.models.wavenet.model import WaveNet, InputSizeError


LAYER_SIZE = 5  # 10 in paper
STACK_SIZE = 2  # 5 in paper
IN_CHANNELS = 1  # 256 in paper. quantized and one-hot input.
RES_CHANNELS = 512  # 512 in paper
OUT_CLASSES = 256


def generate_dummy(dummy_length):
    # x = np.arange(0, dummy_length, dtype=np.float32)
    x = np.random.uniform(low=-1, high=1, size=(1, dummy_length)).astype(np.float32)
    x = np.reshape(x, [1, dummy_length // IN_CHANNELS, IN_CHANNELS])  # [batch, timestep, channels]
    x = torch.from_numpy(x)
    x_sl = [dummy_length]
    return x, x_sl


@pytest.fixture
def wavenet():
    net = WaveNet(
        layer_size=LAYER_SIZE,
        stack_size=STACK_SIZE,
        in_channels=IN_CHANNELS,
        res_channels=RES_CHANNELS,
        out_classes=OUT_CLASSES,
    )
    return net


def test_wavenet_output_size(wavenet):
    x, x_sl = generate_dummy(wavenet.receptive_field + 1)

    loss, output = wavenet(x, x_sl)

    # input size = receptive field size + 1
    # output size = input size - receptive field size
    #             = 1
    assert output.logits.shape == torch.Size([1, 256, 1])

    x, x_sl = generate_dummy(wavenet.receptive_field * 2)

    loss, output = wavenet(x, x_sl)

    # input size = receptive field size * 2
    # output size = input size - receptive field size
    #             = receptive field size
    assert output.logits.shape == torch.Size([1, 256, wavenet.receptive_field])


def test_wavenet_fail_with_short_input(wavenet):
    x, x_sl = generate_dummy(wavenet.receptive_field)

    with pytest.raises(InputSizeError):
        l, o = wavenet(x, x_sl)  # Should fail. Input size is too short.
