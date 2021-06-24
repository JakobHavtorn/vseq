"""
Test WaveNet model
"""

import numpy as np
import torch

import pytest

from vseq.models.wavenet.model import WaveNet, InputSizeError


N_LAYERS = 5  # 10 in paper
N_STACKS = 2  # 5 in paper
IN_CHANNELS = 1  # 256 in paper. quantized and one-hot input.
RES_CHANNELS = 512  # 512 in paper
OUT_CLASSES = 256


def generate_dummy(dummy_length):
    # x = np.arange(0, dummy_length, dtype=np.float32)
    x = np.random.uniform(low=-1, high=1, size=(1, dummy_length)).astype(np.float32)
    x = np.reshape(x, [1, dummy_length, IN_CHANNELS])  # [B, T, C]
    x = torch.from_numpy(x)
    x_sl = torch.LongTensor([dummy_length])
    return x, x_sl


@pytest.fixture
def wavenet():
    net = WaveNet(
        n_layers=N_LAYERS,
        n_stacks=N_STACKS,
        in_channels=IN_CHANNELS,
        res_channels=RES_CHANNELS,
        out_classes=OUT_CLASSES,
    )
    return net


def test_wavenet_output_size(wavenet):
    """Input and output sizes must be the same"""
    x, x_sl = generate_dummy(wavenet.receptive_field + 1)

    loss, metrics, output = wavenet(x, x_sl)

    # input size = receptive field size + 1
    # output size = receptive field size + 1
    assert output.logits.shape == torch.Size([1, 256, wavenet.receptive_field + 1])

    x, x_sl = generate_dummy(wavenet.receptive_field * 2)

    loss, metrics, output = wavenet(x, x_sl)

    # input size = receptive field size * 2
    # output size = receptive field size * 2
    assert output.logits.shape == torch.Size([1, 256, wavenet.receptive_field * 2])


def test_wavenet_fail_with_short_input(wavenet):
    """Wavenet must fail when given too short input"""
    x, x_sl = generate_dummy(wavenet.receptive_field)

    with pytest.raises(InputSizeError):
        loss, metrics, output = wavenet(x, x_sl)  # Should fail. Input size is too short.


def test_wavenet_causality_gradient_full(wavenet):
    """Test causality using gradient computed on full output"""
    x, x_sl = generate_dummy(wavenet.receptive_field + 1)  # (1, 65, 1)

    x.requires_grad_(True)
    x.retain_grad()

    loss, metrics, output = wavenet(x, x_sl)

    loss.backward()

    assert (x.grad[:, :-1, :] != 0).all(), "Gradient of loss wrt. full input is nonzero everywhere except last timestep"
    assert x.grad[:, -1, :] == 0, "Gradient of loss wrt. full input can never reach last timestep"


@pytest.mark.parametrize("slice_idx", [1, 5, 30, 64, 65])
def test_wavenet_causality_gradient_slice(wavenet, slice_idx):
    """Test causality using gradient computed on sliced output"""
    x, x_sl = generate_dummy(wavenet.receptive_field + 1)  # (1, 65, 1)

    x.requires_grad_(True)
    x.retain_grad()

    loss, metrics, output = wavenet(x, x_sl)

    output.log_prob[:, :slice_idx].sum().backward()

    assert (
        x.grad[:, : slice_idx - 1, :] != 0
    ).all(), "Gradient of loss wrt. sliced input must be nonzero before sliced timestep"
    assert (
        x.grad[:, slice_idx:, :] == 0
    ).all(), "Gradient of loss wrt. sliced input must be zero after sliced timestep"
