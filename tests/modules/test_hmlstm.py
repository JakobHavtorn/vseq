import pytest
import torch

from vseq.modules.hmlstm import HMLSTMCell, HMLSTM
from vseq.models.hmrnn import HMLM


BATCH_SIZE = 64

NUM_EMBEDDINGS = 29
EMBEDDING_DIM = 100
LENGTH = 20

BELOW_SIZE = 128
HIDDEN_SIZE = 256
ABOVE_SIZE = 512


def generate_dummy_model_input(num_embeddings: int = NUM_EMBEDDINGS, length: int = LENGTH, n_examples: int = BATCH_SIZE):
    x = torch.randint(low=0, high=num_embeddings, size=(n_examples, length))
    x_sl = torch.ones(n_examples) * length
    return x, x_sl


def generate_dummy_lstm_input(input_size: int = EMBEDDING_DIM, length: int = LENGTH, n_examples: int = BATCH_SIZE):
    x = torch.randn(size=(n_examples, length, input_size))
    return x


def generate_dummy_cell_input(n_examples: int = BATCH_SIZE, hidden_size: int = HIDDEN_SIZE, below_size: int = BELOW_SIZE, above_size: int = ABOVE_SIZE):
    c = torch.randn(hidden_size, n_examples)
    h_bottom = torch.randn(below_size, n_examples)
    h = torch.randn(hidden_size, n_examples)
    h_top = torch.randn(above_size, n_examples)
    z = torch.randint(0, 2, size=(n_examples,))
    z_bottom = torch.randint(0, 2, size=(n_examples,))
    return c, h_bottom, h, h_top, z, z_bottom


@pytest.fixture
def hmlstm_middle_cell():
    cell = HMLSTMCell(
        hidden_size=HIDDEN_SIZE,
        below_size=BELOW_SIZE,
        above_size=ABOVE_SIZE
    )
    return cell


@pytest.fixture
def hmlstm_top_cell():
    cell = HMLSTMCell(
        hidden_size=ABOVE_SIZE,
        below_size=HIDDEN_SIZE,
        above_size=None
    )
    return cell


def test_hmlstm_cell_init(hmlstm_middle_cell, hmlstm_top_cell):
    assert not hmlstm_middle_cell.is_top_layer
    assert hmlstm_top_cell.is_top_layer


def test_hmlstm_cell_forward(hmlstm_middle_cell):
    # get input
    c, h_bottom, h, h_top, z, z_bottom = generate_dummy_cell_input(n_examples=64)

    # update only
    z = torch.zeros_like(z)
    z_bottom = torch.ones_like(z_bottom)
    h_new, c_new, z_new = hmlstm_middle_cell(c, h_bottom, h, h_top, z, z_bottom)
    assert not (c == c_new).all()
    assert not (h == h_new).all()

    # copy only
    z = torch.zeros_like(z)
    z_bottom = torch.zeros_like(z_bottom)
    h_new, c_new, z_new = hmlstm_middle_cell(c, h_bottom, h, h_top, z, z_bottom)
    assert (c == c_new).all()
    assert (h == h_new).all()

    # flush only
    z = torch.ones_like(z)
    h_new, c_new, z_new = hmlstm_middle_cell(c, h_bottom, h, h_top, z, z_bottom)
    assert not (c == c_new).all()
    assert not (h == h_new).all()


def test_hmlstm_init():
    hmlstm = HMLSTM(
        input_size=EMBEDDING_DIM,
        sizes=[128, 256]
    )
    assert hmlstm.sizes[-1] == 256


@pytest.mark.parametrize('sizes', 
    [
        [128, 256],
        [32, 64, 128, 256]
    ]
)
def test_hmlstm_forward(sizes):
    x = generate_dummy_lstm_input()
    
    hmlstm = HMLSTM(
        input_size=EMBEDDING_DIM,
        sizes=sizes
    )
    h, c, z, (h_out, c_out, z_out) = hmlstm(x)

    for i in range(hmlstm.num_layers):
        assert h_out[i].shape == (sizes[i], BATCH_SIZE)
        assert c_out[i].shape == (sizes[i], BATCH_SIZE)
        assert z_out[i].shape == (1, BATCH_SIZE)
        assert h[i].shape == (BATCH_SIZE, LENGTH, sizes[i])
        assert c[i].shape == (BATCH_SIZE, LENGTH, sizes[i])
        assert z[i].shape == (BATCH_SIZE, LENGTH, 1)


def test_hmrnn_init():
    x, x_sl = generate_dummy_model_input()
    hmlm = HMLM(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        sizes=[128, 256, 512],
    )

    out = hmlm(x, x_sl)
