from collections import namedtuple

import pytest
import torch
import torch.nn as nn

from vseq.modules.custom_recurrent import LSTMCell, LSTMCellLayerNorm, LSTMLayerL2R, LSTMLayerR2L, LSTMLayerBidirectional, LSTMStack1, LSTMStack2, LSTM
from vseq.utils.timing import timeit, report_timings
from vseq.utils.device import get_free_gpus

LSTMState = namedtuple("LSTMState", ["hx", "cx"])


@pytest.mark.parametrize(
    "seq_len, batch, input_size, hidden_size",
    [
        [5, 2, 8, 4]
    ]
)
def test_script_lstm_script_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(seq_len, batch, input_size)
    state = LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
    lstm_script = LSTMLayerL2R(LSTMCell, input_size, hidden_size)
    out, out_state = lstm_script(inp, state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
    for lstm_param, custom_param in zip(lstm.all_weights[0], lstm_script.parameters()):
        assert lstm_param.shape == custom_param.shape
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


@pytest.mark.parametrize(
    "seq_len, batch, input_size, hidden_size, num_layers",
    [
        [5, 2, 8, 4, 3]
    ]
)
def test_script_stacked_lstm_script(seq_len, batch, input_size, hidden_size, num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    h_state = torch.randn(num_layers, batch, hidden_size)
    c_state = torch.randn(num_layers, batch, hidden_size)
    states = LSTMState(h_state, c_state)
    lstm_script = LSTM(input_size, hidden_size, num_layers)
    out, out_state = lstm_script(inp, states)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers)
    for layer in range(num_layers):
        custom_params = list(lstm_script.parameters())[4 * layer : 4 * (layer + 1)]
        for lstm_param, custom_param in zip(lstm.all_weights[layer], custom_params):
            assert lstm_param.shape == custom_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, states)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


@pytest.mark.parametrize(
    "seq_len, batch, input_size, hidden_size, num_layers",
    [
        [5, 2, 8, 4, 3]
    ]
)
def test_script_stacked_bidir_lstm_script(seq_len, batch, input_size, hidden_size, num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    h_state = torch.randn(2 * num_layers, batch, hidden_size)
    c_state = torch.randn(2 * num_layers, batch, hidden_size)
    states = LSTMState(h_state, c_state)
    lstm_script = LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True)
    out, out_state = lstm_script(inp, states)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
    for layer in range(num_layers):
        for direct in range(2):
            index = 2 * layer + direct
            custom_params = list(lstm_script.parameters())[4 * index : 4 * index + 4]
            for lstm_param, custom_param in zip(lstm.all_weights[index], custom_params):
                assert lstm_param.shape == custom_param.shape
                with torch.no_grad():
                    lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, states)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


@pytest.mark.parametrize(
    "seq_len, batch, input_size, hidden_size, num_layers",
    [
        [5, 2, 8, 4, 3]
    ]
)
def test_script_stacked_lstm_dropout(seq_len, batch, input_size, hidden_size, num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = LSTMState(torch.randn(num_layers, batch, hidden_size), torch.randn(num_layers, batch, hidden_size))
    lstm_script = LSTM(input_size, hidden_size, num_layers=num_layers, dropout=0.3)

    # just a smoke test
    out, out_state = lstm_script(inp, states)


@pytest.mark.parametrize(
    "seq_len, batch, input_size, hidden_size, num_layers",
    [
        [5, 2, 8, 4, 3]
    ]
)
def test_script_stacked_lnlstm(seq_len, batch, input_size, hidden_size, num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = LSTMState(torch.randn(num_layers, batch, hidden_size), torch.randn(num_layers, batch, hidden_size))
    lstm_script = LSTM(input_size, hidden_size, num_layers=num_layers)

    # just a smoke test
    out, out_state = lstm_script(inp, states)

@pytest.mark.parametrize(
    "seq_len, batch, input_size, hidden_size, num_layers",
    [
        [5, 2, 8, 4, 3]
    ]
)
def test_script_lstm_layer_norm_script_layer(seq_len, batch, input_size, hidden_size, num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = LSTMState(torch.randn(num_layers, batch, hidden_size), torch.randn(num_layers, batch, hidden_size))
    lstm_script = LSTM(input_size, hidden_size, layer_norm=True)

   # just a smoke test
    out, out_state = lstm_script(inp, states)


# @pytest.mark.parametrize(
#     "seq_len, batch, input_size, hidden_size, num_layers, device",
#     [
#         [5, 2, 8, 4, 3, "cpu"],
#         [100, 32, 128, 256, 3, "cpu"],
#         [5, 2, 8, 4, 3, get_free_gpus(1, require_unused=False)],
#         [100, 32, 128, 256, 3, get_free_gpus(1, require_unused=False)],
#     ]
# )
# def test_script_lstm_runtime_forward(seq_len, batch, input_size, hidden_size, num_layers, device):
#     inp = torch.randn(seq_len, batch, input_size, device=device)
#     lstm_script_state = [LSTMState(torch.randn(batch, hidden_size, device=device), torch.randn(batch, hidden_size, device=device)) for _ in range(num_layers)]
#     lstm_script = LSTM(input_size, hidden_size, num_layers=num_layers).to(device)

#     lstm_state = flatten_states(lstm_script_state)
#     lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers).to(device)

#     print()

#     timings_script = timeit("lstm(inp, state)", globals=dict(lstm=lstm_script, inp=inp, state=lstm_script_state))
#     report_timings(timings_script, "Scripted LSTM")

#     timings_native = timeit("lstm(inp, state)", globals=dict(lstm=lstm, inp=inp, state=lstm_state))
#     report_timings(timings_native, "Native LSTM")

#     print(f"Native is {1 + (timings_script.min - timings_native.min) / timings_native.min:.2f}X faster")

#     if device == "cpu":
#         assert timings_script.min < timings_native.min * 2
#     else:
#         assert timings_script.min < timings_native.min * 10


# @pytest.mark.parametrize(
#     "seq_len, batch, input_size, hidden_size, num_layers, device",
#     [
#         [5, 2, 8, 4, 3, "cpu"],
#         [100, 32, 128, 256, 3, "cpu"],
#         [5, 2, 8, 4, 3, get_free_gpus(1, require_unused=False)],
#         [100, 32, 128, 256, 3, get_free_gpus(1, require_unused=False)],
#     ]
# )
# def test_script_lstm_runtime_backward(seq_len, batch, input_size, hidden_size, num_layers, device):
#     inp = torch.randn(seq_len, batch, input_size, device=device)
#     lstm_script_state = [LSTMState(torch.randn(batch, hidden_size, device=device), torch.randn(batch, hidden_size, device=device)) for _ in range(num_layers)]
#     lstm_script = LSTM(input_size, hidden_size, num_layers=num_layers).to(device)

#     lstm_state = flatten_states(lstm_script_state)
#     lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers).to(device)

#     print()

#     timings_script = timeit("lstm(inp, state)[0].sum().backward()", globals=dict(lstm=lstm_script, inp=inp, state=lstm_script_state))
#     report_timings(timings_script, "Scripted LSTM")

#     timings_native = timeit("lstm(inp, state)[0].sum().backward()", globals=dict(lstm=lstm, inp=inp, state=lstm_state))
#     report_timings(timings_native, "Native LSTM")

#     print(f"Native is {1 + (timings_script.min - timings_native.min) / timings_native.min:.2f}X faster")

#     if device == "cpu":
#         assert timings_script.min < timings_native.min * 2
#     else:
#         assert timings_script.min < timings_native.min * 10
