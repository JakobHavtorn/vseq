import torch
import torch.nn as nn
import torch.jit as jit
from torch.nn.modules.rnn import GRUCell

from vseq.utils.device import get_free_gpus
from vseq.utils.timing import timeit


seq_len = 50
batch_size = 64
in_features = 300
out_features = 256

device = get_free_gpus(1, require_unused=True)
# device = "cpu"
print(device)

gru_layer = nn.GRU(in_features, out_features).to(device)
gru_cell = nn.GRUCell(in_features, out_features).to(device)


def iterate_cell(inputs, ht):
    inputs = inputs.unbind(0)
    outputs = []
    for t in range(len(inputs)):
        ht = gru_cell(inputs[t], ht)
        outputs.append(ht)

    return torch.stack(outputs)


def iterate_layer(inputs, ht):
    inputs = [inp.unsqueeze(0) for inp in inputs.unbind(0)]
    outputs = []
    for t in range(len(inputs)):
        output, ht = gru_layer(inputs[t], ht)
        outputs.append(output)

    return torch.stack(outputs)


@jit.script
def iterate_cell_scripted(inputs: torch.Tensor, ht: torch.Tensor):
    inputs = inputs.unbind(0)
    outputs = []
    for t in range(len(inputs)):
        ht = gru_cell(inputs[t], ht)
        outputs.append(ht)

    return torch.stack(outputs)


@jit.script
def iterate_layer_scripted(inputs: torch.Tensor, ht: torch.Tensor):
    inputs = [inp.unsqueeze(0) for inp in inputs.unbind(0)]
    outputs = []
    for t in range(len(inputs)):
        output, ht = gru_layer(inputs[t], ht)
        outputs.append(output)

    return torch.stack(outputs)


def get_x():
    x = torch.randn(seq_len, batch_size, in_features).to(device)
    return x


def get_cell_state():
    state = torch.randn(batch_size, out_features, device=device)
    return state


def get_layer_state():
    state = torch.randn(1, batch_size, out_features, device=device)
    return state


print(f"forward() internal iter                         | GRU    : {timeit('gru_layer(get_x(), get_layer_state())', globals=globals())}")
print(f"forward() external iter                         | GRU    : {timeit('iterate_layer(get_x(), get_layer_state())', globals=globals())}")
print(f"forward() external iter                         | GRUCell: {timeit('iterate_cell(get_x(), get_cell_state())', globals=globals())}")
print(f"forward() external iter scripted                | GRU    : {timeit('iterate_layer_scripted(get_x(), get_layer_state())', globals=globals())}")
print(f"forward() external iter scripted                | GRUCell: {timeit('iterate_cell_scripted(get_x(), get_cell_state())', globals=globals())}")

print(f"forward()+backward() internal iter              | GRU    : {timeit('gru_layer(get_x(), get_layer_state())[0].sum().backward()', globals=globals())}")
print(f"forward()+backward() external iter              | GRU    : {timeit('iterate_layer(get_x(), get_layer_state()).sum().backward()', globals=globals())}")
print(f"forward()+backward() external iter              | GRUCell: {timeit('iterate_cell(get_x(), get_cell_state()).sum().backward()', globals=globals())}")
print(f"forward()+backward() external iter scripted     | GRU    : {timeit('iterate_layer_scripted(get_x(), get_layer_state()).sum().backward()', globals=globals())}")
print(f"forward()+backward() external iter scripted     | GRUCell: {timeit('iterate_cell_scripted(get_x(), get_cell_state()).sum().backward()', globals=globals())}")

