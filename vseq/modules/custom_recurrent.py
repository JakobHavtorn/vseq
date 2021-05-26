"""Module with custom recurrent cells.

Structure:
    LSTM:
    - Cell type
    - Layer type
    - Stack type
"""

import warnings

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.jit as jit

from torch import Tensor
from torch.nn import Parameter


# class LSTMCell(jit.ScriptModule):
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    # @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


# class LSTMCellLayerNorm(jit.ScriptModule):
class LSTMCellLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # No bias since the layernorms provide learnable biases
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)

    # @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


# class LSTMLayerL2R(jit.ScriptModule):
class LSTMLayerL2R(nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        self.cell = cell(*cell_args, **cell_kwargs)

    # @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        # outputs: (seq_len, batch, num_directions * hidden_size)
        # state:  tuple of (num_layers * num_directions, batch, hidden_size)
        return torch.stack(outputs), state


# class LSTMLayerR2L(jit.ScriptModule):
class LSTMLayerR2L(nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        self.cell = cell(*cell_args, **cell_kwargs)

    # @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        print(input.shape)
        inputs = torch.flip(input, dims=[0]).unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        print(len(inputs))
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
            print(inputs[i].shape, out.shape, len(outputs))
        # outputs: (seq_len, batch, num_directions * hidden_size)
        # state:  tuple of (num_layers * num_directions, batch, hidden_size)
        return torch.flip(torch.stack(outputs), dims=[0]), state


# class LSTMLayerBidirectional(jit.ScriptModule):
class LSTMLayerBidirectional(nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        self.directions = nn.ModuleList(
            [
                LSTMLayerL2R(cell, *cell_args, **cell_kwargs),
                LSTMLayerR2L(cell, *cell_args, **cell_kwargs),
            ]
        )

    # @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        for i, direction in enumerate(self.directions):
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
        # outputs: (seq_len, batch, num_directions * hidden_size)
        # state:  tuple of (num_layers * num_directions, batch, hidden_size)
        return torch.cat(outputs, -1), output_states


def init_stacked_lstm(num_layers, layer, first_layer_kwargs, other_layer_kwargs):
    layers = [layer(**first_layer_kwargs)] + [layer(**other_layer_kwargs) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


# class LSTMStack1(jit.ScriptModule):
class LSTMStack1(nn.Module):
    def __init__(self, num_layers, layer, first_layer_kwargs, other_layer_kwargs):
        super().__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_kwargs, other_layer_kwargs)

    # @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
        return output, output_states


# Differs from LSTMStack1 in that its forward method takes
# List[List[Tuple[Tensor,Tensor]]]. It would be nice to subclass LSTMStack1
# except we don't support overriding script methods.
# https://github.com/pytorch/pytorch/issues/10733
# class LSTMStack2(jit.ScriptModule):
class LSTMStack2(nn.Module):
    def __init__(self, num_layers, layer, first_layer_kwargs, other_layer_kwargs):
        super().__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_kwargs, other_layer_kwargs)

    # @jit.script_method
    def forward(
        self, input: Tensor, states: List[List[Tuple[Tensor, Tensor]]]
    ) -> Tuple[Tensor, List[List[Tuple[Tensor, Tensor]]]]:
        # List[List[LSTMState]]: The outer list is for layers,
        #                        inner list is for directions.
        output_states = jit.annotate(List[List[Tuple[Tensor, Tensor]]], [])
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
        return output, output_states


class LSTMStackDropout(jit.ScriptModule):
    def __init__(self, num_layers, layer, first_layer_kwargs, other_layer_kwargs, dropout):
        """Introduces a Dropout layer on the outputs of each LSTM layer except the last layer.

        Args:
            num_layers ([type]): [description]
            layer ([type]): [description]
            first_layer_kwargs ([type]): [description]
            other_layer_kwargs ([type]): [description]
        """
        super().__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_kwargs, other_layer_kwargs)

        self.num_layers = num_layers

        if num_layers == 1:
            warnings.warn(
                "dropout lstm adds dropout layers after all but last "
                "recurrent layer, it expects num_layers greater than "
                "1, but got num_layers = 1"
            )

        self.dropout_layer = nn.Dropout(dropout)

    @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
        return output, output_states


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        bidirectional: bool = False,
        dropout: Optional[float] = None,
        layer_norm: bool = False,
        batch_first: bool = False,
    ):
        """

        Args:
            input_size: The number of expected features in the input `x`
            hidden_size: The number of features in the hidden state `h`
            num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
                would mean stacking two LSTMs together to form a `stacked LSTM`,
                with the second LSTM taking in outputs of the first LSTM and
                computing the final results. Default: 1
            bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
                Default: ``True``
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``False``
            dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
                LSTM layer except the last layer, with dropout probability equal to
                :attr:`dropout`. Default: 0
            bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
            proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0
        """
        assert not batch_first
        assert bool(dropout) * bidirectional == False, "Bidrectional with dropout not implemented"
        assert bias

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_first = batch_first

        if bidirectional:
            stack_type = LSTMStack2
            layer_type = LSTMLayerBidirectional
            n_dirs = 2
            stack_kwargs = dict(num_layers=num_layers, layer=layer_type)
        elif dropout:
            stack_type = LSTMStackDropout
            layer_type = LSTMLayerL2R
            n_dirs = 1
            stack_kwargs = dict(num_layers=num_layers, layer=layer_type, dropout=dropout)
        else:
            stack_type = LSTMStack1
            layer_type = LSTMLayerL2R
            n_dirs = 1
            stack_kwargs = dict(num_layers=num_layers, layer=layer_type)

        if layer_norm:
            cell_type = LSTMCellLayerNorm
        else:
            cell_type = LSTMCell

        self.lstm = stack_type(
            **stack_kwargs,
            first_layer_kwargs=dict(
                cell=cell_type,
                input_size=input_size,
                hidden_size=hidden_size,
            ),
            other_layer_kwargs=dict(
                cell=cell_type,
                input_size=hidden_size * n_dirs,
                hidden_size=hidden_size,
            ),
        )

    def forward(self, x: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None):
        # x: (seq_len, batch, input_size)
        batch = x.size(1)

        if states is None:
            if self.bidirectional:
                states = [
                    [(torch.zeros(batch, self.hidden_size), torch.zeros(batch, self.hidden_size)) for _ in range(2)]
                    for _ in range(self.num_layers)
                ]
            else:
                states = [(torch.zeros(batch, self.hidden_size), torch.zeros(batch, self.hidden_size)) for _ in range(self.num_layers)]

        return self.lstm(x, states)
