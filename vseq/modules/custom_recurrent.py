"""Module with custom recurrent cells.

Structure:
    LSTM:
    - Cell type
    - Layer type
    - Stack type
"""

import math
import warnings

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.jit as jit

from torch import Tensor
from torch.nn import Parameter


# class LSTMCell(jit.ScriptModule):
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = Parameter(torch.empty(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.uniform_(self.weight_hh, -stdv, stdv)
        init.uniform_(self.weight_ih, -stdv, stdv)
        init.zeros_(self.bias_hh)
        init.zeros_(self.bias_ih)

    @jit.export
    def in2hidden(self, input):
        return torch.matmul(input, self.weight_ih.t()) + self.bias_ih

    # @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input (Tensor): (batch, input_size)
            state (Tuple[Tensor, Tensor]): ((batch, hidden_size), (batch, hidden_size)))

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: ((batch, hidden_size), (batch, hidden_size))
        """
        hx, cx = state
        gates = input + torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy


# class LSTMCellLayerNorm(jit.ScriptModule):
class LSTMCellLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # No bias since the layernorms provide learnable biases
        self.weight_ih = Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        init.uniform_(self.weight_hh, -stdv, stdv)
        init.uniform_(self.weight_ih, -stdv, stdv)

    # @jit.script_method
    @jit.export
    def in2hidden(self, input):
        return self.layernorm_i(torch.matmul(input, self.weight_ih.t()))

    # @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        hx, cx = state
        igates = input
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, cy


# class LSTMLayerL2R(jit.ScriptModule):
class LSTMLayerL2R(nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        self.cell = cell(*cell_args, **cell_kwargs)

    # @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        input = self.cell.in2hidden(input)
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs += [state[0]]
        # outputs: (seq_len, batch, num_directions * hidden_size)
        # state:  tuple of (batch, hidden_size)
        return torch.stack(outputs), state


# class LSTMLayerR2L(jit.ScriptModule):
class LSTMLayerR2L(nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        self.cell = cell(*cell_args, **cell_kwargs)

    # @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        input = torch.flip(input, dims=[0])
        input = self.cell.in2hidden(input)
        inputs = input.unbind(0)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            state = self.cell(inputs[i], state)
            outputs += [state[0]]
        # outputs: (seq_len, batch, num_directions * hidden_size)
        # state:  tuple of (batch, hidden_size)
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
        # state:  tuple of (num_directions, batch, hidden_size)
        h_states = torch.stack([output_states[i][0] for i in range(len(output_states))])
        c_states = torch.stack([output_states[i][1] for i in range(len(output_states))])
        return torch.cat(outputs, -1), (h_states, c_states)


def init_stacked_lstm(num_layers, layer, first_layer_kwargs, other_layer_kwargs):
    layers = [layer(**first_layer_kwargs)] + [layer(**other_layer_kwargs) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


# class LSTMStack1(jit.ScriptModule):
class LSTMStack1(nn.Module):
    def __init__(self, num_layers, layer, first_layer_kwargs, other_layer_kwargs):
        super().__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_kwargs, other_layer_kwargs)

    # @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]

        # outputs: (seq_len, batch, num_directions * hidden_size)
        # state:  tuple of (num_layers * num_directions, batch, hidden_size)
        h_states = torch.stack([output_states[i][0] for i in range(len(output_states))])
        c_states = torch.stack([output_states[i][1] for i in range(len(output_states))])
        return output, (h_states, c_states)


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
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # List[List[LSTMState]]: The outer list is for layers,
        #                        inner list is for directions.
        output_states = jit.annotate(List[List[Tuple[Tensor, Tensor]]], [])
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]

        # outputs: (seq_len, batch, num_directions * hidden_size )
        # state:  tuple of (num_layers * num_directions, batch, hidden_size)
        h_states = torch.cat([output_states[i][0] for i in range(len(output_states))], dim=0)
        c_states = torch.cat([output_states[i][1] for i in range(len(output_states))], dim=0)
        return output, (h_states, c_states)


# class LSTMStackDropout(jit.ScriptModule):
class LSTMStackDropout(nn.Module):
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

    # @jit.script_method
    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        for i, rnn_layer in enumerate(self.layers):
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]

        # outputs: (seq_len, batch, num_directions * hidden_size)
        # state:  tuple of (num_layers * num_directions, batch, hidden_size)
        h_states = torch.stack([output_states[i][0] for i in range(len(output_states))])
        c_states = torch.stack([output_states[i][1] for i in range(len(output_states))])
        return output, (h_states, c_states)


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

        self.num_directions = 1 + int(self.bidirectional)

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

        lstm = stack_type(
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
        # self.lstm = torch.jit.script(lstm)
        self.lstm = lstm

    def forward(self, x: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None):
        # x: (seq_len, batch, input_size)
        # states: (num_layers * num_directions, batch, hidden_size)
        batch = x.size(1)

        if states is None:
            if self.bidirectional:
                states = [
                    [(torch.zeros(batch, self.hidden_size, device=x.device), torch.zeros(batch, self.hidden_size, device=x.device)) for _ in range(2)]
                    for _ in range(self.num_layers)
                ]
            else:
                states = [(torch.zeros(batch, self.hidden_size, device=x.device), torch.zeros(batch, self.hidden_size, device=x.device)) for _ in range(self.num_layers)]
        else:
            if self.bidirectional:
                h_state = states[0].view(self.num_layers, self.num_directions, batch, self.hidden_size)
                c_state = states[1].view(self.num_layers, self.num_directions, batch, self.hidden_size)

                states = [
                    [
                        (h_l_d, c_l_d)
                        for h_l_d, c_l_d in zip(h_l.unbind(0), c_l.unbind(0))
                    ]
                    for h_l, c_l in zip(h_state.unbind(0), c_state.unbind(0))
                ]
            else:
                h_states, c_states = states[0].unbind(0), states[1].unbind(0)
                states = [(h_states[i], c_states[i]) for i in range(self.num_layers)]

        return self.lstm(x, states)
