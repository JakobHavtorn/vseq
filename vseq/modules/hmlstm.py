from typing import List, Optional, Union, Tuple

import math

import torch
import torch.nn as nn

from torch import sigmoid, tanh
from torch.nn import Parameter
from torchtyping import TensorType

from vseq.modules.straight_through import BernoulliSTE, BinaryThresholdSTE


def hard_sigmoid(x, slope: float = 1):
    temp = torch.div(torch.add(torch.mul(x, slope), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output


class HMLSTMCell(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        below_size: int,
        above_size: Optional[int] = None,
        threshold_fn: str = "threshold",
    ):
        """Hierarchical Multilevel LSTM cell (HM-LSTM) as described in [1].

        Parameters:
        W_10 is the state transition parameters from layer l-1 (bottom layer) to layer l
        U_11 is the state transition parameters from layer l (current layer) to layer l
        U_21 is the state transition parameters from layer l+1 (top layer) to layer l

        Internal representations are (D, B), i.e. batch last and dimension first. This saves a transpose.

        Args:
            hidden_size (int): Dimensionality of hidden layer (transition from `l` to `l`).
            above_size (int): Dimensionality of above layer (transition from `l+1` to `l`). Defaults to None.
            below_size (int): Dimensionality of below layer (transition frmo `l-1` to `l`).

        [1] Hierarchical Multiscale Recurrent Neural Networks. http://arxiv.org/abs/1609.01704
        """
        super().__init__()

        self.below_size = below_size
        self.hidden_size = hidden_size
        self.above_size = above_size
        self.threshold_fn = threshold_fn
        self.is_top_layer = above_size is None

        self.gates_size = 4 * self.hidden_size + 1

        self.W_10 = Parameter(torch.FloatTensor(self.gates_size, self.below_size))
        self.U_11 = Parameter(torch.FloatTensor(self.gates_size, self.hidden_size))
        if not self.is_top_layer:
            self.U_21 = Parameter(torch.FloatTensor(self.gates_size, self.above_size))
        self.bias = Parameter(torch.FloatTensor(self.gates_size))

        if threshold_fn == "threshold":
            self.threshold_ste = BinaryThresholdSTE(threshold=0.5)
        elif threshold_fn == "bernoulli":
            self.threshold_ste = BernoulliSTE()
        elif threshold_fn == "soft":
            self.threshold_ste = None
        else:
            raise ValueError(f"Unknown threshold function `{threshold_fn}`")

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in self.parameters():
            par.data.uniform_(-stdv, stdv)

    def forward(
        self,
        c: TensorType["D", "B"],
        h_below: TensorType["D", "B"],
        h: TensorType["D", "B"],
        h_above: TensorType["D", "B"],
        z: TensorType[1, "B"],
        z_below: TensorType[1, "B"],
        a: float = 1,
    ) -> Tuple[TensorType["D", "B"], TensorType["D", "B"], TensorType["D", 1]]:
        """Perform HM-LSTM forward pass.

        Update logic:
            if z == 1: (FLUSH)
                c_new = i * g
                h_new = o * F.tanh(c_new)
            elif z_below == 0: (COPY)
                c_new = c
                h_new = h
            else: (UPDATE)
                c_new = f * c + i * g
                h_new = o * F.tanh(c_new)

        Update logic alternative (but seemingly slower) implementation:
            c_new = torch.zeros_like(f)
            z = z.expand_as(f)
            flush = z == 1
            update = torch.logical_and(z == 0, z_below == 1)
            copy = torch.logical_and(z == 0, z_below == 0)
            c_new[:, flush] = (i[:, flush] * g[:, flush])
            c_new[:, update] = (f[:, update] * c[:, update] + i[:, update] * g[:, update])
            c_new[:, copy] = c[:, copy]

        Args:
            c (torch.Tensor): Previous time step cell state for this layer
            h_below (torch.Tensor): Current time step hidden state from layer below
            h (torch.Tensor): Previous time step hidden state for this layer
            h_above (torch.Tensor): Previous time step hidden state for layer above (if any, otherwise ignored)
            z (torch.Tensor): Previous time step boundary detector from this layer
            z_below (torch.Tensor): Current time step boundary detector from layer below
            a (float, optional): Slope of hard sigmoid activation for boundary detector. Defaults to 1.

        Returns:
            tuple: (h, c, z) for this time step
        """
        s_recurrent = torch.mm(self.U_11, h)

        if self.is_top_layer:
            s_topdown = torch.zeros_like(s_recurrent)
        else:
            s_topdown = z * torch.mm(self.U_21, h_above)

        s_bottomup = z_below * torch.mm(self.W_10, h_below)

        f_slice = s_recurrent + s_topdown + s_bottomup + self.bias.unsqueeze(1)

        forgetgate, ingate, outgate, cellgate = f_slice[:-1, :].chunk(chunks=4, dim=0)
        z_gate = f_slice[self.hidden_size * 4 : self.hidden_size * 4 + 1]

        f = sigmoid(forgetgate)
        i = sigmoid(ingate)
        o = sigmoid(outgate)
        g = tanh(cellgate)
        z_hat = hard_sigmoid(z_gate, slope=a)

        one = torch.ones_like(f)
        c_new = z * (i * g) + (one - z) * (one - z_below) * c + (one - z) * z_below * (f * c + i * g)
        h_new = (
            z * o * tanh(c_new) + (one - z) * (one - z_below) * h + (one - z) * z_below * o * tanh(c_new)
        )

        z_new = self.threshold_ste(z_hat)

        return h_new, c_new, z_new

    def extra_repr(self) -> str:
        return f"below_size={self.below_size}, hidden_size={self.hidden_size}, above_size={self.above_size}"


class LayerNormHMLSTMCell(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        below_size: int,
        above_size: Optional[int] = None,
        threshold_fn: str = "threshold",
        elementwise_affine: bool = True,
    ):
        """Hierarchical Multilevel LSTM cell (HM-LSTM) as described in [1].

        Parameters:
        W_10 is the state transition parameters from layer l-1 (bottom layer) to layer l
        U_11 is the state transition parameters from layer l (current layer) to layer l
        U_21 is the state transition parameters from layer l+1 (top layer) to layer l

        Internal representations are (D, B), i.e. batch last and dimension first. This saves a transpose.

        Args:
            hidden_size (int): Dimensionality of hidden layer (transition from `l` to `l`).
            above_size (int): Dimensionality of above layer (transition from `l+1` to `l`). Defaults to None.
            below_size (int): Dimensionality of below layer (transition frmo `l-1` to `l`).

        [1] Hierarchical Multiscale Recurrent Neural Networks. http://arxiv.org/abs/1609.01704
        """
        super().__init__()

        self.below_size = below_size
        self.hidden_size = hidden_size
        self.above_size = above_size
        self.threshold_fn = threshold_fn
        self.is_top_layer = above_size is None

        self.gates_size = 4 * self.hidden_size + 1

        self.W_10 = Parameter(torch.FloatTensor(self.gates_size, self.below_size))
        self.U_11 = Parameter(torch.FloatTensor(self.gates_size, self.hidden_size))
        if not self.is_top_layer:
            self.U_21 = Parameter(torch.FloatTensor(self.gates_size, self.above_size))

        self.ln_10 = nn.LayerNorm(self.gates_size, elementwise_affine=elementwise_affine)
        self.ln_11 = nn.LayerNorm(self.gates_size, elementwise_affine=elementwise_affine)
        self.ln_21 = nn.LayerNorm(self.gates_size, elementwise_affine=elementwise_affine)

        if threshold_fn == "threshold":
            self.threshold_ste = BinaryThresholdSTE()
        elif threshold_fn == "bernoulli":
            self.threshold_ste = BernoulliSTE()
        elif threshold_fn == "soft":
            self.threshold_ste = None
        else:
            raise ValueError(f"Unknown threshold function `{threshold_fn}`")

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in self.parameters():
            par.data.uniform_(-stdv, stdv)

    def forward(
        self,
        c: TensorType["D", "B"],
        h_below: TensorType["D", "B"],
        h: TensorType["D", "B"],
        h_above: TensorType["D", "B"],
        z: TensorType[1, "B"],
        z_below: TensorType[1, "B"],
        a: float = 1,
    ) -> Tuple[TensorType["D", "B"], TensorType["D", "B"], TensorType["D", 1]]:
        """Perform HM-LSTM forward pass.

        Update logic:
            if z == 1: (FLUSH)
                c_new = i * g
                h_new = o * F.tanh(c_new)
            elif z_below == 0: (COPY)
                c_new = c
                h_new = h
            else: (UPDATE)
                c_new = f * c + i * g
                h_new = o * F.tanh(c_new)

        Update logic alternative (but seemingly slower) implementation:
            c_new = torch.zeros_like(f)
            z = z.expand_as(f)
            flush = z == 1
            update = torch.logical_and(z == 0, z_below == 1)
            copy = torch.logical_and(z == 0, z_below == 0)
            c_new[:, flush] = (i[:, flush] * g[:, flush])
            c_new[:, update] = (f[:, update] * c[:, update] + i[:, update] * g[:, update])
            c_new[:, copy] = c[:, copy]

        Args:
            c (torch.Tensor): Previous time step cell state for this layer
            h_below (torch.Tensor): Current time step hidden state from layer below
            h (torch.Tensor): Previous time step hidden state for this layer
            h_above (torch.Tensor): Previous time step hidden state for layer above (if any, otherwise ignored)
            z (torch.Tensor): Previous time step boundary detector from this layer
            z_below (torch.Tensor): Current time step boundary detector from layer below
            a (float, optional): Slope of hard sigmoid activation for boundary detector. Defaults to 1.

        Returns:
            tuple: (h, c, z) for this time step
        """
        s_recurrent = self.ln_11(torch.mm(self.U_11, h).T).T

        if self.is_top_layer:
            s_topdown = torch.zeros_like(s_recurrent)
        else:
            s_topdown = z * self.ln_21(torch.mm(self.U_21, h_above).T).T

        s_bottomup = z_below * self.ln_10(torch.mm(self.W_10, h_below).T).T

        f_slice = s_recurrent + s_topdown + s_bottomup

        forgetgate, ingate, outgate, cellgate = f_slice[:-1, :].chunk(chunks=4, dim=0)
        z_gate = f_slice[self.hidden_size * 4 : self.hidden_size * 4 + 1]

        f = sigmoid(forgetgate)
        i = sigmoid(ingate)
        o = sigmoid(outgate)
        g = tanh(cellgate)
        z_hat = hard_sigmoid(z_gate, slope=a)

        one = torch.ones_like(f)
        c_new = z * (i * g) + (one - z) * (one - z_below) * c + (one - z) * z_below * (f * c + i * g)
        h_new = (z * o * tanh(c_new) + (one - z) * (one - z_below) * h + (one - z) * z_below * o * tanh(c_new))

        z_new = self.threshold_ste(z_hat)

        return h_new, c_new, z_new

    def extra_repr(self) -> str:
        return f"below_size={self.below_size}, hidden_size={self.hidden_size}, above_size={self.above_size}"


class HMLSTM(nn.Module):
    def __init__(
        self, input_size: int, sizes: Union[int, List[int]], num_layers: Optional[int] = None, layer_norm: bool = False
    ):
        super().__init__()

        assert (
            (isinstance(sizes, list) and num_layers is None) or (isinstance(sizes, int) and num_layers is not None),
            "Must give `sizes` as list and not `num_layers` OR `sizes` as int along with a number of layers",
        )

        self.input_size = input_size
        self.sizes = sizes
        self.num_layers = len(sizes) if num_layers is None else num_layers
        self.layer_norm = layer_norm

        cell = LayerNormHMLSTMCell if layer_norm else HMLSTMCell

        sizes = [input_size, *sizes, None]
        cells = torch.nn.ModuleList()
        for l in range(self.num_layers):
            cells.append(cell(below_size=sizes[l], hidden_size=sizes[l + 1], above_size=sizes[l + 2]))
        self.cells = cells

    def forward(
        self,
        x: TensorType["B", "T", "D"],
        h_init: Optional[List[TensorType["B", "T", "D"]]] = None,
        c_init: Optional[List[TensorType["B", "T", "D"]]] = None,
        z_init: Optional[List[TensorType["B", "T", 1]]] = None,
        a: float = 1,
    ):
        # x.size = (B, T, D)
        time_steps = x.size(1)
        batch_size = x.size(0)
        device = x.device

        if h_init is None:
            h = [[torch.zeros(self.sizes[l], batch_size, device=device)] for l in range(self.num_layers)]
        else:
            h = [[h.permute(2, 1, 0)] for h in h_init]  # (B, T, D) to (D, T, B)

        if c_init is None:
            c = [[torch.zeros(self.sizes[l], batch_size, device=device)] for l in range(self.num_layers)]
        else:
            c = [[c.permute(2, 1, 0)] for c in c_init]  # (B, T, D) to (D, T, B)

        if z_init is None:
            z = [[torch.zeros(1, batch_size, device=device)] for l in range(self.num_layers)]
        else:
            z = [[z.permute(2, 1, 0)] for z in z_init]  # (B, T, D) to (D, T, B)

        # create a fictive top layer that gives `h=None` for all time steps.
        # used as `h_above` input for the actual top layer.
        # h.append([torch.zeros(1, batch_size, device=device)] * time_steps)
        h.append([None] * time_steps)

        # z_below for layer 0
        z_one = torch.ones(1, batch_size, device=device)
        x = x.permute(2, 1, 0)  # (B, T, D) to (D, T, B)

        for t in range(time_steps):
            # input layer
            l = 0
            h_tl, c_tl, z_tl = self.cells[l](
                c=c[l][t],
                h=h[l][t],
                h_below=x[:, t, :],
                h_above=h[l + 1][t],
                z=z[l][t],
                z_below=z_one,  # Never skip input by copying (forces UPDATE or FLUSH)
                a=a,
            )
            h[l].append(h_tl)
            c[l].append(c_tl)
            z[l].append(z_tl)

            # additional layers
            for l in range(1, self.num_layers):
                h_tl, c_tl, z_tl = self.cells[l](
                    c=c[l][t],
                    h=h[l][t],
                    h_below=h[l - 1][t + 1],
                    h_above=h[l + 1][t],
                    z=z[l][t],
                    z_below=z[l - 1][t + 1],
                    a=a,
                )
                h[l].append(h_tl)
                c[l].append(c_tl)
                z[l].append(z_tl)

        h = [hl[1:] for hl in h[:-1]]  # Remove initial value and fictive layer
        c = [cl[1:] for cl in c]  # Remove initial value
        z = [zl[1:] for zl in z]  # Remove initial value

        # collect final timestep per layer
        h_out = [h[l][-1] for l in range(self.num_layers)]
        c_out = [c[l][-1] for l in range(self.num_layers)]
        z_out = [z[l][-1] for l in range(self.num_layers)]
        return (
            [torch.stack(hl, dim=1).permute(2, 1, 0) for hl in h],  # (B, T, D)
            [torch.stack(cl, dim=1).permute(2, 1, 0) for cl in c],  # (B, T, D)
            [torch.stack(zl, dim=1).permute(2, 1, 0) for zl in z],  # (B, T, 1)
            (h_out, c_out, z_out),  # (B, D), (B, D), (B, 1)
        )

    def realized_operations(
        self, z: List[TensorType["B", "T", 1]], x_sl: TensorType["B", int], seq_mask: TensorType["B", "T", bool]
    ):
        """Return the boolean masks incidating where the different operations took place and compute the clockrates"""
        update_ops, copy_ops, flush_ops = [], [], []
        update_rates, copy_rates, flush_rates = [], [], []
        x_sl = x_sl.to(z[0].device)
        for l, z_l in enumerate(z):
            z_below = torch.ones_like(z[0]) if l == 0 else z[l - 1]

            update_ops.append(((z_l[:, :-1, :] == 0) * (z_below[:, 1:, :] == 1)).squeeze())
            copy_ops.append(((z_l[:, :-1, :] == 0) * (z_below[:, 1:, :] == 0)).squeeze())
            flush_ops.append((z_l[:, :-1, :] == 1).squeeze())

            update_rates.append((update_ops[l] * seq_mask[:, 1:]).sum(1) / x_sl)
            copy_rates.append((copy_ops[l] * seq_mask[:, 1:]).sum(1) / x_sl)
            flush_rates.append((flush_ops[l] * seq_mask[:, 1:]).sum(1) / x_sl)

        return update_ops, copy_ops, flush_ops, update_rates, copy_rates, flush_rates


if __name__ == "__main__":
    import timeit
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch = 32
    hidden_size = 256
    below_size = 128
    above_size = 512

    cell = HMLSTMCell(
        hidden_size=hidden_size,
        below_size=below_size,
        above_size=above_size,
        threshold_fn="threshold",
    ).to(device)

    lncell = HMLSTMCell(
        hidden_size=hidden_size,
        below_size=below_size,
        above_size=above_size,
        threshold_fn="threshold",
    ).to(device)

    lstm = nn.LSTMCell(
        input_size=hidden_size,
        hidden_size=hidden_size,
    ).to(device)

    c = torch.randn(hidden_size, batch).to(device)
    h = torch.randn(hidden_size, batch).to(device)
    z = torch.randint(low=0, high=2, size=(1, batch)).to(device)
    h_above = torch.randn(above_size, batch).to(device)
    h_below = torch.randn(below_size, batch).to(device)
    z_below = torch.randint(low=0, high=2, size=(1, batch)).to(device)

    cell(c, h_below, h, h_above, z, z_below)

    timer = timeit.Timer("lstm(h, (h, c))", globals=dict(lstm=lstm, h=h.T, c=c.T))
    number, time_taken = timer.autorange()
    timings = timer.repeat(repeat=10, number=number)
    print(f"LSTM: {number=:d}, {min(timings):.3e} +- {np.std(timings):.3e} s")

    timer = timeit.Timer("cell(c, h_below, h, h_above, z, z_below)", globals=globals())
    number, time_taken = timer.autorange()
    timings = timer.repeat(repeat=10, number=number)
    print(f"HM-LSTM: {number=:d}, {min(timings):.3e} +- {np.std(timings):.3e} s")

    timer = timeit.Timer("lncell(c, h_below, h, h_above, z, z_below)", globals=globals())
    number, time_taken = timer.autorange()
    timings = timer.repeat(repeat=10, number=number)
    print(f"HM-LSTM LayerNorm: {number=:d}, {min(timings):.3e} +- {np.std(timings):.3e} s")
