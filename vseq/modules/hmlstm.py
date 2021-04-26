from typing import List, Optional, Union, Tuple

import math

import torch
import torch.nn as nn
import torch.jit as jit

from torch import Tensor
from torch.nn import Module, Parameter

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
        W_hh is the state transition parameters from layer l-1 (bottom layer) to layer l
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

        self.W_hh = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.below_size))
        self.U_11 = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.hidden_size))
        if not self.is_top_layer:
            self.U_21 = Parameter(
                torch.FloatTensor(4 * self.hidden_size + 1, self.above_size if self.above_size else 1)
            )
        self.bias = Parameter(torch.FloatTensor(4 * self.hidden_size + 1))

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
        self.bias.data[-1] = 2.0 / math.sqrt(self.hidden_size)  # Bias towards UPDATE behaviour

    def forward(
        self, c: Tensor, h_bottom: Tensor, h: Tensor, h_top: Tensor, z: Tensor, z_bottom: Tensor, a: float = 1
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform HM-LSTM forward pass.

        Update logic:
            if z == 1: (FLUSH)
                c_new = i * g
                h_new = o * F.tanh(c_new)
            elif z_bottom == 0: (COPY)
                c_new = c
                h_new = h
            else: (UPDATE)
                c_new = f * c + i * g
                h_new = o * F.tanh(c_new)

        Update logic alternative (but seemingly slower) implementation:
            c_new = torch.zeros_like(f)
            z = z.expand_as(f)
            flush = z == 1
            update = torch.logical_and(z == 0, z_bottom == 1)
            copy = torch.logical_and(z == 0, z_bottom == 0)
            c_new[:, flush] = (i[:, flush] * g[:, flush])
            c_new[:, update] = (f[:, update] * c[:, update] + i[:, update] * g[:, update])
            c_new[:, copy] = c[:, copy]

        Args:
            c (torch.Tensor): Cell state of this cell at previous timestep
            h_bottom (torch.Tensor): Hidden state of below cell at current timestep
            h (torch.Tensor): Hidden state of this cell at previous timestep
            h_top (torch.Tensor): Hidden state of above cell at previous timestep
            z (torch.Tensor): Boundary detector of this cell at previous timestep
            z_bottom (torch.Tensor): Boundary detector of below cell at current timestep
            a (float, optional): Slop of the hard sigmoid activation of boundary detector. Defaults to 1.

        Returns:
            tuple: h_new, c_new, z_new
        """
        s_recur = torch.mm(self.W_hh, h_bottom)

        if self.is_top_layer:
            s_topdown = torch.zeros_like(s_recur)
        else:
            s_topdown = z * torch.mm(self.U_21, h_top)

        s_bottomup = z * torch.mm(self.U_11, h)

        f_slice = s_recur + s_topdown + s_bottomup + self.bias.unsqueeze(1)

        forgetgate, ingate, outgate, cellgate = f_slice[:-1, :].chunk(chunks=4, dim=0)
        z_gate = f_slice[self.hidden_size * 4 : self.hidden_size * 4 + 1]

        f = torch.sigmoid(forgetgate)
        i = torch.sigmoid(ingate)
        o = torch.sigmoid(outgate)
        g = torch.tanh(cellgate)
        z_hat = hard_sigmoid(z_gate, slope=a)

        one = torch.ones_like(f)
        c_new = z * (i * g) + (one - z) * (one - z_bottom) * c + (one - z) * z_bottom * (f * c + i * g)
        h_new = (
            z * o * torch.tanh(c_new) + (one - z) * (one - z_bottom) * h + (one - z) * z_bottom * o * torch.tanh(c_new)
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
    ):
        """Hierarchical Multilevel LSTM cell (HM-LSTM) as described in [1].

        Parameters:
        W_hh is the state transition parameters from layer l-1 (bottom layer) to layer l
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

        self.W_hh = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.below_size))
        self.U_11 = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.hidden_size))
        if not self.is_top_layer:
            self.U_21 = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.above_size))

        self.ln_hh = nn.LayerNorm(4 * self.hidden_size + 1)
        self.ln_11 = nn.LayerNorm(4 * self.hidden_size + 1)
        self.ln_12 = nn.LayerNorm(4 * self.hidden_size + 1)

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

    def forward(self, c: Tensor, h_bottom: Tensor, h: Tensor, h_top: Tensor, z: Tensor, z_bottom: Tensor, a: float = 1):
        """Perform HM-LSTM forward pass.

        Update logic:
            if z == 1: (FLUSH)
                c_new = i * g
                h_new = o * F.tanh(c_new)
            elif z_bottom == 0: (COPY)
                c_new = c
                h_new = h
            else: (UPDATE)
                c_new = f * c + i * g
                h_new = o * F.tanh(c_new)

        Update logic alternative (but seemingly slower) implementation:
            c_new = torch.zeros_like(f)
            z = z.expand_as(f)
            flush = z == 1
            update = torch.logical_and(z == 0, z_bottom == 1)
            copy = torch.logical_and(z == 0, z_bottom == 0)
            c_new[:, flush] = (i[:, flush] * g[:, flush])
            c_new[:, update] = (f[:, update] * c[:, update] + i[:, update] * g[:, update])
            c_new[:, copy] = c[:, copy]

        Args:
            c ([type]): [description]
            h_bottom ([type]): [description]
            h ([type]): [description]
            h_top ([type]): [description]
            z ([type]): [description]
            z_bottom ([type]): [description]
            a (float, optional): [description]. Defaults to 1.

        Returns:
            tuple: h_new, c_new, z_new
        """
        s_recur = self.ln_hh(torch.mm(self.W_hh, h_bottom).T).T

        if self.is_top_layer:
            s_topdown = torch.zeros_like(s_recur)
        else:
            s_topdown = z * self.ln_12(torch.mm(self.U_21, h_top).T).T

        s_bottomup = z * self.ln_11(torch.mm(self.U_11, h).T).T

        f_slice = s_recur + s_topdown + s_bottomup + 2.0 / math.sqrt(self.hidden_size)

        forgetgate, ingate, outgate, cellgate = f_slice[:-1, :].chunk(chunks=4, dim=0)
        z_gate = f_slice[self.hidden_size * 4 : self.hidden_size * 4 + 1]

        f = torch.sigmoid(forgetgate)
        i = torch.sigmoid(ingate)
        o = torch.sigmoid(outgate)
        g = torch.tanh(cellgate)
        z_hat = hard_sigmoid(z_gate, slope=a)

        one = torch.ones_like(f)
        c_new = z * (i * g) + (one - z) * (one - z_bottom) * c + (one - z) * z_bottom * (f * c + i * g)
        h_new = (
            z * o * torch.tanh(c_new) + (one - z) * (one - z_bottom) * h + (one - z) * z_bottom * o * torch.tanh(c_new)
        )

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
        x: torch.Tensor,
        h_init: Optional[List[torch.Tensor]] = None,
        c_init: Optional[List[torch.Tensor]] = None,
        z_init: Optional[List[torch.Tensor]] = None,
    ):
        # x.size = (B, T, D)
        time_steps = x.size(1)
        batch_size = x.size(0)
        device = x.device

        if h_init is None:
            h = [[torch.zeros(self.sizes[l], batch_size, device=device)] for l in range(self.num_layers)]
        else:
            h = [[h] for h in h_init]

        if c_init is None:
            c = [[torch.zeros(self.sizes[l], batch_size, device=device)] for l in range(self.num_layers)]
        else:
            c = [[c] for c in c_init]

        if z_init is None:
            z = [[torch.zeros(1, batch_size, device=device)] for l in range(self.num_layers)]
        else:
            z = [[z] for z in z_init]

        # create a fictive top layer that gives `h=None` for all time steps.
        # used as `h_top` input for the actual top layer.
        # h.append([torch.zeros(1, batch_size, device=device)] * time_steps)
        h.append([None] * time_steps)

        # z_bottom for layer 0
        z_one = torch.ones(1, batch_size, device=device)

        for t in range(time_steps):
            # input layer
            l = 0
            h_tl, c_tl, z_tl = self.cells[l](
                c=c[l][t],
                h=h[l][t],
                h_bottom=x[:, t, :].t(),
                h_top=h[l + 1][t],
                z=z[l][t],
                z_bottom=z_one,  # Never skip input by copying (forces UPDATE or FLUSH)
            )
            h[l].append(h_tl)
            c[l].append(c_tl)
            z[l].append(z_tl)

            # additional layers
            for l in range(1, self.num_layers):
                h_tl, c_tl, z_tl = self.cells[l](
                    c=c[l][t],
                    h=h[l][t],
                    h_bottom=h[l - 1][t + 1],
                    h_top=h[l + 1][t],
                    z=z[l][t],
                    z_bottom=z[l - 1][t + 1],
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
            [torch.stack(cl, dim=1).permute(2, 1, 0) for cl in c],
            [torch.stack(zl, dim=1).permute(2, 1, 0) for zl in z],
            (h_out, c_out, z_out),
        )


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
    h_top = torch.randn(above_size, batch).to(device)
    h_bottom = torch.randn(below_size, batch).to(device)
    z_bottom = torch.randint(low=0, high=2, size=(1, batch)).to(device)

    cell(c, h_bottom, h, h_top, z, z_bottom)

    timer = timeit.Timer("lstm(h, (h, c))", globals=dict(lstm=lstm, h=h.T, c=c.T))
    number, time_taken = timer.autorange()
    timings = timer.repeat(repeat=10, number=number)
    print(f"LSTM: {number=:d}, {min(timings):.3e} +- {np.std(timings):.3e} s")

    timer = timeit.Timer("cell(c, h_bottom, h, h_top, z, z_bottom)", globals=globals())
    number, time_taken = timer.autorange()
    timings = timer.repeat(repeat=10, number=number)
    print(f"HM-LSTM: {number=:d}, {min(timings):.3e} +- {np.std(timings):.3e} s")

    timer = timeit.Timer("lncell(c, h_bottom, h, h_top, z, z_bottom)", globals=globals())
    number, time_taken = timer.autorange()
    timings = timer.repeat(repeat=10, number=number)
    print(f"HM-LSTM LayerNorm: {number=:d}, {min(timings):.3e} +- {np.std(timings):.3e} s")
