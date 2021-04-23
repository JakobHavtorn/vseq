from copy import deepcopy
from typing import List, Optional, Tuple, Union

import math

import torch
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import Module, Parameter

from .straight_through import BernoulliSTE, BinaryThresholdSTE


def hard_sigmoid(x, slope: float = 1):
    temp = torch.div(torch.add(torch.mul(x, slope), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output


class HMLSTMCell(Module):
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
        super(HMLSTMCell, self).__init__()

        self.below_size = below_size
        self.hidden_size = hidden_size
        self.above_size = above_size
        self.threshold_fn = threshold_fn
        self.is_top_layer = above_size is None

        self.W_hh = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.below_size))
        self.U_11 = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.hidden_size))
        if not self.is_top_layer:
            self.U_21 = Parameter(torch.FloatTensor(4 * self.hidden_size + 1, self.above_size))
        self.bias = Parameter(torch.FloatTensor(4 * self.hidden_size + 1))

        if threshold_fn == "threshold":
            self.threshold_ste_fn = BinaryThresholdSTE()
        elif threshold_fn == "bernoulli":
            self.threshold_ste_fn = BernoulliSTE()
        elif threshold_fn == "soft":
            self.threshold_ste_fn = None
        else:
            raise ValueError(f"Unknown threshold function `{threshold_fn}`")

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in self.parameters():
            par.data.uniform_(-stdv, stdv)

    def forward(self, c, h_bottom, h, h_top, z, z_bottom, a: float = 1):
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
        s_recur = torch.mm(self.W_hh, h_bottom)

        if self.is_top_layer:
            s_topdown = torch.zeros_like(s_recur)
        else:
            s_topdown = z * torch.mm(self.U_21, h_top)

        s_bottomup = z * torch.mm(self.U_11, h)

        f_slice = s_recur + s_topdown + s_bottomup + self.bias.unsqueeze(1)

        f = F.sigmoid(f_slice[0 : self.hidden_size, :])
        i = F.sigmoid(f_slice[self.hidden_size : self.hidden_size * 2, :])
        o = F.sigmoid(f_slice[self.hidden_size * 2 : self.hidden_size * 3, :])
        g = F.tanh(f_slice[self.hidden_size * 3 : self.hidden_size * 4, :])

        z_hat = hard_sigmoid(f_slice[self.hidden_size * 4 : self.hidden_size * 4 + 1, :], slope=a)

        one = torch.ones_like(f)
        c_new = z * (i * g) + (one - z) * (one - z_bottom) * c + (one - z) * z_bottom * (f * c + i * g)
        h_new = z * o * F.tanh(c_new) + (one - z) * (one - z_bottom) * h + (one - z) * z_bottom * o * F.tanh(c_new)
        z_new = self.threshold_ste_fn.apply(z_hat)

        return h_new, c_new, z_new

    def extra_repr(self) -> str:
        return f"below_size={self.below_size}, hidden_size={self.hidden_size}, above_size={self.above_size}"


class HMLSTM(Module):
    def __init__(self, input_size: int, sizes: Union[int, List[int]], num_layers: Optional[int] = None):
        super(HMLSTM, self).__init__()

        assert (
            (isinstance(sizes, list) and num_layers is None) or (isinstance(sizes, int) and num_layers is not None),
            "Must give `sizes` as list and not `num_layers` OR `sizes` as int along with a number of layers",
        )

        self.input_size = input_size
        self.sizes = sizes
        self.num_layers = len(sizes) if num_layers is None else num_layers

        sizes = [input_size, *sizes, None]
        cells = torch.nn.ModuleList()
        for i_layer in range(self.num_layers):
            cells.append(
                HMLSTMCell(below_size=sizes[i_layer], hidden_size=sizes[i_layer + 1], above_size=sizes[i_layer + 2])
            )
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
