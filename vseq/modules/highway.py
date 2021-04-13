import math
from typing import Optional

import torch.nn as nn


class HighwayBlockDense(nn.Module):
    def __init__(self, n_features, t_bias_mean: float = -1):
        """Single Highway network block as in [1].

        Performs the operation:
            y = H(x, WH) * T(x, WT) + x * C(x, WC)
        where
            H(x, WH) = Tanh(Linear(x, WH))
            T(x, WT) = Sigmoid(Linear(x, WT))
            C(x, WT) = 1 - T(x, WT)

        We initialize the bias of the linear layer of T(x, WT) to have mean value `t_bias_mean` to bias the block to
        initially "carry" the input `x` to the output and ignore the transform.

        Args:
            n_features ([type]): Number of features in input `x`, H(x, WH) and T(x, WT).
            t_bias_mean (float, optional): Initial mean value of the bias of T(x, WT). Defaults to -1.

        [1] Highway Networks https://arxiv.org/pdf/1505.00387.pdf
        """
        super().__init__()
        self.n_features = n_features
        self.t_bias_mean = -1 if t_bias_mean is None else t_bias_mean

        self.h = nn.Sequential(nn.Linear(n_features, n_features), nn.Tanh())
        self.t = nn.Sequential(nn.Linear(n_features, n_features), nn.Sigmoid())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.t[0].weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.t[0].bias, -bound, bound) + self.t_bias_mean

    def forward(self, x):
        h = self.h(x)
        t = self.t(x)
        c = 1 - t
        return h * t + x * c


class HighwayStackDense(nn.Module):
    def __init__(self, n_features: int, n_blocks: int, t_bias_mean: Optional[float] = None):
        """A stack of HighwayBlockDense layers"""
        super().__init__()
        self.n_features = n_features
        self.n_blocks = n_blocks

        self.blocks = [HighwayBlockDense(n_features=n_features, t_bias_mean=t_bias_mean) for _ in range(n_blocks)]
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)
