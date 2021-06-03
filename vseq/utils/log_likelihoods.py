import math

import torch


@torch.jit.script
def gaussian_ll(x, mu, var, epsilon: float = 1e-6):
    if epsilon:
        var = var.clone()
        with torch.no_grad():
            var.clamp_(min=epsilon)

    return 0.5 * (torch.log(var) + (x - mu)**2 / var + math.log(2 * math.pi))
