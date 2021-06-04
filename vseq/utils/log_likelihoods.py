import math

import torch
import torch.nn.functional as F


@torch.jit.script
def gaussian_ll(x, mu, var, epsilon: float = 1e-6):
    if epsilon:
        var = var.clone()
        with torch.no_grad():
            var.clamp_(min=epsilon)
    return 0.5 * (torch.log(var) + (x - mu)**2 / var + math.log(2 * math.pi))


@torch.jit.script
def categorical_ll(y: torch.Tensor, logits: torch.Tensor):
    """Compute Categorical log-likelihood

    Args:
        y (torch.LongTensor): Target values in [1, C-1] of any shape.
        logits (torch.Tensor): Event log-probabilities (unnormalized) of same shape (y.shape, C)

    Returns:
        torch.Tensor: Log-probabilities
    """
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    y = y.long().unsqueeze(-1)
    y, logits = torch.broadcast_tensors(y, logits)
    y = y[..., :1]
    return logits.gather(-1, y).squeeze(-1)


def bernoulli_ll(y: torch.Tensor, logits: torch.Tensor):
    """Compute Bernoulli log-likelihood

    Args:
        x (torch.Tensor): Target values in {0, 1} of any shape.
        probs (torch.Tensor): Event log-probabilities (unnormalized) of same shape as `x`.
    """
    y, logits = torch.broadcast_tensors(y, logits)  # 2µs
    return -F.binary_cross_entropy_with_logits(logits, y, reduction='none')
