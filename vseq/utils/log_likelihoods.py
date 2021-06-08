import math

import torch
import torch.nn.functional as F


def gaussian_ll(x, mu, var, epsilon: float = 1e-6):
    """Compute Gaussian log-likelihood

    Clamps the variance at `epsilon` for numerical stability. This does not affect the gradient.

    Args:
        x (torch.Tensor): Targets
        mu (torch.Tensor): Mean of the Gaussian
        var (torch.Tensor): Variance of the Gaussian
        epsilon (float, optional): Minimum variance for numerical stability. Defaults to 1e-6.

    Returns:
        torch.Tensor: Log-probabilities
    """
    if epsilon:
        var = var.clone()
        with torch.no_grad():
            var.clamp_(min=epsilon)
    return - 0.5 * (math.log(2 * math.pi) + torch.log(var) + (x - mu)**2 / var)


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

    Returns:
        torch.Tensor: Log-probabilities
    """
    y, logits = torch.broadcast_tensors(y, logits)  # 2µs
    return -F.binary_cross_entropy_with_logits(logits, y, reduction='none')
