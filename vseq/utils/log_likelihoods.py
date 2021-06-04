import math

import torch


@torch.jit.script
def gaussian_ll(x, mu, var, epsilon: float = 1e-6):
    if epsilon:
        var = var.clone()
        with torch.no_grad():
            var.clamp_(min=epsilon)
    return 0.5 * (torch.log(var) + (x - mu)**2 / var + math.log(2 * math.pi))


@torch.jit.script
def categorical_ll(y, logits: torch.Tensor):
    """Compute Categorical log-likelihood

    Args:
        y (torch.LongTensor): Target values in [1, C-1]
        logits (torch.Tensor): Event log-probabilities (unnormalized)

    Returns:
        torch.Tensor: Log-probabilities
    """
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    y = y.long().unsqueeze(-1)
    y, log_pmf = torch.broadcast_tensors(y, logits)
    y = y[..., :1]
    return log_pmf.gather(-1, y).squeeze(-1)


def bernoulli_ll(y, logits):
    """Compute Bernoulli log-likelihood

    Args:
        x (torch.Tensor): Target values in {0, 1}
        probs (torch.Tensor): Event log-probabilities (unnormalized)
    """
    return categorical_ll(y, logits)
