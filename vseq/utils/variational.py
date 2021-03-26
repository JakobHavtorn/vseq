import torch


def kl_divergence_mc(
    q_distrib: torch.distributions.Distribution, p_distrib: torch.distributions.Distribution, z: torch.Tensor
):
    """Elementwise Monte-Carlo estimation of KL between two distributions KL(q||p) (no reduction applied).

    Any number of dimensions works via broadcasting and correctly set `event_shape` (be careful).

    Args:
        z: Samples
        q_distrib: Variational distribution
        p_distrib: Target distribution

    Returns:
        tuple: Spatial KL divergence and log-likelihood of samples under q and under p (torch.Tensor)
    """
    q_logprob = q_distrib.log_prob(z)
    p_logprob = p_distrib.log_prob(z)
    kl_elementwise = q_logprob - p_logprob
    return kl_elementwise, q_logprob, p_logprob
