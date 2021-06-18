import math

import numpy as np
import torch
from torch._C import TensorType
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
    return -0.5 * (math.log(2 * math.pi) + torch.log(var) + (x - mu) ** 2 / var)


def categorical_ll(x: torch.Tensor, logits: torch.Tensor):
    """Compute Categorical log-likelihood

    Args:
        x (torch.LongTensor): Target values in [1, C-1] of any shape.
        logits (torch.Tensor): Event log-probabilities (unnormalized) of same shape (x.shape, C)

    Returns:
        torch.Tensor: Log-probabilities
    """
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    x = x.long().unsqueeze(-1)
    x, logits = torch.broadcast_tensors(x, logits)
    x = x[..., :1]
    return logits.gather(-1, x).squeeze(-1)


def bernoulli_ll(x: torch.Tensor, logits: torch.Tensor):
    """Compute Bernoulli log-likelihood

    Args:
        x (torch.Tensor): Target values in {0, 1} of any shape.
        probs (torch.Tensor): Event log-probabilities (unnormalized) of same shape as `x`.

    Returns:
        torch.Tensor: Log-probabilities
    """
    x, logits = torch.broadcast_tensors(x, logits)
    return -F.binary_cross_entropy_with_logits(logits, x, reduction="none")


def von_mises_ll(x: torch.Tensor, logits: torch.Tensor):
    """Compute Bernoulli log-likelihood"""
    raise NotImplementedError()


# https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
# https://github.com/NVlabs/NVAE/blob/38eb9977aa6859c6ee037af370071f104c592695/distributions.py#L98
def discretized_logistic_ll(x: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor, num_bins: int = 256):
    """
    Log of the probability mass of the values x under the logistic distribution
    with parameters mean and scale.

    Assume input data to be inside (not at the edge) of `num_bins` equally-sized
    bins between 0 and 1. E.g. if num_bins=256 the 257 bin edges are:
    0, 1/256, ..., 255/256, 1.

    :param x: tensor with shape (batch, channels, dim1, dim2)z
    :param mean: tensor with mean of distribution, shape (batch, channels, dim1, dim2)
    :param log_scale: tensor with log scale of distribution, shape has to be either scalar or broadcastable
    :param num_bins: bin size (default: 256)
    :param double: whether double precision should be used for computations
    :return:
    """
    centered = x - mean  # B, 3, H, W
    inv_stdv = torch.exp(-log_scale)

    plus_in = inv_stdv * (centered + 1.0 / (num_bins - 1))
    cdf_plus = torch.sigmoid(plus_in)

    min_in = inv_stdv * (centered - 1.0 / (num_bins - 1))
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)

    mid_in = inv_stdv * centered
    log_pdf_mid = mid_in - log_scale - 2.0 * F.softplus(mid_in)
    cdf_delta = cdf_plus - cdf_min

    log_prob_mid_safe = torch.where(
        cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-10)), log_pdf_mid - np.log(num_bins / 2)
    )
    # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
    # which is mapped to 0.9922
    log_prob = torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.99, log_one_minus_cdf_min, log_prob_mid_safe))

    return log_prob


# @torch.jit.script
def discretized_logistic_mixture_ll(
    x: torch.Tensor,
    logit_probs: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    num_mix: int,
    num_bins: int = 256,
):
    """Compute log-likelihood for a mixture of discretized logistics.

    Assumes the data has been rescaled to [-1, 1]

    The implementation is partially as in https://arxiv.org/abs/1701.05517 but does not assume
    three RGB colour channels nor does it condition them on each other (as described in Section 2.2).
    Hence, the channels are regarded as independent.

    Noting that the CDF of the standard logistic distribution is simply the sigmoid function, we simply compute the
    probability mass under the logistic distribution per input element by using

        p(x_i | µ_i, s_i ) = CDF(x_i + 1/256 | µ_i, s_i) − CDF(x_i | µ_i, s_i ),

    where the locations µ_i and the log-scales log(s_i) are learned scalar parameters per input element and

        CDF(x | µ, s) = 1 / (1 + exp(-(x-µ)/s)).

    Code has been adapted from https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py

    Args:
        x (torch.Tensor): (*, D)
        logit_probs (torch.Tensor): (*, D, num_mix)
        means (torch.Tensor): (*, D, num_mix)
        log_scales (torch.Tensor): (*, D, num_mix)
        num_mix (int): Number of mixture components
        num_bins (int): Quantization level
    """
    # prepare x for broadcasting
    assert torch.max(x) <= 1.0 and torch.min(x) >= -1.0
    x = x.unsqueeze(-1).expand(*[-1] * x.ndim, num_mix)  # (*, D, 3 x num_mix)

    # compute x-µ and 1/s
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)

    # compute CDF at right bin and left bin and total mass in between
    plus_in = inv_stdv * (centered_x + 1.0 / (num_bins - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / (num_bins - 1))
    cdf_min = torch.sigmoid(min_in)
    cdf_delta = cdf_plus - cdf_min

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)

    # log probability in the center of the bin, to be used in extreme cases where cdf_delta is extremely small
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    log_prob_mid_safe = torch.where(
        cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-10)), log_pdf_mid - np.log(num_bins / 2)
    )

    # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
    # which is mapped to 0.9922 (B, 3, M, H, W)
    log_probs = torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.99, log_one_minus_cdf_min, log_prob_mid_safe))

    log_probs = log_probs + torch.log_softmax(logit_probs, dim=-1)  # torch.Size([32, 383, 200, 10])
    log_probs = torch.logsumexp(log_probs, dim=-1)

    return log_probs


def discretized_logistic_mixture_rgb_ll(x, parameters, num_bins: int = 256):
    # TODO Streamline like discretized_logistic_mixture_ll
    assert torch.max(x) <= 1.0 and torch.min(x) >= -1.0

    xs = [int(s) for s in x.size()]  # B, C, H, W
    assert xs[1] == 3, "only RGB images are considered."
    ps = [int(s) for s in parameters.size()]

    num_mix = int(ps[-1] / 10)
    logit_probs = parameters[:, :, :, :num_mix]
    parameters = parameters[:, :, :, num_mix:].contiguous().view(xs + [num_mix * 3])  # 3 for mean, scale, coef
    means = parameters[:, :, :, :, :num_mix]
    log_scales = torch.clamp(parameters[:, :, :, :, num_mix : 2 * num_mix], min=-7.0)
    coeffs = torch.tanh(parameters[:, :, :, :, 2 * num_mix : 3 * num_mix])

    x = x.unsqueeze(4)  # B, 3, H , W, 1
    x = x.expand(-1, -1, -1, -1, num_mix).permute(0, 1, 4, 2, 3)  # B, 3, M, H, W
    mean1 = means[:, 0, :, :, :]  # B, M, H, W
    mean2 = means[:, 1, :, :, :] + coeffs[:, 0, :, :, :] * x[:, 0, :, :, :]
    mean3 = means[:, 2, :, :, :] + coeffs[:, 1, :, :, :] * x[:, 0, :, :, :] + coeffs[:, 2, :, :, :] * x[:, 1, :, :, :]
    means = torch.stack([mean1, mean2, mean3], dim=1)  # B, 3, M, H, W

    centered = x - means  # B, 3, M, H, W
    inv_stdv = torch.exp(-log_scales)

    plus_in = inv_stdv * (centered + 1.0 / num_bins)
    cdf_plus = torch.sigmoid(plus_in)

    min_in = inv_stdv * (centered - 1.0 / num_bins)
    cdf_min = torch.sigmoid(min_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    log_prob_mid_safe = torch.where(
        cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-10)), log_pdf_mid - np.log(num_bins / 2)
    )
    # the original implementation uses x > 0.999, this ignores the largest possible pixel value (255)
    # which is mapped to 0.9922
    log_probs = torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.99, log_one_minus_cdf_min, log_prob_mid_safe))

    log_probs = torch.sum(log_probs, 1) + F.log_softmax(logit_probs, dim=1)  # B, M, H, W
    return torch.logsumexp(log_probs, dim=1)  # B, H, W
