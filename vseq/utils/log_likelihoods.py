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


def discretized_logistic_ll(
    x: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor, num_bins: int = 256, double: bool = False
):
    """
    Log of the probability mass of the values x under the logistic distribution
    with parameters mean and scale. The sum is taken over all dimensions except
    for the first one (assumed to be batch).

    Assume input data to be inside (not at the edge) of num_bins equally-sized
    bins between 0 and 1. E.g. if num_bins=256 the 257 bin edges are:
    0, 1/256, ..., 255/256, 1.
    If values are at the left edge it's also ok, but let's be on the safe side

    :param x: tensor with shape (batch, channels, dim1, dim2)
    :param mean: tensor with mean of distribution, shape
                 (batch, channels, dim1, dim2)
    :param log_scale: tensor with log scale of distribution, shape has to be either
                  scalar or broadcastable
    :param num_bins: bin size (default: 256)
    :param double: whether double precision should be used for computations
    :return:
    """
    if double:
        log_scale = log_scale.double()
        x = x.double()
        mean = mean.double()
        eps = 1e-14
    else:
        eps = 1e-7

    scale = log_scale.exp()

    # Set values to the left of each bin
    x = torch.floor(x * num_bins) / num_bins

    cdf_plus = torch.ones_like(x)
    idx = x < (num_bins - 1) / num_bins
    cdf_plus[idx] = torch.sigmoid((x[idx] + 1 / num_bins - mean[idx]) / scale[idx])

    cdf_minus = torch.zeros_like(x)
    idx = x >= 1 / num_bins
    cdf_minus[idx] = torch.sigmoid((x[idx] - mean[idx]) / scale[idx])

    log_prob = torch.log(cdf_plus - cdf_minus + eps)

    log_prob = log_prob.sum((1, 2, 3))
    if double:
        log_prob = log_prob.float()
    return log_prob


# @torch.jit.script
def discretized_logistic_mixture_ll(x: torch.Tensor, logits: torch.Tensor, num_bins: int = 256):
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
        logits (torch.Tensor): (*, D x 3 x num_mix)
    """
    xs = [int(s) for s in x.size()]
    ls = [int(s) for s in logits.size()]

    # here and below: unpacking the params of the mixture of logistics
    num_mix = int(ls[-1] / (xs[-1] * 3))  # 3 for mean, scale and mixture coefficients
    logits = logits.view(xs + [num_mix * 3])  # 2 for mean, scale (D, 3 x num_mix)
    coeffients = logits[..., :num_mix]
    means = logits[..., :num_mix]
    log_scales = logits[..., num_mix : 2 * num_mix].clamp(min=-7.0)

    x = x.unsqueeze(-1).repeat(*[1] * x.ndim, num_mix)

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
    # (not actually used in our code)
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    cond = (cdf_delta > 1e-5).float()
    log_prob_inner = cond * cdf_delta.clamp(min=1e-12).log() + (1.0 - cond) * (log_pdf_mid - np.log(num_bins / 2))

    x_min_cond = (x > 0.999).float()
    log_probs = x_min_cond * log_one_minus_cdf_min + (1.0 - x_min_cond) * log_prob_inner
    x_plus_cond = (x < -0.999).float()
    log_probs = x_plus_cond * log_cdf_plus + (1.0 - x_plus_cond) * log_probs

    log_probs = log_probs + torch.log_softmax(coeffients, dim=-1)  # torch.Size([32, 383, 200, 10])
    log_probs = torch.logsumexp(log_probs, dim=-1)

    return log_probs


# # https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
# def discretized_mix_logistic_loss_1d(x, l):
#     """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval"""
#     # import IPython; IPython.embed(using=False)
#     # Pytorch ordering
#     x = x.unsqueeze(-1)  # (B, T, D, 1)
#     l = l.view(*l.shape[:-1], x.shape[2], -1)  # (B, T, D, 3 * num_mix) torch.Size([32, 383, 200, 30])

#     xs = [int(y) for y in x.size()]
#     ls = [int(y) for y in l.size()]

#     # here and below: unpacking the params of the mixture of logistics
#     nr_mix = int(ls[-1] / 3)
#     logit_probs = l[:, :, :, :nr_mix]
#     l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2]) # 2 for mean, scale
#     means = l[:, :, :, :, :nr_mix]  # torch.Size([32, 383, 200, 1, 10])
#     log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
#     # here and below: getting the means and adjusting them based on preceding
#     # sub-pixels
#     x = x.contiguous()
#     x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix]).cuda()

#     # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
#     centered_x = x - means
#     inv_stdv = torch.exp(-log_scales)
#     plus_in = inv_stdv * (centered_x + 1. / 255.)
#     cdf_plus = F.sigmoid(plus_in)
#     min_in = inv_stdv * (centered_x - 1. / 255.)
#     cdf_min = F.sigmoid(min_in)
#     # log probability for edge case of 0 (before scaling)
#     log_cdf_plus = plus_in - F.softplus(plus_in)
#     # log probability for edge case of 255 (before scaling)
#     log_one_minus_cdf_min = -F.softplus(min_in)
#     cdf_delta = cdf_plus - cdf_min  # probability for all other cases
#     mid_in = inv_stdv * centered_x
#     # log probability in the center of the bin, to be used in extreme cases
#     # (not actually used in our code)
#     log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

#     inner_inner_cond = (cdf_delta > 1e-5).float()
#     inner_inner_out  = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
#     inner_cond       = (x > 0.999).float()
#     inner_out        = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
#     cond             = (x < -0.999).float()
#     log_probs        = cond * log_cdf_plus + (1. - cond) * inner_out  # torch.Size([32, 383, 200, 1, 10])
#     log_probs        = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)  # torch.Size([32, 383, 200, 10])

#     return log_sum_exp(log_probs).sum(-1)


# def log_sum_exp(x):
#     """ numerically stable log_sum_exp implementation that prevents overflow """
#     # TF ordering
#     axis  = len(x.size()) - 1
#     m, _  = torch.max(x, dim=axis)
#     m2, _ = torch.max(x, dim=axis, keepdim=True)
#     return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


# def log_prob_from_logits(x):
#     """ numerically stable log_softmax implementation that prevents overflow """
#     # TF ordering
#     axis = len(x.size()) - 1
#     m, _ = torch.max(x, dim=axis, keepdim=True)
#     return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))
