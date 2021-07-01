import math

import numpy as np
import torch
import torch.nn.functional as F


# TODO Discretized Laplace distribution
# TODO Discretized Mixture of Laplacians
# NOTE According to Unsupervised Blind Source Separation with Variational Auto-Encoders 
#      https://www.music.mcgill.ca/~julian/wp-content/uploads/2021/06/2021_eusipco_vae_bss_neri.pdf
#      this might make reconstructions and samples less blurry for VAEs.


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


# def discretized_laplace_ll(x: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor, num_bins: int = 256):
#     """
#     Log of the probability mass of the values x under the Laplace distribution
#     with parameters mean and scale.

#     :param x: tensor with shape (*, D)
#     :param mean: tensor with mean of distribution, shape (*, D)
#     :param log_scale: tensor with log scale of distribution, shape has to be either scalar or broadcastable
#     :param num_bins: bin size (default: 256)
#     :param double: whether double precision should be used for computations
#     :return:
#     """
#     pass


def discretized_logistic_ll(x: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor, num_bins: int = 256):
    """Log of the probability mass of the values x under the logistic distribution with parameters mean and scale.

    All dimensions are treated as independent.

    Assumes input data to be in `num_bins` equally-sized bins between -1 and 1.
    E.g. if num_bins=256, the 257 bin edges are: 
        -1, -254/256, ..., 254/256, 1  or 
        -1, -127/128, ..., 127/128, 1
    i.e. bin widths of 2/256 = 1/128.
    The data should not be exactly at the right bin edge (1).

    Noting that the CDF of the standard logistic distribution is simply the sigmoid function, we simply compute the
    probability mass under the logistic distribution per input element by using

        PDF(x_i | µ_i, s_i ) = CDF(x_i + 1/256 | µ_i, s_i) − CDF(x_i | µ_i, s_i ),

    where the locations µ_i and the log-scales log(s_i) are learned scalar parameters per input element and

        CDF(x | µ, s) = 1 / (1 + exp(-(x-µ)/s)) = Sigmoid((x-µ)/s)).
    
    We also use that

        log CDF(x | µ, s) = - Softplus((x - µ)/s)
        Softplus(x) = x - Softplus(-x)

    References:
        https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
        https://github.com/NVlabs/NVAE/blob/38eb9977aa6859c6ee037af370071f104c592695/distributions.py#L98
        https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py

    Args:
        x (torch.Tensor): targets to evaluate with shape (*).
        mean (torch.Tensor): mean of logistic distribution, shape (*) same as x.
        log_scale (torch.Tensor): log scale of distribution, shape (*) same as x, or either scalar or broadcastable.
        num_bins (int): number of bins, equivalent to specifying number of bits = log2(num_bins). Defaults to 256.
    :return:
    """
    # check input
    assert torch.max(x) <= 1.0 and torch.min(x) >= -1.0

    # compute x-µ and 1/s
    centered_x = x - mean
    inv_stdv = torch.exp(-log_scale)

    # compute CDF at left and right "bin edge" (floating) to compute total mass in between (cdf_delta)
    plus_in = inv_stdv * (centered_x + 1.0 / (num_bins - 1))  # add half a bin width
    cdf_plus = torch.sigmoid(plus_in)
    minus_in = inv_stdv * (centered_x - 1.0 / (num_bins - 1))  # subtract half a bin width
    cdf_minus = torch.sigmoid(minus_in)
    cdf_delta = cdf_plus - cdf_minus

    # log probability for edge case of 0 (mass from 0 to 0.5)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # = log CDF(x+0.5) via softplus(x) = x - softplus(-x)

    # log probability for edge case of 255 (mass from 254.5 to 255)
    log_one_minus_cdf_minus = -F.softplus(minus_in)  # = log 1 - CDF(x-0.5)

    # log probability in the center of the bin, to be used in extreme cases where cdf_delta is extremely small
    # TODO Understand this part
    # TODO Is this causing non-normalization?
    mid_in = inv_stdv * centered_x
    log_prob_mid = mid_in - log_scale - 2.0 * F.softplus(mid_in)
    log_prob_mid_safe = torch.where(
        cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-10)), log_prob_mid - np.log(num_bins / 2)
    )

    # handle edge cases
    log_probs = torch.where(x < 2 / num_bins - 1, log_cdf_plus, log_prob_mid_safe)  # edge case 0, x < -254/256
    log_probs = torch.where(x > 1 - 2 / num_bins, log_one_minus_cdf_minus, log_probs)  # edge case 255, x > 254/256
    return log_probs


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

    The implementation is partially as in https://arxiv.org/abs/1701.05517 but does not assume
    three RGB colour channels nor does it condition them on each other (as described in Section 2.2).
    Hence, the channels, and all other dimensions, are regarded as independent.

    For more details, refer to documentation for `discretized_logistic_ll`.

    Args:
        x (torch.Tensor): (*, D)
        logit_probs (torch.Tensor): (*, D, num_mix)
        means (torch.Tensor): (*, D, num_mix)
        log_scales (torch.Tensor): (*, D, num_mix)
        num_mix (int): Number of mixture components
        num_bins (int): Quantization level
    """
    # check input
    assert torch.max(x) <= 1.0 and torch.min(x) >= -1.0

    # repeat x for broadcasting to mixture dim
    x = x.unsqueeze(-1).expand(*[-1] * x.ndim, num_mix)  # (*, D, 3 x num_mix)

    # compute x-µ and 1/s
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)

    # compute CDF at left and right "bin edge" (floating) to compute total mass in between (cdf_delta)
    plus_in = inv_stdv * (centered_x + 1.0 / (num_bins - 1))
    cdf_plus = torch.sigmoid(plus_in)
    minus_in = inv_stdv * (centered_x - 1.0 / (num_bins - 1))
    cdf_minus = torch.sigmoid(minus_in)
    cdf_delta = cdf_plus - cdf_minus

    # log probability for edge case of 0 (mass from 0 to 0.5)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # = log CDF(x+0.5) via softplus(x) = x - softplus(-x)

    # log probability for edge case of 255 (mass from 254.5 to 255)
    log_one_minus_cdf_minus = -F.softplus(minus_in)  # = log 1 - CDF(x-0.5)

    # log probability in the center of the bin, to be used in extreme cases where cdf_delta is extremely small
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
    log_prob_mid_safe = torch.where(
        cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-10)), log_pdf_mid - np.log(num_bins / 2)
    )

    # handle edge cases
    log_probs = torch.where(x < 2 / num_bins - 1, log_cdf_plus, log_prob_mid_safe)  # edge case 0, x < -254/256
    log_probs = torch.where(x > 1 - 2 / num_bins, log_one_minus_cdf_minus, log_probs)  # edge case 255, x > 254/256

    log_probs = log_probs + torch.log_softmax(logit_probs, dim=-1)  # torch.Size([32, 383, 200, 10])
    log_probs = torch.logsumexp(log_probs, dim=-1)  # Normalize over mixture components (in log-prob space)

    return log_probs


def discretized_logistic_mixture_rgb_ll(x, parameters, num_bins: int = 256):
    """Compute log-likelihood for a mixture of discretized logistics designed for RGB images.

    The implementation is as in https://arxiv.org/abs/1701.05517
    Therefore, it conditions the RGB channels on each other (as described in Section 2.2) (G on R, B on R and G).
    Hence, the channels are regarded as dependent variables. All other dimensions are independent.

    For more details, refer to documentation for `discretized_logistic_ll`.

    Args:
        x (torch.Tensor): (*, D)
        logit_probs (torch.Tensor): (*, D, num_mix)
        means (torch.Tensor): (*, D, num_mix)
        log_scales (torch.Tensor): (*, D, num_mix)
        num_mix (int): Number of mixture components
        num_bins (int): Quantization level
    """
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

    minus_in = inv_stdv * (centered - 1.0 / num_bins)
    cdf_minus = torch.sigmoid(minus_in)

    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_minus = -F.softplus(minus_in)
    cdf_delta = cdf_plus - cdf_minus

    mid_in = inv_stdv * centered
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    log_prob_mid_safe = torch.where(
        cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-10)), log_pdf_mid - np.log(num_bins / 2)
    )

    # handle edge cases
    log_probs = torch.where(x < 2 / num_bins - 1, log_cdf_plus, log_prob_mid_safe)  # edge case 0, x < -254/256
    log_probs = torch.where(x > 1 - 2 / num_bins, log_one_minus_cdf_minus, log_probs)  # edge case 255, x > 254/256

    log_probs = torch.sum(log_probs, 1) + F.log_softmax(logit_probs, dim=1)  # B, M, H, W
    return torch.logsumexp(log_probs, dim=1)  # B, H, W
