import math

from typing import Optional, Union, Tuple

import torch

from torchtyping import TensorType


def kl_divergence(q_distrib: torch.distributions.Distribution, p_distrib: torch.distributions.Distribution):
    """Compute Kullback-Leibler divergence KL(q||p) between two distributions.

        KL(q||p) = \int q(x) \log [q(x) / p(x)] dx = - \int q(x) \log [p(x) / q(x)] dx

    Note that the order of the distributions q and p is flipped compared to the usual order.
    This is done since KL(q||p) is the order used in the ELBO.

    The usual order (which is NOT used here) is

        KL(p || q) = \int p(x) \log [p(x) / q(x)] dx = - \int p(x) \log [q(x) / p(x)] dx

    Consider two probability distributions P and Q.
    Usually, P represents the data, the observations, or a measured probability distribution.
    Distribution Q represents instead a theory, a model, a description or an approximation of P.

    The Kullback–Leibler divergence is then interpreted as the average difference of the number of bits
    required for encoding samples of P using a code optimized for Q rather than one optimized for P.

    In other words, the KL-divergence is the extra number of bits required for encoding samples of P using
    a code optimized for Q instead of one optimized for P.

    Args:
        q_distrib (Distribution): A :class:`~torch.distributions.Distribution` object.
        p_distrib (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape` in units of nats.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    return torch.distributions.kl_divergence(q_distrib, p_distrib)


def kl_divergence_mc(
    q_distrib: torch.distributions.Distribution, p_distrib: torch.distributions.Distribution, z: torch.Tensor
):
    """Elementwise Monte-Carlo estimation of KL between two distributions KL(q||p) (no reduction applied).

    Any number of dimensions works via broadcasting and correctly set `event_shape` (be careful).

    Args:
        z: Sample or samples from the variational distribution `q_distrib`.
        q_distrib (Distribution): A :class:`~torch.distributions.Distribution` object.
        p_distrib (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        tuple: KL divergence and log-likelihood of samples under q and under p (torch.Tensor)
    """
    q_logprob = q_distrib.log_prob(z)
    p_logprob = p_distrib.log_prob(z)
    kl_dwise = q_logprob - p_logprob
    return kl_dwise, q_logprob, p_logprob


@torch.jit.script
def kl_divergence_gaussian(mu_q: torch.Tensor, sd_q: torch.Tensor, mu_p: torch.Tensor, sd_p: torch.Tensor):
    """Elementwise analytical KL divergence between two Gaussian distributions KL(q||p) (no reduction applied)."""
    return sd_p.log() - sd_q.log() + (sd_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sd_p.pow(2)) - 0.5


def discount_free_nats(
    kld: TensorType["B":..., "shared":...],
    free_nats: float = None,
    shared_dims: Union[Tuple[int], int] = None,
) -> torch.Tensor:
    """Free bits as introduced in [1] but renamed to free nats because that's what it really is with log_e.

    In the paper they divide all latents Z into K groups. This implementation assumes a that each KL tensor passed
    to __call__ is one such group.

    By default, this method discounts `free_nats` units of nats elementwise in the KL regardless of its shape.

    If the KL tensor has more dimensions than the batch dimension, the free_nats budget can be optionally
    shared across those dimensions by setting `shared_dims`. E.g. if `kld.shape` is (32, 10) and `shared_dims` is -1,
    each of the 10 elements in the last dimension will get 1/10th of the free nats budget. If `kld.shape` is (32, 10, 10)
    and `shared_dims` is (-2, -1) each of the 10*10=100 elements will get 1 / 100th.

    The returned KL with `free_nats` discounted is equal to max(kld, freebits_per_dim)

    [1] https://arxiv.org/pdf/1606.04934
    """
    if free_nats is None or free_nats == 0:
        return kld

    if isinstance(shared_dims, int):
        shared_dims = (shared_dims,)

    # equally divide free nats budget over the elements in shared_dims
    if shared_dims is not None:
        n_elements = math.prod([kld.shape[d] for d in shared_dims])
        min_kl_per_dim = free_nats / n_elements
    else:
        min_kl_per_dim = free_nats

    min_kl_per_dim = torch.tensor(min_kl_per_dim, dtype=kld.dtype, device=kld.device)
    freenats_kl = torch.maximum(kld, min_kl_per_dim)
    return freenats_kl


def ornstein_uhlenbeck_sample_gaussian(mu: torch.Tensor, sd: torch.Tensor, num_samples: int, smoothing: float = 0.95):
    """Given a Gaussian distribution, return `num_samples` correlated OU samples with rho set to `smoothing` [1].

    Args:
        mu (torch.Tensor): Gaussian mean of shape (*)
        sd (torch.Tensor): Gaussian standard deviation of shape (*)
        num_samples (int): Number of OU samples to return
        smoothing (float): Degree of OU smoothing (symbol rho in paper)

    Returns:
        torch.Tensor: OU samples of shape (num_samples, *)

    [1] http://proceedings.mlr.press/v139/pervez21a.html
    """
    eps1 = torch.randn((num_samples,) + mu.shape, device=mu.device, dtype=mu.dtype)
    eps2 = torch.randn((num_samples,) + mu.shape, device=mu.device, dtype=mu.dtype)
    return sd * (smoothing * eps1 + torch.sqrt(1 - smoothing ** 2) * eps2) + mu


def ornstein_uhlenbeck_samples_from_guassian_sample(
    z: torch.Tensor,
    mu: torch.Tensor,
    sd: torch.Tensor,
    num_samples: int,
    smoothing: float = 0.95,
):
    """Given a Gaussian distribution and reparameterized sample `z`, return `num_samples` correlated OU samples with
    rho set to `smoothing` [1].

    Args:
        mu (torch.Tensor): Gaussian mean of shape (*)
        sd (torch.Tensor): Gaussian standard deviation of shape (*)
        num_samples (int): Number of OU samples to return
        smoothing (float): Degree of OU smoothing (symbol rho in paper)

    Returns:
        torch.Tensor: OU samples of shape (num_samples, *)

    [1] http://proceedings.mlr.press/v139/pervez21a.html
    """
    eps1 = torch.randn((num_samples,) + z.shape, device=z.device, dtype=z.dtype)
    return smoothing * z + (1 - smoothing) * mu + sd * torch.sqrt(1 - smoothing ** 2) * eps1


@torch.jit.script
def rsample_gaussian(mu: torch.Tensor, sd: torch.Tensor):
    """Return a reparameterized sample from a given Gaussian distribution.

    Args:
        mu (torch.Tensor): Gaussian mean of shape (*)
        sd (torch.Tensor): Gaussian standard deviation of shape (*)

    Returns:
        torch.Tensor: Reparameterized sample of shape (*)
    """
    return torch.randn_like(sd).mul(sd).add(mu)


def rsample_gumbel(
    mean: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    size: Optional[tuple] = None,
    fast: bool = True,
    eps: float = 1e-10,
):
    """Sample from a Gumbel distribution.

    Args:
        mean (Optional[torch.Tensor], optional): Gumbel mean. Defaults to None.
        scale (Optional[torch.Tensor], optional): Gumbel scale. Defaults to None.
        size (Optional[torch.Size], optional): Size of the sample (if mean and scale are None). Defaults to None.
        fast (bool, optional): If True, will sample using log(-log(u)) where u ~ Uniform(eps, 1-eps).
                               Otherwise, samples via log(e) where e ~ Exponential(1). Defaults to True.
        eps (float, optional): Small constant for numerical stability in fast sampling. Defaults to 1e-10.

    Returns:
        [type]: [description]
    """
    size = mean.size() if mean is not None else size
    if fast:
        gumbel = torch.log(-torch.log(torch.empty(size).uniform_(eps, 1.0 - eps)))
    else:
        gumbel = torch.empty(size).exponential_().log()

    if mean is None:
        return gumbel

    return mean + scale * gumbel


def rsample_gumbel_softmax(
    logits: torch.Tensor, hard: bool = False, tau: float = 1.0, eps: float = 1e-10, dim: int = -1
):
    """Samples from the Gumbel-Softmax distribution and optionally discretizes [1, 2].

    Args:
        logits: `[..., num_features]` unnormalized log probabilities
        tau: non-negative scalar temperature
        hard: if ``True``, the returned samples will be discretized as one-hot vectors,
              but will be differentiated as if it is the soft sample in autograd
        dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
        Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
        If ``hard=True``, the returned samples will be one-hot, otherwise they will
        be probability distributions that sum to 1 across `dim`.

    Note:
        The main trick for `hard` is to do  `y_hard + (y_soft - y_soft.detach())`
        This achieves two things:
         1. makes the output value exactly one-hot (since we add then subtract y_soft value)
         2. makes the gradient equal to y_soft gradient (since we strip all other gradients)

    Examples:
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> rsample_gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> rsample_gumbel_softmax(logits, tau=1, hard=True)

    [1] https://arxiv.org/abs/1611.00712
    [2] https://arxiv.org/abs/1611.01144
    """

    gumbels = torch.log(-torch.log(torch.empty_like(logits).uniform_(eps, 1.0 - eps)))  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if not hard:
        # Reparametrization trick
        return y_soft

    # Straight through estimator
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    return y_hard + (y_soft - y_soft.detach())


@torch.jit.script
def rsample_logistic(mu: torch.Tensor, log_scale: torch.Tensor, eps: float = 1e-8):
    """
    Returns a sample from Logistic with specified mean and log scale.

    :param mu: a tensor containing the mean.
    :param log_scale: a tensor containing the log scale.
    :return: a reparameterized sample with the same size as the input mean and log scale.
    """
    u = torch.empty_like(mu).uniform_(eps, 1 - eps)  # uniform sample in the interval (eps, 1 - eps)
    sample = mu + torch.exp(log_scale) * (torch.log(u) - torch.log(1 - u))  # transform to logistic
    return sample


def rsample_discretized_logistic(mu: torch.Tensor, log_scale: torch.Tensor, eps: float = 1e-8):
    """Return a sample from a discretized logistic with values standardized to be in [-1, 1]

    This is done by sampling the corresponding continuous logistic and clamping values outside
    the interval to the endpoints.

    We do not further quantize the samples here.
    """
    return rsample_logistic(mu, log_scale, eps).clamp(-1, 1)


def rsample_discretized_logistic_mixture(
    logit_probs: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    num_mix: int,
    eps: float = 1e-5,
    t: float = 1.0,
):
    """Return a reparameterized sample from a given Discretized Logistic Mixture distribution.

    Code taken from PyTorch adaptation of original PixelCNN++ TensorFlow implementation:
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py but does not include the channel specific conditional modelling.

    Args:
        logit_probs (torch.Tensor): (*, D, num_mix)
        means (torch.Tensor): (*, D, num_mix)
        log_scales (torch.Tensor): (*, D, num_mix)
        num_mix (int): Number of mixture components
        eps (float): Bounds [eps, 1-eps] on the uniform rv used to sample the mixture coefficients and the logistic.
        t (float): Temperature for Gumbel sampling

    Returns:
        torch.Tensor: Sample from the DLM `(*, D)`
    """
    # sample mixture indicator from softmax
    gumbel = -torch.log(-torch.log(torch.empty_like(means).uniform_(eps, 1.0 - eps)))
    argmax = torch.argmax(logit_probs / t + gumbel, dim=-1)
    one_hot = torch.nn.functional.one_hot(argmax, num_mix)

    # select logistic parameters
    means = torch.sum(means * one_hot, dim=-1)
    log_scales = torch.sum(log_scales * one_hot, dim=-1)

    # sample from logistic (we don't actually round to the nearest 8bit value)
    x = rsample_discretized_logistic(means, log_scales)
    return x


def rsample_discretized_logistic_mixture_rgb(parameters: torch.Tensor, num_mix: int, eps: float = 1e-5, t: float = 1.0):
    """Return a reparameterized sample from a given Discretized Logistic Mixture distribution for RGB images.

    Code taken from PyTorch adaptation of original PixelCNN++ TensorFlow implementation:
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py.

    Args:
        parameters (torch.Tensor): Mixture coefficients, means and log-scales for logistic mixtures `(*, D * 3 * num_mix)`.
        num_mix (int): Number of mixture components in the the DLM.
        eps (float): Bounds [eps, 1-eps] on the uniform rv used to sample the mixture coefficients and the logistic.

    Returns:
        torch.Tensor: Sample from the DLM `(*, D)`
    """
    # TODO Streamline
    B, C, H, W = parameters.size()

    # unpack parameters
    logit_probs = parameters[:, :num_mix, :, :]  # B, M, H, W
    l = parameters[:, num_mix:, :, :].view(B, 3, 3 * num_mix, H, W)  # B, 3, 3 * M, H, W
    means = l[:, :, :num_mix, :, :]  # B, 3, M, H, W
    log_scales = torch.clamp(l[:, :, num_mix : 2 * num_mix, :, :], min=-7.0)  # B, 3, M, H, W
    coeffs = torch.tanh(l[:, :, 2 * num_mix : 3 * num_mix, :, :])  # B, 3, M, H, W

    gumbel = -torch.log(-torch.log(torch.empty_like(logit_probs).uniform_(eps, 1.0 - eps)))  # B, M, H, W
    argmax = torch.argmax(logit_probs / t + gumbel, 1)
    one_hot = torch.nn.functional.one_hot(argmax, num_mix, dim=1)  # B, M, H, W
    one_hot = one_hot.unsqueeze(1)  # B, 1, M, H, W

    # select logistic parameters
    means = torch.sum(means * one_hot, dim=2)  # B, 3, H, W
    log_scales = torch.sum(log_scales * one_hot, dim=2)  # B, 3, H, W
    coeffs = torch.sum(coeffs * one_hot, dim=2)  # B, 3, H, W

    # cells from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.Tensor(means.size()).uniform_(eps, 1.0 - eps).cuda()  # B, 3, H, W
    x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1.0 - u))  # B, 3, H, W

    x0 = torch.clamp(x[:, 0, :, :], -1, 1.0)  # B, H, W
    x1 = torch.clamp(x[:, 1, :, :] + coeffs[:, 0, :, :] * x0, -1, 1)  # B, H, W
    x2 = torch.clamp(x[:, 2, :, :] + coeffs[:, 1, :, :] * x0 + coeffs[:, 2, :, :] * x1, -1, 1)  # B, H, W

    x = torch.stack([x0, x1, x2], dim=1)
    return x


@torch.jit.script
def rsample_exponential(rate: torch.Tensor, eps: float = 1e-8):
    """Returns samples from the Exponential distribution parameterized with λ=rate

    Args:
        rate (torch.Tensor): The rate of the exponential distribution
        eps (float): Small constant for numerical stability of log( uniform(0, 1) )
    """
    # return torch.empty_like(rate).exponential_() / rate
    return -(-torch.empty_like(rate).uniform_(eps, 1 - eps)).log1p() / rate  # faster and jit supported


@torch.jit.script
def rsample_laplace(loc: torch.Tensor, scale: torch.Tensor, eps: float = 1e-8):
    """
    Returns a sample from laplace with specified mean and log scale.

    :param loc: a tensor containing the mean.
    :param scale: a tensor containing the log scale.
    :return: a reparameterized sample with the same size as the input mean and log scale.
    """
    e1 = rsample_exponential(scale, eps)
    e2 = rsample_exponential(scale, eps)
    return e1 - e2 + loc


def rsample_discretized_laplace(loc: torch.Tensor, scale: torch.Tensor, eps: float = 1e-8):
    """Return a sample from a discretized laplace with values standardized to be in [-1, 1]

    This is done by sampling the corresponding continuous laplace and clamping values outside
    the interval to the endpoints.

    We do not further quantize the samples here.
    """
    return rsample_laplace(loc, scale, eps).clamp(-1, 1)


def rsample_discretized_laplace_mixture(
    logit_probs: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    num_mix: int,
    eps: float = 1e-5,
    t: float = 1.0,
):
    """Return a reparameterized sample from a given Discretized Laplace Mixture distribution.

    Args:
        logit_probs (torch.Tensor): (*, D, num_mix)
        means (torch.Tensor): (*, D, num_mix)
        log_scales (torch.Tensor): (*, D, num_mix)
        num_mix (int): Number of mixture components
        eps (float): Bounds [eps, 1-eps] on the uniform rv used to sample the mixture coefficients and the laplace.
        t (float): Temperature for Gumbel sampling

    Returns:
        torch.Tensor: Sample from the discretized Laplace mixture `(*, D)`
    """
    # sample mixture indicator from softmax
    gumbel = -torch.log(-torch.log(torch.empty_like(means).uniform_(eps, 1.0 - eps)))
    argmax = torch.argmax(logit_probs / t + gumbel, dim=-1)
    one_hot = torch.nn.functional.one_hot(argmax, num_mix)

    # select laplace parameters
    means = torch.sum(means * one_hot, dim=-1)
    log_scales = torch.sum(log_scales * one_hot, dim=-1)

    # sample from laplace (we don't actually round to the nearest 8bit value)
    x = rsample_discretized_laplace(means, log_scales)
    return x
