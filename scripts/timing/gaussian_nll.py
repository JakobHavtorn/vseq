import math

import torch
import torch.jit as jit
import torch.nn.functional as F

from vseq.utils.timing import timeit
from vseq.utils.log_likelihoods import gaussian_ll


epsilon = 1e-3

x = torch.randn(100, 200)
mu = torch.randn(100, 200)
sd = torch.rand(100, 200)


timeit("F.gaussian_nll_loss(mu, x, sd, full=True, eps=epsilon, reduction='none')", globals=globals(), print_results=True)

timeit("gaussian_ll(x, mu, sd, epsilon)", globals=globals(), print_results=True)

timeit("gaussian_ll(x, mu, sd, 0)", globals=globals(), print_results=True)
