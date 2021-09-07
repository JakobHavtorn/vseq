from types import SimpleNamespace
from typing import List
from vseq.utils.variational import rsample_gaussian

import torch
import torch.nn as nn

from torchtyping import TensorType

from vseq.modules import GaussianDense


class RecurrentAttention(nn.Module):
    def __init__(self, h_dim: int, bidirectional: bool = True):
        super().__init__()
        self.h_dim = h_dim
        self.bidirectional = bidirectional

        self.gru = nn.GRU(h_dim, h_dim, bidirectional=bidirectional)
        self.linear = nn.Linear((bidirectional + 1) * h_dim, 1)

    def get_initial_state(self, batch_size: int, device: str = None):
        device = device if device is not None else self.linear.weight.device
        h = torch.zeros(batch_size, self.h_dim, device=device)
        return h

    def compute_context_vector(self, x: TensorType["B", "T", "h_dim"], a: TensorType["B", "T", 1]) -> TensorType["B", "h_dim"]:
        return torch.matmul(x.permute(0, 2, 1), a).squeeze(-1)

    def forward(self, x: TensorType["B", "T", "h_dim"], seq_mask: TensorType["B", "T", int]):
        h, h_n = self.gru(x)
        a_logits = self.linear(h)
        a_logits[~seq_mask] = - float("inf")  # set attention on padding to zeros
        a_weights = a_logits.softmax(1)  # normalize over time
        return a_weights


class GlobalCoder(nn.Module):
    def __init__(self, z_size: int, e_size: List[int]):
        super().__init__()
        self.z_size = z_size
        self.e_size = e_size
        self.num_embeddings = len(e_size)

        self.attentions = nn.ModuleList([RecurrentAttention(e) for e in e_size])
        self.posterior = nn.Sequential(
            nn.Linear(sum(e_size), e_size[-1]),
            nn.ReLU(),
            GaussianDense(e_size[-1], z_size)
        )
        
        self.register_buffer("mu_p", torch.zeros(z_size))
        self.register_buffer("scale_p", torch.ones(z_size))
        self.prior = torch.distributions.Normal(loc=self.mu_p, scale=self.scale_p)

    def forward(self, x: List[TensorType["B", "T_i", "e_i"]], seq_mask: TensorType["B", "T", int], time_factors: List[int], temperature: float = 1.0, use_mode: bool = False):
        cs = []
        for i in range(self.num_embeddings):
            mask = seq_mask[:, :: time_factors[i]]
            a = self.attentions[i](x[i], mask)
            c = self.attentions[i].compute_context_vector(x[i], a)
            cs.append(c)
        
        enc_mu, enc_sd = self.posterior(torch.cat(cs, dim=-1))

        z = self.posterior[-1].rsample((enc_mu, temperature * enc_sd)) if not use_mode else enc_mu

        distributions = SimpleNamespace(z=z, enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=self.mu_p, prior_sd=self.scale_p)
        return z, distributions

    def generate(self, batch_size: int, temperature: float = 1.0, use_mode: bool = False):
        z = rsample_gaussian(self.mu_p, temperature * self.scale_p) if not use_mode else self.mu_p
        distributions = SimpleNamespace(z=z, prior_mu=self.mu_p, prior_sd=self.scale_p)
        return z, distributions
