from types import SimpleNamespace
from typing import List, Tuple

import torch
from torch.functional import Tensor
import torch.nn as nn

from torchtyping import TensorType

from vseq.modules import GaussianDense
from vseq.utils.operations import sequence_mask


class GRUAttention(nn.Module):
    def __init__(self, h_dim: int, num_layers: int = 1, batch_first: bool = True, bidirectional: bool = True):
        super().__init__()
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.gru = nn.GRU(h_dim, h_dim, num_layers=num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.linear = nn.Linear((bidirectional + 1) * h_dim, 1)

    def get_initial_state(self, batch_size: int, device: str = None):
        device = device if device is not None else self.linear.weight.device
        h = torch.zeros(batch_size, self.h_dim, device=device)
        return h

    def compute_context_vector(
        self, x: TensorType["B", "T", "h_dim"], a: TensorType["B", "T", 1]
    ) -> TensorType["B", "h_dim"]:
        return torch.matmul(x.permute(0, 2, 1), a).squeeze(-1)

    def forward(
        self,
        x: TensorType["B", "T", "h_dim"],
        x_sl: TensorType["B", int],
        seq_mask: TensorType["B", "T", int] = None,
    ) -> TensorType["B", "T", 1]:

        if seq_mask is None:
            seq_mask = sequence_mask(x_sl, max_len=x.shape[1], device=x.device)

        ps = torch.nn.utils.rnn.pack_padded_sequence(x, x_sl, batch_first=self.batch_first)
        h, h_n = self.gru(ps)
        h, h_sl = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])

        a_logits = self.linear(h)
        a_logits[~seq_mask] = -float("inf")  # set attention on padding to zeros
        a_weights = a_logits.softmax(1)  # normalize over time
        return a_weights


class GlobalCoder(nn.Module):
    def __init__(self, z_size: int, e_size: List[int]):
        super().__init__()
        self.z_size = z_size
        self.e_size = e_size
        self.num_embeddings = len(e_size)

        self.attentions = nn.ModuleList([GRUAttention(e) for e in e_size])
        self.posterior = nn.Sequential(
            nn.Linear(sum(e_size), sum(e_size)),
            nn.ReLU(),
            nn.Linear(sum(e_size), e_size[-1]),
            nn.ReLU(),
            GaussianDense(e_size[-1], z_size),
        )

        self.register_buffer("mu_p", torch.zeros(z_size))
        self.register_buffer("scale_p", torch.ones(z_size))
        self.prior = torch.distributions.Normal(loc=self.mu_p, scale=self.scale_p)

    def forward(
        self,
        x: List[TensorType["B", "T_i", "e_i"]],
        x_sl: TensorType["B", int],
        time_factors: List[int],
        seq_mask: TensorType["B", "T", bool] = None,
        temperature: float = 1.0,
        use_mode: bool = False,
    ) -> Tuple[TensorType["B", "T", "z_size"], SimpleNamespace]:
        """Encode a list of timeseries into a single global representation via recurrent attention."""

        if seq_mask is None:
            seq_mask = sequence_mask(x_sl, max_len=x[0].shape[1] * time_factors[0], device=x[0].device)

        context_vectors = []
        for i in range(self.num_embeddings):
            mask = seq_mask[:, :: time_factors[i]]
            lens = torch.ceil(x_sl / time_factors[i])

            a = self.attentions[i](x[i], x_sl=lens, seq_mask=mask)
            c = self.attentions[i].compute_context_vector(x[i], a)

            context_vectors.append(c)

        enc_mu, enc_sd = self.posterior(torch.cat(context_vectors, dim=-1))

        z = self.posterior[-1].rsample((enc_mu, temperature * enc_sd)) if not use_mode else enc_mu

        distributions = SimpleNamespace(z=z, enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=self.mu_p, prior_sd=self.scale_p)
        return z, distributions

    def generate(self, batch_size: int, temperature: float = 1.0, use_mode: bool = False):
        if use_mode:
            z = self.mu_p.repeat((batch_size, self.z_size))
        else:
            z = self.mu_p + temperature * self.scale_p * torch.randn((batch_size, self.z_size))

        distributions = SimpleNamespace(z=z, prior_mu=self.mu_p, prior_sd=self.scale_p)
        return z, distributions
