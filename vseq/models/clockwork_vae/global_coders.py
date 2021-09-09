from types import SimpleNamespace
from typing import List, Tuple
from vseq.utils.operations import sequence_mask
from vseq.utils.variational import rsample_gaussian

import torch
import torch.nn as nn

from torchtyping import TensorType

from vseq.modules import GaussianDense


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

        seq_mask = sequence_mask(x_sl, max_len=x.shape[1], device=x.device) if seq_mask is None else seq_mask

        ps = torch.nn.utils.rnn.pack_padded_sequence(x, x_sl, batch_first=self.batch_first)
        h, h_n = self.gru(ps)
        h, h_sl = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

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
        self.posterior = nn.Sequential(nn.Linear(sum(e_size), e_size[-1]), nn.ReLU(), GaussianDense(e_size[-1], z_size))

        self.register_buffer("mu_p", torch.zeros(z_size))
        self.register_buffer("scale_p", torch.ones(z_size))
        self.prior = torch.distributions.Normal(loc=self.mu_p, scale=self.scale_p)

    def forward(
        self,
        x: List[TensorType["B", "T_i", "e_i"]],
        x_sl: TensorType["B", int],
        time_factors: List[int],
        seq_mask: TensorType["B", "T", int] = None,
        temperature: float = 1.0,
        use_mode: bool = False,
    ) -> Tuple[TensorType["B", "T", "z_size"], SimpleNamespace]:
        """Encode a list of timeseries into a single global representation via recurrent attention."""

        seq_mask = sequence_mask(x_sl, max_len=x.shape[1], device=x.device) if seq_mask is None else seq_mask

        cs = []
        for i in range(self.num_embeddings):
            mask = seq_mask[:, :: time_factors[i]]
            lens = torch.ceil(x_sl / time_factors[i])

            a = self.attentions[i](x[i], x_sl=lens, seq_mask=mask)
            c = self.attentions[i].compute_context_vector(x[i], a)

            cs.append(c)

        enc_mu, enc_sd = self.posterior(torch.cat(cs, dim=-1))

        z = self.posterior[-1].rsample((enc_mu, temperature * enc_sd)) if not use_mode else enc_mu

        distributions = SimpleNamespace(z=z, enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=self.mu_p, prior_sd=self.scale_p)
        return z, distributions

    def generate(self, batch_size: int, temperature: float = 1.0, use_mode: bool = False):
        z = (
            self.mu_p + self.scale_p * torch.randn((batch_size, self.z_size))
            if not use_mode
            else self.mu_p.repeat((batch_size, self.z_size))
        )
        # z = rsample_gaussian(self.mu_p, temperature * self.scale_p) if not use_mode else self.mu_p
        distributions = SimpleNamespace(z=z, prior_mu=self.mu_p, prior_sd=self.scale_p)
        return z, distributions
