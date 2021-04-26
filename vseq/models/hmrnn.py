from types import SimpleNamespace
from typing import Union, List, Optional
from vseq.evaluation.metrics import BitsPerDimMetric, LLMetric, LossMetric, PerplexityMetric

import torch
import torch.nn as nn
import torch.distributions as D

from torch.nn import Module
from torch.autograd import Variable

from vseq.utils.operations import sequence_mask
from vseq.modules.hmlstm import HMLSTM
from vseq.models import BaseModel


class HMLM(BaseModel):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        sizes: Union[int, List[int]],
        num_layers: Optional[int] = None,
        layer_norm: bool = False,
    ):
        super().__init__()

        assert (
            (isinstance(sizes, list) and num_layers is None) or (isinstance(sizes, int) and num_layers is not None),
            "Must give `sizes` as list and not `num_layers` OR `sizes` as int along with a number of layers",
        )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sizes = sizes if num_layers is None else [sizes] * num_layers
        self.num_layers = len(sizes) if num_layers is None else num_layers
        self.layer_norm = layer_norm

        self.slices = [(0, self.sizes[0])] + [(self.sizes[i], self.sizes[i + 1]) for i in range(len(self.sizes) - 1)]

        self.dropout = nn.Dropout(p=0.5)

        self.embedding_in = nn.Embedding(num_embeddings, embedding_dim)

        self.hmlstm = HMLSTM(input_size=embedding_dim, sizes=self.sizes, num_layers=num_layers, layer_norm=layer_norm)

        self.weight = nn.Linear(sum(self.sizes), self.num_layers)

        self.embedding_out = nn.Linear(sum(self.sizes), sum(self.sizes))

        self.output = nn.Linear(sum(self.sizes), num_embeddings)

        self.relu = nn.ReLU()

        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        x,
        x_sl,
        h_init: Optional[List[torch.Tensor]] = None,
        c_init: Optional[List[torch.Tensor]] = None,
        z_init: Optional[List[torch.Tensor]] = None,
    ):
        # x : B * T

        y = x[:, 1:].clone().detach()

        emb = self.embedding_in(x[:, :-1])  # B * T * embedding_dim
        emb = self.dropout(emb)
        h, c, z, (h_out, c_out, z_out) = self.hmlstm(emb, h_init, c_init, z_init)  # B * T * hidden_size

        h = torch.cat(h, dim=2)  # B * T * sum(hidden_sizes)
        h = self.dropout(h)

        g = torch.sigmoid(self.weight(h))

        g = [g[..., i : i + 1].expand(*g.shape[:2], self.sizes[i]) for i in range(self.num_layers)]  # Expand to h
        g = torch.cat(g, dim=2)
        h_g = g * h
        h_e = self.relu(self.embedding_out(h_g))

        p_logits = self.output(h_e)
        seq_mask = sequence_mask(x_sl - 1, dtype=float, device=p_logits.device)
        p_x = D.Categorical(logits=p_logits)
        log_prob_twise = p_x.log_prob(y) * seq_mask
        log_prob = log_prob_twise.sum(1)
        loss = -log_prob.sum() / (x_sl - 1).sum()

        clock_rate = [(ze.squeeze() * seq_mask).sum() / (x_sl - 1).sum() for ze in z]

        outputs = SimpleNamespace(loss=loss, ll=log_prob, p_x=p_x)

        clock_rates = [LossMetric(clock_rate[i], name=f"clock_rate_{i}") for i in range(self.num_layers)]
        metrics = [
            LossMetric(loss, weight_by=log_prob.numel()),
            LLMetric(log_prob),
            BitsPerDimMetric(log_prob, reduce_by=x_sl - 1),
            PerplexityMetric(log_prob, reduce_by=x_sl - 1),
            *clock_rates,
        ]

        return loss, metrics, outputs
