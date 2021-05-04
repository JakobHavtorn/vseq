from types import SimpleNamespace
from typing import Union, List, Optional

import torch
import torch.nn as nn
import torch.distributions as D

from torchtyping import TensorType

from vseq.evaluation.metrics import BitsPerDimMetric, LLMetric, LossMetric, PerplexityMetric
from vseq.models import BaseModel
from vseq.modules.hmlstm import HMLSTM
from vseq.utils.operations import sequence_mask


class HMLM(BaseModel):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        sizes: Union[int, List[int]],
        num_layers: Optional[int] = None,
        layer_norm: bool = False,
        dropout_rate_e: float = 0.5,
        dropout_rate_h: float = 0.5,
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
        self.dropout_rate_e = dropout_rate_e
        self.dropout_rate_h = dropout_rate_h

        self.slices = [(0, self.sizes[0])] + [(self.sizes[i], self.sizes[i + 1]) for i in range(len(self.sizes) - 1)]

        self.embedding_in = nn.Embedding(num_embeddings, embedding_dim)
        self.dropout_e = nn.Dropout(p=self.dropout_rate_e)

        self.hmlstm = HMLSTM(input_size=embedding_dim, sizes=self.sizes, num_layers=num_layers, layer_norm=layer_norm)
        self.dropout_h = nn.Dropout(p=self.dropout_rate_h)

        self.weight = nn.Linear(sum(self.sizes), self.num_layers)

        self.embedding_out = nn.Linear(sum(self.sizes), sum(self.sizes))
        self.relu = nn.ReLU()

        self.output = nn.Linear(sum(self.sizes), num_embeddings)

        self.loss = nn.CrossEntropyLoss()

    def compute_loss(
        self,
        logits: TensorType["B", "T", "num_embeddings"],
        targets: TensorType["B", "T"],
        x_sl: TensorType["B", torch.int64],
        seq_mask: TensorType["B", "T", torch.bool] = None,
    ):
        if seq_mask is None:
            seq_mask = sequence_mask(x_sl - 1, dtype=float, device=logits.device)

        p_x = D.Categorical(logits=logits)

        log_prob_twise = p_x.log_prob(targets) if seq_mask is None else p_x.log_prob(targets) * seq_mask
        log_prob = log_prob_twise.sum(1)

        loss = -log_prob.sum() / (x_sl - 1).sum()  # nats per dim
        return loss, log_prob, p_x

    def forward(
        self,
        x: TensorType["B", "T"],
        x_sl: TensorType["B", torch.int64],
        h_init: Optional[List[TensorType["B", "T", "hidden_size"]]] = None,
        c_init: Optional[List[TensorType["B", "T", "hidden_size"]]] = None,
        z_init: Optional[List[TensorType["B", "T", 1]]] = None,
        **kwargs,
    ):
        target = x[:, 1:].clone().detach()

        emb = self.embedding_in(x[:, :-1])  # B * T * embedding_dim
        emb = self.dropout_e(emb)
        h, c, z, (h_out, c_out, z_out) = self.hmlstm(emb, h_init, c_init, z_init, **kwargs)  # B * T * hidden_size

        h = torch.cat(h, dim=2)  # B * T * sum(hidden_sizes)
        h = self.dropout_h(h)

        g = torch.sigmoid(self.weight(h))

        g = [g[..., i : i + 1].expand(*g.shape[:2], self.sizes[i]) for i in range(self.num_layers)]  # Expand to size h
        g = torch.cat(g, dim=2)
        h_g = g * h
        h_e = self.relu(self.embedding_out(h_g))

        p_logits = self.output(h_e)

        seq_mask = sequence_mask(x_sl - 1, dtype=torch.bool, device=p_logits.device)
        loss, log_prob, p_x = self.compute_loss(p_logits, target, x_sl, seq_mask=seq_mask)

        u_ops, c_ops, f_ops, u_rates, c_rates, f_rates = self.hmlstm.realized_operations(z, x_sl - 1, seq_mask)

        rate_metrics = [
            (
                LossMetric(name=f"u_rate_{l}", values=u_ops[l] * seq_mask[:, 1:], reduce_by=x_sl - 1),
                LossMetric(name=f"c_rate_{l}", values=c_ops[l] * seq_mask[:, 1:], reduce_by=x_sl - 1),
                LossMetric(name=f"f_rate_{l}", values=f_ops[l] * seq_mask[:, 1:], reduce_by=x_sl - 1),
            )
            for l in range(self.num_layers)
        ]

        metrics = [
            LossMetric(loss, weight_by=log_prob.numel()),
            LLMetric(log_prob),
            BitsPerDimMetric(log_prob, reduce_by=x_sl - 1),
            PerplexityMetric(log_prob, reduce_by=x_sl - 1),
            *[m for metrics in rate_metrics for m in metrics],
        ]

        outputs = SimpleNamespace(
            loss=loss,
            ll=log_prob,
            p_x=p_x,
            update_ops=u_ops,
            copy_ops=c_ops,
            flush_ops=f_ops,
            update_rates=u_rates,
            copy_rates=c_rates,
            flush_rates=f_rates,
        )

        return loss, metrics, outputs
