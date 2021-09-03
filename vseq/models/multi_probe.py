from types import SimpleNamespace
from typing import Tuple, List

from vseq.evaluation.metrics import LossMetric, WindowMeanMetric

import torch
import torch.nn as nn
import torch.distributions as D

from vseq.utils.operations import sequence_mask
from vseq.evaluation import LossMetric, BitsPerDimMetric, AccuracyMetric

from .base_model import BaseModel


class MultiProbe(BaseModel):
    def __init__(
        self,
        num_probes: int,
        input_size: int,
        output_size: int
    ):
        super().__init__()

        self.num_probes = num_probes
        self.output_size = output_size
        self.probes = []

        for idx in range(self.num_probes):
            probe = nn.Linear(input_size, output_size)
            setattr(self, f"probe_{idx}", probe)  
            self.probes.append(probe)

    def forward(self, x: list, y: torch.Tensor):

        assert len(x) == self.num_probes

        log_probs, p_y, logits = [], [], []
        for x_i, probe_i in zip(x, self.probes):
            logits_i = probe_i(x_i)
            p_y_i = D.Categorical(logits=logits_i)
            log_probs_i = p_y_i.log_prob(y)
            
            logits.append(logits_i)
            p_y.append(p_y_i)
            log_probs.append(log_probs_i)

        loss = - torch.cat(log_probs, dim=0).mean()
        # import IPython; IPython.embed()
        bpd_metrics = [BitsPerDimMetric(values=l, name=f"bpd_{i}") for i, l in enumerate(log_probs)]
        acc_metrics = [AccuracyMetric(predictions=l.argmax(-1), labels=y, name=f"acc_{i}") for i, l in enumerate(log_probs)]

        outputs = SimpleNamespace(
            logits = logits,
            p_y = p_y,
            log_probs = log_probs
        )

        metrics = [
            LossMetric(loss, weight_by=y.size(0), name="total_loss")
        ]

        metrics += bpd_metrics + acc_metrics

        return loss, outputs, metrics
