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
        probe_model: nn.Module,
        probe_args: dict
    ):
        super().__init__()

        self.num_probes = num_probes
        self.probes = []

        for idx in range(self.num_probes):
            probe = probe_model(**probe_args, probe_idx=idx) # nn.Linear(input_size, output_size)
            setattr(self, f"probe_{idx}", probe)  
            self.probes.append(probe)

    def forward(self, x: list, y: torch.Tensor):

        assert len(x) == self.num_probes

        loss, outputs, metrics = [], [], []
        for x_i, probe_i in zip(x, self.probes):
            l, o, m = probe_i(x_i, y)
            loss.append(l)
            outputs.append(o)
            metrics += m

        loss = sum(loss)

        metrics = [LossMetric(loss, weight_by=y.size(0), name="total_loss")] + metrics

        return loss, outputs, metrics
