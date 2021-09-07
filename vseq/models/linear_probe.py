from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.distributions as D

from vseq.evaluation import LossMetric, AccuracyMetric

from .base_model import BaseModel


class LinearProbe(BaseModel):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        probe_idx: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.probe_idx = probe_idx
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: list, y: torch.Tensor):

        logits = self.linear(x)
        p_y = D.Categorical(logits=logits)
        log_probs = p_y.log_prob(y)

        loss = - log_probs.mean()

        outputs = SimpleNamespace(
            logits = logits,
            p_y = p_y,
            log_probs = log_probs
        )

        metrics = [
            LossMetric(values=loss, name=f"loss_{self.probe_idx}"),
            AccuracyMetric(predictions=logits.argmax(-1), labels=y, name=f"acc_{self.probe_idx}")
        ]

        return loss, outputs, metrics
