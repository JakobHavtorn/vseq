from types import SimpleNamespace
from typing import Tuple, List

from vseq.evaluation.metrics import LossMetric

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from vseq.utils.operations import sequence_mask

from .base_model import BaseModel


class OPReconstruct(BaseModel):
    
    def __init__(
        self, 
        num_embeddings: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        num_outputs: int,
        embed_before_outer_product: bool = False):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.embed_before_outer_product = embed_before_outer_product

        
        #self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.embedding = nn.Linear(num_embeddings, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            bidirectional=True,
        )

        self.output = nn.Linear(hidden_size * 2, num_outputs)


    def forward(
        self,
        decoder_logits: torch.Tensor,
        aw: torch.Tensor,
        aw_sl: torch.Tensor,
        x: torch.Tensor,
        x_sl: torch.Tensor,
        tau: float = 1.0,
        hard: bool = True
    ):
        
        # samples = F.gumbel_softmax(decoder_logits, tau=tau, hard=hard)
        samples = (decoder_logits / tau).softmax(dim=2)

        if self.embed_before_outer_product:
            e = self.embedding(samples).unsqueeze(dim=2) # (B, T_z, 1, D_e)
            sm = sequence_mask(aw_sl, max_len=aw_sl.max() + 1, dtype=torch.float32, device=e.device).unsqueeze(dim=2)
            aw = (aw * sm).unsqueeze(dim=3).detach() # (B, T_z, T_x, 1)
            op = torch.matmul(aw, e).sum(dim=1) # (B, T_x, D_e)
        else:
            pass
            # DO OUTER

        # Compute log probs for p(x|z)
        op = torch.nn.utils.rnn.pack_padded_sequence(op, x_sl, batch_first=True)
        h, _ = self.lstm(op)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        # define output distribution
        p_logits = self.output(h)
        p_x = D.Categorical(logits=p_logits)
        seq_mask = sequence_mask(x_sl, dtype=torch.float32, device=h.device)
        log_prob = p_x.log_prob(x) * seq_mask

        weight = x_sl.sum()
        loss = - log_prob.sum() / weight

        acc = (x == p_logits.argmax(dim=2)).sum() / weight

        metrics = [LossMetric(loss, name="rec", weight_by=weight),
                   LossMetric(acc, name="acc", weight_by=weight)]

        return loss, metrics, p_x

