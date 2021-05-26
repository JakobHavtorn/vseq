from types import SimpleNamespace
from typing import Tuple, List

from vseq.evaluation.metrics import LossMetric

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from vseq.utils.operations import sequence_mask

from .base_model import BaseModel


class LSTMEncoder(BaseModel):
    
    def __init__(self, num_embeddings: int, embedding_dim: int, hidden_size: int, num_layers: int):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The input embedding for x. We use one embedding shared between encoder and decoder. This may be inappropriate.
        self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)
        self.mask_token_idx = num_embeddings

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            bidirectional=True,
        )

        self.output = nn.Linear(hidden_size * 2, hidden_size)


    def forward(self, x: torch.Tensor, x_sl: torch.Tensor, word_dropout_rate: float = 0.0):
        
        if self.training and word_dropout_rate > 0:
            mask = torch.bernoulli(torch.full(x.shape, word_dropout_rate)).to(bool)
            x = x.clone()  # We can't modify x in-place
            x[mask] = self.mask_token_idx
        e = self.embedding(x)

        # Compute log probs for p(x|z)
        e = torch.nn.utils.rnn.pack_padded_sequence(e, x_sl, batch_first=True)
        h, _ = self.lstm(e)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        # Define output distribution
        seq_mask = sequence_mask(x_sl, dtype=torch.float32, device=h.device)
        z = F.tanh(self.output(h)) * seq_mask.unsqueeze(2)
        return z
