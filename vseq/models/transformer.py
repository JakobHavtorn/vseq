import math

from types import SimpleNamespace

import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer

from vseq.evaluation.metrics import BitsPerDimMetric, LLMetric, LossMetric, PerplexityMetric
from vseq.models.base_model import BaseModel
from vseq.utils.log_likelihoods import categorical_ll


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerLM(BaseModel):
    def __init__(self, num_embeddings, embedding_dim, num_heads, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, hidden_size, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Linear(embedding_dim, num_embeddings)

        self.reset_parameters()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def reset_parameters(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_sl):
        y = x[:, 1:].clone().detach()
        x, x_sl = x[:, :-1], x_sl - 1

        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        x = self.pos_encoder(x)

        x_mask = self.generate_square_subsequent_mask(x.size(0)).to(x.device)

        logits = self.transformer_encoder(x, x_mask)
        logits = self.decoder(logits)

        log_prob = categorical_ll(y, logits)
        log_prob = log_prob.sum(1)

        loss = -log_prob.sum() / x_sl.sum()

        metrics = [
            LossMetric(loss, weight_by=log_prob.numel()),
            LLMetric(log_prob),
            BitsPerDimMetric(log_prob, reduce_by=x_sl),
            PerplexityMetric(log_prob, reduce_by=x_sl),
        ]

        outputs = SimpleNamespace(loss=loss, ll=log_prob, logits=logits)
        return loss, metrics, outputs
