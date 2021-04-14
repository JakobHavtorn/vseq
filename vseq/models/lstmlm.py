from types import SimpleNamespace
from typing import Tuple, List
from vseq.evaluation.metrics import KLMetric, PerplexityMetric

import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

import vseq.modules
import vseq.modules.activations

from vseq.utils.operations import sequence_mask
from vseq.evaluation import Metric, LLMetric, KLMetric, PerplexityMetric, BitsPerDimMetric

from .base_module import BaseModule


class LSTMLM(BaseModule):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hidden_size: int,
        delimiter_token_idx: int,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.delimiter_token_idx = delimiter_token_idx

        # The input embedding for x. We use one embedding shared between encoder and decoder. This may be inappropriate.
        self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)
        self.mask_token_idx = num_embeddings

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
        )

        self.output = nn.Linear(hidden_size, num_embeddings)

        # TODO WordDropout as module

    def forward(
        self, x, x_sl, word_dropout_rate: float = 0.75) -> Tuple[torch.Tensor, List[Metric], SimpleNamespace]:
        """Autoregressively predict next step of input x of shape (B, T)"""

        log_prob_twise, p_x = self.reconstruct(x=x, x_sl=x_sl, word_dropout_rate=word_dropout_rate)
        log_prob = log_prob_twise.sum(1)  # (B,)
        loss = - log_prob.sum() / (x_sl - 1).sum()

        metrics = [
            LLMetric(name="loss", values=loss, weight_by=x_sl - 1),
            LLMetric(name="ll", values=log_prob),
            BitsPerDimMetric(name="bpd", values=log_prob, reduce_by=x_sl - 1),
            PerplexityMetric(name="pp", values=log_prob, reduce_by=x_sl - 1)
        ]

        outputs = SimpleNamespace(
            loss=loss,
            rec=log_prob,
            p_x=p_x  # NOTE Save 700 MB by not returning p_x
        )
        return loss, metrics, outputs

    def reconstruct(self, x: torch.Tensor, x_sl: torch.Tensor, word_dropout_rate: float = 0.75):
        """
        Computes log-likelihood for x.
        """
        # Prepare inputs (x) and targets (y)
        y = x[:, 1:].clone().detach()  # Remove start token, batch_first=False and prevent from being masked
        if self.training and word_dropout_rate > 0:
            mask = torch.bernoulli(torch.full(x.shape, word_dropout_rate)).to(bool)
            mask[:, 0] = False  # We never mask the start token - or do we?
            x = x.clone()  # We can't modify x in-place
            x[mask] = self.mask_token_idx
        e = self.embedding(x)

        # Compute log probs for p(x|z)
        e = torch.nn.utils.rnn.pack_padded_sequence(e, x_sl - 1, batch_first=True)  # x_sl - 1 --> remove end token
        h, _ = self.lstm(e)

        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        # Define output distribution
        p_logits = self.output(h)  # labo: we could use our embedding matrix here
        seq_mask = sequence_mask(x_sl - 1, dtype=float, device=p_logits.device)
        p_x = D.Categorical(logits=p_logits)
        log_prob = p_x.log_prob(y) * seq_mask
        # log_prob = torch.gather(p_logits.log_softmax(dim=-1), 2, y.unsqueeze(2)).squeeze() * seq_mask  # NOTE -600 MB

        return log_prob, p_x

    def generate(self, n_samples: int = 1, t_max: int = 100, use_mode: bool = False):
        """
        Generates a sequence by autoregressively sampling from p(x_t|x_<t).
        """
        # Setup initial loop conditions
        h_t = torch.zeros([1, n_samples, self.hidden_size])
        c_t = torch.zeros([1, n_samples, self.hidden_size])
        x_t = torch.full([1, n_samples], self.delimiter_token_idx, device=self.device)

        # Sample x from p(x|z)
        log_prob, x, x_sl = [], [], torch.zeros(n_samples, dtype=torch.int)
        seq_active = torch.ones(n_samples, dtype=torch.int)
        all_ended, t = False, 0  # Used to condition while loop
        while not all_ended and t < t_max:

            # Sample x_t from p(x_t|z, x_<t)
            e_t = self.embedding(x_t)
            _, (h_t, c_t) = self.lstm(e_t, (h_t, c_t))
            p_logits = self.output(h_t)  # (T, B, D)
            p = D.Categorical(logits=p_logits)
            x_t = p.logits.argmax(dim=-1) if use_mode else p.sample()
            log_prob_t = p.log_prob(x_t)

            # Update outputs
            x.append(x_t)
            log_prob.append(log_prob_t)

            # Update sequence length
            x_sl += seq_active
            seq_ending = (x_t[0].cpu() == self.delimiter_token_idx).to(int)  # TODO move to cpu once at end instead
            seq_active *= 1 - seq_ending

            # Update loop conditions
            t += 1
            all_ended = torch.all(1 - seq_active).item()

        seq_mask = sequence_mask(x_sl, dtype=int, device=self.device)
        x = torch.cat(x).T * seq_mask
        log_prob = torch.cat(log_prob).T * seq_mask.to(float)

        return (x, x_sl), log_prob
