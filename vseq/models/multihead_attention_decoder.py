from types import SimpleNamespace
from typing import Tuple, List
from random import random
from time import time
import math

from vseq.evaluation.metrics import LossMetric, RunningMeanMetric

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from vseq.utils.operations import sequence_mask

from .base_model import BaseModel

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]

class MultiheadAttentionDecoder(BaseModel):
    
    def __init__(
        self,
        num_embeddings: int,
        hidden_size: int,
        num_heads: int,
        num_outputs: int,
        delimiter_token_idx: int
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_outputs = num_outputs
        self.delimiter_token_idx = delimiter_token_idx

        # EMBEDDING PARAMETERS
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=False)
        self.decoder_att = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.output = nn.Linear(hidden_size, num_outputs)

        # self.combine_input = nn.Linear(embedding_dim + hidden_size, hidden_size)
        # self.decoder_lstm = nn.LSTM(input_size=(embedding_dim + att_dim), hidden_size=hidden_size, bidirectional=False)

        # # ATTEND PARAMETERS
        # self.v = nn.Linear(att_dim, 1, bias=False)
        # self.state_proj = nn.Linear(hidden_size, att_dim)
        # self.value_proj = nn.Linear(memory_dim, att_dim)
        # self.combine_att = nn.Linear(memory_dim + hidden_size, att_dim)


    def forward(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        x: torch.Tensor = None, # (B, T_t)
        x_sl: torch.Tensor = None, # (B,)
        teacher_forcing_frq: float = 1.0,
        arg_max_sample: bool = True,
        max_len: int = None
    ):

        if self.training:
            assert max_len is None, "The max_len argument shouldn't be provided during training."
            return self.predict(memory=memory,
                                       memory_sl=memory_sl,
                                       x=x,
                                       x_sl=x_sl,
                                       teacher_forcing_frq=teacher_forcing_frq,
                                       arg_max_sample=arg_max_sample)
        else:
            assert max_len is not None, "The max_len argument must be provided for evaluation."
            return self.predict(memory=memory,
                                       memory_sl=memory_sl,
                                       x=x,
                                       x_sl=x_sl,
                                       teacher_forcing_frq=teacher_forcing_frq,
                                       arg_max_sample=arg_max_sample)

    def forward_train(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        x: torch.Tensor = None, # (B, T_t)
        x_sl: torch.Tensor = None, # (B,)
        teacher_forcing_frq: float = 1.0,
        arg_max_sample: bool = False
    ):
        batch_size = memory.size(0)
        device = memory.device
        
        y = x[:, 1:].clone().detach()
        sl = x_sl - 1 # this will implicitly remove end-token from x when packed
        kpm = sequence_mask(memory_sl, dtype=bool, device=device, invert=True)
        tm = sequence_mask(sl, dtype=torch.float32, device=device)

        k = memory.transpose(0, 1) # (T_s, B, D_s)
        v = k
        
        # emb = self.embedding(x) # (B, T_t, D_h)
        # emb = torch.nn.utils.rnn.pack_padded_sequence(emb, sl, batch_first=True)
        # q, _ = self.decoder_lstm(emb)
        # q, _ = torch.nn.utils.rnn.pad_packed_sequence(q, batch_first=False) # (T_t, B, D_h)

        emb = self.embedding(x).transpose(0, 1) # (T_t, B, D_h)
        q, _ = self.decoder_lstm(emb)
        q = q[:-1] * tm.T.unsqueeze(2)

        a, aw = self.decoder_att(query=q, key=k, value=v, key_padding_mask=kpm)
        logits = self.output(a).transpose(0, 1)

        p_y = D.Categorical(logits=logits)
        log_prob = p_y.log_prob(y) * tm

        weight = sl.sum()
        loss = - log_prob.sum() / weight

        metrics = [
            LossMetric(loss, weight_by=weight)
        ]

        return loss, logits, log_prob, metrics, p_y, aw


    def forward_eval(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        x: torch.Tensor, # (B, T_t)
        x_sl: torch.Tensor, # (B,)
        arg_max_sample: bool = False,
        max_len: int = None
    ):
        pass

    def predict(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        x: torch.Tensor = None, # (B, T_t)
        x_sl: torch.Tensor = None, # (B,)
        teacher_forcing_frq: float = 1.0,
        arg_max_sample: bool = False
    ):
        batch_size = memory.size(0)
        device = memory.device
        T = x_sl.max().item() - 1
        
        y = x[:, 1:].clone().detach()
        sl = x_sl - 1 # this will implicitly remove end-token from x when packed
        kpm = sequence_mask(memory_sl, dtype=bool, device=device, invert=True)
        tm = sequence_mask(sl, dtype=torch.float32, device=device)

        k = memory.transpose(0, 1) # (T_s, B, D_s)
        v = k
        
        state = None
        logits = None
        q_, aw_, logits_ = [], [], []
        for t in range(T):
            
            if logits is None:
                idxs = x[:,t:t+1] # (B, 1)
            else:
                idxs = logits.argmax(dim=2) # (B, 1)

            emb = self.embedding(idxs).transpose(0, 1) # (1, B, D_h)
            q, state = self.decoder_lstm(emb, state)
            q_.append(q) #q[:-1] * tm.T.unsqueeze(2)

            a, aw = self.decoder_att(query=q, key=k, value=v, key_padding_mask=kpm)
            aw_.append(aw)
            logits = self.output(a).transpose(0, 1)
            logits_.append(logits)

        logits = torch.cat(logits_, dim=1)
        aw = torch.cat(aw_, dim=1)

        p_y = D.Categorical(logits=logits)
        log_prob = p_y.log_prob(y) * tm

        weight = sl.sum()
        loss = - log_prob.sum() / weight

        metrics = [
            LossMetric(loss, weight_by=weight)
        ]

        return loss, logits, log_prob, metrics, p_y, aw