from types import SimpleNamespace
from typing import Tuple, List
from random import random
from time import time

from vseq.evaluation.metrics import LossMetric, RunningMeanMetric

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from vseq.utils.operations import sequence_mask

from .base_model import BaseModel

class BahdanauAttentionOnepassDecoder(BaseModel):
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        memory_dim: int,
        att_dim: int,
        hidden_size: int,
        num_outputs: int,
        delimiter_token_idx: int
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.memory_dim = memory_dim
        self.att_dim = att_dim
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.delimiter_token_idx = delimiter_token_idx

        # RECURRENT PARAMETERS
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.decoder_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=False)

        # ATTEND PARAMETERS
        self.v = nn.Linear(att_dim, 1, bias=False)
        self.q_proj = nn.Linear(hidden_size, att_dim)
        self.v_proj = nn.Linear(memory_dim, att_dim)
        self.combine_att = nn.Linear(memory_dim + hidden_size, att_dim)

        # PREDICT PARAMETERS
        self.output = nn.Linear(att_dim, num_outputs)

        self.t1, self.t2, self.t3 = 0, 0, 0


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
            return self.forward_train(memory=memory,
                                      memory_sl=memory_sl,
                                      x=x,
                                      x_sl=x_sl,
                                      teacher_forcing_frq=teacher_forcing_frq,
                                      arg_max_sample=arg_max_sample)
        else:
            assert max_len is not None, "The max_len argument must be provided for evaluation."
            return self.forward_eval(memory=memory,
                                     memory_sl=memory_sl,
                                     x=x,
                                     x_sl=x_sl,
                                     arg_max_sample=arg_max_sample,
                                     max_len=max_len)

    def forward_train(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        x: torch.Tensor = None, # (B, T_t)
        x_sl: torch.Tensor = None, # (B,)
        teacher_forcing_frq: float = 1.0,
        arg_max_sample: bool = False,
    ):
        batch_size = memory.size(0)
        device = memory.device
        
        y = x[:, 1:].clone().detach()
        sl = x_sl - 1 # this will implicitly remove end-token from x when packed
        tm = sequence_mask(sl, dtype=float, device=device)

        # k = memory.transpose(0, 1) # (T_s, B, D_s)
        # v = k

        # prepare sequences
        # values = self.value_proj(memory).transpose(0, 1) # ((T_s, B, D_att))
        # values_seq_mask = sequence_mask(memory_sl, device=device, invert=True) # (B, T)
        # values_seq_mask = values_seq_mask.T.unsqueeze(-1) # (T_s, B, 1)
        
        emb = self.embedding(x) # (B, T_t, D_h)
        emb = torch.nn.utils.rnn.pack_padded_sequence(emb, sl, batch_first=True)
        s, _ = self.decoder_lstm(emb)
        s, _ = torch.nn.utils.rnn.pad_packed_sequence(s, batch_first=False) # (B, T_t, D_h)
        
        q = self.q_proj(s).unsqueeze(0) # # (1, T_t, B, D_att)
        v = self.v_proj(memory).transpose(0, 1).unsqueeze(1) # ((T_s, 1, B, D_att))
        mask = sequence_mask(memory_sl, device=device, dtype=bool, invert=True).T.unsqueeze(1) # (T_s, 1, B)
        e = self.v(torch.tanh(q + v)).squeeze() # ((T_s, T_t, B))
        aw = e.softmax(dim=0).T # (B, T_t, T_s)
        c = torch.matmul(aw, memory) # (B, T_t, D_s)

        a = torch.tanh(self.combine_att(torch.cat([c, s.transpose(0, 1)], dim=-1))) # (B, T_t, D_att)
        logits = self.output(a) # (B, T_t, D_out)

        p_y = D.Categorical(logits=logits)
        log_prob = p_y.log_prob(y) * tm

        weight = sl.sum()
        loss = - log_prob.sum() / weight

        metrics = [
            LossMetric(loss, weight_by=weight)
        ]

        return loss, logits, log_prob, metrics, aw

    def forward_eval(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        y: torch.Tensor, # (B, T_t)
        y_sl: torch.Tensor, # (B,)
        arg_max_sample: bool = False,
        max_len: int = None
    ):
        pass
    def predict(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        arg_max_sample: bool = False,
        max_len: int = None
    ):
        pass