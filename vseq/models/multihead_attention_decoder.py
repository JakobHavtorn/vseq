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
        self.decoder_lstm = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, bidirectional=False)
        self.decoder_att = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.combine = nn.Linear(hidden_size * 2, hidden_size)
        self.act_combine = nn.ReLU6()
        self.output = nn.Linear(hidden_size, num_outputs)

    def forward(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        y: torch.Tensor = None, # (B, T_t)
        y_sl: torch.Tensor = None, # (B,)
        teacher_forcing_frq: float = 1.0,
        arg_max_sample: bool = True,
        tau: float = 1.0,
        max_len: int = None
    ):

        if self.training:
            if teacher_forcing_frq == 1.0:
                assert max_len is None, "The max_len argument shouldn't be provided during training."
                return self.forward_onepass(memory=memory,
                                            memory_sl=memory_sl,
                                            y=y,
                                            y_sl=y_sl)
            else:
                return self.forward_autoregressive(memory=memory,
                                                   memory_sl=memory_sl,
                                                   y=y,
                                                   y_sl=y_sl,
                                                   teacher_forcing_frq=teacher_forcing_frq,
                                                   arg_max_sample=arg_max_sample,
                                                   tau=tau)
        else:
            assert max_len is not None, "The max_len argument must be provided for evaluation."
            return self.forward_autoregressive(memory=memory,
                                               memory_sl=memory_sl,
                                               y=y,
                                               y_sl=y_sl,
                                               teacher_forcing_frq=0.0,
                                               arg_max_sample=arg_max_sample,
                                               tau=tau,
                                               max_len=max_len)

    def forward_onepass(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        y: torch.Tensor = None, # (B, T_t)
        y_sl: torch.Tensor = None, # (B,)
    ):

        device = memory.device
        
        targets = y[:, 1:].clone().detach() # (B, T_t)
        inputs = y[:, :-1].clone().detach() # (B, T_t)
        sl = y_sl - 1 # this will implicitly remove end-token from y when packed
        kpm = sequence_mask(memory_sl, dtype=bool, device=device, invert=True) # key padding mask
        tm = sequence_mask(sl, dtype=torch.float32, device=device) # target mask

        k = memory.transpose(0, 1) # (T_s, B, D_s)
        v = k # (T_s, B, D_s)

        emb = self.embedding(inputs).transpose(0, 1) # (T_t, B, D_h)
        q, _ = self.decoder_lstm(emb) # (T_t, B, D_h)
        q = q * tm.T.unsqueeze(2) # (T_t, B, D_h)

        a, aw = self.decoder_att(query=q, key=k, value=v, key_padding_mask=kpm) # (T_t, B, D_h)
        c = self.combine(torch.cat([q, a], dim=-1)) # (T_t, B, D_h)
        c = self.act_combine(c) # (T_t, B, D_h)
        logits = self.output(c).transpose(0, 1) # (B, T_t, D_o)

        p_y = logits - logits.logsumexp(dim=-1, keepdim=True)
        log_prob = p_y.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        log_prob = log_prob * tm

        outputs = SimpleNamespace(
            att_weights=aw,
            logits=logits,
            log_prob=log_prob,
            sl=sl,
            samples=None,
            teacher=None,
        )

        return outputs


    def forward_autoregressive(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        y: torch.Tensor = None, # (B, T_t)
        y_sl: torch.Tensor = None, # (B,)
        teacher_forcing_frq: float = 1.0,
        arg_max_sample: bool = False,
        tau: float = 1.0,
        max_len: int = None
    ):
        batch_size = memory.size(0)
        device = memory.device
        
        #y = x[:, 1:].clone().detach()
        targets = y[:, 1:].clone().detach() # (B, T_t)
        inputs = y[:, :-1].clone().detach() # (B, T_t)
        sl = y_sl - 1 # this will implicitly remove end-token from y when packed
        kpm = sequence_mask(memory_sl, dtype=bool, device=device, invert=True)
        # tm = sequence_mask(sl, max_len=max_len, dtype=torch.float32, device=device)
        T = max_len or sl.max().item()

        k = memory.transpose(0, 1) # (T_s, B, D_s)
        v = k
        
        state = None
        q_, aw_, logits_, samples_, teacher = [], [], [], [], []
        at_end = torch.zeros([batch_size], dtype=torch.bool, device=device)
        logits_sl = torch.zeros([batch_size], dtype=torch.int64, device=device)
        c = torch.zeros([1, batch_size, self.hidden_size], dtype=torch.float32, device=device)
        for t in range(T):
            
            if t == 0:
                idxs = torch.full([batch_size, 1], self.delimiter_token_idx, device=device) # (B, 1)
            elif random() < teacher_forcing_frq:
                idxs = inputs[:, t:t+1]
                teacher.append(True)
            else:
                idxs = samples.detach().argmax(dim=2)
                teacher.append(False)
    
            emb = self.embedding(idxs).transpose(0, 1) # (1, B, D_h)
            rnn_in = torch.cat([c.detach(), emb], dim=-1)
            q, state = self.decoder_lstm(rnn_in, state)
            q_.append(q) #q[:-1] * tm.T.unsqueeze(2)

            a, aw = self.decoder_att(query=q, key=k, value=v, key_padding_mask=kpm) # (1, B, D_h)
            # TODO: Should we maybe add an activation here?
            aw_.append(aw)
            c = self.combine(torch.cat([q, a], dim=-1)) # (1, B, D_h)
            c = self.act_combine(c) # (1, B, D_h)
            logits = self.output(c).transpose(0, 1)
            logits_.append(logits)

            samples = logits if arg_max_sample else F.gumbel_softmax(logits, tau=tau)
            samples_.append(samples)

            # logits_sl += torch.logical_not(at_end).to(torch.int64)
            # end_pred = (samples.argmax(dim=2) == self.delimiter_token_idx).squeeze()
            # at_end = torch.logical_or(at_end, end_pred)

        T = len(logits_)
        logits = torch.cat(logits_, dim=1)
        samples = torch.cat(samples_, dim=1)
        aw = torch.cat(aw_, dim=1)

        p_y = logits - logits.logsumexp(dim=-1, keepdim=True)
        if self.training:
            log_prob = p_y.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            tm = sequence_mask(sl, max_len=log_prob.size(1), dtype=torch.float32, device=device)
            log_prob = log_prob * tm
        else:
            log_prob = torch.zeros_like(y, dtype=torch.float) # temp solution - loss not informative in val models

        outputs = SimpleNamespace(
            att_weights=aw,
            logits=logits,
            log_prob=log_prob,
            sl_ref=sl, # replace none by inferred seq len
            sl_hyp=logits_sl.cpu(),
            samples=samples,
            teacher=teacher
        )

        return outputs