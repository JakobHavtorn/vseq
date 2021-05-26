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

class BahdanauAttentionDecoder(BaseModel):
    
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
        self.combine_input = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=(embedding_dim + att_dim), hidden_size=hidden_size, bidirectional=False)

        # ATTEND PARAMETERS
        self.v = nn.Linear(att_dim, 1, bias=False)
        self.state_proj = nn.Linear(hidden_size, att_dim)
        self.value_proj = nn.Linear(memory_dim, att_dim)
        self.combine_att = nn.Linear(memory_dim + hidden_size, att_dim)

        # PREDICT PARAMETERS
        self.output = nn.Linear(att_dim, num_outputs)

        self.t1, self.t2, self.t3 = 0, 0, 0


    def forward(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        y: torch.Tensor = None, # (B, T_t)
        y_sl: torch.Tensor = None, # (B,)
        teacher_forcing_frq: float = 1.0,
        arg_max_sample: bool = True,
        max_len: int = None
    ):

        if self.training:
            assert max_len is None, "The max_len argument shouldn't be provided during training."
            return self.forward_train(memory=memory,
                                      memory_sl=memory_sl,
                                      y=y,
                                      y_sl=y_sl,
                                      teacher_forcing_frq=teacher_forcing_frq,
                                      arg_max_sample=arg_max_sample)
        else:
            assert max_len is not None, "The max_len argument must be provided for evaluation."
            return self.forward_eval(memory=memory,
                                     memory_sl=memory_sl,
                                     y=y,
                                     y_sl=y_sl,
                                     arg_max_sample=arg_max_sample,
                                     max_len=max_len)

    def forward_train(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        memory_sl: torch.Tensor, # (B,)
        y: torch.Tensor = None, # (B, T_t)
        y_sl: torch.Tensor = None, # (B,)
        teacher_forcing_frq: float = 1.0,
        arg_max_sample: bool = False,
    ):
        batch_size = memory.size(0)
        device = memory.device 

        # prepare sequences
        values = self.value_proj(memory).transpose(0, 1) # ((T_s, B, D_att))
        values_seq_mask = sequence_mask(memory_sl, device=device, invert=True) # (B, T)
        values_seq_mask = values_seq_mask.T.unsqueeze(-1) # (T_s, B, 1)
        
        # set up loop variables
        a_t = torch.zeros([1, batch_size, self.att_dim], device=device) # initial a-vector
        pred_t = torch.full([1, batch_size], self.delimiter_token_idx, device=device) # start/prediction tokens

        # initialize LSTM states
        h_t = torch.zeros([1, batch_size, self.hidden_size], device=device) 
        c_t = torch.zeros([1, batch_size, self.hidden_size], device=device) 

        # start autoregressiv decoding
        # assumes y has and end-token and no start-token which is reflected by y_sl
        aw, logits, preds = [], [], []
        for t in range(y_sl.max().item()):
            
            # recurrent step
            if t == 0:
                emb_t = self.embedding(pred_t)
            elif teacher_forcing_frq < 1.0 and random() > teacher_forcing_frq:
                emb_t = self.embedding(pred_t) # sample from output (for now arg max)
            else:
                emb_t = self.embedding(y.T[t - 1:t]) # use teacher forcing
            r_t = torch.cat([a_t, emb_t], dim=2) # recurrent input
            _, (h_t, c_t) = self.decoder_lstm(r_t, (h_t, c_t))

            # compute attention weights
            q_t = self.state_proj(h_t) # (1, B, D_att)
            e_t = self.v(F.tanh(q_t + values)).masked_fill(values_seq_mask, -1e10) # (T_s, B, 1)
            aw_t = e_t.softmax(dim=0) # (T_s, B, 1)
            aw.append(aw_t.permute(1, 2, 0))

            # compute attention vector
            #import IPython; IPython.embed()
            cxt_t = (aw_t * memory.transpose(0, 1)).sum(dim=0, keepdim=True) # (1, B, D_mem)
            # TODO implement as matrix multiplication
            a_t = torch.tanh(self.combine_att(torch.cat([cxt_t, h_t], dim=2))) # (1, B, D_att)

            # make prediction
            logits_t = self.output(a_t) # (1, B, D_out)
            pred_t = logits_t.argmax(dim=2) # (1, B)
            logits.append(logits_t.transpose(0, 1)) 
            preds.append(pred_t.T)
            
        aw = torch.cat(aw, dim=1) # (B, T_t, T_s)
        logits = torch.cat(logits, dim=1) # (B, T_t, D_out)
        preds = torch.cat(preds, dim=1) # (B, T_t)

        p_y = D.Categorical(logits=logits)
        seq_mask = sequence_mask(y_sl, dtype=float, device=device)
        log_prob = p_y.log_prob(y) * seq_mask

        weight = y_sl.sum()
        loss = - log_prob.sum() / weight

        acc = (preds == y).to(float).masked_fill(torch.logical_not(seq_mask), 0.0) / weight

        metrics = [
            LossMetric(loss, weight_by=weight),
            RunningMeanMetric(acc, "acc", None, weight_by=weight)
        ]

        return loss, preds, log_prob, metrics, aw


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