from types import SimpleNamespace
from typing import Tuple, List

from vseq.evaluation.metrics import LossMetric

import torch
import torch.nn as nn
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
        self.decoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=False)

        # ATTEND PARAMETERS
        self.v = nn.Linear(att_dim, 1, bias=False)
        self.state_proj = nn.Linear(hidden_size, att_dim)
        self.value_proj = nn.Linear(memory_dim, att_dim)
        self.combine_att = nn.Linear(memory_dim + hidden_size, att_dim)

        # PREDICT PARAMETERS
        self.output = nn.Linear(att_dim, num_outputs)


    def forward(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        targets: torch.Tensor = None, # (B, T_t, D_t)
        targets_sl: torch.Tensor = None,
        teacher_forcing_frq: float = 1.0,
        arg_max_sample: bool = True,
        max_len: int = None
    ):

        if self.training:
            assert max_len is None, "The max_len argument shouldn't be provided during training."
            return self.forward_train(memory=memory,
                                      targets=targets,
                                      targets_sl=targets_sl,
                                      teacher_forcing_frq=teacher_forcing_frq,
                                      arg_max_sample=arg_max_sample)
        else:
            assert max_len is not None, "The max_len argument must be provided for evaluation."
            return self.forward_eval(memory=memory,
                                     targets=targets,
                                     targets_sl=targets_sl,
                                     arg_max_sample=arg_max_sample,
                                     max_len=max_len)

    def forward_train(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        targets: torch.Tensor = None, # (B, T_t, D_t)
        targets_sl: torch.Tensor = None, # (B,)
        teacher_forcing_frq: float = 1.0,
        arg_max_sample: bool = False,
    ):
        batch_size = memory.size(0)
        device = memory.device 
        import IPython; IPython.embed()
        # prepare sequences
        values = self.value_proj(s).transpose(0, 1) # ((T_s, B, D_att))
        
        # set up loop variables
        a_t = torch.zeros([1, batch_size, self.att_dim], device=device) # initial a-vector
        t_t = torch.full([1, batch_size], self.delimiter_token_idx, device=device) # start tokens

        # initialize LSTM states
        h_t = torch.zeros([1, batch_size, self.hidden_size], device=device) 
        c_t = torch.zeros([1, batch_size, self.hidden_size], device=device) 

        # start autoregressiv decoding
        # assumes targets has and end-token and no start-token which is reflected by targets_sl
        for t in range(targets_sl.max().item()):
            e_t = self.embedding(t_t) 
            r_t = torch.cat([a_t, e_t], dim=2) # recurrent input
            r_t, (h_t, c_t) = self.decoder_lstm(r_t, (h_t, c_t))
            import IPython; IPython.embed()
            break

    def forward_eval(
        self,
        memory: torch.Tensor, # (B, T_s, D_s)
        targets: torch.Tensor, # (B, T_t, D_t)
        targets_sl: torch.Tensor, # (B,)
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