import math
from vseq.modules.dropout import WordDropout

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.nn import functional as F
from torch.nn.modules.sparse import Embedding

from torchtyping import TensorType


def random(*shape):
	return np.random.randn(*shape)


def glorotize(W):
    W *= np.sqrt(6)
    W /= np.sqrt(np.sum(W.shape))
    return W


def orthogonalize(W):
    W, _, _ = np.linalg.svd(W)
    return W


def recurrent_mask(nclocks, hidden_size):
    matrix = []
    for c in range(nclocks, 0, -1):
        zero_blocks = torch.zeros(hidden_size, hidden_size * (nclocks - c))
        one_blocks = torch.ones(hidden_size, hidden_size * (c))
        matrix.append(torch.cat([zero_blocks, one_blocks], axis=1))
    mask = torch.cat(matrix, axis=0)
    return mask


def make_schedule(clock_periods, hidden_size):
    sch = []
    for c in clock_periods:
        for i in range(hidden_size):
            sch.append(c)
    return sch


# https://github.com/tomrunia/ClockworkRNN/blob/master/models/clockwork_rnn.py

class CWRNNLM(nn.Module):
    def __init__(
        self, embedding_dim, num_embeddings, hidden_size, clock_periods, delimiter_token_idx: int, full_recurrence=False, learn_state=True, first_layer=False, word_dropout: float = 0
    ):
        super().__init__()
        nclocks = len(clock_periods)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.clock_periods = clock_periods
        self.delimiter_token_idx = delimiter_token_idx
        self.word_dropout = word_dropout
        self.nclocks = nclocks
        self.full_recurrence = full_recurrence
        self.learn_state = learn_state
        self.first_layer = first_layer

        self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)
        self.mask_token_idx = num_embeddings

        self.word_dropout = WordDropout(word_dropout, mask_value=self.mask_token_idx) if word_dropout else None

        self.schedules = make_schedule(clock_periods, hidden_size)

        self.Wi = nn.Parameter(torch.randn(embedding_dim, nclocks * hidden_size))
        self.Wh = nn.Parameter(torch.randn(nclocks * hidden_size, nclocks * hidden_size))
        self.bi = nn.Parameter(torch.zeros(nclocks * hidden_size))
        self.bh = nn.Parameter(torch.zeros(nclocks * hidden_size))

        # for c in range(self.nclocks):
        #     setattr(self, f"Wi_{c}", Wi[:, c * hidden_size: (c + 1) * hidden_size])
        #     setattr(self, f"Wh_{c}", Wh[:, c * hidden_size: (c + 1) * hidden_size])
        #     setattr(self, f"bi_{c}", bi[c * hidden_size: (c + 1) * hidden_size])
        #     setattr(self, f"bh_{c}", bh[c * hidden_size: (c + 1) * hidden_size])

        self.Wo = nn.Parameter(torch.randn(nclocks * hidden_size, num_embeddings))
        self.bo = nn.Parameter(torch.zeros(num_embeddings))

        self.utri_mask = recurrent_mask(nclocks, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.Wh)
        if not self.full_recurrence:
            self.Wh = nn.Parameter(self.Wh * self.utri_mask)

        stdv = 1.0 / math.sqrt(self.hidden_size)
        for par in [self.Wi, self.Wh]:
            par.data.uniform_(-stdv, stdv)

    def forward(self, x: TensorType["B", "T", int], x_sl: TensorType["B", int]):
        y = x[:, 1:].clone().detach()  # Remove start token, batch_first=False and prevent from being masked
        x, x_sl = x[:, :-1], x_sl - 1  # Remove end token


        B, T = x.size()
        device = x.device

        Ys = torch.zeros((T, self.num_embeddings, B))
        h_prev = torch.zeros((B, self.nclocks * self.hidden_size), device=device)

        x = self.word_dropout(x) if self.word_dropout else x
        x = self.embedding(x)  # (B, T, E)
        x = x.unbind(1)

        # TODO Precompute active rows for all timesteps modulo ...
        # active_all = [[]]
        # for t in range(T):
        #     for i in range(len(self.schedules)):
        #         active_all[t].append(int(t % self.schedules[i] == 0))

        # active = torch.FloatTensor(active_all[t]).view(-1, 1)

        all_h = []
        for t in range(T):

            active = []
            for i in range(len(self.schedules)):
                active.append(int(t % self.schedules[i] == 0))

            active = torch.tensor(active, device=device).view(-1)  # 86µs

            # active = torch.tensor(active, device=device).to(int).view(-1, 1)
            # %timeit self.Wi[active]

            # active = torch.tensor(active, device=device).view(-1)
            # %timeit self.Wi[active, :]

            i_h = torch.mm(x[t], self.Wi) + self.bi  # 23µs
            h_h = torch.mm(h_prev, self.Wh) + self.bh
            # i_h = torch.mm(x[t], self.Wi[:, active]) + self.bi[active]  # 200µs
            # h_h = torch.mm(h_prev, self.Wh[active, :]) + self.bh
            h_new = i_h + h_h

            h_new = F.tanh(h_new)

            h = active.expand_as(h_new) * h_new + (1 - active).expand_as(h_prev) * h_prev

            all_h.append(h)

            h_prev = h

        h = torch.stack(all_h, dim=1)
        y = torch.matmul(h, self.Wo)

        p_x = torch.distributions.Categorical(logits=y)

        import IPython; IPython.embed(using=False)

        return (Ys,)
