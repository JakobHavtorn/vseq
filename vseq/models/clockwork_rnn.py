import math

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.init as init

from torchtyping import TensorType

from vseq.evaluation.metrics import BitsPerDimMetric, LLMetric, LossMetric, PerplexityMetric
from vseq.models.base_model import BaseModel
from vseq.modules.dropout import WordDropout
from vseq.utils.operations import sequence_mask
from vseq.utils.log_likelihoods import categorical_ll


# TODO Refactor:
# TODO Create CWRNNLM class
# TODO Create CWRNNCell class
# TODO Create CWRNN


def recurrent_mask(n_clocks, hidden_size):
    matrix = []
    for c in range(n_clocks, 0, -1):
        zero_blocks = torch.zeros(hidden_size, hidden_size * (n_clocks - c))
        one_blocks = torch.ones(hidden_size, hidden_size * (c))
        matrix.append(torch.cat([zero_blocks, one_blocks], axis=1))
    mask = torch.cat(matrix, axis=0)
    return mask


class CWRNNCell(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorType["B", "timesteps", int], x_sl: TensorType["B", int]):
        pass


class CWRNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorType["B", "timesteps", int], x_sl: TensorType["B", int]):
        pass


class CWRNNLM(BaseModel):
    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        hidden_size,
        clock_periods,
        delimiter_token_idx: int,
        full_recurrence: bool = False,
        learn_state: bool = False,
        dropout_rate: float = 0.0,
        word_dropout_rate: float = 0.0,
    ):
        """Clockwork RNN as introduced in [1].

        Args:
            embedding_dim (int): Size of embedding
            num_embeddings (int): Number of tokens
            hidden_size (int): Size of the hidden state per clock rate
            clock_periods (List[int]): List of clock periods from fastest to slowest e.g. [1, 2, 4]
            delimiter_token_idx (int): Index of delimiter token
            full_recurrence (bool, optional): If True, use full recurrence. If False, condition faster units on slower
                                              units (but not slower on faster). Defaults to False.
            learn_state (bool, optional): If True, the initial state h0 is a learned parameter. Defaults to False.
            dropout_rate (float, optional): Dropout applied for hidden to output transform. Defaults to 0.0.
            word_dropout_rate (float, optional): Word dropout applied to the input. Defaults to 0.0.

        [1] A Clockwork RNN. Koutnik et. al. 2014. http://arxiv.org/abs/1402.3511
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.clock_periods = clock_periods
        self.delimiter_token_idx = delimiter_token_idx
        self.full_recurrence = full_recurrence
        self.learn_state = learn_state
        self.dropout_rate = dropout_rate
        self.word_dropout_rate = word_dropout_rate

        n_clocks = len(clock_periods)
        self.n_clocks = n_clocks

        # TODO Replace with F.linear or nn.Linear?
        self.Wi = nn.Parameter(torch.empty(embedding_dim, n_clocks * hidden_size))
        self.Wh = nn.Parameter(torch.empty(n_clocks * hidden_size, n_clocks * hidden_size))
        self.Wo = nn.Parameter(torch.empty(n_clocks * hidden_size, num_embeddings))
        self.bi = nn.Parameter(torch.zeros(n_clocks * hidden_size))
        self.bh = nn.Parameter(torch.zeros(n_clocks * hidden_size))
        self.bo = nn.Parameter(torch.zeros(num_embeddings))

        utri_mask = recurrent_mask(n_clocks, hidden_size)
        self.register_buffer("utri_mask", utri_mask)

        initial_state = torch.zeros(self.n_clocks * self.hidden_size)
        if learn_state:
            self.initial_state = nn.Parameter(initial_state)
        else:
            self.register_buffer("initial_state", initial_state)

        self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)
        self.mask_token_idx = num_embeddings

        self.word_dropout = (
            WordDropout(word_dropout_rate, mask_value=self.mask_token_idx) if word_dropout_rate else None
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.Wi.data.uniform_(-stdv, stdv)
        self.Wo.data.uniform_(-stdv, stdv)
        init.orthogonal_(self.Wh)
        if not self.full_recurrence:
            self.Wh.data *= self.utri_mask

    def forward(self, x: TensorType["B", "timesteps", int], x_sl: TensorType["B", int]):
        y = x[:, 1:].clone().detach()  # Remove start token, batch_first=False and prevent from being masked
        x, x_sl = x[:, :-1], x_sl - 1  # Remove end token

        batch_size, timesteps = x.size()

        h_prev = self.initial_state.repeat(batch_size, 1)

        x = self.word_dropout(x) if self.word_dropout else x
        x = self.embedding(x)  # (batch_size, timesteps, E)
        x = x.unbind(1)

        all_h = []
        for t in range(timesteps):

            if not self.full_recurrence:
                self.Wh.data *= self.utri_mask

            # ======================================
            # 1. Full multiplication with all parameters and element-wise multiplication with 1 or 0
            # 2. Row-wise indexing into weight matrices and biases to do only needed calculations followed by masked update of state
            # 3. Exploit contiguous property of the active blocks to do efficient indexing and state update via concatenation.
            # ======================================
            # ======================================
            # NOTE 1: 11.6Hz
            # active = []
            # for i in range(len(self.schedules)):
            #     active.append(int(t % self.schedules[i] == 0))
            # active = torch.tensor(active, device=device).view(-1)  # 86µs

            # i_h = torch.mm(x[t], self.Wi) + self.bi  # 23µs
            # h_h = torch.mm(h_prev, self.Wh) + self.bh
            # h_new = F.tanh(i_h + h_h)
            # h_prev = active.expand_as(h_new) * h_new + (1 - active).expand_as(h_prev) * h_prev
            # ======================================

            # ======================================
            # NOTE 2: 5.6 Hz
            # active = []
            # for i in range(len(self.schedules)):
            #     active.append(t % self.schedules[i] == 0)
            # active = torch.tensor(active, device=device).view(-1)  # 86µs

            # i_h = torch.mm(x[t], self.Wi[:, active]) + self.bi[active]  # 200µs
            # h_h = torch.mm(h_prev, self.Wh[:, active]) + self.bh[active]
            # h_new = torch.tanh(i_h + h_h)
            # h_prev = torch.cat([h_new, h_prev[:, ~active]], dim=1)
            # ======================================

            # ======================================
            # NOTE 3: 16.4Hz
            active = [(t % clock_period) == 0 for clock_period in self.clock_periods]
            index = sum(active) * self.hidden_size

            i_h = torch.mm(x[t], self.Wi[:, :index]) + self.bi[:index]
            h_h = torch.mm(h_prev, self.Wh[:, :index]) + self.bh[:index]
            h_new = torch.tanh(i_h + h_h)
            h_prev = torch.cat([h_new, h_prev[:, index:]], dim=1)
            # ======================================

            all_h.append(h_prev)

        h = torch.stack(all_h, dim=1)
        h = self.dropout(h) if self.dropout else h
        o = torch.matmul(h, self.Wo)

        seq_mask = sequence_mask(x_sl, dtype=float, device=o.device)
        log_prob = categorical_ll(y, o) * seq_mask
        log_prob = log_prob.sum(1)  # (batch_size,)
        loss = -log_prob.sum() / x_sl.sum()

        metrics = [
            LossMetric(loss, weight_by=log_prob.numel()),
            LLMetric(log_prob),
            BitsPerDimMetric(log_prob, reduce_by=x_sl),
            PerplexityMetric(log_prob, reduce_by=x_sl),
        ]

        outputs = SimpleNamespace(loss=loss, ll=log_prob, logits=o)
        return loss, metrics, outputs

    def generate(self, n_samples: int = 1, t_max: int = 100, use_mode: bool = False):
        pass


"""
Timing the speed of different kinds of indexing:

active = torch.cat([torch.ones(1536 //2), torch.zeros(1536//2)])
%timeit h_prev[:, active]      59.7 µs ± 2.53 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

active = active.to(int).view(1, 1536)
%timeit h_prev[active]         36.2 µs ± 1.39 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

index = 1536 // 2
%timeit h_prev[:, :index]       4.01 µs ± 134 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
"""
