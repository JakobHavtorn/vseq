from types import SimpleNamespace
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.distributions as D

from torchaudio.transforms import MelSpectrogram, InverseMelScale, GriffinLim
from torchtyping import TensorType


try:
    from haste_pytorch import LayerNormLSTM
except ModuleNotFoundError:
    print("Module `haste_pytorch` not installed preventing use of LayerNormalized LSTM cells")

from vseq.evaluation import Metric, LLMetric, PerplexityMetric, BitsPerDimMetric
from vseq.evaluation.metrics import LossMetric
from vseq.utils.operations import sequence_mask

from .base_model import BaseModel


class LSTM(BaseModel):
    def __init__(
        self,
        # num_embeddings: int,
        embedding_dim: int = 80,
        hidden_size: int = 256,
        num_layers: int = 1,
        layer_norm: bool = False,
        **lstm_kwargs,
    ):
        """Simple LSTM-based with multiple LSTM layers. Copied from LSTMLM for maximum fast dev.

        Args:
            # num_embeddings (int): Number of input tokens.
            embedding_dim (int, optional): Dimensionality of the embedding space. Defaults to 80 for mel spec.
            hidden_size (int, optional): Dimensionality of the hidden space (LSTM gates).
            num_layers (int, optional): Number of LSTM layers. Defaults to 1 (c.f. Bowman)
        """
        super().__init__()

        # self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.lstm_kwargs = lstm_kwargs

        self._loss = nn.MSELoss()

        # self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)

        rnn_layer = LayerNormLSTM if layer_norm else nn.LSTM

        self.lstm = rnn_layer(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=False,
            **lstm_kwargs
        ) # (B, T, E)->(B, T, H)

        # transform to (B, C, T)

        self.output = nn.Linear(hidden_size, embedding_dim) # (B, C, T) -> (B, E, T)

        # Transform to (B, T, E)


        # TODO WordDropout as module

    def reconstruct(self, x: TensorType["B", "C", "T", float], x_sl: TensorType["B", int]):
        """
        Computes log-likelihood for x. Set the word_dropout to 0 as default to make training easy.
        """
        # Prepare inputs (x) and targets (y)
        # X should be 0 padded along 1st dimension
        targets = x.clone().detach()   # B, C, T
        x = nn.functional.pad(x, (1, -1))
        
        x = torch.movedim(x, 2, 0) # T,B,C

        # Compute log probs for p(x|z)
        # h, _ = self.lstm(x)
        # LSTM expects (T, B, C)
        x, (_h,_c) = self.lstm(x) #  (T, B, H)

        x = torch.movedim(x, 0, 1) # (B, T, H)

        # Define output distribution
        preds = self.output(x) # B, T, C
        preds = torch.movedim(preds, 1, 2)
        seq_mask = sequence_mask(x_sl - 1, dtype=float, device=preds.device)
        mse = self._loss(preds, targets) * seq_mask
        # p_x = D.Categorical(logits=p_logits)
        # log_prob = p_x.log_prob(y) * seq_mask
        # log_prob = torch.gather(p_logits.log_softmax(dim=-1), 2, y.unsqueeze(2)).squeeze() * seq_mask  # NOTE -600 MB

        return preds, mse

    def forward(
        self, x: TensorType["B", "C", "T", float], x_sl: TensorType["B", int], 
        ) -> Tuple[torch.Tensor, List[Metric], SimpleNamespace]:
        """Autoregressively predict next step of input x of shape (B, T)"""

        # TODO: finish doing this 

        preds, loss = self.reconstruct(x=x, x_sl=x_sl)

        metrics = [
            LossMetric(loss)#, weight_by=log_prob.numel()),
            # LLMetric(log_prob),
            # BitsPerDimMetric(log_prob, reduce_by=x_sl - 1),
            # PerplexityMetric(log_prob, reduce_by=x_sl - 1)
        ]

        outputs = SimpleNamespace(
            loss=loss,
            # ll=log_prob,
            # p_x=p_x  # NOTE Save 700 MB by not returning p_x
        )
        return loss, metrics, outputs

    def generate(self, n_samples: int = 1, t_max: int = 100, use_mode: bool = False):
        """
        Generates a sequence by autoregressively sampling from p(x_t|x_<t).
        """
        # Setup initial loop conditions
        h_t = torch.zeros([1, n_samples, self.hidden_size])
        c_t = torch.zeros([1, n_samples, self.hidden_size])
        x_t = torch.full([1, n_samples], 0, device=self.device)

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
