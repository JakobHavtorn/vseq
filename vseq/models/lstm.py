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
    print(
        "Module `haste_pytorch` not installed preventing use of LayerNormalized LSTM cells"
    )

from vseq.evaluation import Metric, LLMetric, PerplexityMetric, BitsPerDimMetric
from vseq.evaluation.metrics import LossMetric
from vseq.utils.operations import sequence_mask


from .base_model import BaseModel


class OutConv(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, num_classes: int, conv_layers: int = 1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.input_dim = input_dim

        self._layers = [
            nn.ReLU(),
            nn.Conv1d(input_dim, output_dim * num_classes, kernel_size=1),
        ]
        if conv_layers > 1:
            for _i in range(conv_layers - 1):
                self._layers.extend(
                    [
                        nn.ReLU(),
                        nn.Conv1d(
                            output_dim * num_classes,
                            output_dim * num_classes,
                            kernel_size=1,
                        ),
                    ]
                )
        self.layers = nn.ModuleList(self._layers)
        self.unflatten = torch.nn.Unflatten(2, (output_dim, num_classes))
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        for _layer in self.layers:
            x = _layer(x)
        x = torch.movedim(x, 1, 2)
        x = self.unflatten(x)
        x = self.log_softmax(x)
        return x


class LSTM(BaseModel):
    def __init__(
        self,
        input_size: int = 80,
        hidden_size: int = 256,
        num_classes: int = 256,
        layer_norm: bool = False,
        **lstm_kwargs,
    ):
        """Simple LSTM-based with multiple LSTM layers. Copied from LSTMLM for maximum fast dev.

        Args:
            input_size (int, optional): Dimensionality of the input space. Defaults to 80 for mel spec.
            hidden_size (int, optional): Dimensionality of the hidden space (LSTM gates).
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.lstm_kwargs = lstm_kwargs
        self.num_classes = num_classes

        self.nll_criterion = torch.nn.NLLLoss(reduction="none")

        # self._loss = nn.MSELoss()

        rnn_layer = LayerNormLSTM if layer_norm else nn.LSTM

        self.lstm = rnn_layer(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=False,
            **lstm_kwargs,
        )  # (B, T, E)->(B, T, H)

        # transform to (B, H, T)
        self.linear_out = OutConv(
            input_dim=hidden_size, output_dim=input_size, num_classes=num_classes
        )
        # (B, H, T) -> (B, C, T)

        # Transform to (B, T, C) and calc. loss

    def _get_target(self, x: TensorType["B", "T", "S"]) -> TensorType["B", "T", "S"]:
        target = (x.clone().detach() + 1) / 2  # Transform [-1, 1] to [0, 1]
        # quantize
        target = (target + 1) / 2  # Transform [-1, 1] to [0, 1]
        target = target * (self.num_classes - 1)  # Transform [0, 1] to [0, 255]
        target = target.floor().to(
            torch.long
        )  # To integer (floor because of added noise for dequantization)
        return target

    def forward(
        self,
        x: TensorType["B", "T", "C", float],
        x_sl: TensorType["B", int],
    ) -> Tuple[torch.Tensor, List[Metric], SimpleNamespace]:
        """Autoregressively predict next step of input x of shape (B, T)"""
        # Prepare inputs (x) and targets (y)
        # X should be 0 padded along 1st dimension, so pad the last dimension
        # targets = x.clone().detach()  # B, T, S
        targets = self._get_target(x)
        x = nn.functional.pad(x, (0, 0, 1, -1))  # last dim no pad, 2nd to last 1 / -1
        x = torch.movedim(x, 1, 0)  # T, B, S
        # LSTM expects (T, B, S)

        x, (_h, _c) = self.lstm(x)  # (T, B, H)
        x = torch.movedim(x, 0, 2)  # (B, T, H)
        # x = torch.movedim(x, 1, 2)

        preds = self.linear_out(x)  # B, T, C
        # print(preds.shape)
        categorical = D.Categorical(logits=preds)
        loss, ll = self.compute_loss(
            target=targets, x_sl=x_sl, output=torch.movedim(preds, 3, 1)
        )

        # preds, loss = self.reconstruct(x=x, x_sl=x_sl)

        metrics = [
            LossMetric(loss, weight_by=ll.numel()),
            LLMetric(ll),
            BitsPerDimMetric(ll, reduce_by=x_sl),
        ]

        output = SimpleNamespace(
            loss=loss, ll=ll, logits=preds, categorical=categorical, target=targets
        )
        return loss, metrics, output

    def compute_loss(
        self,
        target: TensorType["B", "T", int],
        x_sl: TensorType["B", int],
        output: TensorType["B", "T", "C", float],
    ):
        """Compute the loss as negative log-likelihood per frame, masked and mormalized according to sequence lengths.

        Args:
            target (torch.LongTensor): Input audio waveform, i.e. the target (B, T) of quantized integers.
            x_sl (torch.LongTensor): Sequence lengths of examples in the batch.
            output (torch.FloatTensor): Model reconstruction with log softmax scores per possible frame value (B, C, T).
        """
        nll = self.nll_criterion(output, target)
        mask = sequence_mask(x_sl, device=nll.device, max_len=target.size(1))
        mask = torch.unsqueeze(mask, dim=-1)
        # print("TARGET:  ", target.shape)
        # print("NLL:     ", nll.shape)
        # print("MASK:    ", mask.shape)
        nll *= mask
        nll = nll.sum(1)  # sum T
        loss = nll.nansum() / x_sl.nansum()  # sum B, normalize by sequence lengths
        return loss, -nll

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

            # Update preds
            x.append(x_t)
            log_prob.append(log_prob_t)

            # Update sequence length
            x_sl += seq_active
            seq_ending = (x_t[0].cpu() == self.delimiter_token_idx).to(
                int
            )  # TODO move to cpu once at end instead
            seq_active *= 1 - seq_ending

            # Update loop conditions
            t += 1
            all_ended = torch.all(1 - seq_active).item()

        seq_mask = sequence_mask(x_sl, dtype=int, device=self.device)
        x = torch.cat(x).T * seq_mask
        log_prob = torch.cat(log_prob).T * seq_mask.to(float)

        return (x, x_sl), log_prob
