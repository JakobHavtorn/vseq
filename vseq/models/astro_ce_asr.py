from types import SimpleNamespace

from vseq.data.tokenizers import word_tokenizer, char_tokenizer

import torch
import torch.nn as nn

from vseq.modules import LSTMBlock
from vseq.evaluation import LossMetric, WindowMeanMetric, ErrorRateMetric
from vseq.utils.decoding import greedy_standard 
from vseq.data.token_map import TokenMap
from vseq.data.tokens import BLANK_TOKEN

from vseq.utils.operations import sequence_mask


class AstroCEASR(nn.Module):
    def __init__(self,
                 token_map: TokenMap,
                 in_channels: int = 1,
                 kernel_size: int = 500,
                 stride: int = 500,
                 hidden_size: int = 256,
                 lstm_layers: int = 2,
                 dropout_prob: float = 0.0
    ):

        super().__init__()
        
        self.output_size = len(token_map)
        self.token_map = token_map
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.p = dropout_prob

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=stride
        )

        self.conv1d_act = nn.ReLU6()
        
        self.lstm_block = LSTMBlock(num_layers=lstm_layers,
                                    input_size=hidden_size,
                                    hidden_size=hidden_size,
                                    dropout_prob=dropout_prob,
                                    return_all=False)

        self.output =  nn.Linear(hidden_size, self.output_size)
        self.ctc_loss = nn.CTCLoss(blank=token_map.token2index[BLANK_TOKEN], reduction='sum')


    def forward(self, x, x_sl, y, y_sl):
        
        if x.ndim == 2:
            x = x.unsqueeze(dim=1)
        z = self.conv1d(x)
        z = self.conv1d_act(z)
        z = z.permute(2, 0, 1)
        z_sl = x_sl // self.stride
        z, z_sl = self.lstm_block(z, z_sl)
        logits = self.output(z)

        p_y = logits.log_softmax(dim=2).transpose(0, 1)
        
        hyps_raw = greedy_standard(logits, z_sl, blank=0)
        hyps_sl = [len(h) for h in hyps_raw]
        hyps = self.token_map.decode_batch(hyps_raw, hyps_sl, "") # assumes char-level prediction
        refs = self.token_map.decode_batch(y, y_sl, "") # assumes char-level prediction

        log_prob = p_y.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        tm = sequence_mask(y_sl, max_len=log_prob.size(1), dtype=torch.float32, device=y.device)
        log_prob = log_prob * tm
        
        loss = - log_prob.sum() / y_sl.sum()
        
        outputs = SimpleNamespace(
            logits=logits.transpose(0, 1),
            sl=z_sl,
            hyps=hyps,
            refs=refs
        )

        metrics = [
            LossMetric(loss, weight_by=z_sl.sum()),
            WindowMeanMetric(loss),
            ErrorRateMetric(refs, hyps, word_tokenizer, name="wer"),
            ErrorRateMetric(refs, hyps, char_tokenizer, name="cer")
        ]

        return loss, metrics, outputs
