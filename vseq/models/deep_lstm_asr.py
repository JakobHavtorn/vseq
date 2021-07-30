from types import SimpleNamespace
from vseq.data.tokenizers import word_tokenizer, char_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

from vseq.modules import Conv2dBlock, LSTMBlock
from vseq.models import MultiheadAttentionDecoder
from vseq.evaluation import LossMetric, WindowMeanMetric, ErrorRateMetric
from vseq.utils.decoding import greedy_ctc #, greedy_aed
from vseq.data.token_map import TokenMap
from vseq.data.tokens import DELIMITER_TOKEN, BLANK_TOKEN


def stack_frames(x):
    
    T, N, D = x.shape
    device = x.device

    if x.size(0) % 2 == 1:
        padding = torch.zeros([1, N, D]).to(device)
        x = torch.cat([x, padding])
        T += 1

    m = (torch.arange(T) % 2 == 0).to(device)
    x = torch.cat([x[m], x[~m]], dim=2)
    return x


class DeepLSTMASR(nn.Module):
    def __init__(self,
                 token_map: TokenMap,
                 hidden_size: int = 320,
                 layers_pr_block: int = 5,
                 dropout_prob: float = 0.0,
                 ctc_model: bool = False
    ):
        """
        Implements both the AED and CTC DeepLSTM model as described in:
        http://www.interspeech2020.org/uploadfile/pdf/Thu-2-8-4.pdf

        It is assumed that input has 80 frequency bins.

        Args:
            ...
        """
        super().__init__()
        
        self.output_size = len(token_map)
        self.token_map = token_map
        self.hidden_size = hidden_size
        self.layers_pr_block = layers_pr_block
        self.p = dropout_prob
        self.ctc_model = ctc_model

        # encoder layers: conv --> block 1 --> block 2 --> output
        self.conv2d_block = Conv2dBlock(dropout_prob=dropout_prob)
        
        self.lstm_block_1 = LSTMBlock(num_layers=layers_pr_block,
                                      input_size=hidden_size,
                                      hidden_size=hidden_size,
                                      dropout_prob=dropout_prob,
                                      return_all=True)
        stack_factor = 1 if ctc_model else 2
        self.bottleneck_1 = nn.Linear(layers_pr_block * hidden_size * stack_factor, hidden_size)
        self.act_1 = nn.ReLU6()
        self.dropout_1 = nn.Dropout(p=dropout_prob)

        self.lstm_block_2 = LSTMBlock(num_layers=layers_pr_block,
                                      input_size=hidden_size,
                                      hidden_size=hidden_size,
                                      dropout_prob=dropout_prob,
                                      return_all=True)
        self.bottleneck_2 = nn.Linear(layers_pr_block * hidden_size, hidden_size)
        self.act_2 = nn.ReLU6()
        self.dropout_2 = nn.Dropout(p=dropout_prob)

        self.bottleneck_out = nn.Linear(2 * hidden_size, hidden_size)
        self.act_out = nn.ReLU6()
        self.dropout_out = nn.Dropout(p=dropout_prob)

        # decoder module (AED or CTC)
        if ctc_model:
            self.output =  nn.Linear(hidden_size, self.output_size)
            self.ctc_loss = nn.CTCLoss(blank=token_map.token2index[BLANK_TOKEN], reduction='sum')
        else: 
            self.decoder = MultiheadAttentionDecoder(num_embeddings=self.output_size,
                                                     hidden_size=hidden_size,
                                                     num_heads=1,
                                                     num_outputs=self.output_size,
                                                     delimiter_token_idx=token_map.token2index[DELIMITER_TOKEN])


    def forward(self, x, x_sl, y, y_sl, teacher_forcing_frq=1.0, max_len=None):
        """
        Args:
            input (Tensor): Input of size NFHT with dtype == float32. Default that F = 1 and H = 80.
            seq_lens (Tensor): The sequence lengths of the input of size N with dtype == int64.
        
        Returns:
            Tensor:  Output of shape TNF with F = output_size.
            Tensor: 'seq_lens' reduced according to temporal stride.
        """

        x = x.unsqueeze(1)
        z, sl = self.conv2d_block(x, x_sl)
        N, D, H, T = z.shape
        z = z.view(N, D * H, T).permute(2, 0, 1)

        z, sl = self.lstm_block_1(z, sl)
        z = torch.cat(z, dim=2)
        if not self.ctc_model:
            z = stack_frames(z)
            sl = torch.ceil(sl / 2).to(torch.long)
        z = self.bottleneck_1(z)
        z = self.act_1(z)
        z = self.dropout_1(z)
        z1 = z

        z, sl = self.lstm_block_2(z, sl)
        z = torch.cat(z, dim=2)
        z = self.bottleneck_2(z)
        z = self.act_2(z)
        z = self.dropout_2(z)
        z2 = z

        z = torch.cat([z1, z2], dim=2)
        z = self.bottleneck_out(z)
        z = self.act_out(z)
        z = self.dropout_out(z) 

        if self.ctc_model:
            return self.ctc_decoder(z, sl, y, y_sl)
        else:
            return self.aed_decoder(z, sl, y, y_sl, teacher_forcing_frq, max_len)

    def ctc_decoder(self, z, z_sl, y, y_sl):

        logits = self.output(z)
        log_probs = logits.log_softmax(dim=2)
        hyps_raw = greedy_ctc(logits, z_sl, blank=0)
        hyps_sl = [len(h) for h in hyps_raw]
        hyps = self.token_map.decode_batch(hyps_raw, hyps_sl, "") # assumes char-level prediction
        refs = self.token_map.decode_batch(y, y_sl, "") # assumes char-level prediction
        loss = self.ctc_loss(log_probs, y, z_sl, y_sl)
        
        outputs = SimpleNamespace(
            logits=logits.transpose(0, 1),
            sl=z_sl,
            hyps=hyps,
            refs=refs
        )

        metrics = [
            LossMetric(loss, weight_by=z_sl.sum()),
            ErrorRateMetric(refs, hyps, word_tokenizer, name="wer"),
            ErrorRateMetric(refs, hyps, char_tokenizer, name="cer")
        ]

        return loss, metrics, outputs
    
    def aed_decoder(self, z, z_sl, y, y_sl, teacher_forcing_frq, max_len):
        
        memory = z.transpose(0, 1)
        memory_sl = z_sl
        outputs = self.decoder(memory=memory,
                                memory_sl=memory_sl,
                                y=y,
                                y_sl=y_sl,
                                teacher_forcing_frq=teacher_forcing_frq,
                                max_len=max_len,
                                arg_max_sample=True)

        hyps_raw = outputs.logits.argmax(dim=2)
        hyps = self.token_map.decode_batch(hyps_raw, outputs.sl_hyp, "") # assumes char-level prediction
        refs = self.token_map.decode_batch(y, y_sl, "") # assumes char-level prediction
        hyps_word = [h.strip("|") for h in hyps] # we don't care about delimiter tokens on the word-level
        refs_word = [r.strip("|") for r in refs] # we don't care about delimiter tokens on the word-level
        weight = outputs.sl_ref.sum()
        loss = - outputs.log_prob.sum() / weight

        metrics = [
            LossMetric(loss, weight_by=weight),
            WindowMeanMetric(loss),
            ErrorRateMetric(refs_word, hyps_word, word_tokenizer, name="wer"),
            ErrorRateMetric(refs, hyps, char_tokenizer, name="cer")
        ]

        return loss, metrics, outputs
