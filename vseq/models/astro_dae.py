from types import SimpleNamespace

from vseq.data.tokenizers import word_tokenizer, char_tokenizer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vseq.modules import LSTMBlock
from vseq.data.transforms import Compose, MuLawEncode, MuLawDecode, Quantize, Scale
from vseq.evaluation import LossMetric, WindowMeanMetric, ErrorRateMetric
from vseq.data.token_map import TokenMap
from vseq.utils.operations import sequence_mask


class AstroDAE(nn.Module):
    def __init__(
        self,
        token_map: TokenMap,
        in_channels: int = 1,
        kernels: list = [10, 10, 5],
        hidden_size: int = 256,
        lstm_layers: int = 2,
        bits: int = 8
    ):

        super().__init__()
        
        self.token_map = token_map
        self.in_channels = in_channels
        self.kernels = kernels
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.bits = bits

        self.stride = np.prod(kernels)
        self.latent_size = len(token_map)

        self.encoder_conv = []
        self.decoder_conv = []

        # encoder layers
        for idx, k in enumerate(kernels):
            in_conv = in_channels if idx == 0 else hidden_size
            conv1d_layer = nn.Conv1d(
                in_channels=in_conv,
                out_channels=hidden_size,
                kernel_size=k,
                stride=k
            )
            act = nn.ReLU6()
            setattr(self, f"enc_conv1d_{idx}", conv1d_layer)
            setattr(self, f"enc_act_{idx}", act)
            self.encoder_conv.append((conv1d_layer, act))

        self.enc_lstm_block = LSTMBlock(
            num_layers=lstm_layers,
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout_prob=0.0,
            return_all=False
        )
        self.enc_linear =  nn.Linear(hidden_size, self.latent_size)

        # decoder layers
        self.dec_linear =  nn.Linear(self.latent_size, hidden_size)
        self.dec_linear_act = nn.ReLU6()
        self.dec_lstm_block = LSTMBlock(
            num_layers=lstm_layers,
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout_prob=0.0,
            return_all=False
        )

        num_convs = len(kernels) - 1
        for idx, k in enumerate(reversed(kernels)):
            out_conv = hidden_size if idx == num_convs else 2 ** self.bits
            conv1d_layer = nn.ConvTranspose1d(
                in_channels=hidden_size,
                out_channels=out_conv,
                kernel_size=k,
                stride=k
            )
            act = nn.Identity() if idx == num_convs else nn.ReLU6()
            setattr(self, f"dec_conv1d_{idx}", conv1d_layer)
            setattr(self, f"dec_act_{idx}", act)
            self.decoder_conv.append((conv1d_layer, act))

        # target encode
        mu_law_encoder = MuLawEncode()
        quantizer = Quantize(bits=bits)
        self.target_encode = Compose(mu_law_encoder, quantizer)

        # target decode
        scale = Scale(min_val=0, max_val=(2 ** bits) - 1)
        mu_law_decoder = MuLawDecode()
        self.target_decode = Compose(scale, mu_law_decoder)



    def set_transform_device(self, device):

        b = self.target_encode.transforms[1].boundaries
        self.target_encode.transforms[1].boundaries = b.to(device)


    def inference(self, x, x_sl, tau=1.0):
        
        if x.ndim == 2:
            x = x.unsqueeze(dim=1)
        
        # z = x
        # for conv, act in self.encoder_conv:
        #     z = conv(z)
        #     z = act(z)
        z = self.enc_conv1d_0(x)
        z = self.enc_act_0(z)

        z = self.enc_conv1d_1(z)
        z = self.enc_act_1(z)

        z = self.enc_conv1d_2(z)
        z = self.enc_act_2(z)

        z = z.permute(2, 0, 1) # (T, B, D_h)
        z_sl = x_sl // self.stride
        z, z_sl = self.enc_lstm_block(z, z_sl) # (T, B, D_h)
        logits = self.enc_linear(z) # (T, B, D_z)

        z = F.gumbel_softmax(logits=logits, tau=tau, hard=True)
        tm = sequence_mask(z_sl, dtype=torch.float32, device=z.device) # (B, T)
        z = z * tm.T.unsqueeze(-1)

        return z, z_sl, logits.transpose(0, 1)

    def reconstruct(self, z, z_sl):

        r = self.dec_linear(z) # (T, B, D_h)
        r = self.dec_linear_act(r) # (T, B, D_h)
        r, _ = self.dec_lstm_block(r, z_sl) # (T, B, D_h)
        r = r.permute(1, 2, 0) # (B, D_h, T)
    
        # for conv, act in self.decoder_conv:
        #     r = conv(r)
        #     r = act(r)

        r = self.dec_conv1d_0(r)
        r = self.dec_act_0(r)

        r = self.dec_conv1d_1(r)
        r = self.dec_act_1(r)

        r = self.dec_conv1d_2(r)
        r = self.dec_act_2(r)

        r = r.transpose(1, 2) # (B, T, D_q)

        return r

    def forward(self, x, x_sl, y, y_sl, tau=1.0):
        
        t = self.target_encode(x)
        z, z_sl, logits = self.inference(x, x_sl, tau=tau)
        r = self.reconstruct(z, z_sl)

        p_y = r - r.logsumexp(dim=-1, keepdim=True)
        log_prob = p_y.gather(-1, t.unsqueeze(-1)).squeeze(-1)
        tm = sequence_mask(x_sl, dtype=torch.float32, device=x.device)
        log_prob = log_prob * tm
        
        loss = - log_prob.sum() / x_sl.sum()

        # hyps_raw = logits.argmax(dim=-1)
        # hyps = self.token_map.decode_batch(hyps_raw, z_sl, "") # assumes char-level prediction
        # refs = self.token_map.decode_batch(y, y_sl, "") # assumes char-level prediction
        
        outputs = SimpleNamespace(
            logits=logits,
            sl=z_sl
            #hyps=hyps,
            #refs=refs
        )

        metrics = [
            LossMetric(loss, weight_by=x_sl.sum()),
            WindowMeanMetric(loss)
            #ErrorRateMetric(refs, hyps, word_tokenizer, name="wer"),
            #ErrorRateMetric(refs, hyps, char_tokenizer, name="cer")
        ]

        return loss, metrics, outputs
