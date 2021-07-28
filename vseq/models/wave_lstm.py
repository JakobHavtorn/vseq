from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from vseq.modules import CausalConv1dBlock, LSTMBlock
from vseq.data.transforms import Compose, MuLawEncode, MuLawDecode, Quantize, Scale
from vseq.utils.operations import sequence_mask
from vseq.evaluation.metrics import LossMetric

LSTM_AR_1D_CONV_FRONTEND = ((32, 2, 1), # num_filters, kernel_size, dilation
                            (64, 2, 2),
                            (96, 2, 4),
                            (128, 2, 8),
                            (256, 2, 16))

class WaveLSTM(nn.Module):
    def __init__(self,
                 hidden_size: int = 320,
                 lstm_layers: int = 3,
                 conv_config: tuple = LSTM_AR_1D_CONV_FRONTEND,
                 output_bits: int = 8
    ):
        super().__init__()
        
        self.conv_config = conv_config
        self.conv1d_block = CausalConv1dBlock(config=conv_config,
                                              bottleneck_size=hidden_size,
                                              dropout_prob=0.0)
        self.lstm_block = LSTMBlock(input_size=hidden_size,
                                    hidden_size=hidden_size,
                                    num_layers=lstm_layers,
                                    bidirectional=False,
                                    sum_directions=False,
                                    dropout_prob=0.0,
                                    return_all=True,
                                    return_states=True)
        self.bottleneck_layer = nn.Linear(lstm_layers * hidden_size, hidden_size)
        self.bottleneck_act = nn.ReLU6()
        self.output_layer = nn.Linear(hidden_size, 2 ** output_bits)
        
        # target encode
        self.mu_law_encoder = MuLawEncode()
        self.quantizer = Quantize(bits=output_bits)
        self.target_encode = Compose(self.mu_law_encoder, self.quantizer)

        # target decode
        self.scale = Scale(min_val=0, max_val=(2 ** output_bits) - 1)
        self.mu_law_decoder = MuLawDecode()
        self.target_decode = Compose(self.scale, self.mu_law_decoder)
    
    def set_transform_device(self, device):
        self.quantizer.boundaries = self.quantizer.boundaries.to(device)
    
    def forward(self, input, sl, states=None):
        
        x = input
        y = self.quantizer(x)
        h = self.conv1d_block(x) # --> (B, T, D)
        h = h.transpose(0, 1) # --> (T, B, D)
        h, _, end_states  = self.lstm_block(h, sl, states=states)
        h = torch.cat(h, 2).transpose(0, 1) # --> (B, T, D)
        h = self.bottleneck_layer(h)
        h = self.bottleneck_act(h)
        logits = self.output_layer(h)
        
        p_y = logits - logits.logsumexp(dim=-1, keepdim=True)
        log_prob = p_y.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        tm = sequence_mask(sl, dtype=torch.float32, device=x.device)
        log_prob = log_prob * tm
        
        loss = - log_prob.sum() / sl.sum()
        
        metrics = [
            LossMetric(loss, weight_by=sl.sum()),
        ]

        outputs = SimpleNamespace(
            loss=loss,
            logits=logits,
            p_y=p_y,
            states=end_states
        )
        
        return loss, metrics, outputs
    
    # def generate(self, num_examples=1, max_len=16000 * 3, device=None):
    #     """
    #     Generates a sequence by autoregressively sampling x_t from p(x_t|x_<t).
    #     """
        
    #     # prepare inputs
    #     B, K = num_examples, self.hidden_size
    #     z_t = torch.zeros([1, B, K], dtype=torch.float32, device=device)
    #     state = None
        
    #     # compute parameters
    #     z = []
    #     for t in range(max_len):
    #         h, state = self.lstm_prior(z_t, state)
    #         w_sample = D.Categorical(logits=self.dense_prior(h)).sample()
    #         p_z_t_logits = self.embedding(w_sample)
    #         mu, log_sigma = p_z_t_logits.chunk(2, dim=2)
    #         sigma = self.std_activation_prior(log_sigma)
    #         p_z_t = D.Normal(mu, sigma)
    #         z_t = p_z_t.rsample()
    #         z.append(z_t)

    #     z = torch.cat(z, dim=0)
    #     samples = self.dense_generate(z).argmax(2).T
        
    #     return samples