from types import SimpleNamespace
from collections import OrderedDict
from typing import Tuple, List
from vseq.evaluation.metrics import KLMetric, LossMetric, PerplexityMetric

import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from vseq.data.token_map import TokenMap

from vseq.utils.operations import sequence_mask
from vseq.evaluation import Metric, LLMetric, KLMetric, SeqAccuracyMetric, BitsPerDimMetric
from vseq.data.transforms import Compose, MuLawEncode, Quantize, Scale, MuLawDecode

from .base_model import BaseModel


class AstroVAE(nn.Module):
    def __init__(
        self,
        kernel_and_strides: list,
        hidden_size: int,
        conv_hidden_size: int,
        prior_hidden_size: int,
        num_components: int,
        output_bits: int,
    ):
        super().__init__()

        assert len(kernel_and_strides) == 2 # simplified for now
        
        self.kernel_and_strides = kernel_and_strides
        self.hidden_size = hidden_size
        self.conv_hidden_size = conv_hidden_size
        self.prior_hidden_size = prior_hidden_size
        self.num_components = num_components
        self.output_bits = output_bits

        ks1, ks2 = kernel_and_strides
        
        # inference
        self.conv1_infer = nn.Conv1d(1, conv_hidden_size, ks1, ks1)
        self.act1_infer = nn.ReLU6()
        self.conv2_infer = nn.Conv1d(conv_hidden_size, conv_hidden_size, ks2, ks2)
        self.act2_infer = nn.ReLU6()
        self.dense_infer = nn.Linear(conv_hidden_size, hidden_size * 2)
        self.std_activation_infer = nn.Softplus(beta=np.log(2))
        
        # generate
        self.dense_generate = nn.Linear(hidden_size, conv_hidden_size)
        self.act1_generate = nn.ReLU6()
        self.conv1_generate = nn.ConvTranspose1d(conv_hidden_size, conv_hidden_size, ks2, ks2)
        self.act2_generate = nn.ReLU6()
        self.conv2_generate = nn.ConvTranspose1d(conv_hidden_size, 2 ** output_bits, ks1, ks1)
        
        # prior (embedding is shared with posterior)
        self.std_activation_prior = nn.Softplus(beta=np.log(2))
        self.embedding = nn.Embedding(num_embeddings=self.num_components, embedding_dim=hidden_size * 2)
        self.lstm_prior = nn.LSTM(
            input_size=hidden_size,
            hidden_size=prior_hidden_size,
            num_layers=1
        )
        self.dense_prior = nn.Linear(prior_hidden_size, self.num_components)
        
        # target encoder
        mu_law_encoder = MuLawEncode()
        quantizer = Quantize(bits=output_bits)
        self.target_encoder = Compose(mu_law_encoder, quantizer)
        
        # target decoder
        scale = Scale(min_val=0, max_val=(2 ** output_bits) - 1)
        mu_law_decoder = MuLawDecode()
        self.target_decode = Compose(scale, mu_law_decoder)

    # def state_dict_prior(self):
    #     prior_submodules = ["embedding", "lstm_prior", "dense_prior"]
    #     state_dict = OrderedDict()
    #     for name, param in self.state_dict().items():
    #         submodule = name.split(".")[0]
    #         if submodule in prior_submodules:
    #             state_dict[name] = param
    #     return state_dict
    
    def set_transform_device(self, device):
        b = self.target_encoder.transforms[1].boundaries
        self.target_encoder.transforms[1].boundaries = b.to(device)
    
    def forward(self, x, x_sl, beta=1.0) -> Tuple[torch.Tensor, List[Metric], SimpleNamespace]:
        """Perform inference and generative passes on input x of shape (B, T)"""
        
        x_q = self.target_encoder(x)
        z_sl = torch.floor_divide(x_sl, np.prod(self.kernel_and_strides))
        x_mask_2d = sequence_mask(x_sl, dtype=x.dtype, device=x.device) 
        z_mask_2d = sequence_mask(z_sl, dtype=x.dtype, device=x.device)
        z_mask_3d = z_mask_2d.unsqueeze(2)

        q_z = self.infer(x, x_sl)
        z = q_z.rsample() * z_mask_3d
        p_z = self.prior(z, z_sl)
        p_x = self.reconstruct(z)

        # import IPython; IPython.embed()
        
        n, w = p_z
        p_z_log_probs = (torch.log(w) + n.log_prob(z.unsqueeze(2)).sum(3)).logsumexp(2)
        q_z_log_probs = q_z.log_prob(z).sum(2)
        kl_bt = (q_z_log_probs - p_z_log_probs) * z_mask_2d
        log_prob_bt = p_x.log_prob(x_q) * x_mask_2d
        
        kl = kl_bt.sum(dim=1) # (B,)
        log_prob = log_prob_bt.sum(dim=1) # (B,)
        elbo = log_prob - kl # (B,)
        
        preds = p_x.logits.argmax(2)
        loss =  - ((log_prob.sum() / x_sl.sum()) - beta * (kl.sum() / z_sl.sum()))
        # loss = - log_prob.sum() / x_sl.sum()

        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            # LLMetric(elbo, name="elbo"),
            # LLMetric(log_prob, name="rec"),
            # KLMetric(kl),
            LLMetric(elbo, name="elbo (npt)", reduce_by=x_sl),
            KLMetric(kl, name="kl (npt)", reduce_by=z_sl),
            LLMetric(log_prob, name="rec (npt)", reduce_by=x_sl),
            BitsPerDimMetric(elbo, reduce_by=x_sl),
            SeqAccuracyMetric(preds, x_q, mask=x_mask_2d, name="acc")
        ]

        outputs = SimpleNamespace(
            loss=loss,
            elbo=elbo,
            rec=log_prob,
            kl=kl,
            p_x=p_x,  # NOTE Save 700 MB by not returning p_x
            q_z=q_z,
            p_z=p_z,
            z=z,
        )
        return loss, metrics, outputs

    def prior(self, z, z_sl):
        """Return the prior distribution without a batch dimension"""

        # prepare inputs
        B, _, K = z.shape
        sos = torch.zeros([B, 1, K], dtype=z.dtype, device=z.device) # TODO: Add learnable vector
        z_shift = torch.cat([sos, z[:, :-1]], dim=1)
        
        # compute parameters
        z_shift = torch.nn.utils.rnn.pack_padded_sequence(z_shift, z_sl, batch_first=True)
        h, _ = self.lstm_prior(z_shift)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        w = self.dense_prior(h).softmax(2)
        
        # parameterize p_z
        mu, log_sigma = self.embedding.weight.chunk(2, dim=1)
        sigma = self.std_activation_prior(log_sigma)
        gmm = D.Normal(mu, sigma) # unweighted GMM
        p_z = (gmm, w)
        
        return p_z

    def infer(self, x: torch.Tensor, x_sl: torch.Tensor):

        # compute parameters
        x = x.unsqueeze(1) # --> (B, 1, T)
        h = self.conv1_infer(x)
        h = self.act1_infer(h)
        h = self.conv2_infer(h)
        h = self.act2_infer(h)
        h = h.transpose(1, 2) # --> (B, T, D)
        q_z_logits = self.dense_infer(h)
        # w = h.softmax(2)
        # q_z_logits = torch.matmul(w, self.embedding.weight)
        
        # parameterize q_z
        mu, log_sigma = q_z_logits.chunk(2, dim=2)
        sigma = self.std_activation_infer(log_sigma)
        q_z = D.Normal(mu, sigma)
        
        return q_z

    def reconstruct(self, z):
        """
        Computes log-likelihood for x under p(x|z).
        """
        
        # compute parameters
        h = self.dense_generate(z)
        h = h.transpose(1, 2) # --> (B, D, T)
        h = self.act1_generate(h)
        h = self.conv1_generate(h)
        h = self.act2_generate(h)
        p_x_logits = self.conv2_generate(h).transpose(1, 2) # --> (B, D, T)
        
        # parameterize p_x
        p_x = D.Categorical(logits=p_x_logits)

        return p_x

    def generate(self, num_examples=1, max_len=100, device=None):
        """
        Generates a sequence by autoregressively sampling z from p(z_t|z_<t) and then x from p(x|z).
        """
        
        # prepare inputs
        B, K = num_examples, self.hidden_size
        z_t = torch.zeros([1, B, K], dtype=torch.float32, device=device)
        state = None
        
        # compute z
        z = []
        for t in range(max_len):
            h, state = self.lstm_prior(z_t, state)
            w_sample = D.Categorical(logits=self.dense_prior(h)).sample()
            p_z_t_logits = self.embedding(w_sample)
            mu, log_sigma = p_z_t_logits.chunk(2, dim=2)
            sigma = self.std_activation_prior(log_sigma)
            p_z_t = D.Normal(mu, sigma)
            z_t = p_z_t.rsample()
            
            z.append(z_t)

        z = torch.cat(z, dim=0).transpose(0, 1) # --> (B, T, D)
        p_x = self.reconstruct(z)
        samples = p_x.sample()
        # target decode here?
        
        return samples