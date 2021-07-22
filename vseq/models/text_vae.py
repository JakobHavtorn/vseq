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

from .base_model import BaseModel


class TextVAE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        prior_hidden_size: int,
        token_map: TokenMap
    ):
        super().__init__()

        # 
        self.hidden_size = hidden_size
        self.prior_hidden_size = prior_hidden_size
        self.num_tokens = len(token_map)

        self.std_activation_infer = nn.Softplus(beta=np.log(2))
        self.std_activation_prior = nn.Softplus(beta=np.log(2))

        self.embedding = nn.Embedding(num_embeddings=self.num_tokens, embedding_dim=hidden_size * 2)
        self.lstm_prior = nn.LSTM(
            input_size=hidden_size,
            hidden_size=prior_hidden_size,
            num_layers=1
        )
        
        self.dense_prior = nn.Linear(prior_hidden_size, self.num_tokens)
        self.dense_generate = nn.Linear(hidden_size, self.num_tokens)

    def state_dict_prior(self):
        prior_submodules = ["embedding", "lstm_prior", "dense_prior"]
        state_dict = OrderedDict()
        for name, param in self.state_dict().items():
            submodule = name.split(".")[0]
            if submodule in prior_submodules:
                state_dict[name] = param
        return state_dict
    
    def forward(self, x, x_sl, beta=1.0) -> Tuple[torch.Tensor, List[Metric], SimpleNamespace]:
        """Perform inference and generative passes on input x of shape (B, T)"""
        
        mask_2d = sequence_mask(x_sl, dtype=x.dtype, device=x.device)
        mask_3d = mask_2d.unsqueeze(2)
        
        q_z = self.infer(x, x_sl)
        z = q_z.rsample() * mask_3d
        p_z = self.prior(z, x_sl)
        p_x = self.reconstruct(z)

        n, w = p_z
        p_z_log_probs = (torch.log(w) + n.log_prob(z.unsqueeze(2)).sum(3)).logsumexp(2)
        q_z_log_probs = q_z.log_prob(z).sum(2)
        kl_bt = (q_z_log_probs - p_z_log_probs) * mask_2d
        # kl_btd = torch.distributions.kl_divergence(q_z, p_z) * mask_3d
        log_prob_bt = p_x.log_prob(x) * mask_2d
        
        kl = kl_bt.sum(dim=1) # (B,)
        log_prob = log_prob_bt.sum(dim=1) # (B,)
        elbo = log_prob - kl # (B,)
        
        preds = p_x.logits.argmax(2)
        loss = - ((log_prob - beta * kl).sum() / x_sl.sum())

        
        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo"),
            LLMetric(log_prob, name="rec"),
            KLMetric(kl),
            KLMetric(kl, name="kl (bpt)", reduce_by=x_sl / math.log(2)),
            BitsPerDimMetric(elbo, reduce_by=x_sl),
            SeqAccuracyMetric(preds, x, mask=mask_2d, name="acc")
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
        q_z_logits = self.embedding(x)
        
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
        p_x_logits = self.dense_generate(z)
        
        # parameterize p_x
        # seq_mask = sequence_mask(x_sl - 1, dtype=float, device=p_logits.device)
        p_x = D.Categorical(logits=p_x_logits)
        # log_prob = p_x.log_prob(y) * seq_mask
        # log_prob = torch.gather(p_logits.log_softmax(dim=-1), 2, y.unsqueeze(2)).squeeze() * seq_mask  # NOTE -600 MB

        return p_x

    def generate(self, num_examples=1, max_len=100, device=None):
        """
        Generates a sequence by autoregressively sampling z from p(z_t|z_<t) and then x from p(x|z).
        """
        
        # prepare inputs
        B, K = num_examples, self.hidden_size
        z_t = torch.zeros([1, B, K], dtype=torch.float32, device=device)
        state = None
        
        # compute parameters
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

        z = torch.cat(z, dim=0)
        samples = self.dense_generate(z).argmax(2).T
        
        return samples