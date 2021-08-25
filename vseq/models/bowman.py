from types import SimpleNamespace
from typing import Tuple, List, Union

from torchtyping.tensor_type import TensorType
from vseq.evaluation.metrics import KLMetric, LatestMeanMetric, LossMetric, PerplexityMetric

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

from vseq.modules import HighwayStackDense, WordDropout
from vseq.modules.activations import InverseSoftplus
from vseq.utils.operations import sequence_mask
from vseq.utils.log_likelihoods import categorical_ll
from vseq.evaluation import Metric, LLMetric, KLMetric, PerplexityMetric, BitsPerDimMetric

from .base_model import BaseModel


class Bowman(BaseModel):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hidden_size: int,
        latent_dim: int,
        n_highway_blocks: int,
        delimiter_token_idx: int,
        word_dropout_rate: float = 0.0,
        random_prior_variance: Union[bool, float] = False,
        trainable_prior: bool = False,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.n_highway_blocks = n_highway_blocks
        self.delimiter_token_idx = delimiter_token_idx
        self.random_prior_variance = random_prior_variance
        self.trainable_prior = trainable_prior
        self.word_dropout_rate = word_dropout_rate

        self.std_activation = nn.Softplus(beta=np.log(2))
        self.std_activation_inverse = InverseSoftplus(beta=np.log(2))

        # The input embedding for x. We use one embedding shared between encoder and decoder. This may be inappropriate.
        self.embedding = nn.Embedding(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)
        self.mask_token_idx = num_embeddings

        self.lstm_encode = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
        )

        if n_highway_blocks > 0:
            self.h_to_z = nn.Sequential(
                HighwayStackDense(n_features=hidden_size, n_blocks=n_highway_blocks),
                nn.Linear(hidden_size, 2 * latent_dim),
            )
            self.z_to_h = nn.Sequential(
                nn.Linear(latent_dim, hidden_size),
                nn.Tanh(),
                HighwayStackDense(n_features=hidden_size, n_blocks=n_highway_blocks),
            )
        else:
            self.h_to_z = nn.Linear(hidden_size, 2 * latent_dim)
            self.z_to_h = nn.Linear(latent_dim, hidden_size)

        self.lstm_decode = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
        )

        self.output = nn.Linear(hidden_size, num_embeddings)

        if random_prior_variance:
            prior_log_sigma = self.std_activation_inverse(1 + float(random_prior_variance) * torch.randn(latent_dim))
        else:
            prior_log_sigma = self.std_activation_inverse(torch.ones(latent_dim))
        prior_logits = torch.cat([torch.zeros(latent_dim), prior_log_sigma])
        if self.trainable_prior:
            self.prior_logits = nn.Parameter(prior_logits)
        else:
            self.register_buffer("prior_logits", prior_logits)

        self.word_dropout = WordDropout(self.word_dropout_rate, mask_value=self.mask_token_idx)

    def prior(self):
        """Return the prior distribution without a batch dimension"""
        mu, log_sigma = self.prior_logits.chunk(2, dim=0)
        sigma = self.std_activation(log_sigma)
        return D.Normal(mu, sigma)

    def compute_elbo(self, log_prob_twise, kl_dwise, x_sl, beta: float = 1):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""
        kl = kl_dwise.sum(2).squeeze()  # (B,)
        log_prob = log_prob_twise.sum(1)  # (B,)
        elbo = log_prob - kl  # (B,)
        loss = -(log_prob - beta * kl).sum() / (x_sl - 1).sum()  # (1,)
        return loss, elbo, log_prob, kl

    def forward(
        self, x: TensorType["B", "T", int], x_sl: TensorType["B", int], beta: float = 1,
    ) -> Tuple[torch.Tensor, List[Metric], SimpleNamespace]:
        """Perform inference and generative passes on input x of shape (B, T)"""
        p_z = self.prior()

        z, q_z = self.infer(x, x_sl)

        kl_dwise = torch.distributions.kl_divergence(q_z, p_z)

        log_prob_twise, logits = self.reconstruct(z=z, x=x, x_sl=x_sl)

        loss, elbo, log_prob, kl = self.compute_elbo(log_prob_twise, kl_dwise, x_sl=x_sl, beta=beta)

        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo"),
            LLMetric(log_prob, name="rec"),
            KLMetric(kl),
            BitsPerDimMetric(elbo, reduce_by=x_sl - 1),
            PerplexityMetric(elbo, reduce_by=x_sl - 1),
            LatestMeanMetric(beta, name="beta"),
        ]

        outputs = SimpleNamespace(
            loss=loss,
            elbo=elbo,
            rec=log_prob,
            kl=kl,
            q_z=q_z,
            p_z=p_z,
            z=z,
            logits=logits,
        )
        return loss, metrics, outputs

    def infer(self, x: torch.Tensor, x_sl: torch.Tensor):
        # Encode input sequence
        # x, x_sl = x[:, :-1], x_sl - 1  # Remove end token
        e = self.embedding(x)
        e = torch.nn.utils.rnn.pack_padded_sequence(e, x_sl, batch_first=True)
        h, (h_t, c_t) = self.lstm_encode(e)
        h_t = h_t.transpose(0, 1)  # (T=1, B, D) --> (B, T=1, D)

        # Compute and sample from q(z|x)
        q_z_logits = self.h_to_z(h_t)
        mu, log_sigma = q_z_logits.chunk(2, dim=2)
        sigma = self.std_activation(log_sigma)
        q_z = D.Normal(mu, sigma)
        z = q_z.rsample()
        return z, q_z  # (B, T=1, D)

    def reconstruct(self, z: torch.Tensor, x: torch.Tensor, x_sl: torch.Tensor):
        """
        Computes log-likelihood for x under p(x|z).
        """
        # Prepare decoder initial states
        h_0 = self.z_to_h(z.transpose(0, 1))  # (B, T, D) -> (T, B, D)
        c_0 = torch.zeros_like(h_0)

        # Prepare inputs (x) and targets (y)
        y = x[:, 1:].clone().detach()  # Remove start token, batch_first=False and prevent from being masked
        x, x_sl = x[:, :-1], x_sl - 1  # Remove end token

        # Dropout and embed
        x = self.word_dropout(x)
        e = self.embedding(x)

        # Compute log probs for p(x|z)
        e = torch.nn.utils.rnn.pack_padded_sequence(e, x_sl, batch_first=True)
        h, (h_n, c_n) = self.lstm_decode(e, (h_0, c_0))
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        # Define output distribution
        logits = self.output(h)  # labo: we could use our embedding matrix here
        seq_mask = sequence_mask(x_sl, dtype=float, device=logits.device)
        log_prob = categorical_ll(y, logits) * seq_mask
        return log_prob, logits

    def generate(self, z: torch.Tensor = None, n_samples: int = 1, t_max: int = 100, use_mode: bool = False):
        """
        Generates a sequence by autoregressively sampling from p(x_t|x_<t, z).
        """
        # Setup initial loop conditions
        n_samples = n_samples if z is None else z.shape[0]
        x_t = torch.full([1, n_samples], self.delimiter_token_idx, device=self.device)
        z = self.prior().sample(torch.Size([n_samples, 1])) if z is None else z  # (B, T, D)
        h_t = self.z_to_h(z.transpose(0, 1))  # (B, T, D) -> (T, B, D)
        c_t = torch.zeros_like(h_t)

        # Sample x from p(x|z)
        log_prob, x, x_sl = [], [], torch.zeros(n_samples, dtype=torch.int)
        seq_active = torch.ones(n_samples, dtype=torch.int)
        all_ended, t = False, 0  # Used to condition while loop
        while not all_ended and t < t_max:

            # Sample x_t from p(x_t|z, x_<t)
            e_t = self.embedding(x_t)
            _, (h_t, c_t) = self.lstm_decode(e_t, (h_t, c_t))
            logits = self.output(h_t)  # labo: again, we could use our embedding matrix here
            p = D.Categorical(logits=logits)
            x_t = p.logits.argmax(dim=-1) if use_mode else p.sample()
            log_prob_t = p.log_prob(x_t)

            # Update outputs
            x.append(x_t)
            log_prob.append(log_prob_t)

            # Update sequence length
            x_sl += seq_active
            seq_ending = (x_t[0].cpu() == self.delimiter_token_idx).to(int)
            seq_active *= 1 - seq_ending

            # Update loop conditions
            t += 1
            all_ended = torch.all(1 - seq_active).item()

        seq_mask = sequence_mask(x_sl, dtype=int, device=self.device)
        x = torch.cat(x).T * seq_mask
        log_prob = torch.cat(log_prob).T * seq_mask.to(float)

        return (x, x_sl), log_prob
