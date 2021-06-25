import math

from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from torchtyping import TensorType

from vseq.evaluation.metrics import BitsPerDimMetric, KLMetric, LLMetric, LatestMeanMetric, LossMetric
from vseq.models.base_model import BaseModel
from vseq.modules.distributions import DiscretizedLogisticMixtureDense, GaussianDense
from vseq.utils.variational import discount_free_nats, kl_divergence_gaussian
from vseq.utils.operations import sequence_mask

from .coders import MultiLevelEncoderAudioDense, DecoderAudioDense


def get_exponential_time_factors(abs_factor, num_levels):
    """Return exponentially increasing temporal abstraction factors with base `abs_factor`"""
    return [abs_factor ** l for l in range(num_levels)]


class RSSMCell(torch.jit.ScriptModule):
    def __init__(self, z_dim: int, h_dim: int, e_dim: int, c_dim: int, residual_posterior: bool = False):
        """Recurrent State Space Model cell

        Args:
            z_dim (int): Dimensionality stochastic state space.
            h_dim (int): Dimensionality of deterministic state space.
            e_dim (int): Dimensionalit of input embedding space.
            c_dim (int): Dimensionality of "context".
        """
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.c_dim = c_dim
        self.residual_posterior = residual_posterior

        self.gru_in = nn.Sequential(nn.Linear(z_dim + c_dim, h_dim), nn.ReLU())
        self.gru_cell = nn.GRUCell(h_dim, h_dim)

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            GaussianDense(h_dim, z_dim),
        )

        self.posterior = nn.Sequential(
            nn.Linear(h_dim + e_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            GaussianDense(h_dim, z_dim),
        )

    def get_initial_state(self, batch_size: int, device: str = None):
        device = device if device is not None else self.prior[0].weight.device
        return (torch.zeros(batch_size, self.z_dim, device=device), torch.zeros(batch_size, self.h_dim, device=device))

    def get_empty_context(self, batch_size: int, device: str = None):
        device = device if device is not None else self.prior[0].weight.device
        return torch.empty(batch_size, 0, device=device)

    def forward(
        self,
        enc_inputs: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        use_mode: bool = False,
    ):
        z, h = state

        gru_in = self.gru_in(torch.cat([z, context], dim=-1))
        h_new = self.gru_cell(gru_in, h)

        prior_mu, prior_sd = self.prior(h_new)

        enc_mu, enc_sd = self.posterior(torch.cat([h_new, enc_inputs], dim=-1))

        if self.residual_posterior:
            enc_mu = enc_mu + prior_mu

        z_new = self.posterior[-1].rsample((enc_mu, enc_sd)) if not use_mode else prior_mu

        distributions = SimpleNamespace(enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=prior_mu, prior_sd=prior_sd)

        return (z_new, h_new), distributions

    @torch.jit.export
    def generate(self, state: Tuple[torch.Tensor, torch.Tensor], context: torch.Tensor, use_mode: bool = False):
        z, h = state

        gru_in = self.gru_in(torch.cat([z, context], dim=-1))
        h_new = self.gru_cell(gru_in, h)

        prior_mu, prior_sd = self.prior(h_new)
        z_new = self.prior[-1].rsample((prior_mu, prior_sd)) if not use_mode else prior_mu

        distributions = SimpleNamespace(prior_mu=prior_mu, prior_sd=prior_sd)

        return (z_new, h_new), distributions


class CWVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        likelihood: nn.Module,
        z_size: Union[int, List[int]],
        h_size: Union[int, List[int]],
        time_factors: List[int],
        residual_posterior: bool = False,
    ):
        super().__init__()

        assert len(z_size) == len(h_size), "Must give equal number of levels for stochastic and deterministic state"
        assert isinstance(time_factors, int) or len(time_factors) == len(
            z_size
        ), "Must give as many time factors as levels"
        assert encoder.num_levels == len(z_size), "Number of levels in encoder and in latent dimensions must match"

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood
        self.num_levels = len(time_factors)
        self.time_factors = time_factors
        self.residual_posterior = residual_posterior

        self.z_size = [z_size] * self.num_levels if isinstance(z_size, int) else z_size
        self.h_size = [h_size] * self.num_levels if isinstance(h_size, int) else h_size
        self.c_size = [z_dim + h_dim for z_dim, h_dim in zip(self.z_size[1:], self.h_size[1:])] + [0]

        cells = []
        for h_dim, z_dim, c_dim, e_dim in zip(self.h_size, self.z_size, self.c_size, encoder.out_size):
            cells.append(
                RSSMCell(h_dim=h_dim, z_dim=z_dim, e_dim=e_dim, c_dim=c_dim, residual_posterior=residual_posterior)
            )
        self.cells = nn.ModuleList(cells)

    def build_metrics(self, x_sl, loss, elbo, log_prob, kld, klds, beta, free_nats):
        kld_metrics_nats = [
            KLMetric(klds[l], name=f"kl_{l} (nats)", log_to_console=False) for l in range(self.num_levels)
        ]
        kld_metrics_bpd = [
            KLMetric(klds[l], name=f"kl_{l} (bpt)", reduce_by=(x_sl / (math.log(2) * self.time_factors[l])))
            for l in range(self.num_levels)
        ]
        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo (nats)"),
            BitsPerDimMetric(elbo, name="elbo (bpt)", reduce_by=x_sl),
            LLMetric(log_prob, name="rec (nats)", log_to_console=False),
            BitsPerDimMetric(log_prob, name="rec (bpt)", reduce_by=x_sl),
            KLMetric(kld, name="kl (nats)"),
            KLMetric(kld, name="kl (bpt)", reduce_by=x_sl / (math.log(2) * self.time_factors[0])),
            *kld_metrics_nats,
            *kld_metrics_bpd,
            LatestMeanMetric(beta, name="beta"),
            LatestMeanMetric(free_nats, name="free_nats"),
        ]
        return metrics

    def compute_elbo(
        self,
        y: TensorType["B", "T", "D"],
        parameters: TensorType["B", "T", "D"],
        kld_layerwise: List[TensorType["B", "T", "latent_size"]],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
    ):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""
        seq_mask = sequence_mask(x_sl, max_len=y.shape[1], dtype=float, device=y.device)

        log_prob_twise = self.likelihood.log_prob(y, parameters) * seq_mask
        log_prob = log_prob_twise.view(y.size(0), -1).sum(1)  # (B,)

        klds, klds_fn = [], []
        seq_mask = seq_mask.unsqueeze(-1)  # Broadcast to latent dimension
        for l in range(self.num_levels):
            mask = seq_mask[:, :: self.time_factors[l]]  # Faster than creating new mask
            klds.append((kld_layerwise[l] * mask).sum((1, 2)))  # (B,)
            klds_fn.append((discount_free_nats(kld_layerwise[l], free_nats, shared_dims=-1) * mask).sum((1, 2)))  # (B,)

        kld, kld_fn = sum(klds), sum(klds_fn)

        elbo = log_prob - kld  # (B,)

        loss = -(log_prob - beta * kld_fn).sum() / x_sl.sum()  # (1,)

        return loss, elbo, log_prob, kld, klds, seq_mask

    def forward(
        self,
        x: TensorType["B", "T", "D"],
        x_sl: TensorType["B", int],
        state0: List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]] = None,
        beta: float = 1,
        free_nats: float = 0,
        y: Optional[TensorType["B", "T", "D"]] = None,
    ):
        # target
        if y is None:
            y = x.clone().detach()

        # compute encodings
        encodings = self.encoder(x)
        encodings_t = [enc.unbind(1) for enc in encodings]

        # initial context for top layer
        context = [self.cells[0].get_empty_context(batch_size=x.size(0))] * len(encodings_t[-1])

        # initial RSSM state (z, h)
        states = [cell.get_initial_state(batch_size=x.size(0)) for cell in self.cells] if state0 is None else state0

        kl_divs = [[] for _ in range(self.num_levels)]
        t = 0
        for l in range(self.num_levels - 1, -1, -1):
            all_states = []
            all_distributions = []
            T = len(encodings_t[l])
            for t in range(T):
                # reset stochastic state whenever the layer above ticks (never reset top)

                # cell forward
                states[l], distributions = self.cells[l](encodings_t[l][t], states[l], context[t])

                all_states.append(states[l])
                all_distributions.append(distributions)

            # update context for below layer as cat(z_l, h_l)
            context = [torch.cat(all_states[t], dim=-1) for t in range(T)]
            if l >= 1:
                # repeat to temporal resolution of below layer
                factor = int(self.time_factors[l] / self.time_factors[l - 1])
                context = [context[i // factor] for i in range(len(context) * factor)]

            # compute kl divergence
            enc_mu = torch.stack([all_distributions[t].enc_mu for t in range(T)], dim=1)
            enc_sd = torch.stack([all_distributions[t].enc_sd for t in range(T)], dim=1)
            prior_mu = torch.stack([all_distributions[t].prior_mu for t in range(T)], dim=1)
            prior_sd = torch.stack([all_distributions[t].prior_sd for t in range(T)], dim=1)

            kld = kl_divergence_gaussian(enc_mu, enc_sd, prior_mu, prior_sd)
            kl_divs[l] = kld

        context = torch.stack(context, dim=1)
        dec = self.decoder(context)

        parameters = self.likelihood(dec)

        loss, elbo, log_prob, kld, klds, seq_mask = self.compute_elbo(y, parameters, kl_divs, x_sl, beta, free_nats)

        metrics = self.build_metrics(x_sl, loss, elbo, log_prob, kld, klds, beta, free_nats)
        outputs = SimpleNamespace(elbo=elbo, log_prob=log_prob, kld=kld, y=y, parameters=parameters, seq_mask=seq_mask)
        outputs.x_hat = self.likelihood.sample(outputs.parameters)
        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        if x is not None:
            raise NotImplementedError("Conditional generation is not implemented")
            # TODO Run a forward pass over x to get conditional initial state0 (assert state0 is None)
            #      Construct context as concatenation of state0 of layer above (empty for top layer)
        else:
            # initial context for top layer
            context = [self.cells[0].get_empty_context(batch_size=n_samples)] * (max_timesteps // self.time_factors[-1])

            # initial RSSM state (z, h)
            states = [cell.get_initial_state(batch_size=n_samples) for cell in self.cells] if state0 is None else state0

        for l in range(self.num_levels - 1, -1, -1):
            all_states = []
            all_distributions = []
            T = max_timesteps // self.time_factors[l]  # The upscaling factor of the decoder
            for t in range(T):
                # reset stochastic state whenever the layer above ticks (never reset top)

                # cell forward
                states[l], distributions = self.cells[l].generate(states[l], context[t], use_mode=use_mode)

                all_states.append(states[l])
                all_distributions.append(distributions)

            # update context for below layer as cat(z_l, h_l)
            context = [torch.cat(all_states[t], dim=-1) for t in range(T)]
            if l >= 1:
                # repeat to temporal resolution of below layer
                factor = int(self.time_factors[l] / self.time_factors[l - 1])
                context = [context[i // factor] for i in range(len(context) * factor)]

        context = torch.stack(context, dim=1)
        dec = self.decoder(context)

        parameters = self.likelihood(dec)

        if use_mode:
            x = self.likelihood.mode(parameters)
        else:
            x = self.likelihood.sample(parameters)

        x_sl = torch.ones(n_samples, dtype=torch.int) * max_timesteps
        outputs = SimpleNamespace(context=context, all_distributions=all_distributions)
        return (x, x_sl), outputs


class CWVAEAudioConv1D(BaseModel):
    def __init__(
        self,
        num_embeddings: Optional[int] = None,
        z_size: Union[int, List[int]] = 64,
        h_size: Union[int, List[int]] = 128,
        time_factors: Union[int, List[int]] = 6,
        num_levels: int = 3,
        residual_posterior: bool = False,
        num_level_layers: int = 3,
        num_mix: int = 10,
        num_bins: int = 256,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.z_size = z_size
        self.h_size = h_size
        self.num_levels = num_levels
        self.residual_posterior = residual_posterior
        self.num_level_layers = num_level_layers
        self.num_mix = num_mix
        self.num_bins = num_bins

        if isinstance(time_factors, int):
            time_factors = get_exponential_time_factors(time_factors, self.num_levels)

        self.time_factors = time_factors

        bot_z_size = z_size if isinstance(z_size, int) else z_size[0]
        bot_h_size = h_size if isinstance(h_size, int) else h_size[0]
        bot_c_size = bot_z_size + bot_h_size

        if num_embeddings is not None:
            self.in_channels = num_embeddings
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=bot_h_size)
        else:
            self.in_channels = 1
            self.embedding = None

        likelihood = DiscretizedLogisticMixtureDense(
            x_dim=3 * num_mix,
            y_dim=1,
            num_mix=num_mix,
            num_bins=num_bins,
            reduce_dim=-1,
        )

        encoder = MultiLevelEncoderConv1d(
            in_channels=self.in_channels,
            h_size=h_size,
            time_factors=time_factors,
            proj_size=None,
            num_level_layers=num_level_layers,
            activation=nn.ReLU,
        )

        decoder = DecoderAudioConv1d(
            in_dim=bot_c_size,
            h_dim=bot_h_size,
            o_dim=likelihood.out_features,
            time_factors=time_factors,
            num_level_layers=num_level_layers,
        )

        self.cwvae = CWVAE(
            encoder=encoder,
            decoder=decoder,
            likelihood=likelihood,
            z_size=z_size,
            h_size=h_size,
            time_factors=time_factors,
            residual_posterior=residual_posterior,
        )

    def forward(
        self,
        x: TensorType["B", "T"],
        x_sl: TensorType["B", int],
        state0: List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]] = None,
        beta: float = 1,
        free_nats: float = 0,
    ):
        y = x.detach().clone().unsqueeze(-1)  # Create target with channel dim for DML
        loss, metrics, outputs = self.cwvae(x, x_sl, state0, beta, free_nats, y)
        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
            x=x,
            state0=state0,
        )


class CWVAEAudioDense(BaseModel):
    def __init__(
        self,
        z_size: Union[int, List[int]] = 64,
        h_size: Union[int, List[int]] = 128,
        time_factors: Union[int, List[int]] = 6,
        num_levels: int = 3,
        residual_posterior: bool = False,
        num_level_layers: int = 3,
        num_mix: int = 10,
        num_bins: int = 256,
    ):
        super().__init__()

        self.z_size = z_size
        self.h_size = h_size
        self.num_levels = num_levels
        self.residual_posterior = residual_posterior
        self.num_level_layers = num_level_layers
        self.num_mix = num_mix
        self.num_bins = num_bins

        if isinstance(time_factors, int):
            time_factors = get_exponential_time_factors(time_factors, self.num_levels)

        self.time_factors = time_factors

        bot_z_size = z_size if isinstance(z_size, int) else z_size[0]
        bot_h_size = h_size if isinstance(h_size, int) else h_size[0]
        bot_c_size = bot_z_size + bot_h_size

        likelihood = DiscretizedLogisticMixtureDense(
            x_dim=3 * num_mix,
            y_dim=1,
            num_mix=num_mix,
            num_bins=num_bins,
            reduce_dim=-1,
        )

        encoder = MultiLevelEncoderAudioDense(
            h_size=h_size,
            time_factors=time_factors,
            num_level_layers=num_level_layers,
        )

        decoder = DecoderAudioDense(
            in_dim=bot_c_size,
            h_dim=bot_h_size,
            o_dim=likelihood.out_features,
            time_factors=time_factors,
            num_level_layers=num_level_layers,
        )

        self.cwvae = CWVAE(
            encoder=encoder,
            decoder=decoder,
            likelihood=likelihood,
            z_size=z_size,
            h_size=h_size,
            time_factors=time_factors,
            residual_posterior=residual_posterior,
        )

    def forward(
        self,
        x: TensorType["B", "T"],
        x_sl: TensorType["B", int],
        state0: List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]] = None,
        beta: float = 1,
        free_nats: float = 0,
    ):
        y = x.detach().clone().unsqueeze(-1)  # Create target with channel dim for DML
        loss, metrics, outputs = self.cwvae(x, x_sl, state0, beta, free_nats, y)
        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
            x=x,
            state0=state0,
        )
