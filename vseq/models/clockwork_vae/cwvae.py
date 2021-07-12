import math

from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from torchtyping import TensorType

from vseq.evaluation.metrics import BitsPerDimMetric, KLMetric, LLMetric, LatestMeanMetric, LossMetric
from vseq.models.base_model import BaseModel
from vseq.modules.distributions import DiscretizedLaplaceMixtureDense, DiscretizedLogisticMixtureDense, GaussianDense
from vseq.utils.variational import discount_free_nats, kl_divergence_gaussian
from vseq.utils.operations import sequence_mask

from .dense_coders import DenseAudioEncoder, DenseAudioDecoder
from .tasnet_coder import TasNetDecoder, TasNetEncoder
from .cpc_coders import CPCDecoder, CPCEncoder
from .conv_coders import AudioEncoderConv1d, AudioDecoderConv1d, ContextDecoderConv1d


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
        z_size: Union[int, List[int]],
        h_size: Union[int, List[int]],
        time_factors: List[int],
        encoder: nn.Module,
        decoder: nn.Module,
        likelihood: nn.Module,
        context_decoder: Optional[nn.ModuleList] = None,
        residual_posterior: bool = False,
        with_resets: bool = False,
    ):
        super().__init__()

        assert isinstance(time_factors, list)

        self.encoder = encoder
        self.decoder = decoder
        self.context_decoder = context_decoder
        self.likelihood = likelihood
        self.num_levels = len(time_factors)
        self.time_factors = time_factors
        self.residual_posterior = residual_posterior
        self.with_resets = with_resets

        self.strides = [self.time_factors[0]]
        self.strides += [self.time_factors[l] // self.time_factors[l - 1] for l in range(1, self.num_levels)]

        self.z_size = [z_size] * self.num_levels if isinstance(z_size, int) else z_size
        self.h_size = [h_size] * self.num_levels if isinstance(h_size, int) else h_size
        if context_decoder is None:
            self.c_size = [z_dim + h_dim for z_dim, h_dim in zip(self.z_size[1:], self.h_size[1:])] + [0]
        else:
            # When using context decoder that changes num_channels to that of layer below
            self.c_size = [h_dim for z_dim, h_dim in zip(self.z_size[:-1], self.h_size[:-1])] + [0]

        assert (
            len(self.z_size) == len(self.h_size) == len(self.c_size)
        ), f"Must have equal lengths: {self.z_size=}, {self.h_size=}, {self.c_size=}"

        cells = []
        for h_dim, z_dim, c_dim, e_dim in zip(self.h_size, self.z_size, self.c_size, encoder.e_size):
            cells.append(
                RSSMCell(
                    h_dim=h_dim,
                    z_dim=z_dim,
                    e_dim=e_dim,
                    c_dim=c_dim,
                    residual_posterior=residual_posterior,
                )
            )
        self.cells = nn.ModuleList(cells)

        self.receptive_field = self.encoder.receptive_field
        self.overall_stride = self.encoder.overall_stride

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
        encodings = [enc.unbind(1) for enc in encodings]

        # initial context for top layer
        context_l = [self.cells[0].get_empty_context(batch_size=x.size(0))] * len(encodings[-1])

        # initial RSSM state (z, h)
        states = [cell.get_initial_state(batch_size=x.size(0)) for cell in self.cells] if state0 is None else state0
        kl_divs = [[] for _ in range(self.num_levels)]
        for l in range(self.num_levels - 1, -1, -1):
            all_states = []
            all_distributions = []
            T_l = len(encodings[l])
            for t in range(T_l):
                # reset stochastic state whenever the layer above ticks (never reset top)
                if self.with_resets and (l < self.num_levels - 1) and (t % self.strides[l + 1] == 0):
                    states[l] = self.cells[l].get_initial_state(batch_size=x.size(0))

                # cell forward
                states[l], distributions = self.cells[l](encodings[l][t], states[l], context_l[t])

                all_states.append(states[l])
                all_distributions.append(distributions)

            # update context_l for below layer as cat(z_l, h_l)
            context_l = [torch.cat(all_states[t], dim=-1) for t in range(T_l)]
            if l >= 1:
                if self.context_decoder is None:
                    # repeat context to temporal resolution of below layer
                    context_l = [context_l[i // self.strides[l]] for i in range(T_l * self.strides[l])]
                else:
                    # use context decoder to increase temporal resolution
                    # TODO context decoder could be a single decoder with as many layers as encoder (or optionally fewer)
                    context_l = torch.stack(context_l, dim=1)
                    context_l = self.context_decoder.levels[l-1](context_l)
                    context_l = context_l.unbind(1)

            # compute kl divergence
            enc_mu = torch.stack([all_distributions[t].enc_mu for t in range(T_l)], dim=1)
            enc_sd = torch.stack([all_distributions[t].enc_sd for t in range(T_l)], dim=1)
            prior_mu = torch.stack([all_distributions[t].prior_mu for t in range(T_l)], dim=1)
            prior_sd = torch.stack([all_distributions[t].prior_sd for t in range(T_l)], dim=1)

            kld = kl_divergence_gaussian(enc_mu, enc_sd, prior_mu, prior_sd)
            kl_divs[l] = kld

        context_l = torch.stack(context_l, dim=1)
        dec = self.decoder(context_l)

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
        use_mode_latents: bool = False,
        use_mode_observations: bool = False,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        # initial RSSM state (z, h)
        states = [cell.get_initial_state(batch_size=n_samples) for cell in self.cells] if state0 is None else state0

        if x is not None:
            raise NotImplementedError("Conditional generation is not implemented")
            # TODO Run a forward pass over x to get conditional initial state0 (assert state0 is None)
            #      Construct context_l as concatenation of state0 of layer above (empty for top layer)
        else:
            # initial context_l for top layer
            context_l = [self.cells[0].get_empty_context(batch_size=n_samples)] * (
                max_timesteps // self.time_factors[-1]
            )

        for l in range(self.num_levels - 1, -1, -1):
            all_states = []
            all_distributions = []
            T_l = max_timesteps // self.time_factors[l] if l == self.num_levels - 1 else len(context_l)
            for t in range(T_l):
                # reset stochastic state whenever the layer above ticks (never reset top)
                if self.with_resets and (l < self.num_levels - 1) and (t % self.strides[l + 1] == 0):
                    states[l] = self.cells[l].get_initial_state(batch_size=x.size(0))

                # cell forward
                states[l], distributions = self.cells[l].generate(
                    states[l], context_l[t], use_mode=use_mode or use_mode_latents
                )

                all_states.append(states[l])
                all_distributions.append(distributions)

            # update context_l for below layer as cat(z_l, h_l)
            context_l = [torch.cat(all_states[t], dim=-1) for t in range(T_l)]
            if l >= 1:
                if self.context_decoder is None:
                    # repeat context to temporal resolution of below layer
                    context_l = [context_l[i // self.strides[l]] for i in range(T_l * self.strides[l])]
                else:
                    # use context decoder to increase temporal resolution
                    # TODO context decoder could be a single decoder with as many layers as encoder (or optionally fewer)
                    context_l = torch.stack(context_l, dim=1)
                    context_l = self.context_decoder.levels[l-1](context_l)
                    context_l = context_l.unbind(1)

        context_l = torch.stack(context_l, dim=1)
        dec = self.decoder(context_l)

        parameters = self.likelihood(dec)

        if use_mode or use_mode_observations:
            x = self.likelihood.mode(parameters)
        else:
            x = self.likelihood.sample(parameters)

        x_sl = torch.ones(x.shape[1], dtype=torch.int) * max_timesteps
        outputs = SimpleNamespace()  # SimpleNamespace(context_l=context_l, all_distributions=all_distributions)
        return (x, x_sl), outputs


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

        encoder = DenseAudioEncoder(
            h_size=h_size,
            time_factors=time_factors,
            num_level_layers=num_level_layers,
        )

        decoder = DenseAudioDecoder(
            in_dim=bot_c_size,
            h_dim=bot_h_size,
            o_dim=likelihood.out_features,
            time_factor=time_factors[0],
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
        self.receptive_field = self.cwvae.receptive_field
        self.overall_stride = self.cwvae.overall_stride

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
        use_mode_latents: bool = False,
        use_mode_observations: bool = False,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
            use_mode_latents=use_mode_latents,
            use_mode_observations=use_mode_observations,
            x=x,
            state0=state0,
        )


class CWVAEAudioCPCPretrained(BaseModel):
    def __init__(
        self,
        z_size: Union[int, List[int]] = 64,
        h_size: Union[int, List[int]] = 256,
        time_factors: List[int] = [5, 20, 40, 80, 160],
        residual_posterior: bool = False,
        num_level_layers: int = 1,
        num_mix: int = 10,
        num_bins: int = 256,
        frozen_encoder: bool = False,
        pretrained_encoder: bool = False,
    ):
        super().__init__()

        assert h_size == 256 or all([hs == 256 for hs in h_size]), "Pretrained model with these dimensions"
        assert time_factors == [5, 20, 40, 80, 160][-len(time_factors) :], "Pretrained model with these time factors"

        self.z_size = z_size
        self.h_size = h_size
        self.residual_posterior = residual_posterior
        self.num_level_layers = num_level_layers
        self.num_mix = num_mix
        self.num_bins = num_bins
        self.time_factors = time_factors
        self.frozen_encoder = frozen_encoder
        self.pretrained_encoder = pretrained_encoder

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
        # likelihood = DiscretizedLaplaceMixtureDense(
        #     x_dim=3 * num_mix,
        #     y_dim=1,
        #     num_mix=num_mix,
        #     num_bins=num_bins,
        #     reduce_dim=-1,
        # )

        encoder = CPCEncoder(
            num_levels=len(time_factors),
            freeze_parameters=frozen_encoder,
            pretrained=pretrained_encoder,
        )

        decoder = CPCDecoder(
            in_dim=bot_c_size,
            o_dim=likelihood.out_features,
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
        self.receptive_field = self.cwvae.receptive_field
        self.overall_stride = self.cwvae.overall_stride

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
        use_mode_latents: bool = False,
        use_mode_observations: bool = False,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
            use_mode_latents=use_mode_latents,
            use_mode_observations=use_mode_observations,
            x=x,
            state0=state0,
        )


class CWVAEAudioConv1d(BaseModel):
    def __init__(
        self,
        z_size: Union[int, List[int]] = 64,
        h_size: int = 128,
        time_factors: Union[int, List[int]] = 6,
        residual_posterior: bool = False,
        num_level_layers: int = 8,
        num_mix: int = 10,
        num_bins: int = 256,
    ):
        super().__init__()

        self.z_size = z_size
        self.h_size = h_size
        self.residual_posterior = residual_posterior
        self.num_level_layers = num_level_layers
        self.num_mix = num_mix
        self.num_bins = num_bins
        self.time_factors = time_factors
        self.num_levels = len(time_factors)

        bot_z_size = z_size if isinstance(z_size, int) else z_size[0]
        bot_h_size = h_size if isinstance(h_size, int) else h_size[0]
        bot_c_size = bot_z_size + bot_h_size

        likelihood = DiscretizedLogisticMixtureDense(
            x_dim=16,
            y_dim=1,
            num_mix=num_mix,
            num_bins=num_bins,
            reduce_dim=-1,
        )

        encoder = AudioEncoderConv1d(num_levels=self.num_levels)

        decoder = AudioDecoderConv1d(num_levels=self.num_levels - 1)

        self.cwvae = CWVAE(
            encoder=encoder,
            decoder=decoder,
            context_decoder=ContextDecoderConv1d(),
            likelihood=likelihood,
            z_size=z_size,
            h_size=h_size,
            time_factors=time_factors,
            residual_posterior=residual_posterior,
        )
        self.receptive_field = self.cwvae.receptive_field
        self.overall_stride = self.cwvae.overall_stride

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
        use_mode_latents: bool = False,
        use_mode_observations: bool = False,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
            use_mode_latents=use_mode_latents,
            use_mode_observations=use_mode_observations,
            x=x,
            state0=state0,
        )


class CWVAEAudioTasNet(BaseModel):
    def __init__(
        self,
        z_size: Union[int, List[int]] = 64,
        h_size: Union[int, List[int]] = 128,
        time_factors: Union[int, List[int]] = [64, 512, 4096],
        residual_posterior: bool = False,
        num_level_layers: int = 8,
        num_mix: int = 10,
        num_bins: int = 256,
        norm_type: str = "ChannelwiseLayerNorm",
    ):
        super().__init__()

        # TODO Kernel size given as overlap factor

        self.z_size = z_size
        self.h_size = h_size
        self.time_factors = time_factors
        self.residual_posterior = residual_posterior
        self.num_level_layers = num_level_layers
        self.num_mix = num_mix
        self.num_bins = num_bins
        self.norm_type = norm_type

        self.num_levels = len(time_factors)

        bot_z_size = z_size if isinstance(z_size, int) else z_size[0]
        bot_h_size = h_size if isinstance(h_size, int) else h_size[0]
        bot_c_size = bot_z_size + bot_h_size

        assert all(h_size[0] == hs for hs in h_size)
        h_size = h_size[0]

        likelihood = DiscretizedLogisticMixtureDense(
            x_dim=3 * num_mix,
            y_dim=1,
            num_mix=num_mix,
            num_bins=num_bins,
            reduce_dim=-1,
        )

        encoder = TasNetEncoder(
            channels_bottleneck=h_size,
            channels_block=4 * h_size,
            time_factors=time_factors,
            kernel_size=5,
            num_blocks=num_level_layers,
            num_levels=self.num_levels,
            norm_type=norm_type,
        )

        decoder = TasNetDecoder(
            time_factor=time_factors[0],
            channels_in=bot_c_size,
            channels_bottleneck=h_size,
            channels_block=4 * h_size,
            channels_out=likelihood.out_features,
            kernel_size=5,
            num_blocks=num_level_layers,
            norm_type=norm_type,
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
        self.receptive_field = self.cwvae.receptive_field
        self.overall_stride = self.cwvae.overall_stride

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
        use_mode_latents: bool = False,
        use_mode_observations: bool = False,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
            use_mode_latents=use_mode_latents,
            use_mode_observations=use_mode_observations,
            x=x,
            state0=state0,
        )
