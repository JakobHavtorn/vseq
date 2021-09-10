import math

from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from torchtyping import TensorType

from vseq.evaluation.metrics import (
    BitsPerDimMetric,
    KLMetric,
    LLMetric,
    LatentActivityMetric,
    LatestMeanMetric,
    LossMetric,
    RunningMeanMetric,
    RunningVarianceMetric,
)
from vseq.models.base_model import BaseModel
from vseq.modules.distributions import DiscretizedLaplaceMixtureDense, DiscretizedLogisticMixtureDense, GaussianDense
from vseq.modules.rssm import RSSMCell, RSSMCellMultiGRU
from vseq.utils.variational import discount_free_nats, kl_divergence_gaussian
from vseq.utils.operations import sequence_mask

from .dense_coders import DenseAudioEncoder, DenseAudioDecoder
from .tasnet_coder import TasNetCoder, TasNetDecoder, TasNetEncoder
from .cpc_coders import CPCDecoder, CPCEncoder
from .conv_coders import AudioEncoderConv1d, AudioDecoderConv1d, ContextDecoderConv1d
from .global_coders import GlobalCoder


def get_exponential_time_factors(abs_factor, num_levels):
    """Return exponentially increasing temporal abstraction factors with base `abs_factor`"""
    return [abs_factor ** l for l in range(num_levels)]


class CWVAE(nn.Module):
    def __init__(
        self,
        z_size: Union[int, List[int]],
        h_size: Union[int, List[int]],
        time_factors: List[int],
        encoder: nn.Module,
        decoder: nn.Module,
        likelihood: nn.Module,
        g_size: Optional[int] = 0,
        context_decoder: Optional[nn.ModuleList] = None,
        residual_posterior: bool = False,
        num_rssm_gru_cells: int = 1,
        with_resets: bool = False,
    ):
        """Clockwork VAE like latent variable model

        Args:
            z_size (Union[int, List[int]]): Size(s) of the temporal latent variables.
            h_size (Union[int, List[int]]): Size(s) of the temporal deterministic variables.
            g_size (Optional[int]): Size of the optional global latent variable. Defaults to 0.
            time_factors (List[int]): Overall strides per layer.
            encoder (nn.Module): Transformation used to infer deterministic representations of input.
            decoder (nn.Module): Transformation used to decode context output by the final layer.
            likelihood (nn.Module): Transformation used to evaluate the likelihood (log_prob) of the reconstruction.
            context_decoder (Optional[nn.ModuleList], optional): Decoder for between-layer contexts. Defaults to None.
            residual_posterior (bool, optional): If True, compute mu_q = mu_q' + mu_p. Defaults to False.
            num_rssm_gru_cells (int, optional): Number of stacked GRU cells used per RSSM cell. Defaults to 1.
            with_resets (bool, optional): Rset state whenever layer above ticks (never reset top). Defaults to False.
        """
        super().__init__()

        assert isinstance(time_factors, list)

        self.encoder = encoder
        self.decoder = decoder
        self.context_decoder = context_decoder
        self.likelihood = likelihood
        self.num_levels = len(time_factors)
        self.time_factors = time_factors
        self.residual_posterior = residual_posterior
        self.num_rssm_gru_cells = num_rssm_gru_cells
        self.g_size = g_size
        self.with_resets = with_resets

        self.strides = [self.time_factors[0]]
        self.strides += [self.time_factors[l] // self.time_factors[l - 1] for l in range(1, self.num_levels)]

        self.z_size = [z_size] * self.num_levels if isinstance(z_size, int) else z_size
        self.h_size = [h_size] * self.num_levels if isinstance(h_size, int) else h_size
        if context_decoder is None:
            self.c_size = [z_dim + h_dim * num_rssm_gru_cells + g_size for z_dim, h_dim in zip(self.z_size[1:], self.h_size[1:])]
            self.c_size += [0 + g_size]  # top context is empty or global latent
        else:
            # When using context decoder that changes num_channels to that of layer below
            self.c_size = [h_dim for z_dim, h_dim in zip(self.z_size[:-1], self.h_size[:-1])] + [0]

        assert (
            len(self.z_size) == len(self.h_size) == len(self.c_size)
        ), f"Must have equal lengths: {self.z_size=}, {self.h_size=}, {self.c_size=}"

        cells = []
        rssm_cell = RSSMCell if num_rssm_gru_cells == 1 else RSSMCellMultiGRU
        for h_dim, z_dim, c_dim, e_dim in zip(self.h_size, self.z_size, self.c_size, encoder.e_size):
            cells.append(
                rssm_cell(
                    h_dim=h_dim,
                    z_dim=z_dim,
                    c_dim=c_dim,
                    e_dim=e_dim,
                    residual_posterior=residual_posterior,
                    n_gru_cells=num_rssm_gru_cells,
                )
            )
        self.cells = nn.ModuleList(cells)

        if g_size > 0:
            self.global_coder = GlobalCoder(g_size, encoder.e_size)

        self.receptive_field = self.encoder.receptive_field
        self.overall_stride = self.encoder.overall_stride

    def build_metrics(self, loss, elbo, log_prob, kld, kld_l, kld_g, enc_mus, prior_mus, x_sl, seq_mask, beta, free_nats):
        kld_layers_metrics_nats = [
            KLMetric(kld_l[l], name=f"kl_{l} (nats)", log_to_console=False) for l in range(self.num_levels)
        ]
        kld_layers_metrics_bpd = [
            KLMetric(kld_l[l] / math.log(2), name=f"kl_{l} (bpt)", reduce_by=(x_sl / self.time_factors[l]))
            for l in range(self.num_levels)
        ]
        latent_activity_metrics_percent = [
            LatentActivityMetric(
                enc_mus[l][:, : x_sl.min()],
                name=f"z_{l} (%)",
                threshold=0.01,
                reduce_by=enc_mus[l].size(0),
                weight_by=enc_mus[l].size(0) * x_sl.min(),
            )
            for l in range(self.num_levels)
        ]
        latent_activity_metrics_variance = [
            LatentActivityMetric(
                enc_mus[l][:, : x_sl.min()],
                name=f"z_{l} (var)",
                reduce_by=enc_mus[l].size(0),
                weight_by=enc_mus[l].size(0) * x_sl.min(),
            )
            for l in range(self.num_levels)
        ]

        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LatestMeanMetric(elbo / math.log(2), name="last elbo (bpt)", reduce_by=-x_sl),
            LLMetric(elbo, name="elbo (nats)"),
            BitsPerDimMetric(elbo, name="elbo (bpt)", reduce_by=x_sl),
            LLMetric(log_prob, name="rec (nats)", log_to_console=False),
            BitsPerDimMetric(log_prob, name="rec (bpt)", reduce_by=x_sl),
            KLMetric(kld, name="kl (nats)"),
            KLMetric(kld / math.log(2), name="kl (bpt)", reduce_by=x_sl / self.time_factors[0]),
            *kld_layers_metrics_nats,
            *kld_layers_metrics_bpd,
            *latent_activity_metrics_percent,
            *latent_activity_metrics_variance,
            LatestMeanMetric(beta, name="beta"),
            LatestMeanMetric(free_nats, name="free_nats"),
        ]

        if kld_g is not None:
            metrics.extend([
                KLMetric(kld_g, name="kl_g (nats)"),
                KLMetric(kld_g / math.log(2), name="kl_g (bits)"),
            ])

        return metrics

    def compute_elbo(
        self,
        y: TensorType["B", "T", "D"],
        seq_mask: TensorType["B", "T", int],
        x_sl: TensorType["B", int],
        parameters: TensorType["B", "T", "D"],
        kld_layerwise: List[TensorType["B", "T", "latent_size"]],
        kld_global: Optional[TensorType["B", "g_size"]] = None,
        beta: float = 1,
        free_nats: float = 0,
    ):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""
        log_prob_twise = self.likelihood.log_prob(y, parameters) * seq_mask
        log_prob = log_prob_twise.view(y.size(0), -1).sum(1)  # (B,)

        kld_l, klds_fn = [], []
        seq_mask = seq_mask.unsqueeze(-1)  # Broadcast to latent dimension
        for l in range(self.num_levels):
            mask = seq_mask[:, :: self.time_factors[l]]  # Mask for subsampled layers  (faster than creating new mask)
            kld_l.append((kld_layerwise[l] * mask).sum((1, 2)))  # (B,)
            klds_fn.append((discount_free_nats(kld_layerwise[l], free_nats, shared_dims=-1) * mask).sum((1, 2)))  # (B,)

        kld, kld_fn = sum(kld_l), sum(klds_fn)

        if kld_global is not None:
            kld_fn_global = discount_free_nats(kld_global, free_nats, shared_dims=-1).sum(-1)
            kld_global = kld_global.sum(-1)

            kld, kld_fn = kld + kld_global, kld_fn + kld_fn_global

        elbo = log_prob - kld  # (B,)

        loss = -(log_prob - beta * kld_fn).sum() / x_sl.sum()  # (1,)

        return loss, elbo, log_prob, kld, kld_l, kld_global

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

        seq_mask = sequence_mask(x_sl, max_len=y.shape[1], dtype=bool, device=y.device)

        # compute encodings
        encodings_list = self.encoder(x)
        encodings = [enc.unbind(1) for enc in encodings_list]  # unbind time dimension (List[Tuple[Tensor]])

        # global latent
        if self.g_size > 0:
            g, g_distributions = self.global_coder(encodings_list, x_sl, self.time_factors, seq_mask=seq_mask)
            kld_g = kl_divergence_gaussian(g_distributions.enc_mu, g_distributions.enc_sd, g_distributions.prior_mu, g_distributions.prior_sd)
        else:
            g = torch.empty(x.size(0), 0, device=y.device)
            kld_g = None

        # initial context for top layer
        context_l = [g] * len(encodings[-1])

        # initial RSSM state (z, h)
        states = [cell.get_initial_state(batch_size=x.size(0)) for cell in self.cells] if state0 is None else state0
        kld_l = [[] for _ in range(self.num_levels)]
        latents = [[] for _ in range(self.num_levels)]
        enc_mus = [[] for _ in range(self.num_levels)]
        prior_mus = [[] for _ in range(self.num_levels)]
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
            context_l = [torch.cat([*all_states[t], g], dim=-1) for t in range(T_l)]
            if l >= 1:
                # TODO context decoder could be a single decoder with as many layers as encoder (or optionally fewer)
                # if self.decoder[l - 1] is None:
                #     # repeat context to temporal resolution of below layer
                #     context_l = [context_l[i // self.strides[l]] for i in range(T_l * self.strides[l])]
                # else:
                #     # use context decoder to increase temporal resolution
                #     context_l = torch.stack(context_l, dim=1)
                #     context_l = self.context_decoder[l - 1](context_l)
                #     context_l = context_l.unbind(1)

                if self.context_decoder is None:
                    # repeat context to temporal resolution of below layer
                    context_l = [context_l[i // self.strides[l]] for i in range(T_l * self.strides[l])]
                else:
                    # use context decoder to increase temporal resolution
                    context_l = torch.stack(context_l, dim=1)
                    context_l = self.context_decoder.levels[l - 1](context_l)
                    context_l = context_l.unbind(1)

            # compute kl divergence
            enc_mu = torch.stack([all_distributions[t].enc_mu for t in range(T_l)], dim=1)
            enc_sd = torch.stack([all_distributions[t].enc_sd for t in range(T_l)], dim=1)
            prior_mu = torch.stack([all_distributions[t].prior_mu for t in range(T_l)], dim=1)
            prior_sd = torch.stack([all_distributions[t].prior_sd for t in range(T_l)], dim=1)

            latents[l] = torch.stack([all_distributions[t].z for t in range(T_l)], dim=1)
            enc_mus[l] = enc_mu
            prior_mus[l] = prior_mu

            kld_l[l] = kl_divergence_gaussian(enc_mu, enc_sd, prior_mu, prior_sd)

        context_l = torch.stack(context_l, dim=1)
        dec = self.decoder(context_l)

        parameters = self.likelihood(dec)
        reconstruction = self.likelihood.sample(parameters)
        reconstruction_mode = self.likelihood.mode(parameters)

        loss, elbo, log_prob, kld, kld_l, kld_g = self.compute_elbo(y, seq_mask, x_sl, parameters, kld_l, kld_g, beta, free_nats)

        metrics = self.build_metrics(
            loss, elbo, log_prob, kld, kld_l, kld_g, enc_mus, prior_mus, x_sl, seq_mask, beta, free_nats
        )

        outputs = SimpleNamespace(
            elbo=elbo,
            log_prob=log_prob,
            kld=kld,
            y=y,
            seq_mask=seq_mask,
            latents=latents,
            enc_mus=enc_mus,
            prior_mus=prior_mus,
            reconstructions=reconstruction,
            reconstructions_mode=reconstruction_mode,
            reconstructions_parameters=parameters,
        )

        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        use_mode_latents: bool = False,
        use_mode_global: bool = False,
        use_mode_observations: bool = False,
        temperature: float = 1,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        # initial RSSM state (z, h)
        states = [cell.get_initial_state(batch_size=n_samples) for cell in self.cells] if state0 is None else state0

        # global latent
        if self.g_size > 0:
            g, g_distributions = self.global_coder.generate(batch_size=n_samples, temperature=temperature, use_mode=use_mode_global)
        else:
            g = self.cells[0].get_empty_context(batch_size=n_samples)

        if x is not None:
            raise NotImplementedError("Conditional generation is not implemented")
            # TODO Run a forward pass over x to get conditional initial state0 (assert state0 is None)
            #      Construct context_l as concatenation of state0 of layer above (empty for top layer)
        else:
            # initial context_l for top layer
            context_l = [g] * (max_timesteps // self.time_factors[-1])

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
                    states[l], context_l[t], temperature=temperature, use_mode=use_mode or use_mode_latents
                )

                all_states.append(states[l])
                all_distributions.append(distributions)

            # update context_l for below layer as cat(z_l, h_l)
            context_l = [torch.cat([*all_states[t], g], dim=-1) for t in range(T_l)]
            if l >= 1:
                if self.context_decoder is None:
                    # repeat context to temporal resolution of below layer
                    context_l = [context_l[i // self.strides[l]] for i in range(T_l * self.strides[l])]
                else:
                    # use context decoder to increase temporal resolution
                    # TODO context decoder could be a single decoder with as many layers as encoder (or optionally fewer)
                    context_l = torch.stack(context_l, dim=1)
                    context_l = self.context_decoder.levels[l - 1](context_l)
                    context_l = context_l.unbind(1)

        context_l = torch.stack(context_l, dim=1)
        dec = self.decoder(context_l)

        parameters = self.likelihood(dec)

        if use_mode or use_mode_observations:
            x = self.likelihood.mode(parameters)
        else:
            x = self.likelihood.sample(parameters)

        x_sl = torch.ones(x.shape[1], dtype=torch.int) * max_timesteps
        outputs = SimpleNamespace(context_l=context_l, all_distributions=all_distributions)
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
        temperature: float = 1,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
            use_mode_latents=use_mode_latents,
            use_mode_observations=use_mode_observations,
            temperature=temperature,
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
        temperature: float = 1,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
            use_mode_latents=use_mode_latents,
            use_mode_observations=use_mode_observations,
            temperature=temperature,
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

        decoder = AudioDecoderConv1d()

        self.cwvae = CWVAE(
            encoder=encoder,
            decoder=decoder,
            context_decoder=ContextDecoderConv1d(num_levels=self.num_levels - 1),
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
        temperature: float = 1,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
            use_mode_latents=use_mode_latents,
            use_mode_observations=use_mode_observations,
            temperature=temperature,
            x=x,
            state0=state0,
        )


class CWVAEAudioTasNet(BaseModel):
    def __init__(
        self,
        z_size: Union[int, List[int]] = 64,
        h_size: Union[int, List[int]] = 128,
        g_size: Optional[int] = 0,
        time_factors: Union[int, List[int]] = [64, 512, 4096],  # Rename strides
        dilations: Union[int, List[int]] = 1,
        residual_posterior: bool = False,
        num_level_layers: int = 8,
        num_rssm_gru_cells: int = 1,
        num_mix: int = 10,
        num_bins: int = 256,
        norm_type: str = "ChannelwiseLayerNorm",
    ):
        super().__init__()

        # TODO Kernel size given as overlap factor

        self.z_size = z_size
        self.h_size = h_size
        self.g_size = g_size
        self.time_factors = time_factors
        self.dilations = dilations
        self.residual_posterior = residual_posterior
        self.num_level_layers = num_level_layers
        self.num_rssm_gru_cells = num_rssm_gru_cells
        self.num_mix = num_mix
        self.num_bins = num_bins
        self.norm_type = norm_type

        self.num_levels = len(time_factors)

        bot_z_size = z_size if isinstance(z_size, int) else z_size[0]
        bot_h_size = h_size if isinstance(h_size, int) else h_size[0]
        bot_c_size = bot_z_size + bot_h_size * num_rssm_gru_cells + g_size

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
            strides=time_factors,
            channels_in=1,
            channels_bottleneck=h_size,
            channels_block=4 * h_size,
            kernel_size=5,
            num_blocks=num_level_layers,
            num_levels=self.num_levels,
            norm_type=norm_type,
            # stride_per_block=2,
            # transposed=False,
        )

        decoder = TasNetDecoder(
            strides=time_factors[0],
            channels_in=bot_c_size,
            channels_bottleneck=h_size,
            channels_block=4 * h_size,
            channels_out=likelihood.out_features,  # [None, None, likelihood.out_features],
            kernel_size=5,
            num_blocks=num_level_layers,
            norm_type=norm_type,
            # stride_per_block=2,
            # transposed=True,
        )

        # if self.num_levels > 1:
        #     context_decoder = TasNetContextDecoder(
        #         time_factor=time_factors[1:],
        #         channels_in=2 * h_size,
        #         channels_bottleneck=h_size,
        #         channels_block=4 * h_size,
        #         channels_out=likelihood.out_features,
        #         kernel_size=5,
        #         num_blocks=num_level_layers,
        #         norm_type=norm_type,
        #     )
        # else:
        #     context_decoder = None

        self.cwvae = CWVAE(
            encoder=encoder,
            decoder=decoder,
            likelihood=likelihood,
            z_size=z_size,
            h_size=h_size,
            time_factors=time_factors,
            residual_posterior=residual_posterior,
            num_rssm_gru_cells=num_rssm_gru_cells,
            g_size=g_size,
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
        temperature: float = 1,
        x: Optional[TensorType["B", "T", "D"]] = None,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
            use_mode_latents=use_mode_latents,
            use_mode_observations=use_mode_observations,
            temperature=temperature,
            x=x,
            state0=state0,
        )
