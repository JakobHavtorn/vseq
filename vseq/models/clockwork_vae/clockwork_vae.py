from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from torchtyping import TensorType

from vseq.evaluation.metrics import BitsPerDimMetric, KLMetric, LLMetric, LatestMeanMetric, LossMetric
from vseq.models.base_model import BaseModel
from vseq.modules.distributions import DiscretizedLogisticMixtureDense, GaussianDense
from vseq.utils.variational import discount_free_nats, kl_divergence_gaussian, rsample_gaussian
from vseq.utils.operations import sequence_mask


def get_exponential_time_factors(abs_factor, n_levels):
    """Return exponentially increasing temporal abstraction factors with base `abs_factor`"""
    return [abs_factor ** l for l in range(n_levels)]


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

        z_new = rsample_gaussian(enc_mu, enc_sd)

        distributions = SimpleNamespace(enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=prior_mu, prior_sd=prior_sd)

        return (z_new, h_new), distributions

    @torch.jit.export
    def generate(self, state: Tuple[torch.Tensor, torch.Tensor], context: torch.Tensor, use_mode: bool = False):
        z, h = state

        gru_in = self.gru_in(torch.cat([z, context], dim=-1))
        h_new = self.gru_cell(gru_in, h)

        prior_mu, prior_sd = self.prior(h_new)

        z_new = rsample_gaussian(prior_mu, prior_sd)

        distributions = SimpleNamespace(prior_mu=prior_mu, prior_sd=prior_sd)

        return (z_new, h_new), distributions


class StackWaveform(nn.Module):
    def __init__(self, stack_size: int, pad_value: float = 0.0):
        super().__init__()
        self.stack_size = stack_size
        self.pad_value = pad_value

    def forward(self, x: TensorType["B":..., "T"], x_sl: TensorType["B", int] = None):
        padding = (self.stack_size - x.size(-1) % self.stack_size) % self.stack_size
        x = torch.cat(
            [
                x,
                torch.full(
                    (
                        *x.shape[:-1],
                        padding,
                    ),
                    fill_value=self.pad_value,
                    device=x.device,
                ),
            ],
            dim=-1,
        )
        x = x.view(*x.shape[:-1], -1, self.stack_size)  # (B, ..., T / stack_size, stack_size)
        if x_sl is None:
            return x, padding
        x_sl = (x_sl + padding) // self.stack_size
        return x, x_sl, padding

    def reverse(self, x: TensorType["B":..., "T"], padding: Optional[int] = None, x_sl: TensorType["B", int] = None):
        x = x.view(*x.shape[:-2], x.shape[-2] * self.stack_size)
        if padding is None:
            return x

        x = x[..., :-padding]
        if x_sl is None:
            return x

        x_sl = x_sl + self.stack_size - padding
        return x, x_sl


class MultiLevelEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorType["B", "T", "C"]) -> List[TensorType["B", "T", "D"]]:
        raise NotImplementedError()


class MultiLevelEncoderAudioDense(MultiLevelEncoder):
    def __init__(
        self,
        h_size: Union[int, List[int]],
        time_factors: List[int],
        proj_size: Union[int, List[int]] = None,
        n_dense: int = 3,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.time_factors = time_factors
        self.proj_size = proj_size
        self.n_dense = n_dense
        self.activation = activation

        n_levels = len(time_factors)

        h_size = [h_size] * n_levels if isinstance(h_size, int) else h_size
        proj_size = [proj_size] * n_levels if isinstance(proj_size, int) else proj_size

        project_out = (proj_size is not None) and (proj_size > 0)

        self.stack_waveform = StackWaveform(time_factors[0], pad_value=0)  # float('nan'))

        self.levels = nn.ModuleList()
        self.levels.extend([self.get_level(time_factors[0], h_size[0], n_dense, activation)])
        self.levels.extend([self.get_level(h_size[l - 1], h_size[l], n_dense, activation) for l in range(1, n_levels)])

        if project_out:
            self.out_proj = nn.ModuleList(
                [self.get_level(h_size[l], proj_size, 1, activation) for l in range(n_levels)]
            )

        self.n_levels = n_levels
        self.h_size = h_size
        self.project_out = project_out
        self.out_size = proj_size if project_out else h_size

    @staticmethod
    def get_level(in_dim, h_dim, n_dense, activation, o_dim: int = None):
        o_dim = h_dim if o_dim is None else o_dim

        level = [nn.Linear(in_dim, h_dim), activation()]
        for _ in range(1, n_dense - 1):
            level.extend([nn.Linear(h_dim, h_dim), activation()])

        level.extend([nn.Linear(h_dim, o_dim), activation()])
        return nn.Sequential(*level)

    def compute_encoding(self, level: int, pre_enc: TensorType["B", "T", "D"]) -> TensorType["B", "T//factor", "D"]:
        B, T, D = pre_enc.shape
        n_merge_steps = int(
            self.time_factors[level] / self.time_factors[0]
        )  # first step merge done by stacking waveform
        n_pad_steps = (n_merge_steps - T % n_merge_steps) % n_merge_steps
        padding = (0, 0, 0, n_pad_steps, 0, 0)  # pad D-dim by (0, 0) and T-dim by (0, N) and B-dim by (0, 0)
        pre_enc = torch.nn.functional.pad(pre_enc, padding, mode="constant", value=0)
        pre_enc = pre_enc.view(B, -1, n_merge_steps, D)
        enc = pre_enc.sum(2)
        return enc

    def forward(self, x: TensorType["B", "T", "D", float]) -> List[TensorType["B", "T", "D", float]]:
        """Encode a sequence of inputs to multiple representations at different timescales.

        The representation at level `l` will be of length `T // time_factors[l]`.

        Args:
            x (torch.Tensor): Input sequence

        Returns:
            List[torch.Tensor]: List of encodings per level
        """
        encodings = []
        hidden, padding = self.stack_waveform(x)
        # import IPython; IPython.embed(using=False)
        for l in range(self.n_levels):
            hidden = self.levels[l](hidden)
            pre_enc = self.out_proj[l](hidden) if self.project_out else hidden
            encoding = self.compute_encoding(l, pre_enc) if self.time_factors[l] != 1 else pre_enc
            encodings.append(encoding)
        return encodings


class MultiLevelEncoderConv1D(MultiLevelEncoder):
    def __init__(
        self,
        in_dim: int,
        h_size: Union[int, List[int]],
        time_factors: List[int],
        proj_size: Union[int, List[int]] = None,
        n_dense: int = 3,
        activation: nn.Module = nn.ReLU,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.time_factors = time_factors
        self.proj_size = proj_size
        self.n_dense = n_dense
        self.activation = activation
        self.pad_mode = pad_mode
        self.pad_value = pad_value

        n_levels = len(time_factors)

        h_size = [h_size] * n_levels if isinstance(h_size, int) else h_size
        proj_size = [proj_size] * n_levels if isinstance(proj_size, int) else proj_size

        project_out = (proj_size is not None) and (proj_size > 0)

        self.levels = nn.ModuleList()
        self.levels.extend([self.get_level(in_dim, h_size[0], n_dense, activation)])
        self.levels.extend([self.get_level(h_size[l - 1], h_size[l], n_dense, activation) for l in range(1, n_levels)])

        if project_out:
            self.out_proj = nn.ModuleList(
                [self.get_level(h_size[l], proj_size, 1, activation) for l in range(n_levels)]
            )

        self.n_levels = n_levels
        self.h_size = h_size
        self.project_out = project_out
        self.out_size = proj_size if project_out else h_size

    @staticmethod
    def get_level(in_dim, h_dim, n_dense, activation):
        level = [nn.Linear(in_dim, h_dim), activation()]
        for _ in range(1, n_dense):
            level.extend([nn.Linear(h_dim, h_dim), activation()])
        return nn.Sequential(*level)

    def compute_encoding(self, level: int, pre_enc: TensorType["B", "T", "D"]) -> TensorType["B", "T//factor", "D"]:
        B, T, D = pre_enc.shape
        n_merge_steps = self.time_factors[level]
        n_pad_steps = (n_merge_steps - T % n_merge_steps) % n_merge_steps
        padding = (0, 0, 0, n_pad_steps, 0, 0)  # pad D-dim by (0, 0) and T-dim by (0, N) and B-dim by (0, 0)
        pre_enc = torch.nn.functional.pad(pre_enc, padding, mode=self.pad_mode, value=self.pad_value)
        pre_enc = pre_enc.view(B, -1, n_merge_steps, D)
        enc = pre_enc.sum(2)
        return enc

    def forward(self, x: TensorType["B", "T", "D", float]) -> List[TensorType["B", "T", "D", float]]:
        """Encode a sequence of inputs to multiple representations at different timescales.

        The representation at level `l` will be of length `T // time_factors[l]`.

        Args:
            x (torch.Tensor): Input sequence

        Returns:
            List[torch.Tensor]: List of encodings per level
        """
        hidden = x
        encodings = []
        for l in range(self.n_levels):
            hidden = self.levels[l](hidden)
            pre_enc = self.out_proj[l](hidden) if self.project_out else hidden
            encoding = self.compute_encoding(l, pre_enc) if self.time_factors[l] != 1 else pre_enc
            encodings.append(encoding)
        return encodings


class EncoderAudioDense(MultiLevelEncoderAudioDense):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        proj_dim: int,
        n_dense: int,
        activation: nn.Module,
        pad_mode: str,
        pad_value: float,
    ):
        super().__init__(
            in_dim=in_dim,
            h_size=h_dim,
            proj_size=proj_dim,
            n_dense=n_dense,
            activation=activation,
            pad_mode=pad_mode,
            pad_value=pad_value,
            time_factors=[1],
        )

    def forward(self, x: TensorType["B", "T", "D", float]) -> TensorType["B", "T", "D", float]:
        return super().forward(self, x)[0]


class DecoderAudioDense(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        o_dim: int,
        time_factors: List[int],
        n_dense: int = 3,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.o_dim = o_dim
        self.time_factors = time_factors
        self.n_dense = n_dense
        self.activation = activation
        self.decoder = MultiLevelEncoderAudioDense.get_level(
            in_dim, h_dim, n_dense, activation, o_dim=o_dim * time_factors[0]
        )
        self.stack_waveform = StackWaveform(time_factors[0], pad_value=0)  # float('nan'))

    def forward(self, x):
        # import IPython; IPython.embed()
        hidden = self.decoder(x)
        hidden = hidden.view(hidden.size(0), -1, self.o_dim)
        # hidden = self.stack_waveform.reverse(hidden)
        return hidden


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
        assert encoder.n_levels == len(z_size), "Number of levels in encoder and in latent dimensions must match"

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood
        self.n_levels = len(time_factors)
        self.time_factors = time_factors
        self.residual_posterior = residual_posterior

        self.z_size = [z_size] * self.n_levels if isinstance(z_size, int) else z_size
        self.h_size = [h_size] * self.n_levels if isinstance(h_size, int) else h_size
        self.c_size = [z_dim + h_dim for z_dim, h_dim in zip(self.z_size[1:], self.h_size[1:])] + [0]

        cells = []
        for h_dim, z_dim, c_dim, e_dim in zip(self.h_size, self.z_size, self.c_size, encoder.out_size):
            cells.append(
                RSSMCell(h_dim=h_dim, z_dim=z_dim, e_dim=e_dim, c_dim=c_dim, residual_posterior=residual_posterior)
            )
        self.cells = nn.ModuleList(cells)

    def build_metrics(self, x_sl, loss, elbo, log_prob, kld, klds, beta, free_nats):
        kld_metrics_nats = [
            KLMetric(klds[l], name=f"kl_{l} (nats)", log_to_console=False) for l in range(self.n_levels)
        ]
        kld_metrics_bpd = [
            BitsPerDimMetric(-klds[l], name=f"kl_{l} (bpt)", reduce_by=(x_sl / self.time_factors[l]))
            for l in range(self.n_levels)
        ]
        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo (nats)"),
            BitsPerDimMetric(elbo, name="elbo (bpt)", reduce_by=x_sl),
            LLMetric(log_prob, name="rec (nats)", log_to_console=False),
            BitsPerDimMetric(log_prob, name="rec (bpt)", reduce_by=x_sl),
            KLMetric(kld, name="kl (nats)"),
            BitsPerDimMetric(-kld, name="kl (bpt)", reduce_by=x_sl / self.time_factors[0]),
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
        for l in range(self.n_levels):
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

        kl_divs = [[] for _ in range(self.n_levels)]
        t = 0
        for l in range(self.n_levels - 1, -1, -1):
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

        for l in range(self.n_levels - 1, -1, -1):
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


class CWVAEAudio(BaseModel):
    def __init__(
        self,
        num_embeddings: Optional[int] = None,
        z_size: Union[int, List[int]] = 64,
        h_size: Union[int, List[int]] = 128,
        time_factors: Union[int, List[int]] = 6,
        n_levels: int = 3,
        residual_posterior: bool = False,
        n_dense: int = 3,
        num_mix: int = 10,
        num_bins: int = 256,
        stack_size: int = 200,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.z_size = z_size
        self.h_size = h_size
        self.n_levels = n_levels
        self.residual_posterior = residual_posterior
        self.n_dense = n_dense
        self.num_mix = num_mix
        self.num_bins = num_bins
        self.stack_size = stack_size

        if isinstance(time_factors, int):
            time_factors = get_exponential_time_factors(time_factors, self.n_levels)

        self.time_factors = time_factors

        bot_z_size = z_size if isinstance(z_size, int) else z_size[0]
        bot_h_size = h_size if isinstance(h_size, int) else h_size[0]
        bot_c_size = bot_z_size + bot_h_size

        if num_embeddings is not None:
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=bot_h_size)
        else:
            self.embedding = None

        encoder = MultiLevelEncoderAudioConv(
            in_dim=input_size,
            h_size=h_size,
            time_factors=time_factors,
            n_dense=n_dense,
        )

        decoder = DecoderAudioConv(
            in_dim=bot_c_size,
            h_dim=bot_h_size,
            n_dense=n_dense,
        )

        likelihood = DiscretizedLogisticMixtureDense(
            x_dim=bot_h_size,
            y_dim=input_size,
            num_mix=num_mix,
            num_bins=num_bins,
            reduce_dim=-1,
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
        #     x, x_sl_stacked, padding = self.stack_waveform(x, x_sl)
        #     loss, metrics, outputs = self.cwvae(x, x_sl_stacked, state0, beta, free_nats, x_sl_norm=x_sl)
        #     outputs.x_hat, _ = self.stack_waveform.reverse(outputs.x_hat, x_sl_stacked, padding)

        loss, metrics, outputs = self.cwvae(x, x_sl_stacked, state0, beta, free_nats)
        return loss, metrics, outputs


class CWVAEAudioDense(BaseModel):
    def __init__(
        self,
        z_size: Union[int, List[int]] = 64,
        h_size: Union[int, List[int]] = 128,
        time_factors: Union[int, List[int]] = 6,
        n_levels: int = 3,
        residual_posterior: bool = False,
        n_dense: int = 3,
        num_mix: int = 10,
        num_bins: int = 256,
        stack_size: int = 200,
    ):
        super().__init__()

        self.z_size = z_size
        self.h_size = h_size
        self.n_levels = n_levels
        self.residual_posterior = residual_posterior
        self.n_dense = n_dense
        self.num_mix = num_mix
        self.num_bins = num_bins
        self.stack_size = stack_size

        if isinstance(time_factors, int):
            time_factors = get_exponential_time_factors(time_factors, self.n_levels)

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
            n_dense=n_dense,
        )

        decoder = DecoderAudioDense(
            in_dim=bot_c_size,
            h_dim=bot_h_size,
            o_dim=likelihood.out_features,
            time_factors=time_factors,
            n_dense=n_dense,
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

