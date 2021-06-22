from types import SimpleNamespace
from typing import List, Optional, Tuple, Union
from vseq.evaluation.metrics import BitsPerDimMetric, KLMetric, LLMetric, LatestMeanMetric, LossMetric

import torch
import torch.nn as nn

from torchtyping import TensorType

from vseq.models.base_model import BaseModel
from vseq.modules.distributions import DiscretizedLogisticMixtureDense, GaussianDense
from vseq.utils.variational import discount_free_nats, kl_divergence_gaussian, rsample_gaussian
from vseq.utils.operations import sequence_mask


# class RSSMCell(torch.jit.ScriptModule):
class RSSMCell(nn.Module):
    def __init__(self, z_dim: int, h_dim: int, e_dim: int, c_dim: int):
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

    def forward(
        self,
        enc_inputs: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        use_mode: bool = False,
    ):
        # context is the state of the above cell (zeros for top most)
        z, h = state

        gru_in = self.gru_in(torch.cat([z, context], dim=-1))
        h_new = self.gru_cell(gru_in, h)

        prior_mu, prior_sd = self.prior(h_new)

        enc_mu, enc_sd = self.posterior(torch.cat([h_new, enc_inputs], dim=-1))

        z_new = rsample_gaussian(enc_mu, enc_sd)

        distributions = SimpleNamespace(enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=prior_mu, prior_sd=prior_sd)

        return (z_new, h_new), distributions

    def generate(self, state: Tuple[torch.Tensor, torch.Tensor], context: torch.Tensor, use_mode: bool = False):
        z, h = state

        gru_in = self.gru_in(torch.cat([z, context], dim=-1))
        h_new = self.gru_cell(gru_in, h)

        prior_mu, prior_sd = self.prior(h_new)

        z_new = rsample_gaussian(prior_mu, prior_sd)

        distributions = SimpleNamespace(prior_mu=prior_mu, prior_sd=prior_sd)

        return (z_new, h_new), distributions


class MultiLevelEncoder(nn.Module):
    def __init__(self, n_levels: int):
        super().__init__()
        self.n_levels = n_levels

    def forward(self, x: TensorType["B", "T", "C"]) -> List[TensorType["B", "T", "D"]]:
        raise NotImplementedError()


class MultiLevelEncoderAudioStacked(MultiLevelEncoder):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        time_factors: List[int],
        proj_dim: int = None,
        n_dense: int = 3,
        activation: nn.Module = nn.ReLU,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
    ):
        super().__init__(n_levels=len(time_factors))

        self.time_factors = time_factors
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.h_dim = h_dim
        self.n_dense = n_dense
        self.activation = activation
        self.pad_mode = pad_mode
        self.pad_value = pad_value

        n_levels = len(time_factors)

        project_out = (proj_dim is not None) and (proj_dim > 0)

        self.levels = nn.ModuleList()
        self.levels.extend([self.get_level(in_dim, h_dim, n_dense, activation)])
        self.levels.extend([self.get_level(h_dim, h_dim, n_dense, activation) for _ in range(1, n_levels)])

        if project_out:
            self.out_proj = nn.ModuleList([self.get_level(h_dim, proj_dim, 1, activation) for _ in range(n_levels)])

        self.n_levels = n_levels
        self.project_out = project_out
        self.n_dense = n_dense
        self.out_dim = proj_dim if project_out else h_dim

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


class EncoderAudioStacked(MultiLevelEncoderAudioStacked):
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
            h_dim=h_dim,
            proj_dim=proj_dim,
            n_dense=n_dense,
            activation=activation,
            pad_mode=pad_mode,
            pad_value=pad_value,
            time_factors=[1],
        )

    def forward(self, x: TensorType["B", "T", "D", float]) -> TensorType["B", "T", "D", float]:
        return super().forward(self, x)[0]


class DecoderAudioStacked(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        n_dense: int = 3,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.decoder = MultiLevelEncoderAudioStacked.get_level(in_dim, h_dim, n_dense, activation)

    def forward(self, x):
        return self.decoder(x)


class CWVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        likelihood: nn.Module,
        z_size: Union[int, List[int]],
        h_size: Union[int, List[int]],
        time_factors: Optional[List[int]] = None,
    ):
        super().__init__()

        assert len(z_size) == len(h_size), "Must give equal number of levels for stochastic and deterministic state"
        assert time_factors is None or len(time_factors) == len(z_size), "Must give as many time factors as levels"
        assert encoder.n_levels == len(z_size), "Number of levels in encoder and in latent dimensions must match"

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood

        if time_factors is None:
            self.time_factors = self.get_exponential_time_factors(encoder.n_levels)
        else:
            self.time_factors = time_factors

        self.z_size = [z_size] * encoder.n_levels if isinstance(z_size, int) else z_size
        self.h_size = [h_size] * encoder.n_levels if isinstance(h_size, int) else h_size
        self.c_size = [z_dim + h_dim for z_dim, h_dim in zip(self.z_size[1:], self.h_size[1:])] + [0]

        self.n_levels = encoder.n_levels

        cells = []
        for h_dim, z_dim, c_dim in zip(self.h_size, self.z_size, self.c_size):
            cells.append(RSSMCell(h_dim=h_dim, z_dim=z_dim, e_dim=encoder.out_dim, c_dim=c_dim))
        self.cells = nn.ModuleList(cells)

    @staticmethod
    def get_exponential_time_factors(n_levels):
        return [2 ** l for l in range(n_levels)]

    def compute_elbo(
        self,
        y: TensorType["B", "T"],
        parameters: TensorType["B", "T", "D"],
        kld_layerwise: List[TensorType["B", "T", "latent_size"]],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
    ):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""

        seq_mask = sequence_mask(x_sl, dtype=float, device=y.device)
        seq_mask = seq_mask.unsqueeze(-1)

        log_prob_twise = self.likelihood.log_prob(y, parameters) * seq_mask
        log_prob = log_prob_twise.view(y.size(0), -1).sum(1)  # (B,)

        klds = []
        klds_fn = []
        for l in range(self.n_levels):
            mask = seq_mask[:, :: self.time_factors[l]]
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
        d0: TensorType["num_layers", "B", "h_dim"] = None,
        a0: TensorType["num_layers", "B", "h_dim"] = None,
        z0: TensorType["B", "z_dim"] = None,
        beta: float = 1,
        free_nats: float = 0,
    ):
        # target
        y = x.clone().detach()

        # compute encodings
        encodings = self.encoder(x)
        encodings_t = [enc.unbind(1) for enc in encodings]

        # initial context for top layer
        # context = torch.zeros(x.size(0), self.c_size[-1])
        context = [torch.empty(x.size(0), 0, device=x.device)] * len(encodings_t[-1])

        # initial RSSM state (z, h)
        states = [cell.get_initial_state(batch_size=x.size(0)) for cell in self.cells]

        # all_states = [[] for _ in range(self.n_levels)]
        # all_distributions = [[] for _ in range(self.n_levels)]
        all_kl_divergences = [[] for _ in range(self.n_levels)]
        t = 0
        for l in range(self.n_levels - 1, -1, -1):
            all_states = []
            all_distributions = []
            T = len(encodings_t[l])
            for t in range(T):
                # reset stochastic state whenever the layer above ticks (never reset top)
                # concate actions to context??

                # cell forward
                states[l], distributions = self.cells[l](
                    enc_inputs=encodings_t[l][t], state=states[l], context=context[t]
                )

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
            all_kl_divergences[l] = kld

        context = torch.stack(context, dim=1)
        dec = self.decoder(context)

        parameters = self.likelihood(dec)

        loss, elbo, log_prob, kld, klds, seq_mask = self.compute_elbo(
            y, parameters, all_kl_divergences, x_sl, beta, free_nats
        )

        kld_metrics = [KLMetric(klds[l], name=f"kl_{l}") for l in range(self.n_levels)]
        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo"),
            LLMetric(log_prob, name="rec"),
            KLMetric(kld, name="kl"),
            *kld_metrics,
            BitsPerDimMetric(elbo, reduce_by=x_sl),
            LatestMeanMetric(beta, name="beta"),
            LatestMeanMetric(free_nats, name="free_nats"),
        ]
        outputs = SimpleNamespace(elbo=elbo, log_prob=log_prob, kld=kld, y=y, parameters=parameters, seq_mask=seq_mask)
        outputs.x_hat = self.likelihood.sample(parameters)  # TODO Remove
        return loss, metrics, outputs

    def generate(self):
        pass


class CWVAEAudioStacked(BaseModel):
    def __init__(
        self,
        input_size: int,
        z_size: List[int],
        h_size: int,
        time_factors: Union[int, List[int]],
        n_dense: int = 3,
    ):
        super().__init__()

        self.input_size = input_size
        self.z_size = z_size
        self.h_size = h_size
        self.n_dense = n_dense
        self.time_factors = time_factors

        encoder = MultiLevelEncoderAudioStacked()
        decoder = DecoderAudioStacked()
        likelihood = DiscretizedLogisticMixtureDense()
        self.cwvae = CWVAE(encoder, decoder, likelihood, z_size=z_size, time_factors=time_factors)

    def forward(self, x, x_sl):
        return self.cwvae(x, x_sl)
