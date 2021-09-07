from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn

from vseq.modules import GaussianDense


class RSSMCell(torch.jit.ScriptModule):
    def __init__(
        self, z_dim: int, h_dim: int, e_dim: int, c_dim: int, n_gru_cells: int = 3, residual_posterior: bool = False
    ):
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
        self.n_gru_cells = n_gru_cells
        self.residual_posterior = residual_posterior

        self.gru_in = nn.ModuleList()
        self.gru_in.extend([nn.Sequential(nn.Linear(z_dim + c_dim, h_dim), nn.ReLU())])
        self.gru_in.extend([nn.Sequential(nn.Linear(h_dim + c_dim, h_dim), nn.ReLU()) for _ in range(1, n_gru_cells)])

        self.gru_cells = nn.ModuleList([nn.GRUCell(h_dim, h_dim) for _ in range(n_gru_cells)])

        self.prior = nn.Sequential(
            nn.Linear(h_dim * n_gru_cells, h_dim),
            nn.ReLU(),
            GaussianDense(h_dim, z_dim),
        )

        self.posterior = nn.Sequential(
            nn.Linear(h_dim * n_gru_cells + e_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            GaussianDense(h_dim, z_dim),
        )

    def get_initial_state(self, batch_size: int, device: str = None):
        device = device if device is not None else self.prior[0].weight.device
        z = torch.zeros(batch_size, self.z_dim, device=device)
        hs = [torch.zeros(batch_size, self.h_dim, device=device) for _ in range(self.n_gru_cells)]
        return (z, *hs)

    def get_empty_context(self, batch_size: int, device: str = None):
        device = device if device is not None else self.prior[0].weight.device
        return torch.empty(batch_size, 0, device=device)

    def forward(
        self,
        enc_inputs: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        temperature: float = 1.0,
        use_mode: bool = False,
    ):
        z, hs = state[0], state[1:]

        hs_new = [None] * self.n_gru_cells + [z]
        for i in range(self.n_gru_cells):
            gru_in = self.gru_in[i](torch.cat([hs_new[i-1], context], dim=-1))
            hi_new = self.gru_cells[i](gru_in, hs[i])
            hs_new[i] = hi_new
        hs_new.pop(-1)

        hs_new_cat = torch.cat(hs_new, dim=-1)
        prior_mu, prior_sd = self.prior(hs_new_cat)

        enc_mu, enc_sd = self.posterior(torch.cat([hs_new_cat, enc_inputs], dim=-1))

        if self.residual_posterior:
            enc_mu = enc_mu + prior_mu

        z_new = self.posterior[-1].rsample((enc_mu, temperature * enc_sd)) if not use_mode else enc_mu

        distributions = SimpleNamespace(z=z_new, enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=prior_mu, prior_sd=prior_sd)
        return (z_new, *hs_new), distributions

    @torch.jit.export
    def generate(
        self,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        temperature: float = 1.0,
        use_mode: bool = False,
    ):
        z, hs = state[0], state[1:]

        hs_new = [None] * self.n_gru_cells + [z]
        for i in range(self.n_gru_cells):
            gru_in = self.gru_in[i](torch.cat([hs_new[i-1], context], dim=-1))
            hi_new = self.gru_cells[i](gru_in, hs[i])
            hs_new[i] = hi_new
        hs_new.pop(-1)

        hs_new_cat = torch.cat(hs_new, dim=-1)
        prior_mu, prior_sd = self.prior(hs_new_cat)

        z_new = self.prior[-1].rsample((prior_mu, temperature * prior_sd)) if not use_mode else prior_mu

        distributions = SimpleNamespace(z=z_new, prior_mu=prior_mu, prior_sd=prior_sd)
        return (z_new, *hs_new), distributions


class RSSMCellSingleGRU(torch.jit.ScriptModule):
    def __init__(self, z_dim: int, h_dim: int, e_dim: int, c_dim: int, residual_posterior: bool = False, **kwargs):
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
        temperature: float = 1.0,
        use_mode: bool = False,
    ):
        z, h = state

        gru_in = self.gru_in(torch.cat([z, context], dim=-1))
        h_new = self.gru_cell(gru_in, h)

        prior_mu, prior_sd = self.prior(h_new)

        enc_mu, enc_sd = self.posterior(torch.cat([h_new, enc_inputs], dim=-1))

        if self.residual_posterior:
            enc_mu = enc_mu + prior_mu

        z_new = self.posterior[-1].rsample((enc_mu, temperature * enc_sd)) if not use_mode else enc_mu

        distributions = SimpleNamespace(z=z_new, enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=prior_mu, prior_sd=prior_sd)

        return (z_new, h_new), distributions

    @torch.jit.export
    def generate(
        self,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        temperature: float = 1.0,
        use_mode: bool = False,
    ):
        z, h = state

        gru_in = self.gru_in(torch.cat([z, context], dim=-1))
        h_new = self.gru_cell(gru_in, h)

        prior_mu, prior_sd = self.prior(h_new)
        z_new = self.prior[-1].rsample((prior_mu, temperature * prior_sd)) if not use_mode else prior_mu

        distributions = SimpleNamespace(z=z_new, prior_mu=prior_mu, prior_sd=prior_sd)

        return (z_new, h_new), distributions
