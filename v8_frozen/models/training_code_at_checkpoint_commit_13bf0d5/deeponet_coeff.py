"""Coefficient DeepONet: payoff branch + parameter branch -> combined latent,
combined (elementwise) with a time trunk, decoded to a coefficient
correction -- with the SAME exact initial-condition embedding as the MLP
baseline (learning.mlp_coeff):

    c_theta(tau) = c_h^0 + tau * DeepONet_theta(tau_norm, mu_tilde, psi)

so c_theta(0) = c_h^0 holds at machine precision regardless of weights,
exactly as required by the shared CoefficientModel interface
(learning.interface).

Architecture (a common practical generalization of the scalar-output
DeepONet of Lu et al. to vector-valued outputs): the payoff branch encodes
psi, the parameter branch encodes mu_tilde, their concatenation is
projected to a shared latent code; the time trunk encodes tau_norm into a
latent code of the same dimension; the two latent codes are combined by an
elementwise product (the DeepONet-style bilinear coupling between the
branch's "what" and the trunk's "when"); a final linear decoder maps the
combined latent to the n_dof coefficient correction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from learning.interface import to_tensor


def _mlp(in_dim: int, out_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    dims = (in_dim, *hidden_dims)
    layers: list[nn.Module] = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), nn.Tanh()]
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


class DeepONetCoeffNet(nn.Module):
    """DeepONet: branch(psi, mu) (elementwise-)combined with trunk(tau) -> decoder -> coefficients.

    Args:
        mu_dim: Dimension of the normalized parameter vector (6 for CGMY).
        psi_dim: Dimension of the payoff coefficient vector (n_dof).
        latent_dim: Shared latent dimension between the combined branch and the trunk.
        branch_hidden: Hidden layer sizes for both branch sub-networks.
        trunk_hidden: Hidden layer sizes for the time trunk.
        psi_scale: Divisor applied to raw psi in predict_trajectory (see
            learning.mlp_coeff.MLPCoeffNet for why this matters -- same
            saturated-tanh failure mode applies here).
    """

    def __init__(
        self,
        mu_dim: int = 6,
        psi_dim: int = 66,
        latent_dim: int = 64,
        branch_hidden: tuple[int, ...] = (128, 128),
        trunk_hidden: tuple[int, ...] = (64, 64),
        psi_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.mu_dim = mu_dim
        self.psi_dim = psi_dim
        self.latent_dim = latent_dim
        self.psi_scale = psi_scale
        self.out_dim = psi_dim

        self.payoff_branch = _mlp(psi_dim, latent_dim, branch_hidden)
        self.param_branch = _mlp(mu_dim, latent_dim, branch_hidden)
        self.branch_combine = nn.Linear(2 * latent_dim, latent_dim)
        self.time_trunk = _mlp(1, latent_dim, trunk_hidden)
        self.decoder = nn.Linear(latent_dim, self.out_dim)

    def forward_correction(self, tau_norm: torch.Tensor, mu_tilde: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """The DeepONet's raw output (before multiplying by tau), for inspection/testing."""
        b_psi = self.payoff_branch(psi)
        b_mu = self.param_branch(mu_tilde)
        branch_latent = self.branch_combine(torch.cat([b_psi, b_mu], dim=-1))
        trunk_latent = self.time_trunk(tau_norm)
        combined = branch_latent * trunk_latent
        return self.decoder(combined)

    def forward(
        self,
        tau: torch.Tensor,
        tau_norm: torch.Tensor,
        mu_tilde: torch.Tensor,
        psi: torch.Tensor,
        c0: torch.Tensor,
    ) -> torch.Tensor:
        """c_theta(tau) = c0 + tau * DeepONet_theta(tau_norm, mu_tilde, psi)."""
        correction = self.forward_correction(tau_norm, mu_tilde, psi)
        return c0 + tau * correction

    def predict_trajectory(
        self,
        mu_tilde: np.ndarray,
        psi: np.ndarray,
        c0: np.ndarray,
        time_grid: np.ndarray,
        T: float,
    ) -> np.ndarray:
        """CoefficientModel interface: predict the whole trajectory at once.
        ``psi`` is RAW (unscaled); rescaled internally by self.psi_scale."""
        from learning.normalization import normalize_tau

        dtype = next(self.parameters()).dtype
        n_steps1 = len(time_grid)
        tau = to_tensor(np.asarray(time_grid).reshape(-1, 1), dtype)
        tau_norm = to_tensor(np.array([[normalize_tau(t, T)] for t in time_grid]), dtype)
        mu_tilde_b = to_tensor(np.tile(mu_tilde, (n_steps1, 1)), dtype)
        psi_b = to_tensor(np.tile(psi / self.psi_scale, (n_steps1, 1)), dtype)
        c0_b = to_tensor(np.tile(c0, (n_steps1, 1)), dtype)

        self.eval()
        with torch.no_grad():
            pred = self(tau, tau_norm, mu_tilde_b, psi_b, c0_b)
        return pred.numpy()
