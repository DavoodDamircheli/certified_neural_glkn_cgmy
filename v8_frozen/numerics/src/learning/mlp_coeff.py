"""MLP baseline mapping (tau, mu, payoff) to a spline coefficient vector,
with the initial condition embedded structurally rather than learned:

    c_theta(tau) = c_h^0 + tau * N_theta(tau_norm, mu_tilde, psi)

so c_theta(0) = c_h^0 holds at machine precision for ANY weights (random or
trained) -- see tests/test_exact_initial_condition.py. This is the simplest
possible candidate architecture for stage 4 of the six-stage dependency
chain: an integration test of the plumbing (dataset -> normalization ->
network -> certification), not a research model (see Stage E's Gate 3).

Inputs to N_theta itself are (tau_norm, mu_tilde, psi): a normalized time
coordinate, normalized CGMY parameters, and the payoff's B-spline
coefficient vector. c_h^0 is supplied to forward() separately and added
outside the network, exactly matching the formula above -- it is not one of
N_theta's own inputs.

Certification always uses de-normalized physical coefficients: forward()
returns c_theta directly in the same physical units as c_h^0 (no output
normalization to invert), so its output can be handed straight to
certification.residuals without any further transformation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from learning.interface import to_tensor


class MLPCoeffNet(nn.Module):
    """Feed-forward network implementing N_theta; c_theta is assembled outside it.

    Args:
        mu_dim: Dimension of the normalized parameter vector (6 for CGMY:
            C, G, M, Y, r, T).
        psi_dim: Dimension of the payoff coefficient vector (n_dof of the
            fixed mesh; e.g. 66 for N=64, p=3).
        hidden_dims: Sizes of hidden layers.
        psi_scale: Divisor applied to raw psi ONLY inside predict_trajectory
            (the shared CoefficientModel interface) -- forward() itself
            still expects pre-scaled psi, matching Stage E's training
            script, which scales psi externally. Defaults to 100.0 (the
            fixed strike in configs/base.yaml); see learning.normalization
            for why this scaling matters (Gate 3's first, failed overfit
            attempt was exactly this: unnormalized psi saturating tanh).
    """

    def __init__(
        self,
        mu_dim: int = 6,
        psi_dim: int = 66,
        hidden_dims: tuple[int, ...] = (128, 128, 128),
        psi_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.mu_dim = mu_dim
        self.psi_dim = psi_dim
        self.psi_scale = psi_scale
        self.out_dim = psi_dim  # N_theta's output lives in the same space as psi/c_h^0

        in_dim = 1 + mu_dim + psi_dim  # tau_norm, mu_tilde, psi
        dims = (in_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.Tanh()]
        layers.append(nn.Linear(dims[-1], self.out_dim))
        self.net = nn.Sequential(*layers)

    def forward_n_theta(self, tau_norm: torch.Tensor, mu_tilde: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """N_theta(tau_norm, mu, psi) alone, for inspection/testing.

        Args:
            tau_norm: (batch, 1) normalized time coordinate.
            mu_tilde: (batch, mu_dim) normalized CGMY parameters.
            psi: (batch, psi_dim) payoff coefficient vector.
        """
        x = torch.cat([tau_norm, mu_tilde, psi], dim=-1)
        return self.net(x)

    def forward(
        self,
        tau: torch.Tensor,
        tau_norm: torch.Tensor,
        mu_tilde: torch.Tensor,
        psi: torch.Tensor,
        c0: torch.Tensor,
    ) -> torch.Tensor:
        """c_theta(tau) = c0 + tau * N_theta(tau_norm, mu_tilde, psi).

        Args:
            tau: (batch, 1) RAW physical time (the multiplicative prefactor
                -- must be the true tau, not tau_norm, for c_theta(0)=c0 to
                hold exactly).
            tau_norm: (batch, 1) normalized time coordinate, fed to N_theta.
            mu_tilde: (batch, mu_dim) normalized CGMY parameters.
            psi: (batch, psi_dim) payoff coefficient vector.
            c0: (batch, psi_dim) physical initial coefficients c_h^0.

        Returns:
            (batch, psi_dim) physical coefficient vector c_theta(tau).
        """
        correction = self.forward_n_theta(tau_norm, mu_tilde, psi)
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

        ``psi`` here is the RAW (unscaled) payoff coefficient vector;
        internally rescaled by ``self.psi_scale`` before being fed to the
        network (see class docstring).
        """
        from learning.normalization import normalize_tau  # local: avoid import cycle

        dtype = next(self.parameters()).dtype
        n_steps1 = len(time_grid)
        tau = to_tensor(np.asarray(time_grid).reshape(-1, 1), dtype)
        tau_norm = to_tensor(
            np.array([[normalize_tau(t, T)] for t in time_grid]), dtype
        )
        mu_tilde_b = to_tensor(np.tile(mu_tilde, (n_steps1, 1)), dtype)
        psi_b = to_tensor(np.tile(psi / self.psi_scale, (n_steps1, 1)), dtype)
        c0_b = to_tensor(np.tile(c0, (n_steps1, 1)), dtype)

        self.eval()
        with torch.no_grad():
            pred = self(tau, tau_norm, mu_tilde_b, psi_b, c0_b)
        return pred.numpy()
