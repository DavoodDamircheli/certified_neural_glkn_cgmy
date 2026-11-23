"""Galerkin-consistent projection of FNO grid outputs onto the spline space.

The raw FNO (learning.fno.FNO1d) output is a set of pointwise values on a
physical grid -- NOT an element of V_h, and not certifiable directly:
``certification.residuals`` needs a B-spline coefficient vector, and there
is no guarantee a grid function even has an exact representation in V_h.
This module implements Pi_h, the projection of a gridded function onto V_h,
and wraps FNO1d so its PROJECTED output satisfies the shared
CoefficientModel interface (learning.interface) -- including the exact
initial-condition property, via the same shift trick every other model in
this package uses:

    c_theta(tau) = c_h^0 + [Pi_h(u_FNO(tau)) - Pi_h(u_FNO(0))]

Pi_h here is a discrete (collocation/quadrature) L2 projection: build the
B-spline basis-value matrix at the FNO's grid nodes (Phi_jk = phi_k(x_j),
via cgmy_bspline.bspline_basis.phi, reused unchanged), approximate the L2
inner product integral(u * phi_k dx) by the trapezoidal rule on the grid,
and solve with the SAME cached M_h factorization (OperatorCache.apply_M_inv)
every other certification-facing quantity in this project uses. This is a
discrete analogue of assemble_payoff_rhs's exact quadrature (which needs a
closed-form payoff function, not available for a black-box grid output).

``ProjectedFNO`` reports BOTH the raw grid output and its projection
(``predict_trajectory`` returns the projected/certifiable coefficients;
``raw_grid_trajectory`` returns the raw grid values for comparison), per
the task's explicit "report both raw and projected."
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from cgmy.operators import OperatorCache
from cgmy.reference_solver import Mesh
from cgmy_bspline.bspline_basis import active_indices, phi
from cgmy_bspline.projection import eval_solution
from learning.fno import FNO1d
from learning.interface import to_tensor


class ProjectedFNO(nn.Module):
    """FNO whose grid output is projected onto the certified B-spline space.

    Args:
        fno: The underlying grid-based FNO model. Its in_channels must
            equal ``1 + 1 + mu_dim`` (payoff-on-grid, tau_norm, mu_tilde
            broadcast) and out_channels must be 1.
        mesh: Spatial discretization defining the target spline space AND
            the FNO's own grid (grid nodes = linspace(x_min, x_max, N+1)).
            Must be the SAME fixed mesh ``ops`` (below) was built from.
        ops: OperatorCache for ``mesh``, supplying the cached M_h
            factorization Pi_h uses.
        mu_dim: Dimension of the normalized parameter vector.
    """

    def __init__(self, fno: FNO1d, mesh: Mesh, ops: OperatorCache, mu_dim: int = 6) -> None:
        super().__init__()
        if fno.out_channels != 1:
            raise ValueError(f"ProjectedFNO expects a scalar-output FNO1d, got out_channels={fno.out_channels}")
        self.fno = fno
        self.mesh = mesh
        self.ops = ops
        self.mu_dim = mu_dim

        grid_nodes = np.linspace(mesh.x_min, mesh.x_max, mesh.N + 1)
        self.register_buffer("grid_nodes", torch.tensor(grid_nodes, dtype=torch.float64))

        # Pi_h ingredients, built once (mesh-only): basis values at grid
        # nodes and trapezoidal quadrature weights, both reused for every
        # prediction.
        ks = active_indices(mesh.N, mesh.p)
        Phi = np.zeros((len(grid_nodes), len(ks)))
        for j, k in enumerate(ks):
            Phi[:, j] = phi(grid_nodes, k, mesh.p, ops.h, mesh.x_min)
        self._Phi = Phi  # (n_grid, n_dof)

        w = np.full(len(grid_nodes), ops.h)
        w[0] *= 0.5
        w[-1] *= 0.5
        self._quad_weights = w  # trapezoidal rule

    def pi_h(self, grid_values: np.ndarray) -> np.ndarray:
        """Discrete L2 projection of a gridded function onto V_h.

        Args:
            grid_values: (n_grid,) values at self.grid_nodes.
        Returns:
            (n_dof,) B-spline coefficients.
        """
        b = self._Phi.T @ (self._quad_weights * grid_values)
        return self.ops.apply_M_inv(b)

    def _build_input(self, tau_norm: torch.Tensor, mu_tilde: torch.Tensor, psi_grid: torch.Tensor) -> torch.Tensor:
        """Assemble the FNO's (batch, n_grid, in_channels) input: payoff-on-grid
        plus tau_norm/mu_tilde broadcast as constant channels."""
        batch, n_grid = psi_grid.shape
        tau_ch = tau_norm.view(batch, 1, 1).expand(batch, n_grid, 1)
        mu_ch = mu_tilde.view(batch, 1, -1).expand(batch, n_grid, self.mu_dim)
        return torch.cat([psi_grid.unsqueeze(-1), tau_ch, mu_ch], dim=-1)

    def raw_grid_output(self, tau_norm: torch.Tensor, mu_tilde: torch.Tensor, psi_grid: torch.Tensor) -> torch.Tensor:
        """The raw FNO output on the grid, NOT in V_h. (batch, n_grid)."""
        x = self._build_input(tau_norm, mu_tilde, psi_grid)
        return self.fno(x).squeeze(-1)

    def predict_trajectory(
        self,
        mu_tilde: np.ndarray,
        psi: np.ndarray,
        c0: np.ndarray,
        time_grid: np.ndarray,
        T: float,
    ) -> np.ndarray:
        """CoefficientModel interface: PROJECTED coefficients, exact IC via the
        shift trick (see module docstring). ``psi`` is the RAW physical
        payoff coefficient vector; evaluated on the grid via eval_solution
        before being fed to the FNO (which operates on a physical grid, not
        coefficient space)."""
        from learning.normalization import normalize_tau

        dtype = next(self.fno.parameters()).dtype
        grid_info = {"N": self.mesh.N, "h": self.ops.h, "x_min": self.mesh.x_min}
        psi_grid_np = eval_solution(psi, grid_info, self.mesh.p, self.grid_nodes.numpy())

        n_steps1 = len(time_grid)
        tau_norm = to_tensor(np.array([normalize_tau(t, T) for t in time_grid]), dtype)
        mu_tilde_b = to_tensor(np.tile(mu_tilde, (n_steps1, 1)), dtype)
        psi_grid_b = to_tensor(np.tile(psi_grid_np, (n_steps1, 1)), dtype)

        self.eval()
        with torch.no_grad():
            raw = self.raw_grid_output(tau_norm, mu_tilde_b, psi_grid_b).numpy()  # (n_steps1, n_grid)

        projected = np.array([self.pi_h(raw[n]) for n in range(n_steps1)])
        return c0 + (projected - projected[0])

    def raw_grid_trajectory(
        self, mu_tilde: np.ndarray, psi: np.ndarray, time_grid: np.ndarray, T: float
    ) -> np.ndarray:
        """The raw (unprojected, uncertifiable) grid output over the whole
        trajectory, for the "report both raw and projected" comparison.
        Returns (len(time_grid), n_grid)."""
        from learning.normalization import normalize_tau

        dtype = next(self.fno.parameters()).dtype
        grid_info = {"N": self.mesh.N, "h": self.ops.h, "x_min": self.mesh.x_min}
        psi_grid_np = eval_solution(psi, grid_info, self.mesh.p, self.grid_nodes.numpy())

        n_steps1 = len(time_grid)
        tau_norm = to_tensor(np.array([normalize_tau(t, T) for t in time_grid]), dtype)
        mu_tilde_b = to_tensor(np.tile(mu_tilde, (n_steps1, 1)), dtype)
        psi_grid_b = to_tensor(np.tile(psi_grid_np, (n_steps1, 1)), dtype)

        self.eval()
        with torch.no_grad():
            raw = self.raw_grid_output(tau_norm, mu_tilde_b, psi_grid_b).numpy()
        return raw
