"""Loss functions for training the neural surrogates in ``learning``.

The mass-weighted loss (``mass_weighted_loss``) is THE training objective
for the coefficient-regression models in this package:

    L_data = (c_theta - c_h)^T M_h (c_theta - c_h) / (c_h^T M_h c_h + eps)

not plain Euclidean/MSE coefficient error. ``euclidean_coefficient_error``
exists ONLY for logging/diagnostics (e.g. Gate 3's report) and must never be
minimized -- the B-spline coefficients are not an orthonormal basis (M_h is
not the identity), so Euclidean distance between coefficient vectors is not
a meaningful error in the underlying function space.
"""

from __future__ import annotations

import torch

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters
from certification.residuals import _interior_mask
from learning.differentiable_operators import apply_A_batch, apply_M_batch, apply_M_inv_batch


def mass_weighted_loss(
    c_theta: torch.Tensor, c_h: torch.Tensor, mass_matrix: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """L_data = (c_theta - c_h)^T M_h (c_theta - c_h) / (c_h^T M_h c_h + eps), averaged over the batch.

    Args:
        c_theta: (batch, n_dof) predicted coefficients.
        c_h: (batch, n_dof) reference coefficients.
        mass_matrix: (n_dof, n_dof) dense mass matrix M_h (same for every
            row in the batch -- all samples share one fixed mesh).
        eps: Denominator regularizer, guarding against c_h == 0 (e.g. at
            tau=0 for a payoff whose L2 projection happens to vanish).

    Returns:
        Scalar: the per-sample ratio above, averaged over the batch. The
        formula as given is for one (c_theta, c_h) pair; batching by
        averaging the per-sample ratio (not by summing numerators/
        denominators separately first) is this implementation's convention.
    """
    diff = c_theta - c_h
    num = torch.einsum("bi,ij,bj->b", diff, mass_matrix, diff)
    den = torch.einsum("bi,ij,bj->b", c_h, mass_matrix, c_h) + eps
    return (num / den).mean()


def l2_error(c_theta: torch.Tensor, c_h: torch.Tensor, mass_matrix: torch.Tensor) -> torch.Tensor:
    """Absolute physical L2(Omega) error per sample: sqrt((c_theta-c_h)^T M_h (c_theta-c_h)).

    Logging/diagnostic quantity (Gate 3's "L2 error"), not a loss: it is the
    numerator of mass_weighted_loss, unnormalized, since
    ||u_h||_{L2}^2 = c^T M_h c for B-spline coefficients c.
    """
    diff = c_theta - c_h
    return torch.sqrt(torch.clamp(torch.einsum("bi,ij,bj->b", diff, mass_matrix, diff), min=0.0))


def euclidean_coefficient_error(c_theta: torch.Tensor, c_h: torch.Tensor) -> torch.Tensor:
    """Plain Euclidean ||c_theta - c_h|| per sample. LOGGING ONLY -- never a training loss
    (see module docstring)."""
    return torch.linalg.norm(c_theta - c_h, dim=-1)


def torch_cn_residual(
    c_theta_n: torch.Tensor,
    c_theta_np1: torch.Tensor,
    fbar: torch.Tensor,
    dt: float,
    mus: list[CGMYParameters],
    ops: OperatorCache,
    bc_dofs: tuple[list[int], list[int]],
) -> torch.Tensor:
    """Differentiable Crank-Nicolson residual r_theta^{n+1/2}:

        r_theta^{n+1/2} = M_h (c_theta^{n+1}-c_theta^n)/dt + A_h cbar_theta^{n+1/2} - fbar^{n+1/2}

    evaluated through the SAME operator code
    (cgmy.operators.OperatorCache.apply_A/apply_M) that
    certification.residuals.compute_cn_residual uses, via
    learning.differentiable_operators -- not a torch reimplementation of
    A_h. Zero at Dirichlet-boundary dofs, matching
    certification.residuals' convention exactly (see its module docstring
    for why those rows aren't part of "the residual").

    Args:
        c_theta_n, c_theta_np1: (batch, n_dof) predicted coefficients at
            consecutive time levels.
        fbar: (batch, n_dof) CN-averaged forcing (traj.forcing_vectors[n],
            tiled across the batch as needed) -- the SAME stored forcing
            certification uses; never recompute it here (CLAUDE.md Hard
            Invariant #2).
        dt: Time step size.
        mus: Length-batch list of CGMYParameters, one per row (rows may
            come from different trajectories/mu).
        ops: OperatorCache for the shared fixed mesh every row's
            trajectory was solved on.
        bc_dofs: (left, right) Dirichlet dof index lists, from
            traj.solver_metadata.bc_dofs.

    Returns:
        (batch, n_dof) residual, zero at Dirichlet dofs.
    """
    c_bar = 0.5 * (c_theta_np1 + c_theta_n)
    r = apply_M_batch((c_theta_np1 - c_theta_n) / dt, ops) + apply_A_batch(c_bar, mus, ops) - fbar

    mask = _interior_mask(ops.n_dof, bc_dofs)
    mask_t = torch.as_tensor(mask, dtype=r.dtype, device=r.device)
    return r * mask_t


def residual_loss(residuals: torch.Tensor, ops: OperatorCache) -> torch.Tensor:
    """L_res = (1/N_T) sum_n ||r^{n+1/2}||^2_{M_h^{-1}}: the mean, over the
    batch (N_T = number of residual rows in it), of r^T M_h^{-1} r.

    Args:
        residuals: (batch, n_dof) residual vectors, e.g. from torch_cn_residual.
        ops: OperatorCache supplying the cached M_h factorization.
    """
    Minv_r = apply_M_inv_batch(residuals, ops)
    quad = (residuals * Minv_r).sum(dim=-1)
    return quad.mean()
