"""Exact Dirichlet boundary-condition values and projection for the CGMY
European-put discretization.

The trusted reference solver (``cgmy_bspline.solver_cn.solve_cn``) overwrites
two blocks of degrees of freedom at every time step -- the same ``p-1`` dofs
at each end of the domain, dof-indexed by ``bc_dofs_for_mesh`` -- with a
deep-ITM value at the left boundary and zero at the right. ``correction.
krylov.compute_initial_coefficients``/``correct_trajectory`` reproduce this
identically (trusted, unchanged formula) at tau=0 and at every corrected
step; this module is the single place that formula is written down, so noone
else re-derives it independently (CLAUDE.md's "one operator implementation"
spirit, applied to boundary data rather than A_h/M_h/S_h).

A coefficient trajectory NOT produced by ``solve_cn`` itself -- a neural
prediction, or a "skeleton" trajectory built from one -- has no reason to
already satisfy this constraint at tau>0 (the exact IC only fixes tau=0).
Nothing in the residual machinery can catch a violation either:
``certification.residuals`` deliberately zeroes the residual rows at these
same dofs, because at Dirichlet rows the PIDE equation is replaced by an
identity constraint, so the raw operator's row there is not the equation
that was actually solved. ``project_bc`` is the one function that enforces
the constraint directly on a predicted trajectory; ``boundary_error`` is the
matching diagnostic (``E_boundary`` in the working prompt).
"""

from __future__ import annotations

import numpy as np

from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, MeshInfo


def bc_dofs_for_mesh(mesh_info: MeshInfo) -> tuple[list[int], list[int]]:
    """(bc_left, bc_right) Dirichlet dof index lists -- mesh-only (depend on
    N, p alone), identical to solve_cn's own bc_left/bc_right and to
    traj.solver_metadata.bc_dofs for any trajectory on this mesh.
    """
    p = mesh_info.p
    n_dof = mesh_info.n_dof
    return list(range(p - 1)), list(range(n_dof - p + 1, n_dof))


def dirichlet_bc_values(
    mu: CGMYParameters, payoff: EuropeanPutPayoff, mesh_info: MeshInfo, tau: float
) -> tuple[float, float]:
    """Exact Dirichlet boundary values (c_L, c_R) at time tau for a European
    put, matching solve_cn's own overwrite step at every time level
    (tau=0 included: exp(-r*0) == 1 recovers compute_initial_coefficients'
    c_L0). c_R is always 0 (deep-OTM / far-field decay of a put)."""
    c_L = payoff.strike * np.exp(-mu.r * tau) * np.sqrt(mesh_info.h)
    c_R = 0.0
    return c_L, c_R


def boundary_error(
    coefficients: np.ndarray,
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh_info: MeshInfo,
    time_grid: np.ndarray,
) -> float:
    """E_boundary = max over n, j in I_BC of |coefficients[n,j] - c_BC[n,j]|.

    Pure diagnostic, no mutation. Call before ``project_bc`` to see how much
    a raw prediction violated the Dirichlet constraint; call again after to
    confirm the constraint now holds at machine precision (the quantity the
    working prompt's Task 1 requires be <= 1e-12 in every experiment output).
    """
    bc_left, bc_right = bc_dofs_for_mesh(mesh_info)
    worst = 0.0
    for n, tau in enumerate(time_grid):
        c_L, c_R = dirichlet_bc_values(mu, payoff, mesh_info, float(tau))
        row = coefficients[n]
        if bc_left:
            worst = max(worst, float(np.max(np.abs(row[bc_left] - c_L))))
        if bc_right:
            worst = max(worst, float(np.max(np.abs(row[bc_right] - c_R))))
    return worst


def project_bc(
    coefficients: np.ndarray,
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh_info: MeshInfo,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Overwrite the Dirichlet-dof entries of every row of ``coefficients``
    (shape (len(time_grid), n_dof)) with the exact BC value at that row's
    time -- the single sanctioned way to make a candidate trajectory (a
    neural prediction, or a GMRES-corrected trajectory) satisfy the
    Dirichlet constraint at machine precision, structurally rather than by
    any loss term or tolerance. Returns a new array; ``coefficients`` is not
    modified in place.
    """
    if coefficients.shape[0] != len(time_grid):
        raise ValueError(
            f"coefficients has {coefficients.shape[0]} time rows but "
            f"time_grid has {len(time_grid)} entries"
        )
    bc_left, bc_right = bc_dofs_for_mesh(mesh_info)
    out = coefficients.copy()
    for n, tau in enumerate(time_grid):
        c_L, c_R = dirichlet_bc_values(mu, payoff, mesh_info, float(tau))
        for i in bc_left:
            out[n, i] = c_L
        for i in bc_right:
            out[n, i] = c_R
    return out
