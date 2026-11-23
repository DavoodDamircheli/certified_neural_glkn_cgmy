"""Krylov-subspace correction that EXPLOITS a rejected neural prediction,
never discards it: neural coefficients seed (warm-start) the GMRES solve
at every Crank-Nicolson step, rather than the correction re-solving from a
cold (zero) start.

Reuses the exact same operator and stepping structure the trusted solver
(cgmy_bspline.solver_cn.solve_cn, mode='toeplitz') uses -- OperatorCache.
apply_A/apply_M for the operator action (verified bit-identical to
solve_cn's own _A_matvec in Stage A/G), the same boundary-row-override
trick for Dirichlet dofs, the same GMRES tolerance/restart (rtol=1e-10,
restart=50), and the same (currently always-zero) forcing via
cgmy_bspline.boundary.boundary_forcing_vector. This module does not
reimplement A_h, M_h, or the CN update formula -- only the stepping loop
and the warm-start choice are new bookkeeping, not new numerics.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from cgmy import _external  # noqa: F401  (must precede any cgmy_bspline import)

from cgmy_bspline import boundary as cgmy_boundary
from cgmy_bspline.parameters import CGMYParams
from cgmy_bspline.projection import assemble_put_rhs

from cgmy.dirichlet_bc import bc_dofs_for_mesh, dirichlet_bc_values, project_bc
from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, MeshInfo


def compute_initial_coefficients(
    mu: CGMYParameters, payoff: EuropeanPutPayoff, mesh_info: MeshInfo, ops: OperatorCache
) -> tuple[np.ndarray, list[int], list[int]]:
    """Reproduce solve_cn's c^0 (L2 payoff projection + Dirichlet BC
    overwrite) WITHOUT running any time-stepping -- a single cheap
    mass-solve (via the cached M_h factorization), not a full reference
    solve. Wraps cgmy_bspline.projection.assemble_put_rhs (trusted,
    unchanged).

    Every experiment through Stage I obtained c0 via a full solve_cgmy
    call (since GalerkinTrajectory bundles it with the whole trajectory).
    That is fine for certifying an already-solved reference, but Stage J's
    "predict" step must NOT secretly require a full solve just to get c0,
    or the runtime comparison in exp08 would be dishonest.

    Returns:
        (c0, bc_left, bc_right) -- bc_left/bc_right are mesh-only
        (depend on N, p alone), matching
        traj.solver_metadata.bc_dofs exactly.
    """
    params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=payoff.strike, T=mu.T)
    grid_info = {"N": mesh_info.N, "h": mesh_info.h, "x_min": mesh_info.x_min}
    b0 = assemble_put_rhs(params, grid_info, mesh_info.p)
    c0 = ops.apply_M_inv(b0).copy()

    bc_left, bc_right = bc_dofs_for_mesh(mesh_info)
    c_L0, c_R0 = dirichlet_bc_values(mu, payoff, mesh_info, 0.0)
    for i in bc_left:
        c0[i] = c_L0
    for i in bc_right:
        c0[i] = c_R0
    return c0, bc_left, bc_right


@dataclass(frozen=True)
class CorrectionResult:
    """Outcome of a full-trajectory Krylov correction pass.

    Attributes:
        coefficients: (n_steps+1, n_dof) corrected trajectory.
        n_iterations_per_step: GMRES iteration count at each of the
            n_steps linear solves.
        converged_per_step: Whether GMRES hit rtol at each step (info==0).
        total_time_s: Wall-clock time for the whole correction pass.
        warm_start: Whether neural coefficients were used to seed GMRES
            (True) or GMRES started cold/zero (False) -- see
            experiments/exp09_warm_start.py for the direct comparison.
    """

    coefficients: np.ndarray
    n_iterations_per_step: list[int]
    converged_per_step: list[bool]
    total_time_s: float
    warm_start: bool


def correct_trajectory(
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh_info: MeshInfo,
    time_grid: np.ndarray,
    ops: OperatorCache,
    seed_coefficients: np.ndarray | None,
    *,
    warm_start: bool = True,
    gmres_rtol: float = 1e-10,
    gmres_restart: int = 50,
) -> CorrectionResult:
    """Re-solve every Crank-Nicolson step's linear system via GMRES,
    seeded by seed_coefficients (typically a neural prediction) rather
    than a cold/zero start, using the SAME operator and stepping structure
    solve_cn uses (see module docstring).

    Args:
        seed_coefficients: (len(time_grid), n_dof) warm-start source (e.g.
            a neural model's predict_trajectory output). Required if
            warm_start=True; ignored if warm_start=False.
        warm_start: If True, GMRES at step n is seeded with
            seed_coefficients[n+1]. If False, GMRES starts cold (scipy's
            default: zeros) -- used by exp09 for the direct comparison.
    """
    if warm_start and seed_coefficients is None:
        raise ValueError("seed_coefficients is required when warm_start=True")

    n_dof = mesh_info.n_dof
    dt = float(time_grid[1] - time_grid[0])
    c0, bc_left, bc_right = compute_initial_coefficients(mu, payoff, mesh_info, ops)
    bc_dofs = bc_left + bc_right

    def A_matvec(v: np.ndarray) -> np.ndarray:
        return ops.apply_A(mu, v)

    def B_matvec(v: np.ndarray) -> np.ndarray:
        return ops.apply_M(v) - (dt / 2.0) * A_matvec(v)

    def F_matvec(v: np.ndarray) -> np.ndarray:
        w = ops.apply_M(v) + (dt / 2.0) * A_matvec(v)
        w = w.copy()
        for i in bc_dofs:
            w[i] = v[i]
        return w

    F_op = spla.LinearOperator((n_dof, n_dof), matvec=F_matvec, dtype=float)

    grid_info = {"N": mesh_info.N, "h": mesh_info.h, "x_min": mesh_info.x_min}
    params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=payoff.strike, T=mu.T)

    coeffs = [c0.copy()]
    n_iters: list[int] = []
    converged: list[bool] = []
    c = c0.copy()
    tau = 0.0

    t0 = time.time()
    for n in range(len(time_grid) - 1):
        tau_new = float(time_grid[n + 1])
        f_old = cgmy_boundary.boundary_forcing_vector(mesh_info.N, mesh_info.p, mesh_info.h,
                                                        mesh_info.x_min, params, tau, None)
        f_new = cgmy_boundary.boundary_forcing_vector(mesh_info.N, mesh_info.p, mesh_info.h,
                                                        mesh_info.x_min, params, tau_new, None)
        rhs = B_matvec(c) + (dt / 2.0) * (f_new + f_old)
        c_L, c_R = dirichlet_bc_values(mu, payoff, mesh_info, tau_new)
        for i in bc_left:
            rhs[i] = c_L
        for i in bc_right:
            rhs[i] = c_R

        x0 = seed_coefficients[n + 1].copy() if warm_start else None
        step_iters = [0]

        def _cb(xk: np.ndarray, _ctr=step_iters) -> None:
            _ctr[0] += 1

        c_next, info = spla.gmres(
            F_op, rhs, x0=x0, rtol=gmres_rtol, restart=gmres_restart,
            callback=_cb, callback_type="pr_norm",
        )
        if info != 0:
            warnings.warn(f"correction GMRES did not converge at step {n + 1}: info={info}")
        n_iters.append(step_iters[0])
        converged.append(info == 0)
        coeffs.append(c_next.copy())
        c = c_next
        tau = tau_new

    corrected = np.array(coeffs)
    # F_matvec's identity rows at bc_dofs already force GMRES toward the
    # exact Dirichlet value at every step, but only to gmres_rtol -- project
    # explicitly so the returned trajectory satisfies the constraint at
    # machine precision, not just solver tolerance (Task 1, working prompt).
    corrected = project_bc(corrected, mu, payoff, mesh_info, time_grid)

    return CorrectionResult(
        coefficients=corrected,
        n_iterations_per_step=n_iters,
        converged_per_step=converged,
        total_time_s=time.time() - t0,
        warm_start=warm_start,
    )
