"""Discrete residual evaluation for a candidate coefficient trajectory.

Stage 2 of the six-stage dependency chain -- the mathematical heart of the
certificate. Given a ``GalerkinTrajectory`` (from the reference solver, or
later a neural/corrected candidate reshaped into the same form), computes
the per-step discrete residual of the CGMY semidiscrete system using the
*same* operator (``cgmy.operators.OperatorCache``) the trajectory was
produced with (CLAUDE.md Hard Invariant #1), and the *same, stored* forcing
vectors the trajectory was produced with (Hard Invariant #2) -- never a
recomputed or substituted forcing.

Boundary (Dirichlet) degrees of freedom are excluded from the residual by
construction here: at those rows, the trusted solver replaces the PIDE
equation with an identity constraint (see
``cgmy_bspline.solver_cn.solve_cn``'s ``bc_dofs`` handling), so the raw
semidiscrete operator's row there is not the equation that was actually
solved, and evaluating it there measures the boundary-lifting mismatch, not
solver error. ``compute_cn_residual``/``compute_be_residual`` return the
full-length vector (zero at boundary dofs) so callers can inspect it, but
``residual_norm`` -- the quantity Gate 1 checks -- is always taken over the
interior (non-Dirichlet) dofs only.
"""

from __future__ import annotations

import numpy as np

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, GalerkinTrajectory

_TOL = 1e-9


def _interior_mask(n_dof: int, bc_dofs: tuple[list[int], list[int]]) -> np.ndarray:
    """Boolean mask, True on degrees of freedom NOT constrained by a Dirichlet BC."""
    mask = np.ones(n_dof, dtype=bool)
    left, right = bc_dofs
    mask[left] = False
    mask[right] = False
    return mask


def _check_cache_matches_mesh(traj: GalerkinTrajectory, ops: OperatorCache) -> None:
    mesh = traj.mesh_information
    if ops.N != mesh.N or ops.p != mesh.p:
        raise ValueError(
            f"OperatorCache mesh (N={ops.N}, p={ops.p}) does not match "
            f"trajectory mesh (N={mesh.N}, p={mesh.p}); build the cache from "
            f"traj.mesh_information."
        )
    if abs(ops.x_min - mesh.x_min) > 1e-12 or abs(ops.x_max - mesh.x_max) > 1e-12:
        raise ValueError(
            f"OperatorCache domain ({ops.x_min}, {ops.x_max}) does not match "
            f"trajectory domain ({mesh.x_min}, {mesh.x_max}); build the cache "
            f"from traj.mesh_information."
        )


def _forcing_at(traj: GalerkinTrajectory, n: int) -> np.ndarray:
    if traj.forcing_vectors is None:
        raise ValueError(
            f"traj.forcing_vectors is None (solve_cgmy was called with "
            f"store_forcing=False): cannot compute the residual for step {n} "
            f"without the exact forcing the trajectory was produced with. "
            f"Re-run solve_cgmy with store_forcing=True; do not recompute "
            f"the forcing here (CLAUDE.md Hard Invariant #2)."
        )
    if n < 0 or n >= len(traj.forcing_vectors):
        raise ValueError(
            f"step {n} out of range: traj.forcing_vectors has "
            f"{len(traj.forcing_vectors)} entries (steps 0..{len(traj.forcing_vectors) - 1})"
        )
    return traj.forcing_vectors[n]


def compute_cn_residual(
    traj: GalerkinTrajectory, mu: CGMYParameters, ops: OperatorCache, n: int
) -> np.ndarray:
    """Crank-Nicolson discrete residual at step n:

        r^{n+1/2} = M_h (c^{n+1} - c^n) / dt + A_h(mu) (c^{n+1} + c^n) / 2 - fbar^{n+1/2}

    where fbar^{n+1/2} = traj.forcing_vectors[n] (the CN-averaged forcing
    the trajectory was produced with -- see ``GalerkinTrajectory``'s
    docstring for exactly what is stored there). Zero at Dirichlet-boundary
    dofs (see module docstring).

    Raises:
        ValueError: If traj.forcing_vectors is None, if n is out of range,
            or if ``ops`` was not built from ``traj.mesh_information``.
    """
    _check_cache_matches_mesh(traj, ops)
    f_bar = _forcing_at(traj, n)

    c_n = traj.coefficients[n]
    c_np1 = traj.coefficients[n + 1]
    dt = traj.solver_metadata.dt

    c_bar = 0.5 * (c_np1 + c_n)
    r = ops.apply_M((c_np1 - c_n) / dt) + ops.apply_A(mu, c_bar) - f_bar

    mask = _interior_mask(ops.n_dof, traj.solver_metadata.bc_dofs)
    r = r.copy()
    r[~mask] = 0.0
    return r


def compute_be_residual(
    traj: GalerkinTrajectory, mu: CGMYParameters, ops: OperatorCache, n: int
) -> np.ndarray:
    """Backward-Euler discrete residual at step n (for Rannacher startup steps):

        r^{n+1} = M_h (c^{n+1} - c^n) / dt + A_h(mu) c^{n+1} - f^{n+1}

    where f^{n+1} = traj.forcing_vectors[n]. Note this is the *endpoint*
    forcing at tau_{n+1}, not the CN-averaged fbar^{n+1/2} that
    ``compute_cn_residual`` expects at the same index -- a BE step and a CN
    step at the same index mean different things by "the forcing vector for
    step n"; whatever produced the trajectory is responsible for storing
    the quantity appropriate to that step's own scheme. Zero at
    Dirichlet-boundary dofs (see module docstring).

    The trusted reference solver (``cgmy_bspline.solver_cn.solve_cn``) has
    no Rannacher/backward-Euler startup as of Stage A -- every step it
    produces is 'CN'. This function exists for the day that changes (and is
    tested against a synthetically-constructed BE step in
    ``tests/test_reference_residual.py``, since no real BE trajectory exists
    yet to test it against).

    Raises:
        ValueError: If traj.forcing_vectors is None, if n is out of range,
            or if ``ops`` was not built from ``traj.mesh_information``.
    """
    _check_cache_matches_mesh(traj, ops)
    f_np1 = _forcing_at(traj, n)

    c_n = traj.coefficients[n]
    c_np1 = traj.coefficients[n + 1]
    dt = traj.solver_metadata.dt

    r = ops.apply_M((c_np1 - c_n) / dt) + ops.apply_A(mu, c_np1) - f_np1

    mask = _interior_mask(ops.n_dof, traj.solver_metadata.bc_dofs)
    r = r.copy()
    r[~mask] = 0.0
    return r


def compute_all_residuals(
    traj: GalerkinTrajectory, mu: CGMYParameters, ops: OperatorCache
) -> np.ndarray:
    """Residual at every step, dispatching per-step on solver_metadata.scheme_per_step.

    Returns:
        Array of shape (n_steps, n_dof).

    Raises:
        ValueError: If a scheme other than 'CN' or 'BE' is recorded for some
            step (or any of the errors compute_cn_residual/compute_be_residual
            can raise).
    """
    n_steps = traj.solver_metadata.n_steps
    schemes = traj.solver_metadata.scheme_per_step
    out = np.empty((n_steps, ops.n_dof))
    for n in range(n_steps):
        scheme = schemes[n]
        if scheme == "CN":
            out[n] = compute_cn_residual(traj, mu, ops, n)
        elif scheme == "BE":
            out[n] = compute_be_residual(traj, mu, ops, n)
        else:
            raise ValueError(f"unknown scheme {scheme!r} recorded at step {n}")
    return out


def residual_norm(
    r: np.ndarray, ops: OperatorCache, bc_dofs: tuple[list[int], list[int]]
) -> float:
    """Euclidean norm of a residual vector, restricted to interior (non-Dirichlet) dofs.

    This is the quantity Gate 1 checks -- not the norm of the raw
    full-length vector, which is expected to be nonzero at Dirichlet dofs
    for reasons explained in the module docstring.
    """
    mask = _interior_mask(ops.n_dof, bc_dofs)
    return float(np.linalg.norm(r[mask]))
