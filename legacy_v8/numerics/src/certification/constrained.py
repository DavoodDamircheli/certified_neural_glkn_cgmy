"""Constrained (free-coordinate) certificate machinery -- CGMY_V4_REMAINING_WORK.md
item 1 / OVERNIGHT_CLAUDE_CODE_PROMPT.md Phase 1.

The manuscript's boundary-space decision: the raw n_dof coefficient vector
decomposes as ``c = E z + b(t)``, where ``z in R^{n_free}`` are the genuine
variational unknowns (interior/free dofs) and ``b(t)`` carries the
time-dependent Dirichlet-prescribed data (zero on the free dofs, the exact
boundary value on the ``p-1``-per-end boundary dofs -- see
``cgmy.dirichlet_bc``, the single place that formula is written down).
``E`` is never materialized as a dense matrix here: "inject" (apply E),
"restrict" (apply E^T) and the free indices themselves (from
``cgmy.dirichlet_bc.bc_dofs_for_mesh``, never hardcoded) are the only pieces
needed, and every free-space operator action is exactly
``inject -> raw OperatorCache.apply_* -> restrict``.

Design (recorded in reports/v4_overnight_log.md Phase 1 section): rather
than writing a second, free-space implementation of the certificate math
(stability constants, eta_sharp/eta_guar, PCG, preconditioners, the mass
indicator -- all already implemented exactly once against an
``OperatorCache``-shaped object), this module provides ``FreeOperatorCache``,
a duck-typed wrapper exposing the identical interface
(``apply_M``, ``apply_M_inv``, ``apply_A``, ``apply_AT``, ``apply_S``,
``diag_S``, ``banded_preconditioner_lu``, ``n_dof``, ``N``, ``p``, ``x_min``,
``x_max``) over ``R^{n_free}``. Every certification-critical consumer in
this project runs UNCHANGED when handed a ``FreeOperatorCache`` in place of
a raw ``cgmy.operators.OperatorCache`` -- this is what "move positivity
checks, PCG, preconditioners, mass indicators... into one declared
coordinate space" means operationally. ``operators.py`` itself is never
modified and never gets a second ``apply_M``/``apply_A``/``apply_S``
(CLAUDE.md Hard Invariant #1 intact).

IMPORTANT boundary-error caveat (Acceptance Check 6): the free-space
residual/certificate, by construction, never reads a candidate
trajectory's own boundary-dof entries -- ``b(t)`` is always the EXACT
prescribed value (``cgmy.dirichlet_bc.dirichlet_bc_values``), never
whatever happens to sit in the candidate's array at those dofs. This means
a candidate with wrong boundary data can still show a small/zero free
residual: the free-space machinery is mathematically blind to boundary
correctness by construction, exactly as the raw residual is blind to it by
explicit row-zeroing (``certification.residuals``' module docstring). A
free-space acceptance decision therefore MUST separately gate on
``cgmy.dirichlet_bc.boundary_error`` (see ``is_free_accepted`` below) --
never rely on "the free rows look fine" as implicit license to skip that
check. Zeroing/never-reading boundary rows is not itself a boundary check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from cgmy import _external  # noqa: F401  (must precede any cgmy_bspline import)

from cgmy_bspline import matrices as cgmy_matrices
from cgmy_bspline.parameters import CGMYParams

from cgmy.dirichlet_bc import bc_dofs_for_mesh, dirichlet_bc_values
from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, GalerkinTrajectory, MeshInfo


@dataclass(frozen=True)
class FreeSpace:
    """The free/prescribed dof partition for one mesh -- mesh-only (depends
    on N, p alone via ``bc_dofs_for_mesh``), never on mu.

    Attributes:
        n_dof: Raw (full) dimension.
        bc_left: Left-boundary prescribed dof indices (``p-1`` of them).
        bc_right: Right-boundary prescribed dof indices (``p-1`` of them).
        free_idx: Sorted ascending array of the ``n_dof - len(bc_left) -
            len(bc_right)`` free (variational) dof indices.
    """

    n_dof: int
    bc_left: tuple[int, ...]
    bc_right: tuple[int, ...]
    free_idx: np.ndarray

    @property
    def n_free(self) -> int:
        return int(self.free_idx.shape[0])

    @property
    def bc_idx(self) -> np.ndarray:
        return np.asarray(sorted(list(self.bc_left) + list(self.bc_right)), dtype=int)


def free_space_for_mesh(mesh_info: MeshInfo) -> FreeSpace:
    """Build the free/prescribed partition from the existing boundary-DOF
    function -- 62/4 at the production mesh (N=64, p=3), never hardcoded."""
    bc_left, bc_right = bc_dofs_for_mesh(mesh_info)
    bc_set = set(bc_left) | set(bc_right)
    free_idx = np.array(
        [i for i in range(mesh_info.n_dof) if i not in bc_set], dtype=int
    )
    return FreeSpace(
        n_dof=mesh_info.n_dof,
        bc_left=tuple(bc_left),
        bc_right=tuple(bc_right),
        free_idx=free_idx,
    )


def inject(fs: FreeSpace, z: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """c = E z + b. ``b`` (full-length, zero on ``free_idx`` by construction
    -- see ``lifting_vector``) supplies the prescribed part; ``b=None``
    means a zero lifting, which is what every HOMOGENEOUS operator action
    (M_I, A_I, S_I below) uses -- those never see prescribed data, only the
    forcing term does."""
    c = np.zeros(fs.n_dof, dtype=float)
    c[fs.free_idx] = np.asarray(z, dtype=float)
    if b is not None:
        bc_idx = fs.bc_idx
        if bc_idx.size:
            c[bc_idx] = np.asarray(b, dtype=float)[bc_idx]
    return c


def restrict(fs: FreeSpace, full_vec: np.ndarray) -> np.ndarray:
    """E^T v -- the free-row restriction of a full-length vector."""
    return np.asarray(full_vec, dtype=float)[fs.free_idx]


def lifting_vector(
    mu: CGMYParameters, payoff: EuropeanPutPayoff, mesh_info: MeshInfo, fs: FreeSpace, tau: float
) -> np.ndarray:
    """b(tau): full-length, zero on free_idx, the EXACT prescribed Dirichlet
    value on each boundary dof -- reuses ``cgmy.dirichlet_bc.dirichlet_bc_values``
    (the single place that formula is written down), never re-derived here."""
    c_L, c_R = dirichlet_bc_values(mu, payoff, mesh_info, tau)
    b = np.zeros(fs.n_dof, dtype=float)
    for i in fs.bc_left:
        b[i] = c_L
    for i in fs.bc_right:
        b[i] = c_R
    return b


class FreeOperatorCache:
    """Presents the SAME interface as ``cgmy.operators.OperatorCache`` over
    ``R^{n_free}`` -- see module docstring. Every method is
    inject -> raw ``ops.apply_*`` -> restrict; no formula here is a second
    implementation of M_h/A_h/S_h.
    """

    def __init__(self, ops: OperatorCache, fs: FreeSpace) -> None:
        if ops.n_dof != fs.n_dof:
            raise ValueError(
                f"OperatorCache n_dof={ops.n_dof} does not match FreeSpace "
                f"n_dof={fs.n_dof}: build both from the same mesh_info."
            )
        self.ops = ops
        self.fs = fs
        # So existing code written against `ops.n_dof`/`ops.N`/etc. (e.g.
        # certification.constants.compute_stability_constants's max_n_dof
        # guard, certification.riesz_solver.solve_riesz_pcg's LinearOperator
        # shape) runs unchanged when handed `self` in place of `ops`.
        self.n_dof = fs.n_free
        self.N = ops.N
        self.p = ops.p
        self.x_min = ops.x_min
        self.x_max = ops.x_max
        self._M_I_lu: spla.SuperLU | None = None
        self._banded_precond_cache: dict[tuple[float, ...], spla.SuperLU] = {}

    def apply_M(self, v: np.ndarray) -> np.ndarray:
        return restrict(self.fs, self.ops.apply_M(inject(self.fs, v)))

    def apply_A(self, mu: CGMYParameters, v: np.ndarray) -> np.ndarray:
        return restrict(self.fs, self.ops.apply_A(mu, inject(self.fs, v)))

    def apply_AT(self, mu: CGMYParameters, v: np.ndarray) -> np.ndarray:
        return restrict(self.fs, self.ops.apply_AT(mu, inject(self.fs, v)))

    def apply_S(self, mu: CGMYParameters, v: np.ndarray) -> np.ndarray:
        """S_I(mu) = sym(A_I(mu)) = E^T S_raw(mu) E exactly (symmetrization
        and the E^T(.)E congruence commute: E^T(A+A^T)E = E^T A E + (E^T A E)^T)."""
        return 0.5 * (self.apply_A(mu, v) + self.apply_AT(mu, v))

    def _mass_lu(self) -> spla.SuperLU:
        if self._M_I_lu is None:
            idx = self.fs.free_idx
            M_I = self.ops.M_h[idx, :][:, idx].tocsc()
            self._M_I_lu = spla.splu(M_I)
        return self._M_I_lu

    def apply_M_inv(self, v: np.ndarray) -> np.ndarray:
        """Solve M_I x = v via a direct factorization of M_I = E^T M_h E --
        NOT the restriction of the full M_h^{-1} (E^T M_h^{-1} E != M_I^{-1}
        in general; the former would be the wrong operator for a free-space
        mass preconditioner)."""
        return self._mass_lu().solve(np.asarray(v, dtype=float))

    def diag_S(self, mu: CGMYParameters) -> np.ndarray:
        """diag(S_I(mu)) == diag(S_raw(mu))[free_idx] exactly: E selects
        whole rows/columns, so extracting the diagonal commutes with the
        E^T(.)E congruence. Reuses ops.diag_S verbatim -- no new formula."""
        return self.ops.diag_S(mu)[self.fs.free_idx]

    def banded_preconditioner_lu(self, mu: CGMYParameters, bandwidth: int = 64) -> spla.SuperLU:
        """Free-space banded preconditioner: rebuild the SAME trusted
        truncated-bandwidth approximation ``OperatorCache.banded_preconditioner_lu``
        uses (``cgmy_bspline.matrices.operator_matrix``, bandwidth-limited --
        not a second production-operator implementation, see that method's
        own docstring on why reuse for preconditioning purposes is
        sanctioned), then slice to free_idx and factor the submatrix
        directly -- not a restriction of the full banded LU, which cannot be
        sliced after factorization."""
        key = (mu.C, mu.G, mu.M, mu.Y, mu.r, float(bandwidth))
        lu = self._banded_precond_cache.get(key)
        if lu is None:
            bw = min(bandwidth, self.ops.n_dof)
            params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=1.0, T=1.0)
            mats = cgmy_matrices.operator_matrix(
                self.ops.N, self.ops.p, self.ops.h, self.ops.x_min, params, bandwidth=bw
            )
            A_bnd = mats["A_h"]
            S_bnd = 0.5 * (A_bnd + A_bnd.T)
            idx = self.fs.free_idx
            S_bnd_I = S_bnd[idx, :][:, idx].tocsc()
            lu = spla.splu(S_bnd_I)
            self._banded_precond_cache[key] = lu
        return lu


def free_cn_forcing(
    fs: FreeSpace,
    ops: OperatorCache,
    mu: CGMYParameters,
    dt: float,
    f_bar_raw: np.ndarray,
    b_old: np.ndarray,
    b_new: np.ndarray,
) -> np.ndarray:
    """f_I^{n+1/2} = E^T[f_raw - M_raw (b_new-b_old)/dt - A_raw (b_new+b_old)/2].

    NOT zero merely because f_bar_raw is zero (the trusted solver's
    ``boundary_forcing_vector`` is currently an unconditional zero -- see
    ``cgmy.reference_solver``'s module docstring -- but the b-dependent
    terms below are generally nonzero whenever the boundary data varies in
    time, i.e. whenever r != 0, since b_old != b_new)."""
    b_term = ops.apply_M((np.asarray(b_new) - np.asarray(b_old)) / dt) + ops.apply_A(
        mu, 0.5 * (np.asarray(b_new) + np.asarray(b_old))
    )
    return restrict(fs, np.asarray(f_bar_raw, dtype=float) - b_term)


def free_cn_residual(
    free_ops: FreeOperatorCache,
    mu: CGMYParameters,
    z_n: np.ndarray,
    z_np1: np.ndarray,
    dt: float,
    f_I_bar: np.ndarray,
) -> np.ndarray:
    """r_I^{n+1/2} = M_I (z^{n+1}-z^n)/dt + A_I zbar^{n+1/2} - f_I^{n+1/2},
    computed DIRECTLY in R^{n_free} (never forming or solving the full
    n_dof system) via ``free_ops`` (a ``FreeOperatorCache``)."""
    z_bar = 0.5 * (np.asarray(z_np1, dtype=float) + np.asarray(z_n, dtype=float))
    return (
        free_ops.apply_M((np.asarray(z_np1, dtype=float) - np.asarray(z_n, dtype=float)) / dt)
        + free_ops.apply_A(mu, z_bar)
        - np.asarray(f_I_bar, dtype=float)
    )


def compute_all_free_residuals(
    traj: GalerkinTrajectory,
    mu: CGMYParameters,
    ops: OperatorCache,
    fs: FreeSpace,
    free_ops: FreeOperatorCache,
) -> np.ndarray:
    """Free-space residual at every CN step, shape (n_steps, n_free).

    Computed DIRECTLY (explicit free CN forcing + M_I/A_I actions) -- NOT by
    restricting ``certification.residuals.compute_all_residuals``'s output.
    That the two agree is Acceptance Check 2, verified independently in
    ``tests/test_constrained_free.py::test_check2_...`` -- not assumed here.

    ``z_n = restrict(fs, traj.coefficients[n])`` reads only the free rows of
    the candidate; ``b_old``/``b_new`` are always the EXACT prescribed
    values (never read from the candidate's own boundary rows) -- see
    module docstring's boundary-error caveat. Callers that need a boundary
    check on this candidate must call ``cgmy.dirichlet_bc.boundary_error``
    separately (``is_free_accepted`` below does this for the top-level
    accept decision).

    Raises:
        ValueError: If ``traj.forcing_vectors`` is None (mirrors
            ``certification.residuals``' same requirement -- Hard
            Invariant #2, never substitute a recomputed/zero forcing).
    """
    if traj.forcing_vectors is None:
        raise ValueError(
            "traj.forcing_vectors is None: cannot compute the free residual "
            "without the exact forcing the trajectory was produced with "
            "(CLAUDE.md Hard Invariant #2). Re-run solve_cgmy with "
            "store_forcing=True, or build the skeleton trajectory with its "
            "forcing populated (see pipeline.predict_certify_correct.build_skeleton_trajectory)."
        )
    n_steps = traj.solver_metadata.n_steps
    dt = traj.solver_metadata.dt
    out = np.empty((n_steps, fs.n_free))
    for n in range(n_steps):
        tau_n = float(traj.time_grid[n])
        tau_np1 = float(traj.time_grid[n + 1])
        b_old = lifting_vector(mu, traj.payoff, traj.mesh_information, fs, tau_n)
        b_new = lifting_vector(mu, traj.payoff, traj.mesh_information, fs, tau_np1)
        f_bar_raw = traj.forcing_vectors[n]
        f_I_bar = free_cn_forcing(fs, ops, mu, dt, f_bar_raw, b_old, b_new)
        z_n = restrict(fs, traj.coefficients[n])
        z_np1 = restrict(fs, traj.coefficients[n + 1])
        out[n] = free_cn_residual(free_ops, mu, z_n, z_np1, dt, f_I_bar)
    return out


def free_cn_step(
    free_ops: FreeOperatorCache,
    mu: CGMYParameters,
    dt: float,
    z_n: np.ndarray,
    f_I_bar: np.ndarray,
    *,
    x0: np.ndarray | None = None,
    gmres_rtol: float = 1e-10,
    gmres_restart: int = 50,
    gmres_maxiter: int | None = None,
) -> tuple[np.ndarray, int, bool]:
    """One free-space CN step: solve (M_I/dt + A_I/2) z^{n+1} = (M_I/dt - A_I/2)
    z^n + f_I_bar for z^{n+1}, via matrix-free GMRES on ``free_ops``
    (same CN discretization and GMRES settings -- rtol=1e-10, restart=50 --
    the trusted ``vendor/cgmy_bspline/solver_cn.py`` uses for its own
    mode='toeplitz' step; see Acceptance Check 1). Unlike the raw stepper
    (``cgmy_bspline.solver_cn``/``correction.krylov``), there is no
    boundary-row override here -- the free system has no boundary rows at
    all, by construction.

    Returns:
        (z_np1, n_gmres_iterations, converged).
    """
    n = free_ops.n_dof

    def F_matvec(v: np.ndarray) -> np.ndarray:
        return free_ops.apply_M(v) / dt + 0.5 * free_ops.apply_A(mu, v)

    F_op = spla.LinearOperator((n, n), matvec=F_matvec, dtype=float)
    rhs = free_ops.apply_M(z_n) / dt - 0.5 * free_ops.apply_A(mu, z_n) + np.asarray(f_I_bar, dtype=float)

    iters = [0]

    def _cb(xk: np.ndarray, _ctr=iters) -> None:
        _ctr[0] += 1

    z_np1, info = spla.gmres(
        F_op, rhs, x0=x0, rtol=gmres_rtol, restart=gmres_restart, maxiter=gmres_maxiter,
        callback=_cb, callback_type="pr_norm",
    )
    return z_np1, iters[0], info == 0


def solve_free_cn_trajectory(
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh_info: MeshInfo,
    time_grid: np.ndarray,
    ops: OperatorCache,
    fs: FreeSpace,
    free_ops: FreeOperatorCache,
    z0: np.ndarray,
    *,
    gmres_rtol: float = 1e-10,
    gmres_restart: int = 50,
) -> np.ndarray:
    """Time-march the free-space CN system from z0, with the trusted
    solver's own (currently all-zero, see cgmy.reference_solver's module
    docstring) ``boundary_forcing_vector`` and the exact lifting ``b(t)`` --
    never a recomputed-from-scratch forcing (Hard Invariant #2). Used by
    Acceptance Check 1 to verify this reproduces ``restrict(fs, raw
    trajectory)`` at every time level, to GMRES-tolerance-level agreement
    (two independent GMRES solves, in different-dimensional Krylov spaces,
    at the same rtol, are not expected to be bit-identical -- see
    tests/test_constrained_free.py for the stated allowance).

    Returns:
        z trajectory, shape (len(time_grid), n_free).
    """
    from cgmy_bspline import boundary as cgmy_boundary
    from cgmy_bspline.parameters import CGMYParams

    params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=payoff.strike, T=mu.T)
    n_steps = len(time_grid) - 1
    dt = float(time_grid[1] - time_grid[0])

    zs = [np.asarray(z0, dtype=float).copy()]
    z = zs[0]
    tau = float(time_grid[0])
    for n in range(n_steps):
        tau_new = float(time_grid[n + 1])
        f_old_raw = cgmy_boundary.boundary_forcing_vector(
            mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, tau, None
        )
        f_new_raw = cgmy_boundary.boundary_forcing_vector(
            mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, tau_new, None
        )
        f_bar_raw = 0.5 * (f_old_raw + f_new_raw)
        b_old = lifting_vector(mu, payoff, mesh_info, fs, tau)
        b_new = lifting_vector(mu, payoff, mesh_info, fs, tau_new)
        f_I_bar = free_cn_forcing(fs, ops, mu, dt, f_bar_raw, b_old, b_new)

        z_next, n_it, converged = free_cn_step(
            free_ops, mu, dt, z, f_I_bar, x0=z,
            gmres_rtol=gmres_rtol, gmres_restart=gmres_restart,
        )
        if not converged:
            # v5 (roadmap §2.4): the flag used to be discarded here.
            raise FreeSolveNonconvergence(
                f"free CN GMRES did not converge at step {n + 1} after {n_it} iterations "
                f"(rtol={gmres_rtol}); use march_free_cn for a per-step status record"
            )
        zs.append(z_next.copy())
        z = z_next
        tau = tau_new

    return np.array(zs)


def solve_free_cn_trajectory_direct(
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh_info: MeshInfo,
    time_grid: np.ndarray,
    ops: OperatorCache,
    fs: FreeSpace,
    free_ops: FreeOperatorCache,
    z0: np.ndarray,
) -> np.ndarray:
    """Like ``solve_free_cn_trajectory``, but via a DENSE direct solve (LU
    factorization of the dense (n_free, n_free) CN system matrix, refactored
    once since the free system's A_I(mu) does not change across steps) --
    not GMRES. Practical only at small n_free (production: 62).

    Used as the "exact" ground truth for the POD free-space ROM's full-rank
    recovery test (CGMY_V4_REMAINING_WORK.md item 3): GMRES-based stepping
    (``solve_free_cn_trajectory``, matching the trusted solver's own
    rtol=1e-10) introduces its own ~1e-10-level per-step noise, which is
    the wrong yardstick for validating that the ROM's EQUATIONS (not its
    linear-solve tolerance) are correct at full rank -- this direct solve
    removes that confound.
    """
    n = fs.n_free
    A_I = np.array([free_ops.apply_A(mu, e) for e in np.eye(n)]).T
    M_I = np.array([free_ops.apply_M(e) for e in np.eye(n)]).T
    dt = float(time_grid[1] - time_grid[0])
    F_lu = sla.lu_factor(M_I / dt + 0.5 * A_I)
    B = M_I / dt - 0.5 * A_I

    from cgmy_bspline import boundary as cgmy_boundary
    from cgmy_bspline.parameters import CGMYParams

    params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=payoff.strike, T=mu.T)

    z = np.asarray(z0, dtype=float).copy()
    zs = [z.copy()]
    tau = float(time_grid[0])
    for n_step in range(len(time_grid) - 1):
        tau_new = float(time_grid[n_step + 1])
        f_old_raw = cgmy_boundary.boundary_forcing_vector(
            mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, tau, None
        )
        f_new_raw = cgmy_boundary.boundary_forcing_vector(
            mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, tau_new, None
        )
        f_bar_raw = 0.5 * (f_old_raw + f_new_raw)
        b_old = lifting_vector(mu, payoff, mesh_info, fs, tau)
        b_new = lifting_vector(mu, payoff, mesh_info, fs, tau_new)
        f_I_bar = free_cn_forcing(fs, ops, mu, dt, f_bar_raw, b_old, b_new)

        rhs = B @ z + f_I_bar
        z = sla.lu_solve(F_lu, rhs)
        zs.append(z.copy())
        tau = tau_new

    return np.array(zs)


def is_free_accepted(
    certificate_eta: float,
    tolerance: float,
    boundary_error: float,
    *,
    boundary_tol: float = 1e-9,
) -> bool:
    """The free-space accept decision: eta <= tolerance AND boundary_error
    <= boundary_tol. Both are required -- the free certificate alone cannot
    see a boundary discrepancy (module docstring's Acceptance Check 6
    caveat: it never reads a candidate's own boundary rows), so an accept
    decision that only checked eta would let an incorrect boundary slip
    through uncaught. ``boundary_error`` should come from
    ``cgmy.dirichlet_bc.boundary_error`` evaluated on the SAME candidate the
    certificate was computed on, before any ``project_bc`` call that would
    hide the discrepancy.
    """
    return certificate_eta <= tolerance and boundary_error <= boundary_tol


# ---------------------------------------------------------------------------
# v5 (roadmap §2.4): free CN time marching with an honest per-step status
# record. The forcing f_I is passed in (the problem's frozen FreeCNForcing.f_I),
# never regenerated here.
# ---------------------------------------------------------------------------


class FreeSolveNonconvergence(RuntimeError):
    """A free CN linear solve did not reach its tolerance."""


@dataclass(frozen=True)
class FreeStepRecord:
    """One CN step of a free march.

    Attributes:
        step: 1-based index of the time level produced (z^{step}).
        iterations: GMRES inner iterations (0 for the dense direct solve).
        converged: GMRES info == 0 (for the direct solve: finite solution).
        gmres_info: scipy GMRES ``info`` (None for the direct solve).
        relative_linear_residual: ||F z - rhs|| / ||rhs||, recomputed explicitly
            after the solve (F = M_I/dt + A_I/2), not taken from the solver.
        exception: repr of an exception raised by the step, else None.
    """

    step: int
    iterations: int
    converged: bool
    gmres_info: int | None
    relative_linear_residual: float
    exception: str | None


@dataclass(frozen=True)
class FreeMarchResult:
    z: np.ndarray | None  # (n_steps+1, n_free); None if the march aborted
    steps: list[FreeStepRecord]
    all_converged: bool
    aborted: bool
    solver: str  # "gmres" | "direct_dense"


def _cn_matvecs(free_ops: FreeOperatorCache, mu: CGMYParameters, dt: float):
    def F(v: np.ndarray) -> np.ndarray:
        return free_ops.apply_M(v) / dt + 0.5 * free_ops.apply_A(mu, v)

    def B(v: np.ndarray) -> np.ndarray:
        return free_ops.apply_M(v) / dt - 0.5 * free_ops.apply_A(mu, v)

    return F, B


def _rel_residual(F, z: np.ndarray, rhs: np.ndarray) -> float:
    rn = float(np.linalg.norm(rhs))
    res = float(np.linalg.norm(F(z) - rhs))
    return res / rn if rn > 0 else res


def march_free_cn(
    free_ops: FreeOperatorCache,
    mu: CGMYParameters,
    dt: float,
    z0: np.ndarray,
    f_I: np.ndarray,
    *,
    warm_start: np.ndarray | None = None,
    gmres_rtol: float = 1e-10,
    gmres_restart: int = 50,
    gmres_maxiter: int | None = None,
) -> FreeMarchResult:
    """GMRES free CN march from z0. ``warm_start`` (shape (n_steps+1, n_free)),
    if given, seeds step n's GMRES with warm_start[n+1] (the projected
    candidate); otherwise the previous level seeds it. A nonconverged step is
    recorded and the march continues (the iterate is still a candidate that a
    separately computed certificate may qualify); an exception aborts the march
    and is recorded."""
    f_I = np.asarray(f_I, dtype=float)
    n_steps = f_I.shape[0]
    if warm_start is not None and np.shape(warm_start) != (n_steps + 1, free_ops.n_dof):
        raise ValueError(f"warm_start shape {np.shape(warm_start)} != {(n_steps + 1, free_ops.n_dof)}")
    F, B = _cn_matvecs(free_ops, mu, dt)
    z = np.asarray(z0, dtype=float).copy()
    zs = [z.copy()]
    records: list[FreeStepRecord] = []
    for n in range(n_steps):
        x0 = np.asarray(warm_start[n + 1], dtype=float) if warm_start is not None else z
        try:
            z_next, n_it, ok = free_cn_step(
                free_ops, mu, dt, z, f_I[n], x0=x0,
                gmres_rtol=gmres_rtol, gmres_restart=gmres_restart, gmres_maxiter=gmres_maxiter,
            )
            rhs = B(z) + f_I[n]
            rel = _rel_residual(F, z_next, rhs)
            info = 0 if ok else 1
        except Exception as exc:  # recorded, never swallowed silently
            records.append(FreeStepRecord(n + 1, 0, False, None, float("nan"), repr(exc)))
            return FreeMarchResult(z=None, steps=records, all_converged=False, aborted=True, solver="gmres")
        records.append(FreeStepRecord(n + 1, n_it, bool(ok), info, rel, None))
        zs.append(z_next.copy())
        z = z_next
    return FreeMarchResult(
        z=np.array(zs), steps=records, all_converged=all(r.converged for r in records),
        aborted=False, solver="gmres",
    )


def march_free_cn_direct(
    free_ops: FreeOperatorCache,
    mu: CGMYParameters,
    dt: float,
    z0: np.ndarray,
    f_I: np.ndarray,
    *,
    max_n_free: int = 400,
) -> FreeMarchResult:
    """Dense-LU free CN march (fallback / ground truth), same F, B and f_I as
    ``march_free_cn``. Practical only for small n_free."""
    n = free_ops.n_dof
    if n > max_n_free:
        raise ValueError(f"n_free={n} exceeds max_n_free={max_n_free} for a dense direct march")
    f_I = np.asarray(f_I, dtype=float)
    F, B = _cn_matvecs(free_ops, mu, dt)
    records: list[FreeStepRecord] = []
    try:
        A_I = np.array([free_ops.apply_A(mu, e) for e in np.eye(n)]).T
        M_I = np.array([free_ops.apply_M(e) for e in np.eye(n)]).T
        F_lu = sla.lu_factor(M_I / dt + 0.5 * A_I)
    except Exception as exc:
        records.append(FreeStepRecord(1, 0, False, None, float("nan"), repr(exc)))
        return FreeMarchResult(z=None, steps=records, all_converged=False, aborted=True, solver="direct_dense")
    z = np.asarray(z0, dtype=float).copy()
    zs = [z.copy()]
    for k in range(f_I.shape[0]):
        rhs = B(z) + f_I[k]
        z_next = sla.lu_solve(F_lu, rhs)
        ok = bool(np.all(np.isfinite(z_next)))
        records.append(FreeStepRecord(k + 1, 0, ok, None, _rel_residual(F, z_next, rhs), None))
        zs.append(z_next.copy())
        z = z_next
    return FreeMarchResult(
        z=np.array(zs), steps=records, all_converged=all(r.converged for r in records),
        aborted=False, solver="direct_dense",
    )
