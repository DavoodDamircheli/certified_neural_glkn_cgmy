"""One constrained free-coordinate CN problem per query (roadmap §2.1, §2.4, §2.5).

``FreeProblem`` bundles, for one (mesh, mu, payoff, time grid), everything the
integrated free path (``pipeline.free_workflow``) needs, built ONCE and shared by
prediction, residual, Riesz solves, stability constants, preconditioners,
correction and recertification:

- the index partition ``fs`` and the free operator ``free_ops``
  (``certification.constrained.FreeOperatorCache``: inject -> raw
  ``OperatorCache`` action -> restrict; no second operator implementation);
- the time nodes, using the trusted solver's own convention
  ``tau_n = n * (T / n_steps)`` (``vendor/cgmy_bspline/solver_cn.py``), never a
  separately generated grid;
- the forcing ``FreeCNForcing``: the raw CN-averaged forcing actually used
  (stored, or produced by the identical trusted call sequence), the lifting
  ``b^n`` at every node, and

      f_I^{n+1/2} = P^T ( f_raw^{n+1/2} - M (b^{n+1}-b^n)/dt - A (b^{n+1}+b^n)/2 ),

  all frozen (read-only) and hashed, so certification and correction provably
  consume the same bytes;
- the discrete initial target ``z_h^0 = P^T c_h^0`` (see ``build_free_problem``);
- ``FreeMeta``: dimensions, mesh identity, mu, strike, time-node hash and the
  coordinate-space tag, checked at every module boundary.

Notation: P = ``constrained.inject`` (zero lifting), P^T = ``constrained.restrict``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from cgmy import _external  # noqa: F401  (must precede any cgmy_bspline import)

from cgmy_bspline import boundary as cgmy_boundary
from cgmy_bspline.bspline_basis import active_indices, phi as bspline_phi
from cgmy_bspline.parameters import CGMYParams

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, MeshInfo
from certification.constants import StabilityConstants, compute_stability_constants, kappa_operator_norm
from certification.constrained import (
    FreeOperatorCache,
    FreeSpace,
    free_cn_forcing,
    free_cn_residual,
    free_space_for_mesh,
    lifting_vector,
    restrict,
)
from certification.guaranteed_certificate import Certificate, eta_guar, eta_sharp
from certification.riesz_solver import Preconditioner
from common.methods import KNOWN_METHODS, validate_method_id
from common.run_manifest import sha256_array

COORDINATE_SPACE_FREE = "constrained_free"
TIME_NODE_CONVENTION = "tau_n = n * (T / n_steps)  (trusted cgmy_bspline.solver_cn convention)"
INITIAL_TARGET_CONVENTION = (
    "z_h^0 = P^T c_h^0, c_h^0 = correction.krylov.compute_initial_coefficients: raw L2 payoff "
    "projection M_h^{-1} b^0 with Dirichlet rows overwritten (the trusted solver's c^0 convention)"
)
FORCING_SOURCE_TRUSTED = "trusted_boundary_forcing_vector"
FORCING_SOURCE_STORED = "stored"

# Rounding allowance for comparing a supplied time grid with the trusted nodes
# n*(T/n_steps): both np.linspace and n*dt carry at most ~eps*|tau| <= eps*T
# rounding per node, so two such grids differ by <= 2*eps*T per node and their
# steps by <= 4*eps*T. A genuinely nonuniform grid differs by O(dt) >> eps*T.
_GRID_ROUNDING_FACTOR = 4.0


class FreePathInputError(ValueError):
    """An input to the free path is invalid or incompatible; nothing was computed from it."""


class IncompatibleMetadataError(FreePathInputError):
    pass


class UnsupportedSchemeError(FreePathInputError):
    pass


class ForcingError(FreePathInputError):
    pass


# ---------------------------------------------------------------- metadata

@dataclass(frozen=True)
class FreeMeta:
    method_id: str
    coordinate_space: str
    N: int
    p: int
    n_dof: int
    x_min: float
    x_max: float
    n_free: int
    free_idx_sha256: str
    mu: CGMYParameters
    strike: float
    n_steps: int
    dt: float
    time_nodes_sha256: str

    def require_same(self, other: "FreeMeta", where: str) -> None:
        diff = [f.name for f in fields(self) if getattr(self, f.name) != getattr(other, f.name)]
        if diff:
            raise IncompatibleMetadataError(
                f"{where}: metadata mismatch in {diff}: "
                + ", ".join(f"{k}={getattr(self, k)!r} vs {getattr(other, k)!r}" for k in diff)
            )


def check_ops_matches_mesh(ops: OperatorCache, mesh_info: MeshInfo) -> None:
    """Exact identity of the operator's mesh with the problem mesh (both are
    built from the same configs/base.yaml floats, so no tolerance is needed)."""
    got = (ops.N, ops.p, ops.n_dof, float(ops.x_min), float(ops.x_max))
    want = (mesh_info.N, mesh_info.p, mesh_info.n_dof, float(mesh_info.x_min), float(mesh_info.x_max))
    if got != want:
        raise IncompatibleMetadataError(f"OperatorCache mesh {got} != problem mesh {want}")


# ---------------------------------------------------------------- time grid

def validate_time_grid(
    time_grid: np.ndarray,
    T: float,
    *,
    schemes: list[str] | None = None,
    n_steps: int | None = None,
) -> tuple[int, float, np.ndarray]:
    """Check a time grid against what this version supports: finite, starting
    at 0, strictly increasing, uniform, ending at T, every step CN.

    Returns (n_steps, dt, tau_nodes) with dt = T/n_steps and tau_nodes[n] =
    n*dt -- the trusted solver's own nodes, used for every lifting/forcing
    evaluation downstream.

    Raises:
        FreePathInputError: malformed grid, length mismatch with ``n_steps``.
        UnsupportedSchemeError: nonuniform grid, or any step not 'CN'
            (BE/Rannacher steps need their own residual/accumulation formulas,
            which the free path does not implement).
    """
    if not (np.isfinite(T) and T > 0):
        raise FreePathInputError(f"maturity T must be finite and > 0, got {T!r}")
    tg = np.asarray(time_grid, dtype=float)
    if tg.ndim != 1 or tg.size < 2:
        raise FreePathInputError(f"time grid must be 1-D with >= 2 nodes, got shape {tg.shape}")
    if not np.all(np.isfinite(tg)):
        raise FreePathInputError("time grid contains non-finite values")
    steps = np.diff(tg)
    if np.any(steps <= 0):
        raise FreePathInputError("time grid is not strictly increasing")
    m = tg.size - 1
    if n_steps is not None and n_steps != m:
        raise FreePathInputError(f"time grid has {m} steps but metadata says n_steps={n_steps}")
    if schemes is not None:
        if len(schemes) != m:
            raise FreePathInputError(f"{len(schemes)} scheme entries for {m} steps")
        bad = sorted({s for s in schemes if s != "CN"})
        if bad:
            raise UnsupportedSchemeError(
                f"schemes {bad} are not supported by constrained_free_cn_v1 (pure uniform CN only)"
            )
    if tg[0] != 0.0:
        raise FreePathInputError(f"time grid must start at tau=0, got {tg[0]!r}")
    dt = T / m
    tau_nodes = np.arange(m + 1, dtype=float) * dt
    allowance = _GRID_ROUNDING_FACTOR * np.finfo(float).eps * T
    if abs(tg[-1] - T) > allowance:
        raise FreePathInputError(f"time grid ends at {tg[-1]!r}, expected T={T!r}")
    dev = float(np.max(np.abs(tg - tau_nodes)))
    if dev > allowance:
        raise UnsupportedSchemeError(
            f"nonuniform time grid (max node deviation {dev:.3e} from n*T/n_steps > rounding "
            f"allowance {allowance:.3e}); constrained_free_cn_v1 supports uniform CN only"
        )
    tau_nodes.setflags(write=False)
    return m, dt, tau_nodes


# ---------------------------------------------------------------- forcing

def trusted_raw_forcing(
    mu: CGMYParameters, payoff: EuropeanPutPayoff, mesh_info: MeshInfo, tau_nodes: np.ndarray
) -> np.ndarray:
    """The CN-averaged raw forcing produced by the SAME call sequence as
    ``cgmy_bspline.solver_cn.solve_cn`` (``forcing_history``): f_bar[n] =
    0.5*(f(tau_n) + f(tau_{n+1})) with the trusted ``boundary_forcing_vector``."""
    params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=payoff.strike, T=mu.T)
    n_steps = len(tau_nodes) - 1
    out = np.empty((n_steps, mesh_info.n_dof))
    for n in range(n_steps):
        f_old = cgmy_boundary.boundary_forcing_vector(
            mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, float(tau_nodes[n]), None
        )
        f_new = cgmy_boundary.boundary_forcing_vector(
            mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, float(tau_nodes[n + 1]), None
        )
        out[n] = 0.5 * (f_old + f_new)
    return out


@dataclass(frozen=True)
class FreeCNForcing:
    meta: FreeMeta
    source: str
    f_bar_raw: np.ndarray  # (n_steps, n_dof)
    lifting: np.ndarray  # (n_steps+1, n_dof), zero on free rows
    f_I: np.ndarray  # (n_steps, n_free)
    f_bar_raw_sha256: str
    lifting_sha256: str
    f_I_sha256: str


def build_free_cn_forcing(
    meta: FreeMeta,
    fs: FreeSpace,
    ops: OperatorCache,
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh_info: MeshInfo,
    tau_nodes: np.ndarray,
    f_bar_raw: np.ndarray | None,
    source: str,
) -> FreeCNForcing:
    if f_bar_raw is None:
        raise ForcingError("forcing is absent; refusing to substitute zero or regenerated forcing")
    f_bar_raw = np.array(f_bar_raw, dtype=float, copy=True)
    want = (meta.n_steps, meta.n_dof)
    if f_bar_raw.shape != want:
        raise ForcingError(f"raw forcing has shape {f_bar_raw.shape}, expected {want}")
    if not np.all(np.isfinite(f_bar_raw)):
        raise ForcingError("raw forcing contains non-finite values")
    lifting = np.array([lifting_vector(mu, payoff, mesh_info, fs, float(t)) for t in tau_nodes])
    f_I = np.array([
        free_cn_forcing(fs, ops, mu, meta.dt, f_bar_raw[n], lifting[n], lifting[n + 1])
        for n in range(meta.n_steps)
    ])
    for a in (f_bar_raw, lifting, f_I):
        a.setflags(write=False)
    return FreeCNForcing(
        meta=meta, source=source, f_bar_raw=f_bar_raw, lifting=lifting, f_I=f_I,
        f_bar_raw_sha256=sha256_array(f_bar_raw), lifting_sha256=sha256_array(lifting),
        f_I_sha256=sha256_array(f_I),
    )


# ---------------------------------------------------------------- the problem

@dataclass(frozen=True)
class FreeProblem:
    meta: FreeMeta
    mu: CGMYParameters
    payoff: EuropeanPutPayoff
    mesh_info: MeshInfo
    ops: OperatorCache
    fs: FreeSpace
    free_ops: FreeOperatorCache
    n_steps: int
    dt: float
    tau_nodes: np.ndarray
    forcing: FreeCNForcing
    c0_target: np.ndarray  # raw, (n_dof,)
    z0_target: np.ndarray  # free, (n_free,)

    def lift(self, z: np.ndarray) -> np.ndarray:
        """c^n = P z^n + b^n at every node, with THIS problem's lifting."""
        z = self.check_free_trajectory(z, "lift")
        c = np.array(self.forcing.lifting, copy=True)
        c[:, self.fs.free_idx] = z
        return c

    def check_free_trajectory(self, z: np.ndarray, where: str) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        want = (self.n_steps + 1, self.fs.n_free)
        if z.shape != want:
            raise IncompatibleMetadataError(f"{where}: free trajectory shape {z.shape} != {want}")
        return z

    def mass_norm(self, v: np.ndarray) -> float:
        """||v||_{M_I}; raises if the computed square is negative (M_I is SPD,
        so a negative value means a defect, never something to clamp)."""
        q = float(np.asarray(v, dtype=float) @ self.free_ops.apply_M(v))
        if not np.isfinite(q) or q < 0.0:
            raise FloatingPointError(f"mass quadratic form is {q!r}")
        return float(np.sqrt(q))


def build_free_problem(
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh_info: MeshInfo,
    ops: OperatorCache,
    time_grid: np.ndarray,
    *,
    method_id: str,
    schemes: list[str] | None = None,
    n_steps: int | None = None,
    f_bar_raw: np.ndarray | None = None,
    forcing_source: str = FORCING_SOURCE_TRUSTED,
    free_ops: FreeOperatorCache | None = None,
) -> FreeProblem:
    """Build the per-query free problem.

    Forcing: with ``forcing_source='trusted_boundary_forcing_vector'`` and no
    ``f_bar_raw``, the raw forcing is produced by ``trusted_raw_forcing`` (the
    trusted solver's own call sequence; byte-identity with a stored
    ``solve_cgmy`` forcing is tested). With ``forcing_source='stored'`` the
    caller must pass the stored ``f_bar_raw``; absence raises ``ForcingError``.
    ``free_ops`` (optional) must wrap ``ops`` and this mesh's partition; it lets a
    caller reuse a prebuilt mesh-only M_I factorization.
    """
    from correction.krylov import compute_initial_coefficients

    validate_method_id(method_id)
    if KNOWN_METHODS[method_id]["coordinate_space"] != COORDINATE_SPACE_FREE:
        raise IncompatibleMetadataError(f"method {method_id!r} is not a free-coordinate method")
    check_ops_matches_mesh(ops, mesh_info)
    m, dt, tau_nodes = validate_time_grid(time_grid, mu.T, schemes=schemes, n_steps=n_steps)
    fs = free_space_for_mesh(mesh_info)
    if free_ops is None:
        free_ops = FreeOperatorCache(ops, fs)
    elif free_ops.ops is not ops or not np.array_equal(free_ops.fs.free_idx, fs.free_idx):
        # reusable-setup benchmarks may pass a FreeOperatorCache carrying a
        # prebuilt (mesh-only) M_I factorization; it must wrap THIS ops/partition.
        raise IncompatibleMetadataError("supplied free_ops does not wrap this ops / free partition")
    meta = FreeMeta(
        method_id=method_id, coordinate_space=COORDINATE_SPACE_FREE,
        N=mesh_info.N, p=mesh_info.p, n_dof=mesh_info.n_dof,
        x_min=float(mesh_info.x_min), x_max=float(mesh_info.x_max),
        n_free=fs.n_free, free_idx_sha256=sha256_array(fs.free_idx.astype(np.int64)),
        mu=mu, strike=float(payoff.strike), n_steps=m, dt=dt,
        time_nodes_sha256=sha256_array(tau_nodes),
    )
    if forcing_source == FORCING_SOURCE_TRUSTED:
        if f_bar_raw is not None:
            raise ForcingError("pass forcing_source='stored' when supplying f_bar_raw")
        f_bar_raw = trusted_raw_forcing(mu, payoff, mesh_info, tau_nodes)
    elif forcing_source != FORCING_SOURCE_STORED:
        raise ForcingError(f"unknown forcing_source {forcing_source!r}")
    forcing = build_free_cn_forcing(meta, fs, ops, mu, payoff, mesh_info, tau_nodes, f_bar_raw, forcing_source)

    c0, _, _ = compute_initial_coefficients(mu, payoff, mesh_info, ops)
    c0.setflags(write=False)
    z0 = restrict(fs, c0)
    z0.setflags(write=False)
    return FreeProblem(
        meta=meta, mu=mu, payoff=payoff, mesh_info=mesh_info, ops=ops, fs=fs, free_ops=free_ops,
        n_steps=m, dt=dt, tau_nodes=tau_nodes, forcing=forcing, c0_target=c0, z0_target=z0,
    )


# ---------------------------------------------------------------- free-space numerics

def free_residuals(problem: FreeProblem, z: np.ndarray, forcing: FreeCNForcing | None = None) -> np.ndarray:
    """r_I^{n+1/2} = M_I (z^{n+1}-z^n)/dt + A_I zbar - f_I^{n+1/2} for every step,
    shape (n_steps, n_free). ``forcing`` defaults to the problem's own; if
    given, its metadata must match the problem's exactly."""
    forcing = problem.forcing if forcing is None else forcing
    if forcing is None:
        raise ForcingError("forcing is absent")
    forcing.meta.require_same(problem.meta, "free_residuals(forcing)")
    z = problem.check_free_trajectory(z, "free_residuals")
    if not np.all(np.isfinite(z)):
        raise FreePathInputError("free trajectory contains non-finite values")
    return np.array([
        free_cn_residual(problem.free_ops, problem.mu, z[n], z[n + 1], problem.dt, forcing.f_I[n])
        for n in range(problem.n_steps)
    ])


def free_stability_constants(
    problem: FreeProblem, *, coercivity_strategy: str, max_n_dof: int = 400
) -> StabilityConstants:
    """The SPD gate and alpha <= lambda_min(S_I) for the free operator only.
    ``max_n_dof`` is the dense-eigendecomposition size guard (raise it
    deliberately for the resolution study; the cost is recorded there)."""
    return compute_stability_constants(
        problem.mu, problem.free_ops, max_n_dof, coercivity_strategy=coercivity_strategy,
        coordinate_space=COORDINATE_SPACE_FREE,
    )


def free_eta_res_upper(
    problem: FreeProblem,
    residuals: np.ndarray,
    alpha: float,
    *,
    max_pcg_iterations: int,
    preconditioner: Preconditioner,
) -> Certificate:
    """sqrt(dt * sum_n u_n): PCG Riesz upper bracket on S_I at every step."""
    return eta_guar(
        residuals, problem.mu, problem.free_ops, problem.dt, alpha,
        max_pcg_iterations=max_pcg_iterations, preconditioner=preconditioner,
        coordinate_space=COORDINATE_SPACE_FREE,
    )


def free_eta_sharp(problem: FreeProblem, residuals: np.ndarray) -> Certificate:
    return eta_sharp(residuals, problem.mu, problem.free_ops, problem.dt, coordinate_space=COORDINATE_SPACE_FREE)


def free_kappa(problem: FreeProblem) -> float:
    """kappa_I = ||S_I^{-1/2} A_I S_I^{-1/2}||_2 (manuscript eq. kappa-def, free operator)."""
    return kappa_operator_norm(problem.mu, problem.free_ops)


def raw_point_functional(mesh_info: MeshInfo, x: float) -> np.ndarray:
    """phi(x) with phi(x)^T c = V_h(x), using the trusted basis evaluation in the
    same dof ordering as ``cgmy_bspline.projection.eval_solution``."""
    ks = active_indices(mesh_info.N, mesh_info.p)
    xs = np.array([float(x)])
    return np.array([float(bspline_phi(xs, k, mesh_info.p, mesh_info.h, mesh_info.x_min)[0]) for k in ks])


def free_point_functional(problem: FreeProblem, x: float) -> np.ndarray:
    """phi_I(x) = P^T phi(x)."""
    return restrict(problem.fs, raw_point_functional(problem.mesh_info, x))


def beta_I(problem: FreeProblem, x: float) -> float:
    """beta_I(x) = sqrt(phi_I^T M_I^{-1} phi_I), with M_I^{-1} applied by the
    factorization of the principal free mass matrix. For candidates sharing the
    target's boundary data, |e_h(x)| <= beta_I(x) * eta_aug. The displayed price is
    V_h(x) = phi(x)^T c (no extra scaling), so this is directly a price bound."""
    ph = free_point_functional(problem, x)
    q = float(ph @ problem.free_ops.apply_M_inv(ph))
    if not np.isfinite(q) or q < 0.0:
        raise FloatingPointError(f"beta_I quadratic form is {q!r}")
    return float(np.sqrt(q))


def residual_rounding_eta(problem: FreeProblem, z: np.ndarray, lambda_min_S: float) -> float:
    """Allowance for rounding in the COMPUTED free residual, expressed in the
    eta scale: sqrt(dt * sum_n n_free * d_n^2 / lambda_min(S_I)), where
    d_n = GAMMA*(||M_I||_inf ||dz||_inf/dt + ||A_I||_inf ||zbar||_inf + ||f_I||_inf)
    bounds ||fl(r_n) - r_n||_inf and GAMMA = 4*n_dof*eps covers the sparse and
    FFT-Toeplitz matvecs (see tests/test_v5_free_path.py module docstring).
    A floating-point allowance, not an enclosure."""
    from certification.constants import dense_from_apply

    n = problem.fs.n_free
    M = dense_from_apply(problem.free_ops.apply_M, n)
    A = dense_from_apply(lambda v: problem.free_ops.apply_A(problem.mu, v), n)
    Minf, Ainf = float(np.max(np.sum(np.abs(M), 1))), float(np.max(np.sum(np.abs(A), 1)))
    g = 4 * problem.meta.n_dof * np.finfo(float).eps
    z = problem.check_free_trajectory(z, "residual_rounding_eta")
    tot = 0.0
    for k in range(problem.n_steps):
        d = g * (Minf * np.max(np.abs(z[k + 1] - z[k])) / problem.dt
                 + Ainf * np.max(np.abs(0.5 * (z[k + 1] + z[k])))
                 + np.max(np.abs(problem.forcing.f_I[k])))
        tot += n * d * d
    return float(np.sqrt(problem.dt * tot / lambda_min_S))
