"""Integrated free-coordinate predict -> project -> certify -> correct ->
recertify -> deliver workflow, method ``constrained_free_cn_v1``
(roadmap §2, milestones M1-M2).

Per query:

 1. problem:   ``certification.free_problem.build_free_problem`` -- one free
               operator, one index partition, one frozen forcing, the trusted
               time nodes, the discrete initial target z_h^0. Rejects
               nonuniform grids, non-CN steps, mesh/method mismatches and
               absent/incompatible forcing before anything is computed.
 2. predict:   any predictor returning raw (n_T+1, n_dof) coefficients (the
               neural models keep their raw interface).
 3. project:   z = P^T c_pred; the candidate is c = P z + b with THIS problem's
               lifting at every node. The pre-projection boundary discrepancy is
               kept as a diagnostic only. The projected trajectory is the one
               certified and, if accepted, returned.
 4. gate:      SPD gate and alpha <= lambda_min(S_I) on the free operator only.
 5. certify:   initial_error beta_0 = ||z^0 - z_h^0||_{M_I} (computed, never
               assumed 0), residual_certificate eta_res (PCG upper bracket on S_I
               at every step), augmented_certificate eta_aug =
               sqrt(beta_0^2 + eta_res^2); accept iff eta_aug <= tolerance and
               every ingredient is finite and valid. The certificate records the
               sha256 of the exact returned coefficient array.
 6. correct:   free CN march from z_h^0 with the same f_I, GMRES warm-started by
               the projected candidate; per-step iterations / convergence /
               recomputed linear residual / exceptions recorded.
 7. recertify: any changed trajectory is certified again from scratch.
 8. fallback:  if correction aborts or fails to certify, an optional dense
               direct free march (recertified); otherwise an explicit failure
               status. Nothing is delivered without a passing certificate.

Scope: the reliability statements are exact-arithmetic theorems (under
alpha <= lambda_min(S_I), the CN scheme actually run, the stored forcing and
lifting); every number here is ordinary double precision
(``common.methods.NUMERICAL_SCOPE``).
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable

import numpy as np

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, MeshInfo
from certification.constrained import FreeMarchResult, march_free_cn, march_free_cn_direct
from certification.free_problem import (
    FORCING_SOURCE_TRUSTED,
    INITIAL_TARGET_CONVENTION,
    TIME_NODE_CONVENTION,
    FreePathInputError,
    FreeProblem,
    beta_I,
    build_free_problem,
    free_eta_res_upper,
    free_kappa,
    free_residuals,
    free_stability_constants,
    raw_point_functional,
)
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1, NUMERICAL_SCOPE
from common.run_manifest import sha256_array

STATUS_ACCEPTED_DIRECT = "accepted_direct"
STATUS_ACCEPTED_CORRECTED = "accepted_after_correction"
STATUS_ACCEPTED_CORRECTED_NONCONVERGED = "accepted_certificate_qualified_nonconverged_correction"
STATUS_ACCEPTED_FALLBACK = "accepted_fallback_direct_free"
STATUS_UNCERTIFIABLE = "uncertifiable_free_gate"
STATUS_NOT_CERTIFIED = "corrected_not_certified"
STATUS_CORRECTION_FAILED = "correction_failed"
STATUS_NEEDS_CORRECTION_NOT_RUN = "needs_correction_not_run"
ACCEPTED_STATUSES = frozenset({
    STATUS_ACCEPTED_DIRECT, STATUS_ACCEPTED_CORRECTED,
    STATUS_ACCEPTED_CORRECTED_NONCONVERGED, STATUS_ACCEPTED_FALLBACK,
})

_PRECONDITIONERS = ("none", "mass", "jacobi", "banded")
# Only a strategy whose alpha is a lower bound on lambda_min(S_I) in exact
# arithmetic satisfies the upper-bracket theorem's hypothesis.
_COERCIVITY_STRATEGIES = ("B_weyl_residual_enclosure",)
_FALLBACKS = ("none", "direct_free_dense")


class InvalidPolicyError(FreePathInputError):
    pass


def _finite_positive(name: str, x: Any) -> float:
    if isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(float(x)) or float(x) <= 0:
        raise InvalidPolicyError(f"{name} must be a finite real > 0, got {x!r}")
    return float(x)


def _int_at_least(name: str, x: Any, lo: int) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or int(x) < lo:
        raise InvalidPolicyError(f"{name} must be an integer >= {lo}, got {x!r}")
    return int(x)


@dataclass(frozen=True)
class FreeWorkflowPolicy:
    tolerance: float
    quick_pcg_iterations: int
    max_pcg_iterations: int
    preconditioner: str = "none"
    coercivity_strategy: str = "B_weyl_residual_enclosure"
    gmres_rtol: float = 1e-10
    gmres_restart: int = 50
    gmres_maxiter: int | None = None
    correction_warm_start: bool = True
    fallback: str = "direct_free_dense"
    fallback_max_n_free: int = 400

    def validate(self) -> "FreeWorkflowPolicy":
        _finite_positive("tolerance", self.tolerance)
        _int_at_least("quick_pcg_iterations", self.quick_pcg_iterations, 0)
        _int_at_least("max_pcg_iterations", self.max_pcg_iterations, 0)
        if self.preconditioner not in _PRECONDITIONERS:
            raise InvalidPolicyError(f"preconditioner {self.preconditioner!r} not in {_PRECONDITIONERS}")
        if self.coercivity_strategy not in _COERCIVITY_STRATEGIES:
            raise InvalidPolicyError(
                f"coercivity_strategy {self.coercivity_strategy!r} not allowed; the upper bracket "
                f"requires alpha <= lambda_min(S_I): {_COERCIVITY_STRATEGIES}"
            )
        rtol = _finite_positive("gmres_rtol", self.gmres_rtol)
        if rtol >= 1.0:
            raise InvalidPolicyError(f"gmres_rtol must be < 1, got {rtol!r}")
        _int_at_least("gmres_restart", self.gmres_restart, 1)
        if self.gmres_maxiter is not None:
            _int_at_least("gmres_maxiter", self.gmres_maxiter, 1)
        if self.fallback not in _FALLBACKS:
            raise InvalidPolicyError(f"fallback {self.fallback!r} not in {_FALLBACKS}")
        _int_at_least("fallback_max_n_free", self.fallback_max_n_free, 1)
        return self

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides: Any) -> "FreeWorkflowPolicy":
        c, s = cfg["certificate"], cfg["solver"]
        kw = dict(
            tolerance=c["acceptance_tolerance"], quick_pcg_iterations=c["quick_pcg_iterations"],
            max_pcg_iterations=c["max_pcg_iterations"], preconditioner=c["preconditioner"],
            coercivity_strategy=c["coercivity_strategy"], gmres_rtol=s["gmres_rtol"],
            gmres_restart=s["gmres_restart"], gmres_maxiter=s["gmres_maxiter"],
            correction_warm_start=s["correction_warm_start"], fallback=s["fallback"],
            fallback_max_n_free=s["fallback_max_n_free"],
        )
        kw.update(overrides)
        return cls(**kw).validate()


# ---------------------------------------------------------------- certificate

@dataclass(frozen=True)
class CertificateRecord:
    """Augmented free-space certificate of ONE specific coefficient array.

    ``trajectory_sha256`` is the hash of the exact raw (n_T+1, n_dof) array
    this certificate refers to; ``verify_identity`` re-checks it.
    """

    method_id: str
    coordinate_space: str
    trajectory_sha256: str
    forcing_f_I_sha256: str
    initial_error: float  # beta_0 = ||z^0 - z_h^0||_{M_I}
    initial_error_sq: float
    residual_certificate: float  # eta_res (upper bracket)
    augmented_certificate: float  # sqrt(beta_0^2 + eta_res^2)
    per_step_upper: list[float]
    pcg_iterations_per_step: list[int]
    pcg_iteration_cap: int
    alpha_lower_estimate: float
    tolerance: float
    bound_status: str  # "valid" or "invalid: <reason>"
    accepted: bool

    def verify_identity(self, coefficients: np.ndarray) -> bool:
        return sha256_array(np.asarray(coefficients, dtype=float)) == self.trajectory_sha256


def bound_validity(initial_error_sq: float, per_step_upper: np.ndarray, eta_res: float) -> str:
    """Upper bounds of nonnegative quantities must be finite and >= 0; a
    violation is a numerical defect and is reported, never clamped."""
    if not math.isfinite(initial_error_sq):
        return "invalid: initial_error_sq non-finite"
    if initial_error_sq < 0:
        return "invalid: initial_error_sq negative"
    u = np.asarray(per_step_upper, dtype=float)
    if not np.all(np.isfinite(u)):
        return "invalid: residual upper bracket non-finite"
    if np.any(u < 0):
        return "invalid: residual upper bracket negative"
    if not math.isfinite(eta_res) or eta_res < 0:
        return "invalid: residual certificate non-finite or negative"
    return "valid"


def certify_free_trajectory(
    problem: FreeProblem,
    coefficients: np.ndarray,
    *,
    alpha: float,
    pcg_iterations: int,
    preconditioner: str,
    tolerance: float,
) -> CertificateRecord:
    """Certify a raw coefficient array that already carries THIS problem's
    lifting on the prescribed rows (raises otherwise -- project first)."""
    c = np.asarray(coefficients, dtype=float)
    bc = problem.fs.bc_idx
    if c.shape != (problem.n_steps + 1, problem.meta.n_dof):
        raise FreePathInputError(f"coefficients shape {c.shape} incompatible with problem")
    if not np.array_equal(c[:, bc], problem.forcing.lifting[:, bc]):
        raise FreePathInputError("prescribed rows differ from the problem lifting; project before certifying")
    z = c[:, problem.fs.free_idx]
    r = free_residuals(problem, z)
    cert = free_eta_res_upper(problem, r, alpha, max_pcg_iterations=pcg_iterations, preconditioner=preconditioner)
    d0 = z[0] - problem.z0_target
    b0_sq = float(d0 @ problem.free_ops.apply_M(d0))
    status = bound_validity(b0_sq, cert.per_step_quadratic_form, cert.eta)
    if status == "valid":
        beta0 = math.sqrt(b0_sq)
        eta_aug = math.sqrt(b0_sq + cert.eta**2)
    else:
        beta0 = eta_aug = float("nan")
    return CertificateRecord(
        method_id=problem.meta.method_id, coordinate_space=problem.meta.coordinate_space,
        trajectory_sha256=sha256_array(c), forcing_f_I_sha256=problem.forcing.f_I_sha256,
        initial_error=beta0, initial_error_sq=b0_sq, residual_certificate=float(cert.eta),
        augmented_certificate=eta_aug,
        per_step_upper=[float(u) for u in cert.per_step_quadratic_form],
        pcg_iterations_per_step=[int(k) for k in cert.n_pcg_iterations],
        pcg_iteration_cap=int(pcg_iterations), alpha_lower_estimate=float(alpha),
        tolerance=float(tolerance), bound_status=status,
        accepted=(status == "valid" and eta_aug <= tolerance),
    )


# ---------------------------------------------------------------- trace

@dataclass
class CorrectionRecord:
    solver: str
    warm_start: bool
    aborted: bool
    all_converged: bool
    iterations_per_step: list[int]
    converged_per_step: list[bool]
    relative_linear_residual_per_step: list[float]
    exceptions: list[str]

    @classmethod
    def from_march(cls, m: FreeMarchResult, warm_start: bool) -> "CorrectionRecord":
        return cls(
            solver=m.solver, warm_start=warm_start, aborted=m.aborted, all_converged=m.all_converged,
            iterations_per_step=[s.iterations for s in m.steps],
            converged_per_step=[s.converged for s in m.steps],
            relative_linear_residual_per_step=[s.relative_linear_residual for s in m.steps],
            exceptions=[f"step {s.step}: {s.exception}" for s in m.steps if s.exception],
        )


@dataclass
class FreeTrace:
    method_id: str
    coordinate_space: str
    numerical_scope: str
    query_id: str | None
    mu: dict[str, float]
    strike: float
    meta: dict[str, Any]
    time_node_convention: str
    initial_target_convention: str
    forcing: dict[str, str]
    policy: dict[str, Any]
    final_status: str = ""
    output_delivered: bool = False
    delivered_trajectory_sha256: str | None = None
    prediction_finite: bool | None = None
    boundary_discrepancy_pre_projection: float = float("nan")
    boundary_discrepancy_post_projection: float = float("nan")
    gate: dict[str, Any] = field(default_factory=dict)
    kappa_I: float = float("nan")
    candidate_certificate: CertificateRecord | None = None
    correction: CorrectionRecord | None = None
    correction_certificate: CertificateRecord | None = None
    fallback: CorrectionRecord | None = None
    fallback_certificate: CertificateRecord | None = None
    final_certificate: CertificateRecord | None = None
    point_evaluation: list[dict[str, float]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)
    delivered_coefficients: np.ndarray | None = field(default=None, repr=False)
    candidate_coefficients: np.ndarray | None = field(default=None, repr=False)

    def to_json_dict(self) -> dict[str, Any]:
        def conv(x: Any) -> Any:
            if dataclasses.is_dataclass(x):
                return {k: conv(v) for k, v in dataclasses.asdict(x).items()}
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [conv(v) for v in x]
            if isinstance(x, np.ndarray):
                return None  # arrays go to HDF5/npz, never into the JSON trace
            if isinstance(x, (np.floating, np.integer, np.bool_)):
                return x.item()
            return x

        return {k: conv(v) for k, v in dataclasses.asdict(self).items()
                if k not in ("delivered_coefficients", "candidate_coefficients")}


Predictor = Callable[[FreeProblem], np.ndarray]


def neural_predictor(model: Any, stats: Any, time_grid: np.ndarray) -> Predictor:
    """Wrap a ``learning.interface.CoefficientModel`` (raw interface unchanged:
    mu_tilde, psi = raw L2 payoff projection, raw c0, time grid, T)."""
    from data import payoff_family
    from learning.normalization import normalize_mu

    def predict(problem: FreeProblem) -> np.ndarray:
        mu_tilde = normalize_mu(problem.mu, stats)
        psi = payoff_family.european_put(problem.payoff.strike, problem.ops)
        c0 = np.array(problem.c0_target, copy=True)
        return np.asarray(model.predict_trajectory(mu_tilde, psi, c0, np.asarray(time_grid), problem.mu.T), dtype=float)

    return predict


def run_free_workflow(
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh_info: MeshInfo,
    ops: OperatorCache,
    time_grid: np.ndarray,
    predictor: Predictor,
    policy: FreeWorkflowPolicy,
    *,
    query_id: str | None = None,
    schemes: list[str] | None = None,
    n_steps: int | None = None,
    f_bar_raw: np.ndarray | None = None,
    forcing_source: str = FORCING_SOURCE_TRUSTED,
    eval_points: tuple[float, ...] = (),
    run_correction: bool = True,
    keep_coefficients: bool = True,
    free_ops: Any = None,
) -> FreeTrace:
    """Run steps 1-8 of the module docstring for one query. Input errors
    (policy, grid, scheme, mesh, forcing, prediction shape) raise
    ``FreePathInputError`` subclasses before any certificate is formed."""
    policy.validate()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    problem = build_free_problem(
        mu, payoff, mesh_info, ops, time_grid, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
        schemes=schemes, n_steps=n_steps, f_bar_raw=f_bar_raw, forcing_source=forcing_source,
        free_ops=free_ops,
    )
    timings["setup"] = time.perf_counter() - t0
    meta = dataclasses.asdict(problem.meta)
    meta["mu"] = dataclasses.asdict(problem.meta.mu)
    trace = FreeTrace(
        method_id=problem.meta.method_id, coordinate_space=problem.meta.coordinate_space,
        numerical_scope=NUMERICAL_SCOPE, query_id=query_id, mu=dataclasses.asdict(mu),
        strike=float(payoff.strike), meta=meta, time_node_convention=TIME_NODE_CONVENTION,
        initial_target_convention=INITIAL_TARGET_CONVENTION,
        forcing={"source": problem.forcing.source, "f_bar_raw_sha256": problem.forcing.f_bar_raw_sha256,
                 "lifting_sha256": problem.forcing.lifting_sha256, "f_I_sha256": problem.forcing.f_I_sha256},
        policy=dataclasses.asdict(policy), timings_s=timings,
    )

    # 2. predict
    t0 = time.perf_counter()
    raw = np.asarray(predictor(problem), dtype=float)
    timings["predict"] = time.perf_counter() - t0
    if raw.shape != (problem.n_steps + 1, problem.meta.n_dof):
        raise FreePathInputError(
            f"predictor returned shape {raw.shape}, expected {(problem.n_steps + 1, problem.meta.n_dof)}"
        )
    trace.prediction_finite = bool(np.all(np.isfinite(raw)))

    # 3. project onto this problem's lifting
    t0 = time.perf_counter()
    bc = problem.fs.bc_idx
    candidate = None
    if trace.prediction_finite:
        trace.boundary_discrepancy_pre_projection = float(np.max(np.abs(raw[:, bc] - problem.forcing.lifting[:, bc])))
        candidate = problem.lift(raw[:, problem.fs.free_idx])
        trace.boundary_discrepancy_post_projection = float(np.max(np.abs(candidate[:, bc] - problem.forcing.lifting[:, bc])))
        trace.events.append("prediction projected onto the problem lifting")
    else:
        trace.events.append("prediction contains non-finite values: not certified, correction runs cold")
    timings["project"] = time.perf_counter() - t0
    if keep_coefficients:
        trace.candidate_coefficients = candidate

    # 4. free SPD gate
    t0 = time.perf_counter()
    const = free_stability_constants(problem, coercivity_strategy=policy.coercivity_strategy)
    timings["gate"] = time.perf_counter() - t0
    trace.gate = {
        "space": const.coordinate_space, "n": problem.fs.n_free,
        "lambda_min_numerical_S_I": const.lambda_min_numerical,
        "alpha_lower_estimate": const.alpha_lower_estimate,
        "alpha_estimator": f"{const.coercivity_strategy} (exact-arithmetic Weyl bound evaluated in floating point)",
        "is_spd_by_estimate": const.is_spd,
        "lambda_min_M_I": const.lambda_min_M_h, "lambda_max_M_I": const.lambda_max_M_h,
    }
    if not const.is_spd:
        trace.final_status = STATUS_UNCERTIFIABLE
        trace.events.append("free SPD gate: no positive lower estimate for lambda_min(S_I); nothing delivered")
        return trace
    alpha = const.alpha_lower_estimate
    t0 = time.perf_counter()
    trace.kappa_I = free_kappa(problem)
    timings["kappa"] = time.perf_counter() - t0

    def certify(c: np.ndarray, cap: int) -> CertificateRecord:
        return certify_free_trajectory(problem, c, alpha=alpha, pcg_iterations=cap,
                                       preconditioner=policy.preconditioner, tolerance=policy.tolerance)

    def deliver(c: np.ndarray, rec: CertificateRecord, status: str) -> FreeTrace:
        if not rec.verify_identity(c):  # pragma: no cover - defensive, would be a bug
            raise AssertionError("certificate does not refer to the trajectory being delivered")
        trace.final_status = status
        trace.output_delivered = True
        trace.final_certificate = rec
        trace.delivered_trajectory_sha256 = rec.trajectory_sha256
        if keep_coefficients:
            trace.delivered_coefficients = np.array(c, copy=True)
        for x in eval_points:
            b = beta_I(problem, x)
            trace.point_evaluation.append({
                "x": float(x), "spot": float(np.exp(x)),
                "price_V_h": float(raw_point_functional(problem.mesh_info, x) @ c[-1]),
                "beta_I": b, "price_error_bound_vs_exact_discrete": b * rec.augmented_certificate,
            })
        return trace

    # 5. certify the projected candidate
    if candidate is not None:
        t0 = time.perf_counter()
        trace.candidate_certificate = certify(candidate, policy.quick_pcg_iterations)
        timings["certify_candidate"] = time.perf_counter() - t0
        if trace.candidate_certificate.accepted:
            trace.events.append("candidate accepted on eta_aug")
            return deliver(candidate, trace.candidate_certificate, STATUS_ACCEPTED_DIRECT)
        trace.events.append(f"candidate rejected ({trace.candidate_certificate.bound_status}, "
                            f"eta_aug={trace.candidate_certificate.augmented_certificate:.3e})")
    if not run_correction:
        trace.final_status = STATUS_NEEDS_CORRECTION_NOT_RUN
        return trace

    # 6-7. correction from z_h^0 with the same f_I, then recertification
    warm = candidate[:, problem.fs.free_idx] if (candidate is not None and policy.correction_warm_start) else None
    t0 = time.perf_counter()
    m = march_free_cn(problem.free_ops, mu, problem.dt, problem.z0_target, problem.forcing.f_I,
                      warm_start=warm, gmres_rtol=policy.gmres_rtol, gmres_restart=policy.gmres_restart,
                      gmres_maxiter=policy.gmres_maxiter)
    timings["correct"] = time.perf_counter() - t0
    trace.correction = CorrectionRecord.from_march(m, warm is not None)
    if m.aborted:
        trace.events.append(f"correction aborted: {trace.correction.exceptions}")
    else:
        if not m.all_converged:
            trace.events.append("correction GMRES did not converge at some steps (recorded, not relabelled)")
        corrected = problem.lift(m.z)
        t0 = time.perf_counter()
        trace.correction_certificate = certify(corrected, policy.max_pcg_iterations)
        timings["recertify"] = time.perf_counter() - t0
        if trace.correction_certificate.accepted:
            status = STATUS_ACCEPTED_CORRECTED if m.all_converged else STATUS_ACCEPTED_CORRECTED_NONCONVERGED
            return deliver(corrected, trace.correction_certificate, status)
        trace.events.append(f"corrected trajectory not accepted ({trace.correction_certificate.bound_status}, "
                            f"eta_aug={trace.correction_certificate.augmented_certificate:.3e})")

    # 8. fallback or explicit failure
    if policy.fallback == "direct_free_dense" and problem.fs.n_free <= policy.fallback_max_n_free:
        t0 = time.perf_counter()
        fb = march_free_cn_direct(problem.free_ops, mu, problem.dt, problem.z0_target, problem.forcing.f_I,
                                  max_n_free=policy.fallback_max_n_free)
        timings["fallback"] = time.perf_counter() - t0
        trace.fallback = CorrectionRecord.from_march(fb, False)
        trace.events.append("fallback: dense direct free march executed")
        if not fb.aborted:
            c_fb = problem.lift(fb.z)
            t0 = time.perf_counter()
            trace.fallback_certificate = certify(c_fb, policy.max_pcg_iterations)
            timings["recertify_fallback"] = time.perf_counter() - t0
            if trace.fallback_certificate.accepted:
                return deliver(c_fb, trace.fallback_certificate, STATUS_ACCEPTED_FALLBACK)
        trace.final_status = STATUS_NOT_CERTIFIED
        return trace
    trace.final_status = STATUS_CORRECTION_FAILED if m.aborted else STATUS_NOT_CERTIFIED
    return trace


def outcome_row(trace: FreeTrace) -> dict[str, Any]:
    """Roadmap §5.4-style row (adapted) for ResultGroupWriter."""
    fc = trace.final_certificate
    return {
        "query_id": trace.query_id, "method_id": trace.method_id, "coordinate_space": trace.coordinate_space,
        "final_status": trace.final_status, "output_delivered": trace.output_delivered,
        "free_gate_passed": bool(trace.gate.get("is_spd_by_estimate", False)),
        "candidate_initial_error": trace.candidate_certificate.initial_error if trace.candidate_certificate else "",
        "candidate_residual_certificate": trace.candidate_certificate.residual_certificate if trace.candidate_certificate else "",
        "candidate_augmented_certificate": trace.candidate_certificate.augmented_certificate if trace.candidate_certificate else "",
        "correction_executed": trace.correction is not None,
        "correction_all_converged": trace.correction.all_converged if trace.correction else "",
        "fallback_executed": trace.fallback is not None,
        "final_augmented_certificate": fc.augmented_certificate if fc else "",
        "tolerance": trace.policy["tolerance"],
        "returned_coefficients_sha256": trace.delivered_trajectory_sha256 or "",
        "total_time_s": sum(trace.timings_s.values()),
    }
