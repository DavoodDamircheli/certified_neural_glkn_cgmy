"""v5 M2: initial error, boundary projection, scheme/grid validation, solver
status and input validation in the integrated free workflow (roadmap §2.2-2.4
and the remaining rows of the §2 "Required small tests" table).

Tolerances are tied to solver residuals or the rounding model stated in
tests/test_v5_free_path.py; acceptance tolerances passed to the workflow are
the quantity under test, not test allowances.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import certification.constrained as constrained_mod
from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.constants import compute_stability_constants
from certification.constrained import march_free_cn_direct, solve_free_cn_trajectory, FreeSolveNonconvergence
from certification.free_problem import (
    FreePathInputError,
    UnsupportedSchemeError,
    build_free_problem,
    validate_time_grid,
)
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1
from common.run_manifest import sha256_array
from pipeline import free_workflow as fw
from pipeline.legacy_guard import LegacyCallError, forbid_legacy_calls, summarize
from pipeline.predict_certify_correct import build_mesh_info

EPS = np.finfo(float).eps
X_MIN, X_MAX = -3.394829814011908, 12.605170185988092
PAYOFF = EuropeanPutPayoff(strike=100.0)
N_STEPS = 20
MU_SPD = CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.5)
MU_FULL_INDEFINITE = CGMYParameters(
    C=0.4874301809351891, G=1.4997461959719658, M=8.837858855724335,
    Y=1.4543537443503738, r=0.06294871459715069, T=0.10040483502671123,
)


@pytest.fixture(scope="module")
def mesh():
    return Mesh(N=64, p=3, x_min=X_MIN, x_max=X_MAX)


@pytest.fixture(scope="module")
def mesh_info(mesh):
    return build_mesh_info(mesh)


@pytest.fixture(scope="module")
def ops(mesh):
    return OperatorCache(mesh, X_MIN, X_MAX)


def _tg(mu, n=N_STEPS):
    return np.linspace(0.0, mu.T, n + 1)


@pytest.fixture(scope="module")
def prob(mesh_info, ops):
    return build_free_problem(MU_SPD, PAYOFF, mesh_info, ops, _tg(MU_SPD), method_id=METHOD_CONSTRAINED_FREE_CN_V1)


@pytest.fixture(scope="module")
def exact(prob):
    """Dense-LU solution of the constrained problem from z_h^0 (the reference
    'exact discrete' trajectory up to its own small residual)."""
    m = march_free_cn_direct(prob.free_ops, prob.mu, prob.dt, prob.z0_target, prob.forcing.f_I)
    return m.z


def policy(**kw):
    base = dict(tolerance=1e-3, quick_pcg_iterations=10, max_pcg_iterations=200)
    base.update(kw)
    return fw.FreeWorkflowPolicy(**base).validate()


def run(mesh_info, ops, raw, pol, mu=MU_SPD, **kw):
    return fw.run_free_workflow(mu, PAYOFF, mesh_info, ops, _tg(mu), lambda p: raw, pol, **kw)


# ---------------------------------------------------------------- §2.2 initial error

def test_nonzero_initial_error_with_small_residual_is_not_accepted(prob, exact, mesh_info, ops):
    """A candidate following the homogeneous error equation from a perturbed
    initial value has a residual at rounding level, yet its true error is
    ~||delta||_{M_I}. Acceptance must see beta_0: eta_res alone would accept."""
    rng = np.random.default_rng(10)
    delta = rng.standard_normal(prob.fs.n_free)
    delta *= 1e-4 / prob.mass_norm(delta)
    pert = march_free_cn_direct(prob.free_ops, prob.mu, prob.dt, prob.z0_target + delta, prob.forcing.f_I).z
    raw = prob.lift(pert)
    tol = 1e-6
    tr = run(mesh_info, ops, raw, policy(tolerance=tol))
    cc = tr.candidate_certificate
    assert cc.residual_certificate < tol  # residual-only acceptance would wrongly accept
    # The candidate's z^0 is fl(z_h^0 + delta): forming it and subtracting z_h^0
    # again perturbs each entry by <= eps*(|z_h^0| + |delta|) (two roundings),
    # i.e. by up to eps*||z_h^0|| / ||delta|| relative -- cancellation, not a
    # certificate error. Allowance: 2*eps*|| |z_h^0| + |delta| ||_{M_I}.
    cancel = 2 * EPS * prob.mass_norm(np.abs(prob.z0_target) + np.abs(delta))
    assert abs(cc.initial_error - 1e-4) <= cancel + 4 * prob.fs.n_free * EPS * 1e-4
    assert cc.augmented_certificate >= cc.initial_error and not cc.accepted
    # theorem check: true max-in-time error <= eta_aug, and > eta_res
    true_err = max(prob.mass_norm(pert[n] - exact[n]) for n in range(N_STEPS + 1))
    assert true_err <= cc.augmented_certificate
    assert true_err > cc.residual_certificate
    # workflow then corrects from z_h^0: initial error of the delivered output is computed as 0
    assert tr.final_status == fw.STATUS_ACCEPTED_CORRECTED
    assert tr.final_certificate.initial_error == 0.0


def test_initial_error_is_computed_not_assumed(prob, exact, mesh_info, ops):
    tr = run(mesh_info, ops, prob.lift(exact), policy())
    assert tr.final_status == fw.STATUS_ACCEPTED_DIRECT
    assert tr.final_certificate.initial_error_sq == 0.0  # exact IC row => d0 == 0 bitwise
    assert tr.final_certificate.augmented_certificate == pytest.approx(
        tr.final_certificate.residual_certificate, rel=4 * EPS)


# ---------------------------------------------------------------- §2.3 boundary projection

@pytest.mark.parametrize("tol", [1e-3, 1e-10])
def test_corrupted_boundary_rows_projected_and_identity_matches(prob, exact, mesh_info, ops, tol):
    """Garbage on prescribed rows: the returned candidate carries the exact
    lifting, its hash is the certificate's hash, and budgets below 1e-9 work
    (no fixed boundary threshold is involved any more)."""
    raw = prob.lift(exact)
    raw[:, prob.fs.bc_idx] += 0.37 * np.arange(1, len(prob.fs.bc_idx) + 1)
    tr = run(mesh_info, ops, raw, policy(tolerance=tol))
    assert tr.boundary_discrepancy_pre_projection > 0.3
    assert tr.boundary_discrepancy_post_projection == 0.0
    assert tr.output_delivered and tr.final_status == fw.STATUS_ACCEPTED_DIRECT
    out = tr.delivered_coefficients
    assert np.array_equal(out[:, prob.fs.bc_idx], prob.forcing.lifting[:, prob.fs.bc_idx])
    assert tr.final_certificate.verify_identity(out)
    assert not tr.final_certificate.verify_identity(raw)
    assert tr.delivered_trajectory_sha256 == sha256_array(out)
    tampered = out.copy()
    tampered[5, 10] = np.nextafter(tampered[5, 10], np.inf)
    assert not tr.final_certificate.verify_identity(tampered)


def test_certifying_unprojected_coefficients_is_refused(prob, exact):
    raw = prob.lift(exact)
    raw[3, prob.fs.bc_idx[0]] += 1e-14
    with pytest.raises(FreePathInputError):
        fw.certify_free_trajectory(prob, raw, alpha=0.05, pcg_iterations=5, preconditioner="none", tolerance=1e-3)


def test_recertification_after_correction_uses_the_corrected_hash(prob, exact, mesh_info, ops):
    rng = np.random.default_rng(11)
    raw = prob.lift(exact + 1e-2 * rng.standard_normal(exact.shape))
    tr = run(mesh_info, ops, raw, policy())
    assert tr.final_status == fw.STATUS_ACCEPTED_CORRECTED
    assert tr.candidate_certificate.trajectory_sha256 != tr.final_certificate.trajectory_sha256
    assert tr.final_certificate is tr.correction_certificate
    assert tr.final_certificate.verify_identity(tr.delivered_coefficients)
    assert tr.final_certificate.pcg_iteration_cap == 200


# ---------------------------------------------------------------- §2.4 time grid / scheme

@pytest.mark.parametrize("T,n", [(0.1, 60), (0.5179218421690166, 60), (1.0, 7), (0.10040483502671123, 60), (0.73, 1000)])
def test_linspace_grids_accepted_and_nodes_are_trusted_convention(T, n):
    m, dt, nodes = validate_time_grid(np.linspace(0.0, T, n + 1), T, schemes=["CN"] * n, n_steps=n)
    assert m == n and dt == T / n
    assert np.array_equal(nodes, np.arange(n + 1) * (T / n))


def test_nonuniform_grid_rejected():
    T, n = 0.5, 20
    g = T * (np.linspace(0, 1, n + 1) ** 1.5)
    with pytest.raises(UnsupportedSchemeError):
        validate_time_grid(g, T)


@pytest.mark.parametrize("schemes", [["BE"] * 2 + ["CN"] * 18, ["CN"] * 19 + ["RANNACHER"]])
def test_non_cn_scheme_rejected(schemes):
    with pytest.raises(UnsupportedSchemeError):
        validate_time_grid(np.linspace(0, 0.5, 21), 0.5, schemes=schemes)


@pytest.mark.parametrize("bad", ["decreasing", "nan", "start", "end", "length", "short"])
def test_malformed_grid_rejected(bad):
    g = np.linspace(0.0, 0.5, 21)
    kw = {}
    if bad == "decreasing":
        g[5], g[6] = g[6], g[5]
    elif bad == "nan":
        g[3] = np.nan
    elif bad == "start":
        g = g + 1e-3
    elif bad == "end":
        g = np.linspace(0.0, 0.49, 21)
    elif bad == "length":
        kw["n_steps"] = 19
    elif bad == "short":
        g = g[:1]
    with pytest.raises(FreePathInputError):
        validate_time_grid(g, 0.5, **kw)


def test_workflow_rejects_wrong_scheme(prob, exact, mesh_info, ops):
    with pytest.raises(UnsupportedSchemeError):
        run(mesh_info, ops, prob.lift(exact), policy(), schemes=["BE"] + ["CN"] * (N_STEPS - 1))


# ---------------------------------------------------------------- §2.4 solver status

def test_forced_gmres_nonconvergence_not_relabelled_fallback_runs(prob, exact, mesh_info, ops):
    """gmres_maxiter=1, restart=2 cannot reach rtol=1e-10 from a poor warm start:
    every step is recorded nonconverged; the recertified iterate fails, the
    dense fallback runs and is recertified; nothing claims convergence."""
    rng = np.random.default_rng(12)
    raw = prob.lift(exact + 1e-1 * rng.standard_normal(exact.shape))
    tr = run(mesh_info, ops, raw, policy(gmres_maxiter=1, gmres_restart=2))
    assert tr.correction is not None and not tr.correction.all_converged
    assert not any(tr.correction.converged_per_step)
    assert all(r > 1e-10 for r in tr.correction.relative_linear_residual_per_step)
    assert not tr.correction_certificate.accepted
    assert tr.fallback is not None and tr.final_status == fw.STATUS_ACCEPTED_FALLBACK
    assert tr.final_certificate is tr.fallback_certificate


def test_forced_nonconvergence_without_fallback_is_a_failure(prob, exact, mesh_info, ops):
    rng = np.random.default_rng(12)
    raw = prob.lift(exact + 1e-1 * rng.standard_normal(exact.shape))
    tr = run(mesh_info, ops, raw, policy(gmres_maxiter=1, gmres_restart=2, fallback="none"))
    assert tr.final_status == fw.STATUS_NOT_CERTIFIED
    assert not tr.output_delivered and tr.delivered_coefficients is None and tr.final_certificate is None


def test_nonconverged_but_certificate_qualified(prob, exact, mesh_info, ops):
    """rtol=1e-17 is below the attainable relative-residual floor (~3e-16 for
    this system, i.e. a few eps), so GMRES with maxiter=1 must report
    nonconvergence even from an essentially exact warm start; the recomputed
    certificate still meets the tolerance -> certificate-qualified acceptance,
    never 'converged'."""
    raw = prob.lift(exact)
    raw[0, prob.fs.free_idx] += 1e-2  # reject the candidate via beta_0 only
    tr = run(mesh_info, ops, raw, policy(gmres_rtol=1e-17, gmres_maxiter=1, gmres_restart=1))
    assert not tr.candidate_certificate.accepted
    assert not tr.correction.all_converged
    assert tr.final_status == fw.STATUS_ACCEPTED_CORRECTED_NONCONVERGED


def test_correction_exception_recorded(prob, exact, mesh_info, ops, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(constrained_mod, "free_cn_step", boom)
    rng = np.random.default_rng(13)
    raw = prob.lift(exact + 1e-2 * rng.standard_normal(exact.shape))
    tr = run(mesh_info, ops, raw, policy(fallback="none"))
    assert tr.correction.aborted and "injected solver failure" in tr.correction.exceptions[0]
    assert tr.final_status == fw.STATUS_CORRECTION_FAILED and not tr.output_delivered


def test_driver_no_longer_discards_convergence_flag(prob, mesh_info, monkeypatch):
    real = constrained_mod.free_cn_step

    def nonconv(*a, **k):
        z, it, _ = real(*a, **k)
        return z, it, False

    monkeypatch.setattr(constrained_mod, "free_cn_step", nonconv)
    with pytest.raises(FreeSolveNonconvergence):
        solve_free_cn_trajectory(MU_SPD, PAYOFF, mesh_info, _tg(MU_SPD), prob.ops, prob.fs, prob.free_ops,
                                 prob.z0_target)


# ---------------------------------------------------------------- invalid inputs / bounds

@pytest.mark.parametrize("field,value", [
    ("tolerance", -1e-3), ("tolerance", 0.0), ("tolerance", float("nan")), ("tolerance", float("inf")),
    ("tolerance", "1e-3"), ("tolerance", True), ("quick_pcg_iterations", -1), ("max_pcg_iterations", 2.5),
    ("gmres_rtol", 0.0), ("gmres_rtol", 1.5), ("gmres_rtol", float("nan")), ("gmres_restart", 0),
    ("gmres_maxiter", 0), ("preconditioner", "ilu"), ("coercivity_strategy", "C_safety_margin"),
    ("fallback", "legacy"),
])
def test_invalid_policy_rejected(field, value):
    kw = dict(tolerance=1e-3, quick_pcg_iterations=10, max_pcg_iterations=200)
    kw[field] = value
    with pytest.raises(fw.InvalidPolicyError):
        fw.FreeWorkflowPolicy(**kw).validate()


@pytest.mark.parametrize("b0,u,eta,expect", [
    (0.0, [0.0, 1e-20], 1e-10, "valid"),
    (-1e-30, [1e-20], 1e-10, "invalid: initial_error_sq negative"),
    (float("nan"), [1e-20], 1e-10, "invalid: initial_error_sq non-finite"),
    (0.0, [1e-20, float("inf")], 1e-10, "invalid: residual upper bracket non-finite"),
    (0.0, [1e-20, -1e-25], 1e-10, "invalid: residual upper bracket negative"),
    (0.0, [1e-20], float("nan"), "invalid: residual certificate non-finite or negative"),
])
def test_bound_validity(b0, u, eta, expect):
    assert fw.bound_validity(b0, np.array(u), eta) == expect


def test_invalid_bound_never_accepts(prob, exact, mesh_info, ops, monkeypatch):
    import certification.free_problem as fp
    from certification.guaranteed_certificate import Certificate

    def nan_cert(residuals, *a, **k):
        n = len(residuals)
        return Certificate(eta=float("nan"), per_step_quadratic_form=np.full(n, np.nan),
                           n_pcg_iterations=np.zeros(n, int), coordinate_space="constrained_free")

    monkeypatch.setattr(fp, "eta_guar", nan_cert)
    tr = run(mesh_info, ops, prob.lift(exact), policy(fallback="none"))
    assert tr.candidate_certificate.bound_status.startswith("invalid")
    assert not tr.output_delivered and tr.final_status == fw.STATUS_NOT_CERTIFIED


def test_predictor_shape_mismatch_rejected(prob, mesh_info, ops):
    with pytest.raises(FreePathInputError):
        run(mesh_info, ops, np.zeros((N_STEPS + 1, prob.fs.n_free)), policy())


# ---------------------------------------------------------------- legacy failure, free gate passes

def test_legacy_gate_failure_processed_in_free_coordinates_without_legacy_calls(mesh, mesh_info, ops):
    """Q_audit_0000: full S_h indefinite, free S_I SPD. Under forbid_legacy_calls
    the whole workflow runs; every certification call received the free
    operator (dim n_free); no legacy entry point was reached."""
    assert not compute_stability_constants(MU_FULL_INDEFINITE, ops).is_spd  # outside the guard
    traj = solve_cgmy(MU_FULL_INDEFINITE, PAYOFF, mesh, TimeGrid(n_steps=N_STEPS), mode="toeplitz")
    rng = np.random.default_rng(14)
    raw = traj.coefficients + 1e-3 * rng.standard_normal(traj.coefficients.shape)
    with forbid_legacy_calls() as log:
        tr = run(mesh_info, ops, raw, policy(tolerance=1e-6), mu=MU_FULL_INDEFINITE)
    s = summarize(log)
    assert s["blocked"] == []
    audited = [e for e in log if "ops_class" in e]
    assert audited and all(e["ops_class"] == "FreeOperatorCache" and e["dim"] == 62 for e in audited)
    assert tr.gate["space"] == "constrained_free" and tr.gate["is_spd_by_estimate"]
    assert tr.final_status in fw.ACCEPTED_STATUSES


def test_guard_actually_blocks_legacy_paths(prob, ops):
    from certification.residuals import compute_all_residuals  # noqa: F401  (module import before patching)
    import certification.residuals as res

    with forbid_legacy_calls() as log:
        with pytest.raises(LegacyCallError):
            res.compute_all_residuals(None, MU_SPD, ops)
        with pytest.raises(LegacyCallError):
            ops.apply_S(MU_SPD, np.zeros(ops.n_dof))
        with pytest.raises(LegacyCallError):
            ops.apply_M_inv(np.zeros(ops.n_dof))
        import certification.free_problem as fp
        with pytest.raises(LegacyCallError):
            fp.compute_stability_constants(MU_SPD, ops)
    assert len(summarize(log)["blocked"]) == 4
    assert ops.apply_S(MU_SPD, np.zeros(ops.n_dof)).shape == (ops.n_dof,)  # restored


# ---------------------------------------------------------------- point evaluation on delivered output

def test_pointwise_price_bound_on_delivered_output(prob, exact, mesh_info, ops):
    """|V_h(x) - V_delivered(x)| <= beta_I(x) * eta_aug at every time level,
    with V_h from the dense direct (exact-discrete up to its residual) solution."""
    rng = np.random.default_rng(15)
    raw = prob.lift(exact + 2e-6 * rng.standard_normal(exact.shape))
    xs = (math.log(80.0), math.log(100.0), math.log(120.0))
    tr = run(mesh_info, ops, raw, policy(tolerance=1e-3), eval_points=xs)
    assert tr.final_status == fw.STATUS_ACCEPTED_DIRECT
    from certification.free_problem import free_eta_sharp, free_point_functional, free_residuals
    ref_allow = free_eta_sharp(prob, free_residuals(prob, exact)).eta  # reference's own distance to exact
    for pe in tr.point_evaluation:
        ph = free_point_functional(prob, pe["x"])
        err = max(abs(ph @ (tr.delivered_coefficients[n, prob.fs.free_idx] - exact[n])) for n in range(N_STEPS + 1))
        assert err <= pe["beta_I"] * (tr.final_certificate.augmented_certificate + ref_allow)
