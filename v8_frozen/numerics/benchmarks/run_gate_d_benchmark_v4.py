"""v4 Phase 4: corrected Gate D benchmark harness -- CGMY_V4_REMAINING_WORK.md
item 4.

Historical ``benchmarks/run_gate_d_benchmark.py`` and its existing output
CSVs (``benchmarks/raw_timings.csv``, ``accuracy_matched_summary.csv``,
``run_provenance.json``) are UNCHANGED (ground rule 2) -- this is a
separate script, writing to
``results/v4_constrained_free/gate_d_v4_<smoke|full>_<timestamp>/``.

Corrections implemented (CGMY_V4_REMAINING_WORK.md item 4), each flagged
with how it was actually resolved:

1. **Distinct status flags.** Every row now carries ``solver_converged``,
   ``certificate_computed``, ``target_met``, ``fallback_executed``,
   ``price_delivered`` -- never collapsed into one ``success`` bool.
2. **neural_direct_riesz no longer reports success when the certificate
   exceeds target.** The historical script hardcoded ``success=True``
   regardless of ``cert.eta <= TOLERANCE`` (only recorded in a free-text
   note). Fixed: ``target_met = (cert.eta <= TOLERANCE)``,
   ``price_delivered = target_met``.
3. **Failed queries' diagnostic prices excluded from accuracy aggregates,
   included in denominators.** ``price`` is still recorded for every row
   (the diagnostic number, if any was computed); the accuracy aggregate
   (price-error stats) is computed ONLY over rows with
   ``price_delivered=True``; the attempted-population outcome rate
   (target_met rate) is computed over ALL attempted rows. The two are
   reported as separate tables, never conflated into one number.
4. **Actual fallback work charged.** ``neural_pcg_workflow``/
   ``certified_pod`` already time correction via
   ``predict_certify_correct``'s own phase timers (``t_correction``) --
   confirmed, not re-derived. ``fallback_executed`` is now an explicit
   column (``trace.correction_ran``), not left implicit in the branch
   string.
5. **Dense LU and cold GMRES get the SAME residual-based accuracy
   assessment as the certified methods** -- rather than treating their
   output as definitionally correct. Both now compute ``eta_sharp`` on
   their resulting trajectory via the SAME ``certification.guaranteed_certificate``
   machinery every other certified method uses, and ``target_met`` /
   ``price_delivered`` gate on it exactly the same way.
6. **Per-repeat independent method-order shuffling.** The historical
   script shuffled method order ONCE per query, before the repeat loop, so
   every repeat of a given query ran methods in the identical order. Fixed:
   the loop is restructured repeat-outer, query-order-inner -- a fresh
   ``rng.shuffle`` for every (query, repeat) pair.
7. **Cold-setup fairness (the "shared operator cache created outside the
   alternative methods' timers, while the wrapped reference creates setup
   inside its timer" bug).** Every custom-ops-based method (dense_lu,
   cold_gmres, warm_gmres, neural_direct_riesz, neural_pcg_workflow,
   certified_pod) now builds a FRESH ``OperatorCache`` per QUERY, inside an
   explicitly timed ``operator_cache_setup`` phase charged to every one of
   them -- equalizing against ``reference_galerkin``/``cos``, which always
   pay a from-scratch setup cost internally every call (``solve_cn``
   builds its own matrices every invocation; there is no way to inject an
   external warm cache into the vendored, unmodified trusted solver).
   SCOPED REDUCTION (recorded honestly, not silently): this fixes the
   COLD-setup fairness bug but does NOT implement a full separate
   "amortised reuse" workload variant on top of it (item 4's "measure cold
   setup and amortised reuse as separate workloads" asks for both) -- doing
   that properly requires its own warm-cache-reuse-across-queries pass,
   deferred; this run reports the COLD-every-query numbers only, labeled
   as such in every output file and in the morning summary.
8. **Query-level bootstrap.** Already correct in the historical script
   (resamples over query ids, not repeats) -- verified, not re-implemented.
9. **Full enclosing wall-clock timers, warmup, repetitions, thread
   limits.** Unchanged pattern (N_WARMUP + N_REPEATS, perf_counter, thread
   env vars pinned before numpy/scipy/torch import).
10. **Per-query records of coefficient hashes, terminal prices, branch,
    bound, norm target, phase times.** Added: ``coeff_hash`` (SHA-256 of
    the final coefficient vector), ``branch``, ``eta_final``,
    ``norm_target`` (=TOLERANCE) columns.

Run (smoke, this session): ``python benchmarks/run_gate_d_benchmark_v4.py --smoke``
Full Q_bench120 run (NOT executed tonight -- see reports/v4_overnight_log.md
Phase 4 section for why): ``python benchmarks/run_gate_d_benchmark_v4.py --full``
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import torch

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT))

import exp_common  # noqa: E402
from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, TimeGrid, solve_cgmy  # noqa: E402
from cgmy.dirichlet_bc import dirichlet_bc_values, project_bc  # noqa: E402
from cgmy import _external  # noqa: F401,E402
from cgmy_bspline import boundary as cgmy_boundary  # noqa: E402
from cgmy_bspline.parameters import CGMYParams  # noqa: E402
from certification.constants import compute_stability_constants  # noqa: E402
from certification.guaranteed_certificate import eta_sharp as compute_eta_sharp  # noqa: E402
from certification.residuals import compute_all_residuals  # noqa: E402
from correction.krylov import compute_initial_coefficients, correct_trajectory  # noqa: E402
from data import payoff_family  # noqa: E402
from learning.normalization import normalize_mu  # noqa: E402
from pipeline.predict_certify_correct import build_skeleton_trajectory, predict_certify_correct  # noqa: E402
from validation.cgmy_reference_pricing import CGMYMarketParams  # noqa: E402
from validation.cos_pricer import cos_put_price  # noqa: E402

RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
N_SMOKE_QUERIES = 8
N_FULL_QUERIES = 120
N_REPEATS = 5
N_WARMUP = 1
SEED = 314159
QUICK_PCG_ITERATIONS = 10
MAX_PCG_ITERATIONS = 200
TOLERANCE = 1.0e-3

PHASE_NAMES = [
    "operator_cache_setup", "initial_projection", "forcing_assembly", "solver_setup",
    "inference", "residual_evaluation", "alpha_setup", "riesz_setup", "riesz_solve",
    "correction", "recertification", "total",
]


def empty_phases() -> dict[str, float]:
    return {p: 0.0 for p in PHASE_NAMES}


def coeff_hash(c: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(c, dtype=np.float64).tobytes()).hexdigest()[:16]


def load_queries(path: Path, n: int) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))[:n]
    return [{k: (float(v) if k not in ("query_id", "source") else v) for k, v in r.items()} for r in rows]


def _dense_from_ops(ops: OperatorCache, mu: CGMYParameters) -> tuple[np.ndarray, np.ndarray]:
    n = ops.n_dof
    M = np.zeros((n, n))
    A = np.zeros((n, n))
    e = np.zeros(n)
    for j in range(n):
        e[j] = 1.0
        M[:, j] = ops.apply_M(e)
        A[:, j] = ops.apply_A(mu, e)
        e[j] = 0.0
    return M, A


def _certify_trajectory(coeffs, mu, payoff, mesh_info, time_grid, ops) -> tuple[bool, float, bool]:
    """Shared residual-based certification path for dense_lu/cold_gmres --
    item 4 correction 5: the same assessment the certified neural methods
    get, not an output assumed correct by construction. Uses
    ``build_skeleton_trajectory`` exactly as ``pipeline.predict_certify_correct``
    does for a neural prediction -- its forcing is recomputed via the SAME
    (currently always-zero) ``boundary_forcing_vector`` call every other
    skeleton trajectory in this project uses, bit-identical, not a
    substituted value (Hard Invariant #2)."""
    traj = build_skeleton_trajectory(mu, payoff, mesh_info, time_grid, coeffs)
    residuals = compute_all_residuals(traj, mu, ops)
    const = compute_stability_constants(mu, ops)
    if not const.is_spd:
        return False, float("nan"), False
    cert = compute_eta_sharp(residuals, mu, ops, traj.solver_metadata.dt)
    target_met = cert.eta <= TOLERANCE
    return True, cert.eta, target_met


def run_dense_lu(mu, payoff, mesh_info, time_grid, ops) -> dict:
    phases = empty_phases()
    t0 = time.perf_counter()
    c0, bc_left, bc_right = compute_initial_coefficients(mu, payoff, mesh_info, ops)
    phases["initial_projection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    M, A = _dense_from_ops(ops, mu)
    dt = float(time_grid[1] - time_grid[0])
    F = M + 0.5 * dt * A
    B = M - 0.5 * dt * A
    bc_dofs = bc_left + bc_right
    for i in bc_dofs:
        F[i, :] = 0.0
        F[i, i] = 1.0
    phases["solver_setup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lu, piv = sla.lu_factor(F)
    phases["solver_setup"] += time.perf_counter() - t0

    params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=payoff.strike, T=mu.T)
    c = c0.copy()
    coeffs = [c0.copy()]
    tau = 0.0
    t0 = time.perf_counter()
    for n in range(len(time_grid) - 1):
        tau_new = float(time_grid[n + 1])
        f_old = cgmy_boundary.boundary_forcing_vector(mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, tau, None)
        f_new = cgmy_boundary.boundary_forcing_vector(mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, tau_new, None)
        rhs = B @ c + 0.5 * dt * (f_new + f_old)
        c_L, c_R = dirichlet_bc_values(mu, payoff, mesh_info, tau_new)
        for i in bc_left:
            rhs[i] = c_L
        for i in bc_right:
            rhs[i] = c_R
        c = sla.lu_solve((lu, piv), rhs)
        coeffs.append(c.copy())
        tau = tau_new
    phases["correction"] = time.perf_counter() - t0

    coeffs = np.array(coeffs)
    coeffs = project_bc(coeffs, mu, payoff, mesh_info, time_grid)
    price = exp_common.option_price(coeffs[-1], ops)

    t0 = time.perf_counter()
    cert_computed, eta_final, target_met = _certify_trajectory(coeffs, mu, payoff, mesh_info, time_grid, ops)
    phases["recertification"] = time.perf_counter() - t0
    phases["total"] = sum(v for k, v in phases.items() if k != "total")
    return dict(
        phases=phases, price=price, coeff_hash=coeff_hash(coeffs[-1]),
        solver_converged=True, certificate_computed=cert_computed,
        target_met=bool(target_met), fallback_executed=False,
        price_delivered=bool(target_met), eta_final=eta_final, branch="dense_lu_direct", note="",
    )


def run_cold_gmres(mu, payoff, mesh_info, time_grid, ops) -> dict:
    phases = empty_phases()
    t0 = time.perf_counter()
    result = correct_trajectory(mu, payoff, mesh_info, time_grid, ops, seed_coefficients=None, warm_start=False)
    phases["correction"] = time.perf_counter() - t0
    price = exp_common.option_price(result.coefficients[-1], ops)
    converged = all(result.converged_per_step)

    # Item 4 correction 5: the SAME residual-based assessment the certified
    # methods get, not an output assumed correct merely because GMRES
    # reported convergence.
    t0 = time.perf_counter()
    cert_computed, eta_final, target_met = _certify_trajectory(
        result.coefficients, mu, payoff, mesh_info, time_grid, ops
    )
    phases["recertification"] = time.perf_counter() - t0
    phases["total"] = phases["correction"] + phases["recertification"]
    return dict(
        phases=phases, price=price, coeff_hash=coeff_hash(result.coefficients[-1]),
        solver_converged=converged, certificate_computed=cert_computed,
        target_met=bool(target_met and converged), fallback_executed=False,
        price_delivered=bool(target_met and converged), eta_final=eta_final, branch="cold_gmres",
        note="" if converged else "GMRES did not converge at every step",
    )


def run_warm_gmres(mu, payoff, mesh_info, time_grid, ops, models, stats) -> dict:
    phases = empty_phases()
    t0 = time.perf_counter()
    c0, _, _ = compute_initial_coefficients(mu, payoff, mesh_info, ops)
    mu_tilde = normalize_mu(mu, stats)
    psi = payoff_family.european_put(payoff.strike, ops)
    pred = models["DeepONet"].predict_trajectory(mu_tilde, psi, c0, time_grid, mu.T)
    phases["inference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = correct_trajectory(mu, payoff, mesh_info, time_grid, ops, seed_coefficients=pred, warm_start=True)
    phases["correction"] = time.perf_counter() - t0
    phases["total"] = phases["inference"] + phases["correction"]
    price = exp_common.option_price(result.coefficients[-1], ops)
    converged = all(result.converged_per_step)
    return dict(
        phases=phases, price=price, coeff_hash=coeff_hash(result.coefficients[-1]),
        solver_converged=converged, certificate_computed=False,
        target_met=converged, fallback_executed=True,
        price_delivered=converged, eta_final=float("nan"), branch="warm_gmres",
        note="" if converged else "GMRES did not converge at every step",
    )


def run_neural_direct_riesz(mu, payoff, mesh_info, time_grid, ops, models, stats) -> dict:
    """Item 4 correction 2: target_met/price_delivered now gate on
    cert.eta <= TOLERANCE -- the historical script's `success=True`
    regardless of this comparison was the named bug."""
    phases = empty_phases()
    t0 = time.perf_counter()
    c0, _, _ = compute_initial_coefficients(mu, payoff, mesh_info, ops)
    mu_tilde = normalize_mu(mu, stats)
    psi = payoff_family.european_put(payoff.strike, ops)
    pred_raw = models["DeepONet"].predict_trajectory(mu_tilde, psi, c0, time_grid, mu.T)
    pred = project_bc(pred_raw, mu, payoff, mesh_info, time_grid)
    phases["inference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    traj = build_skeleton_trajectory(mu, payoff, mesh_info, time_grid, pred)
    residuals = compute_all_residuals(traj, mu, ops)
    phases["residual_evaluation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    const = compute_stability_constants(mu, ops)
    phases["alpha_setup"] = time.perf_counter() - t0

    if not const.is_spd:
        phases["total"] = sum(v for k, v in phases.items() if k != "total")
        return dict(
            phases=phases, price=float("nan"), coeff_hash=coeff_hash(pred[-1]),
            solver_converged=True, certificate_computed=False, target_met=False,
            fallback_executed=False, price_delivered=False, eta_final=float("nan"),
            branch="not_spd_uncertifiable", note="S_h(mu) not certified SPD",
        )

    t0 = time.perf_counter()
    dt = traj.solver_metadata.dt
    cert = compute_eta_sharp(residuals, mu, ops, dt)
    phases["riesz_solve"] = time.perf_counter() - t0
    phases["total"] = sum(v for k, v in phases.items() if k != "total")
    price = exp_common.option_price(pred[-1], ops)
    target_met = cert.eta <= TOLERANCE
    return dict(
        phases=phases, price=price, coeff_hash=coeff_hash(pred[-1]),
        solver_converged=True, certificate_computed=True, target_met=bool(target_met),
        fallback_executed=False, price_delivered=bool(target_met), eta_final=cert.eta,
        branch="neural_direct_riesz", note=f"eta_sharp={cert.eta:.4g}",
    )


def run_neural_pcg_workflow(mu, payoff, mesh, time_grid, ops, models, stats, model_name: str) -> dict:
    trace = predict_certify_correct(
        mu, payoff, mesh, time_grid, models[model_name], stats, ops,
        tolerance=TOLERANCE, quick_pcg_iterations=QUICK_PCG_ITERATIONS,
        max_pcg_iterations=MAX_PCG_ITERATIONS, preconditioner="none",
        keep_coefficients=True,
    )
    phases = empty_phases()
    phases["inference"] = trace.t_predict
    phases["residual_evaluation"] = trace.t_residual
    phases["alpha_setup"] = trace.t_mass_indicator
    phases["riesz_solve"] = trace.t_riesz_certificate
    phases["correction"] = trace.t_correction
    phases["recertification"] = trace.t_recertify
    phases["total"] = trace.t_total
    price = exp_common.option_price(trace.final_coefficients[-1], ops) if trace.final_coefficients is not None else float("nan")
    certificate_computed = not np.isnan(trace.eta_final)
    return dict(
        phases=phases, price=price,
        coeff_hash=coeff_hash(trace.final_coefficients[-1]) if trace.final_coefficients is not None else "",
        solver_converged=(trace.correction_converged if trace.correction_converged is not None else True),
        certificate_computed=certificate_computed, target_met=bool(trace.accepted),
        fallback_executed=bool(trace.correction_ran), price_delivered=bool(trace.accepted),
        eta_final=trace.eta_final, branch=trace.branch, note=f"branch={trace.branch}",
    )


def run_cos(mu: CGMYParameters, strike: float) -> dict:
    """COS is a terminal-price-only method with no trajectory/certificate
    -- certificate_computed is structurally False (not "failed", not
    applicable), and it always delivers A price (its accuracy, not its
    delivery, is what is being measured) -- item 4's "add COS at the same
    verified total-price target" is satisfied by including it in the SAME
    accuracy-aggregate table as every certified method, compared against
    the same reference price, rather than inventing a separate pass/fail
    budget this session has no spec for."""
    phases = empty_phases()
    t0 = time.perf_counter()
    mkt = CGMYMarketParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, T=mu.T)
    result = cos_put_price(mkt, S0=strike, K=strike)
    phases["total"] = time.perf_counter() - t0
    return dict(
        phases=phases, price=result["price"], coeff_hash="",
        solver_converged=True, certificate_computed=False, target_met=True,
        fallback_executed=False, price_delivered=True, eta_final=float("nan"),
        branch="cos", note=f"N={result['N']}",
    )


def run_reference_galerkin(mu, payoff, mesh, time_grid, ops) -> dict:
    phases = empty_phases()
    t0 = time.perf_counter()
    time_grid_obj = TimeGrid(n_steps=len(time_grid) - 1)
    traj = solve_cgmy(mu, payoff, mesh, time_grid_obj, mode="toeplitz", store_forcing=False)
    phases["total"] = time.perf_counter() - t0
    price = exp_common.option_price(traj.coefficients[-1], ops)
    return dict(
        phases=phases, price=price, coeff_hash=coeff_hash(traj.coefficients[-1]),
        solver_converged=True, certificate_computed=False, target_met=True,
        fallback_executed=False, price_delivered=True, eta_final=float("nan"),
        branch="reference_galerkin", note="",
    )


METHODS_NEEDING_OWN_OPS = (
    "dense_lu", "cold_gmres", "warm_gmres", "neural_direct_riesz", "neural_pcg_workflow", "certified_pod",
)


def run_method(method: str, q: dict, mesh, mesh_info, time_grid, models, stats, ops_cold) -> dict:
    if method == "dense_lu":
        return run_dense_lu(q["mu"], q["payoff"], mesh_info, time_grid, ops_cold)
    if method == "cold_gmres":
        return run_cold_gmres(q["mu"], q["payoff"], mesh_info, time_grid, ops_cold)
    if method == "warm_gmres":
        return run_warm_gmres(q["mu"], q["payoff"], mesh_info, time_grid, ops_cold, models, stats)
    if method == "neural_direct_riesz":
        return run_neural_direct_riesz(q["mu"], q["payoff"], mesh_info, time_grid, ops_cold, models, stats)
    if method == "neural_pcg_workflow":
        return run_neural_pcg_workflow(q["mu"], q["payoff"], mesh, time_grid, ops_cold, models, stats, "DeepONet")
    if method == "certified_pod":
        raise LegacyPODRetiredError(
            "v5 (roadmap §3.2): the legacy POD checkpoint is retired from this harness. The "
            "repaired mass-weighted free POD (learning.pod_free.load_pod_artifact + "
            "pipeline.free_workflow, method constrained_free_cn_v1) belongs in the repaired "
            "v5 harness (M5); adding it here would mix methods in one result group."
        )
    if method == "cos":
        return run_cos(q["mu"], q["payoff"].strike)
    if method == "reference_galerkin":
        return run_reference_galerkin(q["mu"], q["payoff"], mesh, time_grid, ops_cold)
    raise ValueError(method)


class LegacyPODRetiredError(RuntimeError):
    pass


# v5: "certified_pod" removed (legacy POD retired, see run_method).
METHODS = [
    "dense_lu", "cold_gmres", "warm_gmres", "neural_direct_riesz",
    "neural_pcg_workflow", "cos", "reference_galerkin",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help=f"{N_SMOKE_QUERIES} queries, N=64 -- tonight's run")
    g.add_argument("--full", action="store_true", help=f"{N_FULL_QUERIES} queries (Q_bench120) -- NOT run tonight")
    args = parser.parse_args()

    label = "smoke" if args.smoke else "full"
    n_queries = N_SMOKE_QUERIES if args.smoke else N_FULL_QUERIES
    out_dir = REPO_ROOT / "results" / "v4_constrained_free" / f"gate_d_v4_{label}_{RUN_STAMP}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    mesh = exp_common.FIXED_MESH
    mesh_info = exp_common.FIXED_MESH_INFO
    payoff = exp_common.PAYOFF
    box, stats = exp_common.get_box_and_stats()
    models = exp_common.load_models(exp_common.get_ops(), stats, include_legacy_pod=False)

    q_path = REPO_ROOT / "experiments" / "raw_results" / "query_sets" / "Q_bench120.csv"
    queries = load_queries(q_path, n_queries)
    for q in queries:
        q["mu"] = CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
        q["payoff"] = payoff

    raw_rows = []
    t_start = time.time()
    for qi, q in enumerate(queries):
        time_grid_q = np.linspace(0.0, q["mu"].T, exp_common.TIME_GRID.n_steps + 1)

        # Correction 7: a FRESH per-query OperatorCache, built once per
        # query (not per repeat -- that would multiply an already-timed
        # cost N_REPEATS-fold for no comparative benefit) and shared by
        # every repeat/method THIS query -- its build cost is charged
        # explicitly to "operator_cache_setup" below, in every row.
        t0 = time.perf_counter()
        ops_cold = OperatorCache(mesh, mesh.x_min, mesh.x_max)
        cache_setup_s = time.perf_counter() - t0

        # Correction 6: fresh shuffle EVERY repeat, not once per query.
        for rep in range(N_WARMUP + N_REPEATS):
            order = METHODS.copy()
            rng.shuffle(order)
            for method_order_index, method in enumerate(order):
                try:
                    out = run_method(method, q, mesh, mesh_info, time_grid_q, models, stats, ops_cold)
                    out["phases"]["operator_cache_setup"] = cache_setup_s if method in METHODS_NEEDING_OWN_OPS else 0.0
                    out["phases"]["total"] += out["phases"]["operator_cache_setup"]
                except Exception as exc:  # noqa: BLE001 -- a failed query is a result, not something to hide
                    out = dict(
                        phases=empty_phases(), price=float("nan"), coeff_hash="",
                        solver_converged=False, certificate_computed=False, target_met=False,
                        fallback_executed=False, price_delivered=False, eta_final=float("nan"),
                        branch="exception", note=f"EXCEPTION: {exc}",
                    )
                timed = rep >= N_WARMUP
                if timed:
                    row = {
                        "query_id": q["query_id"], "method": method, "repeat": rep - N_WARMUP,
                        "method_order_index": method_order_index,
                        "solver_converged": out["solver_converged"],
                        "certificate_computed": out["certificate_computed"],
                        "target_met": out["target_met"], "fallback_executed": out["fallback_executed"],
                        "price_delivered": out["price_delivered"],
                        "price": out["price"], "coeff_hash": out["coeff_hash"],
                        "branch": out["branch"], "eta_final": out["eta_final"],
                        "norm_target": TOLERANCE, "note": out["note"],
                    }
                    row.update(out["phases"])
                    raw_rows.append(row)
        elapsed = time.time() - t_start
        print(f"query {qi+1}/{len(queries)} done ({elapsed:.1f}s elapsed)", flush=True)

    with open(out_dir / "raw_timings.csv", "w", newline="") as f:
        fieldnames = [
            "query_id", "method", "repeat", "method_order_index",
            "solver_converged", "certificate_computed", "target_met", "fallback_executed",
            "price_delivered", "price", "coeff_hash", "branch", "eta_final", "norm_target", "note",
            *PHASE_NAMES,
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)

    # ---- Attempted-population outcomes (ALL rows in the denominator) ----
    outcome_rows = []
    for method in METHODS:
        rows = [r for r in raw_rows if r["method"] == method]
        n = len(rows)
        outcome_rows.append(dict(
            method=method, n_attempted=n,
            solver_converged_rate=sum(r["solver_converged"] for r in rows) / n,
            certificate_computed_rate=sum(r["certificate_computed"] for r in rows) / n,
            target_met_rate=sum(r["target_met"] for r in rows) / n,
            fallback_executed_rate=sum(r["fallback_executed"] for r in rows) / n,
            price_delivered_rate=sum(r["price_delivered"] for r in rows) / n,
        ))
    with open(out_dir / "attempted_population_outcomes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(outcome_rows[0].keys()))
        w.writeheader()
        w.writerows(outcome_rows)

    # ---- Successful-query accuracy (price_delivered=True rows ONLY) ----
    ref_rows = [r for r in raw_rows if r["method"] == "reference_galerkin" and r["price_delivered"]]
    ref_by_query = {}
    for r in ref_rows:
        ref_by_query.setdefault(r["query_id"], []).append(r["price"])
    ref_median_by_query = {qid: float(np.median(v)) for qid, v in ref_by_query.items()}

    accuracy_rows = []
    for method in METHODS:
        delivered = [r for r in raw_rows if r["method"] == method and r["price_delivered"]]
        errs = [
            abs(r["price"] - ref_median_by_query[r["query_id"]])
            for r in delivered if r["query_id"] in ref_median_by_query
        ]
        accuracy_rows.append(dict(
            method=method, n_price_delivered=len(delivered), n_comparable_to_reference=len(errs),
            mean_abs_price_error=float(np.mean(errs)) if errs else float("nan"),
            median_abs_price_error=float(np.median(errs)) if errs else float("nan"),
            max_abs_price_error=float(np.max(errs)) if errs else float("nan"),
        ))
    with open(out_dir / "successful_query_accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(accuracy_rows[0].keys()))
        w.writeheader()
        w.writerows(accuracy_rows)

    # ---- Timing ratios, bootstrapped at the QUERY level (correction 8: already-correct pattern, reused) ----
    def method_query_medians(method: str) -> dict[str, float]:
        rows = [r for r in raw_rows if r["method"] == method]
        by_q: dict[str, list[float]] = {}
        for r in rows:
            by_q.setdefault(r["query_id"], []).append(r["total"])
        return {qid: float(np.median(v)) for qid, v in by_q.items()}

    ref_medians = method_query_medians("reference_galerkin")
    timing_rows = []
    for method in METHODS:
        if method == "reference_galerkin":
            continue
        m = method_query_medians(method)
        common_ids = [qid for qid in ref_medians if qid in m]
        if not common_ids:
            continue
        ref_vals = np.array([ref_medians[qid] for qid in common_ids])
        m_vals = np.array([m[qid] for qid in common_ids])
        per_query_ratio = ref_vals / m_vals
        n_boot = 2000
        boot_idx = rng.integers(0, len(common_ids), size=(n_boot, len(common_ids)))
        boot_medians = np.median(per_query_ratio[boot_idx], axis=1)
        ci_lo, ci_hi = np.percentile(boot_medians, [2.5, 97.5])
        timing_rows.append(dict(
            method=method, n_queries=len(common_ids),
            median_querywise_ratio_ref_over_method=float(np.median(per_query_ratio)),
            bootstrap_median_ratio_ci95_lo=float(ci_lo), bootstrap_median_ratio_ci95_hi=float(ci_hi),
        ))
    with open(out_dir / "timing_ratios.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(timing_rows[0].keys()))
        w.writeheader()
        w.writerows(timing_rows)

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip())
    provenance = dict(
        label=label, git_sha=git_sha, git_dirty=git_dirty,
        n_queries=n_queries, n_repeats=N_REPEATS, n_warmup=N_WARMUP, seed=SEED,
        tolerance=TOLERANCE, python=platform.python_version(),
        numpy=np.__version__, scipy=__import__("scipy").__version__, torch=torch.__version__,
        thread_limits={v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        scope_note=(
            "cold-setup fairness fix applied (operator_cache_setup charged per query to every "
            "ops-based method); NO separate amortised-reuse workload variant implemented tonight "
            "-- see benchmarks/run_gate_d_benchmark_v4.py module docstring correction 7."
        ),
    )
    with open(out_dir / "run_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"\nwrote {out_dir}")
    print("\n=== Attempted-population outcomes ===")
    for row in outcome_rows:
        print(f"{row['method']:24s} target_met_rate={row['target_met_rate']:.2f}  "
              f"price_delivered_rate={row['price_delivered_rate']:.2f}  n={row['n_attempted']}")
    print("\n=== Successful-query accuracy (price_delivered rows only) ===")
    for row in accuracy_rows:
        print(f"{row['method']:24s} n_delivered={row['n_price_delivered']:3d}  "
              f"mean_abs_price_err={row['mean_abs_price_error']:.3e}")
    print("\n=== Timing ratios (reference_galerkin / method, >1 means method faster) ===")
    for row in timing_rows:
        print(f"{row['method']:24s} median_ratio={row['median_querywise_ratio_ref_over_method']:.3f}  "
              f"95% CI=[{row['bootstrap_median_ratio_ci95_lo']:.3f}, {row['bootstrap_median_ratio_ci95_hi']:.3f}]")


if __name__ == "__main__":
    main()
