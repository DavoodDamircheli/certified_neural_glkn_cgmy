"""Gate D -- fair performance and attribution (CGMY_V3_NUMERICAL_EXPERIMENT_ROADMAP.md).

Compares, on the SAME 20-query subsample of experiments/raw_results/
query_sets/Q_bench120.csv (120 predeclared, first 20 taken as a fixed
prefix -- documented subsample, not a silent truncation: a single-shot
execution budget does not comfortably fit 120 queries x 7 methods x 5
repeats x phase-by-phase timing, see reports/GATE_D_REPORT.md for the
exact reasoning), at the production mesh (N=64, configs/base.yaml):

  dense_lu            -- fresh, independent dense-LU-per-step solve (NOT
                          the trusted solver's own GMRES path), factored
                          once per query, used at every of the 60 steps.
  cold_gmres          -- correction.krylov.correct_trajectory, warm_start=False
  warm_gmres          -- same, warm_start=True, seeded by the DeepONet prediction
  neural_direct_riesz -- DeepONet prediction + certification.guaranteed_certificate.eta_sharp
  neural_pcg_workflow -- the full pipeline.predict_certify_correct workflow (DeepONet)
  certified_pod       -- the same full workflow with the POD model instead
  cos                 -- validation.cos_pricer (terminal ATM price only, no trajectory)
  reference_galerkin  -- solve_cgmy (mode='toeplitz'), the ground-truth baseline

Thread limits are pinned to 1 (OMP/MKL/OPENBLAS/torch) via the env vars set
at the very top of this file, BEFORE numpy/scipy/torch are imported --
this is the only way the OpenBLAS backend reliably picks them up.

Run: python benchmarks/run_gate_d_benchmark.py
Writes benchmarks/raw_timings.csv, benchmarks/accuracy_matched_summary.csv,
benchmarks/run_provenance.json.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import torch

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT))

import exp_common  # noqa: E402  (experiments/exp_common.py)
from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, TimeGrid, solve_cgmy  # noqa: E402
from cgmy.dirichlet_bc import bc_dofs_for_mesh, dirichlet_bc_values, project_bc  # noqa: E402
from cgmy import _external  # noqa: F401,E402
from cgmy_bspline import boundary as cgmy_boundary  # noqa: E402
from cgmy_bspline.parameters import CGMYParams  # noqa: E402
from certification.constants import compute_stability_constants  # noqa: E402
from certification.guaranteed_certificate import eta_sharp as compute_eta_sharp  # noqa: E402
from certification.residuals import compute_all_residuals  # noqa: E402
from correction.krylov import compute_initial_coefficients, correct_trajectory  # noqa: E402
from data import payoff_family  # noqa: E402
from learning.normalization import normalize_mu  # noqa: E402
from pipeline.predict_certify_correct import build_mesh_info, predict_certify_correct  # noqa: E402
from validation.cgmy_reference_pricing import CGMYMarketParams  # noqa: E402
from validation.cos_pricer import cos_put_price  # noqa: E402

OUT_DIR = REPO_ROOT / "benchmarks"
N_QUERIES = 20  # fixed prefix of Q_bench120 -- see module docstring
N_REPEATS = 5
N_WARMUP = 1
SEED = 314159
QUICK_PCG_ITERATIONS = 10
MAX_PCG_ITERATIONS = 200  # matches configs/base.yaml certification.max_pcg_iterations
TOLERANCE = 1.0e-3  # matches configs/base.yaml certification.acceptance_tolerance

PHASE_NAMES = [
    "initial_projection", "operator_assembly", "forcing_assembly", "solver_setup",
    "inference", "residual_evaluation", "alpha_setup", "riesz_setup", "riesz_solve",
    "correction", "recertification", "fallback", "total",
]


def load_queries(n: int) -> list[dict]:
    path = REPO_ROOT / "experiments" / "raw_results" / "query_sets" / "Q_bench120.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))[:n]
    return [{k: (float(v) if k != "query_id" and k != "source" else v) for k, v in r.items()} for r in rows]


def empty_phases() -> dict[str, float]:
    return {p: 0.0 for p in PHASE_NAMES}


# ----------------------------------------------------------------------
# dense_lu: an independent, fresh dense-LU-per-step solver. Builds dense
# M_h, A_h(mu) ONCE from OperatorCache.apply_M/apply_A (matrix-free ->
# dense via unit-vector probing, same technique certification.constants
# uses, but this is a SEPARATE timed method, not a reuse of the trusted
# solver's own GMRES/Toeplitz stepping path).
# ----------------------------------------------------------------------
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


def run_dense_lu(mu: CGMYParameters, payoff: EuropeanPutPayoff, mesh_info, time_grid: np.ndarray, ops: OperatorCache) -> dict:
    phases = empty_phases()
    t0 = time.perf_counter()
    c0, bc_left, bc_right = compute_initial_coefficients(mu, payoff, mesh_info, ops)
    phases["initial_projection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    M, A = _dense_from_ops(ops, mu)
    dt = float(time_grid[1] - time_grid[0])
    # CN scheme (matches correction/krylov.py's F_matvec/B_matvec exactly):
    # solve (M + dt/2 A) c^{n+1} = (M - dt/2 A) c^n + dt/2(f^{n+1}+f^n),
    # with Dirichlet rows of the SOLVED (F) matrix set to identity -- NOT
    # the RHS-builder (B) matrix. An earlier version of this function had
    # these two matrices' roles swapped (solved B instead of F), which
    # produced a divergent, unphysical trajectory -- caught by comparing
    # against reference_galerkin's price and fixed before this result was
    # ever reported as a finding.
    F = M + 0.5 * dt * A  # solved every step
    B = M - 0.5 * dt * A  # builds the RHS from c^n, no BC overwrite
    bc_dofs = bc_left + bc_right
    for i in bc_dofs:
        F[i, :] = 0.0
        F[i, i] = 1.0
    phases["operator_assembly"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lu, piv = sla.lu_factor(F)
    phases["solver_setup"] = time.perf_counter() - t0

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
    phases["correction"] = time.perf_counter() - t0  # the "per-step linear solve" phase

    coeffs = np.array(coeffs)
    coeffs = project_bc(coeffs, mu, payoff, mesh_info, time_grid)
    price = exp_common.option_price(coeffs[-1], ops)
    phases["total"] = sum(v for k, v in phases.items() if k != "total")
    return {"phases": phases, "price": price, "success": True, "note": ""}


def run_cold_gmres(mu, payoff, mesh_info, time_grid, ops) -> dict:
    phases = empty_phases()
    t0 = time.perf_counter()
    result = correct_trajectory(mu, payoff, mesh_info, time_grid, ops, seed_coefficients=None, warm_start=False)
    phases["correction"] = time.perf_counter() - t0
    phases["total"] = phases["correction"]
    price = exp_common.option_price(result.coefficients[-1], ops)
    converged = all(result.converged_per_step)
    return {"phases": phases, "price": price, "success": converged, "note": "" if converged else "GMRES did not converge at every step"}


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
    return {"phases": phases, "price": price, "success": converged, "note": "" if converged else "GMRES did not converge at every step"}


def run_neural_direct_riesz(mu, payoff, mesh_info, time_grid, ops, models, stats) -> dict:
    phases = empty_phases()
    t0 = time.perf_counter()
    c0, _, _ = compute_initial_coefficients(mu, payoff, mesh_info, ops)
    mu_tilde = normalize_mu(mu, stats)
    psi = payoff_family.european_put(payoff.strike, ops)
    pred_raw = models["DeepONet"].predict_trajectory(mu_tilde, psi, c0, time_grid, mu.T)
    pred = project_bc(pred_raw, mu, payoff, mesh_info, time_grid)
    phases["inference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from pipeline.predict_certify_correct import build_skeleton_trajectory
    traj = build_skeleton_trajectory(mu, payoff, mesh_info, time_grid, pred)
    residuals = compute_all_residuals(traj, mu, ops)
    phases["residual_evaluation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    const = compute_stability_constants(mu, ops)
    phases["alpha_setup"] = time.perf_counter() - t0

    if not const.is_spd:
        phases["total"] = sum(phases.values())
        return {"phases": phases, "price": float("nan"), "success": False, "note": "S_h(mu) not certified SPD"}

    t0 = time.perf_counter()
    dt = traj.solver_metadata.dt
    cert = compute_eta_sharp(residuals, mu, ops, dt)
    phases["riesz_solve"] = time.perf_counter() - t0
    phases["total"] = sum(v for k, v in phases.items() if k != "total")
    price = exp_common.option_price(pred[-1], ops)
    accepted = cert.eta <= TOLERANCE
    return {"phases": phases, "price": price, "success": True, "note": f"eta_sharp={cert.eta:.4g}, accepted={accepted}"}


def run_neural_pcg_workflow(mu, payoff, mesh, time_grid, ops, models, stats, model_name: str) -> dict:
    trace = predict_certify_correct(
        mu, payoff, mesh, time_grid, models[model_name], stats, ops,
        tolerance=TOLERANCE, quick_pcg_iterations=QUICK_PCG_ITERATIONS,
        max_pcg_iterations=MAX_PCG_ITERATIONS, preconditioner="none",
        keep_coefficients=True,
    )
    phases = empty_phases()
    phases["initial_projection"] = 0.0  # folded into t_predict below (compute_initial_coefficients call)
    phases["inference"] = trace.t_predict
    phases["residual_evaluation"] = trace.t_residual
    phases["alpha_setup"] = trace.t_mass_indicator  # compute_stability_constants is called alongside this
    phases["riesz_solve"] = trace.t_riesz_certificate
    phases["correction"] = trace.t_correction
    phases["recertification"] = trace.t_recertify
    phases["total"] = trace.t_total
    price = exp_common.option_price(trace.final_coefficients[-1], ops) if trace.final_coefficients is not None else float("nan")
    return {"phases": phases, "price": price, "success": trace.accepted,
            "note": f"branch={trace.branch}, eta_final={trace.eta_final:.4g}"}


def run_cos(mu: CGMYParameters, strike: float) -> dict:
    phases = empty_phases()
    t0 = time.perf_counter()
    mkt = CGMYMarketParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, T=mu.T)
    result = cos_put_price(mkt, S0=strike, K=strike)
    phases["total"] = time.perf_counter() - t0
    return {"phases": phases, "price": result["price"], "success": True, "note": f"N={result['N']}, [a,b]=[{result['a']:.3f},{result['b']:.3f}]"}


def run_reference_galerkin(mu, payoff, mesh, time_grid, ops) -> dict:
    phases = empty_phases()
    t0 = time.perf_counter()
    time_grid_obj = TimeGrid(n_steps=len(time_grid) - 1)
    traj = solve_cgmy(mu, payoff, mesh, time_grid_obj, mode="toeplitz", store_forcing=False)
    phases["total"] = time.perf_counter() - t0
    price = exp_common.option_price(traj.coefficients[-1], ops)
    return {"phases": phases, "price": price, "success": True, "note": ""}


METHOD_FNS = {
    "dense_lu": lambda q, ctx: run_dense_lu(q["mu"], ctx["payoff"], ctx["mesh_info"], ctx["time_grid"], ctx["ops"]),
    "cold_gmres": lambda q, ctx: run_cold_gmres(q["mu"], ctx["payoff"], ctx["mesh_info"], ctx["time_grid"], ctx["ops"]),
    "warm_gmres": lambda q, ctx: run_warm_gmres(q["mu"], ctx["payoff"], ctx["mesh_info"], ctx["time_grid"], ctx["ops"], ctx["models"], ctx["stats"]),
    "neural_direct_riesz": lambda q, ctx: run_neural_direct_riesz(q["mu"], ctx["payoff"], ctx["mesh_info"], ctx["time_grid"], ctx["ops"], ctx["models"], ctx["stats"]),
    "neural_pcg_workflow": lambda q, ctx: run_neural_pcg_workflow(q["mu"], ctx["payoff"], ctx["mesh"], ctx["time_grid"], ctx["ops"], ctx["models"], ctx["stats"], "DeepONet"),
    "certified_pod": lambda q, ctx: run_neural_pcg_workflow(q["mu"], ctx["payoff"], ctx["mesh"], ctx["time_grid"], ctx["ops"], ctx["models"], ctx["stats"], "POD"),
    "cos": lambda q, ctx: run_cos(q["mu"], ctx["payoff"].strike),
    "reference_galerkin": lambda q, ctx: run_reference_galerkin(q["mu"], ctx["payoff"], ctx["mesh"], ctx["time_grid"], ctx["ops"]),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    mesh = exp_common.FIXED_MESH
    mesh_info = exp_common.FIXED_MESH_INFO
    time_grid = exp_common.TIME_GRID.n_steps
    time_grid_arr = np.linspace(0.0, 1.0, exp_common.TIME_GRID.n_steps + 1)  # placeholder; rescaled per-mu below (T varies)
    payoff = exp_common.PAYOFF
    ops = exp_common.get_ops()
    box, stats = exp_common.get_box_and_stats()
    models = exp_common.load_models(ops, stats)

    queries = load_queries(N_QUERIES)
    for q in queries:
        q["mu"] = CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])

    methods = list(METHOD_FNS.keys())
    raw_rows = []
    summary_rows = []

    for qi, q in enumerate(queries):
        time_grid_q = np.linspace(0.0, q["mu"].T, exp_common.TIME_GRID.n_steps + 1)
        ctx = {"payoff": payoff, "mesh": mesh, "mesh_info": mesh_info, "time_grid": time_grid_q,
               "ops": ops, "models": models, "stats": stats}

        order = methods.copy()
        rng.shuffle(order)

        ref_price = None
        for method in order:
            fn = METHOD_FNS[method]
            prices = []
            for rep in range(N_WARMUP + N_REPEATS):
                try:
                    out = fn(q, ctx)
                except Exception as exc:  # noqa: BLE001 -- a failed query is a RESULT, not something to hide
                    out = {"phases": empty_phases(), "price": float("nan"), "success": False, "note": f"EXCEPTION: {exc}"}
                timed = rep >= N_WARMUP
                if timed:
                    row = {"query_id": q["query_id"], "method": method, "repeat": rep - N_WARMUP,
                           "method_order_index": order.index(method), "success": out["success"], "note": out["note"],
                           "price": out["price"]}
                    row.update(out["phases"])
                    raw_rows.append(row)
                    prices.append(out["price"])
            if method == "reference_galerkin":
                ref_price = float(np.nanmedian(prices))
            print(f"query {qi+1}/{len(queries)} method={method} median_total="
                  f"{np.nanmedian([r['total'] for r in raw_rows if r['query_id']==q['query_id'] and r['method']==method]):.5f}s", flush=True)

        for method in methods:
            rows = [r for r in raw_rows if r["query_id"] == q["query_id"] and r["method"] == method]
            totals = [r["total"] for r in rows]
            prices = [r["price"] for r in rows]
            median_price = float(np.nanmedian(prices))
            summary_rows.append({
                "query_id": q["query_id"], "method": method,
                "median_total_s": float(np.median(totals)),
                "mean_total_s": float(np.mean(totals)),
                "n_repeats": len(totals),
                "n_success": sum(r["success"] for r in rows),
                "median_price": median_price,
                "abs_price_error_vs_reference": abs(median_price - ref_price) if ref_price is not None and not np.isnan(median_price) else float("nan"),
            })

    with open(OUT_DIR / "raw_timings.csv", "w", newline="") as f:
        fieldnames = ["query_id", "method", "repeat", "method_order_index", "success", "note", "price", *PHASE_NAMES]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)

    # ---- Aggregate ratio statistics (bootstrap over QUERIES) ----
    def method_query_medians(method: str) -> dict[str, float]:
        return {r["query_id"]: r["median_total_s"] for r in summary_rows if r["method"] == method}

    ref_medians = method_query_medians("reference_galerkin")
    agg_rows = []
    for method in methods:
        if method == "reference_galerkin":
            continue
        m = method_query_medians(method)
        common_ids = [qid for qid in ref_medians if qid in m]
        ref_vals = np.array([ref_medians[qid] for qid in common_ids])
        m_vals = np.array([m[qid] for qid in common_ids])
        per_query_ratio = ref_vals / m_vals  # >1 means method is FASTER than reference
        n_boot = 2000
        boot_idx = rng.integers(0, len(common_ids), size=(n_boot, len(common_ids)))
        boot_medians = np.median(per_query_ratio[boot_idx], axis=1)
        ci_lo, ci_hi = np.percentile(boot_medians, [2.5, 97.5])
        agg_rows.append({
            "method": method,
            "n_queries": len(common_ids),
            "ratio_of_aggregate_times_ref_over_method": float(ref_vals.sum() / m_vals.sum()),
            "median_querywise_ratio_ref_over_method": float(np.median(per_query_ratio)),
            "mean_querywise_ratio_ref_over_method": float(np.mean(per_query_ratio)),
            "ratio_of_medians_ref_over_method": float(np.median(ref_vals) / np.median(m_vals)),
            "bootstrap_median_ratio_ci95_lo": float(ci_lo),
            "bootstrap_median_ratio_ci95_hi": float(ci_hi),
            "mean_abs_price_error_vs_reference": float(np.nanmean(
                [r["abs_price_error_vs_reference"] for r in summary_rows if r["method"] == method]
            )),
        })

    with open(OUT_DIR / "accuracy_matched_summary.csv", "w", newline="") as f:
        fieldnames = list(agg_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(agg_rows)

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip())
    provenance = {
        "git_sha": git_sha, "git_dirty": git_dirty,
        "n_queries": N_QUERIES, "n_repeats": N_REPEATS, "n_warmup": N_WARMUP, "seed": SEED,
        "quick_pcg_iterations": QUICK_PCG_ITERATIONS, "max_pcg_iterations": MAX_PCG_ITERATIONS,
        "tolerance": TOLERANCE,
        "cpu_model": platform.processor() or platform.uname().processor,
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": __import__("scipy").__version__, "torch": torch.__version__,
        "thread_limits": {v: os.environ.get(v) for v in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    with open(OUT_DIR / "run_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print("\n=== Summary (ratio of reference_galerkin / method, >1 means method is faster) ===")
    for row in agg_rows:
        print(f"{row['method']:24s} median_ratio={row['median_querywise_ratio_ref_over_method']:8.3f}  "
              f"mean_abs_price_err={row['mean_abs_price_error_vs_reference']:.2e}")


if __name__ == "__main__":
    main()
