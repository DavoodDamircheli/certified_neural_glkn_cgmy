"""Task 9 (working prompt): re-select the PCG preconditioner.

configs/base.yaml's own comment already flagged this as open: "Stage C
found mass-preconditioning increases, not decreases, kappa(M_h^{-1}
S_h(mu)) relative to unpreconditioned kappa(S_h(mu)) for the baseline test
point... 'none' may be a better default." This script checks that across
several random queries at the production mesh, benchmarking all
preconditioners this project can build:

  - none:   no preconditioner (M = I).
  - mass:   M_h^{-1} (the current default).
  - jacobi: diag(S_h(mu))^{-1} -- NEW (Task 9), via cgmy.operators.
            OperatorCache.diag_S, no dense assembly needed.
  - banded: a truncated-bandwidth approximation of S_h(mu), factored once
            and cached per mu -- NEW (Task 9), "operator-aware" in the
            sense the task means it: reuses cgmy_bspline.matrices.
            operator_matrix (trusted, unchanged), the SAME construction
            the trusted solver's own mode='toeplitz_precond' already uses
            to precondition ITS GMRES solve.

Incomplete Cholesky is NOT implemented: S_h(mu) has no natural sparse
structure (K_G/K_M are dense Toeplitz operators in this project's exact,
untruncated construction) for a standard sparse-ILU routine to exploit
without an ad hoc banding approximation -- which is exactly what 'banded'
above already is, built the trusted-and-reused way rather than invented
from scratch here.

For each of several random (SPD) mu, for each preconditioner: N_iter
(iterations to converge or hit the cap), T_PCG (wall-clock), and an
ESTIMATED condition number of the preconditioned operator M^{-1} S_h(mu)
(via the generalized eigenvalue problem S x = lambda M x, the SAME
technique certification.constants already uses for
mass_preconditioned_condition_number -- extended here to all four M's).

IMPORTANT methodology note, found while running this benchmark (see
notes/09_preconditioner.md for the full account): 'banded' pays a real,
large, ONE-TIME cost per mu to build and factor the banded system
(~540ms), but that factorization is cached per mu in OperatorCache
(banded_preconditioner_lu) -- every SUBSEQUENT solve_riesz_pcg call for
the SAME mu reuses it at ~0.6ms. Since certification.guaranteed_
certificate.eta_guar calls solve_riesz_pcg once PER TIME STEP within a
single query (N_STEPS_PER_QUERY times, matching configs/base.yaml's
production n_time_steps), a benchmark that calls solve_riesz_pcg ONCE per
(query, preconditioner) drastically overstates 'banded''s real per-call
cost and understates its production viability. This script therefore
simulates a FULL query's worth of PCG solves (N_STEPS_PER_QUERY calls,
one per synthetic residual row, same mu) per preconditioner, and reports
BOTH the first (cold) call's cost and the AMORTIZED mean cost across the
whole simulated query -- the amortized number is what should drive the
config decision, since that is what a real query actually pays.

Outputs: results/preconditioner_benchmark.csv, figures/preconditioner_benchmark.png.
Usage: PYTHONPATH=src python3 scripts/preconditioner_benchmark.py
"""

from __future__ import annotations

import csv
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.linalg as sla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgmy.operators import OperatorCache
from cgmy.reference_solver import Mesh
from certification.constants import compute_stability_constants, dense_from_apply
from certification.riesz_solver import solve_riesz_pcg
from data.sampling import load_parameter_box, sample_sobol

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

N_QUERIES = 10
N_STEPS_PER_QUERY = 60  # matches configs/base.yaml's production n_time_steps -- see module docstring
SEED = 90900
MAX_PCG_ITERATIONS = 300  # generous -- observe REAL convergence, not just a cap
PCG_TOL = 1e-14
PRECONDITIONERS = ["none", "mass", "jacobi", "banded"]

# Production mesh -- the resolution these numbers actually apply to.
N_MESH, P = 64, 3
X_MIN, X_MAX = float(np.log(100) - 8), float(np.log(100) + 8)

CSV_FIELDS = [
    "query", "C", "G", "M", "Y", "r", "T", "preconditioner",
    "mean_n_pcg_iterations", "t_pcg_first_call_ms", "t_pcg_amortized_mean_ms",
    "kappa_estimated", "mean_bracket_gap_rel",
]


def _dense_preconditioner_matrix(name: str, ops: OperatorCache, mu, S: np.ndarray) -> np.ndarray:
    n = ops.n_dof
    if name == "none":
        return np.eye(n)
    if name == "mass":
        return dense_from_apply(ops.apply_M, n)
    if name == "jacobi":
        return np.diag(ops.diag_S(mu))
    if name == "banded":
        lu = ops.banded_preconditioner_lu(mu)
        # Recover the dense banded matrix by solving against unit vectors of its OWN factorization
        # is circular; instead rebuild it the same way banded_preconditioner_lu does, densely, for
        # the kappa estimate only (a diagnostic, not on the PCG hot path).
        from cgmy_bspline import matrices as cgmy_matrices
        from cgmy_bspline.parameters import CGMYParams
        params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=1.0, T=1.0)
        bw = min(64, n)
        mats = cgmy_matrices.operator_matrix(ops.N, ops.p, ops.h, ops.x_min, params, bandwidth=bw)
        A_bnd = mats["A_h"]
        return np.asarray((0.5 * (A_bnd + A_bnd.T)).todense())
    raise ValueError(name)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    box = load_parameter_box(CONFIG_PATH)
    mesh = Mesh(N=N_MESH, p=P, x_min=X_MIN, x_max=X_MAX)
    ops = OperatorCache(mesh, X_MIN, X_MAX)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = sample_sobol(box, N_QUERIES, seed=SEED)

    print(f"Task 9: preconditioner benchmark, {N_QUERIES} queries at N={N_MESH} (n_dof={ops.n_dof})\n")

    rows = []
    rng = np.random.default_rng(0)
    n_checked = 0
    for qi, (mu, payoff) in enumerate(samples):
        const = compute_stability_constants(mu, ops)
        if not const.is_spd:
            print(f"  query {qi}: mu={mu} not SPD, skipping (see notes/04_spd_study.md)")
            continue
        n_checked += 1

        S = dense_from_apply(lambda v: ops.apply_S(mu, v), ops.n_dof)
        # One residual per (synthetic) time step, matching how eta_guar
        # actually calls solve_riesz_pcg once per row of compute_all_residuals'
        # output -- see module docstring's methodology note.
        residual_rows = [rng.standard_normal(ops.n_dof) for _ in range(N_STEPS_PER_QUERY)]

        print(f"  query {qi}: mu=(C={mu.C:.3f},G={mu.G:.3f},M={mu.M:.3f},Y={mu.Y:.3f},r={mu.r:.3f},T={mu.T:.3f})")
        for precond in PRECONDITIONERS:
            # Fresh OperatorCache per preconditioner so 'banded'/'jacobi'
            # caches don't leak across preconditioners within this query.
            ops_fresh = OperatorCache(mesh, X_MIN, X_MAX)
            n_iters, t_calls, gaps = [], [], []
            for r_vec in residual_rows:
                t0 = time.perf_counter()
                bracket = solve_riesz_pcg(
                    r_vec, mu, ops_fresh, const.alpha_certified,
                    max_iter=MAX_PCG_ITERATIONS, preconditioner=precond, pcg_tol=PCG_TOL,
                )
                t_calls.append(time.perf_counter() - t0)
                n_iters.append(bracket.n_pcg_iterations)
                gaps.append(
                    (bracket.upper_bound - bracket.lower_bound) / bracket.upper_bound
                    if bracket.upper_bound > 0 else float("nan")
                )

            M_dense = _dense_preconditioner_matrix(precond, ops, mu, S)
            eigs = sla.eigh(S, M_dense, eigvals_only=True)
            eigs = eigs[eigs > 0]  # guard: banded/jacobi M could be indefinite in principle
            kappa = float(eigs[-1] / eigs[0]) if len(eigs) > 0 and eigs[0] > 0 else float("nan")

            t_first_ms = 1000 * t_calls[0]
            t_amortized_ms = 1000 * float(np.mean(t_calls))
            mean_iter = float(np.mean(n_iters))
            mean_gap = float(np.nanmean(gaps))
            rows.append({
                "query": qi, "C": mu.C, "G": mu.G, "M": mu.M, "Y": mu.Y, "r": mu.r, "T": mu.T,
                "preconditioner": precond, "mean_n_pcg_iterations": mean_iter,
                "t_pcg_first_call_ms": t_first_ms, "t_pcg_amortized_mean_ms": t_amortized_ms,
                "kappa_estimated": kappa, "mean_bracket_gap_rel": mean_gap,
            })
            print(f"    {precond:8s}  mean_n_iter={mean_iter:6.1f}  "
                  f"first_call={t_first_ms:9.3f}ms  amortized={t_amortized_ms:7.4f}ms  "
                  f"kappa~{kappa:10.3e}  mean_gap_rel={mean_gap:.2e}")

    csv_path = RESULTS_DIR / "preconditioner_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}  ({n_checked}/{N_QUERIES} queries were SPD)")

    # ── Aggregate ────────────────────────────────────────────────────────
    print("\n=== aggregate (mean over SPD queries; T_PCG is AMORTIZED per-call, see module docstring) ===")
    print(f"{'preconditioner':12s} {'mean n_iter':>12s} {'first call (ms)':>16s} {'amortized (ms)':>16s} {'mean kappa':>14s} {'mean gap_rel':>14s}")
    agg = {}
    for precond in PRECONDITIONERS:
        sub = [r for r in rows if r["preconditioner"] == precond]
        mean_iter = float(np.mean([r["mean_n_pcg_iterations"] for r in sub]))
        mean_t_first = float(np.mean([r["t_pcg_first_call_ms"] for r in sub]))
        mean_t_amortized = float(np.mean([r["t_pcg_amortized_mean_ms"] for r in sub]))
        mean_kappa = float(np.nanmean([r["kappa_estimated"] for r in sub]))
        mean_gap = float(np.nanmean([r["mean_bracket_gap_rel"] for r in sub]))
        agg[precond] = {
            "n_iter": mean_iter, "t_first_ms": mean_t_first, "t_amortized_ms": mean_t_amortized,
            "kappa": mean_kappa, "gap_rel": mean_gap,
        }
        print(f"{precond:12s} {mean_iter:12.1f} {mean_t_first:16.3f} {mean_t_amortized:16.4f} {mean_kappa:14.3e} {mean_gap:14.3e}")

    best = min(agg, key=lambda k: agg[k]["n_iter"])
    print(f"\nbest by mean PCG iteration count: {best}")
    best_t = min(agg, key=lambda k: agg[k]["t_amortized_ms"])
    print(f"best by AMORTIZED wall-clock (the production-relevant number): {best_t}")

    # ── Figure ───────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    names = PRECONDITIONERS
    ax1.bar(names, [agg[n]["n_iter"] for n in names], color="C0", alpha=0.75)
    ax1.set_ylabel("mean PCG iterations")
    ax1.set_title("Task 9: iterations by preconditioner")

    ax2.bar(names, [agg[n]["kappa"] for n in names], color="C1", alpha=0.75)
    ax2.set_yscale("log")
    ax2.set_ylabel("mean estimated kappa(M^-1 S)")
    ax2.set_title("Task 9: preconditioned condition number")

    fig.tight_layout()
    fig_path = FIGURES_DIR / "preconditioner_benchmark.png"
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
