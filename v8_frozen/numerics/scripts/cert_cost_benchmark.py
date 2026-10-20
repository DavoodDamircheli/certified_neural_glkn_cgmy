"""Task 8 (working prompt): direct (eta_sharp) vs PCG (eta_guar) certification
cost scaling, N = 32, 64, 128, 256, 512 -- the crossover resolution at which
iterative/matrix-free certification wins.

Scope note, worth stating up front: alpha_certified (Task 3's
Weyl-residual-enclosure lower bound, certification.constants) requires the
SAME dense assembly + eigh that eta_sharp's direct solve needs, and BOTH
eta_sharp and eta_guar require alpha_certified before either can run at all
-- so "dense assembly + eigendecomposition" is not itself part of the
direct-vs-PCG choice this task asks about; it is a shared, mandatory cost
either path pays today. What varies between eta_sharp and eta_guar, GIVEN
alpha_certified already computed, is exactly: a Cholesky factor + solve
against the (already-assembled) dense S_h(mu), vs. a matrix-free PCG solve
that never needs the dense matrix at all. This script times all three
pieces (assembly+eigh, direct factor+solve, PCG) separately, so the
crossover reported is unambiguous about which comparison it is.

compute_stability_constants' production max_n_dof=400 guard is
deliberately overridden here (BENCHMARK_MAX_N_DOF) to actually measure cost
up to N=512 (n_dof=514) -- the guard itself already encodes an engineering
judgment about where dense assembly stops being "practical"; this script
exists to check that judgment empirically rather than assume it, at
exactly the resolutions the working prompt asks for. The production
default (400) is NOT changed anywhere else.

FINDING worth stating up front (see notes/08_certification_cost.md for the
full account): profiling the "shared/mandatory" cost above shows it is NOT
dominated by eigh (the eigendecomposition Task 3's Weyl bound needs) -- it
is dominated by np.linalg.eigh(S)'s PREREQUISITE, densely MATERIALIZING
S_h(mu) at all (dense_from_apply, n_dof individual matrix-free apply_S
calls). At N=512 this is ~190ms vs. eigh's own ~17ms and the Weyl bound
matmul's ~2ms -- assembly alone is ~10x eigh's cost. Since alpha_certified
(hence is_spd) is required before EITHER eta_sharp or eta_guar can be used
at all, and alpha_certified requires this same dense assembly, PCG's
matrix-free advantage (the entire reason to prefer it over a direct dense
solve) is largely NOT realized in this project's current architecture: the
expensive step is already paid for, mandatorily, before the direct-vs-PCG
choice is even made.

FIX (fix/efficiency-norm-kappa, Step 6): the pre-submission-fixes pass
(item 1.1, below) fixed what this script CALLED the shared setup cost, but
left the actual crossover comparison unfair to direct: `pcg_wins_apples_to_apples`
compared raw `t_pcg_ms` (PCG's solve alone) against `direct_total_ms`
(setup+factor+solve) -- but PCG also needs `alpha_certified` from that same
stability-constant setup (`eta_guar`/`solve_riesz_pcg` takes
`coercivity_constant` as a required argument), so PCG pays this cost too,
and the old flag silently gave PCG a free pass on it. This has been
replaced by `pcg_wins_total_cost` = (t_setup+t_pcg) < (t_setup+t_direct),
matching the actual comparison "T_setup+T_PCG vs T_setup+T_direct". Since
t_setup is identical on both sides, this is mathematically the same
ordering as `pcg_wins_incremental` (factor+solve alone) -- both are now
reported and, unlike before, are internally consistent with each other
rather than disagreeing because one silently dropped a shared term.

RELABELING (pre-submission fixes, item 1.1): the timed region this script
calls "assemble+eigh" is actually one call to
`certification.constants.compute_stability_constants`, and a full read of
that function (not just its docstring) shows it computes MORE than
assembly+eigh of S alone -- it ALSO densely assembles M_h
(`dense_from_apply(ops.apply_M, n)`), computes `eigvalsh(M)`, AND solves a
SEPARATE generalized eigenproblem `scipy.linalg.eigh(S, M)` for the
diagnostic-only mass-preconditioned condition number / PCG
convergence-rate fields -- none of which the phrase "assemble+eigh" would
lead a reader to expect. The field is renamed here to
`t_stability_constant_setup_ms` ("stability-constant setup", not
"certificate setup cost"): this timed region produces every field of the
`StabilityConstants` dataclass (kappa, kappa_precond, rho, lambda_min/max
of both S and M, alpha_certified, safety_ratio) -- it IS the stability-
constant computation, literally, by name. It does NOT compute a
certificate (eta_sharp/eta_guar are separate, later calls, timed
separately below as `t_direct_*`/`t_pcg_ms`) -- "certificate setup cost"
would misleadingly suggest this region is part of producing eta_sharp/
eta_guar themselves, when it is a shared PREREQUISITE both certificates
consume via `alpha_certified`, not part of either certificate's own
computation. New `t_M_assemble_ms`/`t_eigvalsh_M_ms`/
`t_generalized_eigh_ms` fields sub-break-down this previously-undocumented
M-side cost, mirroring the existing S-side breakdown.

Memory is reported as an analytic proxy (floats: n_dof^2 for the dense
path, n_dof*n_pcg_iterations for PCG), matching exp03_certificate_levels.py's
own established convention in this project -- not a measured RSS delta
(noisy at these problem sizes, dominated by Python/BLAS overhead, not the
certificate computation itself).

Outputs: results/cert_cost_benchmark.csv, figures/cert_cost_benchmark.png.
Usage: PYTHONPATH=src python3 scripts/cert_cost_benchmark.py
"""

from __future__ import annotations

import csv
import datetime
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, Mesh
from certification.constants import compute_stability_constants, dense_from_apply, weyl_residual_enclosure
from certification.riesz_solver import solve_riesz_pcg

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

RESOLUTIONS = [32, 64, 128, 256, 512]
P = 3
X_MIN, X_MAX = float(np.log(100) - 8), float(np.log(100) + 8)  # production domain, fixed across N
# Same well-conditioned interior mu used throughout this project's own test
# suite (tests/test_pipeline_predict_certify_correct.py,
# tests/test_certification_assumptions.py, etc.) -- SPD-ness across N is a
# separate concern (Task 4); this benchmark is about cost, not robustness,
# so a known-safe point avoids conflating the two.
MU = CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.4)
PRECONDITIONER = "none"  # pre-submission fixes, item 1.1: matches the FINAL production default
MAX_PCG_ITERATIONS = 200  # matches configs/base.yaml's production default
PCG_TOL = 1e-14  # matches riesz_solver.solve_riesz_pcg's default
BENCHMARK_MAX_N_DOF = 600  # override ONLY in this script -- see module docstring
N_REPEATS = 9  # wall-clock repeats per measurement, report the median (exp03's convention)


def _assert_preconditioner_matches_production() -> None:
    """Fail loudly if PRECONDITIONER drifts from configs/base.yaml's
    actual production default again -- this exact drift (script said
    'mass', production config had already moved to 'none') is what made
    this benchmark stale enough to need item 1.1 in the first place."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    production_default = cfg["certification"]["preconditioner"]
    if PRECONDITIONER != production_default:
        raise AssertionError(
            f"cert_cost_benchmark.py's PRECONDITIONER={PRECONDITIONER!r} no longer matches "
            f"configs/base.yaml's certification.preconditioner={production_default!r} -- "
            f"update the PRECONDITIONER constant above (and rerun this benchmark) before "
            f"trusting these numbers. This assertion exists specifically so this staleness "
            f"can't recur silently (pre-submission fixes, item 1.1)."
        )


CSV_FIELDS = [
    "N", "n_dof",
    "t_stability_constant_setup_ms",  # shared/mandatory: full compute_stability_constants call
                                       # (dense S assembly + eigh(S) + Weyl bound + dense M
                                       # assembly + eigvalsh(M) + generalized eigh(S,M) -- see
                                       # module docstring's RELABELING note for why this is NOT
                                       # just "assemble+eigh")
    "t_dense_assemble_S_ms",  # sub-breakdown: JUST materializing S densely (n_dof apply_S calls)
    "t_eigh_ms",  # sub-breakdown: JUST np.linalg.eigh(S)
    "t_weyl_bound_ms",  # sub-breakdown: JUST the Weyl residual-enclosure matmul
    "t_M_assemble_ms",  # sub-breakdown: JUST materializing M_h densely
    "t_eigvalsh_M_ms",  # sub-breakdown: JUST np.linalg.eigvalsh(M)
    "t_generalized_eigh_ms",  # sub-breakdown: JUST scipy.linalg.eigh(S, M) (mass-preconditioned kappa)
    "alpha_certified", "safety_ratio",
    "t_direct_factor_ms", "t_direct_solve_ms",  # eta_sharp-specific, GIVEN the dense S above
    "eta_sharp",
    "t_pcg_ms", "n_pcg_iterations", "eta_guar",
    "bracket_gap_abs", "bracket_gap_rel",
    "direct_memory_floats", "pcg_memory_floats",
    "direct_total_ms",  # t_stability_constant_setup_ms + t_direct_factor_ms + t_direct_solve_ms
    "pcg_total_ms",  # t_stability_constant_setup_ms + t_pcg_ms -- PCG ALSO needs alpha_certified
                      # from the same shared setup (eta_guar/solve_riesz_pcg requires
                      # coercivity_constant), so this is the correct total PCG cost, not t_pcg_ms alone
    "pcg_wins_total_cost",  # (Step 6, working prompt) pcg_total_ms < direct_total_ms, i.e.
                             # T_setup+T_PCG vs T_setup+T_direct -- the actual like-for-like
                             # comparison. REPLACES the old, unfair-to-direct "pcg_wins_apples_to_apples"
                             # (which compared t_pcg_ms alone, silently dropping PCG's own setup cost).
    "pcg_wins_incremental",  # pcg_wins vs JUST factor+solve, t_setup already paid either way --
                              # mathematically the SAME ordering as pcg_wins_total_cost since t_setup
                              # cancels from both sides; reported separately for transparency.
]


def _median_timed(fn, repeats: int = N_REPEATS):
    """Run fn() `repeats` times, return (median_seconds, last_return_value)."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def benchmark_one_resolution(N: int) -> dict:
    mesh = Mesh(N=N, p=P, x_min=X_MIN, x_max=X_MAX)
    ops = OperatorCache(mesh, X_MIN, X_MAX)
    n_dof = ops.n_dof
    rng = np.random.default_rng(0)
    r = rng.standard_normal(n_dof)  # a representative residual vector; PCG/Cholesky cost
                                     # depends on S_h(mu) and n_dof, not on r's specific values

    # ── Shared/mandatory: the full stability-constant setup -> alpha_certified ──
    def _setup_stability_constants():
        return compute_stability_constants(MU, ops, max_n_dof=BENCHMARK_MAX_N_DOF)

    t_stability_setup, const = _median_timed(_setup_stability_constants, repeats=7)
    assert const.is_spd, f"N={N}: MU is not SPD here -- pick a different benchmark mu (see notes/04_spd_study.md)"

    # ── Fine-grained sub-breakdown of the "shared/mandatory" cost above --
    # every individual piece compute_stability_constants actually does (see
    # module docstring's RELABELING note: this is more than "assemble+eigh") ──
    t_dense_assemble_S, S = _median_timed(lambda: dense_from_apply(lambda v: ops.apply_S(MU, v), n_dof), repeats=7)
    t_eigh, (eigs, V) = _median_timed(lambda: np.linalg.eigh(S), repeats=7)
    t_weyl_bound, _ = _median_timed(lambda: weyl_residual_enclosure(S, eigs, V), repeats=7)
    t_M_assemble, M = _median_timed(lambda: dense_from_apply(ops.apply_M, n_dof), repeats=7)
    t_eigvalsh_M, _ = _median_timed(lambda: np.linalg.eigvalsh(M), repeats=7)
    t_generalized_eigh, _ = _median_timed(lambda: sla.eigh(S, M, eigvals_only=True), repeats=7)

    # ── Direct (eta_sharp): factor + solve, reusing the freshly-assembled S ─

    def _factor():
        return sla.cho_factor(S)

    t_factor, cho = _median_timed(_factor, repeats=N_REPEATS)

    def _solve():
        return sla.cho_solve(cho, r)

    t_solve, x_direct = _median_timed(_solve, repeats=N_REPEATS)
    eta_sharp_val = float(np.sqrt(max(r @ x_direct, 0.0)))

    # ── PCG (eta_guar): matrix-free, needs only alpha_certified as a scalar ─
    def _pcg():
        return solve_riesz_pcg(
            r, MU, ops, const.alpha_certified,
            max_iter=MAX_PCG_ITERATIONS, preconditioner=PRECONDITIONER, pcg_tol=PCG_TOL,
        )

    t_pcg, bracket = _median_timed(_pcg, repeats=N_REPEATS)
    eta_guar_val = float(np.sqrt(max(bracket.upper_bound, 0.0)))

    gap_abs = bracket.upper_bound - bracket.lower_bound
    gap_rel = gap_abs / bracket.upper_bound if bracket.upper_bound > 0 else float("nan")

    direct_total_ms = 1000 * (t_stability_setup + t_factor + t_solve)
    pcg_ms = 1000 * t_pcg
    pcg_total_ms = 1000 * t_stability_setup + pcg_ms

    return {
        "N": N, "n_dof": n_dof,
        "t_stability_constant_setup_ms": 1000 * t_stability_setup,
        "t_dense_assemble_S_ms": 1000 * t_dense_assemble_S,
        "t_eigh_ms": 1000 * t_eigh,
        "t_weyl_bound_ms": 1000 * t_weyl_bound,
        "t_M_assemble_ms": 1000 * t_M_assemble,
        "t_eigvalsh_M_ms": 1000 * t_eigvalsh_M,
        "t_generalized_eigh_ms": 1000 * t_generalized_eigh,
        "alpha_certified": const.alpha_certified, "safety_ratio": const.safety_ratio,
        "t_direct_factor_ms": 1000 * t_factor, "t_direct_solve_ms": 1000 * t_solve,
        "eta_sharp": eta_sharp_val,
        "t_pcg_ms": pcg_ms, "n_pcg_iterations": bracket.n_pcg_iterations, "eta_guar": eta_guar_val,
        "bracket_gap_abs": gap_abs, "bracket_gap_rel": gap_rel,
        "direct_memory_floats": n_dof * n_dof,
        "pcg_memory_floats": n_dof * max(bracket.n_pcg_iterations, 1),
        "direct_total_ms": direct_total_ms,
        "pcg_total_ms": pcg_total_ms,
        "pcg_wins_total_cost": pcg_total_ms < direct_total_ms,
        "pcg_wins_incremental": pcg_ms < 1000 * (t_factor + t_solve),
    }


def _supersede_if_exists(*paths: Path) -> list[Path]:
    """Move each existing path to results/_superseded/<date>/<same-
    relative-name> before it's overwritten (Ground Rule 2) -- never a
    silent overwrite."""
    date_dir = REPO_ROOT / "results" / "_superseded" / datetime.date.today().isoformat()
    moved = []
    for p in paths:
        if not p.exists():
            continue
        rel = p.relative_to(REPO_ROOT)
        dest = date_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
        moved.append(dest)
    return moved


def _git_commit_hash() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _write_provenance(csv_path: Path, start_time: float, git_commit: str) -> Path:
    """<name>.config.json sidecar (Ground Rule 3): git commit,
    preconditioner, solver tolerances, mesh resolutions, seed, sample
    counts, hardware, library versions, wall-clock start/end."""
    import os
    import platform

    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    blas_name = "unknown"
    try:
        blas_name = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
    except Exception:  # noqa: BLE001
        pass

    payload = {
        "git_commit": git_commit,
        "preconditioner": PRECONDITIONER,
        "solver_tolerances": {
            "pcg_rtol": PCG_TOL, "pcg_atol": 0.0, "max_pcg_iterations": MAX_PCG_ITERATIONS,
            "eigensolver": "numpy.linalg.eigh / scipy.linalg.eigh (dense, exact -- no tolerance parameter)",
            "cholesky_factor_solve": "scipy.linalg.cho_factor/cho_solve (exact direct solve)",
        },
        "mesh_resolutions": RESOLUTIONS,
        "benchmark_mu": {"C": MU.C, "G": MU.G, "M": MU.M, "Y": MU.Y, "r": MU.r, "T": MU.T},
        "n_repeats_per_measurement": N_REPEATS,
        "hardware": {
            "cpu_model": cpu_model, "cpu_count_logical": os.cpu_count(),
            "thread_env_vars": {k: os.environ.get(k) for k in
                                 ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "blas_backend": blas_name, "python": platform.python_version(),
        },
        "library_versions": {"numpy": np.__version__, "scipy": __import__("scipy").__version__},
        "wall_clock_start": start_time, "wall_clock_end": time.time(),
        "wall_clock_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_time)),
        "wall_clock_end_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = csv_path.with_suffix(".config.json")
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def main() -> None:
    t_start = time.time()
    git_commit = _git_commit_hash()  # captured BEFORE any output is written (Step 8, working
                                      # prompt) -- see exp_common.capture_git_commit_hash's docstring
                                      # for why checking git status AFTER writing tracked output
                                      # is self-referentially "-dirty" even right after a clean commit
    _assert_preconditioner_matches_production()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Task 8 / pre-submission fixes item 1.1: direct (eta_sharp) vs PCG (eta_guar) "
          f"certification cost, N={RESOLUTIONS}")
    print(f"mu={MU}, domain=[{X_MIN:.4f},{X_MAX:.4f}], preconditioner={PRECONDITIONER}, "
          f"max_pcg_iterations={MAX_PCG_ITERATIONS}\n")

    rows = []
    for N in RESOLUTIONS:
        print(f"N={N:4d} ...", end=" ", flush=True)
        row = benchmark_one_resolution(N)
        rows.append(row)
        print(
            f"n_dof={row['n_dof']:4d}  stability_setup={row['t_stability_constant_setup_ms']:8.2f}ms "
            f"(dense_assemble_S={row['t_dense_assemble_S_ms']:7.2f}ms eigh_S={row['t_eigh_ms']:6.2f}ms "
            f"weyl={row['t_weyl_bound_ms']:5.2f}ms dense_assemble_M={row['t_M_assemble_ms']:6.2f}ms "
            f"eigvalsh_M={row['t_eigvalsh_M_ms']:6.2f}ms generalized_eigh={row['t_generalized_eigh_ms']:7.2f}ms)  "
            f"direct(factor+solve)={row['t_direct_factor_ms'] + row['t_direct_solve_ms']:6.2f}ms  "
            f"pcg={row['t_pcg_ms']:8.2f}ms ({row['n_pcg_iterations']:3d} iters)  "
            f"gap_rel={row['bracket_gap_rel']:.2e}  "
            f"pcg_wins(T_setup+T_pcg vs T_setup+T_direct)={row['pcg_wins_total_cost']}  "
            f"pcg_wins(factor+solve only)={row['pcg_wins_incremental']}"
        )

    csv_path = RESULTS_DIR / "cert_cost_benchmark.csv"
    fig_path = FIGURES_DIR / "cert_cost_benchmark.png"
    moved = _supersede_if_exists(csv_path, fig_path, csv_path.with_suffix(".config.json"))
    for m in moved:
        print(f"superseded existing output -> {m}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    # ── Crossover verdict ────────────────────────────────────────────────
    crossover_total_cost = next((r["N"] for r in rows if r["pcg_wins_total_cost"]), None)
    crossover_incremental = next((r["N"] for r in rows if r["pcg_wins_incremental"]), None)
    print("\n=== crossover resolution ===")
    print(f"PCG beats direct on T_setup+T_pcg vs T_setup+T_direct (Step 6, working prompt), starting at N="
          f"{crossover_total_cost if crossover_total_cost else 'never (within tested range)'}")
    print(f"PCG beats direct factor+solve ALONE (stability-constant setup already paid either way), starting at N="
          f"{crossover_incremental if crossover_incremental else 'never (within tested range)'}")
    if crossover_total_cost != crossover_incremental:
        print(f"NOTE: these two crossovers disagree ({crossover_total_cost} vs {crossover_incremental}) -- "
              f"since t_setup is identical on both sides of pcg_wins_total_cost, they should be "
              f"mathematically equivalent; a disagreement here would indicate a bug, not just noise.")

    # ── Figure: cost vs N, log-log ───────────────────────────────────────
    Ns = [r["N"] for r in rows]
    direct_total = [r["direct_total_ms"] for r in rows]
    direct_incremental = [r["t_direct_factor_ms"] + r["t_direct_solve_ms"] for r in rows]
    pcg_total = [r["pcg_total_ms"] for r in rows]
    pcg_incremental = [r["t_pcg_ms"] for r in rows]
    setup_cost = [r["t_stability_constant_setup_ms"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(Ns, direct_total, "o-", label="direct total (T_setup+T_factor+T_solve)")
    ax1.plot(Ns, direct_incremental, "s--", label="direct factor+solve only", alpha=0.6)
    ax1.plot(Ns, pcg_total, "^-", label=f"PCG total (T_setup+T_pcg, max_iter={MAX_PCG_ITERATIONS})")
    ax1.plot(Ns, pcg_incremental, "v--", label="PCG solve only", alpha=0.6)
    ax1.plot(Ns, setup_cost, "d:", label="shared T_setup alone", alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("N (spatial elements)")
    ax1.set_ylabel(f"wall-clock (ms, median of {N_REPEATS} repeats)")
    ax1.set_title("Task 8 / Step 6: certification cost vs. resolution")
    ax1.legend(fontsize=7)

    gaps = [r["bracket_gap_rel"] for r in rows]
    ax2.plot(Ns, gaps, "o-", color="C3")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("N (spatial elements)")
    ax2.set_ylabel("PCG bracket relative gap (upper-lower)/upper")
    ax2.set_title(f"Riesz bracket gap at max_iter={MAX_PCG_ITERATIONS}")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")

    prov_path = _write_provenance(csv_path, t_start, git_commit)
    print(f"wrote {prov_path}")


if __name__ == "__main__":
    main()
