"""exp15 -- E6 (Final experiment matrix, working prompt) / pre-submission
fixes item 1.2: runtime scaling at N = 32...512, broken down into
Galerkin / inference / certification / correction / total.

Fills a genuine gap the final matrix surfaced: `scripts/
cert_cost_benchmark.py` (Task 8) already sweeps N=32...512 but only for
the CERTIFICATION piece (direct vs. PCG); `experiments/exp08_runtime.py`
already reports the full phase breakdown (predict/residual/certify/
correct/recertify) but only at ONE fixed N (=64, the production mesh).
Neither alone answers "how does the FULL workflow's cost scale with N."
This script does, reusing both pieces' own validated methodology rather
than inventing new timing code:

  - Galerkin: `cgmy.reference_solver.solve_cgmy`, timed directly (same as
    exp08).
  - Inference: ProjectedFNO ONLY. Task 13's own finding applies again
    here without re-deriving it: MLP/DeepONet/POD are ARCHITECTURALLY
    INCOMPATIBLE with any N other than their training resolution (64),
    so they structurally cannot appear in an N=32...512 sweep at all --
    not a new limitation, the same one exp10_cross_resolution.py already
    established and this script inherits rather than re-litigates.
  - Certification: `certification.constants.compute_stability_constants`
    + `certification.guaranteed_certificate.eta_guar` (quick, capped PCG),
    called directly (not through `pipeline.predict_certify_correct`,
    which hardcodes `compute_stability_constants`'s default
    `max_n_dof=400` -- too small to reach N=512's n_dof=514). Reuses
    Task 8's own precedent of overriding `max_n_dof` LOCALLY in this one
    diagnostic script only, never touching the production default
    anywhere else.
  - Correction: `correction.krylov.correct_trajectory`, warm-started by
    the ProjectedFNO prediction, same GMRES convention
    `predict_certify_correct` itself uses.
  - Total: inference + certification + correction. **This is the FULL /
    FORCED-CORRECTION PATH** (T_NN + T_cert + T_correct) -- `tolerance=
    1e-9` is used throughout, tight enough that every query needs
    correction, so this number is NOT the average cost of the real,
    tolerance-dependent predict-certify-correct workflow (which often
    skips correction entirely -- see `experiments/
    exp11_tolerance_sweep.py`, whose own numbers show direct acceptance
    ranging from ~0% to ~88% depending on tolerance). This script's
    "total" is what a query costs in the WORST case (correction always
    required); `exp11_tolerance_sweep.py` remains the primary end-to-end
    workflow timing experiment, since it reports the ACTUAL,
    tolerance-dependent mix of accepted-directly vs. corrected queries.
    This is stated here, in the printed output, and in the LaTeX table's
    own caption row, not just in this docstring.

CHANGED (pre-submission fixes, item 1.2): `N_QUERIES` was 5, with no
attempt to use the SAME queries across every resolution -- at N=32
(binding constraint: SPD loss is more common at coarser meshes, see
`notes/04_spd_study.md`), 3/5 of even that small sample were non-SPD,
leaving almost no usable timing data there. Now: a COMMON pool of
parameter samples is oversampled (candidates drawn from the same Sobol
distribution as before) until at least `TARGET_N_COMMON` (30) are found
SPD/certifiable at EVERY one of the 5 resolutions (minimum acceptable:
`MIN_N_COMMON` = 20); if that floor can't be reached within a bounded
candidate budget, this script reports exactly that and falls back to
independent per-resolution valid sample sets instead (never silently
substitutes a smaller N or claims cross-resolution comparability it
doesn't have). Every resolution now also reports median, Q1/Q3 (IQR),
mean +/- std, n_valid, and n_non_SPD -- not just a median.

Tolerance note: `tolerance=1e-9` is used throughout (tight enough that
every query needs correction) so T_correction is always actually
measured at every N, rather than being skipped whenever the quick
certificate happens to pass -- the same "force every phase to run"
logic exp08's own calibration pass already established, applied here to
get a complete row at every resolution rather than a mix of complete and
partial rows.

Outputs: results/exp15_runtime_scaling_matrix.csv,
figures/exp15_runtime_scaling_matrix.png,
tables/exp15_runtime_scaling_matrix.tex,
results/exp15_runtime_scaling_matrix.config.json.
"""

from __future__ import annotations

import csv
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cgmy.operators import OperatorCache
from cgmy.reference_solver import Mesh, TimeGrid, solve_cgmy
from certification.constants import compute_stability_constants
from certification.guaranteed_certificate import eta_guar as compute_eta_guar
from certification.residuals import compute_all_residuals
from correction.krylov import compute_initial_coefficients, correct_trajectory
from data import payoff_family
from data.sampling import sample_sobol
from exp_common import (
    FIGURES_DIR,
    MAX_PCG_ITERATIONS,
    PRECONDITIONER,
    RESULTS_DIR,
    TABLES_DIR,
    fmt,
    get_box_and_stats,
    load_models,
    capture_git_commit_hash,
    provenance_start,
    supersede_if_exists,
    write_latex_table,
    write_provenance,
)
from learning.normalization import normalize_mu
from learning.projected_fno import ProjectedFNO
from pipeline.predict_certify_correct import build_mesh_info, build_skeleton_trajectory

X_MIN = float(np.log(100) - 8)
X_MAX = float(np.log(100) + 8)
MESH_P = 3
RESOLUTIONS = [32, 64, 128, 256, 512]  # ascending -- SPD pre-check short-circuits on the cheapest (N=32) first
TARGET_N_COMMON = 30
MIN_N_COMMON = 20
CANDIDATE_BUDGET = 400  # hard cap on candidates drawn during the common-sample search
SEED = 160160
CANDIDATE_SEED = 160161  # disjoint from SEED (used for the warm-up draw)
TOLERANCE = 1e-9  # tight: forces correction to actually run at every N (see module docstring)
QUICK_PCG_ITERATIONS = 5
MAX_N_DOF_OVERRIDE = 600  # Task 8's own precedent: local override only, production default (400) untouched elsewhere


def _mesh_at(N: int) -> Mesh:
    return Mesh(N=N, p=MESH_P, x_min=X_MIN, x_max=X_MAX)


def _is_spd_at(mu, ops: OperatorCache) -> bool:
    return compute_stability_constants(mu, ops, max_n_dof=MAX_N_DOF_OVERRIDE).is_spd


def find_common_valid_samples(box, ops_by_N: dict[int, OperatorCache]) -> tuple[list, dict]:
    """Oversample until >= TARGET_N_COMMON parameter samples are found
    SPD/certifiable at EVERY resolution in ops_by_N, or CANDIDATE_BUDGET
    candidates have been drawn. RESOLUTIONS is checked ascending (cheapest
    first) with short-circuit rejection -- most candidates fail at the
    cheap N=32 check, so this stays fast even though the N=512 check
    alone costs ~300ms/candidate.

    Returns (common_samples, stats) where stats records n_drawn,
    n_kept, and per-resolution rejection counts (how many drawn
    candidates failed SPD specifically at that resolution, among those
    that reached it -- diagnostic, not just a final tally).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = sample_sobol(box, CANDIDATE_BUDGET, seed=CANDIDATE_SEED)

    common: list = []
    n_drawn = 0
    n_failed_at: dict[int, int] = {N: 0 for N in RESOLUTIONS}
    for mu, payoff in candidates:
        n_drawn += 1
        ok = True
        for N in RESOLUTIONS:
            if not _is_spd_at(mu, ops_by_N[N]):
                n_failed_at[N] += 1
                ok = False
                break  # short-circuit: don't pay for the more expensive higher-N checks
        if ok:
            common.append((mu, payoff))
        if len(common) >= TARGET_N_COMMON:
            break

    stats = {
        "n_drawn": n_drawn, "n_kept": len(common),
        "n_failed_at_resolution": n_failed_at,
        "reached_target": len(common) >= TARGET_N_COMMON,
        "reached_minimum": len(common) >= MIN_N_COMMON,
    }
    return common, stats


def find_per_resolution_valid_samples(box, ops_by_N: dict[int, OperatorCache], n_target: int) -> dict[int, list]:
    """Fallback (item 1.2's own instruction) if a common cross-resolution
    pool can't reach MIN_N_COMMON: independently valid sample sets per
    resolution, no cross-resolution commonality claimed."""
    result = {}
    for N in RESOLUTIONS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidates = sample_sobol(box, CANDIDATE_BUDGET, seed=CANDIDATE_SEED + N)
        kept = [(mu, payoff) for mu, payoff in candidates if _is_spd_at(mu, ops_by_N[N])][:n_target]
        result[N] = kept
    return result


def _iqr_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan"),
                "mean": float("nan"), "std": float("nan")}
    arr = np.array(values)
    return {
        "median": float(np.median(arr)), "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)), "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def main() -> None:
    t_start = provenance_start()
    git_commit = capture_git_commit_hash()  # BEFORE any output is written -- see its docstring
    box, stats = get_box_and_stats()
    ops64 = OperatorCache(_mesh_at(64), X_MIN, X_MAX)
    base_models = load_models(ops64, stats)
    fno_base = base_models["ProjectedFNO"]

    ops_by_N = {N: OperatorCache(_mesh_at(N), X_MIN, X_MAX) for N in RESOLUTIONS}

    print(f"exp15: searching for >= {TARGET_N_COMMON} common valid samples across N={RESOLUTIONS} "
          f"(budget={CANDIDATE_BUDGET} candidates)...")
    common_samples, search_stats = find_common_valid_samples(box, ops_by_N)
    print(f"  drawn={search_stats['n_drawn']}  kept={search_stats['n_kept']}  "
          f"failed_at_resolution={search_stats['n_failed_at_resolution']}")

    used_fallback = False
    if search_stats["reached_target"]:
        print(f"  reached target: {len(common_samples)} common valid samples.")
        samples_by_N = {N: common_samples for N in RESOLUTIONS}
    elif search_stats["reached_minimum"]:
        print(f"  did not reach target ({TARGET_N_COMMON}) but reached the minimum "
              f"({MIN_N_COMMON}): using {len(common_samples)} common valid samples.")
        samples_by_N = {N: common_samples for N in RESOLUTIONS}
    else:
        used_fallback = True
        print(f"  FAILED to reach the minimum ({MIN_N_COMMON}) common valid samples within "
              f"the {CANDIDATE_BUDGET}-candidate budget ({len(common_samples)} found). "
              f"Falling back to independent per-resolution valid samples (item 1.2's own "
              f"instruction) -- cross-resolution comparability is NOT claimed for this run.")
        samples_by_N = find_per_resolution_valid_samples(box, ops_by_N, TARGET_N_COMMON)
        for N in RESOLUTIONS:
            print(f"    N={N}: {len(samples_by_N[N])} valid samples found")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warmup_samples = sample_sobol(box, 1, seed=SEED + 999)

    rows = []
    for N in RESOLUTIONS:
        mesh_N = _mesh_at(N)
        ops_N = ops_by_N[N]
        mesh_info_N = build_mesh_info(mesh_N)
        time_grid_req = TimeGrid(n_steps=60)
        wrapped_fno = ProjectedFNO(fno_base.fno, mesh_N, ops_N, mu_dim=fno_base.mu_dim)
        samples = samples_by_N[N]

        # Warm-up: one full, untimed query at THIS resolution before any
        # timing starts (exp08_runtime.py's own established convention,
        # repeated per-N since ProjectedFNO's Pi_h wrapper rebuilds per
        # mesh -- see notes/final-matrix writeup for the artifact this
        # caught before it was added).
        wu_mu, wu_payoff = warmup_samples[0]
        wu_time_grid = np.linspace(0.0, wu_mu.T, time_grid_req.n_steps + 1)
        wu_c0, _, _ = compute_initial_coefficients(wu_mu, wu_payoff, mesh_info_N, ops_N)
        wu_psi = payoff_family.european_put(wu_payoff.strike, ops_N)
        _ = wrapped_fno.predict_trajectory(normalize_mu(wu_mu, stats), wu_psi, wu_c0, wu_time_grid, wu_mu.T)

        print(f"exp15: N={N} (n_dof={ops_N.n_dof}, n_queries={len(samples)})...")

        t_galerkin_l, t_inference_l, t_cert_l, t_correction_l, t_total_l = [], [], [], [], []
        n_not_spd = 0
        for mu, payoff in samples:
            time_grid = np.linspace(0.0, mu.T, time_grid_req.n_steps + 1)

            t0 = time.perf_counter()
            traj_true = solve_cgmy(mu, payoff, mesh_N, time_grid_req, mode="toeplitz")
            t_galerkin = time.perf_counter() - t0

            t0 = time.perf_counter()
            c0, bc_left, bc_right = compute_initial_coefficients(mu, payoff, mesh_info_N, ops_N)
            mu_tilde = normalize_mu(mu, stats)
            psi = payoff_family.european_put(payoff.strike, ops_N)
            pred = wrapped_fno.predict_trajectory(mu_tilde, psi, c0, time_grid, mu.T)
            t_inference = time.perf_counter() - t0

            t_galerkin_l.append(t_galerkin)
            t_inference_l.append(t_inference)

            t0 = time.perf_counter()
            const = compute_stability_constants(mu, ops_N, max_n_dof=MAX_N_DOF_OVERRIDE)
            t_cert = None
            if const.is_spd:
                traj_pred = build_skeleton_trajectory(mu, payoff, mesh_info_N, time_grid, pred)
                residuals = compute_all_residuals(traj_pred, mu, ops_N)
                dt = traj_pred.solver_metadata.dt
                guar_quick = compute_eta_guar(
                    residuals, mu, ops_N, dt, const.alpha_certified,
                    max_pcg_iterations=QUICK_PCG_ITERATIONS, preconditioner=PRECONDITIONER,
                )
                t_cert = time.perf_counter() - t0
                t_cert_l.append(t_cert)
            else:
                n_not_spd += 1  # should be 0 by construction for the common/fallback-valid pool; recorded anyway

            t0 = time.perf_counter()
            correction = correct_trajectory(
                mu, payoff, mesh_info_N, time_grid, ops_N, seed_coefficients=pred,
                warm_start=True, gmres_rtol=1e-10, gmres_restart=50,
            )
            t_correction = time.perf_counter() - t0
            t_correction_l.append(t_correction)

            if t_cert is not None:
                t_total_l.append(t_inference + t_cert + t_correction)

        galerkin_stats = _iqr_stats(t_galerkin_l)
        inference_stats = _iqr_stats(t_inference_l)
        cert_stats = _iqr_stats(t_cert_l)
        correction_stats = _iqr_stats(t_correction_l)
        total_stats = _iqr_stats(t_total_l)

        row = {"N": N, "n_dof": ops_N.n_dof, "n_valid": len(samples), "n_not_spd": n_not_spd}
        for label, s in (("galerkin", galerkin_stats), ("inference", inference_stats),
                          ("certification", cert_stats), ("correction", correction_stats),
                          ("total", total_stats)):
            for stat_name, val in s.items():
                row[f"t_{label}_{stat_name}_ms"] = 1000 * val if val == val else float("nan")
        rows.append(row)
        print(f"  galerkin_median={row['t_galerkin_median_ms']:.2f}ms  "
              f"inference_median={row['t_inference_median_ms']:.2f}ms  "
              f"certification_median={row['t_certification_median_ms']:.2f}ms  "
              f"correction_median={row['t_correction_median_ms']:.2f}ms  "
              f"total_median(FULL/FORCED-CORRECTION PATH)={row['t_total_median_ms']:.2f}ms  "
              f"[Q1={row['t_total_q1_ms']:.2f}, Q3={row['t_total_q3_ms']:.2f}]  "
              f"mean+/-std={row['t_total_mean_ms']:.2f}+/-{row['t_total_std_ms']:.2f}ms  "
              f"n_valid={row['n_valid']}  not_spd={n_not_spd}")

    print("\n*** IMPORTANT: the 'total' column above is the FULL / FORCED-CORRECTION PATH "
          "(T_NN + T_cert + T_correct, tolerance=1e-9 forces correction on every query). "
          "It is NOT the average cost of the real, tolerance-dependent predict-certify-"
          "correct workflow, which often skips correction entirely -- see "
          "experiments/exp11_tolerance_sweep.py, the primary end-to-end workflow timing "
          "experiment, for the tolerance-dependent numbers. ***\n")

    # ── Never overwrite results silently ────────────────────────────────
    csv_path = RESULTS_DIR / "exp15_runtime_scaling_matrix.csv"
    fig_path = FIGURES_DIR / "exp15_runtime_scaling_matrix.png"
    tex_path = TABLES_DIR / "exp15_runtime_scaling_matrix.tex"
    moved = supersede_if_exists(csv_path, fig_path, tex_path, csv_path.with_suffix(".config.json"))
    for m in moved:
        print(f"superseded existing output -> {m}")

    # ── CSV ──────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    # ── Figure: median with IQR error bars ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    Ns = [r["N"] for r in rows]
    for key, label, marker in [
        ("galerkin", "Galerkin (reference solve)", "o"),
        ("inference", "inference (ProjectedFNO)", "s"),
        ("certification", "certification (quick eta_guar)", "^"),
        ("correction", "correction (warm-started GMRES)", "v"),
        ("total", "total -- FULL/FORCED-CORRECTION PATH", "D"),
    ]:
        med = [r[f"t_{key}_median_ms"] for r in rows]
        q1 = [r[f"t_{key}_q1_ms"] for r in rows]
        q3 = [r[f"t_{key}_q3_ms"] for r in rows]
        yerr = [[m - lo for m, lo in zip(med, q1)], [hi - m for m, hi in zip(med, q3)]]
        ax.errorbar(Ns, med, yerr=yerr, marker=marker, label=label, capsize=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mesh resolution N"); ax.set_ylabel("time (ms, median with IQR)")
    ax.set_title(f"exp15: runtime scaling, N=32...512 (n>={min(r['n_valid'] for r in rows)}/resolution, "
                 f"ProjectedFNO)\nFULL/FORCED-CORRECTION PATH -- NOT the tolerance-dependent workflow average "
                 f"(see exp11_tolerance_sweep.py)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    header = ["N", "n\\_dof", "n\\_valid", "Galerkin median (ms)", "total median (ms)",
               "total Q1 (ms)", "total Q3 (ms)", "total mean $\\pm$ std (ms)", "not-SPD"]
    rows_tex = [
        [str(r["N"]), str(r["n_dof"]), str(r["n_valid"]), fmt(r["t_galerkin_median_ms"]),
         fmt(r["t_total_median_ms"]), fmt(r["t_total_q1_ms"]), fmt(r["t_total_q3_ms"]),
         f"{fmt(r['t_total_mean_ms'])} $\\pm$ {fmt(r['t_total_std_ms'])}", str(r["n_not_spd"])]
        for r in rows
    ]
    write_latex_table(tex_path, header, rows_tex)
    # Append the required caveat sentence directly below the table (item 1.2's own instruction:
    # "Add a sentence in the text stating it is not the average cost of the tolerance-dependent
    # workflow, and that the tolerance sweep is the primary end-to-end timing experiment.")
    with open(tex_path, "a") as f:
        f.write(
            "\n% NOTE: \"total\" above is the FULL / FORCED-CORRECTION PATH "
            "(T_NN + T_cert + T_correct, tolerance=1e-9 forces correction on every query) -- "
            "it is NOT the average cost of the real, tolerance-dependent predict-certify-correct "
            "workflow, which often skips correction entirely. "
            "experiments/exp11_tolerance_sweep.py is the primary end-to-end workflow timing "
            "experiment; this table characterizes worst-case (always-correct) cost scaling only.\n"
        )
    print(f"wrote {tex_path}")

    # ── Provenance sidecar ───────────────────────────────────────────────
    prov_path = write_provenance(
        csv_path, start_time=t_start, seed={"queries": SEED, "candidates": CANDIDATE_SEED},
        git_commit_override=git_commit,
        n_requested=TARGET_N_COMMON, n_kept=search_stats["n_kept"],
        n_rejected=search_stats["n_drawn"] - search_stats["n_kept"],
        mesh_resolutions=RESOLUTIONS,
        extra={
            "tolerance": TOLERANCE, "quick_pcg_iterations": QUICK_PCG_ITERATIONS,
            "max_n_dof_override": MAX_N_DOF_OVERRIDE,
            "used_fallback_per_resolution_sampling": used_fallback,
            "common_sample_search_stats": search_stats,
            "labeling": "total = FULL/FORCED-CORRECTION PATH, NOT the tolerance-dependent "
                        "workflow average -- see exp11_tolerance_sweep.py for that",
        },
    )
    print(f"wrote {prov_path}")


if __name__ == "__main__":
    main()
