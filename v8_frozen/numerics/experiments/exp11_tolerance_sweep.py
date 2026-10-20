"""exp11 -- Task 10 (working prompt): the tolerance sweep that replaces the
single "calibrated eps ~ 7.3" headline speedup number (exp08_runtime.py)
with a full characterization across the tolerance axis.

exp08's own docstring already says this plainly: configs/base.yaml's frozen
certification.acceptance_tolerance is 1e-3, far below the eta_guar
magnitudes actually observed for this model (single digits to tens -- see
exp01-03), so at that nominal value nearly every query is sent to
correction and the "speedup" number describes the EXPENSIVE branch almost
exclusively. exp08 sidesteps this with one calibrated illustrative value
(median eta_guar_quick over a small sample). A single number -- calibrated
or nominal -- does not characterize a method whose behavior changes
qualitatively across the tolerance axis (nearly-always-correct vs
nearly-always-accept). This script sweeps the full axis instead.

Design (efficiency without changing what's measured): for a FIXED tolerance
value, whether a query is accepted immediately, corrected, or found not-SPD
is a genuine function of tolerance (eta_guar_quick <= tolerance), but the
reference (Galerkin) trajectory and its solve time are NOT functions of
tolerance at all -- solve_cgmy never sees a tolerance. So each of the
N_QUERIES (mu, payoff) pairs is solved by the reference solver exactly
ONCE and reused across every tolerance value in the sweep (identical in
spirit to reusing OperatorCache within a query loop -- this project's
established pattern for "compute once, reuse where a repeated computation
provably does not change"), while predict_certify_correct itself is
re-run, fresh, once per (query, tolerance) pair -- the full six-step
workflow, correction included whenever the tolerance demands it, never
skipped via run_correction=False (unlike exp08's UN-timed calibration
pass, every number in this sweep comes from a real, timed, fully-executed
workflow).

True error (E_i) and effectivity reuse exp02's established definitions
(exp_common.true_error_Ei) for consistency with the rest of this
project's certificate-vs-error reporting, rather than inventing a new
"true error" convention for this one script. The guaranteed-certificate
violation check (CLAUDE.md Hard Invariant #5: eta_guar must NEVER be
smaller than the true error) is run across every tolerance and query,
same as exp02.

Terminology (pre-submission fixes, item 7.1): this script reports TWO
different rates, and neither is bare "acceptance rate" -- that phrase is
ambiguous between them (this project's own exp12_seed_variance.py used
it for a THIRD, differently-scoped number until this same fix):
  - "direct accept" / direct_accept_rate: fraction of queries where
    eta(u_NN) <= tolerance with NO correction run at all (branch ==
    accepted_guar_quick).
  - "final_certified_success_rate": fraction of queries where the
    COMPLETE predict-certify-correct pipeline ultimately produces a
    certified solution meeting the tolerance, whether via direct
    acceptance OR after correction (direct_accept_rate + the fraction of
    corrected queries that end up certified). Reported explicitly below
    (previously computed as `certified_acceptance_rate` but never
    surfaced in any table/print) so it can be compared directly against
    exp12_seed_variance.py's own (now identically-named, same
    definition) metric: at tolerance=5, this script's
    final_certified_success_rate and exp12's acceptance number
    (calibrated tolerance ~5.17, close to this sweep's tolerance=5
    point) should land close together, since both measure the SAME
    thing (final workflow success), just at slightly different
    tolerances -- unlike direct_accept_rate, which measures a narrower,
    genuinely different quantity and is NOT expected to match either.

Outputs: results/exp11_tolerance_sweep.h5, figures/exp11_tolerance_sweep.png,
tables/exp11_tolerance_sweep_headline.tex (the task's literal 6-column
table), tables/exp11_tolerance_sweep_full.tex (every metric the task text
asks to be reported, not just the headline table's columns).
"""

from __future__ import annotations

import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.sampling import sample_sobol
from exp_common import (
    FIGURES_DIR,
    FIXED_MESH,
    FINAL_MODEL_NAME,
    MAX_PCG_ITERATIONS,
    PRECONDITIONER,
    RESULTS_DIR,
    TABLES_DIR,
    fmt,
    get_box_and_stats,
    get_ops,
    load_models,
    capture_git_commit_hash,
    provenance_start,
    solve_reference,
    supersede_if_exists,
    true_error_Ei,
    write_h5,
    write_latex_table,
    write_provenance,
)
from pipeline.predict_certify_correct import (
    BRANCH_ACCEPTED_AFTER_CORRECTION_GUAR,
    BRANCH_ACCEPTED_GUAR_QUICK,
    BRANCH_CORRECTED_BUT_NOT_CERTIFIED,
    BRANCH_NOT_SPD_UNCERTIFIABLE,
    predict_certify_correct,
)

N_QUERIES = 120  # >=100 (task minimum), with margin for not-SPD/solver-failure dropouts
QUICK_PCG_ITERATIONS = 5  # matches exp08's convention for the cheap escalation step
SEED = 111011  # disjoint from every other experiment's seed in this project
TOLERANCES = [1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 10.0]


def main() -> None:
    t_start = provenance_start()
    git_commit = capture_git_commit_hash()  # BEFORE any output is written -- see its docstring
    ops = get_ops()
    box, stats = get_box_and_stats()
    models = load_models(ops, stats)
    model = models[FINAL_MODEL_NAME]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = sample_sobol(box, N_QUERIES, seed=SEED)

    # ── Solve the reference trajectory ONCE per query, reused across every
    #    tolerance (see module docstring) ───────────────────────────────
    print(f"exp11: solving {N_QUERIES} reference trajectories (once, reused across all "
          f"{len(TOLERANCES)} tolerances)...")
    queries = []
    for i, (mu, payoff) in enumerate(samples):
        t0 = time.perf_counter()
        try:
            traj = solve_reference(mu)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] reference solve failed: {exc}; skipping query")
            continue
        t_full_galerkin = time.perf_counter() - t0
        queries.append({"mu": mu, "payoff": payoff, "traj": traj, "t_full_galerkin": t_full_galerkin})
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_QUERIES}")
    n_q = len(queries)
    print(f"  {n_q}/{N_QUERIES} reference trajectories solved successfully.")

    sweep = {}  # tol -> dict of aggregate metrics
    per_tol_arrays = {}  # tol -> dict of per-query arrays (for HDF5)
    violations_total = 0

    for tol in TOLERANCES:
        print(f"\nexp11: tolerance={tol:.3e} -- running {n_q} full predict-certify-correct workflows...")
        branches = []
        t_totals, t_galerkins, speedups = [], [], []
        E_list, eta_list, eff_list = [], [], []

        for q in queries:
            trace = predict_certify_correct(
                q["mu"], q["payoff"], FIXED_MESH, q["traj"].time_grid, model, stats, ops,
                tolerance=tol, quick_pcg_iterations=QUICK_PCG_ITERATIONS,
                max_pcg_iterations=MAX_PCG_ITERATIONS, preconditioner=PRECONDITIONER,
                keep_coefficients=True,
            )
            branches.append(trace.branch)
            t_totals.append(trace.t_total)
            t_galerkins.append(q["t_full_galerkin"])
            speedups.append(q["t_full_galerkin"] / trace.t_total if trace.t_total > 0 else float("nan"))

            if trace.is_spd and trace.final_coefficients is not None:
                E_i = true_error_Ei(trace.final_coefficients, q["traj"].coefficients, ops)
                eta_final = trace.eta_final
                E_list.append(E_i)
                eta_list.append(eta_final)
                eff_list.append(eta_final / E_i if E_i > 0 else float("nan"))
                if not np.isnan(eta_final) and eta_final < E_i:
                    violations_total += 1
                    print(f"  !!! GUARANTEED CERTIFICATE VIOLATED at tol={tol:.3e}: "
                          f"eta_final={eta_final:.4e} < E_i={E_i:.4e} for mu={q['mu']} !!!")

        branches_arr = np.array(branches)
        t_totals_arr = np.array(t_totals)
        t_galerkins_arr = np.array(t_galerkins)
        speedups_arr = np.array(speedups)
        E_arr = np.array(E_list)
        eta_arr = np.array(eta_list)
        eff_arr = np.array(eff_list)

        n = len(branches_arr)
        n_direct_accept = int(np.sum(branches_arr == BRANCH_ACCEPTED_GUAR_QUICK))
        n_correction = int(np.sum(
            (branches_arr == BRANCH_ACCEPTED_AFTER_CORRECTION_GUAR)
            | (branches_arr == BRANCH_CORRECTED_BUT_NOT_CERTIFIED)
        ))
        n_not_spd = int(np.sum(branches_arr == BRANCH_NOT_SPD_UNCERTIFIABLE))
        n_certified_total = n_direct_accept + int(np.sum(branches_arr == BRANCH_ACCEPTED_AFTER_CORRECTION_GUAR))

        agg = {
            "n": n,
            "n_not_spd": n_not_spd,
            "direct_accept_rate": n_direct_accept / n,
            "correction_rate": n_correction / n,
            # Renamed from "certified_acceptance_rate" (item 7.1: no bare
            # "acceptance" label) and now actually surfaced in the full
            # table/print output, not just computed and left in the h5.
            "final_certified_success_rate": n_certified_total / n,
            "mean_runtime": float(np.mean(t_totals_arr)),
            "median_runtime": float(np.median(t_totals_arr)),
            "p95_runtime": float(np.percentile(t_totals_arr, 95)),
            "mean_galerkin_runtime": float(np.mean(t_galerkins_arr)),
            "median_galerkin_runtime": float(np.median(t_galerkins_arr)),
            # Pre-submission fixes item 6.1: THREE distinct statistics, not
            # interchangeable -- explicit formulas, not just names:
            #   mean_speedup    = S_mean = (1/N) sum_i T_G,i / T_W,i   (mean of per-query ratios)
            #   median_speedup  = median_i (T_G,i / T_W,i)             (median of per-query ratios)
            #   ratio_of_medians = median_i(T_G,i) / median_i(T_W,i)   (ratio of the two medians --
            #       a DIFFERENT statistic from median_speedup; do not conflate them)
            "mean_speedup": float(np.mean(speedups_arr)),
            "median_speedup": float(np.median(speedups_arr)),
            "ratio_of_medians": (float(np.median(t_galerkins_arr) / np.median(t_totals_arr))
                                  if np.median(t_totals_arr) > 0 else float("nan")),
            "mean_true_error": float(np.mean(E_arr)) if len(E_arr) else float("nan"),
            "max_true_error": float(np.max(E_arr)) if len(E_arr) else float("nan"),
            "mean_certificate": float(np.mean(eta_arr)) if len(eta_arr) else float("nan"),
            "mean_effectivity": float(np.nanmean(eff_arr)) if len(eff_arr) else float("nan"),
        }
        sweep[tol] = agg
        per_tol_arrays[tol] = {
            "branch": branches_arr, "t_total": t_totals_arr, "t_galerkin": t_galerkins_arr,
            "speedup": speedups_arr, "E_i": E_arr, "eta_final": eta_arr, "effectivity": eff_arr,
        }

        print(f"  n={n} (not-SPD: {n_not_spd}/{n})  direct_accept_rate={100*agg['direct_accept_rate']:.1f}%  "
              f"correction={100*agg['correction_rate']:.1f}%  "
              f"final_certified_success_rate={100*agg['final_certified_success_rate']:.1f}%  "
              f"median_runtime={1000*agg['median_runtime']:.3f}ms  "
              f"mean_speedup(S_mean=mean_i[T_G,i/T_W,i])={agg['mean_speedup']:.2f}x  "
              f"median_speedup(median_i[T_G,i/T_W,i])={agg['median_speedup']:.2f}x  "
              f"ratio_of_medians(median[T_G]/median[T_W])={agg['ratio_of_medians']:.2f}x  "
              f"mean_true_error={agg['mean_true_error']:.3e}  max_true_error={agg['max_true_error']:.3e}  "
              f"mean_certificate={agg['mean_certificate']:.3e}  mean_effectivity={agg['mean_effectivity']:.2f}")

    print("\n" + "=" * 78)
    if violations_total == 0:
        print(f"GUARANTEED CERTIFICATE: 0 violations across all {len(TOLERANCES)} tolerances "
              f"x {n_q} queries. PASS.")
    else:
        print(f"!!! GUARANTEED CERTIFICATE VIOLATED {violations_total} times -- see above. "
              f"DO NOT trust eta_guar until this is root-caused. !!!")
    print("=" * 78)

    # ── Sanity check: not-SPD fraction should be identical across tolerances
    #    (SPD-ness of S_h(mu) does not depend on the acceptance tolerance) ──
    not_spd_fracs = {tol: sweep[tol]["n_not_spd"] for tol in TOLERANCES}
    if len(set(not_spd_fracs.values())) > 1:
        print(f"NOTE: not-SPD count varies across tolerances ({not_spd_fracs}) -- unexpected, "
              f"since SPD-ness does not depend on tolerance; investigate if this recurs.")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    # write_arrays (common.io) already converts unicode-string arrays to
    # h5py variable-length strings, so the branch arrays need no manual
    # encoding here.
    arrays_out = {
        f"tol_{tol:.0e}/{name}": arr
        for tol in TOLERANCES for name, arr in per_tol_arrays[tol].items()
    }
    h5_path = RESULTS_DIR / "exp11_tolerance_sweep.h5"
    moved = supersede_if_exists(
        h5_path, FIGURES_DIR / "exp11_tolerance_sweep.png",
        TABLES_DIR / "exp11_tolerance_sweep_headline.tex", TABLES_DIR / "exp11_tolerance_sweep_full.tex",
        h5_path.with_suffix(".config.json"),
    )
    for m in moved:
        print(f"superseded existing output -> {m}")
    write_h5(
        h5_path, arrays_out,
        attrs={
            "n_queries_requested": N_QUERIES, "n_queries_solved": n_q, "seed": SEED,
            "model": FINAL_MODEL_NAME, "quick_pcg_iterations": QUICK_PCG_ITERATIONS,
            "tolerances": np.array(TOLERANCES),
            "violations_total": violations_total,
            **{f"tol_{tol:.0e}_{k}": v for tol in TOLERANCES for k, v in sweep[tol].items()},
        },
    )

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    tols_arr = np.array(TOLERANCES)

    axes[0, 0].plot(tols_arr, [100 * sweep[t]["direct_accept_rate"] for t in TOLERANCES], "o-", label="direct accept")
    axes[0, 0].plot(tols_arr, [100 * sweep[t]["correction_rate"] for t in TOLERANCES], "s-", label="correction")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel(r"tolerance $\epsilon$")
    axes[0, 0].set_ylabel("% of queries")
    axes[0, 0].set_title("exp11: decision-branch rate vs. tolerance")
    axes[0, 0].legend()

    axes[0, 1].plot(tols_arr, [1000 * sweep[t]["median_runtime"] for t in TOLERANCES], "o-", color="C2")
    axes[0, 1].axhline(1000 * np.mean([sweep[t]["mean_galerkin_runtime"] for t in TOLERANCES]),
                        color="k", linestyle="--", label="mean Galerkin runtime")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel(r"tolerance $\epsilon$")
    axes[0, 1].set_ylabel("median runtime (ms)")
    axes[0, 1].set_title("exp11: workflow runtime vs. tolerance")
    axes[0, 1].legend()

    axes[1, 0].plot(tols_arr, [sweep[t]["mean_speedup"] for t in TOLERANCES], "o-", color="C3")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel(r"tolerance $\epsilon$")
    axes[1, 0].set_ylabel("mean speedup")
    axes[1, 0].set_title("exp11: speedup vs. tolerance")

    axes[1, 1].plot(tols_arr, [sweep[t]["mean_true_error"] for t in TOLERANCES], "o-", label="mean true error")
    axes[1, 1].plot(tols_arr, [sweep[t]["max_true_error"] for t in TOLERANCES], "s-", label="max true error")
    axes[1, 1].plot(tols_arr, [sweep[t]["mean_certificate"] for t in TOLERANCES], "^-", label="mean certificate")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel(r"tolerance $\epsilon$")
    axes[1, 1].set_ylabel("error / certificate")
    axes[1, 1].set_title("exp11: error and certificate vs. tolerance")
    axes[1, 1].legend()

    fig.suptitle(f"exp11: tolerance sweep ({FINAL_MODEL_NAME}, n={n_q} queries/tolerance)")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp11_tolerance_sweep.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {fig_path}")

    # ── LaTeX tables ─────────────────────────────────────────────────────
    # Headline table: the task's literal 6-column ask.
    header_headline = ["tolerance", "direct accept", "correction", "median runtime", "speedup", "max true error"]
    rows_headline = [
        [fmt(tol), f"{100*sweep[tol]['direct_accept_rate']:.1f}\\%", f"{100*sweep[tol]['correction_rate']:.1f}\\%",
         f"{1000*sweep[tol]['median_runtime']:.3f}ms", f"{sweep[tol]['mean_speedup']:.2f}x",
         fmt(sweep[tol]["max_true_error"])]
        for tol in TOLERANCES
    ]
    write_latex_table(TABLES_DIR / "exp11_tolerance_sweep_headline.tex", header_headline, rows_headline)
    print(f"wrote {TABLES_DIR / 'exp11_tolerance_sweep_headline.tex'}")

    # Full table: every metric the task text asks to be reported. Pre-
    # submission fixes item 6.1: mean_speedup, median_speedup, and
    # ratio_of_medians are THREE DIFFERENT statistics (see the "agg" dict
    # comment above for the exact formulas) -- all three are reported
    # here, not just "mean speedup", so a reader can't mistake one for
    # another the way the original single-number table invited.
    header_full = [
        "tolerance", "direct accept rate", "correction", "final certified success rate", "not-SPD",
        "mean runtime", "median runtime",
        "p95 runtime", "Galerkin runtime (median)", "mean speedup $S_{mean}$", "median speedup",
        "ratio of medians", "mean true err", "max true err", "mean cert.", "mean effectivity",
    ]
    rows_full = [
        [fmt(tol), f"{100*sweep[tol]['direct_accept_rate']:.1f}\\%", f"{100*sweep[tol]['correction_rate']:.1f}\\%",
         f"{100*sweep[tol]['final_certified_success_rate']:.1f}\\%",
         f"{sweep[tol]['n_not_spd']}/{sweep[tol]['n']}",
         f"{1000*sweep[tol]['mean_runtime']:.3f}ms", f"{1000*sweep[tol]['median_runtime']:.3f}ms",
         f"{1000*sweep[tol]['p95_runtime']:.3f}ms", f"{1000*sweep[tol]['median_galerkin_runtime']:.3f}ms",
         f"{sweep[tol]['mean_speedup']:.2f}x", f"{sweep[tol]['median_speedup']:.2f}x",
         f"{sweep[tol]['ratio_of_medians']:.2f}x",
         fmt(sweep[tol]["mean_true_error"]), fmt(sweep[tol]["max_true_error"]),
         fmt(sweep[tol]["mean_certificate"]), f"{sweep[tol]['mean_effectivity']:.2f}"]
        for tol in TOLERANCES
    ]
    write_latex_table(TABLES_DIR / "exp11_tolerance_sweep_full.tex", header_full, rows_full)
    with open(TABLES_DIR / "exp11_tolerance_sweep_full.tex", "a") as f:
        f.write(
            "\n% Speedup definitions (pre-submission fixes, item 6.1) -- three DIFFERENT statistics:\n"
            "%   mean speedup S_mean     = (1/N) sum_i T_Galerkin,i / T_workflow,i   (mean of per-query ratios)\n"
            "%   median speedup          = median_i (T_Galerkin,i / T_workflow,i)    (median of per-query ratios)\n"
            "%   ratio of medians        = median_i(T_Galerkin,i) / median_i(T_workflow,i)\n"
            "% These are NOT interchangeable -- e.g. at tolerance=5, mean speedup and\n"
            "% (median Galerkin runtime)/(median workflow runtime) can differ substantially\n"
            "% (mean-of-ratios is not the same as ratio-of-medians in general); both are\n"
            "% reported explicitly above rather than presenting one under the ambiguous\n"
            "% label \"speedup\".\n"
        )
    print(f"wrote {TABLES_DIR / 'exp11_tolerance_sweep_full.tex'}")

    prov_path = write_provenance(
        h5_path, start_time=t_start, seed=SEED, n_requested=N_QUERIES, n_kept=n_q,
        git_commit_override=git_commit,
        extra={"tolerances": list(TOLERANCES), "quick_pcg_iterations": QUICK_PCG_ITERATIONS,
               "violations_total": violations_total},
    )
    print(f"wrote {prov_path}")


if __name__ == "__main__":
    main()
