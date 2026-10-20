"""exp02 — per-query effectivity of the three certificates.

For each test query, evaluates the FINAL model (exp_common.FINAL_MODEL_NAME
-- DeepONet, chosen on Stage H validation performance; exp01 is the
cross-model comparison, this and every later experiment focus on
certifying one model's predictions) and computes:

    E_i = max_m ||U_h^m - U_theta,h^m||_{L2}        (task's literal definition)
    eta_mass_indicator_i (heuristic, NOT a certificate -- see
        certification.mass_indicator), eta_sharp_i, eta_guar_i
    I_eff,i = eta_i / E_i                             for each certificate

Also compares observed effectivity against the theoretical bound
(main.tex, thm:efficiency)

    eta_sharp^2 / |||e|||^2 <= 2 * max{1, kappa(mu)^2}

CHANGED (fix/efficiency-norm-kappa, Steps 1-2): |||e|||^2 and kappa(mu) are
now the manuscript's EXACT definitions --
certification.constants.triple_norm_squared (eq:triple-norm: a genuine
space-time norm, dt*sum_n [ ||M_h de/dt||_{S_h^-1}^2 + ||ebar||_{S_h}^2 ],
with BOTH the time-derivative and midpoint-energy terms) and
certification.constants.kappa_operator_norm (eq:kappa-def:
||S_h^-1/2 A_h S_h^-1/2||_2, NOT cond(S_h)). A presubmission review
(CGMY_presubmission_review_and_efficiency_fix_notes.md) found that an
earlier pass -- made without manuscript access -- had gotten BOTH
quantities wrong: |||e|||^2 was mass/L2-only (no time-derivative term at
all, not even the wrong norm family), and kappa was cond(S_h(mu)), which
main.tex's Lemma~\\ref{lem:kappa} explicitly is not (e.g. A_h(mu)=S_h(mu)
gives kappa=1 exactly regardless of how large cond(S_h(mu)) is -- see
tests/test_efficiency_norm_kappa.py). The old mass/L2 quantity is still
computed and logged (mass_l2_error_squared_legacy) purely for comparison,
never consumed by the actual pass/fail bound check below.

Also computes, for every certified sample, the Step-4 identity-discrepancy
diagnostic (exp_common.identity_discrepancy): how well r_theta + M_h*de/dt
+ A_h*ebar vanishes -- i.e. whether the residual evaluator and the
triple-norm's own terms are actually consistent with each other on this
project's real operators/trajectories, not just individually plausible.

**The guaranteed certificate's violation count MUST be 0** (eta_guar >= E_i
for every query). If it is not, this script prints a loud, unmissable
FAILURE banner rather than quietly reporting the count as just another
number -- CLAUDE.md Hard Invariant #5: "guaranteed means guaranteed."

Scale note: N_TEST=100 (Task 11, working prompt: "certification test:
N_cert >= 100" -- exp02 is this project's per-query certification/
effectivity test, so this is where that floor applies).

Outputs: results/exp02_effectivity.h5, figures/exp02_effectivity.png,
tables/exp02_effectivity.tex, results/exp02_per_sample.csv.
"""

from __future__ import annotations

import csv
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from certification.constants import (
    kappa_operator_norm,
    mass_l2_error_squared_legacy,
    triple_norm_squared,
)
from data.sampling import sample_sobol
from exp_common import (
    FIGURES_DIR,
    FINAL_MODEL_NAME,
    RESULTS_DIR,
    TABLES_DIR,
    capture_git_commit_hash,
    compute_certificates,
    fmt,
    get_box_and_stats,
    get_ops,
    identity_discrepancy,
    load_models,
    predict,
    provenance_start,
    solve_reference,
    supersede_if_exists,
    true_error_Ei,
    write_h5,
    write_latex_table,
    write_provenance,
)

N_TEST = 100
SEED = 424243  # exp01's seed + 1, still disjoint from Stage H's seeds


def main() -> None:
    t_start = provenance_start()
    git_commit = capture_git_commit_hash()  # BEFORE any output is written -- see its docstring
    ops = get_ops()
    box, stats = get_box_and_stats()
    models = load_models(ops, stats)
    model = models[FINAL_MODEL_NAME]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_samples = sample_sobol(box, N_TEST, seed=SEED)

    E_i_list, eta_mass_indicator_list, eta_sharp_list, eta_guar_list = [], [], [], []
    kappa_list, bound_lhs_list, bound_rhs_list = [], [], []
    mass_l2_legacy_list, idc_max_list, idc_median_list, idc_p99_list = [], [], [], []
    per_sample_rows = []
    n_skipped_not_spd = 0

    print(f"exp02: certifying {FINAL_MODEL_NAME} on {N_TEST} queries...")
    for i, (mu, payoff) in enumerate(test_samples):
        traj = solve_reference(mu)
        pred = predict(model, mu, ops, stats, traj.initial_coefficients, traj.time_grid)

        cert = compute_certificates(traj, pred, mu, ops)
        if not cert.is_spd:
            n_skipped_not_spd += 1
            continue

        dt = traj.solver_metadata.dt
        N_T = traj.solver_metadata.n_steps
        E_i = true_error_Ei(pred, traj.coefficients, ops)
        e_sq = triple_norm_squared(pred, traj.coefficients, mu, ops, dt)
        mass_l2_legacy = mass_l2_error_squared_legacy(pred, traj.coefficients, ops, dt)
        kappa = kappa_operator_norm(mu, ops)
        idc = identity_discrepancy(traj, pred, mu, ops)
        bound_rhs = 2.0 * max(1.0, kappa ** 2)
        bound_lhs = (cert.eta_sharp ** 2) / e_sq if e_sq > 0 else float("nan")
        passed = bool(bound_lhs <= bound_rhs) if bound_lhs == bound_lhs else False

        E_i_list.append(E_i)
        eta_mass_indicator_list.append(cert.eta_mass_indicator)
        eta_sharp_list.append(cert.eta_sharp)
        eta_guar_list.append(cert.eta_guar)
        kappa_list.append(kappa)
        bound_lhs_list.append(bound_lhs)
        bound_rhs_list.append(bound_rhs)
        mass_l2_legacy_list.append(mass_l2_legacy)
        idc_max_list.append(idc["max"])
        idc_median_list.append(idc["median"])
        idc_p99_list.append(idc["p99"])

        per_sample_rows.append({
            "sample_id": i,
            "C": mu.C, "G": mu.G, "M": mu.M, "Y": mu.Y, "r": mu.r, "T": mu.T,
            "N": ops.N, "dt": dt, "N_T": N_T,
            "eta_sharp_sq": cert.eta_sharp ** 2, "triple_norm_sq": e_sq,
            "mass_l2_legacy_sq": mass_l2_legacy,
            "kappa": kappa, "cond_S_h": cert.condition_number,
            "bound_rhs": bound_rhs, "ratio": bound_lhs, "pass": passed,
            "identity_discrepancy_max": idc["max"],
            "identity_discrepancy_median": idc["median"],
            "identity_discrepancy_p99": idc["p99"],
        })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_TEST}")

    if n_skipped_not_spd:
        print(f"NOTE: {n_skipped_not_spd}/{N_TEST} queries skipped (S_h(mu) not SPD)")

    E_i_arr = np.array(E_i_list)
    eta_mass_indicator_arr = np.array(eta_mass_indicator_list)
    eta_sharp_arr = np.array(eta_sharp_list)
    eta_guar_arr = np.array(eta_guar_list)
    kappa_arr = np.array(kappa_list)
    bound_lhs_arr = np.array(bound_lhs_list)
    bound_rhs_arr = np.array(bound_rhs_list)
    idc_max_arr = np.array(idc_max_list)

    I_eff_mass_indicator = eta_mass_indicator_arr / E_i_arr
    I_eff_sharp = eta_sharp_arr / E_i_arr
    I_eff_guar = eta_guar_arr / E_i_arr

    # ── THE hard check: guaranteed certificate violations must be 0 ────────
    violations_guar = int(np.sum(eta_guar_arr < E_i_arr))
    violations_sharp = int(np.sum(eta_sharp_arr < E_i_arr))
    bound_violations = int(np.sum(bound_lhs_arr > bound_rhs_arr))

    print("\n" + "=" * 78)
    if violations_guar == 0:
        print(f"GUARANTEED CERTIFICATE: 0/{len(E_i_arr)} violations (eta_guar >= E_i always). PASS.")
    else:
        print("!" * 78)
        print(f"!!! GUARANTEED CERTIFICATE VIOLATED: {violations_guar}/{len(E_i_arr)} queries had "
              f"eta_guar < E_i !!!")
        print("!!! This means either a bug or a broken assumption (e.g. coercivity_constant")
        print("!!! is not actually a valid lower bound on lambda_min(S_h(mu))). DO NOT trust")
        print("!!! eta_guar as a guaranteed bound until this is root-caused. !!!")
        print("!" * 78)
    print("=" * 78)

    def summary(name, arr):
        return (f"{name:10s} min={arr.min():.3e} median={np.median(arr):.3e} mean={arr.mean():.3e} "
                f"p95={np.percentile(arr, 95):.3e} max={arr.max():.3e}")

    print(summary("I_eff_mass_ind", I_eff_mass_indicator))
    print(summary("I_eff_sharp", I_eff_sharp))
    print(summary("I_eff_guar", I_eff_guar))
    print(f"eta_sharp violations (eta_sharp < E_i): {violations_sharp}/{len(E_i_arr)} "
          f"(eta_sharp is NOT claimed guaranteed, so this is informational)")
    print(f"theoretical bound eta_sharp^2/|||e|||^2 <= 2*max(1,kappa^2) violations: "
          f"{bound_violations}/{len(E_i_arr)}")
    print(f"Step 4 identity discrepancy ||r_theta + M_h*de/dt + A_h*ebar|| / (sum of term norms): "
          f"max={idc_max_arr.max():.3e} median={np.median(idc_max_arr):.3e} "
          f"p99={np.percentile(idc_max_arr, 99):.3e} "
          f"(root cause of the ~1e-6..1e-7 floor: the trusted, vendored reference solver's own "
          f"per-step GMRES rtol=1e-10 -- see fix/efficiency-norm-kappa's Step 4 investigation; "
          f"NOT a bug in this project's residual/triple-norm code)")

    # ── Never overwrite results silently: move any existing outputs to
    #    results/_superseded/<date>/ before writing new ones in their place ──
    h5_path = RESULTS_DIR / "exp02_effectivity.h5"
    fig_path = FIGURES_DIR / "exp02_effectivity.png"
    tex_path = TABLES_DIR / "exp02_effectivity.tex"
    csv_path = RESULTS_DIR / "exp02_per_sample.csv"
    moved = supersede_if_exists(h5_path, fig_path, tex_path, h5_path.with_suffix(".config.json"), csv_path)
    for m in moved:
        print(f"superseded existing output -> {m}")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    write_h5(
        RESULTS_DIR / "exp02_effectivity.h5",
        {
            "E_i": E_i_arr, "eta_mass_indicator": eta_mass_indicator_arr,
            "eta_sharp": eta_sharp_arr, "eta_guar": eta_guar_arr,
            "I_eff_mass_indicator": I_eff_mass_indicator, "I_eff_sharp": I_eff_sharp, "I_eff_guar": I_eff_guar,
            "kappa": kappa_arr, "bound_lhs": bound_lhs_arr, "bound_rhs": bound_rhs_arr,
            "mass_l2_legacy": np.array(mass_l2_legacy_list),
            "identity_discrepancy_max": idc_max_arr,
        },
        attrs={
            "n_test": N_TEST, "seed": SEED, "model": FINAL_MODEL_NAME,
            "n_skipped_not_spd": n_skipped_not_spd,
            "violations_guar": violations_guar, "violations_sharp": violations_sharp,
            "bound_violations": bound_violations,
            "identity_discrepancy_max_overall": float(idc_max_arr.max()),
        },
    )

    # ── Per-sample CSV (Step 5, working prompt) ─────────────────────────
    if per_sample_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_rows)
        print(f"wrote {csv_path}")

    # ── Figure: log-log scatter of E_i vs eta_i, with y=x ───────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    lims = [min(E_i_arr.min(), eta_guar_arr.min()) * 0.5, max(eta_mass_indicator_arr.max(), E_i_arr.max()) * 2]
    ax.plot(lims, lims, "k--", lw=1, label=r"$y=x$")
    ax.scatter(E_i_arr, eta_mass_indicator_arr, s=20, alpha=0.6,
               label=r"$\eta_{\mathrm{mass\ indicator}}$ (heuristic, not a bound)", marker="^")
    ax.scatter(E_i_arr, eta_sharp_arr, s=20, alpha=0.6, label=r"$\eta_{sharp}$", marker="s")
    ax.scatter(E_i_arr, eta_guar_arr, s=20, alpha=0.6, label=r"$\eta_{guar}$", marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"true error $E_i$")
    ax.set_ylabel(r"certificate $\eta_i$")
    ax.set_title(f"exp02: certificate vs. true error ({FINAL_MODEL_NAME}, n={len(E_i_arr)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    header = ["quantity", "min $I_{eff}$", "median", "mean", "p95", "max", "violations"]
    rows = [
        ["eta_mass_indicator (heuristic, not a certificate)",
         fmt(I_eff_mass_indicator.min()), fmt(np.median(I_eff_mass_indicator)), fmt(I_eff_mass_indicator.mean()),
         fmt(np.percentile(I_eff_mass_indicator, 95)), fmt(I_eff_mass_indicator.max()), "n/a (not guaranteed)"],
        ["eta_sharp", fmt(I_eff_sharp.min()), fmt(np.median(I_eff_sharp)), fmt(I_eff_sharp.mean()),
         fmt(np.percentile(I_eff_sharp, 95)), fmt(I_eff_sharp.max()), str(violations_sharp)],
        ["eta_guar", fmt(I_eff_guar.min()), fmt(np.median(I_eff_guar)), fmt(I_eff_guar.mean()),
         fmt(np.percentile(I_eff_guar, 95)), fmt(I_eff_guar.max()), str(violations_guar) + (" (MUST be 0)")],
    ]
    write_latex_table(tex_path, header, rows)
    print(f"wrote {tex_path}")

    # ── Provenance sidecar (Ground Rule 3) ──────────────────────────────
    prov_path = write_provenance(
        h5_path, start_time=t_start, seed=SEED,
        n_requested=N_TEST, n_kept=len(E_i_arr), n_rejected=n_skipped_not_spd,
        git_commit_override=git_commit,
        extra={
            "model": FINAL_MODEL_NAME,
            "norm_for_triple_bar_e": "main.tex eq:triple-norm exactly (M_h de/dt in S_h^-1, "
                                      "ebar in S_h, both terms) -- fix/efficiency-norm-kappa Step 1; "
                                      "see CGMY_presubmission_review_and_efficiency_fix_notes.md",
            "kappa_definition": "main.tex eq:kappa-def exactly, ||S_h^-1/2 A_h S_h^-1/2||_2 -- "
                                 "fix/efficiency-norm-kappa Step 2, NOT cond(S_h)",
            "violations_guar": violations_guar, "violations_sharp": violations_sharp,
            "bound_violations": bound_violations,
            "identity_discrepancy_max_overall": float(idc_max_arr.max()),
        },
    )
    print(f"wrote {prov_path}")


if __name__ == "__main__":
    main()
