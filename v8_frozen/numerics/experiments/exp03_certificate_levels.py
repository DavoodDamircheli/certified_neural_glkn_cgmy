"""exp03 — cheap, heuristic eta_mass_indicator (NOT a certificate -- see
certification.mass_indicator) vs. direct sharp (eta_sharp) vs. iterative
guaranteed (eta_guar at a range of PCG iteration caps): effectivity,
runtime, PCG iteration count, and bracket tightness. Produces the
sharpness/effort trade-off figure (certificate quality vs. PCG iteration
count).

"Memory" (per the task) is reported as an analytic proxy (n_dof floats per
stored PCG vector x iterations), not a measured RSS delta -- measuring
process-level RSS deltas per call is noisy at this problem size (n_dof=66)
and would mostly measure Python/PyTorch overhead, not the certificate
computation itself. Runtime is measured directly (wall-clock, several
repeats).

Outputs: results/exp03_certificate_levels.h5,
figures/exp03_certificate_levels.png, tables/exp03_certificate_levels.tex.
"""

from __future__ import annotations

import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from certification.guaranteed_certificate import eta_guar as compute_eta_guar
from certification.guaranteed_certificate import eta_sharp as compute_eta_sharp
from certification.mass_indicator import eta_mass_indicator as compute_eta_mass_indicator
from certification.residuals import compute_all_residuals
from certification.riesz_solver import solve_riesz_pcg
from data.sampling import sample_sobol
from exp_common import (
    FIGURES_DIR,
    FINAL_MODEL_NAME,
    MAX_PCG_ITERATIONS,
    PRECONDITIONER,
    RESULTS_DIR,
    TABLES_DIR,
    fmt,
    get_box_and_stats,
    get_ops,
    load_models,
    predict,
    solve_reference,
    true_error_Ei,
    write_h5,
    write_latex_table,
)
from certification.constants import compute_stability_constants
from learning.interface import wrap_as_trajectory

N_TEST = 15
SEED = 424244
PCG_LEVELS = [1, 2, 5, 10, 20, 50, 100, 200]


def main() -> None:
    ops = get_ops()
    box, stats = get_box_and_stats()
    models = load_models(ops, stats)
    model = models[FINAL_MODEL_NAME]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_samples = sample_sobol(box, N_TEST, seed=SEED)

    # Per-query, per-level records.
    eta_guar_by_level = {k: [] for k in PCG_LEVELS}
    time_guar_by_level = {k: [] for k in PCG_LEVELS}
    bracket_width_by_level = {k: [] for k in PCG_LEVELS}  # single representative residual
    pcg_iters_actual_by_level = {k: [] for k in PCG_LEVELS}

    eta_mass_indicator_list, eta_sharp_list, E_i_list = [], [], []
    time_mass_indicator_list, time_sharp_list = [], []
    n_skipped = 0

    print(f"exp03: {N_TEST} queries x {len(PCG_LEVELS)} PCG levels...")
    for qi, (mu, payoff) in enumerate(test_samples):
        traj = solve_reference(mu)
        pred = predict(model, mu, ops, stats, traj.initial_coefficients, traj.time_grid)

        const = compute_stability_constants(mu, ops)
        if not const.is_spd:
            n_skipped += 1
            continue

        traj_pred = wrap_as_trajectory(traj, pred)
        residuals = compute_all_residuals(traj_pred, mu, ops)
        dt = traj.solver_metadata.dt
        E_i = true_error_Ei(pred, traj.coefficients, ops)

        t0 = time.perf_counter()
        em = compute_eta_mass_indicator(residuals, ops, dt, const.alpha_certified)
        time_mass_indicator_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        es = compute_eta_sharp(residuals, mu, ops, dt)
        time_sharp_list.append(time.perf_counter() - t0)

        eta_mass_indicator_list.append(em)
        eta_sharp_list.append(es.eta)
        E_i_list.append(E_i)

        # Representative single residual (final step) for the bracket-width panel.
        r_rep = residuals[-1]

        for k in PCG_LEVELS:
            t0 = time.perf_counter()
            eg = compute_eta_guar(residuals, mu, ops, dt, const.alpha_certified,
                                   max_pcg_iterations=k, preconditioner=PRECONDITIONER)
            time_guar_by_level[k].append(time.perf_counter() - t0)
            eta_guar_by_level[k].append(eg.eta)
            pcg_iters_actual_by_level[k].append(float(np.mean(eg.n_pcg_iterations)))

            bracket = solve_riesz_pcg(r_rep, mu, ops, const.alpha_certified,
                                       max_iter=k, preconditioner=PRECONDITIONER)
            bracket_width_by_level[k].append(bracket.upper_bound - bracket.lower_bound)

        if (qi + 1) % 5 == 0:
            print(f"  {qi + 1}/{N_TEST}")

    if n_skipped:
        print(f"NOTE: {n_skipped}/{N_TEST} queries skipped (S_h(mu) not SPD)")

    E_i_arr = np.array(E_i_list)
    eta_mass_indicator_arr = np.array(eta_mass_indicator_list)
    eta_sharp_arr = np.array(eta_sharp_list)
    n_dof = ops.n_dof

    eta_guar_mean = np.array([np.mean(eta_guar_by_level[k]) for k in PCG_LEVELS])
    time_guar_mean = np.array([np.mean(time_guar_by_level[k]) for k in PCG_LEVELS])
    bracket_width_mean = np.array([np.mean(bracket_width_by_level[k]) for k in PCG_LEVELS])
    pcg_iters_actual_mean = np.array([np.mean(pcg_iters_actual_by_level[k]) for k in PCG_LEVELS])
    I_eff_guar_mean = np.array([np.mean(np.array(eta_guar_by_level[k]) / E_i_arr) for k in PCG_LEVELS])

    mean_time_mass_indicator = float(np.mean(time_mass_indicator_list))
    mean_time_sharp = float(np.mean(time_sharp_list))
    mean_I_eff_mass_indicator = float(np.mean(eta_mass_indicator_arr / E_i_arr))
    mean_I_eff_sharp = float(np.mean(eta_sharp_arr / E_i_arr))

    print(f"\neta_mass_indicator (heuristic): mean time={mean_time_mass_indicator * 1000:.3f} ms, "
          f"mean I_eff={mean_I_eff_mass_indicator:.3e}, memory proxy={n_dof} floats")
    print(f"eta_sharp: mean time={mean_time_sharp * 1000:.3f} ms, mean I_eff={mean_I_eff_sharp:.3e}, "
          f"memory proxy={n_dof * n_dof} floats (dense S_h)")
    for k, t, ie, bw in zip(PCG_LEVELS, time_guar_mean, I_eff_guar_mean, bracket_width_mean):
        print(f"eta_guar(k={k:4d}): mean time={t * 1000:8.3f} ms, mean I_eff={ie:.3e}, "
              f"mean bracket width (final-step residual)={bw:.3e}")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    write_h5(
        RESULTS_DIR / "exp03_certificate_levels.h5",
        {
            "pcg_levels": np.array(PCG_LEVELS),
            "eta_guar_mean": eta_guar_mean, "time_guar_mean_s": time_guar_mean,
            "bracket_width_mean": bracket_width_mean, "pcg_iters_actual_mean": pcg_iters_actual_mean,
            "I_eff_guar_mean": I_eff_guar_mean,
            "eta_mass_indicator_mean": np.array([eta_mass_indicator_arr.mean()]),
            "eta_sharp_mean": np.array([eta_sharp_arr.mean()]),
            "time_mass_indicator_mean_s": np.array([mean_time_mass_indicator]),
            "time_sharp_mean_s": np.array([mean_time_sharp]),
        },
        attrs={"n_test": N_TEST, "seed": SEED, "model": FINAL_MODEL_NAME, "n_skipped_not_spd": n_skipped,
               "n_dof": n_dof, "preconditioner": PRECONDITIONER},
    )

    # ── Figure: certificate quality (I_eff) and bracket width vs PCG iters ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(PCG_LEVELS, I_eff_guar_mean, "o-", label=r"$I_{eff}(\eta_{guar})$")
    ax1.axhline(mean_I_eff_sharp, color="C1", ls="--", label=r"$I_{eff}(\eta_{sharp})$ (direct)")
    ax1.axhline(mean_I_eff_mass_indicator, color="C2", ls=":",
                label=r"$I_{eff}(\eta_{\mathrm{mass\ indicator}})$ (cheap, heuristic)")
    ax1.set_xscale("log")
    ax1.set_xlabel("max PCG iterations")
    ax1.set_ylabel("mean effectivity index")
    ax1.set_title("Certificate sharpness vs. PCG effort")
    ax1.legend(fontsize=8)

    ax2.plot(PCG_LEVELS, bracket_width_mean, "o-", color="C3")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("max PCG iterations")
    ax2.set_ylabel("mean bracket width (final-step residual)")
    ax2.set_title("Riesz bracket collapse vs. PCG effort")

    fig.suptitle(f"exp03: certificate levels ({FINAL_MODEL_NAME}, n={len(E_i_arr)} queries)")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp03_certificate_levels.png"
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    header = ["quantity", "mean time (ms)", "mean $I_{eff}$", "memory proxy (floats)"]
    rows = [
        ["eta_mass_indicator (cheap, heuristic, not a certificate)",
         f"{mean_time_mass_indicator * 1000:.3f}", fmt(mean_I_eff_mass_indicator), str(n_dof)],
        ["eta_sharp (direct)", f"{mean_time_sharp * 1000:.3f}", fmt(mean_I_eff_sharp), str(n_dof * n_dof)],
    ]
    for k, t, ie in zip(PCG_LEVELS, time_guar_mean, I_eff_guar_mean):
        rows.append([f"eta_guar (PCG k={k})", f"{t * 1000:.3f}", fmt(ie), str(n_dof * k)])
    tex_path = TABLES_DIR / "exp03_certificate_levels.tex"
    write_latex_table(tex_path, header, rows)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
