"""exp09 -- warm start: for queries that fail the quick guaranteed
(eta_guar, capped PCG) certificate and so genuinely need correction
(exp08's "% needing correction" bucket -- screened here via
predict_certify_correct with tolerance=1e-12, forcing eta_guar_quick to
actually run rather than being skipped by the default-disabled
eta_mass_indicator early-reject path), compare
correction.krylov.correct_trajectory warm-started by the neural prediction
against the same correction run cold (GMRES's default zero start).

Even a model whose raw output fails certification is still valuable if it
cuts correction cost -- this experiment measures exactly that: Krylov
iteration count, wall-clock, final residual, and final certificate, warm
vs. cold, same mu, same mesh, same GMRES tolerance/restart, same process.

Outputs: results/exp09_warm_start.h5, figures/exp09_warm_start.png,
tables/exp09_warm_start.tex.
"""

from __future__ import annotations

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from certification.constants import compute_stability_constants
from certification.mass_indicator import eta_mass_indicator as compute_eta_mass_indicator
from certification.residuals import compute_all_residuals, residual_norm
from correction.krylov import compute_initial_coefficients, correct_trajectory
from data.sampling import sample_sobol
from exp_common import (
    FIGURES_DIR,
    FINAL_MODEL_NAME,
    MAX_PCG_ITERATIONS,
    PAYOFF,
    PRECONDITIONER,
    RESULTS_DIR,
    TABLES_DIR,
    TIME_GRID,
    FIXED_MESH,
    fmt,
    get_box_and_stats,
    get_ops,
    load_models,
    write_h5,
    write_latex_table,
)
from learning.normalization import normalize_mu
from data import payoff_family
from pipeline.predict_certify_correct import build_mesh_info, build_skeleton_trajectory, predict_certify_correct

N_CANDIDATES = 60  # queries screened; only those needing correction are used
SEED = 90900
QUICK_PCG_ITERATIONS = 5


def _time_grid_for(mu) -> np.ndarray:
    return np.linspace(0.0, mu.T, TIME_GRID.n_steps + 1)


def main() -> None:
    ops = get_ops()
    box, stats = get_box_and_stats()
    models = load_models(ops, stats)
    model = models[FINAL_MODEL_NAME]
    mesh_info = build_mesh_info(FIXED_MESH)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        candidates = sample_sobol(box, N_CANDIDATES, seed=SEED)

    print(f"exp09: screening {N_CANDIDATES} candidates for ones genuinely needing correction...")
    needs_correction = []
    for mu, payoff in candidates:
        trace = predict_certify_correct(
            mu, payoff, FIXED_MESH, _time_grid_for(mu), model, stats, ops,
            tolerance=1e-12, quick_pcg_iterations=QUICK_PCG_ITERATIONS,
            max_pcg_iterations=MAX_PCG_ITERATIONS, preconditioner=PRECONDITIONER,
            run_correction=False,
        )
        if trace.is_spd and trace.branch == "needs_correction_not_run":
            needs_correction.append((mu, payoff))
    print(f"  {len(needs_correction)}/{N_CANDIDATES} genuinely need correction "
          f"(fail the quick guaranteed certificate)")

    if not needs_correction:
        print("no candidates needed correction at this tolerance/model -- nothing to compare. "
              "(Reporting this honestly rather than fabricating a comparison.)")
        write_h5(RESULTS_DIR / "exp09_warm_start.h5", {}, attrs={"n_candidates": N_CANDIDATES, "n_needing_correction": 0})
        return

    warm_iters_total, cold_iters_total = [], []
    warm_time, cold_time = [], []
    warm_final_resid, cold_final_resid = [], []
    warm_final_eta_mass_indicator, cold_final_eta_mass_indicator = [], []

    print(f"\nexp09: running warm-vs-cold correction on {len(needs_correction)} queries...")
    for i, (mu, payoff) in enumerate(needs_correction):
        time_grid = _time_grid_for(mu)
        c0, _, _ = compute_initial_coefficients(mu, payoff, mesh_info, ops)
        mu_tilde = normalize_mu(mu, stats)
        psi = payoff_family.european_put(payoff.strike, ops)
        pred_coeffs = model.predict_trajectory(mu_tilde, psi, c0, time_grid, mu.T)

        const = compute_stability_constants(mu, ops)
        dt = mu.T / TIME_GRID.n_steps

        warm = correct_trajectory(mu, payoff, mesh_info, time_grid, ops,
                                   seed_coefficients=pred_coeffs, warm_start=True)
        cold = correct_trajectory(mu, payoff, mesh_info, time_grid, ops,
                                   seed_coefficients=None, warm_start=False)

        for label, result, iters_acc, time_acc, resid_acc, eta_acc in [
            ("warm", warm, warm_iters_total, warm_time, warm_final_resid, warm_final_eta_mass_indicator),
            ("cold", cold, cold_iters_total, cold_time, cold_final_resid, cold_final_eta_mass_indicator),
        ]:
            traj = build_skeleton_trajectory(mu, payoff, mesh_info, time_grid, result.coefficients)
            residuals = compute_all_residuals(traj, mu, ops)
            max_r = max(residual_norm(residuals[n], ops, traj.solver_metadata.bc_dofs)
                        for n in range(residuals.shape[0]))
            eta_mass_indicator = compute_eta_mass_indicator(residuals, ops, dt, const.alpha_certified)
            iters_acc.append(sum(result.n_iterations_per_step))
            time_acc.append(result.total_time_s)
            resid_acc.append(max_r)
            eta_acc.append(eta_mass_indicator)

        print(f"  [{i+1}/{len(needs_correction)}] warm: {sum(warm.n_iterations_per_step)} total iters, "
              f"{warm.total_time_s*1000:.1f}ms  |  cold: {sum(cold.n_iterations_per_step)} total iters, "
              f"{cold.total_time_s*1000:.1f}ms")

    warm_iters_total = np.array(warm_iters_total)
    cold_iters_total = np.array(cold_iters_total)
    warm_time = np.array(warm_time)
    cold_time = np.array(cold_time)
    warm_final_resid = np.array(warm_final_resid)
    cold_final_resid = np.array(cold_final_resid)
    warm_final_eta_mass_indicator = np.array(warm_final_eta_mass_indicator)
    cold_final_eta_mass_indicator = np.array(cold_final_eta_mass_indicator)

    iter_reduction = 1.0 - warm_iters_total.mean() / cold_iters_total.mean()
    time_reduction = 1.0 - warm_time.mean() / cold_time.mean()

    print(f"\nmean total GMRES iterations: warm={warm_iters_total.mean():.1f}  "
          f"cold={cold_iters_total.mean():.1f}  (reduction={100*iter_reduction:.1f}%)")
    print(f"mean correction wall-clock:  warm={warm_time.mean()*1000:.2f}ms  "
          f"cold={cold_time.mean()*1000:.2f}ms  (reduction={100*time_reduction:.1f}%)")
    print(f"mean final residual norm:    warm={warm_final_resid.mean():.3e}  cold={cold_final_resid.mean():.3e}")
    print(f"mean final eta_mass_indicator (heuristic): warm={warm_final_eta_mass_indicator.mean():.3e}  "
          f"cold={cold_final_eta_mass_indicator.mean():.3e}")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    write_h5(
        RESULTS_DIR / "exp09_warm_start.h5",
        {
            "warm_iters_total": warm_iters_total, "cold_iters_total": cold_iters_total,
            "warm_time_s": warm_time, "cold_time_s": cold_time,
            "warm_final_residual": warm_final_resid, "cold_final_residual": cold_final_resid,
            "warm_final_eta_mass_indicator": warm_final_eta_mass_indicator,
            "cold_final_eta_mass_indicator": cold_final_eta_mass_indicator,
        },
        attrs={
            "n_candidates": N_CANDIDATES, "n_needing_correction": len(needs_correction),
            "model": FINAL_MODEL_NAME, "seed": SEED,
            "mean_iter_reduction_pct": 100 * iter_reduction, "mean_time_reduction_pct": 100 * time_reduction,
        },
    )

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    x = np.arange(len(needs_correction))
    width = 0.35
    axes[0].bar(x - width/2, cold_iters_total, width, label="cold", color="C3", alpha=0.75)
    axes[0].bar(x + width/2, warm_iters_total, width, label="warm (neural)", color="C0", alpha=0.75)
    axes[0].set_xlabel("query"); axes[0].set_ylabel("total GMRES iterations")
    axes[0].set_title("exp09: iteration count, warm vs. cold")
    axes[0].legend()

    axes[1].scatter(cold_time * 1000, warm_time * 1000, color="C2", alpha=0.7)
    lims = [0, max(cold_time.max(), warm_time.max()) * 1000 * 1.05]
    axes[1].plot(lims, lims, "k--", alpha=0.5, label="warm = cold")
    axes[1].set_xlabel("cold correction wall-clock (ms)")
    axes[1].set_ylabel("warm correction wall-clock (ms)")
    axes[1].set_title("exp09: wall-clock, warm vs. cold")
    axes[1].legend()

    axes[2].boxplot([cold_iters_total, warm_iters_total], tick_labels=["cold", "warm"])
    axes[2].set_ylabel("total GMRES iterations")
    axes[2].set_title(f"exp09: {100*iter_reduction:.1f}% mean iteration reduction")

    fig.suptitle(f"exp09: warm vs. cold correction ({FINAL_MODEL_NAME}, n={len(needs_correction)})")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp09_warm_start.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    header = ["quantity", "warm (neural)", "cold"]
    rows = [
        ["mean total GMRES iterations", fmt(warm_iters_total.mean()), fmt(cold_iters_total.mean())],
        ["mean wall-clock (ms)", fmt(warm_time.mean()*1000), fmt(cold_time.mean()*1000)],
        ["mean final residual norm", fmt(warm_final_resid.mean()), fmt(cold_final_resid.mean())],
        ["mean final eta\\_mass\\_indicator (heuristic)",
         fmt(warm_final_eta_mass_indicator.mean()), fmt(cold_final_eta_mass_indicator.mean())],
    ]
    tex_path = TABLES_DIR / "exp09_warm_start.tex"
    write_latex_table(tex_path, header, rows)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
