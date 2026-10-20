"""exp01 — interpolation: unseen interior parameter combinations.

Compares MLP, coefficient DeepONet, POD-Galerkin, and projected FNO (the
four models actually trained in Stage H) against the reference solver on
held-out interior Sobol points (never seen during training -- a fresh seed
disjoint from Stage H's training/validation seeds). Reports relative L2,
relative energy, and option-price error per model.

Payoff scope note: every query uses the European put (fixed strike=100),
the only payoff solve_cgmy can produce a reference trajectory for (see
cgmy/reference_solver.py's module docstring) -- "payoff combinations" here
means varying mu only, not payoff family, for that reason.

Scale note: N_TEST=100 interior points (Task 11, working prompt: "main
interpolation test: N_test >= 100 (prefer 200-500)" -- 100 is the floor
this task requires; 200-500 was not reached, for the same
interactive-session wall-clock reasons already flagged throughout Stages
G-H). Change N_TEST below to rerun larger.

Outputs: results/exp01_interpolation.h5, figures/exp01_interpolation.png,
tables/exp01_interpolation.tex.
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.sampling import sample_sobol
from exp_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    fmt,
    get_box_and_stats,
    get_ops,
    load_models,
    option_price,
    predict,
    relative_energy_error,
    relative_l2_error,
    solve_reference,
    write_h5,
    write_latex_table,
)

N_TEST = 100
SEED = 424242  # disjoint from Stage H's training (seed=100+) and validation (999_999) seeds
MODEL_ORDER = ["MLP", "DeepONet", "POD", "ProjectedFNO"]


def main() -> None:
    ops = get_ops()
    box, stats = get_box_and_stats()
    models = load_models(ops, stats)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_samples = sample_sobol(box, N_TEST, seed=SEED)

    rel_l2 = {name: [] for name in MODEL_ORDER}
    rel_energy = {name: [] for name in MODEL_ORDER}
    price_err = {name: [] for name in MODEL_ORDER}
    mus_arr = []

    print(f"exp01: evaluating {N_TEST} interior interpolation queries x {len(MODEL_ORDER)} models...")
    for i, (mu, payoff) in enumerate(test_samples):
        traj = solve_reference(mu)
        mus_arr.append([mu.C, mu.G, mu.M, mu.Y, mu.r, mu.T])
        true_price = option_price(traj.coefficients[-1], ops)
        for name in MODEL_ORDER:
            pred = predict(models[name], mu, ops, stats, traj.initial_coefficients, traj.time_grid)
            rel_l2[name].append(relative_l2_error(pred, traj.coefficients, ops))
            rel_energy[name].append(relative_energy_error(pred, traj.coefficients, mu, ops))
            price_err[name].append(abs(option_price(pred[-1], ops) - true_price))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_TEST}")

    mus_arr = np.array(mus_arr)

    # ── HDF5 ─────────────────────────────────────────────────────────────
    arrays = {"mu": mus_arr}
    for name in MODEL_ORDER:
        arrays[f"rel_l2_{name}"] = np.array(rel_l2[name])
        arrays[f"rel_energy_{name}"] = np.array(rel_energy[name])
        arrays[f"price_err_{name}"] = np.array(price_err[name])
    write_h5(RESULTS_DIR / "exp01_interpolation.h5", arrays, attrs={"n_test": N_TEST, "seed": SEED})

    # ── Figure: box plots of relative L2 / energy / price error per model ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (metric_name, data) in zip(axes, [
        ("relative L2 error", rel_l2), ("relative energy error", rel_energy), ("option-price error", price_err)
    ]):
        ax.boxplot([data[name] for name in MODEL_ORDER], tick_labels=MODEL_ORDER, showfliers=True)
        ax.set_yscale("log")
        ax.set_title(metric_name)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"exp01: interpolation errors across {N_TEST} unseen interior points")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp01_interpolation.png"
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    header = ["model", "median rel L2", "mean rel L2", "median rel energy", "mean rel energy", "median price err"]
    rows = []
    for name in MODEL_ORDER:
        rows.append([
            name,
            fmt(np.median(rel_l2[name])), fmt(np.mean(rel_l2[name])),
            fmt(np.median(rel_energy[name])), fmt(np.mean(rel_energy[name])),
            fmt(np.median(price_err[name])),
        ])
    tex_path = TABLES_DIR / "exp01_interpolation.tex"
    write_latex_table(tex_path, header, rows)
    print(f"wrote {tex_path}")

    print("\n" + "=" * 90)
    print(f"{'model':14s} {'median relL2':>14} {'mean relL2':>14} {'median relE':>14} {'mean relE':>14} {'median $err':>14}")
    for name in MODEL_ORDER:
        print(f"{name:14s} {np.median(rel_l2[name]):14.3e} {np.mean(rel_l2[name]):14.3e} "
              f"{np.median(rel_energy[name]):14.3e} {np.mean(rel_energy[name]):14.3e} "
              f"{np.median(price_err[name]):14.3e}")


if __name__ == "__main__":
    main()
