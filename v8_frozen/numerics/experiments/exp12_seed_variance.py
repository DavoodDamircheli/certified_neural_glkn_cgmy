"""exp12 -- Task 11 (working prompt): train the main model (DeepONet,
exp_common.FINAL_MODEL_NAME) with >=3 independent seeds and report
mean +/- std ACROSS SEEDS for: relative L2 error, energy error, pricing
error, effectivity, final certified workflow success rate, and runtime.

Terminology (pre-submission fixes, item 7.1): the "accepted" metric below
is `trace.accepted` from a FULL `pipeline.predict_certify_correct` call --
True whether the query was accepted directly OR after correction. This is
the **final certified workflow success rate**, NOT "direct NN acceptance
rate" (the fraction accepted with NO correction at all -- see
`experiments/exp11_tolerance_sweep.py`'s `direct_accept_rate`, a
genuinely different, narrower quantity). The label "acceptance rate" used
here previously was ambiguous between the two; renamed throughout this
script's table/printout. The ~98% figure this script reports and
exp11_tolerance_sweep.py's own `final_certified_success_rate` at a
nearby tolerance (its tolerance=5 point, close to this script's own
calibrated tolerance ~5.17) are the SAME quantity and land close
together, by construction -- exp11's much lower `direct_accept_rate`
(~53% at tolerance=5) is a different, narrower number and is not
expected to match either.

Does NOT touch checkpoints/deeponet_final.pt (the Stage H canonical
checkpoint every other exp01-11 script loads via exp_common.load_models)
-- each seed's model is trained fresh, in memory, here and never saved to
disk, the same convention exp05_ood.py already uses for its own
throwaway model. Trains on the SAME production dataset and the SAME
n_train=800 trajectories the canonical checkpoint used (read back from
data/production_dataset.h5; n_train=800 matches
checkpoints/deeponet_final.pt's own recorded "n_train" metadata field --
see scripts/train_final_models.py), so seed-to-seed variance is isolated
from any change in training data. SEEDS = (0, 1, 2): seed 0 deliberately
matches the canonical checkpoint's own training seed -- a sanity check,
since this script recomputes it via the identical architecture/recipe on
the identical data, so its held-out metrics should land close to (not
necessarily bit-identical to, since this script's own held-out test
queries differ from Stage H's original validation set) the canonical
checkpoint's.

Two different kinds of per-query metric, computed separately and combined
only at the seed-aggregate level:
  - relative L2 error, relative energy error, pricing error, effectivity:
    computed on each model's RAW prediction (exp_common.predict), exactly
    exp01/exp02's own established convention -- pure model-quality
    metrics, independent of any acceptance tolerance.
  - final certified workflow success rate, runtime: undefined without a
    tolerance, since they
    are workflow-level (pipeline.predict_certify_correct) properties, not
    model properties. A SINGLE tolerance, calibrated ONCE (median
    eta_guar_quick over a small calibration pass using the seed-0 model --
    exp08's own established calibration method), is applied uniformly to
    all 3 seeds' evaluation: calibrating separately per seed would let
    tolerance variation confound the seed-variance signal this task is
    actually asking for.

For each seed: aggregate (mean) each of the six metrics over the held-out
test set, giving 3 seed-level numbers per metric. Report mean +/- std
ACROSS those 3 numbers -- i.e. std here measures training-seed
sensitivity of the model's overall performance, not query-to-query
spread within one seed (that spread is what exp01/exp02's own per-query
statistics already report).

Outputs: results/exp12_seed_variance.h5, figures/exp12_seed_variance.png,
tables/exp12_seed_variance.tex.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import io as common_io
from data.dataset import read_trajectory
from data import payoff_family
from data.sampling import sample_sobol
from exp_common import (
    FIGURES_DIR,
    FIXED_MESH,
    MAX_PCG_ITERATIONS,
    PRECONDITIONER,
    REPO_ROOT,
    RESULTS_DIR,
    TABLES_DIR,
    TIME_GRID,
    compute_certificates,
    fmt,
    get_box_and_stats,
    get_ops,
    option_price,
    predict,
    relative_energy_error,
    relative_l2_error,
    solve_reference,
    true_error_Ei,
    write_h5,
    write_latex_table,
)
from learning.deeponet_coeff import DeepONetCoeffNet
from learning.losses import mass_weighted_loss
from learning.normalization import normalize_mu, normalize_psi, normalize_tau
from pipeline.predict_certify_correct import predict_certify_correct

torch.set_default_dtype(torch.float64)

DATASET_PATH = REPO_ROOT / "data" / "production_dataset.h5"
N_TRAIN = 800  # matches checkpoints/deeponet_final.pt's own recorded n_train
N_EPOCHS = 3000  # matches scripts/train_final_models.py's N_EPOCHS
LR = 2e-3
SEEDS = (0, 1, 2)  # Task 11: ">=3 independent seeds"; seed 0 matches the canonical checkpoint's own seed

N_TEST = 50
TEST_SEED = 121121  # disjoint from every other experiment's seed in this project

N_CALIBRATION = 20
CALIBRATION_SEED = 121122
QUICK_PCG_ITERATIONS = 5

DEEPONET_KWARGS = {"mu_dim": 6, "psi_dim": None, "latent_dim": 64,
                    "branch_hidden": (128, 128), "trunk_hidden": (64, 64)}


def load_train_samples(dataset_path: Path, n_samples: int):
    """Read back the first n_samples successful production-dataset
    trajectories -- same logic as scripts/generate_production_dataset.
    load_all_train_samples, reimplemented here to keep this experiment
    self-contained (experiments/ imports from src/, not scripts/, same as
    every other exp0X script)."""
    ids = common_io.list_groups(dataset_path, parent="samples")
    trajs = []
    for sid in sorted(ids, key=int):
        if int(sid) >= n_samples:
            continue
        traj = read_trajectory(dataset_path, int(sid))
        trajs.append((traj.parameter_vector, traj.payoff, traj))
    return trajs


def build_tensors(trajs, ops, stats):
    tau_l, tau_norm_l, mu_tilde_l, psi_l, c0_l, target_l = [], [], [], [], [], []
    for mu, payoff, traj in trajs:
        mu_tilde = normalize_mu(mu, stats)
        psi = normalize_psi(payoff_family.european_put(payoff.strike, ops), payoff.strike)
        for n, tau in enumerate(traj.time_grid):
            tau_l.append([tau]); tau_norm_l.append([normalize_tau(tau, mu.T)])
            mu_tilde_l.append(mu_tilde); psi_l.append(psi)
            c0_l.append(traj.initial_coefficients); target_l.append(traj.coefficients[n])
    return {
        "tau": torch.tensor(np.array(tau_l)), "tau_norm": torch.tensor(np.array(tau_norm_l)),
        "mu_tilde": torch.tensor(np.array(mu_tilde_l)), "psi": torch.tensor(np.array(psi_l)),
        "c0": torch.tensor(np.array(c0_l)), "target": torch.tensor(np.array(target_l)),
    }


def train_one_seed(seed: int, tensors, mass_matrix_dense, n_dof: int):
    kwargs = dict(DEEPONET_KWARGS, psi_dim=n_dof)
    torch.manual_seed(seed)
    net = DeepONetCoeffNet(**kwargs)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 1e-3)
    t0 = time.time()
    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = net(tensors["tau"], tensors["tau_norm"], tensors["mu_tilde"], tensors["psi"], tensors["c0"])
        loss = mass_weighted_loss(pred, tensors["target"], mass_matrix_dense)
        loss.backward()
        optimizer.step()
        scheduler.step()
    net.eval()
    return net, time.time() - t0, float(loss.item())


def main() -> None:
    ops = get_ops()
    box, stats = get_box_and_stats()
    n_dof = ops.n_dof

    print(f"exp12: loading {N_TRAIN} training trajectories from {DATASET_PATH}...")
    train_trajs = load_train_samples(DATASET_PATH, N_TRAIN)
    print(f"  loaded {len(train_trajs)} trajectories")
    tensors = build_tensors(train_trajs, ops, stats)
    mass_matrix_dense = torch.tensor(np.array([ops.apply_M(e) for e in np.eye(n_dof)]).T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_samples = sample_sobol(box, N_TEST, seed=TEST_SEED)
        calib_samples = sample_sobol(box, N_CALIBRATION, seed=CALIBRATION_SEED)

    print(f"exp12: solving {N_TEST} held-out test trajectories (shared across all seeds)...")
    test_queries = []
    for mu, payoff in test_samples:
        traj = solve_reference(mu)
        test_queries.append((mu, payoff, traj))

    models = {}
    train_times = {}
    train_losses = {}
    for seed in SEEDS:
        print(f"\nexp12: training DeepONet, seed={seed} ({N_EPOCHS} epochs, n_train={N_TRAIN})...")
        net, t_train, final_loss = train_one_seed(seed, tensors, mass_matrix_dense, n_dof)
        print(f"  ...done in {t_train:.1f}s, final train loss={final_loss:.4e}")
        models[seed] = net
        train_times[seed] = t_train
        train_losses[seed] = final_loss

    # ── Calibrate ONE tolerance (seed-0 model, exp08's method) used for
    #    every seed's acceptance-rate/runtime evaluation ──────────────────
    print(f"\nexp12: calibrating one shared acceptance tolerance ({N_CALIBRATION} queries, seed-0 model)...")
    calib_values = []
    for mu, payoff in calib_samples:
        time_grid = np.linspace(0.0, mu.T, TIME_GRID.n_steps + 1)
        trace = predict_certify_correct(
            mu, payoff, FIXED_MESH, time_grid, models[SEEDS[0]], stats, ops,
            tolerance=1e-12, quick_pcg_iterations=QUICK_PCG_ITERATIONS,
            max_pcg_iterations=MAX_PCG_ITERATIONS, preconditioner=PRECONDITIONER,
            run_correction=False,
        )
        if trace.is_spd:
            calib_values.append(trace.eta_final)
    calibrated_tol = float(np.median(calib_values)) if calib_values else float("nan")
    print(f"  calibrated tolerance (shared across all seeds): {calibrated_tol:.3e}")

    # ── Evaluate every seed's model on the SAME held-out test set ──────────
    metric_names = ["rel_l2", "rel_energy", "price_err", "effectivity", "accepted", "runtime"]
    per_seed_means = {m: [] for m in metric_names}
    per_seed_per_query = {seed: {m: [] for m in metric_names} for seed in SEEDS}

    for seed in SEEDS:
        print(f"\nexp12: evaluating seed={seed} on {N_TEST} held-out queries...")
        net = models[seed]
        for i, (mu, payoff, traj) in enumerate(test_queries):
            pred = predict(net, mu, ops, stats, traj.initial_coefficients, traj.time_grid)
            rel_l2 = relative_l2_error(pred, traj.coefficients, ops)
            rel_energy = relative_energy_error(pred, traj.coefficients, mu, ops)
            true_price = option_price(traj.coefficients[-1], ops)
            price_err = abs(option_price(pred[-1], ops) - true_price)

            cert = compute_certificates(traj, pred, mu, ops)
            E_i = true_error_Ei(pred, traj.coefficients, ops)
            effectivity = (cert.eta_guar / E_i) if (cert.is_spd and E_i > 0) else float("nan")

            trace = predict_certify_correct(
                mu, payoff, FIXED_MESH, traj.time_grid, net, stats, ops,
                tolerance=calibrated_tol, quick_pcg_iterations=QUICK_PCG_ITERATIONS,
                max_pcg_iterations=MAX_PCG_ITERATIONS, preconditioner=PRECONDITIONER,
            )

            per_seed_per_query[seed]["rel_l2"].append(rel_l2)
            per_seed_per_query[seed]["rel_energy"].append(rel_energy)
            per_seed_per_query[seed]["price_err"].append(price_err)
            per_seed_per_query[seed]["effectivity"].append(effectivity)
            per_seed_per_query[seed]["accepted"].append(float(trace.accepted))
            per_seed_per_query[seed]["runtime"].append(trace.t_total)

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{N_TEST}")

        for m in metric_names:
            vals = np.array(per_seed_per_query[seed][m])
            per_seed_means[m].append(float(np.nanmean(vals)))
        print(f"  seed={seed} means: " + ", ".join(f"{m}={per_seed_means[m][-1]:.4e}" for m in metric_names))

    # ── Mean +/- std ACROSS seeds (the task's actual deliverable) ──────────
    print("\n" + "=" * 78)
    print(f"exp12: mean +/- std across {len(SEEDS)} seeds (n_test={N_TEST}, calibrated_tol={calibrated_tol:.3e})")
    print("=" * 78)
    summary = {}
    for m in metric_names:
        vals = np.array(per_seed_means[m])
        summary[m] = (float(np.mean(vals)), float(np.std(vals)))
        print(f"  {m:12s}: {summary[m][0]:.4e} +/- {summary[m][1]:.4e}   (per-seed: "
              f"{', '.join(f'{v:.4e}' for v in vals)})")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    arrays = {"calibrated_tol": np.array([calibrated_tol])}
    for seed in SEEDS:
        for m in metric_names:
            arrays[f"seed_{seed}/{m}"] = np.array(per_seed_per_query[seed][m])
    write_h5(
        RESULTS_DIR / "exp12_seed_variance.h5", arrays,
        attrs={
            "seeds": np.array(SEEDS), "n_train": N_TRAIN, "n_epochs": N_EPOCHS, "n_test": N_TEST,
            "test_seed": TEST_SEED, "calibration_seed": CALIBRATION_SEED, "calibrated_tol": calibrated_tol,
            **{f"train_time_seed{seed}": train_times[seed] for seed in SEEDS},
            **{f"final_train_loss_seed{seed}": train_losses[seed] for seed in SEEDS},
            **{f"{m}_mean": summary[m][0] for m in metric_names},
            **{f"{m}_std": summary[m][1] for m in metric_names},
        },
    )

    # ── Figure: per-metric seed-to-seed spread ──────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, m in zip(axes.flat, metric_names):
        vals = per_seed_means[m]
        ax.bar([str(s) for s in SEEDS], vals, color="C0", alpha=0.75)
        ax.axhline(summary[m][0], color="k", linestyle="--", label=f"mean={summary[m][0]:.3e}")
        ax.set_title(m)
        ax.set_xlabel("seed")
        ax.legend(fontsize=7)
    fig.suptitle(f"exp12: DeepONet seed-to-seed variance ({len(SEEDS)} seeds, n_test={N_TEST})")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp12_seed_variance.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    header = ["metric", "mean", "std", "seed values"]
    metric_labels = {
        "rel_l2": "relative $L_2$ error", "rel_energy": "relative energy error",
        "price_err": "pricing error", "effectivity": "effectivity $\\eta_{guar}/E_i$",
        "accepted": "final certified workflow success rate", "runtime": "runtime (s)",
    }
    rows = []
    for m in metric_names:
        seed_vals = ", ".join(fmt(v) for v in per_seed_means[m])
        rows.append([metric_labels[m], fmt(summary[m][0]), fmt(summary[m][1]), seed_vals])
    tex_path = TABLES_DIR / "exp12_seed_variance.tex"
    write_latex_table(tex_path, header, rows)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
