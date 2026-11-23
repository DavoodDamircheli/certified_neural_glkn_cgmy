"""Stage H item 1: generate the production dataset in tranches, extending
only while the learning curve (validation error vs. dataset size) shows
real improvement.

Default tranche sizes are the production scale (1000, 5000, 10000+); pass
--tranche-sizes to override (e.g. for a smaller validation run of this same
pipeline -- see the run this script was actually invoked with in the Stage H
report, which used a reduced scale to fit an interactive session's
wall-clock budget; the logic and stopping rule are identical at any scale).

All trajectories -- production and the fixed validation set -- are solved
on ONE common mesh domain (same convention as Stage F/G: POD and any
cross-sample coefficient comparison require it).
"""

from __future__ import annotations

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cgmy.operators import OperatorCache
from cgmy.reference_solver import Mesh, TimeGrid
from data.dataset import generate_one_sample, write_manifest, write_sample
from data.sampling import load_parameter_box, sample_sobol
from learning.losses import mass_weighted_loss
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import normalize_mu, normalize_psi, normalize_tau, stats_from_box
from data import payoff_family

torch.set_default_dtype(torch.float64)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
FIXED_MESH = Mesh(N=64, p=3, x_min=float(np.log(100) - 8), x_max=float(np.log(100) + 8))
TIME_GRID = TimeGrid(n_steps=60)
MODE = "toeplitz"
N_EPOCHS = 3000
LR = 2e-3
VAL_SEED = 999_999  # far from any tranche seed, fixed validation set never grows


def _worker(mu, payoff):
    return generate_one_sample(mu, payoff, FIXED_MESH, TIME_GRID, mode=MODE)


def generate_tranche(output_path: Path, n_new: int, seed: int, start_index: int, workers: int | None):
    box = load_parameter_box(CONFIG_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = sample_sobol(box, n_new, seed=seed)

    n_ok = n_failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, mu, payoff): i for i, (mu, payoff) in enumerate(samples)}
        for fut in as_completed(futures):
            i = futures[fut]
            result = fut.result()
            write_sample(output_path, start_index + i, "train", result)
            if result.ok:
                n_ok += 1
            else:
                n_failed += 1
    return n_ok, n_failed


def build_val_set(n_val: int, seed: int = VAL_SEED):
    box = load_parameter_box(CONFIG_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = sample_sobol(box, n_val, seed=seed)
    return [(mu, payoff, generate_one_sample(mu, payoff, FIXED_MESH, TIME_GRID, mode=MODE).traj)
            for mu, payoff in samples]


def load_all_train_samples(output_path: Path, n_samples: int):
    """Read back every successful sample from the dataset file (indices 0..n_samples-1)."""
    from common import io as common_io
    from data.dataset import read_trajectory

    ids = common_io.list_groups(output_path, parent="samples")
    trajs = []
    for sid in sorted(ids, key=int):
        if int(sid) >= n_samples:
            continue
        traj = read_trajectory(output_path, int(sid))
        trajs.append((traj.parameter_vector, traj.payoff, traj))
    return trajs


def train_mlp(train_trajs, ops, stats, seed=0):
    tau_list, tau_norm_list, mu_tilde_list, psi_list, c0_list, target_list = [], [], [], [], [], []
    for mu, payoff, traj in train_trajs:
        mu_tilde = normalize_mu(mu, stats)
        psi = normalize_psi(payoff_family.european_put(payoff.strike, ops), payoff.strike)
        for n, tau in enumerate(traj.time_grid):
            tau_list.append([tau])
            tau_norm_list.append([normalize_tau(tau, mu.T)])
            mu_tilde_list.append(mu_tilde)
            psi_list.append(psi)
            c0_list.append(traj.initial_coefficients)
            target_list.append(traj.coefficients[n])

    tau_t = torch.tensor(np.array(tau_list))
    tau_norm_t = torch.tensor(np.array(tau_norm_list))
    mu_tilde_t = torch.tensor(np.array(mu_tilde_list))
    psi_t = torch.tensor(np.array(psi_list))
    c0_t = torch.tensor(np.array(c0_list))
    target_t = torch.tensor(np.array(target_list))

    n_dof = ops.n_dof
    mass_matrix_dense = torch.tensor(np.array([ops.apply_M(e) for e in np.eye(n_dof)]).T)

    torch.manual_seed(seed)
    net = MLPCoeffNet(mu_dim=6, psi_dim=n_dof, hidden_dims=(128, 128, 128))
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 1e-3)

    t0 = time.time()
    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = net(tau_t, tau_norm_t, mu_tilde_t, psi_t, c0_t)
        loss = mass_weighted_loss(pred, target_t, mass_matrix_dense)
        loss.backward()
        optimizer.step()
        scheduler.step()
    train_time = time.time() - t0
    return net, train_time, mass_matrix_dense


def evaluate_mlp(net, val_trajs, ops, stats):
    l2_errs = []
    for mu, payoff, traj in val_trajs:
        mu_tilde = normalize_mu(mu, stats)
        psi = payoff_family.european_put(payoff.strike, ops)
        pred = net.predict_trajectory(mu_tilde, psi, traj.initial_coefficients, traj.time_grid, mu.T)
        diff = pred - traj.coefficients
        l2_errs.extend(np.sqrt(d @ ops.apply_M(d)) for d in diff)
    return float(np.mean(l2_errs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tranche-sizes", type=str, default="1000,5000,10000",
                         help="Comma-separated CUMULATIVE tranche totals.")
    parser.add_argument("--n-val", type=int, default=30)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "data" / "production_dataset.h5"))
    parser.add_argument("--flatten-threshold", type=float, default=0.05,
                         help="Stop extending if relative improvement falls below this.")
    args = parser.parse_args()

    tranche_sizes = [int(s) for s in args.tranche_sizes.split(",")]
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    box = load_parameter_box(CONFIG_PATH)
    stats = stats_from_box(box)
    ops = OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)

    print(f"building fixed validation set (n={args.n_val}, never grows)...")
    val_trajs = build_val_set(args.n_val)

    prev_count = 0
    errors = []
    stopped_early = False
    for tranche_idx, target in enumerate(tranche_sizes):
        n_new = target - prev_count
        print(f"\n=== tranche {tranche_idx}: extending {prev_count} -> {target} "
              f"(+{n_new} new trajectories) ===")
        t0 = time.time()
        n_ok, n_failed = generate_tranche(output_path, n_new, seed=100 + tranche_idx,
                                           start_index=prev_count, workers=args.workers)
        print(f"  generated {n_ok} ok, {n_failed} failed, in {time.time() - t0:.1f}s")
        prev_count = target

        print(f"  loading all {prev_count} accumulated training trajectories...")
        train_trajs = load_all_train_samples(output_path, prev_count)
        print(f"  training MLP on {len(train_trajs)} trajectories...")
        net, train_time, _ = train_mlp(train_trajs, ops, stats)
        val_err = evaluate_mlp(net, val_trajs, ops, stats)
        errors.append(val_err)
        print(f"  train_time={train_time:.1f}s  validation L2 error={val_err:.4e}")

        if len(errors) >= 2:
            rel_improvement = (errors[-2] - errors[-1]) / errors[-2]
            print(f"  relative improvement over previous tranche: {rel_improvement:.1%}")
            if rel_improvement < args.flatten_threshold:
                print(f"  improvement below --flatten-threshold={args.flatten_threshold:.0%}: "
                      f"curve has flattened, stopping here.")
                stopped_early = True
                break

    write_manifest(output_path, {"train": prev_count}, n_failures=0,
                    config_snapshot=CONFIG_PATH.read_text(), seed=100)

    sizes = tranche_sizes[:len(errors)]
    print("\n" + "=" * 60)
    print("LEARNING CURVE (dataset size -> validation L2 error)")
    print("=" * 60)
    for n, e in zip(sizes, errors):
        print(f"  N={n:6d}  val_L2={e:.4e}")
    print(f"stopped early: {stopped_early}")

    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot(sizes, errors, "o-")
    plt.xlabel("training dataset size (trajectories)")
    plt.ylabel("validation L2 error (mass-weighted)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Stage H learning curve")
    plt.tight_layout()
    fig_path = figures_dir / "learning_curve.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nwrote {fig_path}")

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    import csv
    csv_path = results_dir / "learning_curve.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_size", "val_l2_error"])
        w.writerows(zip(sizes, errors))
    print(f"wrote {csv_path}")
    print(f"\nfinal dataset: {output_path} ({prev_count} trajectories)")


if __name__ == "__main__":
    main()
