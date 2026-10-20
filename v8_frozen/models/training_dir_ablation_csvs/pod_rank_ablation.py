"""Gate E4: POD rank ablation at ranks {8, 16, 32, 48, full available rank}.

Reuses the real production dataset (data/production_dataset.h5, 800
Sobol-sampled trajectories, seed 0 -- same file scripts/train_final_models.py
trained the committed pod_final.npz from) and the same fixed validation set
(scripts/generate_production_dataset.py:build_val_set, VAL_SEED=999_999, 30
queries) -- not synthetic data, and not a resample of either set.

For each rank, reports SEPARATELY (never conflated):
  - mass_weighted_projection_error: ||snapshot - projection||_M, the best-
    approximation floor a rank can achieve regardless of dynamics (computed
    directly from the SVD, not from running any CN recurrence).
  - reduced_initial_error: the IC-projection error specific to POD (c0 is
    NOT reproduced exactly by POD, unlike every neural model in this
    project -- this is the "irreducible reduced-space projection error"
    the roadmap itself warns about).
  - reduced_dynamics_error: full-trajectory error once the reduced CN
    recurrence actually runs (>= the projection floor, since running the
    dynamics in a truncated subspace adds its own propagated error).
  - runtime_s: wall time to fit the basis at that rank (SVD is shared
    across ranks -- only the truncation differs -- so this is amortized/
    documented explicitly, not re-run from scratch for honesty).

Full-rank recovery: at rank = n_dof (66, the largest a 66-dim coefficient
space can support -- see module docstring in src/learning/pod.py, U from
SVD(mean-centered snapshots) has at most min(n_dof, n_snapshots) non-trivial
columns, and n_snapshots >> n_dof here), POD should recover the identical
discrete reference to near machine precision (zero truncation).

Run: PYTHONPATH=src:experiments python training/pod_rank_ablation.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cgmy.operators import OperatorCache  # noqa: E402
from data.sampling import load_mesh_and_time_grid, load_parameter_box  # noqa: E402
from learning.normalization import stats_from_box  # noqa: E402
from learning.pod import PODBasis, PODGalerkinROM, compute_pod_basis, project_onto_basis, reconstruct_from_basis  # noqa: E402
from generate_production_dataset import build_val_set, load_all_train_samples  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
DATASET_PATH = REPO_ROOT / "data" / "production_dataset.h5"
N_TRAIN = 800  # same as the committed pod_final.npz (checkpoints/pod_final.npz: n_train=800)
N_VAL = 30
RANKS = [8, 16, 32, 48]  # "full" appended below once n_dof is known


def mass_norm(v: np.ndarray, ops: OperatorCache) -> float:
    return float(np.sqrt(v @ ops.apply_M(v)))


def main():
    FIXED_MESH, TIME_GRID = load_mesh_and_time_grid(CONFIG_PATH)
    box = load_parameter_box(CONFIG_PATH)
    stats = stats_from_box(box)
    ops = OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)
    n_dof = ops.n_dof

    print(f"loading {N_TRAIN} training trajectories from {DATASET_PATH} ...")
    train_trajs = load_all_train_samples(DATASET_PATH, N_TRAIN)
    print(f"building {N_VAL}-query fixed validation set (VAL_SEED=999_999) ...")
    val_trajs = build_val_set(N_VAL)

    snapshots = np.concatenate([t.coefficients for _, _, t in train_trajs], axis=0)
    n_snapshots = snapshots.shape[0]
    full_rank = min(n_dof, n_snapshots - 1)
    ranks = RANKS + ([full_rank] if full_rank not in RANKS else [])
    print(f"snapshots: {n_snapshots} rows, n_dof={n_dof}, full_rank={full_rank}")

    t0 = time.perf_counter()
    mean = snapshots.mean(axis=0)
    X = (snapshots - mean).T
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    svd_time = time.perf_counter() - t0
    print(f"one shared SVD computed in {svd_time:.2f}s (amortized across all ranks below)")

    rows = []
    for rank in sorted(set(ranks)):
        t0 = time.perf_counter()
        basis = PODBasis(modes=U[:, :rank], singular_values=S[:rank], mean=mean)

        # Mass-weighted best-projection error over ALL snapshots (the
        # theoretical floor for this rank, independent of any dynamics).
        proj_errs = []
        for i in range(0, n_snapshots, 500):  # chunked, avoid materializing huge residual arrays at once
            chunk = snapshots[i:i + 500]
            reduced = (chunk - mean) @ basis.modes
            recon = mean + reduced @ basis.modes.T
            resid = chunk - recon
            proj_errs.extend(mass_norm(r, ops) for r in resid)
        mass_weighted_projection_error = float(np.mean(proj_errs))

        rom = PODGalerkinROM(basis, ops, stats=stats)

        reduced_initial_errs = []
        reduced_dynamics_errs = []
        for mu, payoff, traj in val_trajs:
            from learning.normalization import normalize_mu
            import data.payoff_family as payoff_family
            mu_tilde = normalize_mu(mu, stats)
            psi = payoff_family.european_put(payoff.strike, ops)
            pred = rom.predict_trajectory(mu_tilde, psi, traj.initial_coefficients, traj.time_grid, mu.T)
            reduced_initial_errs.append(mass_norm(pred[0] - traj.coefficients[0], ops))
            full_errs = [mass_norm(pred[n] - traj.coefficients[n], ops) for n in range(len(traj.time_grid))]
            reduced_dynamics_errs.append(float(np.mean(full_errs)))

        runtime_s = time.perf_counter() - t0
        row = dict(
            rank=rank,
            mass_weighted_projection_error=mass_weighted_projection_error,
            reduced_initial_error=float(np.mean(reduced_initial_errs)),
            reduced_initial_error_max=float(np.max(reduced_initial_errs)),
            reduced_dynamics_error=float(np.mean(reduced_dynamics_errs)),
            reduced_dynamics_error_max=float(np.max(reduced_dynamics_errs)),
            runtime_s=runtime_s,
            is_full_rank=(rank == full_rank),
        )
        rows.append(row)
        print(f"rank={rank:3d}  proj_floor={mass_weighted_projection_error:.3e}  "
              f"IC_err={row['reduced_initial_error']:.3e}  "
              f"dyn_err={row['reduced_dynamics_error']:.3e}  ({runtime_s:.2f}s)")

    out_path = REPO_ROOT / "training" / "pod_rank_ablation.csv"
    with open(out_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    sv_path = REPO_ROOT / "training" / "pod_singular_values_full.npz"
    np.savez(sv_path, singular_values=S, n_snapshots=n_snapshots, n_dof=n_dof, n_train=N_TRAIN)

    print(f"\nwrote {out_path}")
    print(f"wrote {sv_path} (full singular-value spectrum, {len(S)} values)")

    full_row = next(r for r in rows if r["is_full_rank"])
    print(f"\nFull-rank ({full_rank}) recovery check: "
          f"mass_weighted_projection_error={full_row['mass_weighted_projection_error']:.3e} "
          f"(should be ~machine precision if POD is working correctly)")


if __name__ == "__main__":
    main()
