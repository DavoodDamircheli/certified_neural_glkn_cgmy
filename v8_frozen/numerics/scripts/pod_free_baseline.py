"""v4 Phase 3: repaired (free-coordinate, mass-weighted) POD ROM baseline --
CGMY_V4_REMAINING_WORK.md item 3.

Reuses the SAME training/validation data the legacy ablation
(training/pod_rank_ablation.py) uses -- data/production_dataset.h5 (800
Sobol trajectories) + the same fixed 30-query validation set
(generate_production_dataset.build_val_set, VAL_SEED=999_999) -- for a fair
apples-to-apples comparison against the retained legacy numbers (not a
resample, not synthetic data). No neural training happens here (no
gradient steps, no optimizer) -- `generate_one_sample`/`build_val_set` only
run reference Galerkin solves, the same thing every experiment script in
this project does to get a validation trajectory.

Reports the projection floor, initial error, and dynamics error SEPARATELY
at ranks {8, 16, 32, 48, 62 (=n_free, full)} -- never conflated -- plus a
dedicated full-rank-only recovery check against a DIRECT (non-GMRES) dense
solve of the free CN system (near machine precision, per the task file's
decisive test), separate from the GMRES-based rank-sweep numbers (which
inherit the reference solver's own ~1e-10 GMRES-tolerance noise floor, see
reports/v4_overnight_log.md's Phase 1 Check 2 finding).

Run: PYTHONPATH=src:experiments:scripts python3 scripts/pod_free_baseline.py
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cgmy.operators import OperatorCache  # noqa: E402
from data.sampling import load_mesh_and_time_grid  # noqa: E402
from certification.constrained import (  # noqa: E402
    FreeOperatorCache,
    free_space_for_mesh,
    restrict,
    solve_free_cn_trajectory_direct,
)
from learning.pod_free import FreePODGalerkinROM, compute_free_pod_basis, oracle_mass_projection_error  # noqa: E402
from generate_production_dataset import build_val_set, load_all_train_samples  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
DATASET_PATH = REPO_ROOT / "data" / "production_dataset.h5"
N_TRAIN = 800  # same as training/pod_rank_ablation.py and checkpoints/pod_final.npz
N_VAL = 30
RANKS = [8, 16, 32, 48]

RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = REPO_ROOT / "results" / "v4_constrained_free" / f"pod_free_{RUN_STAMP}"


def mass_norm_free(v_free: np.ndarray, free_ops: FreeOperatorCache) -> float:
    return float(np.sqrt(max(v_free @ free_ops.apply_M(v_free), 0.0)))


def main() -> None:
    FIXED_MESH, TIME_GRID = load_mesh_and_time_grid(CONFIG_PATH)
    ops = OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)

    print(f"loading {N_TRAIN} training trajectories from {DATASET_PATH} ...")
    train_trajs = load_all_train_samples(DATASET_PATH, N_TRAIN)
    print(f"building {N_VAL}-query fixed validation set (VAL_SEED=999_999, same as legacy ablation) ...")
    val_trajs = build_val_set(N_VAL)

    mesh_info = train_trajs[0][2].mesh_information
    fs = free_space_for_mesh(mesh_info)
    free_ops = FreeOperatorCache(ops, fs)
    n_free = fs.n_free
    print(f"n_dof={fs.n_dof}, n_free={n_free}")

    # Homogeneous free snapshots: z(tau) = restrict(fs, c(tau)) -- no
    # lifting to subtract, restrict already excludes the boundary rows.
    snapshots_free = np.concatenate(
        [t.coefficients[:, fs.free_idx] for _, _, t in train_trajs], axis=0
    )
    n_snapshots = snapshots_free.shape[0]
    full_rank = min(n_free, n_snapshots - 1)
    ranks = sorted(set(RANKS + [full_rank]))
    print(f"snapshots: {n_snapshots} rows, n_free={n_free}, full_rank={full_rank}")

    rows = []
    for rank in ranks:
        t0 = time.perf_counter()
        basis = compute_free_pod_basis(snapshots_free, free_ops, rank)
        # Check: V^T M_I V == I (mass-orthonormality actually achieved).
        VtMV = basis.modes.T @ np.array([free_ops.apply_M(basis.modes[:, j]) for j in range(rank)]).T
        ortho_err = float(np.max(np.abs(VtMV - np.eye(rank))))

        proj_errs = [oracle_mass_projection_error(z, basis, free_ops) for z in snapshots_free[::37]]
        projection_floor = float(np.mean(proj_errs))

        rom = FreePODGalerkinROM(basis)
        ic_errs, dyn_errs = [], []
        for mu, payoff, traj in val_trajs:
            z_true = traj.coefficients[:, fs.free_idx]
            z0 = z_true[0]
            alpha0 = rom.project_initial(z0, free_ops)
            z0_recon = basis.modes @ alpha0
            ic_errs.append(mass_norm_free(z0_recon - z0, free_ops))

            z_rom = rom.solve_reduced_trajectory(
                mu, payoff, mesh_info, traj.time_grid, fs, ops, free_ops, z0
            )
            step_errs = [mass_norm_free(z_rom[n] - z_true[n], free_ops) for n in range(len(traj.time_grid))]
            dyn_errs.append(float(np.mean(step_errs)))

        runtime_s = time.perf_counter() - t0
        row = dict(
            rank=rank,
            mass_orthonormality_max_error=ortho_err,
            mass_weighted_projection_error=projection_floor,
            reduced_initial_error=float(np.mean(ic_errs)),
            reduced_initial_error_max=float(np.max(ic_errs)),
            reduced_dynamics_error=float(np.mean(dyn_errs)),
            reduced_dynamics_error_max=float(np.max(dyn_errs)),
            runtime_s=runtime_s,
            is_full_rank=(rank == full_rank),
        )
        rows.append(row)
        print(
            f"rank={rank:3d}  ortho_err={ortho_err:.1e}  proj_floor={projection_floor:.3e}  "
            f"IC_err={row['reduced_initial_error']:.3e}  dyn_err={row['reduced_dynamics_error']:.3e}  "
            f"({runtime_s:.2f}s)"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "pod_free_rank_ablation.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    # Dedicated full-rank recovery check: direct (non-GMRES) dense solve of
    # the free CN system is the "identical constrained CN system" ground
    # truth, not the GMRES-based val trajectory (which carries its own
    # ~1e-10 GMRES-tolerance noise -- see Phase 1 Check 2 finding).
    print("\n=== Dedicated full-rank recovery check (direct dense solve ground truth) ===")
    mu, payoff, traj = val_trajs[0]
    z0 = traj.coefficients[0, fs.free_idx]
    z_direct = solve_free_cn_trajectory_direct(mu, payoff, mesh_info, traj.time_grid, ops, fs, free_ops, z0)

    full_basis = compute_free_pod_basis(snapshots_free, free_ops, full_rank)
    rom_full = FreePODGalerkinROM(full_basis)
    z_rom_full = rom_full.solve_reduced_trajectory(mu, payoff, mesh_info, traj.time_grid, fs, ops, free_ops, z0)
    full_rank_recovery_error = float(np.max(np.abs(z_rom_full - z_direct)))
    print(
        f"max|z_rom(full_rank={full_rank}) - z_direct| over the whole trajectory = "
        f"{full_rank_recovery_error:.3e} (expect near machine precision if the ROM equations are correct)"
    )

    with open(OUT_DIR / "full_rank_recovery_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["full_rank", "max_abs_error_vs_direct_solve", "query_id"])
        w.writeheader()
        w.writerow(dict(full_rank=full_rank, max_abs_error_vs_direct_solve=full_rank_recovery_error, query_id=0))
    print(f"wrote {OUT_DIR / 'full_rank_recovery_check.csv'}")


if __name__ == "__main__":
    main()
