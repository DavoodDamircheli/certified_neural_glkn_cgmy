"""Stage D debug dataset: 100-500 trajectories, parallelized across parameters.

Samples the frozen parameter box (configs/base.yaml) into the five named
splits (train_interior, interpolation_test, edge_test, stress_test,
ood_test), solves each point via cgmy.reference_solver.solve_cgmy in a
process pool, and writes every result -- success or failure -- to one HDF5
file via data.dataset.write_sample. Every failed solve is logged with its
full traceback (data.dataset.SampleResult.error), never dropped silently.

Also demonstrates (separately, not counted among the trajectories above)
that data.payoff_family can build p-vectors for the other 5 payoff
families, even though solve_cgmy can only produce full trajectories for
European put today.

Usage:
    python scripts/generate_debug_dataset.py [--n-total 200] [--workers N]
        [--output data/debug_dataset.h5] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cgmy.reference_solver import CGMYParameters, Mesh, TimeGrid
from cgmy.operators import OperatorCache
from data import payoff_family
from data.dataset import SampleResult, generate_one_sample, write_manifest, write_sample
from data.sampling import ParameterBox, generate_named_splits, load_parameter_box, save_splits

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

MESH = Mesh(N=64, p=3)
TIME_GRID = TimeGrid(n_steps=100)
MODE = "toeplitz"


def _worker(mu: CGMYParameters, payoff, split: str) -> tuple[str, SampleResult, float]:
    t0 = time.time()
    result = generate_one_sample(mu, payoff, MESH, TIME_GRID, mode=MODE)
    return split, result, time.time() - t0


def _demo_other_payoff_families(box: ParameterBox) -> None:
    """Illustrate (not part of the 100-500 trajectory count) that the other 5
    payoff families produce valid p-vectors today, even without a solved
    trajectory to go with them."""
    print("\n--- payoff_family p-vector demo (not solved via solve_cgmy) ---")
    K = box.strike
    log_k = float(np.log(K))
    x_min, x_max = log_k - 3.0, log_k + 3.0  # domain must actually contain log(K)
    mesh = Mesh(N=64, p=3, x_min=x_min, x_max=x_max)
    ops = OperatorCache(mesh, x_min, x_max)
    demos = {
        "european_call": payoff_family.european_call(K, ops),
        "call_spread": payoff_family.call_spread(0.9 * K, 1.1 * K, ops),
        "put_spread": payoff_family.put_spread(0.9 * K, 1.1 * K, ops),
        "butterfly": payoff_family.butterfly(0.9 * K, K, 1.1 * K, ops),
        "smooth_synthetic_combination": payoff_family.smooth_synthetic_combination(
            [payoff_family.SmoothCombinationLeg(1.0, 0.9 * K),
             payoff_family.SmoothCombinationLeg(-2.0, K),
             payoff_family.SmoothCombinationLeg(1.0, 1.1 * K)],
            ops,
        ),
    }
    for name, p in demos.items():
        print(f"  {name:30s} p-vector norm = {np.linalg.norm(p):.4f}  (SOLVABLE_PAYOFFS: "
              f"{name in payoff_family.SOLVABLE_PAYOFFS})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=200)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "data" / "debug_dataset.h5"))
    parser.add_argument("--splits-output", type=str,
                         default=str(REPO_ROOT / "data" / "debug_dataset_splits.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (100 <= args.n_total <= 500):
        raise ValueError(f"--n-total must be in [100, 500], got {args.n_total}")

    box = load_parameter_box(CONFIG_PATH)
    frac = {"train_interior": 0.5, "interpolation_test": 0.15,
            "edge_test": 0.15, "stress_test": 0.1, "ood_test": 0.1}
    counts = {name: max(1, round(f * args.n_total)) for name, f in frac.items()}

    print(f"Parameter box: {box}")
    print(f"Split sizes (n_total={args.n_total}): {counts}")
    print(f"Mesh: N={MESH.N}, p={MESH.p} (auto domain per mu); "
          f"time_grid: n_steps={TIME_GRID.n_steps}; mode={MODE}")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sobol's "not a power of 2" balance warning
        splits = generate_named_splits(box, seed=args.seed, counts=counts)
    save_splits(splits, args.splits_output)
    print(f"Splits permanently recorded at {args.splits_output}")

    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    tasks = [(mu, payoff, split) for split, samples in splits.items() for mu, payoff in samples]

    n_ok = 0
    n_fail = 0
    solve_times: list[float] = []
    failures_log: list[dict] = []
    slow_or_fragile: list[dict] = []

    index = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, mu, payoff, split) for mu, payoff, split in tasks]
        for fut in as_completed(futures):
            split, result, wall_time = fut.result()
            solve_times.append(wall_time)
            write_sample(output_path, index, split, result)

            if result.ok:
                n_ok += 1
                iters = result.traj.solver_metadata.mean_gmres_iters or 0.0
                if wall_time > 5.0 or iters > 60 or (result.is_spd is False):
                    slow_or_fragile.append({
                        "mu": result.mu.__dict__, "split": split,
                        "wall_time": wall_time, "mean_gmres_iters": iters,
                        "is_spd": result.is_spd,
                    })
            else:
                n_fail += 1
                failures_log.append({
                    "mu": result.mu.__dict__, "split": split,
                    "error": result.error,
                })
                print(f"[FAILED] split={split} mu={result.mu} error={result.error.splitlines()[-1]}")

            index += 1
            if index % 20 == 0 or index == len(tasks):
                print(f"  progress: {index}/{len(tasks)}  (ok={n_ok}, failed={n_fail})")

    total_wall = time.time() - t_start

    with open(CONFIG_PATH) as f:
        config_snapshot = f.read()
    write_manifest(output_path, {k: len(v) for k, v in splits.items()}, n_fail, config_snapshot, args.seed)

    failures_path = output_path.with_suffix(".failures.json")
    failures_path.write_text(json.dumps(failures_log, indent=2))

    _demo_other_payoff_families(box)

    print("\n" + "=" * 78)
    print("STAGE D DATASET GENERATION REPORT")
    print("=" * 78)
    print(f"Output file: {output_path}")
    print(f"Total attempted: {len(tasks)}   ok: {n_ok}   failed: {n_fail}")
    split_sizes = {name: len(samples) for name, samples in splits.items()}
    print(f"Split sizes: {split_sizes}")
    print(f"Total wall-clock time: {total_wall:.1f}s over {args.workers or 'default'} workers")
    if solve_times:
        arr = np.array(solve_times)
        print(f"Per-trajectory wall time: mean={arr.mean():.3f}s  median={np.median(arr):.3f}s  "
              f"min={arr.min():.3f}s  max={arr.max():.3f}s")
    print(f"Failures logged to: {failures_path} ({n_fail} entries)")
    print(f"Slow/fragile samples flagged: {len(slow_or_fragile)} "
          f"(wall_time>5s, mean_gmres_iters>60, or S_h not SPD)")
    if slow_or_fragile:
        for s in slow_or_fragile[:10]:
            print(f"    split={s['split']:20s} mu={s['mu']}  wall={s['wall_time']:.2f}s  "
                  f"gmres_iters={s['mean_gmres_iters']:.1f}  is_spd={s['is_spd']}")
        if len(slow_or_fragile) > 10:
            print(f"    ... and {len(slow_or_fragile) - 10} more")
    print("=" * 78)


if __name__ == "__main__":
    main()
