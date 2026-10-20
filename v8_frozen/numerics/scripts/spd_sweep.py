"""Task 4 (working prompt): systematic SPD-failure sweep over the frozen
parameter box, at the production mesh and at 2x spatial resolution.

Mirrors the validation methodology CLAUDE.md's own account of the prior
Stage D narrowing describes (Sobol + exact vertices + a second mesh
resolution), so this sweep's results are directly comparable to that prior
claim ("validated at 0/500 fresh Sobol points, 0/64 exact vertices, and a
second mesh resolution (N=96)") -- if this sweep finds 0 failures too, that
confirms the prior claim still holds for the CURRENT box; if it finds
failures, that is the honest, surprising result Task 4 exists to surface.

Every failure found is appended to results/spd_failures.csv via
certification.spd_diagnostics.check_and_log_spd -- never silently skipped.

Usage: PYTHONPATH=src python3 scripts/spd_sweep.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

from cgmy.operators import OperatorCache
from cgmy.reference_solver import Mesh
from certification.spd_diagnostics import check_and_log_spd
from data.sampling import (
    load_parameter_box,
    sample_grid,
    sample_sobol,
    _edge_samples,
    _stress_samples,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
CSV_PATH = REPO_ROOT / "results" / "spd_failures.csv"

N_SOBOL = 500
N_STRESS = 200
SEED = 40400

# Production mesh geometry (domain fixed across resolutions -- only N
# changes, matching Stage D's own "second mesh resolution" comparison).
X_MIN, X_MAX = -3.394829814011908, 12.605170185988092  # log(100) +- 8
N_PRODUCTION = 64
N_DOUBLED = 128
P = 3


def _sweep_one_resolution(box, N: int, source_label: str) -> dict[str, int]:
    mesh = Mesh(N=N, p=P, x_min=X_MIN, x_max=X_MAX)
    ops = OperatorCache(mesh, X_MIN, X_MAX)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sobol = sample_sobol(box, N_SOBOL, seed=SEED)
        vertices = sample_grid(box, n_per_dim=2)  # 2^6 = 64 exact vertices
        stress = _stress_samples(box, N_STRESS, seed=SEED + 1)
        edges = _edge_samples(box, N_STRESS, seed=SEED + 2)

    counts = {}
    for name, samples in (("sobol", sobol), ("vertices", vertices), ("stress", stress), ("edges", edges)):
        n_fail = 0
        t0 = time.time()
        for mu, payoff in samples:
            dt = mu.T / 60.0  # production n_time_steps
            const = check_and_log_spd(mu, ops, CSV_PATH, dt=dt, source=f"{source_label}/{name}")
            if not const.is_spd:
                n_fail += 1
        elapsed = time.time() - t0
        print(f"  [{source_label}] {name:10s}: {n_fail}/{len(samples)} failed  ({elapsed:.1f}s)")
        counts[name] = n_fail
    return counts


def main() -> None:
    box = load_parameter_box(CONFIG_PATH)
    print(f"parameter box: C={box.C} G={box.G} M={box.M} Y={box.Y} r={box.r} maturity={box.maturity}")
    print(f"domain: [{X_MIN:.6f}, {X_MAX:.6f}]  (width {X_MAX - X_MIN:.4f})")
    print(f"writing failures to {CSV_PATH}\n")

    print(f"=== N={N_PRODUCTION} (production resolution) ===")
    prod_counts = _sweep_one_resolution(box, N_PRODUCTION, f"N{N_PRODUCTION}")

    print(f"\n=== N={N_DOUBLED} (2x spatial resolution) ===")
    doubled_counts = _sweep_one_resolution(box, N_DOUBLED, f"N{N_DOUBLED}")

    print("\n=== summary ===")
    total_prod = sum(prod_counts.values())
    total_prod_n = N_SOBOL + 64 + N_STRESS + N_STRESS
    total_doubled = sum(doubled_counts.values())
    print(f"N={N_PRODUCTION}: {total_prod}/{total_prod_n} total failures across all sample sets")
    print(f"N={N_DOUBLED}: {total_doubled}/{total_prod_n} total failures across all sample sets")
    if CSV_PATH.exists():
        print(f"\nfailure detail written to {CSV_PATH}")
    else:
        print("\nno failures found at either resolution -- no CSV written")


if __name__ == "__main__":
    main()
