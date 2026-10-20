"""Gate E4: regression tests for the POD rank ablation's headline claims.

Reads training/pod_rank_ablation.csv (produced by
training/pod_rank_ablation.py against the real production dataset) rather
than recomputing the SVD/dynamics in the test itself -- these are the
cheap, structural sanity checks on the ALREADY-COMPUTED result, not a
from-scratch numerical verification (that would duplicate the ablation
script's own ~10s runtime on every test run).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "training" / "pod_rank_ablation.csv"


def _load_rows():
    if not CSV_PATH.exists():
        pytest.skip(f"{CSV_PATH} not generated yet -- run training/pod_rank_ablation.py first")
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def test_projection_floor_decreases_monotonically_with_rank():
    rows = sorted(_load_rows(), key=lambda r: int(r["rank"]))
    floors = [float(r["mass_weighted_projection_error"]) for r in rows]
    for a, b in zip(floors, floors[1:]):
        assert b <= a * 1.01, (
            f"mass_weighted_projection_error should not increase with rank "
            f"(best-approximation floor is monotone non-increasing by "
            f"construction of a truncated SVD); got {floors}"
        )


def test_full_rank_recovery_is_near_machine_precision():
    rows = _load_rows()
    full_rows = [r for r in rows if r["is_full_rank"] in ("True", "true", "1")]
    assert len(full_rows) == 1, f"expected exactly one full-rank row, got {full_rows}"
    floor = float(full_rows[0]["mass_weighted_projection_error"])
    assert floor < 1e-10, (
        f"full-rank POD should recover the snapshot matrix to ~machine "
        f"precision in the mass norm (no truncation at full rank); got {floor:.3e}"
    )


def test_reduced_initial_error_is_exactly_zero_every_rank():
    """PODGalerkinROM uses the same exact-IC structural trick as every
    other model in this project (c_theta(0) = c_h^0 exactly, never a
    POD-projected c0) -- this is a structural guarantee, not an empirical
    convergence property, so it should hold at EVERY rank, not just full
    rank."""
    rows = _load_rows()
    for r in rows:
        ic_err = float(r["reduced_initial_error"])
        assert ic_err < 1e-10, (
            f"rank={r['rank']}: reduced_initial_error={ic_err:.3e} is not ~0 -- "
            f"this breaks CLAUDE.md Hard Invariant #4 (exact initial condition) "
            f"if it ever fails, since POD is supposed to enforce it structurally "
            f"like every other model in this project"
        )


def test_dynamics_error_does_not_vanish_at_full_rank():
    """Documents (as a regression test, not just prose) that the reduced-
    dynamics error for THIS architecture is dominated by something other
    than basis truncation -- it should NOT collapse to ~0 even at full
    rank, consistent with src/learning/pod.py's own documented limitation
    (Dirichlet BCs not exactly enforced in the reduced dynamics). If this
    test ever starts failing because dynamics error DOES vanish at full
    rank, that's a real architectural change worth knowing about, not a
    regression to silently update away."""
    rows = _load_rows()
    full_row = next(r for r in rows if r["is_full_rank"] in ("True", "true", "1"))
    dyn_err = float(full_row["reduced_dynamics_error"])
    assert dyn_err > 1.0, (
        f"expected the known, documented full-rank dynamics-error floor "
        f"(previously measured ~9.46) to persist; got {dyn_err:.3e}, which "
        f"would mean the Dirichlet-BC/reduced-dynamics limitation this "
        f"test exists to document may have been fixed -- worth investigating, "
        f"not silently accepting"
    )
