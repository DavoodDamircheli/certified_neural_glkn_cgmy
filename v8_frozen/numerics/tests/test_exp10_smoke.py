"""Toy-scale smoke test for experiments/exp10_cross_resolution.py: exercises
run_experiment() (the core logic, factored out of main() precisely so it can
be called without touching disk outputs) at a tiny scale -- 2 resolutions,
2 queries, 3 time steps -- to catch import/shape/plumbing regressions fast,
not to check the scientific result itself (see exp10's own printed
claim-check for that). Must finish well under 30s.
"""

from __future__ import annotations

import time

from cgmy.reference_solver import TimeGrid
from exp10_cross_resolution import MODEL_NAMES, TRAIN_N, run_experiment


def test_exp10_runs_at_toy_scale_under_30s():
    t0 = time.time()
    result = run_experiment(
        resolutions=[32, TRAIN_N], n_test_mus=2, time_grid_req=TimeGrid(n_steps=3), seed=1,
    )
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"exp10 toy run took {elapsed:.1f}s, expected < 30s"

    records = result["records"]
    assert set(records.keys()) == {(m, N) for m in MODEL_NAMES for N in [32, TRAIN_N]}

    # At the training resolution, every model must produce SOME successful
    # prediction (the toy run's 2 queries might individually lose SPD, but
    # not-SPD is orthogonal to model architecture -- a total wipeout at the
    # training resolution would indicate a real plumbing break, not noise).
    for model_name in MODEL_NAMES:
        rows = records[(model_name, TRAIN_N)]
        assert len(rows) == 2
        assert any(r["success"] for r in rows), (
            f"{model_name} had zero successful predictions at its OWN training "
            f"resolution N={TRAIN_N} -- likely a plumbing regression, not a "
            f"legitimate cross-resolution finding"
        )

    # Off-training resolution: MLP/DeepONet/POD must fail every query
    # (Claim A); ProjectedFNO must succeed (Claim B's precondition).
    for model_name in ("MLP", "DeepONet", "POD"):
        rows = records[(model_name, 32)]
        assert all(not r["success"] for r in rows), (
            f"{model_name} unexpectedly succeeded at an off-training resolution "
            f"in the toy run -- see exp10's Claim A"
        )
    assert any(r["success"] for r in records[("ProjectedFNO", 32)])
