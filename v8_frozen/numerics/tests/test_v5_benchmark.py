"""v5 M5: benchmark harness invariants that do not need a run directory."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from common.run_manifest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
import gate_d_v5_summary as s  # noqa: E402


@pytest.mark.parametrize("delivered,verified,emp,expect", [
    (False, False, "", "unavailable"), (False, True, True, "unavailable"),
    (True, True, False, "verified"), (True, False, True, "empirically_checked"),
    (True, False, False, "failed"), (True, False, "", "failed"),
])
def test_target_status_vocabulary(delivered, verified, emp, expect):
    assert s.target_status(delivered, verified, emp) == expect


def test_delivery_alone_is_never_verified():
    assert s.target_status(True, False, "") != "verified"


def test_bootstrap_deterministic_and_over_queries():
    r = np.array([0.5, 0.6, 0.55, 0.7, 0.65])
    a = s.geomean_ci(r, s.bootstrap_seed(1, "x"), 200, 0.95)
    b = s.geomean_ci(r, s.bootstrap_seed(1, "x"), 200, 0.95)
    assert a == b and a[1] <= a[0] <= a[2]


def test_full_campaign_requires_explicit_approval(monkeypatch):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(v, "1")
    import run_gate_d_benchmark_v5 as h

    with pytest.raises(SystemExit, match="approval"):
        h.main(["--full"])
