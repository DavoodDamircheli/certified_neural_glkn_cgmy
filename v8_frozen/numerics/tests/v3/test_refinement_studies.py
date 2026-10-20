"""Gate C refinement-study regression tests (validation/refinement_studies.py).

These check the STUDY SETUP is sound (comparable mesh width across domain
buffers; monotone error decrease on at least one axis for at least one
query) -- not a re-derivation of the Galerkin solver's own correctness,
which is Gates A/B's job.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from validation.refinement_studies import PRODUCTION_H  # noqa: E402

DOMAIN_CSV = REPO_ROOT / "validation" / "domain_refinement.csv"
SPACE_CSV = REPO_ROOT / "validation" / "space_refinement.csv"
TIME_CSV = REPO_ROOT / "validation" / "time_refinement.csv"


def _read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.mark.skipif(not DOMAIN_CSV.exists(), reason="domain_refinement.csv not generated yet")
def test_domain_refinement_holds_mesh_width_fixed():
    """The whole point of separating domain from space refinement: h must
    be IDENTICAL across every buffer tested, not merely similar."""
    rows = _read(DOMAIN_CSV)
    hs = {float(r["h"]) for r in rows}
    assert len(hs) == 1, f"domain refinement must hold h fixed across buffers, found {hs}"
    assert abs(hs.pop() - PRODUCTION_H) < 1e-12


@pytest.mark.skipif(not DOMAIN_CSV.exists(), reason="domain_refinement.csv not generated yet")
def test_domain_refinement_buffers_give_distinct_N():
    """N must scale with buffer (width/h), i.e. domain width is genuinely
    varying, not held accidentally constant."""
    rows = _read(DOMAIN_CSV)
    buffer_to_N = {int(float(r["buffer"])): int(float(r["N"])) for r in rows}
    assert len(buffer_to_N) == 4, buffer_to_N
    Ns = sorted(buffer_to_N.values())
    assert Ns == sorted(set(Ns)), "N must be distinct per buffer (strictly increasing with buffer)"
    assert Ns == [48, 64, 80, 96], f"expected N=[48,64,80,96] for buffers [6,8,10,12] at h={PRODUCTION_H}, got {Ns}"


@pytest.mark.skipif(not SPACE_CSV.exists(), reason="space_refinement.csv not generated yet")
def test_space_refinement_error_shrinks_on_at_least_one_query():
    """At fixed moneyness, at least one query's error should generally
    shrink as N increases from 32->256 (not required to be monotone at
    every single step -- GMRES/quadrature noise can perturb a single
    resolution -- but the coarsest and finest points must differ in the
    expected direction for the majority of (query, moneyness) pairs)."""
    rows = _read(SPACE_CSV)
    by_key: dict[tuple[str, str], dict[int, float]] = {}
    for r in rows:
        if r.get("sample", "full") != "full":
            continue
        key = (r["query_id"], r["moneyness"])
        by_key.setdefault(key, {})[int(float(r["N"]))] = float(r["abs_error"])

    improved = 0
    total = 0
    for key, errs in by_key.items():
        if 32 not in errs or 256 not in errs:
            continue
        total += 1
        if errs[256] <= errs[32]:
            improved += 1
    assert total > 0, "no (query, moneyness) pair had both N=32 and N=256"
    assert improved / total >= 0.5, (
        f"expected error at N=256 to be <= error at N=32 for most (query, moneyness) "
        f"pairs, got {improved}/{total}"
    )


@pytest.mark.skipif(not TIME_CSV.exists(), reason="time_refinement.csv not generated yet")
def test_time_refinement_produces_finite_errors():
    """Sanity only (the roadmap explicitly does not require second-order
    time convergence near the payoff kink) -- every row must at least be a
    finite, non-negative error, not NaN/inf from a broken solve."""
    rows = _read(TIME_CSV)
    assert len(rows) > 0
    for r in rows:
        err = float(r["abs_error"])
        assert np.isfinite(err) and err >= 0, f"non-finite or negative error: {r}"
