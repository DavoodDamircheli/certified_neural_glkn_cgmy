"""Task 4 (working prompt): the SPD-failure diagnostic harness must never
silently skip a failure, and must be able to reproduce a logged failure
standalone from its saved row.
"""

from __future__ import annotations

import csv

import pytest

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, Mesh
from certification.spd_diagnostics import CSV_FIELDS, check_and_log_spd, reproduce_failure

# A known SPD point and a known non-SPD point (production mesh) --
# see notes/04_spd_study.md.
GOOD_MU = CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.4)
BAD_MU = CGMYParameters(C=1.0, G=1.0, M=9.0, Y=1.8, r=0.1, T=1.0)

X_MIN, X_MAX = -3.394829814011908, 12.605170185988092
N, P = 64, 3


@pytest.fixture()
def ops():
    return OperatorCache(Mesh(N=N, p=P, x_min=X_MIN, x_max=X_MAX), X_MIN, X_MAX)


def test_good_mu_does_not_write_a_row(tmp_path, ops):
    csv_path = tmp_path / "spd_failures.csv"
    const = check_and_log_spd(GOOD_MU, ops, csv_path, dt=0.4 / 60, source="test")
    assert const.is_spd
    assert not csv_path.exists()


def test_bad_mu_writes_exactly_one_row_with_every_field(tmp_path, ops):
    csv_path = tmp_path / "spd_failures.csv"
    const = check_and_log_spd(BAD_MU, ops, csv_path, dt=1.0 / 60, source="test")
    assert not const.is_spd

    assert csv_path.exists()
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == set(CSV_FIELDS)
    assert float(row["C"]) == BAD_MU.C
    assert float(row["M"]) == BAD_MU.M
    assert float(row["lambda_min"]) < 0
    assert row["alpha_certified"] in ("nan", "")  # NaN -- not SPD, no certified bound
    assert row["source"] == "test"


def test_never_silently_skips_repeated_failures(tmp_path, ops):
    """Two failures in a row must produce two rows, not one (dedup would be
    silently dropping information the working prompt says never to drop)."""
    csv_path = tmp_path / "spd_failures.csv"
    check_and_log_spd(BAD_MU, ops, csv_path, source="a")
    check_and_log_spd(BAD_MU, ops, csv_path, source="b")
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 2
    assert [r["source"] for r in rows] == ["a", "b"]


def test_reproduce_failure_matches_the_original_computation(tmp_path, ops):
    csv_path = tmp_path / "spd_failures.csv"
    const_original = check_and_log_spd(BAD_MU, ops, csv_path, source="test")
    row = next(csv.DictReader(open(csv_path)))

    const_reproduced = reproduce_failure(row)
    assert not const_reproduced.is_spd
    assert const_reproduced.lambda_min_numerical == pytest.approx(
        const_original.lambda_min_numerical, rel=1e-9
    )
