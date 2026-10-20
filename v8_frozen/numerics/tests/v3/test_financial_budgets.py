"""Gate E1-E2 soundness tests for financial/beta_eval.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "financial"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "vendor"))

from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import Mesh  # noqa: E402

import beta_eval as be  # noqa: E402


@pytest.fixture(scope="module")
def ops():
    N, p = 32, 3
    x_min, x_max = np.log(100.0) - 8.0, np.log(100.0) + 8.0
    return OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)


def test_beta_eval_nonnegative(ops):
    xs = np.linspace(ops.x_min + 1.0, ops.x_max - 1.0, 25)
    vals = [be.beta_eval(x, ops) for x in xs]
    assert all(v >= 0.0 for v in vals)
    assert any(v > 0.0 for v in vals)  # not vacuously zero everywhere


def test_enclosed_sup_is_at_least_the_sampled_max(ops):
    window = (np.log(80.0), np.log(120.0))
    enc = be.enclosed_sup_beta_eval(window, ops, n_per_cell=15)
    assert enc["enclosed_sup"] >= enc["sampled_max"]
    assert enc["safety_margin"] >= 0.0


def test_window_gram_generalized_eigenvalue_inequality_holds(ops):
    """The defining property of lambda_max(W, M): e^T W e <= lambda_max *
    e^T M e for EVERY vector e (not just the eigenvector that achieves
    equality). This is the actual mathematical content of the window-Gram
    bound; it is NOT the same quantity as (and is not comparable to) a
    pointwise beta_eval value -- W integrates over a window, beta_eval
    evaluates at one point, and a narrow window can have a smaller
    integrated quantity than a sharp pointwise peak. Do not conflate them."""
    window = (np.log(80.0), np.log(120.0))
    W = be.window_gram_matrix(window, ops)
    lam_max = be.window_gram_max_eigenvalue(W, ops)

    rng = np.random.default_rng(0)
    n = ops.n_dof
    for _ in range(20):
        e = rng.standard_normal(n)
        lhs = float(e @ W @ e)
        rhs = lam_max * float(e @ ops.apply_M(e))
        assert lhs <= rhs + 1e-9 * max(abs(rhs), 1.0), (lhs, rhs)


def test_floor_is_positive_and_below_tightest_dollar_budget_in_relative_terms():
    # FLOOR (1e-5) should be well below the tightest relative level (1e-4)
    # scaled by a representative price (~K=100) -- i.e. it should not itself
    # be the binding constraint at the levels this project actually sweeps.
    assert be.FLOOR > 0.0
    representative_price = 20.0  # typical ATM put price in this project's box
    assert be.FLOOR < min(be.RELATIVE_LEVELS) * representative_price


def test_no_certificate_bound_violation_in_committed_sweep():
    """Regression check on the committed financial/budget_sweep.csv: every
    row with a finite bound and a finite certified error must have
    bound >= certified error (effectivity >= 1), matching Hard Invariant #5
    ("eta_guar uses the upper Riesz bracket bound... zero certificate
    violations is a correctness requirement")."""
    import csv
    path = REPO_ROOT / "financial" / "budget_sweep.csv"
    if not path.exists():
        pytest.skip("financial/budget_sweep.csv not generated yet")
    violations = []
    with open(path) as f:
        for row in csv.DictReader(f):
            bound = row["final_bound"]
            err = row["final_err_vs_discrete_CERTIFIED"]
            if bound in ("", "nan") or err in ("", "nan"):
                continue
            bound_f, err_f = float(bound), float(err)
            if bound_f < err_f:
                violations.append((row["query_id"], row["S_over_K"], bound_f, err_f))
    assert not violations, f"certificate bound violated: {violations}"
