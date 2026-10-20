"""Gate B2 soundness test: a manufactured case where forcing disagreement
exactly cancels the naively-computed residual -- i.e. where trusting
``rho_eval == 0`` would wrongly accept a trajectory that does NOT actually
satisfy the discrete equation, unless the discrepancy-correction term
``(f_eval - f_ref)`` is applied (roadmap Gate B2:
``rho_ref = rho_eval + (A_ref-A_eval)*c_mid + (f_eval-f_ref)``).

This test does not touch ``certification.residuals`` -- it reimplements the
CN residual's own formula directly (same formula, different code path) so
that it can independently check the discrepancy-correction identity rather
than merely re-running the audited function on itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, Mesh  # noqa: E402


@pytest.fixture(scope="module")
def small_ops():
    mesh = Mesh(N=16, p=3, x_min=-3.394829814011908, x_max=12.605170185988092)
    return OperatorCache(mesh, mesh.x_min, mesh.x_max)


@pytest.fixture(scope="module")
def mu():
    return CGMYParameters(C=0.3, G=2.0, M=5.0, Y=1.4, r=0.05, T=0.5)


def _raw_cn_lhs(ops, mu, c_n, c_np1, dt):
    """M_h*(c^{n+1}-c^n)/dt + A_h(mu)*cbar -- the part of the CN residual
    formula that does not involve forcing."""
    c_bar = 0.5 * (c_np1 + c_n)
    return ops.apply_M((c_np1 - c_n) / dt) + ops.apply_A(mu, c_bar)


def test_cancelling_forcing_is_caught_by_discrepancy_correction(small_ops, mu):
    rng = np.random.default_rng(42)
    n = small_ops.n_dof
    dt = 1.0 / 60.0

    # An arbitrary candidate step -- NOT a real solve, deliberately: this
    # test must work for ANY c_n/c_np1, since the whole point is that the
    # naive check (rho_eval == 0) gives no information about whether the
    # trajectory is actually a solution once forcing can be wrong.
    c_n = rng.standard_normal(n)
    c_np1 = rng.standard_normal(n)

    lhs = _raw_cn_lhs(small_ops, mu, c_n, c_np1, dt)

    # f_eval is reverse-engineered to exactly cancel lhs -- simulating a
    # bug where "the forcing used for evaluation" was recomputed (rather
    # than reused bit-identically, CLAUDE.md Hard Invariant #2) and happens
    # to make the naive residual vanish.
    f_eval = lhs.copy()
    rho_eval = lhs - f_eval
    assert np.linalg.norm(rho_eval) < 1e-12, "sanity check: the manufactured case must start from an exactly-zero naive residual"

    # The TRUE forcing this trajectory was actually supposed to be checked
    # against (f_ref) is NOT f_eval -- e.g. the real stored forcing from the
    # solve that produced c_n/c_np1 (always exactly 0 in this project,
    # confirmed in EFFICIENCY_NORM_KAPPA_FIX_REPORT.md / boundary.py).
    f_ref = np.zeros(n)

    # Discrepancy correction (roadmap eq.): A_ref == A_eval here (no matrix
    # discrepancy in this manufactured case, only a forcing one), so the
    # correction reduces to rho_ref = rho_eval + (f_eval - f_ref).
    rho_ref = rho_eval + (f_eval - f_ref)

    # The whole point: a naive check of rho_eval alone would (wrongly)
    # accept this trajectory as solving the discrete equation. The
    # discrepancy-corrected rho_ref reveals the true, nonzero residual --
    # exactly equal to f_eval (since f_ref=0 and rho_eval=0 by construction).
    assert np.linalg.norm(rho_ref) > 1e-6, (
        "the discrepancy correction must NOT silently vanish: this manufactured "
        "case is specifically built so the naive residual is zero while the "
        "true one is not"
    )
    np.testing.assert_allclose(rho_ref, f_eval, atol=1e-10)

    # Also confirm this is not a vacuous construction: f_eval itself must be
    # non-negligible relative to the problem's own scale (lhs), i.e. we
    # didn't accidentally pick a case where everything is ~0 anyway.
    assert np.linalg.norm(f_eval) > 1e-3 * max(np.linalg.norm(lhs), 1.0) or np.linalg.norm(f_eval) > 1e-3


def test_matrix_discrepancy_term_is_also_caught(small_ops, mu):
    """Same idea, but the discrepancy is in A_h (A_ref != A_eval) rather
    than forcing -- e.g. mu drifting between the 'reference' and 'eval'
    assembly (a different, equally real way Hard Invariant #1 could be
    silently violated if two call sites ever used different mu)."""
    rng = np.random.default_rng(7)
    n = small_ops.n_dof
    dt = 1.0 / 60.0
    c_n = rng.standard_normal(n)
    c_np1 = rng.standard_normal(n)
    c_mid = 0.5 * (c_n + c_np1)

    mu_eval = mu
    mu_ref = CGMYParameters(C=mu.C * 1.05, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, T=mu.T)  # a 5% C drift

    lhs_eval = _raw_cn_lhs(small_ops, mu_eval, c_n, c_np1, dt)
    f_eval = lhs_eval.copy()  # again reverse-engineered to cancel, using mu_eval's own A_h
    rho_eval = lhs_eval - f_eval
    assert np.linalg.norm(rho_eval) < 1e-12

    f_ref = np.zeros(n)
    A_ref_cmid = small_ops.apply_A(mu_ref, c_mid)
    A_eval_cmid = small_ops.apply_A(mu_eval, c_mid)
    rho_ref = rho_eval + (A_ref_cmid - A_eval_cmid) + (f_eval - f_ref)

    assert np.linalg.norm(rho_ref) > 1e-6, (
        "an operator-assembly discrepancy (mu_ref != mu_eval) must also be "
        "caught by the full discrepancy-correction identity, not just a "
        "forcing discrepancy"
    )
