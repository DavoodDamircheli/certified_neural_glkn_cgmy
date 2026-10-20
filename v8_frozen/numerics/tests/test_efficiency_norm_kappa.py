"""Step 3 (fix/efficiency-norm-kappa working prompt): unit tests that must
pass BEFORE rerunning the efficiency experiment.

Tests the manuscript's exact triple norm (main.tex, eq:triple-norm) and
exact kappa (main.tex, eq:kappa-def, Lemma~\\ref{lem:kappa}), implemented in
certification.constants.triple_norm_squared / kappa_operator_norm /
kappa_operator_norm_matrix_free -- NOT the old mass/L2-only norm or the old
cond(S_h)-as-kappa convention these replace (see that module's docstrings).

Tests 1-3 use small, hand-built random SPD matrices via a minimal fake
OperatorCache-like double (``_FakeOps``) rather than the real CGMY operator,
so the closed-form checks are exact and independent of any CGMY-specific
formula -- exactly what "manufactured error field on small random SPD
S_h/M_h" (Step 3, item 3) calls for. Test 4 uses the real OperatorCache at a
small mesh, since it is specifically checking dense-vs-matrix-free agreement
on the real operator.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from certification.constants import (
    kappa_operator_norm,
    kappa_operator_norm_matrix_free,
    triple_norm_squared,
)
from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, Mesh

RNG = np.random.default_rng(12345)


def _random_spd(n: int, rng: np.random.Generator, cond_boost: float = 1.0) -> np.ndarray:
    """A random SPD matrix, optionally with an artificially large condition
    number (cond_boost >> 1) by scaling up the smallest eigenvalue's gap --
    used to make cond(S) huge while keeping kappa(mu)=1 exactly (test 1)."""
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.geomspace(1.0, cond_boost, n)
    return (Q * eigs) @ Q.T


class _FakeOps:
    """Duck-types the subset of OperatorCache used by triple_norm_squared /
    kappa_operator_norm / kappa_operator_norm_matrix_free: n_dof, apply_A,
    apply_S, apply_M, apply_AT. `mu` is accepted and ignored by every apply_*
    method (the fake matrices don't depend on it)."""

    def __init__(self, A: np.ndarray, S: np.ndarray, M: np.ndarray) -> None:
        self.A = A
        self.S = S
        self.M = M
        self.n_dof = A.shape[0]

    def apply_A(self, mu, v):
        return self.A @ v

    def apply_AT(self, mu, v):
        return self.A.T @ v

    def apply_S(self, mu, v):
        return self.S @ v

    def apply_M(self, v):
        return self.M @ v


DUMMY_MU = object()  # never dereferenced by _FakeOps


# ── Test 1: A_h = S_h => kappa == 1, while cond(S_h) may be large ─────────
#
# v5 M4 (roadmap §4.1). The original single test asserted |kappa - 1| <= 1e-10
# at cond(S) = 1e8 and failed (kappa = 1.0000000015738064, single-threaded;
# 1 + 4.9e-11 multi-threaded -- the value depends on BLAS rounding). The
# investigation (scripts/v5_kappa_investigation.py; record in the M4 run
# directory) found:
#   * The stored test matrix (Q*eigs) @ Q.T is not exactly symmetric
#     (|S - S^T| up to 3.7e-9), while scipy's Cholesky reads one triangle and
#     A = S is used in full. In 60-digit arithmetic on the SAME stored entries,
#     kappa(A, lower-triangle S) = 1 + 2.9e-10 -- the represented problem's
#     exact kappa is not 1. With S = sym(A) formed exactly, kappa = 1 (to 1e-20).
#   * Cholesky and both triangular solves meet their componentwise backward
#     error bounds (gamma_{n+1}, gamma_n); the production value differs from the
#     exact value of its own data by 1.3e-9 <= n*eps*cond(S) = 2.7e-7.
# So no transformation/orientation error: the 1e-10 tolerance ignored the
# O(eps * cond(S)) forward error of L^{-1} S L^{-T}. The replacement tests use
# exactly symmetric input (so the exact kappa of the represented data is 1), a
# strict well-conditioned check, and for the ill-conditioned case backward-error
# checks plus a forward tolerance derived from cond(S). No A = S shortcut exists
# in production code; kappa_operator_norm runs its general path here.


def _symmetric_exactly(S: np.ndarray) -> np.ndarray:
    """(S + S^T)/2 is bitwise symmetric: entries (i,j) and (j,i) are the same
    two floats added in the same order... up to commutativity, which IEEE
    addition satisfies exactly."""
    S = 0.5 * (S + S.T)
    assert np.array_equal(S, S.T)
    return S


def _forward_tol(n: int, cond_S: float) -> float:
    """First-order forward-error bound for ||fl(L^{-1} S L^{-T}) - I||_2 >= |kappa - 1|.

    Cholesky: L L^T = S + dS, |dS| <= gamma_{n+1} |L||L^T| (Higham Thm 10.3), so
    ||dS||_2 <= gamma_{n+1} ||L||_F^2 = gamma_{n+1} trace(S) <= gamma_{n+1} n ||S||_2,
    and ||L^{-1} dS L^{-T}||_2 <= ||dS||_2 ||S^{-1}||_2 (to first order)
    <= n gamma_{n+1} cond(S). Each of the two triangular solves is backward stable
    with |dL| <= gamma_n |L| (Higham Thm 8.5), contributing at most another
    n gamma_n cond(S) each after the outer congruence. Total: 3 n gamma_{n+1} cond(S).
    """
    eps = np.finfo(float).eps
    gamma = (n + 1) * eps / (1 - (n + 1) * eps)
    return 3 * n * gamma * cond_S


def test_kappa_is_one_when_A_equals_S_well_conditioned_strict():
    """Strict: cond(S) = 50, exactly symmetric S. Tolerance 3 n gamma cond(S)
    is ~5.2e-12 here (vs 1e-10 in the old test at cond 1e8)."""
    rng = np.random.default_rng(2024)
    n = 12
    S = _symmetric_exactly(_random_spd(n, rng, cond_boost=50.0))
    cond_S = np.linalg.cond(S)
    kappa = kappa_operator_norm(DUMMY_MU, _FakeOps(A=S, S=S, M=np.eye(n)))
    tol = _forward_tol(n, cond_S)
    assert tol < 1e-11  # setup sanity: the strict case really is strict
    assert abs(kappa - 1.0) <= tol, (kappa, tol)


def test_kappa_is_one_when_A_equals_S_ill_conditioned_backward_and_forward():
    """cond(S) = 1e8 (same seed/construction as the original test, input made
    exactly symmetric). Checks (1) Cholesky componentwise backward error
    |L L^T - S| <= gamma_{n+1} |L||L^T|, (2) both triangular solves'
    componentwise residuals <= gamma_n |L||X|, (3) |kappa - 1| <= 3 n gamma cond(S),
    and (4) kappa >= 1 - tol (Lemma kappa: kappa >= 1). A wrong Cholesky
    orientation (L^{-T} S L^{-1}) would violate (3) by orders of magnitude
    (tests/test_v5_free_path.py checks that sensitivity on random A)."""
    rng = np.random.default_rng(12345)
    n = 12
    S = _symmetric_exactly(_random_spd(n, rng, cond_boost=1e8))
    cond_S = np.linalg.cond(S)
    assert cond_S > 1e6, f"test setup failed: cond(S)={cond_S:.3e} is not large"
    eps = np.finfo(float).eps
    g_n = n * eps / (1 - n * eps)
    g_n1 = (n + 1) * eps / (1 - (n + 1) * eps)

    L = sla.cholesky(S, lower=True)
    aL = np.abs(L)
    assert np.all(np.abs(L @ L.T - S) <= g_n1 * (aL @ aL.T) * (1 + 4 * n * eps))
    X = sla.solve_triangular(L, S, lower=True)
    assert np.all(np.abs(L @ X - S) <= g_n * (aL @ np.abs(X)) * (1 + 4 * n * eps))
    Bt = sla.solve_triangular(L, X.T, lower=True)
    lhs, rhs = np.abs(L @ Bt - X.T), g_n * (aL @ np.abs(Bt)) * (1 + 4 * n * eps)
    assert np.all(lhs <= rhs)

    kappa = kappa_operator_norm(DUMMY_MU, _FakeOps(A=S, S=S, M=np.eye(n)))
    tol = _forward_tol(n, cond_S)
    assert abs(kappa - 1.0) <= tol, (kappa, tol)
    assert kappa >= 1.0 - tol


# ── Test 2: A_h = c * S_h => kappa == |c| ──────────────────────────────────

@pytest.mark.parametrize("c", [3.7, -2.0, 0.05])
def test_kappa_scales_with_scalar_multiple_of_S(c: float):
    n = 10
    S = _random_spd(n, RNG, cond_boost=50.0)
    A = c * S
    kappa = kappa_operator_norm(DUMMY_MU, _FakeOps(A=A, S=S, M=np.eye(n)))
    assert kappa == pytest.approx(abs(c), rel=1e-9)


# ── Test 3: manufactured error field, known closed-form triple norm ───────

def test_triple_norm_matches_closed_form_on_random_spd():
    n = 8
    n_steps = 5
    dt = 0.037
    S = _random_spd(n, RNG, cond_boost=20.0)
    M = _random_spd(n, RNG, cond_boost=5.0)
    ops = _FakeOps(A=S, S=S, M=M)  # A unused by triple_norm_squared

    e = RNG.standard_normal((n_steps + 1, n))
    pred = e
    true = np.zeros_like(e)  # so pred - true == e exactly

    cho = sla.cho_factor(S)
    expected = 0.0
    for k in range(n_steps):
        Mde = M @ ((e[k + 1] - e[k]) / dt)
        time_term = Mde @ sla.cho_solve(cho, Mde)
        ebar = 0.5 * (e[k + 1] + e[k])
        space_term = ebar @ (S @ ebar)
        expected += dt * (time_term + space_term)

    got = triple_norm_squared(pred, true, DUMMY_MU, ops, dt)
    assert got == pytest.approx(expected, rel=1e-12)


def test_triple_norm_is_zero_for_zero_error():
    n = 6
    S = _random_spd(n, RNG)
    M = _random_spd(n, RNG)
    ops = _FakeOps(A=S, S=S, M=M)
    traj = RNG.standard_normal((4, n))
    got = triple_norm_squared(traj, traj, DUMMY_MU, ops, dt=0.1)
    assert got == pytest.approx(0.0, abs=1e-14)


def test_triple_norm_requires_both_terms():
    """A pure space-only (no time-derivative) computation must NOT agree
    with triple_norm_squared in general -- guards against silently
    reintroducing the old, incomplete norm."""
    n = 6
    n_steps = 4
    dt = 0.1
    S = _random_spd(n, RNG, cond_boost=10.0)
    M = _random_spd(n, RNG, cond_boost=10.0)
    ops = _FakeOps(A=S, S=S, M=M)
    e = RNG.standard_normal((n_steps + 1, n))

    full = triple_norm_squared(e, np.zeros_like(e), DUMMY_MU, ops, dt)
    space_only = sum(
        dt * (0.5 * (e[k + 1] + e[k])) @ (S @ (0.5 * (e[k + 1] + e[k])))
        for k in range(n_steps)
    )
    assert full > space_only + 1e-8, (
        "triple_norm_squared must include a genuine, nonzero time-derivative "
        "contribution beyond the space-only midpoint term"
    )


# ── Test 4: dense vs. matrix-free kappa agree at small N (real operator) ──

def test_dense_and_matrix_free_kappa_agree_on_real_operator():
    N, p, x_min, x_max = 16, 3, -3.0, 3.0
    ops = OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)
    mu = CGMYParameters(C=0.3, G=3.0, M=4.0, Y=1.4, r=0.05, T=0.5)

    kappa_dense = kappa_operator_norm(mu, ops)
    kappa_mf = kappa_operator_norm_matrix_free(mu, ops, tol=1e-11, max_iter=2000)

    assert kappa_dense == pytest.approx(kappa_mf, rel=1e-5), (
        f"dense kappa={kappa_dense!r} vs matrix-free kappa={kappa_mf!r} disagree "
        f"beyond tolerance"
    )
    assert kappa_dense >= 1.0 - 1e-8  # eq:kappa-bound: 1 <= kappa(mu)
