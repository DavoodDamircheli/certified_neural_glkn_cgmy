"""Stage A operator-consistency tests (CLAUDE.md Gate 1 prerequisites).

These tests check that ``cgmy.operators.OperatorCache`` reproduces the
trusted external solver's own operator exactly -- not an approximation of
it -- on small meshes where dense/explicit reference computations are
affordable. They do not test the reference solver's own numerics (that is
``cgmy_bspline``'s test suite); they test that this repository's wrapper
calls it correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from cgmy import _external  # noqa: F401  (must precede any cgmy_bspline import)

from cgmy_bspline import matrices as cgmy_matrices
from cgmy_bspline.connection_coefficients import tempered_coeffs_via_ifft
from cgmy_bspline.parameters import CGMYParams

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, Mesh

MU = CGMYParameters(C=1.0, G=5.0, M=5.0, Y=0.5, r=0.05, T=0.5)
STRIKE = 100.0


def _small_cache(N: int = 16, p: int = 3, x_min: float = -3.0, x_max: float = 3.0) -> OperatorCache:
    return OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)


# ── Test 6: matrix-free apply_A vs explicitly assembled A_h ──────────────────


def test_apply_A_matches_dense_assembly_small_mesh():
    """apply_A(mu, v) matches cgmy_bspline.matrices.operator_matrix's dense A_h @ v.

    Uses bandwidth = 2*n_dof in operator_matrix ("full coverage", per
    cgmy_bspline's own CLAUDE.md) so the banded assembly's IFFT grid (L =
    bandwidth) matches build_fractional_toeplitz's internal L = 2*n_dof,
    and no lag is truncated -- the two should then agree to floating-point
    precision, not just approximately.
    """
    N, p, x_min, x_max = 16, 3, -3.0, 3.0
    h = (x_max - x_min) / N
    cache = _small_cache(N, p, x_min, x_max)
    n_dof = cache.n_dof

    params = CGMYParams(C=MU.C, G=MU.G, M=MU.M, Y=MU.Y, r=MU.r, K=STRIKE, T=MU.T)
    dense = cgmy_matrices.operator_matrix(N, p, h, x_min, params, bandwidth=2 * n_dof)
    A_h_dense = dense["A_h"]

    rng = np.random.default_rng(0)
    v = rng.standard_normal(n_dof)

    expected = A_h_dense @ v
    actual = cache.apply_A(MU, v)

    assert np.max(np.abs(actual - expected)) < 1e-10


# ── Test 7: FFT/Toeplitz application vs direct dense assembly ───────────────


def test_toeplitz_matvec_matches_dense_assembly_small_N():
    """The Toeplitz piece of apply_A matches an explicit dense Toeplitz matrix.

    Builds K_G_minus densely from the same Lambda sequence
    (tempered_coeffs_via_ifft, called unchanged) that
    cgmy_bspline.toeplitz.build_fractional_toeplitz uses internally, then
    compares K_G @ v to OperatorCache's cached Toeplitz matvec.
    """
    N, p, x_min, x_max = 16, 3, -3.0, 3.0
    cache = _small_cache(N, p, x_min, x_max)
    n_dof = cache.n_dof
    h = cache.h

    delta_G = MU.G * h
    L = 2 * n_dof
    lam = tempered_coeffs_via_ifft(delta_G, MU.Y, p, "minus", L=L) * (h ** (-MU.Y))
    first_col = lam[:n_dof]
    first_row = lam[(-np.arange(n_dof)) % L]
    K_G_dense = np.array(
        [
            [first_col[i - j] if i >= j else first_row[j - i] for j in range(n_dof)]
            for i in range(n_dof)
        ]
    )

    rng = np.random.default_rng(1)
    v = rng.standard_normal(n_dof)

    pair = cache._toeplitz_pair(MU)
    expected = K_G_dense @ v
    actual = pair.K_G.matvec(v)

    assert np.max(np.abs(actual - expected)) < 1e-8


# ── Symmetry and SPD check for S_h ───────────────────────────────────────────


def _dense_S(cache: OperatorCache, mu: CGMYParameters) -> np.ndarray:
    n = cache.n_dof
    S = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        S[:, j] = cache.apply_S(mu, e)
    return S


def test_S_h_is_symmetric():
    cache = _small_cache()
    S = _dense_S(cache, MU)
    assert np.max(np.abs(S - S.T)) < 1e-10


@pytest.mark.parametrize("Y", [0.15, 0.5, 1.05, 1.5, 1.85])
def test_S_h_smallest_eigenvalue_positive(Y: float):
    """S_h = (A_h + A_h^T)/2 is SPD on this small mesh across the Y range.

    This is an empirical check on one small mesh/parameter point, not a
    proof of coercivity -- it is a Stage-A sanity gate, not Gate 2/3 from
    CLAUDE.md (which require the full Riesz-bracket and residual machinery).
    """
    mu = CGMYParameters(C=MU.C, G=MU.G, M=MU.M, Y=Y, r=MU.r, T=MU.T)
    cache = _small_cache()
    S = _dense_S(cache, mu)
    eigs = np.linalg.eigvalsh(S)
    assert eigs.min() > 0.0, f"S_h not SPD for Y={Y}: min eig = {eigs.min():.3e}"


# ── apply_M / apply_M_inv / apply_AT sanity ──────────────────────────────────


def test_apply_M_inv_is_left_inverse_of_apply_M():
    cache = _small_cache()
    rng = np.random.default_rng(2)
    v = rng.standard_normal(cache.n_dof)
    v_back = cache.apply_M_inv(cache.apply_M(v))
    assert np.max(np.abs(v_back - v)) < 1e-10


def test_apply_AT_is_the_adjoint_of_apply_A():
    cache = _small_cache()
    rng = np.random.default_rng(3)
    v = rng.standard_normal(cache.n_dof)
    w = rng.standard_normal(cache.n_dof)
    lhs = np.dot(cache.apply_A(MU, v), w)
    rhs = np.dot(v, cache.apply_AT(MU, w))
    assert abs(lhs - rhs) < 1e-9 * max(abs(lhs), 1.0)


def test_diag_S_matches_dense_assembly():
    """Task 9 (working prompt): diag_S must match a full dense assembly's
    diagonal exactly (to machine precision) -- it exists specifically so a
    Jacobi/diagonal PCG preconditioner never needs dense assembly at all."""
    from certification.constants import dense_from_apply

    cache = _small_cache()
    S = dense_from_apply(lambda v: cache.apply_S(MU, v), cache.n_dof)
    diag_dense = np.diag(S)
    diag_cheap = cache.diag_S(MU)
    assert np.max(np.abs(diag_dense - diag_cheap)) < 1e-10
