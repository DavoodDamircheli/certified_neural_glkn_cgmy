"""Plan 2.1 done-when criteria (EIM of the symbol family).

Quick mode: 100 test queries at N = 64 (dense 2-norm), 30 at N = 512 (4x-finer symbol sup-norm) of which
5 also with the dense 2-norm.  CGMY_FULL_TESTS=1: the 1000 Sobol test queries (seed 2026) at both N, 50 of
them with the dense 2-norm at N = 512.  The EIM files are built on first use (data/eim/, ~1-3 min).
"""
import numpy as np
import pytest
from scipy.linalg import norm

from conftest import FULL
from cgmy_cert.params import Query
from cgmy_cert.symbols import fractional_alias, free_matrices, ToeplitzOp, theta_grid
from cgmy_cert.eim import load_eim, family, family_grid, eim_test_queries, stack, mirror


@pytest.fixture(scope="module")
def eim64():
    return load_eim(64, nproc=8)


@pytest.fixture(scope="module")
def eim512():
    return load_eim(512, nproc=8)


def test_family_matches_fractional_alias():
    q = Query(0.4, 1.7, 6.3, 1.45, 0.03)
    th = theta_grid(310); h = 16 / 64
    a, da = fractional_alias(th, q, h, 100, with_derivative=True)
    b, db = family(th, (q.G, q.M, q.Y), h, 100, with_derivative=True)
    assert np.max(np.abs(a - b)) < 1e-13 * np.max(np.abs(a))
    assert np.max(np.abs(da - db)) < 1e-12 * np.max(np.abs(da))


def test_symmetrised_family_is_what_A_I_assembles():
    """The real part of the Toeplitz entries of the raw samples = the entries of the symmetrised samples."""
    q = Query(0.4, 1.7, 6.3, 1.45, 0.03)
    fs = free_matrices(q, 64)
    sym_a = fs.sym.a
    zm = np.conj(sym_a[mirror(sym_a.size)])
    op = ToeplitzOp(0.5 * (sym_a + zm), fs.n, extra=2)
    assert np.max(np.abs(op.t.imag)) < 1e-14 * np.max(np.abs(op.t))
    assert np.max(np.abs(op.t.real - fs.A.t.real)) < 1e-14 * np.max(np.abs(op.t))


def test_toeplitz_from_lags():
    q = Query(0.4, 1.7, 6.3, 1.45, 0.03)
    fs = free_matrices(q, 64)
    op = ToeplitzOp.from_lags(fs.A.t, fs.n, 2)
    v = np.random.default_rng(1).standard_normal(fs.n)
    assert np.max(np.abs(op.matvec(v) - fs.A.matvec(v))) < 1e-13 * norm(fs.A.matvec(v))
    assert op.entry(fs.n + 1) == fs.A.entry(fs.n + 1)


@pytest.mark.parametrize("which", ["eim64", "eim512"])
def test_eim_structure(which, request):
    e = request.getfixturevalue(which)
    Q = e.Q
    assert np.all(np.diag(e.B) == 1.0) and np.all(np.triu(e.B, 1) == 0.0)
    Q_tol = e.meta.get("Q_tol", Q)          # plan rule: smallest Q with test sup-error <= 1e-9
    assert e.test_err[Q_tol] <= 1e-9 and e.test_err[Q_tol - 1] > 1e-9
    assert Q >= Q_tol and 15 <= Q <= 40
    # interpolation: I_Q Phi_nu equals Phi_nu at the magic points; the stored lags are T(xi_q)
    q = eim_test_queries(3)[2]
    ell = e.ell((q.G, q.M, q.Y))
    I, _ = e.interpolant(ell)
    phi, _ = family_grid(e.L, (q.G, q.M, q.Y), e.h, e.K)
    assert np.max(np.abs(stack(I - phi)[e.magic[:Q]])) < 1e-12 * np.max(np.abs(phi))
    op = ToeplitzOp(e.xi[Q - 1], e.n, e.extra)
    assert np.max(np.abs(op.t.real - e.lags[Q - 1])) < 1e-15 * np.max(np.abs(op.t))
    # the derivative samples are the derivatives of the basis functions (recursion on another grid)
    xi4, dxi4 = e.basis_on(4 * e.L, Q=6, with_derivative=True)
    assert np.max(np.abs(xi4[:, ::4] - e.xi[:6])) < 1e-12
    assert np.max(np.abs(dxi4[:, ::4] - e.dxi[:6])) < 1e-11 * np.max(np.abs(e.dxi[:6]))


def _queries(n_quick):
    return eim_test_queries(1000 if FULL else n_quick)


def test_eps_Q_dense_N64(eim64):
    """||A_I - A_Q||_2 <= eps_Q <= 1e-8 ||A_I||_2 (dense 2-norm) on the Sobol test queries at N = 64."""
    e = eim64
    worst = 0.0
    for q in _queries(100):
        fs = free_matrices(q, 64)
        A = fs.A.dense()
        info = e.eps_Q(q)
        err = norm(A - e.A_Q(q, ell=info["ell"]).dense(), 2)
        nA = norm(A, 2)
        assert err <= info["eps_Q"], q
        assert info["eps_Q"] <= 1e-8 * nA, (q, info["eps_Q"] / nA)
        worst = max(worst, err / info["eps_Q"])
    assert worst > 0.3          # the bound is not vacuous


def test_eps_Q_N512(eim512):
    """At N = 512: eps_Q dominates the symbol sup-norm of the difference on a 4x finer grid for every test
    query, and the dense 2-norm on the first 5 (quick) / 50 (full) queries; eps_Q <= 1e-8 ||A_I||_2."""
    e = eim512
    Lf = 4 * e.L
    xif = e.basis_on(Lf)
    n_dense = 50 if FULL else 5
    for i, q in enumerate(_queries(30)):
        info = e.eps_Q(q)
        phi, _ = family_grid(Lf, (q.G, q.M, q.Y), e.h, e.K)
        fine = info["scale"] * np.max(np.abs(phi - info["ell"] @ xif))
        assert fine <= info["eps_Q"], (q, fine / info["eps_Q"])
        A = free_matrices(q, 512).A.dense()
        nA = norm(A, 2)
        assert info["eps_Q"] <= 1e-8 * nA, (q, info["eps_Q"] / nA)
        if i < n_dense:
            assert norm(A - e.A_Q(q, ell=info["ell"]).dense(), 2) <= info["eps_Q"], q


def test_affine_free_system(eim64):
    q = eim_test_queries(5)[4]
    fs_Q, info = eim64.affine_free_system(q)
    fs = free_matrices(q, 64)
    v = np.random.default_rng(2).standard_normal(fs.n)
    assert norm(fs_Q.A.matvec(v) - fs.A.matvec(v)) <= info["eps_Q"] * norm(v)
    assert abs(fs_Q.A.entry(fs.n + 1) - fs.A.entry(fs.n + 1).real) <= info["eps_Q"]
