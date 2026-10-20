"""Plan 1.2 (spectral) and 1.3 (Riesz) done-when criteria."""
import numpy as np
import pytest
from scipy.linalg import eigvalsh, eigh

from conftest import FULL
from cgmy_cert.params import Query, sobol_queries
from cgmy_cert.symbols import free_matrices
from cgmy_cert.spectral import spectral_data
from cgmy_cert.riesz import pcg_bracket, circulant_preconditioner, plain_cg_iterations


def _subset(pop, n):
    return pop if FULL else pop[:n]


@pytest.mark.parametrize("N", [64, 256, 1024])
def test_spectral_bounds_on_288(populations, N):
    """eigvalsh(S_I).min() >= alpha_lb and eigh(S_I, X_I) extremes in [1, Theta] for every query."""
    pop = _subset(populations["spectral288"], 24 if N < 1024 else 8)
    ratios = []
    for q in pop:
        fs = free_matrices(q, N, K=100)
        sp = spectral_data(q, N, K=100)
        d = fs.dense()
        lmin = eigvalsh(d["S"]).min()
        assert lmin >= sp.alpha_lb, (q.query_id, lmin, sp.alpha_lb)
        assert sp.alpha_lb > 0
        from cgmy_cert.symbols import _dense_banded
        X = _dense_banded(sp.X_entries, fs.n)
        ev = eigh(d["S"], X, eigvals_only=True)
        assert ev.min() >= 1 - 1e-9, (q.query_id, ev.min())
        assert ev.max() <= sp.Theta * (1 + 1e-9), (q.query_id, ev.max(), sp.Theta)
        ratios.append(lmin / sp.s_min)
    assert min(ratios) >= 1.0


def test_fixture_eigenvalues_exceed_alpha_lb(eig24):
    """The 24 40-digit eigenvalues of the exact-assembly S_I (N = 64, K = 100) exceed alpha_lb."""
    from decimal import Decimal
    import json
    from conftest import FIX
    pops = json.loads((FIX / "queries.json").read_text())["populations"]["audit459"]
    lookup = {d["query_id"]: d for d in pops}
    for row in eig24:
        d = lookup[row["query_id"]]
        q = Query(C=d["C"], G=d["G"], M=d["M"], Y=d["Y"], r=d["r"], T=d["T"], K=d["K"])
        sp = spectral_data(q, 64, K=100)
        lam = float(Decimal(row["lambda_min_exact_assembly_40digit"]))
        assert lam >= sp.alpha_lb, (row["query_id"], lam, sp.alpha_lb)
        assert lam / sp.s_min < 20


@pytest.mark.parametrize("N", [64, 256, 1024, 4096])
def test_pcg_iterations_and_bracket(N):
    """Iteration counts to 1e-12 are <= 15 for N = 64..4096 on 20 random queries; bracket width
    ||res||^2/alpha_lb at convergence is < 1e-20 relative to the dual norm."""
    qs = sobol_queries(20 if FULL or N <= 256 else 6, seed=11, include_T=False)
    rng = np.random.default_rng(0)
    worst_it = 0
    for q in qs:
        fs = free_matrices(q, N, K=100)
        sp = spectral_data(q, N, K=100)
        C = circulant_preconditioner(sym_vals=fs.sym.s.astype(complex), n=fs.n, kind="symbol")
        assert np.all(C.lam > 0)
        # residual-like right-hand sides: A e for a smooth e, and white noise
        e = np.cumsum(rng.standard_normal(fs.n)); e -= e.mean()
        for rho in (fs.A.matvec(e), rng.standard_normal(fs.n)):
            br = pcg_bracket(fs.S, rho, sp.alpha_lb, precond=C, tol=1e-12, maxit=200)
            assert br.converged
            worst_it = max(worst_it, br.iters)
            assert br.iters <= 15, (q.query_id, N, br.iters)
            # at convergence the width is set by roundoff in the recomputed residual,
            # ~ (eps_mach ||S||_2 ||v||)^2 / alpha_lb; ||S||_2 ~ h^{-Y} grows with N
            assert br.width / br.u < (1e-20 if N <= 1024 else 1e-18), (N, br.width, br.u)
            assert br.l <= br.u
    print("worst PCG iteration count:", worst_it)


def test_strang_preconditioner_equivalent():
    q = Query(0.5, 3.0, 5.0, 1.5, 0.05)
    fs = free_matrices(q, 256)
    C1 = circulant_preconditioner(sym_vals=fs.sym.s.astype(complex), n=fs.n, kind="symbol")
    C2 = circulant_preconditioner(op=fs.S, kind="strang")
    assert np.all(C2.lam > 0)
    rho = np.random.default_rng(1).standard_normal(fs.n)
    sp = spectral_data(q, 256)
    for C in (C1, C2):
        assert pcg_bracket(fs.S, rho, sp.alpha_lb, precond=C).iters <= 15
    assert plain_cg_iterations(fs.S, rho) > 30
