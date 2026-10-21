"""Exact-assembly direct solver and references (plan 1.5).

Conventions (v8 Section 2.6, kept):
    raw coefficients c^n = E z^n + b^n;  b^n carries the prescribed coefficients
        left (p-1) raw entries  = K e^{-r tau_n} sqrt(h),   right (p-1) raw entries = 0;
    free CN system  M_I (z^{n+1}-z^n)/dt + A_I (z^{n+1}+z^n)/2 = f_I^{n+1/2},
        f_I^{n+1/2} = -E^T [ Mbar (b^{n+1}-b^n)/dt + Abar (b^{n+1}+b^n)/2 ]      (raw forcing fbar = 0);
    z_h^0 = free part of the raw L2 payoff projection with the prescribed rows overwritten.

v9 change (documented in Section 2 of the paper): the raw matrices Mbar, Abar are the Toeplitz
sections T_{N+p-1}(m), T_{N+p-1}(a_mu) of the full cardinal splines, i.e. the prescribed boundary
functions are not truncated to Omega.  For the mass and drift couplings this coincides with v8's
Omega-quadrature (the overlaps with free functions lie inside Omega); for the fractional coupling it
is the exact Galerkin value for the full-line spline.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve_banded
from scipy.sparse.linalg import gmres, LinearOperator

from .params import Query, Mesh, P_ORDER
from .symbols import FreeSystem, free_matrices, banded_from_toeplitz, mass_symbol, _poly_entries
from .riesz import Circulant, circulant_preconditioner


# ---------------------------------------------------------------- cardinal B-splines

def cardinal_bspline(x, p: int):
    x = np.asarray(x, dtype=float)
    if p <= 0:
        return np.zeros_like(x)
    if p == 1:
        return np.where((x >= 0) & (x < 1), 1.0, 0.0)
    return (x / (p - 1)) * cardinal_bspline(x, p - 1) + ((p - x) / (p - 1)) * cardinal_bspline(x - 1, p - 1)


def raw_basis_eval(x, msh: Mesh) -> np.ndarray:
    """Matrix Phi[i, k] = phi_{h,k}(x_i) for the raw basis k = -(p-1), ..., N-1."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    xi = (x - msh.x_min) / msh.h
    ks = np.arange(-(msh.p - 1), msh.N)
    return msh.h ** (-0.5) * cardinal_bspline(xi[:, None] - ks[None, :], msh.p)


def free_basis_eval(x, msh: Mesh) -> np.ndarray:
    return raw_basis_eval(x, msh)[:, msh.free_slice]


# ---------------------------------------------------------------- payoff projection

def _gauss_legendre(n: int):
    return np.polynomial.legendre.leggauss(n)


def payoff_load_vector(msh: Mesh, K: float, n_gauss: int = 12) -> np.ndarray:
    """b^0_k = int_Omega (K - e^x)^+ phi_k(x) dx, element-wise Gauss--Legendre (the kink at
    log K is a grid node because log K - x_min = 8 = (N/2) h, so every element integrand is smooth)."""
    xg, wg = _gauss_legendre(n_gauss)
    nodes = msh.nodes
    b = np.zeros(msh.n_raw)
    for j in range(msh.N):
        a, c = nodes[j], nodes[j + 1]
        x = 0.5 * (c - a) * xg + 0.5 * (c + a)
        w = 0.5 * (c - a) * wg
        psi = np.maximum(K - np.exp(x), 0.0)
        if not np.any(psi):
            continue
        xi = (x - msh.x_min) / msh.h
        for k in range(j - msh.p + 1, j + 1):          # active functions on element j
            phi = msh.h ** (-0.5) * cardinal_bspline(xi - k, msh.p)
            b[k + msh.p - 1] += np.sum(w * psi * phi)
    return b


def raw_mass_omega_banded(msh: Mesh, n_gauss: int = 8) -> np.ndarray:
    """The raw mass matrix (Mbar)_{kk'} = int_Omega phi_k phi_k' dx in LAPACK banded storage.
    It equals the Toeplitz section T(m) except in the (p-1) x (p-1) corner blocks, where the
    boundary functions are truncated to Omega (v8 convention for the initial L2 projection;
    assembled element-wise with Gauss--Legendre, exact for the polynomial integrands)."""
    xg, wg = _gauss_legendre(n_gauss)
    p = msh.p; bw = p - 1
    ab = np.zeros((2 * bw + 1, msh.n_raw))
    for j in range(msh.N):
        xi = 0.5 * msh.h * (xg + 1) / msh.h + j          # local coordinate on element j: xi in [j, j+1]
        w = 0.5 * msh.h * wg
        ks = list(range(j - p + 1, j + 1))
        vals = [msh.h ** (-0.5) * cardinal_bspline(xi - k, p) for k in ks]
        for a_, ka in enumerate(ks):
            for b_, kb in enumerate(ks):
                i, jj = ka + p - 1, kb + p - 1
                ab[bw + i - jj, jj] += np.sum(w * vals[a_] * vals[b_])
    return ab


def raw_mass_banded(msh: Mesh) -> np.ndarray:
    """Toeplitz raw mass T_{N+p-1}(m) (full cardinal splines) in banded storage."""
    Me = _poly_entries(lambda t: mass_symbol(t, msh.p), msh.p)
    return banded_from_toeplitz(Me, msh.n_raw, msh.p - 1)


def initial_raw_coefficients(q: Query, msh: Mesh) -> np.ndarray:
    """c^0 = Mbar_Omega^{-1} b^0 with the prescribed rows overwritten (v8 convention; Mbar_Omega is
    the Omega-quadrature mass matrix, so that constants are reproduced exactly up to the boundary)."""
    b0 = payoff_load_vector(msh, q.K)
    bw = msh.p - 1
    c = solve_banded((bw, bw), raw_mass_omega_banded(msh), b0)
    c[: bw] = q.K * np.sqrt(msh.h)
    c[msh.n_raw - bw:] = 0.0
    return c


def initial_free_vector(q: Query, msh: Mesh) -> np.ndarray:
    return initial_raw_coefficients(q, msh)[msh.free_slice]


# ---------------------------------------------------------------- lifting and forcing

def lifting(q: Query, msh: Mesh, tau: np.ndarray) -> np.ndarray:
    """Left prescribed coefficient value b_L(tau) = K e^{-r tau} sqrt(h) (same for the p-1 left rows)."""
    return q.K * np.exp(-q.r * np.asarray(tau)) * np.sqrt(msh.h)


@dataclass
class FreeForcing:
    """f_I^{n+1/2} for n = 0..N_T-1 as columns, plus the coupling vectors."""
    tau: np.ndarray
    dt: float
    bL: np.ndarray            # b_L(tau_n)
    uM: np.ndarray            # E^T Mbar e_bdry  (sum over the p-1 left boundary functions)
    uA: np.ndarray            # E^T Abar e_bdry
    F: np.ndarray             # n_free x N_T

    def col(self, n: int) -> np.ndarray:
        return self.F[:, n]


def free_forcing(fs: FreeSystem, T: float, N_T: int) -> FreeForcing:
    msh, q = fs.mesh, fs.q
    p = msh.p; n = msh.n_free
    tau = np.arange(N_T + 1) * (T / N_T)
    dt = T / N_T
    bL = lifting(q, msh, tau)
    # free index k (0..n-1) couples to boundary raw indices k' = -(p-1)..-1 through t_{k - k'} = t_{k+j}, j=1..p-1
    uM = np.zeros(n); uA = np.zeros(n)
    for j in range(1, p):
        for k in range(0, min(n, p)):
            uM[k] += fs.M_entries.get(k + j, 0.0)
        lags = np.arange(n) + j
        uA += np.array([fs.A.entry(int(m)) for m in lags]).real
    db = (bL[1:] - bL[:-1]) / dt
    sb = 0.5 * (bL[1:] + bL[:-1])
    F = -(uM[:, None] * db[None, :] + uA[:, None] * sb[None, :])
    return FreeForcing(tau=tau, dt=dt, bL=bL, uM=uM, uA=uA, F=F)


# ---------------------------------------------------------------- CN march

@dataclass
class Trajectory:
    Z: np.ndarray             # n_free x (N_T+1)
    iters: list               # per-step iteration counts (GMRES) or []
    solver: str
    dt: float
    T: float
    N_T: int


def _B_dense(fs: FreeSystem, dt: float, sign: float) -> np.ndarray:
    n = fs.n
    A = fs.A.dense()
    from .symbols import _dense_banded
    M = _dense_banded(fs.M_entries, n)
    return M + sign * 0.5 * dt * A


def cn_march(fs: FreeSystem, z0: np.ndarray, T: float, N_T: int, solver: str = "gmres",
             forcing: FreeForcing | None = None, rtol: float = 1e-13, restart: int = 50,
             maxiter: int = 200, precond: str = "symbol", x0_traj: np.ndarray | None = None) -> Trajectory:
    """B_+ z^{n+1} = B_- z^n + dt f^{n+1/2},  B_pm = M_I +- dt/2 A_I.
    solver = 'lu' (dense, reference only, N <= 1024) or 'gmres' (matrix-free, circulant preconditioner,
    warm-started at z^n, or at x0_traj[:, n+1] when a candidate trajectory is given: the correction fallback)."""
    n = fs.n
    dt = T / N_T
    if forcing is None:
        forcing = free_forcing(fs, T, N_T)
    Z = np.zeros((n, N_T + 1)); Z[:, 0] = z0
    iters = []
    Mmv = fs.M_matvec

    def Bminus(v):
        return Mmv(v) - 0.5 * dt * fs.A.matvec(v)

    def Bplus(v):
        return Mmv(v) + 0.5 * dt * fs.A.matvec(v)

    if solver == "lu":
        lu = lu_factor(_B_dense(fs, dt, +1.0))
        for k in range(N_T):
            rhs = Bminus(Z[:, k]) + dt * forcing.col(k)
            Z[:, k + 1] = lu_solve(lu, rhs)
        return Trajectory(Z=Z, iters=[], solver="lu", dt=dt, T=T, N_T=N_T)
    if solver == "gmres":
        sym = fs.sym
        lam_B = sym.m + 0.5 * dt * sym.a         # symbol of B_+
        C = circulant_preconditioner(sym_vals=lam_B, n=n, kind="symbol") if precond == "symbol" else None
        Bop = LinearOperator((n, n), matvec=Bplus, dtype=float)
        Pop = LinearOperator((n, n), matvec=(lambda v: C.solve(v).real), dtype=float) if C is not None else None
        for k in range(N_T):
            rhs = Bminus(Z[:, k]) + dt * forcing.col(k)
            cnt = [0]

            def cb(x, _c=cnt):
                _c[0] += 1
            x0 = Z[:, k].copy() if x0_traj is None else np.array(x0_traj[:, k + 1], dtype=float)
            x, info = gmres(Bop, rhs, x0=x0, M=Pop, rtol=rtol, atol=0.0, restart=restart,
                            maxiter=maxiter, callback=cb, callback_type="pr_norm")
            if info != 0:
                raise RuntimeError(f"GMRES failed at step {k}: info={info}")
            Z[:, k + 1] = x
            iters.append(cnt[0])
        return Trajectory(Z=Z, iters=iters, solver="gmres", dt=dt, T=T, N_T=N_T)
    raise ValueError(solver)


def reference_trajectory(q: Query, N: int, N_T: int, K: int = 100, solver: str = "gmres", **kw):
    """Convenience: assemble, project, march.  Returns (fs, forcing, trajectory)."""
    fs = free_matrices(q, N, K=K)
    z0 = initial_free_vector(q, fs.mesh)
    forcing = free_forcing(fs, q.T, N_T)
    traj = cn_march(fs, z0, q.T, N_T, solver=solver, forcing=forcing, **kw)
    return fs, forcing, traj


def mass_norm(fs: FreeSystem, v: np.ndarray) -> float:
    return float(np.sqrt(max(v @ fs.M_matvec(v), 0.0)))


def spot_price_factor(fs: FreeSystem, x) -> np.ndarray:
    """beta_I(x) = sqrt(Phi_I(x)^T M_I^{-1} Phi_I(x)) (v8 eq. (4.17))."""
    Phi = free_basis_eval(x, fs.mesh)
    bw = fs.mesh.p - 1
    sol = solve_banded((bw, bw), fs.M_banded(), Phi.T)
    return np.sqrt(np.sum(Phi.T * sol, axis=0))
