"""Empirical interpolation of the fractional symbol family (plan 2.1; Lemma 5.1 of the note).

Family:  Phi_nu(theta), nu = (G, M, Y), sampled on the L-grid of `symbols()` (theta_j = -pi + 2 pi j/L,
L = N_h max(4, ceil(256/N_h))), alias sum truncated at |k| <= K = 100.  The family depends on N through
h (Gh, Mh), so there is one EIM per N.  Real and imaginary parts are stacked into a vector of length 2L;
magic points are indices into that stacked vector.

Offline (`build_eim`): training set = tensor grid 16^3 in (log G, log M, Y) over the box, each snapshot
normalised by its own sup-norm; standard greedy EIM (largest residual sup-norm -> snapshot, largest
entry of that residual -> magic point); basis functions xi_q = residual / residual(magic_q), so the
interpolation matrix B_{iq} = xi_q(magic_i) is lower triangular with unit diagonal.  The greedy runs to
q_max; because B is lower triangular, the first Q coefficients of the q_max interpolant are those of
the Q-term interpolant, so one run gives the error curve for every Q.  Q is the smallest value with
test sup-error <= tol (relative to max|Phi_nu|, per nu) on 1000 Sobol nu (seed 2026).

Stored per N (`data/eim/eim_N{N}_K{K}.npz`): magic points, B, the basis samples xi_q and xi_q' on the
L-grid, the Toeplitz entries of T(xi_q) for lags |m| <= n_free + p - 2 (what `free_forcing` needs for
the lifting columns), the selected nu_q, the training-grid definition and the error histories.

Online, for a query mu:
    ell(nu)   = B^{-1} Phi_nu(magic)                      O(Q K) symbol evaluations + triangular solve
    A_Q(mu)   = -(r - omega_S) D_I + lambda_0 M_I - C Gamma(-Y) h^{-Y} sum_q ell_q T(xi_q)
    eps_Q(mu) = |C Gamma(-Y)| h^{-Y} ( max_grid |e| + (pi/L) max_grid |e'| ),  e = Phi_nu - I_Q Phi_nu,
with the exact (K-truncated) symbol and its derivative at the query, cost O(L K).

Remark (rigour).  The assembled A_I and A_Q are Toeplitz sections whose entries are the L-point DFT
coefficients of the samples; since L >= 2 n_free, T_n is a compression of the L x L circulant with
eigenvalues equal to the samples (up to the unitary diagonal (-1)^m), so ||A_I - A_Q||_2 <=
scale * max_grid |e| already.  The derivative term makes eps_Q bound the continuous sup-norm of the
symbol difference as well (the only non-enclosed step being sup|e'| taken on the grid, as in
`spectral.py`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import pi
from scipy.linalg import solve_triangular
from scipy.stats import qmc

from .params import Query, Mesh, BOX, P_ORDER
from .symbols import (bspline_ft_sq, bspline_ft_sq_prime, theta_grid, fourier_coefficients, ToeplitzOp,
                      FreeSystem, free_matrices, _poly_entries, mass_symbol, drift_symbol)

ROOT = Path(__file__).resolve().parents[1]
EIM_DIR = ROOT / "data" / "eim"
TRAIN_SHAPE = (16, 16, 16)
TEST_SEED = 2026
N_TEST = 1000
TOL_DEFAULT = 1e-9
TOL_SWEEP = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11)
GRADED_L0, GRADED_NFINE = 96, 32          # N-independent node set of eps_Q_graded (161 nodes)


# ------------------------------------------------------------------ family evaluation

def half_alias(theta: np.ndarray, delta: float, Y: float, sign: int, K: int, p: int = P_ORDER,
               with_derivative: bool = False):
    """sum_{|k|<=K} |N_p^(theta_k)|^2 (delta + i sign theta_k)^Y and its theta-derivative.
    Phi_nu = half_alias(Gh, +1) + half_alias(Mh, -1)  (same arithmetic as `fractional_alias`)."""
    theta = np.asarray(theta, dtype=float)
    out = np.zeros(theta.shape, dtype=complex)
    dout = np.zeros(theta.shape, dtype=complex) if with_derivative else None
    chunk = max(1, int(2e6 // max(theta.size, 1)))
    ks = np.arange(-K, K + 1)
    for i in range(0, ks.size, chunk):
        tt = theta[None, :] + 2 * pi * ks[i:i + chunk][:, None]
        w = bspline_ft_sq(tt, p)
        z = delta + 1j * sign * tt
        pz = z ** Y
        out += np.sum(w * pz, axis=0)
        if with_derivative:
            dout += np.sum(bspline_ft_sq_prime(tt, p) * pz + w * (1j * sign * Y) * pz / z, axis=0)
    return out, dout


def family(theta: np.ndarray, nu, h: float, K: int, with_derivative: bool = False):
    """Phi_nu(theta) (and Phi_nu'(theta)) for nu = (G, M, Y)."""
    G, M, Y = nu
    a, da = half_alias(theta, G * h, Y, +1, K, with_derivative=with_derivative)
    b, db = half_alias(theta, M * h, Y, -1, K, with_derivative=with_derivative)
    return a + b, (da + db if with_derivative else None)


def mirror(L: int) -> np.ndarray:
    """Index of -theta_j on the grid theta_j = -pi + 2 pi j/L (theta_0 = -pi is its own mirror)."""
    return (-np.arange(L)) % L


def symmetrise(z: np.ndarray, derivative: bool = False) -> np.ndarray:
    """(z(theta) + conj z(-theta))/2 on the uniform grid (last axis), derivative: (z - conj z(-.))/2.
    The K-truncated alias sum is conjugate-symmetric except for O(eps_K) terms at theta near -pi; the
    assembled A_I keeps the real part of its Toeplitz entries, i.e. it is T of the symmetrised samples.
    Symmetrised samples are exactly conjugate-symmetric in floating point, and the greedy rank-one
    updates preserve that, so every basis function has real Toeplitz entries."""
    zm = np.conj(z[..., mirror(z.shape[-1])])
    return 0.5 * (z - zm) if derivative else 0.5 * (z + zm)


def family_grid(L: int, nu, h: float, K: int, with_derivative: bool = False):
    """Symmetrised Phi_nu (and Phi_nu') on theta_grid(L)."""
    phi, dphi = family(theta_grid(L), nu, h, K, with_derivative)
    return symmetrise(phi), (symmetrise(dphi, True) if with_derivative else None)


def stack(z: np.ndarray) -> np.ndarray:
    return np.concatenate([z.real, z.imag], axis=-1)


def l_grid(N: int, p: int = P_ORDER) -> int:
    n = Mesh(N=N, p=p).n_free
    return n * max(4, int(np.ceil(256 / n)))


def training_grid(shape=TRAIN_SHAPE) -> dict:
    g = np.exp(np.linspace(np.log(BOX["G"][0]), np.log(BOX["G"][1]), shape[0]))
    m = np.exp(np.linspace(np.log(BOX["M"][0]), np.log(BOX["M"][1]), shape[1]))
    y = np.linspace(BOX["Y"][0], BOX["Y"][1], shape[2])
    return dict(G=g, M=m, Y=y)


def eim_test_nus(n: int = N_TEST, seed: int = TEST_SEED) -> np.ndarray:
    """The (G, M, Y) of `sobol_queries(n, seed, include_T=False)` (the 5-d test queries of the tests)."""
    return np.array([(q.G, q.M, q.Y) for q in eim_test_queries(n, seed)])


def eim_test_queries(n: int = N_TEST, seed: int = TEST_SEED) -> list[Query]:
    import warnings
    from .params import sobol_queries
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # n not a power of 2
        return sobol_queries(n, seed=seed, include_T=False, prefix="eimtest")


def training_snapshots(theta: np.ndarray, h: float, K: int, grid: dict):
    """Tensor-grid snapshots and their theta-derivatives (complex, n_train x L) and nus (n_train x 3),
    using the additive split Phi_(G,M,Y) = P_G(G, Y) + P_M(M, Y)  (2 x 16 x 16 alias sums instead of
    16^3; same floating-point operations as `family`)."""
    g, m, y = grid["G"], grid["M"], grid["Y"]
    hG = [[half_alias(theta, G * h, Y, +1, K, with_derivative=True) for Y in y] for G in g]
    hM = [[half_alias(theta, M * h, Y, -1, K, with_derivative=True) for Y in y] for M in m]
    out = []
    for k in (0, 1):
        PG = np.array([[v[k] for v in row] for row in hG])                    # (nG, nY, L)
        PM = np.array([[v[k] for v in row] for row in hM])
        out.append((PG[:, None, :, :] + PM[None, :, :, :]).reshape(-1, theta.size))
    nus = np.stack(np.meshgrid(g, m, y, indexing="ij"), axis=-1).reshape(-1, 3)
    return out[0], out[1], nus


def _eval_many(L, nus, h, K, nproc=1):
    if nproc <= 1:
        return np.array([family_grid(L, nu, h, K)[0] for nu in nus])
    from multiprocessing import Pool
    with Pool(nproc) as pool:
        return np.array(pool.starmap(_family_only, [(L, tuple(nu), h, K) for nu in nus], chunksize=8))


def _family_only(L, nu, h, K):
    return family_grid(L, nu, h, K)[0]


# ------------------------------------------------------------------ the EIM object

@dataclass
class EIM:
    N: int
    K: int
    L: int
    p: int
    n: int                     # n_free
    extra: int                 # extra Toeplitz lags kept (p - 1)
    magic: np.ndarray          # (q_max,) indices into the stacked 2L vector
    B: np.ndarray              # (q_max, q_max) lower triangular, unit diagonal
    xi: np.ndarray             # (q_max, L) complex basis samples
    dxi: np.ndarray            # (q_max, L) complex basis derivative samples
    lags: np.ndarray           # (q_max, 2(n+extra)-1) real Toeplitz entries of T(xi_q)
    nus: np.ndarray            # (q_max, 3) selected training parameters
    rec: np.ndarray            # (q_max, q_max) greedy recursion: xi_q = (Phi_q/sc_q - sum_l rec[q,l] xi_l)/rho_q
    rho: np.ndarray            # (q_max,)
    sc: np.ndarray             # (q_max,) sup-norm normalisation of the selected snapshots
    Q: int                     # selected rank for tol
    tol: float
    train_err: np.ndarray      # (q_max+1,) max relative training residual after q terms
    test_err: np.ndarray       # (q_max+1,) max relative test residual after q terms
    sv: np.ndarray             # normalised singular values of the training snapshot matrix
    meta: dict

    @property
    def h(self) -> float:
        return Mesh(N=self.N, p=self.p).h

    @property
    def theta(self) -> np.ndarray:
        return theta_grid(self.L)

    def magic_theta(self, Q: int | None = None):
        """(theta at each magic point, part) with part 0 = real, 1 = imaginary."""
        Q = self.Q if Q is None else Q
        mg = self.magic[:Q]
        return self.theta[mg % self.L], mg // self.L

    def Q_for_tol(self, tol: float) -> int:
        ok = np.nonzero(self.test_err <= tol)[0]
        return int(ok[0]) if ok.size else -1

    # -------------------------------------------------------------- online
    def ell(self, nu, Q: int | None = None) -> np.ndarray:
        """Interpolation coefficients: B_Q ell = Phi_nu(magic) (triangular solve)."""
        Q = self.Q if Q is None else Q
        th, part = self.magic_theta(Q)
        thm = self.theta[mirror(self.L)[self.magic[:Q] % self.L]]
        vals = 0.5 * (family(th, nu, self.h, self.K)[0] + np.conj(family(thm, nu, self.h, self.K)[0]))
        rhs = np.where(part == 0, vals.real, vals.imag)
        return solve_triangular(self.B[:Q, :Q], rhs, lower=True, unit_diagonal=True)

    def interpolant(self, ell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q = ell.size
        return ell @ self.xi[:Q], ell @ self.dxi[:Q]

    def eps_Q(self, q: Query, Q: int | None = None, ell: np.ndarray | None = None) -> dict:
        """Evaluated EIM error bound in operator norm (Lemma 5.1), with its two components."""
        Q = self.Q if Q is None else Q
        nu = (q.G, q.M, q.Y)
        if ell is None:
            ell = self.ell(nu, Q)
        phi, dphi = family_grid(self.L, nu, self.h, self.K, with_derivative=True)
        I, dI = self.interpolant(ell)
        scale = abs(q.C * q.gammaY) * self.h ** (-q.Y)
        e_grid = float(np.max(np.abs(phi - I)))
        e_der = float(np.max(np.abs(dphi - dI)))
        return dict(eps_Q=scale * (e_grid + (pi / self.L) * e_der), grid=scale * e_grid,
                    deriv=scale * (pi / self.L) * e_der, scale=scale, ell=ell)

    def eps_Q_curve(self, q: Query, Qs) -> np.ndarray:
        """eps_Q(mu) for several Q (one symbol evaluation; nested coefficients since B is triangular)."""
        nu = (q.G, q.M, q.Y)
        ell = self.ell(nu, max(Qs))
        phi, dphi = family_grid(self.L, nu, self.h, self.K, with_derivative=True)
        scale = abs(q.C * q.gammaY) * self.h ** (-q.Y)
        out = []
        for Q in Qs:
            I, dI = self.interpolant(ell[:Q])
            out.append(scale * (np.max(np.abs(phi - I)) + (pi / self.L) * np.max(np.abs(dphi - dI))))
        return np.array(out)

    # -------------------------------------------------------------- N-independent eps_Q (follow-up 2, 2b)
    def graded(self, L0: int = GRADED_L0, n_fine: int = GRADED_NFINE) -> "GradedNodes":
        """Query-independent node set and precomputed alias data for eps_Q_graded (cached)."""
        key = (L0, n_fine)
        cache = self.__dict__.setdefault("_graded", {})
        if key not in cache:
            cache[key] = GradedNodes(self, L0, n_fine)
        return cache[key]

    def eps_Q_graded(self, q: Query, Q: int | None = None, ell: np.ndarray | None = None,
                     L0: int = GRADED_L0, n_fine: int = GRADED_NFINE) -> dict:
        """eps_Q(mu) from a rigorous bound on sup_theta |e_nu(theta)| over the N-independent node set
        (cell-wise derivative margin, as in spectral._grid_extreme), cost O(n_nodes K), independent of N.
        Valid for the assembled matrices (sup over theta >= max over the L-grid, compression argument) and
        for the continuous symbol."""
        Q = self.Q if Q is None else Q
        nu = (q.G, q.M, q.Y)
        if ell is None:
            ell = self.ell(nu, Q)
        g = self.graded(L0, n_fine)
        sup = g.sup_error(nu, ell)
        scale = abs(q.C * q.gammaY) * self.h ** (-q.Y)
        return dict(eps_Q=scale * sup, scale=scale, ell=ell, n_nodes=g.theta.size)

    def eps_Q_graded_curve(self, q: Query, Qs, **kw) -> np.ndarray:
        nu = (q.G, q.M, q.Y)
        ell = self.ell(nu, max(Qs))
        g = self.graded(**kw)
        scale = abs(q.C * q.gammaY) * self.h ** (-q.Y)
        v, dv = g.symbol(nu)
        return np.array([scale * g.sup_from(v, dv, ell[:Q]) for Q in Qs])

    def Q_for_target(self, q: Query, eps: float, cbar_norm: float | None = None, alpha: float | None = None,
                     T: float | None = None, safety: float = 10.0) -> int:
        """Smallest stored Q with eps_Q(mu) ||cbar||_2 sqrt(T / alpha) <= eps / safety (follow-up 2, 2d).
        Defaults (a selection heuristic; the certificate itself always uses the actual allowance):
        cbar_norm = 1.1 ||c_raw^0||_2 (the payoff dominates the raw coefficient norm), alpha = r min m
        (a rigorous lower bound of alpha_lb since s_mu - r m >= 0).  Returns q_max if none suffices."""
        from .direct import initial_raw_coefficients
        from .symbols import mass_symbol
        T = q.T if T is None else T
        if cbar_norm is None:
            cbar_norm = 1.1 * float(np.linalg.norm(initial_raw_coefficients(q, Mesh(N=self.N, p=self.p))))
        if alpha is None:
            alpha = q.r * float(np.min(mass_symbol(np.linspace(-pi, pi, 2001))))
        Qs = list(range(1, self.magic.size + 1))
        curve = self.eps_Q_graded_curve(q, Qs)
        ok = [Q for Q, e in zip(Qs, curve) if e * cbar_norm * np.sqrt(T / alpha) <= eps / safety]
        return ok[0] if ok else self.magic.size

    def set_Q(self, Q: int, **why) -> None:
        """Raise/lower the operational rank (e.g. after the operator-norm check of scripts/run_eim.py)."""
        self.meta.setdefault("Q_tol", self.Q)
        self.meta.update(Q=Q, Q_selection=why)
        self.Q = Q

    def A_Q_lags(self, q: Query, ell: np.ndarray) -> np.ndarray:
        h = self.h
        nl = self.n + self.extra
        D = _poly_lags(lambda t: 1j / h * drift_symbol(t, self.p), self.p, nl)
        Mm = _poly_lags(lambda t: mass_symbol(t, self.p), self.p, nl)
        cg = q.C * q.gammaY * h ** (-q.Y)
        return -(q.r - q.omega_S) * D + q.lambda0 * Mm - cg * (ell @ self.lags[:ell.size])

    def A_Q(self, q: Query, Q: int | None = None, ell: np.ndarray | None = None) -> ToeplitzOp:
        """A_Q(mu) as one ToeplitzOp (first columns summed, then one operator)."""
        if ell is None:
            ell = self.ell((q.G, q.M, q.Y), Q)
        return ToeplitzOp.from_lags(self.A_Q_lags(q, ell), self.n, self.extra)

    def basis_on(self, L: int, Q: int | None = None, with_derivative: bool = False):
        """xi_q (and xi_q') on theta_grid(L) for any L, by the greedy recursion applied to the selected
        (symmetrised) snapshots evaluated on that grid; for L = self.L these are the stored samples."""
        Q = self.Q if Q is None else Q
        xi = np.zeros((Q, L), dtype=complex)
        dxi = np.zeros_like(xi) if with_derivative else None
        for q in range(Q):
            phi, dphi = family_grid(L, self.nus[q], self.h, self.K, with_derivative)
            xi[q] = (phi / self.sc[q] - self.rec[q, :q] @ xi[:q]) / self.rho[q]
            if with_derivative:
                dxi[q] = (dphi / self.sc[q] - self.rec[q, :q] @ dxi[:q]) / self.rho[q]
        return (xi, dxi) if with_derivative else xi

    def affine_free_system(self, q: Query, Q: int | None = None) -> tuple[FreeSystem, dict]:
        """The free system with A_I replaced by A_Q (S, M, K1, D and the spectral data unchanged:
        built from the exact symbol, as the certificate needs).  Returns (fs_Q, eps info)."""
        fs = free_matrices(q, self.N, K=self.K)
        if fs.sym.L != self.L:
            raise ValueError("L-grid mismatch")
        info = self.eps_Q(q, Q)
        fs.A = self.A_Q(q, ell=info["ell"])
        return fs, info

    # -------------------------------------------------------------- io
    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path is not None else eim_path(self.N, self.K)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, N=self.N, K=self.K, L=self.L, p=self.p, n=self.n, extra=self.extra, magic=self.magic,
                 B=self.B, xi=self.xi, dxi=self.dxi, lags=self.lags, nus=self.nus,
                 rec=self.rec, rho=self.rho, sc=self.sc, Q=self.Q, tol=self.tol,
                 train_err=self.train_err, test_err=self.test_err, sv=self.sv, meta=json.dumps(self.meta))
        return path


def _poly_lags(fun, p: int, nl: int) -> np.ndarray:
    ent = _poly_entries(fun, p)
    t = np.zeros(2 * nl - 1)
    for m, v in ent.items():
        t[nl - 1 + m] = v
    return t


def build_eim(N: int, K: int = 100, q_max: int = 50, tol: float = TOL_DEFAULT, nproc: int = 1,
              n_test: int = N_TEST, verbose: bool = False) -> EIM:
    import time
    t0 = time.time()
    msh = Mesh(N=N); h = msh.h; n = msh.n_free; p = msh.p; extra = p - 1
    L = l_grid(N, p); theta = theta_grid(L)
    grid = training_grid()
    snaps, dsnaps, nus_train = training_snapshots(theta, h, K, grid)
    t_snap = time.time() - t0
    R = stack(symmetrise(snaps)); dR = stack(symmetrise(dsnaps, True))
    del snaps, dsnaps
    sc_all = np.max(np.abs(R), axis=1)
    R /= sc_all[:, None]; dR /= sc_all[:, None]
    sv = np.linalg.svd(R, compute_uv=False); sv = sv / sv[0]
    # greedy; the residual of the derivative snapshots follows the same rank-one updates, so the basis
    # derivatives are exactly the derivatives of the basis functions
    magic, sel, xs, dxs, mult = [], [], [], [], []
    train_err = [float(np.max(np.abs(R)))]
    for q in range(q_max):
        err = np.max(np.abs(R), axis=1)
        j = int(np.argmax(err))
        i = int(np.argmax(np.abs(R[j])))
        col = R[:, i].copy()
        x = R[j] / R[j, i]; dx = dR[j] / R[j, i]
        x[i] = 1.0
        R -= np.outer(col, x); dR -= np.outer(col, dx)
        R[:, i] = 0.0
        magic.append(i); sel.append(j); xs.append(x); dxs.append(dx); mult.append(col)
        train_err.append(float(np.max(np.abs(R))))
        if verbose:
            print(f"N={N} q={q + 1} train err {train_err[-1]:.2e}", flush=True)
        if train_err[-1] < 1e-13:
            break
    del R, dR
    magic = np.array(magic); sel = np.array(sel); q_max = magic.size
    nus = nus_train[sel]
    Xs = np.array(xs); dXs = np.array(dxs); mult = np.array(mult).T          # mult[j, l] = R^(l)_j[i_l]
    xi = Xs[:, :L] + 1j * Xs[:, L:]; dxi = dXs[:, :L] + 1j * dXs[:, L:]
    B = np.tril(Xs[:, magic].T)                                              # exact unit lower triangular
    rec = np.tril(mult[sel], -1); rho = mult[sel, np.arange(q_max)]; sc = sc_all[sel]
    # test error curve (first Q coefficients of the q_max solve = the Q-term interpolant)
    tq = eim_test_nus(n_test)
    St = stack(_eval_many(L, tq, h, K, nproc))
    nrm = np.max(np.abs(St), axis=1)
    coef = solve_triangular(B, St[:, magic].T, lower=True, unit_diagonal=True).T       # (n_test, q_max)
    test_err = [float(np.max(np.max(np.abs(St), axis=1) / nrm))]
    for q in range(q_max):
        St -= np.outer(coef[:, q], Xs[q])
        test_err.append(float(np.max(np.max(np.abs(St), axis=1) / nrm)))
    test_err = np.array(test_err)
    ok = np.nonzero(test_err <= tol)[0]
    if not ok.size:
        raise RuntimeError(f"EIM at N={N}: tol {tol:g} not reached with q_max={q_max} "
                           f"(best {test_err.min():.2e})")
    Q = int(ok[0])
    lags = np.array([fourier_coefficients(xi[q], n + extra) for q in range(q_max)])
    rel_imag = np.max(np.abs(lags[:Q].imag)) / np.max(np.abs(lags[:Q].real))
    lags = lags.real
    meta = dict(N=N, K=K, L=L, p=p, h=h, n_free=n, q_max=q_max, Q=Q, tol=tol,
                training_grid=dict(shape=list(TRAIN_SHAPE), spacing="log G, log M, Y uniform",
                                   G=grid["G"].tolist(), M=grid["M"].tolist(), Y=grid["Y"].tolist()),
                test=dict(n=n_test, seed=TEST_SEED, generator="sobol_queries(n, seed, include_T=False)"),
                normalisation="per-snapshot sup-norm of the stacked (Re, Im) vector",
                error_measure="max over nu of max_j |stacked residual| / max_j |stacked Phi_nu|",
                lags_rel_imag_first_Q=float(rel_imag), time_snapshots_s=t_snap, time_total_s=time.time() - t0,
                Q_vs_tol={f"{t:g}": int(np.nonzero(test_err <= t)[0][0]) if np.any(test_err <= t) else None
                          for t in TOL_SWEEP})
    return EIM(N=N, K=K, L=L, p=p, n=n, extra=extra, magic=magic, B=B, xi=xi, dxi=dxi, lags=lags, nus=nus,
               rec=rec, rho=rho, sc=sc, Q=Q, tol=tol, train_err=np.array(train_err), test_err=test_err,
               sv=sv, meta=meta)


class GradedNodes:
    """N-independent node set for eps_Q: symmetric, contains +-pi as one-sided limits, uniform L0 grid plus a
    geometric refinement on the scale h * min_box min(G, M) = h near theta = 0 (the same construction as
    spectral.graded_theta_grid, but query-independent so that the EIM basis can be precomputed on it).
    Query-independent alias data theta_k, |N^(theta_k)|^2 and its derivative are stored; online only
    (delta +- i theta_k)^Y is evaluated, in real arithmetic.  Symmetrisation pairs node i with node
    n-1-i; node 0 / n-1 carry the one-sided limits at -pi / +pi, and the L-grid's convention value at
    theta = -pi (Re of the unsymmetrised value) is bounded separately."""

    def __init__(self, eim: "EIM", L0: int, n_fine: int, lo: float = 1e-4, hi: float = 400.0):
        h, K = eim.h, eim.K
        d = h * min(BOX["G"][0], BOX["M"][0])
        pos = np.arange(1, L0 // 2 + 1) * 2 * pi / L0                       # (0, pi], pi included
        fine = np.geomspace(d * lo, min(pi * 0.999, hi * d), n_fine)
        pos = np.unique(np.concatenate([pos, fine]))
        self.theta = np.concatenate([-pos[::-1], [0.0], pos])
        self.L0, self.n_fine, self.K, self.h = L0, n_fine, K, h
        k = np.arange(-K, K + 1)
        self.tt = self.theta[None, :] + 2 * pi * k[:, None]
        self.tt2 = self.tt ** 2
        self.w = bspline_ft_sq(self.tt)
        self.wp = bspline_ft_sq_prime(self.tt)
        self.width = np.diff(self.theta)
        # EIM basis on the nodes (all stored terms), by the greedy recursion
        Qm = eim.magic.size
        self.xi = np.zeros((Qm, self.theta.size), dtype=complex); self.dxi = np.zeros_like(self.xi)
        self.xi_conv = np.zeros(Qm)                                         # Re xi(-pi) unsymmetrised
        for qi in range(Qm):
            v, dv = self._raw(tuple(eim.nus[qi]))
            ph, dph = self._sym(v, dv)
            self.xi[qi] = (ph / eim.sc[qi] - eim.rec[qi, :qi] @ self.xi[:qi]) / eim.rho[qi]
            self.dxi[qi] = (dph / eim.sc[qi] - eim.rec[qi, :qi] @ self.dxi[:qi]) / eim.rho[qi]
            self.xi_conv[qi] = (v[0].real / eim.sc[qi] - eim.rec[qi, :qi] @ self.xi_conv[:qi]) / eim.rho[qi]

    def _half(self, d: float, Y: float, sign: int):
        r2 = d * d + self.tt2
        mag = np.exp(0.5 * Y * np.log(r2))
        ph = Y * np.arctan2(sign * self.tt, d)
        pr, pim = mag * np.cos(ph), mag * np.sin(ph)
        zr, zi = d / r2, -sign * self.tt / r2                     # 1/z
        qr, qi = pr * zr - pim * zi, pr * zi + pim * zr           # z^Y / z
        dr, di = -sign * Y * qi, sign * Y * qr                    # i sign Y z^Y / z
        v = (self.w * pr).sum(0) + 1j * (self.w * pim).sum(0)
        dv = (self.wp * pr + self.w * dr).sum(0) + 1j * (self.wp * pim + self.w * di).sum(0)
        return v, dv

    def _raw(self, nu):
        G, M, Y = nu
        a, da = self._half(G * self.h, Y, +1)
        b, db = self._half(M * self.h, Y, -1)
        return a + b, da + db

    @staticmethod
    def _sym(v, dv):
        return 0.5 * (v + np.conj(v[::-1])), 0.5 * (dv - np.conj(dv[::-1]))

    def symbol(self, nu):
        return self._raw(nu)

    def sup_from(self, v, dv, ell) -> float:
        Q = ell.size
        ph, dph = self._sym(v, dv)
        e = np.abs(ph - ell @ self.xi[:Q]); de = np.abs(dph - ell @ self.dxi[:Q])
        cell = np.maximum(e[:-1], e[1:]) + 0.5 * self.width * np.maximum(de[:-1], de[1:])
        conv = abs(v[0].real - ell @ self.xi_conv[:Q])
        return float(max(cell.max(), conv))

    def sup_error(self, nu, ell) -> float:
        v, dv = self._raw(nu)
        return self.sup_from(v, dv, ell)


def eim_path(N: int, K: int = 100) -> Path:
    return EIM_DIR / f"eim_N{N}_K{K}.npz"


def load_eim(N: int, K: int = 100, build_if_missing: bool = True, **kw) -> EIM:
    path = eim_path(N, K)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(path)
        build_eim(N, K, **kw).save(path)
    d = np.load(path, allow_pickle=False)
    return EIM(N=int(d["N"]), K=int(d["K"]), L=int(d["L"]), p=int(d["p"]), n=int(d["n"]), extra=int(d["extra"]),
               magic=d["magic"], B=d["B"], xi=d["xi"], dxi=d["dxi"], lags=d["lags"], nus=d["nus"],
               rec=d["rec"], rho=d["rho"], sc=d["sc"], Q=int(d["Q"]),
               tol=float(d["tol"]), train_err=d["train_err"], test_err=d["test_err"], sv=d["sv"],
               meta=json.loads(str(d["meta"])))


def sha256(path: Path) -> str:
    hsh = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            hsh.update(blk)
    return hsh.hexdigest()
