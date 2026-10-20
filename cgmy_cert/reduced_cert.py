"""Reduced online certificate eta^{X-bar} (plan 2.3), N-independent online.

Cells: 2 x 2 in (log r, log C) over the box.  Per cell the comparison operator is
    X-bar = r-bar M_I + beta-bar K1_I            (banded, symbol x-bar = r-bar m + beta-bar g1),
r-bar = the cell's r_min, beta-bar = 0.99 x min of spectral_data(q, 512).beta over the 5 x 5 (r, C) grid
of the cell times the 3 x 3 x 3 (min, mid, max) grid in (G, M, Y).

Residual of a reduced candidate (free vector V alpha^n) with the affine operator and forcing:
    rho^{n+1/2} = F_M x_n + sum_k theta_k F_k y_n,
    F_k = [O_k V, u_k]  (n x (r+1)),  O_k in {M_I, D_I, T(xi_1), ..., T(xi_Q)},  u_k = boundary coupling,
    theta = (lambda_0, -(r - omega_S), -C Gamma(-Y) h^{-Y} ell_1, ..., -C Gamma(-Y) h^{-Y} ell_Q),
    x_n = [(alpha^{n+1} - alpha^n)/dt ; (b_L^{n+1} - b_L^n)/dt],  y_n = [(alpha^{n+1} + alpha^n)/2 ; (b_L^{n+1} + b_L^n)/2].
Offline per cell: banded Cholesky X-bar = L L^T, W = L^{-1} [F_M, F_D, F_T1, ...] (banded triangular solves),
economy QR W = Q_W R, and store R (m x (Q+2)(r+1), m = min(n, (Q+2)(r+1))).  Online:
    R_theta = sum_k theta_k R_k,   u_n = || R_M x_n + R_theta y_n ||_2^2     (= rho_n^T X-bar^{-1} rho_n),
cost O(m (Q+2) r + m r N_T), bounded independently of N (m <= (Q+2)(r+1)).

Why not the Gram tensor G = F^T X-bar^{-1} F (the plan's formulation, O((Q+2)^2 r^2) online): for good
candidates the residual is ~1e-8 of the individual terms (kappa ~ 5e7 for POD-Galerkin at r = 48), and the
Gram form loses eps * kappa^2 ~ O(1) relative accuracy (measured: per-step errors up to 100%, sums 0.5%),
which can *under*-estimate the bound.  The R-factor form loses eps * kappa ~ 1e-8, the same as the
full-order residual evaluation.  A rigorous rounding term gamma_m ||R_M||_F ||x_n|| + gamma_m ||R_theta||_F
||y_n|| (gamma_m = m u / (1 - m u), u = 2^-53, standard dot-product bound) is added to sqrt(u_n).

Bound (rigorous): with the EIM allowance d_n = eps_Q ||cbar_raw^n||_2 / sqrt(xbar_min) and the conversion
factor c_mu = min_theta (s_mu - eps_K) / x-bar  (S_I >= c_mu X-bar, exact matrix included),
    eta^{X-bar} = ( dt sum_n (sqrt(u_n) + d_n)^2 / c_mu )^{1/2},   eta_aug = (beta_0^2 + eta^2)^{1/2}.
||cbar_raw||_2^2 = ybar^T (V^T V) ybar + (p-1) b-bar^2 uses the stored r x r Gram V^T V; beta_0 =
||z_h^0 - V alpha^0||_{M} is precomputed (z_h^0 depends on the strike only, which is fixed).
c_mu needs one symbol evaluation on the graded grid of `spectral.py` (N-independent size).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded, cholesky_banded

from .params import Query, Mesh, BOX, P_ORDER
from .symbols import (ToeplitzOp, banded_from_toeplitz, _poly_entries, mass_symbol, stiffness_symbol,
                      drift_symbol, symbols, graded_theta_grid, closed_form_derivatives, banded_sym_matvec)
from .spectral import spectral_data, _grid_extreme
from .pod import boundary_couplings, _entries_to_lags
from .direct import initial_free_vector


# ---------------------------------------------------------------- cells

def cell_edges():
    r0, r1 = BOX["r"]; c0, c1 = BOX["C"]
    rm = float(np.sqrt(r0 * r1)); cm = float(np.sqrt(c0 * c1))
    return [(r0, rm), (rm, r1)], [(c0, cm), (cm, c1)]


def cells() -> list[dict]:
    rs, cs = cell_edges()
    return [dict(id=f"r{i}C{j}", r=rs[i], C=cs[j]) for i in range(2) for j in range(2)]


def cell_of(q: Query) -> str:
    rs, cs = cell_edges()
    i = 0 if q.r <= rs[0][1] else 1
    j = 0 if q.C <= cs[0][1] else 1
    return f"r{i}C{j}"


def cell_beta_queries(cell: dict) -> list[Query]:
    """5 x 5 (r, C) grid of the cell (geometric, corners and interior) x 3 x 3 x 3 (G, M, Y) (min, mid, max)."""
    rr = np.geomspace(*cell["r"], 5); cc = np.geomspace(*cell["C"], 5)
    gg = np.linspace(*BOX["G"], 3); mm = np.linspace(*BOX["M"], 3); yy = np.linspace(*BOX["Y"], 3)
    return [Query(C=float(C), G=float(G), M=float(M), Y=float(Y), r=float(r))
            for r in rr for C in cc for G in gg for M in mm for Y in yy]


def xbar_entries(rbar: float, betabar: float, p: int = P_ORDER) -> dict:
    Me = _poly_entries(lambda t: mass_symbol(t, p), p); Ke = _poly_entries(lambda t: stiffness_symbol(t, p), p)
    return {k: rbar * Me[k] + betabar * Ke[k] for k in Me}


def xbar_min(rbar: float, betabar: float, p: int = P_ORDER) -> float:
    """min_theta rbar m + betabar g1 (trigonometric polynomial of degree 2: dense grid + margin)."""
    th = np.linspace(-np.pi, np.pi, 4097)[:-1]
    x = rbar * mass_symbol(th, p) + betabar * stiffness_symbol(th, p)
    dm, _, dg = closed_form_derivatives(th, p)
    return _grid_extreme(x, rbar * dm + betabar * dg, th, "min")[0]


K_LADDER = (10, 20, 40, 100)


def conversion_alias_count(q: Query, N: int, rbar: float, betabar: float, rel: float = 0.02, p: int = P_ORDER) -> int:
    """Smallest K' in K_LADDER whose fractional tail bound eps_{K'} (Lemma 1.2, all |k| > K' of the exact
    symbol) is <= rel * min xbar; the K'-truncated symbol minus eps_{K'} is then a rigorous minorant of s_mu."""
    from .symbols import eps_K_fractional
    h = Mesh(N=N, p=p).h
    xm = xbar_min(rbar, betabar, p)
    for K in K_LADDER:
        if eps_K_fractional(q, h, K, p) <= rel * xm:
            return K
    return K_LADDER[-1]


def conversion_factor(q: Query, N: int, rbar: float, betabar: float, K: int | None = None, p: int = P_ORDER) -> float:
    """c_mu = min_theta (s_{mu,K'}(theta) - eps_{K'}) / xbar(theta) with the cell-wise derivative margin of
    spectral.py on the graded grid (N-independent size).  Since s_mu >= s_{mu,K'} - eps_{K'} for the exact
    symbol and for the K = 100 truncation alike, S_I >= c_mu X-bar for both.  K' is chosen adaptively
    (`conversion_alias_count`) unless given: this cuts the cost ~8x at N = 512 at <= 2% loss in c_mu."""
    if K is None:
        K = conversion_alias_count(q, N, rbar, betabar, p=p)
    th = graded_theta_grid(q, N, p=p)
    sym = symbols(q, N, K=K, p=p, with_derivative=True, theta=th)
    th, s, m, g1, sp = sym.theta, sym.s, sym.m, sym.g1, sym.s_prime
    dm, _, dg = closed_form_derivatives(th, p)
    x = rbar * m + betabar * g1; xp = rbar * dm + betabar * dg
    f = (s - sym.eps_K) / x
    fp = (sp * x - (s - sym.eps_K) * xp) / x ** 2
    return _grid_extreme(f, fp, th, "min")[0]


# ---------------------------------------------------------------- offline tensors

@dataclass
class CellGram:
    cell: str
    rbar: float
    betabar: float
    xbar_min: float
    R: np.ndarray            # (Q+2, m, r+1): R-factor column blocks
    R_fro: np.ndarray        # (Q+2,) Frobenius norms of the blocks (rounding term)


@dataclass
class ReducedCertificate:
    N: int
    r: int
    Q: int
    h: float
    p: int
    VtV: np.ndarray          # r x r
    beta0_sq_const: float    # ||z_h^0||_M^2 (the strike is fixed)
    cells: dict = field(default_factory=dict)      # id -> CellGram
    time_offline_s: float = 0.0

    @staticmethod
    def residual_blocks(V: np.ndarray, eim, Q: int) -> np.ndarray:
        """F = [F_M, F_D, F_T1, ..., F_TQ], each n x (r+1): (Q+2) blocks stacked along a new axis 0."""
        n, p, h = eim.n, eim.p, eim.h
        nl = n + eim.extra
        Me = _poly_entries(lambda t: mass_symbol(t, p), p)
        De = _poly_entries(lambda t: 1j / h * drift_symbol(t, p), p)
        lag_sets = [_entries_to_lags(Me, nl), _entries_to_lags(De, nl)] + [eim.lags[q] for q in range(Q)]
        F = np.zeros((Q + 2, n, V.shape[1] + 1))
        for k, lags in enumerate(lag_sets):
            F[k, :, :-1] = ToeplitzOp.from_lags(lags, n, eim.extra).matmat(V)
            F[k, :, -1] = boundary_couplings(lags, n, p).sum(axis=0)
        return F

    @classmethod
    def build(cls, V: np.ndarray, eim, cell_params: dict, Q: int | None = None) -> "ReducedCertificate":
        """cell_params: {cell id: (rbar, betabar)}."""
        t0 = time.time()
        Q = eim.Q if Q is None else Q
        n, r = V.shape
        p = eim.p; bw = p - 1
        F = cls.residual_blocks(V, eim, Q)
        Fm = F.transpose(1, 0, 2).reshape(n, -1)              # n x (Q+2)(r+1)
        msh = Mesh(N=eim.N, p=p)
        z0 = initial_free_vector(Query(0.5, 3.0, 5.0, 1.5, 0.05), msh)
        Me = _poly_entries(lambda t: mass_symbol(t, p), p)
        Mz0 = banded_sym_matvec(Me, z0)
        rc = cls(N=eim.N, r=r, Q=Q, h=eim.h, p=p, VtV=V.T @ V, beta0_sq_const=float(z0 @ Mz0))
        rc._Vt_Mz0 = V.T @ Mz0
        for cid, (rbar, betabar) in cell_params.items():
            ab = banded_from_toeplitz(xbar_entries(rbar, betabar, p), n, bw)
            Lb = cholesky_banded(ab[bw:], lower=True)            # lower band storage, X-bar = L L^T
            W = solve_banded((bw, 0), Lb, Fm)                    # L^{-1} F
            Rm = np.linalg.qr(W, mode="r")                       # m x (Q+2)(r+1)
            Rb = Rm.reshape(Rm.shape[0], Q + 2, r + 1).transpose(1, 0, 2).copy()
            rc.cells[cid] = CellGram(cell=cid, rbar=rbar, betabar=betabar, xbar_min=xbar_min(rbar, betabar, p),
                                     R=Rb, R_fro=np.sqrt(np.sum(Rb ** 2, axis=(1, 2))))
        rc.time_offline_s = time.time() - t0
        return rc

    # ---------------------------------------------------------------- online
    def alpha0(self) -> np.ndarray:
        """alpha^0 = V^T M_I z_h^0 (the same for every query)."""
        return self._Vt_Mz0.copy()

    def beta0(self, alpha0: np.ndarray) -> float:
        """||z_h^0 - V alpha^0||_M for an M-orthonormal V."""
        d = self.beta0_sq_const - 2 * alpha0 @ self._Vt_Mz0 + alpha0 @ alpha0
        return float(np.sqrt(max(d, 0.0)))

    def theta(self, q: Query, ell: np.ndarray) -> np.ndarray:
        cg = q.C * q.gammaY * self.h ** (-q.Y)
        return np.concatenate([[q.lambda0, -(q.r - q.omega_S)], -cg * ell[: self.Q]])

    def per_step(self, cid: str, theta: np.ndarray, X: np.ndarray, Yv: np.ndarray, with_rounding: bool = False):
        """u_n = rho_n^T X-bar^{-1} rho_n for columns of X = [x_n], Y = [y_n] ((r+1) x N_T); with_rounding also
        returns the rigorous rounding bound on |sqrt(u_n) computed - exact|."""
        c = self.cells[cid]
        Rth = np.tensordot(theta, c.R, axes=([0], [0]))       # m x (r+1)
        Wn = c.R[0] @ X + Rth @ Yv
        u = np.sum(Wn * Wn, axis=0)
        if not with_rounding:
            return u
        m = c.R.shape[1] * (len(theta) + 1)
        g = m * 2.0 ** -53 / (1 - m * 2.0 ** -53)
        rth_fro = float(np.sqrt(np.sum(Rth ** 2)) + g * np.dot(np.abs(theta), c.R_fro))
        rnd = g * (c.R_fro[0] * np.linalg.norm(X, axis=0) + rth_fro * np.linalg.norm(Yv, axis=0))
        return u, rnd

    def certify(self, q: Query, alpha: np.ndarray, dt: float, bL: np.ndarray, ell: np.ndarray,
                eps_Q: float, c_mu: float | None = None, cid: str | None = None, timings: dict | None = None):
        """eta^{X-bar} (augmented with beta_0).  c_mu is computed if not given (graded-grid symbol)."""
        t0 = time.perf_counter()
        cid = cell_of(q) if cid is None else cid
        cell = self.cells[cid]
        X = np.vstack([(alpha[:, 1:] - alpha[:, :-1]) / dt, ((bL[1:] - bL[:-1]) / dt)[None, :]])
        Yv = np.vstack([0.5 * (alpha[:, 1:] + alpha[:, :-1]), (0.5 * (bL[1:] + bL[:-1]))[None, :]])
        u, rnd = self.per_step(cid, self.theta(q, ell), X, Yv, with_rounding=True)
        ab = Yv[:-1]
        craw = np.sqrt(np.sum(ab * (self.VtV @ ab), axis=0) + (self.p - 1) * Yv[-1] ** 2)
        d = eps_Q * craw / np.sqrt(cell.xbar_min)
        resX2 = dt * np.sum((np.sqrt(u) + rnd + d) ** 2)
        b0 = self.beta0(alpha[:, 0])
        t_form = time.perf_counter() - t0
        t0 = time.perf_counter()
        if c_mu is None:
            c_mu = conversion_factor(q, self.N, cell.rbar, cell.betabar)
        t_conv = time.perf_counter() - t0
        if c_mu <= 0:
            raise ValueError(f"conversion factor c_mu = {c_mu} <= 0 for {q} in cell {cid}")
        if timings is not None:
            timings.update(form=t_form, conversion=t_conv)
        return dict(eta_xbar=float(np.sqrt(resX2)), eta_aug=float(np.sqrt(b0 ** 2 + resX2 / c_mu)),
                    beta0=b0, c_mu=c_mu, cell=cid, per_step=u, rounding=rnd, allowance=d)


# ---------------------------------------------------------------- io

def save_rc(rc: ReducedCertificate, path):
    arrs = dict(N=rc.N, r=rc.r, Q=rc.Q, h=rc.h, p=rc.p, VtV=rc.VtV, beta0_sq_const=rc.beta0_sq_const,
                Vt_Mz0=rc._Vt_Mz0, cell_ids=np.array(list(rc.cells)))
    for cid, c in rc.cells.items():
        arrs[f"{cid}|R"] = c.R; arrs[f"{cid}|scal"] = np.array([c.rbar, c.betabar, c.xbar_min])
    np.savez(path, **arrs)


def load_rc(path) -> ReducedCertificate:
    d = np.load(path)
    rc = ReducedCertificate(N=int(d["N"]), r=int(d["r"]), Q=int(d["Q"]), h=float(d["h"]), p=int(d["p"]),
                            VtV=d["VtV"], beta0_sq_const=float(d["beta0_sq_const"]))
    rc._Vt_Mz0 = d["Vt_Mz0"]
    for cid in d["cell_ids"]:
        cid = str(cid); rb, bb, xm = d[f"{cid}|scal"]; Rb = d[f"{cid}|R"]
        rc.cells[cid] = CellGram(cell=cid, rbar=float(rb), betabar=float(bb), xbar_min=float(xm), R=Rb,
                                 R_fro=np.sqrt(np.sum(Rb ** 2, axis=(1, 2))))
    return rc
