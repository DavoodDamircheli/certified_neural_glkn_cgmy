"""Periodised symbols, truncation tail bound and Toeplitz assembly (note, Sections 1-2).

Every assembled matrix in v9 is a Toeplitz *section* T_n(f) of a 2 pi-periodic
symbol f:  (T_n(f))_{kk'} = t_{k-k'},  t_m = (2 pi)^{-1} int_{-pi}^{pi} f(theta) e^{i theta m} d theta.

Symbols (Lemma 1.1 of the note; p = spline order, theta_k = theta + 2 pi k):
    a_mu(theta) = sum_k |N_p^(theta_k)|^2 A_mu(theta_k / h)        evolution operator
    s_mu        = Re a_mu                                           symmetric part
    m(theta)    = sum_k |N_p^(theta_k)|^2                            mass            (trig. polynomial)
    d1(theta)   = sum_k |N_p^(theta_k)|^2 theta_k                    (i/h) d1 = drift (trig. polynomial)
    g1(theta)   = sum_k |N_p^(theta_k)|^2 theta_k^2                  h^2 R^(1)       (trig. polynomial)

m, d1, g1 are trigonometric polynomials of degree p-1 and are evaluated in closed
form (verified against the alias sums in the tests).  Only the fractional part
    Phi_nu(theta) = sum_k |N_p^(theta_k)|^2 [ (Gh + i theta_k)^Y + (Mh - i theta_k)^Y ]
is truncated to |k| <= K; its tail is bounded by Lemma 1.2 (eps_K), recorded with every
assembled operator.  The general three-term bound of Lemma 1.2 is implemented in
`tail_bound` / `eps_K_full`; the operator assembled here only needs the fractional term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy import pi
from scipy.linalg import toeplitz

from .params import Query, Mesh, P_ORDER

# ------------------------------------------------------------------ B-spline FT

def bspline_ft_sq(theta: np.ndarray, p: int = P_ORDER) -> np.ndarray:
    """|N_p^(theta)|^2 = sinc(theta/2)^(2p)   (sinc x = sin x / x)."""
    x = 0.5 * np.asarray(theta, dtype=float)
    out = np.ones_like(x)
    nz = x != 0.0
    out[nz] = np.sin(x[nz]) / x[nz]
    return out ** (2 * p)


def bspline_ft_sq_prime(theta: np.ndarray, p: int = P_ORDER) -> np.ndarray:
    """d/dtheta sinc(theta/2)^(2p)."""
    x = 0.5 * np.asarray(theta, dtype=float)
    s = np.ones_like(x); ds = np.zeros_like(x)
    nz = x != 0.0
    s[nz] = np.sin(x[nz]) / x[nz]
    ds[nz] = (np.cos(x[nz]) - s[nz]) / x[nz]
    return 2 * p * s ** (2 * p - 1) * ds * 0.5


# ------------------------------------------------------------------ alias sums

def alias_sum(theta: np.ndarray, fun, K: int, p: int = P_ORDER) -> np.ndarray:
    """sum_{|k|<=K} |N_p^(theta_k)|^2 fun(theta_k), vectorised over theta.
    Chunked over k so that L ~ 2^14 and K ~ 10^3 fit in memory."""
    theta = np.asarray(theta, dtype=float)
    out = np.zeros(theta.shape, dtype=complex)
    chunk = max(1, int(4e6 // max(theta.size, 1)))
    ks = np.arange(-K, K + 1)
    for i in range(0, ks.size, chunk):
        kk = ks[i:i + chunk][:, None]
        tt = theta[None, :] + 2 * pi * kk
        out += np.sum(bspline_ft_sq(tt, p) * fun(tt), axis=0)
    return out


def alias_sum_prime(theta: np.ndarray, fun, dfun, K: int, p: int = P_ORDER) -> np.ndarray:
    """Derivative in theta of alias_sum(theta, fun, K)."""
    theta = np.asarray(theta, dtype=float)
    out = np.zeros(theta.shape, dtype=complex)
    chunk = max(1, int(2e6 // max(theta.size, 1)))
    ks = np.arange(-K, K + 1)
    for i in range(0, ks.size, chunk):
        kk = ks[i:i + chunk][:, None]
        tt = theta[None, :] + 2 * pi * kk
        out += np.sum(bspline_ft_sq_prime(tt, p) * fun(tt) + bspline_ft_sq(tt, p) * dfun(tt), axis=0)
    return out


# ------------------------------------------------------------------ closed forms (p = 3)

def mass_symbol(theta, p: int = P_ORDER):
    """m(theta) = sum_k |N_p^(theta_k)|^2 = Euler-Frobenius polynomial of N_{2p}."""
    theta = np.asarray(theta, dtype=float)
    if p == 3:
        return 11 / 20 + (13 / 30) * np.cos(theta) + (1 / 60) * np.cos(2 * theta)
    if p == 2:
        return 2 / 3 + (1 / 3) * np.cos(theta)
    raise NotImplementedError(p)


def drift_symbol(theta, p: int = P_ORDER):
    """d1(theta) = sum_k |N_p^(theta_k)|^2 theta_k ;  D_h = T((i/h) d1)."""
    theta = np.asarray(theta, dtype=float)
    if p == 3:
        return (5 / 6) * np.sin(theta) + (1 / 12) * np.sin(2 * theta)
    if p == 2:
        return np.sin(theta)
    raise NotImplementedError(p)


def stiffness_symbol(theta, p: int = P_ORDER):
    """g1(theta) = sum_k |N_p^(theta_k)|^2 theta_k^2 ;  h^2 R^(1) = T(g1)."""
    theta = np.asarray(theta, dtype=float)
    if p == 3:
        return 1.0 - (2 / 3) * np.cos(theta) - (1 / 3) * np.cos(2 * theta)
    if p == 2:
        return 2.0 - 2.0 * np.cos(theta)
    raise NotImplementedError(p)


def closed_form_derivatives(theta, p: int = P_ORDER):
    theta = np.asarray(theta, dtype=float)
    if p != 3:
        raise NotImplementedError(p)
    dm = -(13 / 30) * np.sin(theta) - (1 / 30) * np.sin(2 * theta)
    dd1 = (5 / 6) * np.cos(theta) + (1 / 6) * np.cos(2 * theta)
    dg1 = (2 / 3) * np.sin(theta) + (2 / 3) * np.sin(2 * theta)
    return dm, dd1, dg1


# ------------------------------------------------------------------ tail bound (Lemma 1.2)

def tau_K(K: int, q: float, p: int = P_ORDER) -> float:
    """tau_K(q) = 2^(2p+1) pi^(q-2p) (2K-1)^(q-2p+1) / (2(2p-q-1)),  q in [0, 2p-1)."""
    if K < 1:
        return np.inf
    s = 2 * p - q
    return 2 ** (2 * p + 1) * pi ** (-s) * (2 * K - 1) ** (1 - s) / (2 * (s - 1))


def tail_bound(K: int, q: float, delta: float, Y: float, p: int = P_ORDER) -> float:
    """Bound on sum_{|k|>K} |N_p^(theta_k)|^2 |delta + i theta_k|^q for |theta| <= pi
    (q = 1 drift, q = Y fractional, q = 0 mass).  Lemma 1.2 of the note."""
    c = (1.0 + delta / (pi * (2 * K + 1))) ** q if q > 0 else 1.0
    return c * tau_K(K, q, p)


def eps_K_fractional(q: Query, h: float, K: int, p: int = P_ORDER) -> float:
    """Sup-norm bound on the dropped aliases of C Gamma(-Y) h^{-Y} Phi_nu (the only truncated term)."""
    scale = abs(q.C * q.gammaY) * h ** (-q.Y)
    return scale * (tail_bound(K, q.Y, q.G * h, q.Y, p) + tail_bound(K, q.Y, q.M * h, q.Y, p))


def eps_K_full(q: Query, h: float, K: int, p: int = P_ORDER) -> float:
    """The complete three-term bound (1.2) of the note: drift (q=1) + fractional (q=Y) + mass (q=0).
    Used only when *all three* symbols are truncated (tests); the production assembly uses the
    exact trigonometric polynomials for m, d1 and only truncates Phi_nu."""
    drift = abs(q.r - q.omega_S) / h * tail_bound(K, 1.0, 0.0, q.Y, p)
    mass = abs(q.lambda0) * tail_bound(K, 0.0, 0.0, q.Y, p)
    return drift + eps_K_fractional(q, h, K, p) + mass


# ------------------------------------------------------------------ CGMY symbols

def symbol_A(xi: np.ndarray, q: Query) -> np.ndarray:
    """A_mu(xi) = -(r - omega_S) i xi - C Gamma(-Y) [(M - i xi)^Y + (G + i xi)^Y] + lambda_0."""
    C, G, M, Y, r = q.mu
    return (-(r - q.omega_S) * 1j * xi
            - C * q.gammaY * ((M - 1j * xi) ** Y + (G + 1j * xi) ** Y) + q.lambda0)


def fractional_symbol(theta_k: np.ndarray, q: Query, h: float) -> np.ndarray:
    """(Gh + i theta)^Y + (Mh - i theta)^Y  (the family Phi_nu before alias summation)."""
    return (q.G * h + 1j * theta_k) ** q.Y + (q.M * h - 1j * theta_k) ** q.Y


def fractional_symbol_prime(theta_k: np.ndarray, q: Query, h: float) -> np.ndarray:
    Y = q.Y
    return 1j * Y * (q.G * h + 1j * theta_k) ** (Y - 1) - 1j * Y * (q.M * h - 1j * theta_k) ** (Y - 1)


def fractional_alias(theta: np.ndarray, q: Query, h: float, K: int, p: int = P_ORDER,
                     with_derivative: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Truncated alias sum of Phi_nu and (optionally) its theta-derivative, sharing the power
    evaluations: d/dtheta (delta + i theta)^Y = i Y (delta + i theta)^Y / (delta + i theta)."""
    theta = np.asarray(theta, dtype=float)
    Y = q.Y; dG = q.G * h; dM = q.M * h
    phi = np.zeros(theta.shape, dtype=complex)
    dphi = np.zeros(theta.shape, dtype=complex) if with_derivative else None
    chunk = max(1, int(2e6 // max(theta.size, 1)))
    ks = np.arange(-K, K + 1)
    for i in range(0, ks.size, chunk):
        kk = ks[i:i + chunk][:, None]
        tt = theta[None, :] + 2 * pi * kk
        w = bspline_ft_sq(tt, p)
        zG = dG + 1j * tt; zM = dM - 1j * tt
        pG = zG ** Y; pM = zM ** Y
        f = pG + pM
        phi += np.sum(w * f, axis=0)
        if with_derivative:
            wp = bspline_ft_sq_prime(tt, p)
            fp = 1j * Y * pG / zG - 1j * Y * pM / zM
            dphi += np.sum(wp * f + w * fp, axis=0)
    return phi, dphi


def theta_grid(L: int) -> np.ndarray:
    return -pi + 2 * pi * np.arange(L) / L


@dataclass
class Symbols:
    """Symbol samples on theta_j = -pi + 2 pi j / L and their derivatives."""
    q: Query
    mesh: Mesh
    K: int
    L: int
    theta: np.ndarray
    a: np.ndarray          # complex evolution symbol (truncated at K)
    s: np.ndarray          # Re a
    m: np.ndarray
    d1: np.ndarray
    g1: np.ndarray
    phi: np.ndarray        # Phi_nu truncated
    s_prime: np.ndarray    # d/dtheta s
    eps_K: float           # Lemma 1.2 fractional tail bound (sup-norm, absolute)

    @property
    def h(self) -> float:
        return self.mesh.h


def symbols(q: Query, N: int, K: int = 100, L: int | None = None, p: int = P_ORDER,
            with_derivative: bool = True, theta: np.ndarray | None = None) -> Symbols:
    """Periodised symbols of the free system on theta_j = -pi + 2 pi j / L (default L = a multiple
    of N_h, >= 256) or on an arbitrary sorted grid `theta` in [-pi, pi).  The alias sum of the
    fractional part is truncated at |k| <= K."""
    q.validate()
    msh = Mesh(N=N, p=p)
    h = msh.h
    if theta is None:
        if L is None:
            # a multiple of N_h (so that the N_h-point circulant can be sampled from the grid), >= 256
            L = msh.n_free * max(4, int(np.ceil(256 / msh.n_free)))
        th = theta_grid(L)
    else:
        th = np.asarray(theta, dtype=float); L = th.size
    C, G, M, Y, r = q.mu
    cg = C * q.gammaY * h ** (-Y)
    phi, dphi = fractional_alias(th, q, h, K, p, with_derivative)
    m = mass_symbol(th, p); d1 = drift_symbol(th, p); g1 = stiffness_symbol(th, p)
    a = -(r - q.omega_S) * 1j / h * d1 + q.lambda0 * m - cg * phi
    s = a.real
    s_prime = None
    if with_derivative:
        dm, dd1, dg1 = closed_form_derivatives(th, p)
        s_prime = q.lambda0 * dm - cg * dphi.real
    return Symbols(q=q, mesh=msh, K=K, L=L, theta=th, a=a, s=s, m=m, d1=d1, g1=g1, phi=phi,
                   s_prime=s_prime, eps_K=eps_K_fractional(q, h, K, p))


def graded_theta_grid(q: Query, N: int, L0: int = 2 ** 14, n_fine: int = 1500, p: int = P_ORDER) -> np.ndarray:
    """Uniform L0-point grid plus a geometric refinement near theta = 0 on the scale
    delta = min(G, M) h (the symbol varies on that scale), symmetric, sorted, in [-pi, pi)."""
    h = Mesh(N=N, p=p).h
    delta = min(q.G, q.M) * h
    base = theta_grid(L0)
    lo, hi = delta * 1e-4, min(np.pi * 0.999, 400 * delta)
    if hi <= lo:
        return base
    fine = np.geomspace(lo, hi, n_fine)
    th = np.unique(np.concatenate([base, fine, -fine]))
    return th


# ------------------------------------------------------------------ Toeplitz sections

def fourier_coefficients(sym_vals: np.ndarray, nmax: int) -> np.ndarray:
    """t_m for m = -(nmax-1), ..., nmax-1 from samples on theta_j = -pi + 2 pi j / L:
    t_m = (-1)^m ifft(sym)[m mod L].  Requires 2 nmax - 1 <= L."""
    L = sym_vals.size
    if 2 * nmax - 1 > L:
        raise ValueError(f"need L >= 2 nmax - 1 (L={L}, nmax={nmax})")
    c = np.fft.ifft(sym_vals)
    m = np.arange(-(nmax - 1), nmax)
    return ((-1.0) ** m) * c[np.mod(m, L)]


def toeplitz_section(sym_vals: np.ndarray, n: int, real: bool = True) -> np.ndarray:
    """Dense n x n Toeplitz section T_n(f) from symbol samples (one inverse FFT)."""
    t = fourier_coefficients(sym_vals, n)
    col = t[n - 1:]                 # t_0 ... t_{n-1}
    row = t[:n][::-1]               # t_0, t_{-1}, ..., t_{-(n-1)}
    T = toeplitz(col, row)
    if real:
        imax = np.max(np.abs(T.imag))
        if imax > 1e-9 * max(1.0, np.max(np.abs(T.real))):
            raise ValueError(f"Toeplitz section is not real (max imag {imax:.2e})")
        T = T.real
    return T


class ToeplitzOp:
    """Matrix-free n x n Toeplitz operator with O(n log n) matvec by circulant embedding.
    Also exposes the entries t_m for -(n+extra) <= m <= n+extra (used for the lifting columns)."""

    def __init__(self, sym_vals: np.ndarray, n: int, extra: int = 0):
        self.n = n
        t = fourier_coefficients(sym_vals, n + extra)
        self.t = t                           # lags -(n+extra-1) .. (n+extra-1)
        self.lag0 = n + extra - 1            # index of t_0 in self.t
        col = t[self.lag0:self.lag0 + n]     # t_0 .. t_{n-1}
        row = t[self.lag0 - n + 1:self.lag0 + 1][::-1]   # t_0, t_{-1}, ... t_{-(n-1)}
        self.col, self.row = col, row
        self.real = np.max(np.abs(t.imag)) <= 1e-9 * max(1.0, np.max(np.abs(t.real)))
        m2 = 1 << int(np.ceil(np.log2(2 * n - 1)))
        emb = np.zeros(m2, dtype=complex)
        emb[:n] = col
        emb[m2 - n + 1:] = row[1:][::-1]
        self.m2 = m2
        self.fft_emb = np.fft.fft(emb)

    def entry(self, m: int) -> complex:
        return self.t[self.lag0 + m]

    def matvec(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        out = np.fft.ifft(self.fft_emb * np.fft.fft(v, self.m2))[: self.n]
        if self.real and np.isrealobj(v):
            return out.real
        return out

    def matmat(self, V: np.ndarray) -> np.ndarray:
        V = np.asarray(V)
        out = np.fft.ifft(self.fft_emb[:, None] * np.fft.fft(V, self.m2, axis=0), axis=0)[: self.n]
        if self.real and np.isrealobj(V):
            return out.real
        return out

    def dense(self) -> np.ndarray:
        T = toeplitz(self.col, self.row)
        return T.real if self.real else T

    def transpose_op(self) -> "ToeplitzOp":
        op = ToeplitzOp.__new__(ToeplitzOp)
        op.n, op.lag0, op.real, op.m2 = self.n, self.lag0, self.real, self.m2
        op.t = self.t[::-1]
        op.col, op.row = self.row, self.col
        emb = np.zeros(self.m2, dtype=complex); emb[: self.n] = op.col; emb[self.m2 - self.n + 1:] = op.row[1:][::-1]
        op.fft_emb = np.fft.fft(emb)
        return op


def banded_from_toeplitz(entries: dict[int, float], n: int, bw: int) -> np.ndarray:
    """LAPACK banded storage (ab[bw + i - j, j] = T[i, j]) of a symmetric banded Toeplitz matrix
    with entries {m: t_m}, |m| <= bw."""
    ab = np.zeros((2 * bw + 1, n))
    for m in range(-bw, bw + 1):
        val = entries.get(m, 0.0)
        if val == 0.0:
            continue
        # T[i, j] = t_{i-j} ; row index in ab = bw + (i - j) = bw + m
        if m >= 0:
            ab[bw + m, : n - m] = val
        else:
            ab[bw + m, -m:] = val
    return ab


# ------------------------------------------------------------------ free system

@dataclass
class FreeSystem:
    """Assembled free system (Toeplitz sections) for one query and mesh."""
    q: Query
    mesh: Mesh
    sym: Symbols
    A: ToeplitzOp              # A_I = T_{N_h}(a_mu)
    S: ToeplitzOp              # S_I = T_{N_h}(s_mu)
    M_entries: dict            # mass Toeplitz entries t_m, |m| <= p-1
    K1_entries: dict           # h^2 R^(1) entries
    D_entries: dict            # D_I entries (antisymmetric)
    eps_K: float

    @property
    def n(self) -> int:
        return self.mesh.n_free

    def M_banded(self) -> np.ndarray:
        return banded_from_toeplitz(self.M_entries, self.n, self.mesh.p - 1)

    def K1_banded(self) -> np.ndarray:
        return banded_from_toeplitz(self.K1_entries, self.n, self.mesh.p - 1)

    def M_matvec(self, v):
        return banded_sym_matvec(self.M_entries, v)

    def dense(self) -> dict:
        n = self.n
        return dict(A=self.A.dense(), S=self.S.dense(), M=_dense_banded(self.M_entries, n),
                    K1=_dense_banded(self.K1_entries, n), D=_dense_banded(self.D_entries, n))


def _dense_banded(entries: dict, n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for m, v in entries.items():
        if m >= 0:
            T[np.arange(m, n), np.arange(0, n - m)] = v
        else:
            T[np.arange(0, n + m), np.arange(-m, n)] = v
    return T


def banded_sym_matvec(entries: dict, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    out = entries.get(0, 0.0) * v
    n = v.shape[0]
    for m, val in entries.items():
        if m > 0:
            out[m:] += val * v[: n - m]         # T[i, i-m] = t_m
            out[: n - m] += entries.get(-m, val) * v[m:]
    return out


def _poly_entries(symbol_fun, p: int, scale: float = 1.0) -> dict[int, float]:
    """Exact Toeplitz entries of a trigonometric polynomial symbol of degree p-1."""
    L = 64
    th = theta_grid(L)
    t = fourier_coefficients(symbol_fun(th).astype(complex), p)
    return {m: scale * float(np.round(t[(p - 1) + m].real, 15)) for m in range(-(p - 1), p)}


def free_matrices(q: Query, N: int, K: int = 100, L: int | None = None, p: int = P_ORDER,
                  with_derivative: bool = True, extra_lags: int | None = None) -> FreeSystem:
    """A_I, S_I, M_I, K1_I (and D_I) for the contiguous free index set, from one symbol evaluation.
    `extra_lags` extra Toeplitz entries are kept for the boundary (lifting) columns (default p-1)."""
    sym = symbols(q, N, K=K, L=L, p=p, with_derivative=with_derivative)
    n = sym.mesh.n_free
    extra = (p - 1) if extra_lags is None else extra_lags
    A = ToeplitzOp(sym.a, n, extra=extra)
    S = ToeplitzOp(sym.s.astype(complex), n, extra=extra)
    h = sym.mesh.h
    M_entries = _poly_entries(lambda t: mass_symbol(t, p), p)
    K1_entries = _poly_entries(lambda t: stiffness_symbol(t, p), p)
    D_entries = _poly_entries(lambda t: 1j / h * drift_symbol(t, p), p)
    return FreeSystem(q=q, mesh=sym.mesh, sym=sym, A=A, S=S, M_entries=M_entries,
                      K1_entries=K1_entries, D_entries=D_entries, eps_K=sym.eps_K)


# ------------------------------------------------------------------ legacy (v8) assembly

def legacy_principal_period_free(q: Query, N: int, p: int = P_ORDER) -> dict:
    """The v8 assembly, reproduced for regression tests only: compensated fractional symbols
    (delta -/+ i tau)^Y - delta^Y sampled on L = 2 (N+p-1) points of the principal period,
    exact banded mass and derivative matrices, lambda_0 := r.  Returns dense free A_I, S_I."""
    msh = Mesh(N=N, p=p); h = msh.h; n_raw = msh.n_raw; n = msh.n_free
    L = 2 * n_raw
    tau = np.fft.fftfreq(L) * 2 * pi
    with np.errstate(divide="ignore", invalid="ignore"):
        hatN = np.where(np.abs(tau) < 1e-14, 1.0 + 0j, ((1 - np.exp(-1j * tau)) / (1j * tau)) ** p)
    W = np.abs(hatN) ** 2
    dG, dM, Y = q.G * h, q.M * h, q.Y
    lamG = np.real(np.fft.ifft(W * ((dG + 1j * tau) ** Y - dG ** Y))) * h ** (-Y)
    lamM = np.real(np.fft.ifft(W * ((dM - 1j * tau) ** Y - dM ** Y))) * h ** (-Y)

    def top(lam):
        col = lam[:n_raw]; row = lam[(-np.arange(n_raw)) % L]
        return toeplitz(col, row)

    M = _dense_banded(_poly_entries(lambda t: mass_symbol(t, p), p), n_raw)
    D = _dense_banded(_poly_entries(lambda t: 1j / h * drift_symbol(t, p), p), n_raw)
    A_raw = -(q.r - q.omega_S) * D - q.C * q.gammaY * (top(lamM) + top(lamG)) + q.r * M
    fs = msh.free_slice
    A = A_raw[fs, fs]
    return dict(A=A, S=0.5 * (A + A.T), M=M[fs, fs])
