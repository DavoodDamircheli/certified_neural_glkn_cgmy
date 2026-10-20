"""High-precision (mpmath) evaluation of the *exact* periodised symbol and of the
exact Galerkin Toeplitz entries.  Used to build the real-line fixture and for the
40-digit eigenvalue checks; never used by the production code path.

The alias series of Lemma 1.1 is summed explicitly for |k| <= K0 and its two tails
(k > K0 and k < -K0) are summed in closed form with the Hurwitz zeta function:
for theta_k = theta + 2 pi k > 0,
    sinc(theta_k/2)^(2p) = sin(theta/2)^(2p) (2/theta_k)^(2p),
    (delta + i theta_k)^Y = e^{i pi Y/2} theta_k^Y sum_j C(Y,j) (-i delta / theta_k)^j,
    sum_{k>K0} (theta + 2 pi k)^{-s} = (2 pi)^{-s} zeta(s, K0 + 1 + theta/(2 pi)),
and symmetrically for the negative tail.  The binomial series converges like
(delta / (2 pi K0))^j, so J = 12 terms give far more than 40 digits for K0 >= 20.

The Toeplitz entries t_m = (2 pi)^{-1} int a(theta) e^{i theta m} d theta are then obtained
with the L-point trapezoid rule, which for the 2 pi-periodic analytic symbol is exact up to
aliases t_{m + jL} ~ e^{-delta_min |m + jL|}; L is chosen so that this is below 10^{-45}.
"""

from __future__ import annotations

import numpy as np
import mpmath as mp

from .params import Query, Mesh, P_ORDER


def _binom(Y, j):
    out = mp.mpf(1)
    for i in range(j):
        out *= (Y - i) / (i + 1)
    return out


def exact_fractional_symbol_mp(theta, delta, Y, p: int = P_ORDER, K0: int = 30, J: int = 12):
    """Phi_delta(theta) = sum_{k in Z} |N_p^(theta_k)|^2 (delta + i theta_k)^Y at mpmath precision.
    (The (delta - i theta)^Y block is the complex conjugate: conj(Phi_delta(theta)).)"""
    theta = mp.mpf(theta); delta = mp.mpf(delta); Y = mp.mpf(Y)
    two_pi = 2 * mp.pi
    total = mp.mpc(0)
    # explicit part
    for k in range(-K0, K0 + 1):
        tk = theta + two_pi * k
        if tk == 0:
            w = mp.mpf(1)
        else:
            w = (mp.sin(tk / 2) / (tk / 2)) ** (2 * p)
        total += w * (delta + 1j * tk) ** Y
    # tails: common factor (2 sin(theta/2))^(2p)
    s6 = (2 * mp.sin(theta / 2)) ** (2 * p)
    if s6 != 0:
        eY_p = mp.exp(1j * mp.pi * Y / 2)
        eY_m = mp.exp(-1j * mp.pi * Y / 2)
        a_plus = K0 + 1 + theta / two_pi
        a_minus = K0 + 1 - theta / two_pi
        tail = mp.mpc(0)
        for j in range(J):
            s = 2 * p + j - Y
            bj = _binom(Y, j)
            zp = two_pi ** (-s) * mp.zeta(s, a_plus)
            zm = two_pi ** (-s) * mp.zeta(s, a_minus)
            tail += bj * ((-1j * delta) ** j * eY_p * zp + (1j * delta) ** j * eY_m * zm)
        total += s6 * tail
    return total


def exact_symbol_mp(theta, q: Query, h, p: int = P_ORDER, K0: int = 30, J: int = 12):
    """a_mu(theta) exactly (mpmath): drift and mass in closed form, fractional part via tails."""
    theta = mp.mpf(theta); h = mp.mpf(h)
    C, G, M, Y, r = (mp.mpf(x) for x in q.mu)
    gY = mp.gamma(-Y)
    omega = C * gY * ((M - 1) ** Y - M ** Y + (G + 1) ** Y - G ** Y)
    lam0 = r + C * gY * (M ** Y + G ** Y)
    if p != 3:
        raise NotImplementedError
    m = mp.mpf(11) / 20 + mp.mpf(13) / 30 * mp.cos(theta) + mp.mpf(1) / 60 * mp.cos(2 * theta)
    d1 = mp.mpf(5) / 6 * mp.sin(theta) + mp.mpf(1) / 12 * mp.sin(2 * theta)
    phiG = exact_fractional_symbol_mp(theta, G * h, Y, p, K0, J)
    phiM = mp.conj(exact_fractional_symbol_mp(theta, M * h, Y, p, K0, J))
    return -(r - omega) * 1j / h * d1 + lam0 * m - C * gY * h ** (-Y) * (phiG + phiM)


def exact_toeplitz_entries_mp(q: Query, N: int, lags, dps: int = 40, p: int = P_ORDER,
                              which: str = "a", L: int | None = None, verbose: bool = False):
    """Exact Galerkin Toeplitz entries t_m (mpmath) for the requested lags.
    which = 'a' (evolution symbol), 'G' / 'M' (one fractional block (delta -/+ i theta)^Y,
    without the C Gamma(-Y) h^{-Y} factor)."""
    with mp.workdps(dps):
        msh = Mesh(N=N, p=p); h = mp.mpf(msh.h)
        lags = list(lags)
        mmax = max(abs(int(m)) for m in lags)
        dmin = min(q.G, q.M) * msh.h
        if L is None:
            # aliasing |t_{m + L}| ~ e^{-dmin (L - mmax)} : want < 10^{-(dps+5)}
            L = int(mmax + np.ceil((dps + 5) * np.log(10) / dmin)) + 8
        th = [-mp.pi + 2 * mp.pi * j / L for j in range(L)]
        if which == "a":
            vals = [exact_symbol_mp(t, q, h, p) for t in th]
        elif which == "G":
            vals = [exact_fractional_symbol_mp(t, q.G * msh.h, q.Y, p) for t in th]
        elif which == "M":
            vals = [mp.conj(exact_fractional_symbol_mp(t, q.M * msh.h, q.Y, p)) for t in th]
        else:
            raise ValueError(which)
        out = {}
        for m in lags:
            acc = mp.mpc(0)
            for j in range(L):
                acc += vals[j] * mp.expj(th[j] * m)
            out[int(m)] = acc / L
        return out, L


def eigvals_min_mp(S: np.ndarray, dps: int = 40) -> str:
    """lambda_min of a (double-precision) symmetric matrix computed at `dps` digits."""
    with mp.workdps(dps):
        A = mp.matrix(S.tolist())
        E = mp.eigsy(A, eigvals_only=True)
        return mp.nstr(min(E), dps)
