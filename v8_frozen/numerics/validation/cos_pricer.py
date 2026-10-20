"""Independent COS (Fang-Oosterlee) European put pricer for the CGMY
process defined in main.tex (Gate C, CGMY_V3_NUMERICAL_EXPERIMENT_ROADMAP.md).

Does not import vendor/cgmy_bspline or src/cgmy -- uses only
validation.cgmy_reference_pricing's char_fn_numpy (derived directly from
main.tex's stated symbol, see that module's docstring).

COS method (Fang & Oosterlee, 2008), specialised to a European put. Let
x0 = log(S0/K), y = log(S_T/K) = x0 + X_T. The put payoff as a function of
y is K*max(1-e^y, 0), nonzero only for y<0.

  V(x0,T) ~= e^{-rT} * sum_{k=0}^{N-1}' Re[ phi(k*pi/(b-a); T)
                 * exp(i*k*pi*(x0-a)/(b-a)) ] * V_k

Cosine coefficients V_k of the PUT payoff on [a,b] (nonzero support
intersected with (a,0)) are computed by direct numerical quadrature of
  V_k = (2/(b-a)) * integral_{max(a,-inf)}^{min(b,0)} K*(1-e^y)*cos(k*pi*(y-a)/(b-a)) dy
rather than the textbook closed-form (chi_k, psi_k) algebra, specifically
to avoid transcription risk in a from-scratch "independent" implementation
-- scipy.integrate.quad is accurate to the requested tolerance regardless
of k, and this is cheap enough for the query counts used here.

Truncation range [a, b] is NOT a fixed heuristic constant: it is derived
from the first few cumulants of X_T (computed via mpmath numerical
differentiation of psi at xi=0, not a hand-derived cumulant formula, for
the same transcription-risk reason), using the standard Fang-Oosterlee
recipe a,b = c1 +/- L*sqrt(c2 + sqrt(c4)), and then VALIDATED by an
explicit convergence sweep over L and N (see `converge_price`) rather than
trusted at face value -- the roadmap explicitly warns heuristic COS
truncation can misprice.
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import numpy as np
from scipy.integrate import quad

from validation.cgmy_reference_pricing import CGMYMarketParams, char_fn_numpy, psi_mpmath


def _cumulants_mpmath(mu: CGMYMarketParams, dps: int = 40) -> tuple[float, float, float]:
    """c1, c2, c4 of X_T via mpmath numerical differentiation of
    T*psi(xi) at xi=0 (cumulant generating function is T*psi(-i*t) in the
    real-exponent convention, but since psi here is already defined via
    E[e^{i xi X}], cumulants are c_n = (-i)^n d^n/dxi^n [T*psi(xi)] |_{xi=0}
    = T * (-i)^n psi^{(n)}(0)."""
    # Raise precision well above dps before differentiating: a 4th-order
    # finite difference needs several times dps's worth of extra guard
    # digits to survive cancellation (see psi_mpmath's docstring -- an
    # earlier version of this function forced workdps(dps) inside the
    # differentiated function itself and silently got c2=c4=0.0).
    with mp.workdps(dps + 60):
        def f(xi):
            return mp.mpf(mu.T) * psi_mpmath(xi, mu, dps=dps)

        c1 = complex(-1j * mp.diff(f, 0, 1))
        c2 = complex(-1 * mp.diff(f, 0, 2))
        c4 = complex(1 * mp.diff(f, 0, 4))
        return c1.real, c2.real, c4.real


def truncation_range(mu: CGMYMarketParams, L: float = 12.0) -> tuple[float, float]:
    c1, c2, c4 = _cumulants_mpmath(mu)
    width = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return c1 - width, c1 + width


_COS_DENSE_LIMIT = 1024 * 32768  # entries of the dense cosine matrix (~0.27 GB); covers N <= 1024


def _cosine_coeffs_put(a: float, b: float, N: int, K: float) -> np.ndarray:
    """V_k for k=0..N-1, by direct (vectorized) quadrature -- a single
    shared Gauss-Legendre rule evaluated once and dotted against every
    cos(k*...) basis function, rather than N separate adaptive
    scipy.integrate.quad calls (the latter is correct but was
    prohibitively slow for N in the hundreds across many queries/sweeps;
    Gauss-Legendre with enough nodes integrates the oscillatory
    cos(k*pi*(y-a)/(b-a)) factor exactly up to high polynomial degree, and
    the payoff factor K*(1-e^y) is smooth, so accuracy is governed by node
    count, not by N itself).

    Still an independent implementation choice (no closed-form chi_k/psi_k
    algebra from the COS literature is used anywhere here)."""
    d = min(b, 0.0)
    if d <= a:
        return np.zeros(N)

    # A single high-order leggauss call scales badly (numpy 1.26's
    # companion-eigenvalue algorithm is far worse than linear for node
    # counts in the thousands -- measured ~7.5s for n=1000 on this
    # machine, which made the original single-high-order-rule version of
    # this function the dominant cost of every convergence sweep). Instead
    # use a cheap, FIXED small-order rule (leggauss(8), effectively free)
    # composited over many subintervals of [a, d] -- a standard composite
    # Gauss quadrature construction, linear in the number of subintervals,
    # and still independent of any COS closed-form chi_k/psi_k algebra.
    base_order = 8
    base_nodes, base_weights = np.polynomial.legendre.leggauss(base_order)
    n_sub = max(200, 4 * N)
    sub_edges = np.linspace(a, d, n_sub + 1)
    sub_a, sub_b = sub_edges[:-1], sub_edges[1:]
    sub_half_width = 0.5 * (sub_b - sub_a)  # shape (n_sub,)
    sub_mid = 0.5 * (sub_b + sub_a)  # shape (n_sub,)
    # y[i, j] = mid_i + half_width_i * base_node_j
    y = sub_mid[:, None] + sub_half_width[:, None] * base_nodes[None, :]
    w = sub_half_width[:, None] * base_weights[None, :]
    y = y.ravel()
    w = w.ravel()
    payoff = K * (1.0 - np.exp(y))  # shape (n_sub * base_order,)

    k = np.arange(N)
    if N * y.size <= _COS_DENSE_LIMIT:
        # historical path, unchanged (bitwise identical to every pre-v5 result)
        theta = np.outer(k, (y - a) * np.pi / (b - a))  # shape (N, n_nodes)
        cos_basis = np.cos(theta)
        integral = cos_basis @ (payoff * w)  # shape (N,)
        return (2.0 / (b - a)) * integral
    # v5 M9: above the limit the dense (N, ~32N) cosine matrix does not fit in memory
    # (~34 GB at N=8192); compute it in row blocks. BLAS reduction order differs with
    # the block shape, so values agree with the dense path to rounding (~1e-16
    # relative), not bitwise -- only sizes that never ran before take this path.
    scaled = (y - a) * np.pi / (b - a)
    pw = payoff * w
    integral = np.empty(N)
    for k0 in range(0, N, 256):
        kb = k[k0:k0 + 256]
        integral[k0:k0 + len(kb)] = np.cos(np.outer(kb, scaled)) @ pw
    return (2.0 / (b - a)) * integral


def _cosine_coeffs_put_reference_quad(a: float, b: float, N: int, K: float) -> np.ndarray:
    """Slow, independent cross-check of `_cosine_coeffs_put`: the original
    N-separate-adaptive-quad-calls version, kept only for the unit test
    that validates the vectorized Gauss-Legendre rule against it on a
    small N (cheap at small N, too slow to use as the default at
    production N)."""
    d = min(b, 0.0)
    if d <= a:
        return np.zeros(N)
    Vk = np.zeros(N)
    for k in range(N):
        def integrand(y, _k=k):
            return K * (1.0 - np.exp(y)) * np.cos(_k * np.pi * (y - a) / (b - a))

        val, _ = quad(integrand, a, d, limit=200, epsabs=1e-13, epsrel=1e-13)
        Vk[k] = (2.0 / (b - a)) * val
    return Vk


def cos_put_price(mu: CGMYMarketParams, S0: float, K: float, N: int = 256,
                   trunc_range: tuple[float, float] | None = None,
                   L: float = 12.0) -> dict:
    """European put price via COS. Returns a dict with the price and the
    exact settings used (so callers can record provenance)."""
    if trunc_range is None:
        a, b = truncation_range(mu, L=L)
    else:
        a, b = trunc_range

    x0 = np.log(S0 / K)
    k = np.arange(N)
    u = k * np.pi / (b - a)
    phi = char_fn_numpy(u, mu)
    Vk = _cosine_coeffs_put(a, b, N, K)

    cos_term = np.cos(k * np.pi * (x0 - a) / (b - a))
    sin_term = np.sin(k * np.pi * (x0 - a) / (b - a))
    weighted = (np.real(phi) * cos_term - np.imag(phi) * sin_term) * Vk
    weighted[0] *= 0.5  # COS half-weight on k=0 ("'" in the sum)

    price = np.exp(-mu.r * mu.T) * np.sum(weighted)
    return {
        "price": float(price),
        "N": N,
        "a": float(a),
        "b": float(b),
        "L": L,
    }


def converge_price(mu: CGMYMarketParams, S0: float, K: float,
                    N_list=(32, 64, 128, 256, 512, 1024),
                    L_list=(6.0, 8.0, 10.0, 12.0, 16.0, 20.0),
                    rel_tol: float = 1e-8) -> dict:
    """Doubling sweep over (N, L) until the price stabilizes to rel_tol
    relative change between consecutive settings. Returns the converged
    price plus the settings used and the full convergence trace, so a
    caller never has to trust a single fixed (N, L) pair blindly.
    """
    trace = []
    prev = None
    converged_at = None
    for L in L_list:
        a, b = truncation_range(mu, L=L)
        for N in N_list:
            out = cos_put_price(mu, S0, K, N=N, trunc_range=(a, b), L=L)
            price = out["price"]
            rel_change = None if prev is None else abs(price - prev) / max(abs(price), 1e-12)
            trace.append({"L": L, "N": N, "price": price, "rel_change": rel_change})
            if rel_change is not None and rel_change < rel_tol and converged_at is None:
                converged_at = {"L": L, "N": N, "price": price}
            prev = price
    final = trace[-1]
    return {
        "price": final["price"],
        "N_used": final["N"],
        "L_used": final["L"],
        "converged_at": converged_at,
        "trace": trace,
        "flagged_not_converged": converged_at is None,
    }
