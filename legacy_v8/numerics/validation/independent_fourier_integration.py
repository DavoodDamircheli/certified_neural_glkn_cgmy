"""Second, independent reference pricer (Gate C): Carr-Madan (1999) damped
Fourier integration, using mpmath arbitrary-precision quadrature and the
mpmath characteristic function (validation.cgmy_reference_pricing.psi_mpmath
/char_fn_mpmath) -- a different formula AND a different numerical method
from validation.cos_pricer's cosine-series expansion, so the two do not
share a code path or even a characteristic-function implementation.

Carr-Madan call formula. Let k = ln(K), phi(u;T) = E[e^{iu X_T}] with
X_T = ln(S_T/S0) (char_fn_mpmath), so E[e^{iu ln S_T}] = e^{iu ln S0} *
phi(u;T). For a damping parameter alpha>0 with alpha+1 < M (needed so that
phi(v-(alpha+1)i; T) is finite -- the CGMY exponential-moment strip is
Im(u) in (-M, G), see cgmy_reference_pricing.py's docstring):

  Psi(v) = e^{-rT} * e^{i(v-(alpha+1)i) ln S0} * phi(v-(alpha+1)i; T)
            / (alpha^2 + alpha - v^2 + i(2*alpha+1)*v)

  Call(K) = (exp(-alpha*k) / pi) * integral_0^infty Re[ exp(-i*v*k) * Psi(v) ] dv

Put price recovered via put-call parity: Put = Call - S0 + K*exp(-rT).

This is the standard Carr-Madan damped-transform formula (not re-derived
here from scratch -- it is a well-established, widely checked result,
unlike the model-specific symbol psi which WAS re-derived from main.tex).
Its correctness as implemented is validated empirically below by (a) the
martingale check on phi itself (shared derivation, checked independently
in cgmy_reference_pricing.py) and (b) agreement with the independently
implemented COS method on identical inputs (tests/v3/test_cos_pricer.py).
"""
from __future__ import annotations

import mpmath as mp

from validation.cgmy_reference_pricing import CGMYMarketParams, char_fn_mpmath


def _choose_alpha(mu: CGMYMarketParams) -> float:
    """Damping parameter satisfying alpha + 1 < M with margin, alpha>0."""
    max_alpha = mu.M - 1.0
    alpha = min(0.75, 0.5 * max_alpha)
    if alpha <= 0:
        raise ValueError(f"Cannot choose a valid Carr-Madan alpha for M={mu.M} (need M>1).")
    return alpha


def _find_cutoff(integrand, start: float = 5.0, tol: float = 1e-14, max_v: float = 400.0) -> float:
    """Find v_cut such that |integrand(v)| < tol for v >= v_cut, by doubling."""
    v = start
    while v < max_v:
        if abs(integrand(v)) < tol and abs(integrand(v * 1.5)) < tol:
            return v
        v *= 2.0
    return max_v


def carr_madan_put_price(mu: CGMYMarketParams, S0: float, K: float, dps: int = 40, max_v: float = 400.0,
                         panel_width: float | None = None) -> dict:
    """Carr-Madan put via mpmath quadrature. ``max_v`` caps the cutoff search
    (default 400 = the historical setting); ``cutoff_cap_hit`` in the result is
    True when the search never found integrand decay below tolerance before the
    cap, i.e. the result is cutoff-limited (v5 M9). ``panel_width`` (None = the
    historical 4-panel quadrature) integrates over [0, v_cut] panel by panel; a
    single mp.quad over a very long oscillatory range exhausted >25 GB per process
    in the v5 M9 run, so extended cutoffs must use panels."""
    with mp.workdps(dps):
        alpha = _choose_alpha(mu)
        k = mp.log(K)
        lnS0 = mp.log(S0)
        r, T = mp.mpf(mu.r), mp.mpf(mu.T)

        def Psi(v):
            u = mp.mpc(v, -(alpha + 1.0))
            phi_logS = mp.e ** (1j * u * lnS0) * char_fn_mpmath(u, mu, dps=dps)
            denom = (alpha ** 2 + alpha - v ** 2) + 1j * (2 * alpha + 1) * v
            return mp.e ** (-r * T) * phi_logS / denom

        def integrand(v):
            return (mp.e ** (-1j * v * k) * Psi(v)).real

        v_cut = _find_cutoff(lambda v: integrand(v), start=5.0, tol=mp.mpf(10) ** (-(dps - 5)), max_v=max_v)
        if panel_width is None:
            integral = mp.quad(integrand, [0, v_cut / 4, v_cut / 2, v_cut])
        else:
            edges = [mp.mpf(0)]
            while edges[-1] < v_cut:
                edges.append(min(edges[-1] + panel_width, mp.mpf(v_cut)))
            integral = mp.fsum(mp.quad(integrand, [a_, (a_ + b_) / 2, b_]) for a_, b_ in zip(edges[:-1], edges[1:]))
        call = float((mp.e ** (-alpha * k) / mp.pi) * integral)
        put = call - S0 + K * float(mp.e ** (-r * T))
        return {
            "price": put,
            "call_price": call,
            "alpha": alpha,
            "v_cutoff": float(v_cut),
            "cutoff_cap_hit": bool(v_cut >= max_v),
            "max_v": float(max_v),
            "dps": dps,
            "panel_width": panel_width,
        }
