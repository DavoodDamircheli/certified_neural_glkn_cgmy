"""Independent CGMY characteristic function (Gate C).

Derives the risk-neutral characteristic exponent of the log-price
increment X_T = log(S_T/S_0) for the CGMY process exactly as defined in
main.tex (sec:model, ssec:operator), WITHOUT importing anything from
vendor/cgmy_bspline or src/cgmy -- this module only reads main.tex's
equations and re-derives them from the stated PIDE symbol.

Derivation (from main.tex eq:cgmy-parameters -- eq:symbol):

  A_mu(xi) = -(r-omega_S) i xi - C*Gamma(-Y)*[(M-i xi)^Y + (G+i xi)^Y]
             + lambda_0(mu),
  lambda_0(mu) = r + C*Gamma(-Y)*(M^Y + G^Y),
  omega_S(mu)  = C*Gamma(-Y)*[(M-1)^Y - M^Y + (G+1)^Y - G^Y].

Substituting lambda_0 gives
  A_mu(xi) = r - psi(xi; mu),
  psi(xi; mu) := i*xi*(r-omega_S)
                 + C*Gamma(-Y)*[(M-i*xi)^Y - M^Y + (G+i*xi)^Y - G^Y].

The backward PIDE d_tau V + A V = 0, V(.,0)=payoff, Fourier-transforms (in
x) to d_tau Vhat = -A_mu(xi) Vhat = (psi(xi)-r) Vhat, i.e.
  Vhat(xi,T) = e^{-rT} * e^{T*psi(xi)} * payoff_hat(xi).
This is exactly the standard discounted-expectation representation
  V(x0,T) = e^{-rT} * E[payoff(x0+X_T)],  X_T's char. fn. = exp(T*psi(xi)),
so psi is the Levy/characteristic exponent of the log-return process this
project's PIDE actually prices -- an independent pricer must use this
exact psi (not a generic off-the-shelf CGMY convention) to be pricing the
SAME process the Galerkin solver discretizes.

Martingale check (built into the algebra, verified numerically below, not
assumed): psi(-i) = (1)*(r-omega_S) + C*Gamma(-Y)*[(M-1)^Y-M^Y+(G+1)^Y-G^Y]
 = (r-omega_S) + omega_S = r exactly, so phi(-i;T)=exp(T*psi(-i))=exp(rT),
i.e. E[e^{X_T}] = e^{rT} -- discounted stock is a martingale under psi.

Two independent numerical implementations are provided:
  - `psi_numpy`: vectorized, double precision (for the COS method).
  - `psi_mpmath`: scalar, arbitrary precision via mpmath (for the
    high-precision Fourier-integration cross-check). Implemented
    separately (not calling psi_numpy) so a bug in one is unlikely to be
    reproduced identically in the other.

(C, G, M, Y, r) are all required to satisfy main.tex's eq:cgmy-parameters:
C>0, G>0, M>1, Y in (0,2), Y != 1. No validation against the production
parameter box is performed here; callers pick the box.
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import numpy as np
from scipy.special import gamma as _scipy_gamma


@dataclass(frozen=True)
class CGMYMarketParams:
    C: float
    G: float
    M: float
    Y: float
    r: float
    T: float


def omega_S_numpy(mu: CGMYMarketParams) -> float:
    gY = _scipy_gamma(-mu.Y)
    return mu.C * gY * ((mu.M - 1.0) ** mu.Y - mu.M ** mu.Y
                        + (mu.G + 1.0) ** mu.Y - mu.G ** mu.Y)


def psi_numpy(xi: np.ndarray, mu: CGMYMarketParams) -> np.ndarray:
    """Vectorized characteristic exponent psi(xi; mu), double precision.
    xi may be real or complex (COS only ever calls it with real xi)."""
    xi = np.asarray(xi, dtype=complex)
    gY = _scipy_gamma(-mu.Y)
    omega_S = omega_S_numpy(mu)
    drift = mu.r - omega_S
    term_M = (mu.M - 1j * xi) ** mu.Y - mu.M ** mu.Y
    term_G = (mu.G + 1j * xi) ** mu.Y - mu.G ** mu.Y
    return 1j * xi * drift + mu.C * gY * (term_M + term_G)


def char_fn_numpy(xi: np.ndarray, mu: CGMYMarketParams) -> np.ndarray:
    """phi(xi; T) = E[e^{i xi X_T}] = exp(T * psi(xi; mu))."""
    return np.exp(mu.T * psi_numpy(xi, mu))


def martingale_check_numpy(mu: CGMYMarketParams) -> float:
    """Returns E[e^{X_T}]/e^{rT} - 1, should be ~0 to float64 precision."""
    val = char_fn_numpy(np.array([-1j]), mu)[0]
    expected = np.exp(mu.r * mu.T)
    return float(np.real(val / expected - 1.0))


# ---------------------------------------------------------------------
# Independent mpmath (arbitrary precision) implementation. Deliberately
# NOT sharing code with the numpy version above: Gamma(-Y) is recomputed
# via mpmath.gamma (a different library implementation than
# scipy.special.gamma), and the complex powers use mpmath.mpc arithmetic.
# ---------------------------------------------------------------------

def psi_mpmath(xi, mu: CGMYMarketParams, dps: int = 50):
    """Scalar characteristic exponent via mpmath, arbitrary precision.
    xi may be a real float or an mpmath complex (e.g. -1j for the
    martingale check, or v - i*(alpha+1) for Carr-Madan damping).

    Deliberately does NOT force `mp.workdps(dps)` internally: this
    function is called from inside `mp.diff` (see `cos_pricer._cumulants_mpmath`),
    which raises the AMBIENT working precision well above `dps` to absorb
    finite-difference cancellation for higher-order derivatives -- forcing
    `dps` here would clobber that extra precision and silently return 0
    for the 2nd/4th cumulants (this was caught exactly this way during
    Gate C development: both came back as exact 0.0 before this fix). The
    `dps` parameter is accepted for interface symmetry with
    `char_fn_mpmath` but only used to size default-precision conversions
    when the caller has not already raised the ambient precision higher.
    """
    C, G, M, Y, r = mp.mpf(mu.C), mp.mpf(mu.G), mp.mpf(mu.M), mp.mpf(mu.Y), mp.mpf(mu.r)
    xi_mp = mp.mpc(xi)
    gY = mp.gamma(-Y)
    omega_S = C * gY * ((M - 1) ** Y - M ** Y + (G + 1) ** Y - G ** Y)
    drift = r - omega_S
    term_M = (M - 1j * xi_mp) ** Y - M ** Y
    term_G = (G + 1j * xi_mp) ** Y - G ** Y
    return 1j * xi_mp * drift + C * gY * (term_M + term_G)


def char_fn_mpmath(xi, mu: CGMYMarketParams, dps: int = 50):
    with mp.workdps(dps):
        return mp.e ** (mp.mpf(mu.T) * psi_mpmath(xi, mu, dps=dps))


def martingale_check_mpmath(mu: CGMYMarketParams, dps: int = 50) -> float:
    with mp.workdps(dps):
        val = char_fn_mpmath(mp.mpc(0, -1), mu, dps=dps)
        expected = mp.e ** (mp.mpf(mu.r) * mp.mpf(mu.T))
        return float((val / expected - 1).real)
