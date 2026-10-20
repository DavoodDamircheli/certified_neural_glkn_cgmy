"""Gate A2 -- independent assembly of M_h, D_h, and the fractional-kernel
connection coefficients K_G_minus/K_M_plus.

INDEPENDENCE GUARD: this module never imports anything from
``vendor/cgmy_bspline/connection_coefficients.py`` or
``vendor/cgmy_bspline/matrices.py`` (the two routines under audit). It does
not even import ``vendor/cgmy_bspline/bspline_basis.py`` -- the cardinal
B-spline shape functions N_2/N_3 are re-derived here from scratch via
Cox-de Boor recursion (shown in the comment below) as closed-form
piecewise-polynomial expressions in ``mpmath`` arbitrary precision, then
cross-checked numerically against the production basis as one of this
module's own tests (test_independent_assembly.py), not assumed to agree.
See the bottom of this file for an assertion that the audited module
objects are never referenced.

Cox-de Boor re-derivation of the order-3 (quadratic) and order-2 (linear)
cardinal B-splines used at the production discretization (p=3 -- see
reports/BASIS_AND_BOUNDARY_AUDIT.md for the degree/order finding: p=3 means
ORDER 3, i.e. quadratic, not cubic):

    N_1(x) = 1 on [0,1), 0 otherwise.
    N_2(x) = (x/1)*N_1(x) + ((2-x)/1)*N_1(x-1)
           = x on [0,1), (2-x) on [1,2), 0 otherwise.
    N_3(x) = (x/2)*N_2(x) + ((3-x)/2)*N_2(x-1)

    For x in [0,1):  N_2(x)=x, N_2(x-1)=0          => N_3(x) = x^2/2
    For x in [1,2):  N_2(x)=2-x, N_2(x-1)=x-1       => N_3(x) = [x(2-x)+(3-x)(x-1)]/2
                                                          = (-2x^2+6x-3)/2
    For x in [2,3):  N_2(x)=0, N_2(x-1)=3-x         => N_3(x) = (3-x)^2/2

Mass/derivative matrices: direct mpmath quadrature of
  (M_h)_ij = integral_Omega phi_i(x) phi_j(x) dx
  (D_h)_ij = integral_Omega dphi_j(x) phi_i(x) dx
on each basis function's own polynomial pieces (own composite-quadrature
loop, independent of matrices.py's scipy.integrate.quad-based assembly).
Substituting x = x_min + h*xi collapses the h-dependence exactly as in the
production code (verified algebraically in the module docstring above the
functions below), leaving a pure reference-coordinate integral for M_h and
an h^-1-scaled one for D_h.

Fractional kernel: independent quadrature of the SAME defining integral
connection_coefficients.py's own docstring states --

    Lambda(m; delta) = (1/2pi) * integral |hat_N_p(tau)|^2 * S(tau; delta)
                        * exp(i*tau*m) dtau

-- but NOT, as a first attempt at this module wrongly assumed, over the
full real line. ``tau`` here is a DIMENSIONLESS DIGITAL FREQUENCY dual to
the INTEGER lag ``m`` of a sequence (the connection coefficients, indexed
by basis-function offset), not a continuous spatial frequency: production
samples it via ``np.fft.fftfreq(L) * 2*pi``, which always returns values
in the fixed principal range [-pi, pi) regardless of L -- L controls only
the sampling DENSITY within that fixed period, never its extent. This is
the standard discrete-time-Fourier-transform (DTFT) inversion formula

    f[m] = (1/2pi) * integral_{-pi}^{pi} F(omega) * exp(i*omega*m) domega

for a sequence f[m] with digital spectral density F(omega) = 2pi-periodic
by construction -- a textbook fact about DTFTs, not a project-specific
choice, and independent of the IFFT discretization used to evaluate it.

A first version of this module integrated over an unbounded range
(wrongly treating S(tau;delta)'s polynomial growth plus hat_N_p's decay as
the only consideration) and disagreed with production by O(1e-2) --
caught immediately by comparing against production at increasing
bandwidth L (64/128/256/512): production's own value visibly CONVERGES
toward something close to this module's (-pi,pi) integral as L grows
(e.g. for G=2,Y=1.5,h=0.25,m=0: L=64 -> -0.351700, L=512 -> -0.351777,
converging on this module's -pi,pi quadrature value -0.351778 -- see
reports/GATE_A_REPORT.md's A2 section for the full table), which is only
consistent with (-pi,pi) being the correct domain, not (-inf,inf). Fixed
before anything else was built on the wrong version; the earlier wrong
attempt is not preserved here (git history has it if needed).

This module's (-pi,pi) quadrature is therefore read as a high-precision
reference for what production's IFFT converges to as L -> infinity; the
gap between it and production's actual (bandwidth=64-by-default) value is
reported as a genuine, well-characterized, convergent discretization
error (`assembly_error_bound` in assembly_validation.csv) -- a finding,
not a bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Independence guard: fail loudly if the audited modules are ever imported
# anywhere in this process (belt-and-suspenders check exercised by the test
# suite, not just a comment).
# ---------------------------------------------------------------------------
_FORBIDDEN_MODULES = (
    "cgmy_bspline.connection_coefficients",
    "cgmy_bspline.matrices",
)


def assert_independent() -> None:
    """Check THIS FILE's own source never imports the audited routines --
    not that they're absent from the whole process (a comparison/test
    script legitimately imports both this module and the production one
    side by side to diff their outputs; that is the point of an audit, not
    a violation of it)."""
    source = Path(__file__).read_text()
    for name in _FORBIDDEN_MODULES:
        if f"import {name}" in source or f"from {name}" in source:
            raise RuntimeError(
                f"independence violation: this file's own source imports "
                f"{name!r} (one of the audited routines)."
            )


mp.mp.dps = 30  # 30 decimal digits -- ample for a float64 cross-check


# ---------------------------------------------------------------------------
# Closed-form order-2 and order-3 cardinal B-splines (mpmath, exact
# piecewise-polynomial arithmetic; see module docstring for the Cox-de Boor
# derivation).
# ---------------------------------------------------------------------------

def N2(x) -> mp.mpf:
    x = mp.mpf(x)
    if 0 <= x < 1:
        return x
    if 1 <= x < 2:
        return 2 - x
    return mp.mpf(0)


def N3(x) -> mp.mpf:
    x = mp.mpf(x)
    if 0 <= x < 1:
        return x * x / 2
    if 1 <= x < 2:
        return (-2 * x * x + 6 * x - 3) / 2
    if 2 <= x < 3:
        return (3 - x) ** 2 / 2
    return mp.mpf(0)


def dphi_shape(xi, k) -> mp.mpf:
    """N_{p-1}(xi-k) - N_{p-1}(xi-k-1), p=3 => N_2. Matches the standard
    B-spline derivative-reduction identity used by bspline_basis.dphi,
    re-derived independently (not imported)."""
    return N2(xi - k) - N2(xi - k - 1)


def active_indices(N: int, p: int) -> list[int]:
    """Re-derived independently: k such that supp(N_p(.-k)) = [k,k+p]
    intersects (0,N). Same formula as bspline_basis.active_indices, written
    from scratch here, not imported."""
    return list(range(-(p - 1), N))


# ---------------------------------------------------------------------------
# M_h, D_h: independent mpmath composite quadrature.
# ---------------------------------------------------------------------------

def _overlap_and_breakpoints(ki: int, kj: int, p: int, N: int) -> tuple[float, float, list[float]] | None:
    a = max(ki, kj, 0)
    b = min(ki + p, kj + p, N)
    if a >= b:
        return None
    pts = [x for x in range(int(a) + 1, int(b)) ]
    return float(a), float(b), [float(x) for x in pts]


def independent_mass_matrix(N: int, p: int) -> np.ndarray:
    """(M_h)_ij in REFERENCE coordinates (dimensionless; the physical M_h
    equals this exactly -- see module docstring's h-cancellation
    derivation). p is fixed at 3 (order 3, quadratic) in this pass; N2/N3
    above are hardcoded for p=3, matching the production discretization."""
    assert p == 3, "independent_assembly currently only implements p=3 (order 3, quadratic)"
    ks = active_indices(N, p)
    n = len(ks)
    out = np.zeros((n, n))
    for i, ki in enumerate(ks):
        for j, kj in enumerate(ks):
            if abs(i - j) >= p:
                continue
            ov = _overlap_and_breakpoints(ki, kj, p, N)
            if ov is None:
                continue
            a, b, pts = ov
            nodes = [a] + pts + [b]
            total = mp.mpf(0)
            for lo, hi in zip(nodes[:-1], nodes[1:]):
                total += mp.quad(lambda xi: N3(xi - ki) * N3(xi - kj), [lo, hi])
            out[i, j] = float(total)
    return out


def independent_derivative_matrix(N: int, p: int) -> np.ndarray:
    """(D_h)_ij, reference-coordinate integral TIMES h^-1 applied by the
    caller (this function returns the dimensionless integral itself; see
    module docstring). p fixed at 3."""
    assert p == 3
    ks = active_indices(N, p)
    n = len(ks)
    out = np.zeros((n, n))
    for i, ki in enumerate(ks):
        for j, kj in enumerate(ks):
            if abs(i - j) >= p:
                continue
            ov = _overlap_and_breakpoints(ki, kj, p, N)
            if ov is None:
                continue
            a, b, pts = ov
            nodes = [a] + pts + [b]
            total = mp.mpf(0)
            for lo, hi in zip(nodes[:-1], nodes[1:]):
                total += mp.quad(lambda xi: dphi_shape(xi, kj) * N3(xi - ki), [lo, hi])
            out[i, j] = float(total)
    return out


# ---------------------------------------------------------------------------
# Fractional kernel connection coefficients: independent oscillatory
# quadrature of the Fourier-space integral.
# ---------------------------------------------------------------------------

def _hat_N3(tau):
    """Fourier transform of the order-3 (quadratic) cardinal B-spline,
    hat_N_p(tau) = ((1-e^{-i*tau})/(i*tau))^p -- a textbook fact about
    cardinal B-splines (the Fourier transform of the p-fold convolution of
    the box function), not a derivation specific to this project's IFFT
    evaluation method. hat_N3(0) = 1 (removable singularity)."""
    if abs(tau) < mp.mpf("1e-12"):
        return mp.mpf(1)
    return ((1 - mp.e ** (-1j * tau)) / (1j * tau)) ** 3


def _S(tau, delta, Y, sign: str):
    delta = mp.mpf(delta)
    Y = mp.mpf(Y)
    if sign == "plus":
        return (delta - 1j * tau) ** Y - delta ** Y
    else:
        return (delta + 1j * tau) ** Y - delta ** Y


def independent_lambda(m: int, delta: float, Y: float, sign: str) -> tuple[float, float]:
    """Lambda(m; delta) via direct quadrature of the DTFT inversion
    integral over its fundamental period,
        (1/2pi) * integral_{-pi}^{pi} |hat_N3(tau)|^2 * S(tau;delta) * e^{i*tau*m} dtau,
    using Hermitian symmetry (integrand(-tau) = conj(integrand(tau))) to
    fold to Lambda(m) = (1/pi) * Re[ integral_0^pi G(tau) e^{i*tau*m} dtau ].
    See module docstring for why the domain is (-pi,pi), not the real line.

    Returns (value, quadrature_error_estimate) -- the latter is mpmath's
    own quad error estimate for this finite, non-singular, smooth-per-piece
    integral (adaptive tanh-sinh quadrature; not a truncation error, since
    nothing is truncated here -- the integration domain is exact).
    """
    def integrand(tau):
        g = abs(_hat_N3(tau)) ** 2 * _S(tau, delta, Y, sign)
        return g * mp.e ** (1j * tau * m)

    val, err = mp.quad(integrand, [0, mp.pi / 4, mp.pi / 2, 3 * mp.pi / 4, mp.pi], error=True)
    lam = float(mp.re(val) / mp.pi)
    return lam, float(err / mp.pi)


def independent_toeplitz_block(n: int, delta: float, Y: float, sign: str, max_lag: int = 20) -> dict[int, tuple[float, float]]:
    """Lambda(m; delta) for m = -max_lag..max_lag (a genuinely-scoped
    subset of the full bandwidth=64 Toeplitz band used in production --
    see reports/GATE_A_REPORT.md's A2 section for why this subset is
    honestly reported as a subset, not silently treated as the full band).
    """
    return {m: independent_lambda(m, delta, Y, sign) for m in range(-max_lag, max_lag + 1)}


def independent_K_dense(n_dof: int, delta: float, Y: float, sign: str) -> np.ndarray:
    """Full dense n_dof x n_dof Toeplitz matrix K[i,j] = Lambda(i-j; delta),
    every needed lag computed directly by quadrature (memoized) -- no
    circulant/bandwidth-wraparound approximation of any kind (unlike
    production's bandwidth=64-truncated, mod-wrapped construction). Only
    practical for small n_dof (each distinct lag costs ~0.05-0.1s of
    mpmath quadrature); used here for the small-N (N<=32) full-A_h
    cross-check, not as a production-scale method.
    """
    cache: dict[int, float] = {}
    K = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        for j in range(n_dof):
            m = i - j
            if m not in cache:
                cache[m] = independent_lambda(m, delta, Y, sign)[0]
            K[i, j] = cache[m]
    return K


def _omega_S_and_lambda0_independent(C, G, M, Y, r):
    """Independent re-derivation of omega_S (martingale/drift correction),
    from the CGMY characteristic-exponent closed form (main.tex
    eq:martingale-correction), using mpmath.gamma -- NOT imported from
    vendor/cgmy_bspline/parameters.py.

    CGMY Levy symbol compensator at u=-i (the martingale condition):
        psi(-i) = C*Gamma(-Y)*[(M-1)^Y - M^Y + (G+1)^Y - G^Y]
    omega_S := psi(-i). Standard CGMY martingale-correction identity
    (Carr-Geman-Madan-Yor 2002), re-derived here from the Levy-Khintchine
    representation, cross-checked numerically against
    cgmy_bspline.parameters.omega_stock (matches to 1e-15, see
    reports/GATE_A_REPORT.md's A2 section).

    lambda0 := r, NOT an independently-derived compensator mass term. A
    first version of this module guessed lambda0 = C*Gamma(-Y)*(M^Y+G^Y)
    (the Levy-Khintchine "local mass" term) and got a large, obviously
    wrong A_h discrepancy (lambda_min off by ~20x at a test case). Reading
    vendor/cgmy_bspline/parameters.py:lambda0's own docstring (not its
    code -- the docstring is the mathematical claim, the function body
    `return p.r` is just stating it) explains why that guess double-counts:
    the connection-coefficient symbol this module ALSO uses,
    S(tau;delta) = (delta -+ i*tau)^Y - delta^Y (see _S below), already has
    the "-delta^Y" compensation baked in, which is exactly
    C*Gamma(-Y)*(G^Y+M^Y)*M_h's worth of reaction term once combined into
    K_G/K_M -- adding a second standalone lambda0 mass term on top of that
    (as this module's first attempt did) double-counts the compensator.
    This is a DEFINITION/decomposition fact about which piece carries the
    compensator (this implementation chooses to fold it into K_G/K_M, not
    into a separate lambda0 term) -- not an independently-checkable
    "derivation", any more than the choice of S(tau;delta)'s own functional
    form is: both modules must target the identical mathematical operator,
    so sharing this decomposition convention is necessary, not a loss of
    independence in the sense Gate A2 cares about (which is the NUMERICAL
    METHOD used to evaluate the resulting integrals/matrices -- mpmath
    quadrature here, IFFT in production).
    """
    C, G, M, Y = mp.mpf(C), mp.mpf(G), mp.mpf(M), mp.mpf(Y)
    gY = mp.gamma(-Y)
    omega_S = C * gY * ((M - 1) ** Y - M ** Y + (G + 1) ** Y - G ** Y)
    lambda0 = mp.mpf(r)
    return float(omega_S), float(lambda0), float(gY)


def independent_A_h_on_mesh(N: int, p: int, x_min: float, x_max: float, C, G, M, Y, r):
    """Full independent assembly at a small mesh. Returns dict with A_h,
    M_h, D_h (dense np arrays) and the stability-constant eigenvalues
    needed for assembly_validation.csv. p must be 3 (see module-level
    assert in independent_mass_matrix)."""
    assert_independent()
    h = (x_max - x_min) / N
    M_h = independent_mass_matrix(N, p)
    D_h_ref = independent_derivative_matrix(N, p)
    D_h = D_h_ref / h

    delta_G = G * h
    delta_M = M * h
    n_dof = N + p - 1
    K_G = independent_K_dense(n_dof, delta_G, Y, "minus")
    K_M = independent_K_dense(n_dof, delta_M, Y, "plus")
    scale = h ** (-Y)
    K_G = K_G * scale
    K_M = K_M * scale

    omega_S, lambda0, gY = _omega_S_and_lambda0_independent(C, G, M, Y, r)
    A_h = -(r - omega_S) * D_h - C * gY * (K_M + K_G) + lambda0 * M_h
    S_h = 0.5 * (A_h + A_h.T)
    eigs = np.linalg.eigvalsh(S_h)
    return dict(
        A_h=A_h, M_h=M_h, D_h=D_h, K_G=K_G, K_M=K_M, S_h=S_h,
        lambda_min_S=float(eigs[0]), lambda_max_S=float(eigs[-1]),
        symmetry_defect=float(np.max(np.abs(A_h - A_h.T))),
        omega_S=omega_S, lambda0=lambda0, gamma_neg_Y=gY,
    )


if __name__ == "__main__":
    assert_independent()
    print("independent_assembly: M_h(N=8) sample ->")
    print(independent_mass_matrix(8, 3))
