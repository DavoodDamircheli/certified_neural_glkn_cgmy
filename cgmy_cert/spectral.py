"""Rigorous spectral quantities from the symbol (note, Sections 3-4).

    s_min(mu)   = min_theta s_mu(theta)                       (Theorem 3.2: lambda_min(S_I) >= s_min)
    alpha_lb    = s_min - eps_K                                (valid for the K-truncated and the exact matrix)
    beta_mu     = 0.99 * min_{theta != 0} (s - r m) / g1       (Theorem 4.1)
    x_mu        = r m + beta g1,  X_I = r M_I + beta K1_I      (banded, bandwidth p-1)
    Theta_mu    = max_theta s / x                              (effectivity bound of the cheap certificate)

All minima/maxima are taken on a fine uniform grid (L_fine points) and shifted outward by the
derivative margin (pi / L_fine) * max |f'|, with f' evaluated analytically on the same grid.
The only non-enclosed step is that max |f'| is itself a grid maximum; see the note, Remark 3.4.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy import pi

from .params import Query, Mesh, P_ORDER
from .symbols import symbols, Symbols, closed_form_derivatives, graded_theta_grid


@dataclass
class SpectralData:
    q: Query
    N: int
    K: int
    L_fine: int
    s_min: float          # grid min of s minus derivative margin
    s_min_grid: float
    margin_s: float
    eps_K: float
    alpha_lb: float       # s_min - eps_K
    r_mmin: float         # crude bound r * m_min
    beta: float           # 0.99 * (min ratio - margin)
    beta_grid: float
    Theta: float          # max s/x + margin
    Theta_grid: float
    x_min: float          # min_theta x_mu(theta) (for the EIM allowance with X_I)
    X_entries: dict       # banded entries of X_I = r M_I + beta K1_I


def _grid_extreme(f: np.ndarray, fprime: np.ndarray, theta: np.ndarray, kind: str) -> tuple[float, float, float]:
    """(value with margin, grid value, margin) for the min or max of a smooth 2 pi-periodic
    function sampled on the sorted grid `theta` (cells between consecutive points, plus the
    wrap-around cell).  Cell-wise derivative margin: on a cell of width w the function deviates
    from its endpoint values by at most (w/2) sup_cell |f'|, and sup_cell |f'| is taken as the
    larger endpoint value of |f'|.  (A global margin (pi/L) max|f'| is far too crude when f' is
    large away from the extremum, e.g. s' ~ h^{-Y} at theta = pi for large N.)"""
    w = np.diff(np.concatenate([theta, [theta[0] + 2 * pi]]))
    fp = np.abs(fprime)
    fp_cell = np.maximum(fp, np.roll(fp, -1))
    f_lo = np.minimum(f, np.roll(f, -1)); f_hi = np.maximum(f, np.roll(f, -1))
    if kind == "min":
        cand = f_lo - 0.5 * w * fp_cell
        j = int(np.argmin(cand))
        return float(cand[j]), float(np.min(f)), float(np.min(f) - cand[j])
    cand = f_hi + 0.5 * w * fp_cell
    j = int(np.argmax(cand))
    return float(cand[j]), float(np.max(f)), float(cand[j] - np.max(f))


FAST_L0, FAST_NFINE = 512, 250
FAST_K_LADDER = (10, 20, 40, 100)


def fast_grid(q: Query, N: int, L0: int = FAST_L0, n_fine: int = FAST_NFINE, p: int = P_ORDER) -> np.ndarray:
    return graded_theta_grid(q, N, L0=L0, n_fine=n_fine, p=p)


def fast_alias_count(q: Query, N: int, rel: float = 0.02, p: int = P_ORDER) -> int:
    """Smallest K' in FAST_K_LADDER with eps_K'(Lemma 1.2) <= rel * r min m: the dropped tail (weighted by sin^6) then
    changes s by at most 2% of r m anywhere."""
    from .symbols import eps_K_fractional, mass_symbol
    h = Mesh(N=N, p=p).h
    m_min = float(np.min(mass_symbol(np.linspace(-pi, pi, 4001), p)))
    for K in FAST_K_LADDER:
        if eps_K_fractional(q, h, K, p) <= rel * q.r * m_min:
            return K
    return FAST_K_LADDER[-1]


def spectral_data_fast(q: Query, N: int, p: int = P_ORDER, safety: float = 0.99, K: int | None = None) -> SpectralData:
    """Online-path spectral data (follow-up 1 of Phase 3): coarse graded grid (FAST_L0 uniform + FAST_NFINE geometric
    points on the scale min(G, M) h near 0) and an adaptive alias count K'.  Rigour: the dropped aliases |k| > K' of the
    exact symbol (and of the K = 100 assembly) carry the factor |N^(theta_k)|^2 = sin^{2p}(theta/2) (2/theta_k)^{2p}, so
    |s_exact - s_K'| <= w(theta) eps_K', w = sin^{2p}(theta/2) (Lemma 1.2 bounds the sum with sin <= 1).  alpha_lb and
    beta use the lower envelope s_K' - w eps_K', Theta the upper one, each with the cell-wise derivative margin of
    _grid_extreme (w' included).  The tail vanishes at theta = 0 to order 2p, so beta (a ratio with g1 ~ theta^2) stays
    finite.  Only tightness is lost against the fine mode."""
    th0 = fast_grid(q, N, p=p)
    w0 = np.sin(0.5 * th0) ** (2 * p)
    # alias count: smallest K' whose sin^{2p}-weighted tail bound is <= 2% of s - r m on the grid (theta != 0): the tail
    # then perturbs alpha_lb, beta and Theta by <= ~2%; the bound itself stays rigorous whatever K' is chosen
    for K_try in ((K,) if K is not None else FAST_K_LADDER):
        sym = symbols(q, N, K=K_try, p=p, with_derivative=True, theta=th0)
        nz = np.abs(th0) > 1e-14
        rel = float(np.max(w0[nz] * sym.eps_K / np.maximum(sym.s[nz] - q.r * sym.m[nz], 1e-300)))
        if rel <= 0.02:
            break
    K = K_try
    th, s, m, g1, sp = sym.theta, sym.s, sym.m, sym.g1, sym.s_prime
    dm, _, dg1 = closed_form_derivatives(th, p)
    eps = sym.eps_K
    w = w0
    wp = p * np.sin(0.5 * th) ** (2 * p - 1) * np.cos(0.5 * th)          # d/dtheta sin^{2p}(theta/2)
    sl, slp = s - w * eps, sp - wp * eps
    su, sup_ = s + w * eps, sp + wp * eps
    r = q.r
    s_min, s_min_grid, margin_s = _grid_extreme(sl, slp, th, "min")
    mask = np.abs(th) > 1e-14
    f = sl[mask] - r * m[mask]; g = g1[mask]
    fp = slp[mask] - r * dm[mask]; gp = dg1[mask]
    beta_lb, beta_grid, _ = _grid_extreme(f / g, (fp * g - f * gp) / g ** 2, th[mask], "min")
    beta = safety * beta_lb
    if beta <= 0:
        raise ValueError(f"comparison norm failed (fast mode): beta={beta} (query {q})")
    x = r * m + beta * g1
    xp = r * dm + beta * dg1
    Theta, Theta_grid, _ = _grid_extreme(su / x, (sup_ * x - su * xp) / x ** 2, th, "max")
    x_min, _, _ = _grid_extreme(x, xp, th, "min")
    from .symbols import _poly_entries, mass_symbol, stiffness_symbol
    Me = _poly_entries(lambda t: mass_symbol(t, p), p); Ke = _poly_entries(lambda t: stiffness_symbol(t, p), p)
    X_entries = {k: r * Me[k] + beta * Ke[k] for k in Me}
    return SpectralData(q=q, N=N, K=K, L_fine=th.size, s_min=s_min, s_min_grid=s_min_grid, margin_s=margin_s,
                        eps_K=eps, alpha_lb=s_min, r_mmin=r * float(np.min(m)), beta=beta, beta_grid=beta_grid,
                        Theta=Theta, Theta_grid=Theta_grid, x_min=x_min, X_entries=X_entries)


def spectral_data(q: Query, N: int, K: int = 100, L_fine: int | None = None, p: int = P_ORDER,
                  safety: float = 0.99, sym: Symbols | None = None, mode: str = "fine") -> SpectralData:
    """All symbol-derived spectral quantities for (mu, N) in one pass over a graded fine grid
    (uniform 2^14 points plus a geometric refinement on the scale min(G,M) h near theta = 0).
    mode = "fast": the online-path variant `spectral_data_fast` (coarse grid, adaptive alias count, rigorous)."""
    if mode == "fast":
        return spectral_data_fast(q, N, p=p, safety=safety)
    if sym is None:
        th = graded_theta_grid(q, N, L0=2 ** 14 if L_fine is None else L_fine, p=p)
        sym = symbols(q, N, K=K, p=p, with_derivative=True, theta=th)
    L = sym.L
    th, s, m, g1, sp = sym.theta, sym.s, sym.m, sym.g1, sym.s_prime
    dm, _, dg1 = closed_form_derivatives(th, p)
    r = q.r
    # --- s_min, alpha_lb
    s_min, s_min_grid, margin_s = _grid_extreme(s, sp, th, "min")
    eps = sym.eps_K
    # --- beta: min over theta != 0 of (s - r m)/g1
    mask = np.abs(th) > 1e-14
    f = s[mask] - r * m[mask]; g = g1[mask]
    fp = sp[mask] - r * dm[mask]; gp = dg1[mask]
    ratio = f / g
    ratio_p = (fp * g - f * gp) / g ** 2
    beta_lb, beta_grid, _ = _grid_extreme(ratio, ratio_p, th[mask], "min")
    beta = safety * beta_lb
    if beta <= 0:
        raise ValueError(f"comparison norm failed: beta={beta} (query {q})")
    # --- Theta = max s/x with x = r m + beta g1
    x = r * m + beta * g1
    xp = r * dm + beta * dg1
    Theta, Theta_grid, _ = _grid_extreme(s / x, (sp * x - s * xp) / x ** 2, th, "max")
    x_min, _, _ = _grid_extreme(x, xp, th, "min")
    # banded entries of X_I
    from .symbols import _poly_entries, mass_symbol, stiffness_symbol
    Me = _poly_entries(lambda t: mass_symbol(t, p), p); Ke = _poly_entries(lambda t: stiffness_symbol(t, p), p)
    X_entries = {k: r * Me[k] + beta * Ke[k] for k in Me}
    m_min = float(np.min(m))
    return SpectralData(q=q, N=N, K=K, L_fine=L, s_min=s_min, s_min_grid=s_min_grid, margin_s=margin_s,
                        eps_K=eps, alpha_lb=s_min - eps, r_mmin=r * m_min, beta=beta, beta_grid=beta_grid,
                        Theta=Theta, Theta_grid=Theta_grid, x_min=x_min, X_entries=X_entries)


def s_min(q: Query, N: int, K: int = 100, **kw) -> float:
    return spectral_data(q, N, K=K, **kw).s_min


def alpha_lb(q: Query, N: int, K: int = 100, **kw) -> float:
    return spectral_data(q, N, K=K, **kw).alpha_lb


def comparison_norm(q: Query, N: int, K: int = 100, **kw) -> tuple[float, dict, float]:
    """(beta, X_entries, Theta)."""
    d = spectral_data(q, N, K=K, **kw)
    return d.beta, d.X_entries, d.Theta


def comparison_norm_wide(q: Query, N: int, d: int, K: int = 100, L_fine: int | None = None,
                         p: int = P_ORDER, safety: float = 0.99) -> dict:
    """Optional (plan 1.2): best trigonometric minorant of degree d of s - r m of the form
    sum_{j=1..d} c_j (1 - cos j theta) >= 0 ... Implemented as a convex LP-free heuristic:
    x = r m + beta_d * g_d with g_d(theta) = sum_{j<=d} w_j (1 - cos(j theta)), w_j = 1/j^2
    (a wider-band positive minorant), beta_d = min (s - r m)/g_d.  Returns entries and Theta.
    Only to be used if E0.3 shows Theta > 4 at the production N."""
    th = graded_theta_grid(q, N, p=p)
    sym = symbols(q, N, K=K, p=p, with_derivative=True, theta=th)
    th, s, m, sp = sym.theta, sym.s, sym.m, sym.s_prime; L = th.size
    dm, _, _ = closed_form_derivatives(th, p)
    w = {j: 1.0 / j ** 2 for j in range(1, d + 1)}
    gd = sum(wj * (1 - np.cos(j * th)) for j, wj in w.items())
    gdp = sum(wj * j * np.sin(j * th) for j, wj in w.items())
    mask = np.abs(th) > 1e-14
    f = s[mask] - q.r * m[mask]; fp = sp[mask] - q.r * dm[mask]
    ratio = f / gd[mask]; ratio_p = (fp * gd[mask] - f * gdp[mask]) / gd[mask] ** 2
    beta_lb, _, _ = _grid_extreme(ratio, ratio_p, th[mask], "min")
    beta = safety * beta_lb
    x = q.r * m + beta * gd; xp = q.r * dm + beta * gdp
    Theta, _, _ = _grid_extreme(s / x, (sp * x - s * xp) / x ** 2, th, "max")
    from .symbols import _poly_entries, mass_symbol
    Me = _poly_entries(lambda t: mass_symbol(t, p), p)
    entries = {k: q.r * Me.get(k, 0.0) for k in range(-d, d + 1)}
    entries[0] += beta * sum(w.values())
    for j, wj in w.items():
        entries[j] += -0.5 * beta * wj; entries[-j] += -0.5 * beta * wj
    return dict(beta=beta, entries=entries, Theta=Theta, bandwidth=d)
