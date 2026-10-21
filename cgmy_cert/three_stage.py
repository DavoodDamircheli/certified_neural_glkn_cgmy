"""Three-stage certificate (follow-up 5, item 2): reduced online -> banded full-order -> exact full-order.

Stage 1 (N-independent): eta_red = reduced online bound (cell norm X-bar, EIM allowance, rounding term).
    eta_red_aug <= eps                                   -> reduced_accept
    Lower bound for what the exact stage can achieve (Prop. 4.3 generalised, Lemma 2.5): with
    Theta-bar_mu = sup_theta s_mu / x-bar >= the generalised eigenvalues of (S_I, X-bar),
        ||rho_I^n||_{S^-1} >= (sqrt(u_n) - rnd_n) / sqrt(Theta-bar) - d_n,
    where u_n, rnd_n are the reduced per-step value and its rigorous rounding/compression term and d_n the EIM
    allowance (||rho_I - rho_Q||_{S^-1} <= eps_Q ||cbar_raw|| / sqrt(x-bar_min), since S_I >= X-bar on the cell).
    (beta_0^2 + dt sum max(0, .)^2)^{1/2} > eps          -> reject_cheap
Stage 2 (full order, A_I, the query's own X_I and Theta_mu; certificate.certify mode "two-stage"):
    eta^X_aug <= eps                                     -> banded_accept
    (beta_0^2 + eta^X^2 / Theta_mu)^{1/2} > eps          -> reject_banded
Stage 3 (exact, PCG bracket on S_I):                     -> exact_accept / exact_reject
Each stage is timed separately; Theta-bar costs one graded-grid symbol evaluation with adaptive alias count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .params import Query, P_ORDER
from .symbols import symbols, graded_theta_grid, closed_form_derivatives
from .spectral import _grid_extreme, spectral_data
from .certificate import certify
from .reduced_cert import conversion_alias_count

STATUSES = ("reduced_accept", "reject_cheap", "banded_accept", "reject_banded", "exact_accept", "exact_reject")


def theta_bar(q: Query, N: int, rbar: float, betabar: float, p: int = P_ORDER) -> float:
    """Upper bound of sup_theta s_mu(theta) / x-bar(theta): s_mu <= s_(mu,K') + eps_K' for the exact symbol and for the
    K = 100 truncation; grid maximum plus the cell-wise derivative margin on the graded grid."""
    K = conversion_alias_count(q, N, rbar, betabar, p=p)
    th = graded_theta_grid(q, N, p=p)
    sym = symbols(q, N, K=K, p=p, with_derivative=True, theta=th)
    th, s, m, g1, sp = sym.theta, sym.s, sym.m, sym.g1, sym.s_prime
    dm, _, dg = closed_form_derivatives(th, p)
    x = rbar * m + betabar * g1; xp = rbar * dm + betabar * dg
    f = (s + sym.eps_K) / x
    fp = (sp * x - (s + sym.eps_K) * xp) / x ** 2
    return _grid_extreme(f, fp, th, "max")[0]


@dataclass
class ThreeStage:
    status: str
    stage: int
    eps: float
    eta_reduced: float
    lower_from_reduced: float | None = None
    theta_bar: float | None = None
    eta_banded: float | None = None
    eta_exact: float | None = None
    times: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status.endswith("accept")


def certify_three_stage(fs, forcing, z_h0: np.ndarray, eps: float, *, q: Query, alpha: np.ndarray, V: np.ndarray,
                        rc, ell: np.ndarray, eps_Q: float, cid: str, spec=None, reduced=None, tbar=None,
                        full=None, force_all: bool = False) -> ThreeStage:
    """`reduced`, `tbar`, `full` allow reusing stage results across a sweep of eps (the bounds do not depend on eps).
    force_all=True evaluates every stage (for validation and reporting); the status is still decided by the rule,
    and the times are reported only for the stages the rule actually uses."""
    if force_all:
        stages = stage_values(fs, forcing, z_h0, q=q, alpha=alpha, V=V, rc=rc, ell=ell, eps_Q=eps_Q, cid=cid, spec=spec)
        reduced, tbar, full = stages["reduced"], stages["tbar"], stages["full"]
    times = {}
    dt = forcing.dt
    t0 = time.perf_counter()
    if reduced is None:
        reduced = rc.certify(q, alpha, dt, forcing.bL, ell, eps_Q, cid=cid)
    times["reduced"] = time.perf_counter() - t0
    red = reduced["eta_aug"]
    if red <= eps:
        return ThreeStage("reduced_accept", 1, eps, red, times=times)
    t0 = time.perf_counter()
    cell = rc.cells[cid]
    if tbar is None:
        tbar = theta_bar(q, fs.mesh.N, cell.rbar, cell.betabar)
    lo = np.maximum((np.sqrt(reduced["per_step"]) - reduced["rounding"]) / np.sqrt(tbar) - reduced["allowance"], 0.0)
    lower = float(np.sqrt(reduced["beta0"] ** 2 + dt * np.sum(lo ** 2)))
    times["theta_bar"] = time.perf_counter() - t0
    if lower > eps:
        return ThreeStage("reject_cheap", 1, eps, red, lower, tbar, times=times)
    if full is None:           # lazy (timed path): banded stage now, exact stage only if the banded one is undecided
        full = banded_values(fs, forcing, z_h0, V @ alpha, q, spec)
    times["banded"] = full["t_banded"]
    out = ThreeStage("", 2, eps, red, lower, tbar, eta_banded=full["etaX"], times=times)
    if full["etaX"] <= eps:
        out.status = "banded_accept"
        return out
    if np.sqrt(full["beta0"] ** 2 + full["etaX_res"] ** 2 / full["Theta"]) > eps:
        out.status = "reject_banded"
        return out
    if "etaS" not in full:
        full.update(exact_values(fs, forcing, z_h0, V @ alpha, full["spec"]))
    times["exact"] = full["t_exact"]
    out.stage = 3
    out.eta_exact = full["etaS"]
    out.status = "exact_accept" if full["etaS"] <= eps else "exact_reject"
    return out


def banded_values(fs, forcing, z_h0, Z, q, spec=None) -> dict:
    """Stage 2 only: spectral data (X_I, Theta_mu) and the banded certificate with A_I."""
    t0 = time.perf_counter()
    if spec is None:
        spec = spectral_data(q, fs.mesh.N)
    fb = certify(fs, forcing, Z, z_h0, 0.0, mode="banded", spec=spec)
    return dict(etaX=fb.etaX_aug, etaX_res=fb.etaX, Theta=spec.Theta, beta0=fb.beta0, spec=spec,
                t_banded=time.perf_counter() - t0)


def exact_values(fs, forcing, z_h0, Z, spec) -> dict:
    t0 = time.perf_counter()
    fe = certify(fs, forcing, Z, z_h0, 0.0, mode="exact", spec=spec)
    return dict(etaS=fe.etaS_aug, t_exact=time.perf_counter() - t0)


def full_order_values(fs, forcing, z_h0, Z, q, spec=None) -> dict:
    """Stage 2 and 3 values with the exact operator A_I (no allowance); timed separately (spectral data counted in
    the banded stage, which needs X_I and Theta_mu)."""
    t0 = time.perf_counter()
    if spec is None:
        spec = spectral_data(q, fs.mesh.N)
    t_spec = time.perf_counter() - t0
    fb = certify(fs, forcing, Z, z_h0, 0.0, mode="banded", spec=spec)
    fe = certify(fs, forcing, Z, z_h0, 0.0, mode="exact", spec=spec)
    return dict(etaX=fb.etaX_aug, etaX_res=fb.etaX, etaS=fe.etaS_aug, Theta=spec.Theta, beta0=fb.beta0,
                t_banded=t_spec + fb.time_banded + fb.time_residual, t_exact=fe.time_exact + fe.time_residual)


def stage_values(fs, forcing, z_h0, *, q, alpha, V, rc, ell, eps_Q, cid, spec=None) -> dict:
    """All stage quantities, independent of eps (reuse across the eps sweep)."""
    t0 = time.perf_counter()
    reduced = rc.certify(q, alpha, forcing.dt, forcing.bL, ell, eps_Q, cid=cid)
    t_red = time.perf_counter() - t0
    t0 = time.perf_counter()
    cell = rc.cells[cid]
    tbar = theta_bar(q, fs.mesh.N, cell.rbar, cell.betabar)
    t_tb = time.perf_counter() - t0
    full = full_order_values(fs, forcing, z_h0, V @ alpha, q, spec)
    lo = np.maximum((np.sqrt(reduced["per_step"]) - reduced["rounding"]) / np.sqrt(tbar) - reduced["allowance"], 0.0)
    lower = float(np.sqrt(reduced["beta0"] ** 2 + forcing.dt * np.sum(lo ** 2)))
    return dict(reduced=reduced, tbar=tbar, full=full, lower=lower,
                times=dict(reduced=t_red, theta_bar=t_tb, banded=full["t_banded"], exact=full["t_exact"]))
