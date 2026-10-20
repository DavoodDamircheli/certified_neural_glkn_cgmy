"""Certificate driver (plan 1.4).

Given the assembled free system (M_I, A_I, f_I, b^n), a candidate free trajectory z_theta^n,
the discrete initial target z_h^0 and a tolerance eps, compute

    beta_0      = || z_theta^0 - z_h^0 ||_{M_I}
    rho^{n+1/2} = M_I (z^{n+1}-z^n)/dt + A_I (z^{n+1}+z^n)/2 - f_I^{n+1/2}        (v8 eq. (4.3))
    eta^X       : (dt sum_n rho^T X_I^{-1} rho)^{1/2}        banded Cholesky, Cor. 4.2 of the note
    eta^S       : (dt sum_n u(v_n))^{1/2}                     PCG bracket with alpha_lb, v8 Lemma 4.2
    eta_aug     = (beta_0^2 + eta^2)^{1/2}                    v8 Theorem 4.1

Modes: 'banded' (eta^X only), 'exact' (eta^S only), 'two-stage' (Prop. 4.3 of the note),
'reduced' (reserved for Phase 2: eta^{X-bar} from the offline tensors).

Allowance (Theorem 5.2 of the note): if the candidate's residual was evaluated with an affine
surrogate operator A_Q, pass `allowance[n] = eps_Q ||cbar^{n+1/2}||_2 / sqrt(alpha)` and the
per-step bound becomes (sqrt(u_n) + allowance[n])^2  (v8 Remark 4.4 with d_n explicit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

from .symbols import FreeSystem, banded_from_toeplitz
from .spectral import SpectralData, spectral_data
from .riesz import pcg_bracket, circulant_preconditioner, Circulant
from .direct import FreeForcing, mass_norm

STATUS = ("direct_accept", "reject_cheap", "refined_accept", "refined_reject", "banded_accept",
          "banded_reject", "exact_accept", "exact_reject")


@dataclass
class Certificate:
    status: str
    mode: str
    eps: float
    beta0: float
    etaX: float | None = None          # residual part only (no beta0)
    etaX_aug: float | None = None
    etaS_upper: float | None = None    # residual part, upper bracket
    etaS_lower: float | None = None    # residual part, lower bracket
    etaS_aug: float | None = None      # (beta0^2 + etaS_upper^2)^(1/2)
    sqrt_Theta: float | None = None
    alpha_lb: float | None = None
    eps_K: float | None = None
    pcg_iters: list = field(default_factory=list)
    bracket_rel_width: float | None = None
    residual_norms2: np.ndarray | None = None   # ||rho_n||_2
    per_step_X: np.ndarray | None = None
    per_step_u: np.ndarray | None = None
    per_step_l: np.ndarray | None = None
    time_residual: float = 0.0
    time_banded: float = 0.0
    time_exact: float = 0.0
    time_spectral: float = 0.0

    @property
    def bound(self) -> float:
        """The certified bound returned to the user (the tightest valid one computed)."""
        cands = [b for b in (self.etaX_aug, self.etaS_aug) if b is not None]
        return min(cands) if cands else np.inf

    @property
    def accepted(self) -> bool:
        return self.status.endswith("accept")


def cn_residuals(fs: FreeSystem, Z: np.ndarray, forcing: FreeForcing) -> np.ndarray:
    """rho^{n+1/2} for all n (columns), FFT matvecs for A_I and banded for M_I."""
    dt = forcing.dt
    dZ = (Z[:, 1:] - Z[:, :-1]) / dt
    Zbar = 0.5 * (Z[:, 1:] + Z[:, :-1])
    return fs.M_matvec(dZ) + fs.A.matmat(Zbar) - forcing.F


def banded_dual_norms(entries: dict, n: int, bw: int, R: np.ndarray) -> tuple[np.ndarray, object]:
    """rho_n^T X^{-1} rho_n for every column of R, with one banded Cholesky of X."""
    ab = banded_from_toeplitz(entries, n, bw)
    ab_upper = ab[: bw + 1, :]         # LAPACK upper storage for cholesky_banded
    c = cholesky_banded(ab_upper, lower=False)
    Y = cho_solve_banded((c, False), R)
    return np.sum(R * Y, axis=0), c


def certify(fs: FreeSystem, forcing: FreeForcing, Z: np.ndarray, z_h0: np.ndarray, eps: float,
            mode: str = "two-stage", spec: SpectralData | None = None, allowance: np.ndarray | None = None,
            pcg_tol: float = 1e-12, pcg_maxit: int = 200, precond: Circulant | None = None,
            K_spec: int | None = None, allowance_X: np.ndarray | None = None) -> Certificate:
    """Run the certificate in the requested mode.  `spec` (alpha_lb, X_I, Theta) is computed
    from the symbol if not supplied.  `allowance` is the per-step term d_n for the exact (S_I) mode,
    `allowance_X` the one for the banded (X_I) mode (defaults to `allowance`); for an affine
    surrogate operator d_n = eps_Q ||cbar_n||_2 / sqrt(alpha) with alpha = alpha_lb resp. x_min."""
    n = fs.n
    dt = forcing.dt
    N_T = Z.shape[1] - 1
    t0 = time.perf_counter()
    if spec is None:
        spec = spectral_data(fs.q, fs.mesh.N, K=fs.sym.K if K_spec is None else K_spec)
    t_spec = time.perf_counter() - t0

    t0 = time.perf_counter()
    beta0 = mass_norm(fs, Z[:, 0] - z_h0)
    R = cn_residuals(fs, Z, forcing)
    rn2 = np.linalg.norm(R, axis=0)
    t_res = time.perf_counter() - t0
    cert = Certificate(status="", mode=mode, eps=eps, beta0=beta0, sqrt_Theta=float(np.sqrt(spec.Theta)),
                       alpha_lb=spec.alpha_lb, eps_K=spec.eps_K, residual_norms2=rn2,
                       time_residual=t_res, time_spectral=t_spec)
    if spec.alpha_lb <= 0:
        raise ValueError(f"alpha_lb = {spec.alpha_lb} <= 0: uncertifiable (should not happen on the box)")
    bw = fs.mesh.p - 1
    allow = np.zeros(N_T) if allowance is None else np.asarray(allowance, dtype=float)
    allow_X = allow if allowance_X is None else np.asarray(allowance_X, dtype=float)

    def aug(sum_sq):
        return float(np.sqrt(beta0 ** 2 + sum_sq))

    # ---------------- cheap bound
    if mode in ("banded", "two-stage"):
        t0 = time.perf_counter()
        qf, _ = banded_dual_norms(spec.X_entries, n, bw, R)
        qf = np.maximum(qf, 0.0)
        cert.per_step_X = qf
        resX2 = dt * np.sum((np.sqrt(qf) + allow_X) ** 2)
        cert.etaX = float(np.sqrt(resX2))
        cert.etaX_aug = aug(resX2)
        cert.time_banded = time.perf_counter() - t0
        if mode == "banded":
            cert.status = "banded_accept" if cert.etaX_aug <= eps else "banded_reject"
            return cert
        if cert.etaX_aug <= eps:
            cert.status = "direct_accept"
            return cert
        # the exact residual bound satisfies etaS^2 >= etaX^2 / Theta (same beta0, allowance >= 0):
        lower_possible = aug(resX2 / spec.Theta)
        if lower_possible > eps:
            cert.status = "reject_cheap"
            return cert

    # ---------------- exact bound (bracket)
    t0 = time.perf_counter()
    if precond is None:
        precond = circulant_preconditioner(sym_vals=fs.sym.s.astype(complex), n=n, kind="symbol")
    us = np.zeros(N_T); ls = np.zeros(N_T); its = []
    for k in range(N_T):
        br = pcg_bracket(fs.S, R[:, k], spec.alpha_lb, precond=precond, tol=pcg_tol, maxit=pcg_maxit)
        us[k] = br.u; ls[k] = max(br.l, 0.0); its.append(br.iters)
    cert.per_step_u, cert.per_step_l, cert.pcg_iters = us, ls, its
    resS2_u = dt * np.sum((np.sqrt(np.maximum(us, 0.0)) + allow) ** 2)
    resS2_l = dt * np.sum(ls)
    cert.etaS_upper = float(np.sqrt(resS2_u)); cert.etaS_lower = float(np.sqrt(resS2_l))
    cert.etaS_aug = aug(resS2_u)
    tot_u = np.sum(us); cert.bracket_rel_width = float((np.sum(us) - np.sum(ls)) / tot_u) if tot_u > 0 else 0.0
    cert.time_exact = time.perf_counter() - t0
    if mode == "exact":
        cert.status = "exact_accept" if cert.etaS_aug <= eps else "exact_reject"
    else:
        cert.status = "refined_accept" if cert.etaS_aug <= eps else "refined_reject"
    return cert
