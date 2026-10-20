"""Preconditioned exact Riesz solve S_I v = rho and the v8 two-sided bracket (v8 Lemma 4.2).

    l(v) = 2 v^T rho - v^T S v   <=   rho^T S^{-1} rho   <=   u(v) = l(v) + ||rho - S v||_2^2 / alpha_lb.

The Riesz systems are solved by CG preconditioned with a circulant built from the same symbol
samples (Strang / symbol-sampled; Chan--Strang clustering at 1).  Only alpha_lb and the
preconditioner differ from v8; the bracket code path is kept.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .symbols import ToeplitzOp, Symbols


@dataclass
class Circulant:
    """Circulant preconditioner with eigenvalues `lam` (length n); apply inverse by FFT."""
    lam: np.ndarray

    @property
    def n(self) -> int:
        return self.lam.size

    def solve(self, v: np.ndarray) -> np.ndarray:
        out = np.fft.ifft(np.fft.fft(v) / self.lam)
        return out.real if np.isrealobj(v) and np.isrealobj(self.lam) else out

    def matvec(self, v: np.ndarray) -> np.ndarray:
        out = np.fft.ifft(np.fft.fft(v) * self.lam)
        return out.real if np.isrealobj(v) and np.isrealobj(self.lam) else out


def circulant_preconditioner(op: ToeplitzOp | None = None, sym_vals: np.ndarray | None = None,
                             n: int | None = None, kind: str = "symbol") -> Circulant:
    """kind='symbol': eigenvalues f(2 pi k / n) taken from the L-point symbol samples
    (requires L % n == 0; exact for the n-point circulant whose symbol is the sampled f).
    kind='strang': first column c_m = t_m for |m| <= n/2, eigenvalues fft(c)."""
    if kind == "symbol":
        if sym_vals is None or n is None:
            raise ValueError("kind='symbol' needs sym_vals and n")
        L = sym_vals.size
        if L % n != 0:
            raise ValueError(f"symbol grid L={L} must be a multiple of n={n}")
        step = L // n
        k = np.arange(n)
        idx = (k * step + L // 2) % L           # theta_j = -pi + 2 pi j / L  ->  2 pi k / n
        lam = sym_vals[idx]
        if np.max(np.abs(lam.imag)) <= 1e-12 * np.max(np.abs(lam)):
            lam = lam.real
        return Circulant(lam=lam)
    if kind == "strang":
        if op is None:
            raise ValueError("kind='strang' needs the ToeplitzOp")
        n = op.n
        c = np.zeros(n, dtype=complex)
        half = n // 2
        for m in range(0, half + 1):
            c[m] = op.entry(m)
        for m in range(1, n - half):
            c[n - m] = op.entry(-m)
        lam = np.fft.fft(c)
        if op.real:
            lam = lam.real
        return Circulant(lam=lam)
    raise ValueError(kind)


@dataclass
class Bracket:
    l: float
    u: float
    v: np.ndarray
    iters: int
    res_norm: float
    width: float
    converged: bool


def pcg_bracket(S: ToeplitzOp, rho: np.ndarray, alpha_lb: float, precond: Circulant | None = None,
                tol: float = 1e-12, maxit: int = 200, budget: int | None = None, x0=None) -> Bracket:
    """Conjugate gradients on S v = rho (S SPD) with the bracket evaluated at the final iterate.
    Stops when ||rho - S v||_2 <= tol * ||rho||_2, or after `budget` iterations if given."""
    n = rho.size
    v = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    r = rho - (S.matvec(v) if x0 is not None else 0.0)
    r = np.array(r, dtype=float)
    z = precond.solve(r) if precond is not None else r
    p = z.copy()
    rz = float(r @ z)
    nrho = float(np.linalg.norm(rho))
    it = 0
    limit = maxit if budget is None else min(maxit, budget)
    converged = nrho == 0.0
    while it < limit and not converged:
        Sp = S.matvec(p)
        pSp = float(p @ Sp)
        if pSp <= 0:
            break
        a = rz / pSp
        v += a * p
        r -= a * Sp
        it += 1
        if np.linalg.norm(r) <= tol * nrho:
            converged = True
            break
        z = precond.solve(r) if precond is not None else r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    Sv = S.matvec(v)
    res = rho - Sv
    l = float(2 * v @ rho - v @ Sv)
    width = float(res @ res) / alpha_lb
    return Bracket(l=l, u=l + width, v=v, iters=it, res_norm=float(np.linalg.norm(res)), width=width,
                   converged=converged)


def plain_cg_iterations(S: ToeplitzOp, rho: np.ndarray, tol: float = 1e-12, maxit: int = 2000) -> int:
    return pcg_bracket(S, rho, 1.0, precond=None, tol=tol, maxit=maxit).iters
