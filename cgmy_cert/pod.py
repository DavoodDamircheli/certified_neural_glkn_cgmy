"""Mass-weighted POD and POD--Galerkin on the exact-assembly free system (plan 2.2).

The basis is M_I-orthonormal: V^T M_I V = I.  Two reduced models:

* `pod_galerkin(fs, forcing, V, z0)`: reduced CN march with the full assembled operator (dense
  products V^T A_I V); reference only.
* `AffineROM`: the affine/EIM operator of plan 2.1,
      V^T A_Q V = -(r - omega_S) V^T D_I V + lambda_0 V^T M_I V - C Gamma(-Y) h^{-Y} sum_q ell_q V^T T_q V,
  with Q + 2 precomputed r x r blocks, and the reduced forcing
      V^T f^{n+1/2} = -V^T u_M (b_L^{n+1} - b_L^n)/dt - V^T u_A (b_L^{n+1} + b_L^n)/2,
      V^T u_A = -(r - omega_S) V^T u_D + lambda_0 V^T u_M - C Gamma(-Y) h^{-Y} sum_q ell_q V^T u_q,
  where u_X = E^T Xbar e_bd is the coupling of the free functions to the left boundary functions
  (raw indices -(p-1), ..., -1, all carrying the same prescribed value b_L).  The couplings are
  stored per boundary direction (u_X[j] for raw index -j) and summed online; u_q is built from the
  stored Toeplitz columns of the EIM basis (lags k + j, k = 0..n-1).  Everything online is affine in
  the scalars (ell_q(nu), r - omega_S, lambda_0, C Gamma(-Y) h^{-Y}, b_L(tau_n)) and costs O(Q r^2).

Allowance (Theorem 5.2): the residual of a candidate computed with A_Q differs from the one with
A_I by E^T (Abar_I - Abar_Q) cbar_raw^{n+1/2}, where cbar_raw is the *raw* coefficient vector (free
part and the prescribed boundary values; the forcing difference is the boundary part).  Hence
d_n = eps_Q ||cbar_raw^{n+1/2}||_2 / sqrt(alpha)  (`allowance_vectors`).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky, solve_triangular, lu_factor, lu_solve

from .params import Query
from .symbols import FreeSystem, _dense_banded, _poly_entries, mass_symbol, drift_symbol
from .direct import FreeForcing, lifting


def mass_cholesky(fs: FreeSystem) -> np.ndarray:
    M = _dense_banded(fs.M_entries, fs.n)
    return cholesky(M, lower=True)        # M = L L^T


def pod_basis(snapshots: np.ndarray, fs: FreeSystem, r_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Mass-weighted POD: columns of V (n x r_max) with V^T M V = I; returns (V, singular values)."""
    L = mass_cholesky(fs)
    U, s, _ = np.linalg.svd(L.T @ snapshots, full_matrices=False)
    V = solve_triangular(L.T, U[:, :r_max], lower=False)
    return V, s


def pod_galerkin(fs: FreeSystem, forcing: FreeForcing, V: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """Reduced CN march with V^T A_I V; returns the reconstructed free trajectory V alpha^n
    (n x (N_T+1)), alpha^0 = V^T M_I z0."""
    r = V.shape[1]
    Mr = V.T @ fs.M_matvec(V)                     # = I up to roundoff
    Ar = V.T @ fs.A.matmat(V)
    Fr = V.T @ forcing.F
    dt = forcing.dt
    Bp = Mr + 0.5 * dt * Ar; Bm = Mr - 0.5 * dt * Ar
    lu = lu_factor(Bp)
    N_T = forcing.F.shape[1]
    alpha = np.zeros((r, N_T + 1))
    alpha[:, 0] = V.T @ fs.M_matvec(z0)
    for k in range(N_T):
        alpha[:, k + 1] = lu_solve(lu, Bm @ alpha[:, k] + dt * Fr[:, k])
    return V @ alpha


# ---------------------------------------------------------------- affine reduced model

def boundary_couplings(lags: np.ndarray, n: int, p: int) -> np.ndarray:
    """u[j-1, k] = t_{k+j} (k = 0..n-1, j = 1..p-1): coupling of free function k to the boundary raw
    index -j, from Toeplitz entries `lags` indexed m = -(nl-1)..(nl-1) (nl = n + p - 1).  Works on a
    stack of lag vectors (leading axes)."""
    nl = (lags.shape[-1] + 1) // 2
    lag0 = nl - 1
    return np.stack([lags[..., lag0 + j: lag0 + j + n] for j in range(1, p)], axis=-2)


@dataclass
class ReducedSolution:
    alpha: np.ndarray          # r x (N_T+1)
    ell: np.ndarray
    eps_Q: float
    dt: float
    bL: np.ndarray


class AffineROM:
    """Offline: Q + 2 reduced operator blocks, Q + 2 reduced boundary couplings, for a basis V."""

    def __init__(self, V: np.ndarray, eim, Q: int | None = None):
        from .eim import EIM  # noqa: F401  (type only)
        self.eim = eim
        self.Q = eim.Q if Q is None else Q
        self.V = V
        n, p, h = eim.n, eim.p, eim.h
        if V.shape[0] != n:
            raise ValueError("basis size does not match the EIM mesh")
        nl = n + eim.extra
        self.n, self.p, self.h = n, p, h
        Me = _poly_entries(lambda t: mass_symbol(t, p), p)
        De = _poly_entries(lambda t: 1j / h * drift_symbol(t, p), p)
        self.M_entries = Me
        from .symbols import ToeplitzOp
        D_lags = _entries_to_lags(De, nl); M_lags = _entries_to_lags(Me, nl)
        Dop = ToeplitzOp.from_lags(D_lags, n, eim.extra)
        Mop = ToeplitzOp.from_lags(M_lags, n, eim.extra)
        self.Mr = V.T @ Mop.matmat(V)
        self.Dr = V.T @ Dop.matmat(V)
        self.Tr = np.stack([V.T @ ToeplitzOp.from_lags(eim.lags[q], n, eim.extra).matmat(V) for q in range(self.Q)])
        # boundary couplings, per direction: (p-1, r)
        self.uM = boundary_couplings(M_lags, n, p) @ V          # (p-1, r)
        self.uD = boundary_couplings(D_lags, n, p) @ V
        self.uT = boundary_couplings(eim.lags[:self.Q], n, p) @ V   # (Q, p-1, r)

    # ---- online pieces (all O(Q r^2) or less)
    def scalars(self, q: Query, ell: np.ndarray):
        return -(q.r - q.omega_S), q.lambda0, q.C * q.gammaY * self.h ** (-q.Y)

    def operator(self, q: Query, ell: np.ndarray) -> np.ndarray:
        cd, cm, cg = self.scalars(q, ell)
        return cd * self.Dr + cm * self.Mr - cg * np.tensordot(ell, self.Tr, axes=1)

    def coupling_A(self, q: Query, ell: np.ndarray) -> np.ndarray:
        """V^T u_A summed over the boundary directions (r,)."""
        cd, cm, cg = self.scalars(q, ell)
        per_dir = cd * self.uD + cm * self.uM - cg * np.tensordot(ell, self.uT, axes=1)
        return per_dir.sum(axis=0)

    def forcing(self, q: Query, T: float, N_T: int, ell: np.ndarray):
        from .params import Mesh
        tau = np.arange(N_T + 1) * (T / N_T); dt = T / N_T
        bL = lifting(q, Mesh(N=self.eim.N, p=self.p), tau)
        db = (bL[1:] - bL[:-1]) / dt; sb = 0.5 * (bL[1:] + bL[:-1])
        uA = self.coupling_A(q, ell); uM = self.uM.sum(axis=0)
        return -(uM[:, None] * db[None, :] + uA[:, None] * sb[None, :]), dt, bL

    def solve(self, q: Query, alpha0: np.ndarray, N_T: int, T: float | None = None,
              ell: np.ndarray | None = None, with_eps: bool = True) -> ReducedSolution:
        T = q.T if T is None else T
        eps = np.nan
        if ell is None:
            if with_eps:
                info = self.eim.eps_Q(q, self.Q); ell = info["ell"]; eps = info["eps_Q"]
            else:
                ell = self.eim.ell((q.G, q.M, q.Y), self.Q)
        Ar = self.operator(q, ell)
        Fr, dt, bL = self.forcing(q, T, N_T, ell)
        Bp = self.Mr + 0.5 * dt * Ar; Bm = self.Mr - 0.5 * dt * Ar
        lu = lu_factor(Bp)
        alpha = np.zeros((self.V.shape[1], N_T + 1)); alpha[:, 0] = alpha0
        for k in range(N_T):
            alpha[:, k + 1] = lu_solve(lu, Bm @ alpha[:, k] + dt * Fr[:, k])
        return ReducedSolution(alpha=alpha, ell=ell, eps_Q=eps, dt=dt, bL=bL)

    def project(self, z: np.ndarray) -> np.ndarray:
        """alpha = V^T M_I z (the M-orthogonal projection coefficients)."""
        from .symbols import banded_sym_matvec
        return self.V.T @ banded_sym_matvec(self.M_entries, z)


def _entries_to_lags(entries: dict, nl: int) -> np.ndarray:
    t = np.zeros(2 * nl - 1, dtype=complex)
    for m, v in entries.items():
        t[nl - 1 + m] = v
    return t.real if np.all(t.imag == 0) else t


def allowance_vectors(Z: np.ndarray, bL: np.ndarray, p: int, eps_Q: float, alpha_S: float, alpha_X: float):
    """d_n for the exact and the banded mode: eps_Q ||cbar_raw^{n+1/2}||_2 / sqrt(alpha), where
    cbar_raw stacks the free mean (Z^{n+1}+Z^n)/2 and the p-1 prescribed left values (b_L^{n+1}+b_L^n)/2
    (right prescribed values are 0).  Also returns the free-part-only norms for reporting."""
    zbar = 0.5 * (Z[:, 1:] + Z[:, :-1])
    bbar = 0.5 * (bL[1:] + bL[:-1])
    free2 = np.sum(zbar ** 2, axis=0)
    raw = np.sqrt(free2 + (p - 1) * bbar ** 2)
    return eps_Q * raw / np.sqrt(alpha_S), eps_Q * raw / np.sqrt(alpha_X), raw, np.sqrt(free2)
