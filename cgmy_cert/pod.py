"""Mass-weighted POD and POD--Galerkin on the exact-assembly free system (plan 2.2, first part).

The basis is M_I-orthonormal: V^T M_I V = I.  The reduced CN march uses the full assembled
operator A_I (dense products V^T A_I V); the affine/EIM operator of plan 2.1 replaces it later.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular, lu_factor, lu_solve

from .symbols import FreeSystem, _dense_banded
from .direct import FreeForcing


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
