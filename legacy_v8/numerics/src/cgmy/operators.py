"""Matrix-free CGMY operator application, cached once per mesh.

Provides ``apply_M``, ``apply_M_inv``, ``apply_A``, ``apply_AT``, ``apply_S``
as methods of ``OperatorCache``, built once per mesh and reused for any
number of ``mu`` evaluations. Every array/matrix/factorization here is
constructed by calling the trusted external solver's own functions
(``cgmy_bspline.matrices``, ``cgmy_bspline.toeplitz``,
``cgmy_bspline.parameters``) unchanged -- this module never re-derives the
B-spline mass matrix, the derivative matrix, the connection coefficients, or
the Toeplitz/FFT circulant-embedding construction. It reproduces exactly the
formula used in ``cgmy_bspline.solver_cn.solve_cn``'s mode='toeplitz'
``_A_matvec`` closure, so that the reference solver, residual evaluation,
certification, and correction all apply the identical operator (Hard
Invariant #1 in CLAUDE.md).

Mesh-only quantities (M_h and its LU factorization, D_h, mesh geometry) are
cached once in ``__init__``. The nonlocal Toeplitz operators K_G/K_M depend
on mu through (G*h, M*h, Y), so they cannot be mesh-only cached; they are
built lazily and memoized per distinct (G, M, Y) seen so far.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla
from scipy.special import gamma as sc_gamma

from cgmy import _external  # noqa: F401  (must precede any cgmy_bspline import)

from cgmy_bspline import matrices as cgmy_matrices
from cgmy_bspline.parameters import CGMYParams, lambda0, omega_stock
from cgmy_bspline.toeplitz import ToeplitzOperator, build_fractional_toeplitz

from cgmy.reference_solver import CGMYParameters, Mesh


@dataclass(frozen=True)
class _ToeplitzPair:
    """K_G_minus and K_M_plus Toeplitz operators for one (G, M, Y)."""

    K_G: ToeplitzOperator
    K_M: ToeplitzOperator


class OperatorCache:
    """Caches mesh-dependent quantities once; applies A_h(mu) for any mu.

    Args:
        mesh: Spatial discretization (N, p).
        x_min: Realized left domain endpoint, e.g. from
            ``GalerkinTrajectory.mesh_information.x_min``. Not recomputed
            here: when auto-selected, the domain depends on mu (through
            domain_scale/G, domain_scale/M), so the caller must supply the
            domain actually used by the solve this cache is meant to match.
        x_max: Realized right domain endpoint.
    """

    def __init__(self, mesh: Mesh, x_min: float, x_max: float) -> None:
        self.mesh = mesh
        self.N = mesh.N
        self.p = mesh.p
        self.x_min = x_min
        self.x_max = x_max
        self.h = (x_max - x_min) / mesh.N
        self.n_dof = mesh.N + mesh.p - 1

        self.M_h = cgmy_matrices.mass_matrix(mesh.N, mesh.p, self.h, x_min)
        self.D_h = cgmy_matrices.derivative_matrix(mesh.N, mesh.p, self.h, x_min)
        self._M_h_lu = spla.splu(self.M_h.tocsc())

        self._toeplitz_cache: dict[tuple[float, float, float], _ToeplitzPair] = {}
        self._banded_precond_cache: dict[tuple[float, ...], spla.SuperLU] = {}

    def _toeplitz_pair(self, mu: CGMYParameters) -> _ToeplitzPair:
        key = (mu.G, mu.M, mu.Y)
        pair = self._toeplitz_cache.get(key)
        if pair is None:
            K_G = build_fractional_toeplitz(
                self.n_dof, self.h, mu.G * self.h, self.p, mu.Y, "minus"
            )
            K_M = build_fractional_toeplitz(
                self.n_dof, self.h, mu.M * self.h, self.p, mu.Y, "plus"
            )
            pair = _ToeplitzPair(K_G=K_G, K_M=K_M)
            self._toeplitz_cache[key] = pair
        return pair

    @staticmethod
    def _drift_constants(mu: CGMYParameters) -> tuple[float, float, float]:
        """(omega_S, lambda0, Gamma(-Y)) for mu, via the trusted formulas.

        K is irrelevant to these three quantities (they depend only on
        C, G, M, Y, r), so a placeholder K=1.0 is used to build the
        CGMYParams needed by cgmy_bspline.parameters.omega_stock/lambda0.
        """
        params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=1.0, T=1.0)
        return omega_stock(params), lambda0(params), sc_gamma(-mu.Y)

    def apply_M(self, v: np.ndarray) -> np.ndarray:
        """Apply the B-spline mass matrix M_h."""
        return self.M_h @ np.asarray(v, dtype=float)

    def apply_M_inv(self, v: np.ndarray) -> np.ndarray:
        """Solve M_h x = v via the cached LU factorization of M_h."""
        return self._M_h_lu.solve(np.asarray(v, dtype=float))

    def apply_A(self, mu: CGMYParameters, v: np.ndarray) -> np.ndarray:
        """Apply the semidiscrete CGMY operator A_h(mu), matrix-free.

        A_h = -(r - omega_S) * D_h - C*Gamma(-Y)*(K_M_plus + K_G_minus) + lambda0 * M_h,
        with K_M_plus/K_G_minus applied via the exact (untruncated) Toeplitz
        circulant-embedding matvec -- identical formula and construction to
        ``solve_cn``'s mode='toeplitz' ``_A_matvec`` closure.
        """
        v = np.asarray(v, dtype=float)
        pair = self._toeplitz_pair(mu)
        om_S, lam0, gY = self._drift_constants(mu)
        return (
            -(mu.r - om_S) * (self.D_h @ v)
            - mu.C * gY * pair.K_G.matvec(v)
            - mu.C * gY * pair.K_M.matvec(v)
            + lam0 * (self.M_h @ v)
        )

    def apply_AT(self, mu: CGMYParameters, v: np.ndarray) -> np.ndarray:
        """Apply A_h(mu)^T, matrix-free.

        M_h is symmetric (it is the Gram matrix of the B-spline basis), so
        its term transposes trivially. D_h^T is the sparse transpose of the
        (non-symmetric) derivative-vs-value matrix. Each Toeplitz operator's
        transpose is Toeplitz with its defining first row and first column
        swapped -- a standard fact about Toeplitz matrices, not a new
        derivation -- using the same (first_col, first_row) arrays that
        built the forward operator.
        """
        v = np.asarray(v, dtype=float)
        pair = self._toeplitz_pair(mu)
        om_S, lam0, gY = self._drift_constants(mu)
        K_G_T = ToeplitzOperator(pair.K_G.first_row, pair.K_G.first_col)
        K_M_T = ToeplitzOperator(pair.K_M.first_row, pair.K_M.first_col)
        return (
            -(mu.r - om_S) * (self.D_h.T @ v)
            - mu.C * gY * K_G_T.matvec(v)
            - mu.C * gY * K_M_T.matvec(v)
            + lam0 * (self.M_h.T @ v)
        )

    def apply_S(self, mu: CGMYParameters, v: np.ndarray) -> np.ndarray:
        """Apply S_h(mu) = (A_h(mu) + A_h(mu)^T) / 2, matrix-free."""
        return 0.5 * (self.apply_A(mu, v) + self.apply_AT(mu, v))

    def banded_preconditioner_lu(self, mu: CGMYParameters, bandwidth: int = 64) -> spla.SuperLU:
        """A sparse LU factorization of a BANDED (truncated-bandwidth)
        approximation of S_h(mu), for use as an "operator-aware" PCG
        preconditioner (Task 9, working prompt).

        This is a deliberately APPROXIMATE operator, built by
        ``cgmy_bspline.matrices.operator_matrix`` (trusted, unchanged) --
        the same truncated-bandwidth Toeplitz construction the trusted
        solver's own ``mode='toeplitz_precond'`` already uses to
        precondition its GMRES solve (see ``solver_cn.solve_cn``). Reusing
        it here for the SAME purpose (a cheap, structure-exploiting
        preconditioner, not the exact operator every matvec in this
        project's PCG/residual/correction code still uses via
        ``apply_S``/``apply_A``) does not conflict with CLAUDE.md Hard
        Invariant #1 -- that invariant is about the operator actually being
        SOLVED against, not about what may approximate it for
        preconditioning. Cached per (mesh, mu, bandwidth), mirroring
        ``_toeplitz_pair``'s memoization -- a fresh banded LU factorization
        is real, non-negligible cost (see notes/09_preconditioner.md), not
        something to rebuild on every PCG call within a query.
        """
        key = (mu.C, mu.G, mu.M, mu.Y, mu.r, float(bandwidth))
        lu = self._banded_precond_cache.get(key)
        if lu is None:
            bw = min(bandwidth, self.n_dof)
            params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=1.0, T=1.0)
            mats = cgmy_matrices.operator_matrix(self.N, self.p, self.h, self.x_min, params, bandwidth=bw)
            A_bnd = mats["A_h"]
            S_bnd = 0.5 * (A_bnd + A_bnd.T)
            lu = spla.splu(S_bnd.tocsc())
            self._banded_precond_cache[key] = lu
        return lu

    def diag_S(self, mu: CGMYParameters) -> np.ndarray:
        """diag(S_h(mu)), WITHOUT dense assembly (Task 9, working prompt --
        enables a cheap Jacobi/diagonal PCG preconditioner).

        diag(S_h(mu)) = diag(A_h(mu)) exactly (a matrix and its transpose
        always share the same diagonal, so symmetrizing doesn't change it),
        and every additive piece of A_h(mu) (see apply_A's docstring) has an
        O(1)-or-O(n_dof) accessible diagonal without ever materializing an
        n_dof x n_dof matrix:
          - D_h, M_h are already-stored scipy.sparse matrices -- ``.diagonal()``.
          - K_G_minus/K_M_plus are Toeplitz operators -- a Toeplitz matrix's
            diagonal (lag 0) is the single constant ``first_col[0]``
            (== ``first_row[0]`` by construction, see ToeplitzOperator),
            repeated n_dof times.
        """
        pair = self._toeplitz_pair(mu)
        om_S, lam0, gY = self._drift_constants(mu)
        diag_D = self.D_h.diagonal()
        diag_M = self.M_h.diagonal()
        diag_K_G = np.full(self.n_dof, pair.K_G.first_col[0])
        diag_K_M = np.full(self.n_dof, pair.K_M.first_col[0])
        return (
            -(mu.r - om_S) * diag_D
            - mu.C * gY * diag_K_G
            - mu.C * gY * diag_K_M
            + lam0 * diag_M
        )
