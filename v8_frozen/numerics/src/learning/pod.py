"""Classical POD/reduced-basis ROM baseline -- directly answers "why not just
a reduced-order model?"

IMPORTANT prerequisite, different from every other model in this package:
POD requires every snapshot to live in the SAME discrete space (same knot
vector: same x_min, h) for dof index k to mean the same physical location
across snapshots. ``cgmy.reference_solver.solve_cgmy``'s DEFAULT domain
selection is mu-dependent (domain_scale/G, domain_scale/M -- see Stage A),
so trajectories solved with the default Mesh are NOT directly comparable as
POD snapshots. Callers MUST solve every trajectory used to build or
evaluate a PODGalerkinROM on one common, explicit
``Mesh(x_min=..., x_max=...)``. This is not an implementation detail to
work around -- it is a real structural requirement of any linear-subspace
method, and part of the "why not POD" story: a fixed reduced basis is even
less flexible than a fixed mesh.

PODGalerkinROM reconstructs the exact initial-condition property the same
structural way every model in this package does:

    c_theta(tau) = c_h^0 + [g(tau) - g(0)],   g(tau) = mean + U_r @ alpha(tau)

so c_theta(0) = c_h^0 exactly regardless of the basis or reduced dynamics.
The reduced dynamics themselves are the SAME Crank-Nicolson recurrence
solve_cn uses, Galerkin-projected onto U_r via OperatorCache.apply_A/apply_M
(the single operator implementation) -- not a new time-stepping scheme.

Known limitation, itself part of the answer to "why not POD": the reduced
system does not enforce the reference solver's Dirichlet boundary
conditions (which the full-order solver imposes by *replacing* certain
matrix rows, not a Galerkin operation a reduced basis can represent
without extra lifting machinery this baseline does not implement). Errors
concentrate near the domain edges as a result -- see
scripts/pod_baseline_report.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters
from learning.normalization import NormalizationStats, denormalize_mu


@dataclass(frozen=True)
class PODBasis:
    """A truncated POD basis and its associated singular values.

    Attributes:
        modes: Basis matrix, shape (n_dof, n_modes).
        singular_values: Singular values associated with each mode.
        mean: Snapshot mean subtracted before decomposition, shape (n_dof,).
    """

    modes: np.ndarray
    singular_values: np.ndarray
    mean: np.ndarray


def compute_pod_basis(snapshots: np.ndarray, n_modes: int) -> PODBasis:
    """Compute a truncated POD basis via SVD of the (mean-centered) snapshot matrix.

    Args:
        snapshots: Snapshot matrix, shape (n_snapshots, n_dof) -- rows are
            individual coefficient vectors (any time level of any
            trajectory), ALL solved on the SAME fixed mesh (see module
            docstring).
        n_modes: Number of POD modes to retain.
    """
    mean = snapshots.mean(axis=0)
    X = (snapshots - mean).T  # (n_dof, n_snapshots), matches X = [c^(1), c^(2), ...]
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return PODBasis(modes=U[:, :n_modes], singular_values=S[:n_modes], mean=mean)


def project_onto_basis(coeffs: np.ndarray, basis: PODBasis) -> np.ndarray:
    """Project a coefficient vector onto the reduced POD coordinates."""
    return basis.modes.T @ (coeffs - basis.mean)


def reconstruct_from_basis(reduced_coords: np.ndarray, basis: PODBasis) -> np.ndarray:
    """Reconstruct a full coefficient vector from reduced POD coordinates."""
    return basis.mean + basis.modes @ reduced_coords


def reconstruction_error(coeffs: np.ndarray, basis: PODBasis) -> float:
    """||coeffs - reconstruct(project(coeffs))|| -- the best possible error ANY
    rank-len(basis) linear method could achieve on this vector (an oracle
    bound, since it uses the true coeffs to compute the projection)."""
    reduced = project_onto_basis(coeffs, basis)
    return float(np.linalg.norm(coeffs - reconstruct_from_basis(reduced, basis)))


class PODGalerkinROM:
    """Reduced-order model: Galerkin-project A_h(mu) and M_h onto U_r, then run
    the same Crank-Nicolson recurrence in r dimensions.

    Needs no training/regression (unlike the neural models) and reuses
    OperatorCache.apply_A/apply_M unchanged -- but its accuracy is capped by
    how well the FIXED linear subspace U_r captures a new mu's solution,
    which degrades outside the span of the training snapshots (see
    scripts/pod_baseline_report.py's extrapolation numbers). M_r is built
    once (M_h is mu- and domain-invariant, verified in Stage E); A_r(mu) is
    rebuilt (cheap: n_modes calls to apply_A, then an (n_modes, n_modes)
    dense projection) for each new mu.

    Args:
        basis: A PODBasis built from snapshots on ops's own mesh.
        ops: OperatorCache for the SAME fixed mesh the basis was built on.
        stats: Normalization stats, needed to recover the physical mu from
            the shared interface's mu_tilde (unlike the neural models,
            this ROM needs physical mu to Galerkin-project the operator,
            not just a feature vector). Optional; only required for
            predict_trajectory (internal cost-timing helpers that call
            _A_r(mu) directly don't need it).
    """

    def __init__(self, basis: PODBasis, ops: OperatorCache, stats: NormalizationStats | None = None) -> None:
        self.basis = basis
        self.ops = ops
        self.stats = stats
        self.n_dof = ops.n_dof
        self.n_modes = basis.modes.shape[1]
        AM = np.array([ops.apply_M(basis.modes[:, j]) for j in range(self.n_modes)]).T
        self._M_r = basis.modes.T @ AM

    def _A_r(self, mu: CGMYParameters) -> np.ndarray:
        AU = np.array([self.ops.apply_A(mu, self.basis.modes[:, j]) for j in range(self.n_modes)]).T
        return self.basis.modes.T @ AU

    def predict_trajectory(
        self,
        mu_tilde: np.ndarray,
        psi: np.ndarray,  # unused: the reduced dynamics only need mu and c0
        c0: np.ndarray,
        time_grid: np.ndarray,
        T: float,
    ) -> np.ndarray:
        """CoefficientModel interface. Recovers physical mu from mu_tilde
        (exact inverse, see learning.normalization) to Galerkin-project A_h(mu)."""
        if self.stats is None:
            raise ValueError("PODGalerkinROM.predict_trajectory requires stats to be set at construction")
        mu = denormalize_mu(mu_tilde, self.stats)
        dt = float(time_grid[1] - time_grid[0]) if len(time_grid) > 1 else float(T)
        A_r = self._A_r(mu)
        F = self._M_r + (dt / 2.0) * A_r
        B = self._M_r - (dt / 2.0) * A_r
        F_lu = sla.lu_factor(F)

        alpha0 = self.basis.modes.T @ (c0 - self.basis.mean)
        alphas = [alpha0]
        alpha = alpha0
        for _ in range(len(time_grid) - 1):
            rhs = B @ alpha
            alpha = sla.lu_solve(F_lu, rhs)
            alphas.append(alpha)
        alphas = np.array(alphas)  # (n_steps+1, n_modes)

        g = self.basis.mean + alphas @ self.basis.modes.T  # (n_steps+1, n_dof)
        return c0 + (g - g[0])


def online_cost_report(basis: PODBasis, ops: OperatorCache, mu: CGMYParameters, n_steps: int) -> dict:
    """Time and storage cost of one PODGalerkinROM prediction, for Stage F's
    "online time, storage" report (see scripts/pod_baseline_report.py)."""
    t0 = time.time()
    rom = PODGalerkinROM(basis, ops, stats=None)  # stats unused by _A_r/timing path directly
    A_r = rom._A_r(mu)
    setup_time = time.time() - t0

    t0 = time.time()
    dt = 1.0 / n_steps
    F = rom._M_r + (dt / 2.0) * A_r
    F_lu = sla.lu_factor(F)
    alpha = np.zeros(basis.modes.shape[1])
    for _ in range(n_steps):
        alpha = sla.lu_solve(F_lu, rom._M_r @ alpha - (dt / 2.0) * (A_r @ alpha))
    step_time = time.time() - t0

    storage_floats = basis.modes.size + basis.mean.size + basis.singular_values.size
    return {
        "n_modes": basis.modes.shape[1],
        "setup_time_s": setup_time,
        "n_steps_time_s": step_time,
        "storage_floats": storage_floats,
        "storage_vs_full_order_ratio": storage_floats / (ops.n_dof * ops.n_dof),
    }
