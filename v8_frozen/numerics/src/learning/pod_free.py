"""Mass-weighted free-space POD ROM -- CGMY_V4_REMAINING_WORK.md item 3.

Repairs the diagnosis in ``src/learning/pod.py``'s ``PODGalerkinROM``: full
raw snapshots (boundary rows baked in) projected to a plain-SVD
(Euclidean-orthonormal) basis, then stepped with a HOMOGENEOUS reduced CN
recurrence that has no forcing term at all for the time-dependent Dirichlet
lifting the true dynamics actually have. The task file's own diagnosis:
full-rank snapshot-projection error ~4.74e-15 (the basis spans the
snapshots fine) vs full-rank dynamics error ~9.46 (the reduced EQUATIONS
being stepped are simply wrong, not merely truncated) -- so no amount of
rank is going to fix it.

This module is NOT a second POD implementation layered on top of the old
one for comparison purposes -- ``src/learning/pod.py`` is untouched and
its numbers are retained verbatim as a labeled legacy diagnostic per the
task file's own instruction ("retain the old POD numbers only as an
explicitly limited implementation diagnostic... do not use them to
establish neural superiority over reduced-basis methods"). This is the
replacement, built on ``certification.constrained``'s free-space
primitives (same ``OperatorCache.apply_M``/``apply_A`` -- Hard Invariant #1
intact):

1. Snapshots are the HOMOGENEOUS free trajectory ``z(tau) =
   restrict(fs, c(tau))`` -- the lifting ``b(t)`` is not part of ``z`` by
   construction (unlike the raw snapshots, which bake it into the boundary
   rows), so there is nothing further to "remove."
2. The basis ``V`` (n_free, n_modes) is MASS-ORTHONORMAL,
   ``V^T M_I V = I`` -- built from an ordinary SVD basis ``U`` by a single
   (n_modes, n_modes) Cholesky re-orthonormalization (``_mass_orthonormalize``),
   not a generalized eigenproblem.
3. The reduced system is the free CN recurrence WITH ITS CORRECT FORCING
   (``certification.constrained.free_cn_forcing``), Galerkin-projected:
   substituting ``z ~= V alpha`` into
   ``M_I (z^{n+1}-z^n)/dt + A_I zbar = f_I_bar`` and left-multiplying by
   ``V^T`` gives, using ``V^T M_I V = I``,
   ``(alpha^{n+1}-alpha^n)/dt + A_r alphabar = f_r_bar``, with
   ``A_r = V^T A_I(mu) V`` and ``f_r_bar = V^T f_I_bar`` -- a genuine
   Galerkin projection of the ACTUAL equation, not a homogeneous
   approximation of it.
4. Initial condition: the Galerkin projection of ``z0`` onto ``range(V)``
   in the M_I inner product is ``alpha0`` solving
   ``V^T M_I (V alpha0 - z0) = 0``, i.e. ``alpha0 = (V^T M_I V)^{-1} V^T
   M_I z0 = V^T M_I z0`` (using ``V^T M_I V = I``) -- no ad hoc exact-IC
   offset added on top of the reduced dynamics (the task file's own
   warning against this), just the standard Galerkin IC projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, MeshInfo

from certification.constrained import FreeOperatorCache, FreeSpace, free_cn_forcing, lifting_vector


BASIS_MASS_WEIGHTED = "mass_weighted_pod"
BASIS_EUCLIDEAN_DIAGNOSTIC = "euclidean_svd_subspace_with_mass_orthonormalisation"


@dataclass(frozen=True)
class FreePODBasis:
    """A mass-orthonormal (``modes.T @ M_I @ modes == I``) free-space POD basis.

    Attributes:
        modes: (n_free, n_modes).
        singular_values: Singular values of the snapshot matrix the basis was
            built from (L^T Z for the mass-weighted basis, Z for the Euclidean
            diagnostic), length n_modes.
        basis_type: BASIS_MASS_WEIGHTED (v5 production) or
            BASIS_EUCLIDEAN_DIAGNOSTIC ("Euclidean-SVD subspace with mass
            orthonormalisation": valid, but NOT the M_I-optimal subspace).
    """

    modes: np.ndarray
    singular_values: np.ndarray
    basis_type: str = BASIS_EUCLIDEAN_DIAGNOSTIC


def _mass_orthonormalize(U: np.ndarray, free_ops: FreeOperatorCache) -> np.ndarray:
    """Given a Euclidean-orthonormal U (n_free, n_modes), return V = U L^{-T}
    where G := U^T M_I U = L L^T (Cholesky) -- so V^T M_I V =
    L^{-1} (U^T M_I U) L^{-T} = L^{-1} (L L^T) L^{-T} = I exactly."""
    n_modes = U.shape[1]
    MU = np.empty((U.shape[0], n_modes))
    for j in range(n_modes):
        MU[:, j] = free_ops.apply_M(U[:, j])
    G = U.T @ MU
    L = sla.cholesky(G, lower=True)
    X = sla.solve_triangular(L, U.T, lower=True)  # X = L^{-1} U^T, shape (n_modes, n_free)
    return X.T  # V = U L^{-T} = X^T


def compute_free_pod_basis(
    snapshots_free: np.ndarray, free_ops: FreeOperatorCache, n_modes: int
) -> FreePODBasis:
    """DIAGNOSTIC ONLY (v5): "Euclidean-SVD subspace with mass
    orthonormalisation" -- the Euclidean SVD subspace of the snapshots,
    re-orthonormalised in M_I. Valid, but not the mass-weighted optimal
    subspace; v5 production uses ``compute_mass_weighted_pod_basis``.
    (Pre-v5 this docstring called it "mass-weighted POD"; the name of the
    function is kept so v4 scripts/tests still run.)

    Args:
        snapshots_free: (n_snapshots, n_free) -- rows are
            ``restrict(fs, c(tau))`` at any time level of any trajectory,
            all solved on the SAME fixed mesh. No mean subtraction -- a
            pure linear subspace through the origin (so the Galerkin IC
            projection formula above has no extra offset to derive).
        n_modes: Number of modes to retain (<= n_free).
    """
    Z = snapshots_free.T  # (n_free, n_snapshots)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    U = U[:, :n_modes]
    V = _mass_orthonormalize(U, free_ops)
    return FreePODBasis(modes=V, singular_values=S[:n_modes], basis_type=BASIS_EUCLIDEAN_DIAGNOSTIC)


def oracle_mass_projection_error(z: np.ndarray, basis: FreePODBasis, free_ops: FreeOperatorCache) -> float:
    """||z - V (V^T M_I z)||_{M_I} -- the best possible error ANY rank-n_modes
    linear method in the M_I inner product could achieve on this one
    snapshot (uses the true z, an oracle bound)."""
    V = basis.modes
    alpha = V.T @ free_ops.apply_M(z)
    resid = z - V @ alpha
    return float(np.sqrt(max(resid @ free_ops.apply_M(resid), 0.0)))


class FreePODGalerkinROM:
    """Reduced-order model in the free (constrained) coordinate space: steps
    the Galerkin-projected free CN recurrence (module docstring) on a
    mass-orthonormal basis ``V``.
    """

    def __init__(self, basis: FreePODBasis) -> None:
        self.basis = basis
        self.n_modes = basis.modes.shape[1]

    def project_initial(self, z0: np.ndarray, free_ops: FreeOperatorCache) -> np.ndarray:
        """alpha0 = V^T M_I z0 (module docstring's Galerkin IC projection)."""
        return self.basis.modes.T @ free_ops.apply_M(z0)

    def _A_r(self, mu: CGMYParameters, free_ops: FreeOperatorCache) -> np.ndarray:
        V = self.basis.modes
        AV = np.empty_like(V)
        for j in range(self.n_modes):
            AV[:, j] = free_ops.apply_A(mu, V[:, j])
        return V.T @ AV

    def solve_reduced_trajectory(
        self,
        mu: CGMYParameters,
        payoff: EuropeanPutPayoff,
        mesh_info: MeshInfo,
        time_grid: np.ndarray,
        fs: FreeSpace,
        ops,  # cgmy.operators.OperatorCache -- for the raw f_bar_raw/free_cn_forcing call
        free_ops: FreeOperatorCache,
        z0: np.ndarray,
    ) -> np.ndarray:
        """Time-march the reduced system from z0 (full free-space IC,
        projected via ``project_initial``). Returns the RECONSTRUCTED free
        trajectory ``V @ alpha``, shape (len(time_grid), n_free) -- directly
        comparable to a true free trajectory ``restrict(fs, raw c)``.
        """
        from cgmy_bspline import boundary as cgmy_boundary
        from cgmy_bspline.parameters import CGMYParams

        V = self.basis.modes
        n_modes = self.n_modes
        dt = float(time_grid[1] - time_grid[0])

        A_r = self._A_r(mu, free_ops)
        F = np.eye(n_modes) / dt + 0.5 * A_r
        B = np.eye(n_modes) / dt - 0.5 * A_r
        F_lu = sla.lu_factor(F)

        params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=payoff.strike, T=mu.T)

        alpha = self.project_initial(z0, free_ops)
        alphas = [alpha]
        tau = float(time_grid[0])
        for n in range(len(time_grid) - 1):
            tau_new = float(time_grid[n + 1])
            f_old_raw = cgmy_boundary.boundary_forcing_vector(
                mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, tau, None
            )
            f_new_raw = cgmy_boundary.boundary_forcing_vector(
                mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, tau_new, None
            )
            f_bar_raw = 0.5 * (f_old_raw + f_new_raw)
            b_old = lifting_vector(mu, payoff, mesh_info, fs, tau)
            b_new = lifting_vector(mu, payoff, mesh_info, fs, tau_new)
            f_I_bar = free_cn_forcing(fs, ops, mu, dt, f_bar_raw, b_old, b_new)
            f_r_bar = V.T @ f_I_bar

            rhs = B @ alpha + f_r_bar
            alpha = sla.lu_solve(F_lu, rhs)
            alphas.append(alpha)
            tau = tau_new

        alphas_arr = np.array(alphas)  # (n_steps+1, n_modes)
        return alphas_arr @ V.T  # (n_steps+1, n_free)


# ---------------------------------------------------------------------------
# v5 (roadmap §3, M3): genuinely mass-weighted POD, basis artifact, strict
# loader, and a predictor for pipeline.free_workflow.
#
# Basis. Snapshots are free homogeneous coefficient vectors stored as columns
# of Z (n_free, n_snap). With the current boundary-only lifting, b(t) is zero on
# free rows, so restrict(c) IS the homogeneous unknown -- nothing to subtract.
# With M_I = L L^T (Cholesky of the principal free mass matrix),
#     L^T Z = U Sigma W^T,   V_r = L^{-T} U_r   (triangular solve, no inverse),
# so V_r^T M_I V_r = U_r^T L^{-1} L L^T L^{-T} U_r = I, and range(V_r) minimises
# sum_j ||z_j - P z_j||_{M_I}^2 over rank-r subspaces (Eckart-Young in the M_I
# inner product). Every stored time level of every training trajectory has
# weight 1 (no time/query weights).
#
# Centering. UNCENTRED linear subspace. A mean-shifted (affine) basis would need
# the affine reduced equation z = zbar + V a with forcing term -A_I zbar and an
# initial projection of z^0 - zbar; that is not implemented, so no mean is used.
#
# Reduced dynamics (roadmap §3.2): A_r = V^T A_I V, f_r = V^T f_I, a^0 = V^T M_I z^0,
#     (I/dt + A_r/2) a^{n+1} = (I/dt - A_r/2) a^n + f_r^{n+1/2},
# with f_I the problem's frozen forcing and z^0 = z_h^0. The reconstruction
# V a^n is lifted with the problem's lifting and certified by the same free
# pipeline as the neural models; the finite-rank initial error
# beta_0 = ||V a^0 - z_h^0||_{M_I} is part of that certificate.
# ---------------------------------------------------------------------------

import json  # noqa: E402

POD_ARTIFACT_FORMAT = "cgmy_v5_free_pod_basis_v1"


def mass_cholesky_factor(free_ops: FreeOperatorCache) -> np.ndarray:
    """Lower Cholesky factor L of the principal free mass matrix M_I = L L^T,
    assembled from the exact stored entries M_h[I, I]."""
    idx = free_ops.fs.free_idx
    M_I = free_ops.ops.M_h[idx, :][:, idx].toarray()
    return sla.cholesky(M_I, lower=True)


def compute_mass_weighted_pod_basis(
    snapshots_free: np.ndarray, free_ops: FreeOperatorCache, n_modes: int
) -> FreePODBasis:
    """Mass-weighted POD (see block comment above). ``snapshots_free`` is
    (n_snapshots, n_free) -- rows are snapshots, as in compute_free_pod_basis."""
    Z = np.asarray(snapshots_free, dtype=float).T
    if Z.shape[0] != free_ops.n_dof:
        raise ValueError(f"snapshots have {Z.shape[0]} rows, expected n_free={free_ops.n_dof}")
    if not 1 <= n_modes <= min(Z.shape):
        raise ValueError(f"n_modes={n_modes} outside [1, {min(Z.shape)}]")
    L = mass_cholesky_factor(free_ops)
    U, S, _ = np.linalg.svd(L.T @ Z, full_matrices=False)
    V = sla.solve_triangular(L.T, U[:, :n_modes], lower=False)
    return FreePODBasis(modes=V, singular_values=S[:n_modes], basis_type=BASIS_MASS_WEIGHTED)


def truncate_basis(basis: FreePODBasis, rank: int) -> FreePODBasis:
    """First ``rank`` modes (POD subspaces are nested)."""
    if not 1 <= rank <= basis.modes.shape[1]:
        raise ValueError(f"rank {rank} outside [1, {basis.modes.shape[1]}]")
    return FreePODBasis(modes=basis.modes[:, :rank].copy(), singular_values=basis.singular_values[:rank].copy(),
                        basis_type=basis.basis_type)


class LegacyPODCheckpointError(ValueError):
    """Raised when a non-v5 (legacy or diagnostic) POD artifact is offered to
    the v5 POD loader."""


def save_pod_artifact(path, basis: FreePODBasis, metadata: dict) -> None:
    """Write an .npz basis artifact. ``metadata`` must hold at least
    mass_matrix_identity, training and discretization blocks; basis type,
    rank and singular values are added here."""
    from pathlib import Path

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; basis artifacts are never overwritten")
    meta = dict(metadata)
    meta.update({
        "format": POD_ARTIFACT_FORMAT, "basis_type": basis.basis_type,
        "rank": int(basis.modes.shape[1]), "n_free": int(basis.modes.shape[0]),
        "singular_values": [float(x) for x in basis.singular_values],
    })
    for key in ("mass_matrix_identity", "training", "discretization", "method_id"):
        if key not in meta:
            raise ValueError(f"basis artifact metadata missing {key!r}")
    np.savez(path, modes=basis.modes, singular_values=basis.singular_values,
             metadata_json=np.array(json.dumps(meta, sort_keys=True)))


def mass_matrix_identity(free_ops: FreeOperatorCache) -> dict:
    from common.run_manifest import sha256_array

    idx = free_ops.fs.free_idx
    M_I = free_ops.ops.M_h[idx, :][:, idx].toarray()
    return {
        "M_I_dense_sha256": sha256_array(M_I), "free_idx_sha256": sha256_array(idx.astype(np.int64)),
        "N": int(free_ops.ops.N), "p": int(free_ops.ops.p),
        "x_min": float(free_ops.ops.x_min), "x_max": float(free_ops.ops.x_max),
    }


def load_pod_artifact(
    path, free_ops: FreeOperatorCache, *, rank: int, expected_basis_type: str = BASIS_MASS_WEIGHTED
) -> tuple[FreePODBasis, dict]:
    """Strict v5 loader: refuses the legacy ``checkpoints/pod_final.npz`` (and any
    artifact without v5 metadata), a different basis type, or a basis built
    for a different free mass matrix."""
    data = np.load(path, allow_pickle=False)
    if "metadata_json" not in data.files:
        raise LegacyPODCheckpointError(
            f"{path}: no v5 metadata -- this is a legacy POD checkpoint (raw n_dof modes, no "
            f"forcing in the reduced equation); it cannot be used by constrained_free_cn_v1"
        )
    meta = json.loads(str(data["metadata_json"]))
    if meta.get("format") != POD_ARTIFACT_FORMAT:
        raise LegacyPODCheckpointError(f"{path}: unknown artifact format {meta.get('format')!r}")
    if meta["basis_type"] != expected_basis_type:
        raise LegacyPODCheckpointError(
            f"{path}: basis_type {meta['basis_type']!r} != expected {expected_basis_type!r}"
        )
    got_id = mass_matrix_identity(free_ops)
    if meta["mass_matrix_identity"] != got_id:
        raise LegacyPODCheckpointError(f"{path}: mass-matrix identity mismatch {meta['mass_matrix_identity']} vs {got_id}")
    basis = FreePODBasis(modes=np.array(data["modes"]), singular_values=np.array(data["singular_values"]),
                         basis_type=meta["basis_type"])
    return truncate_basis(basis, rank), meta


def reduced_cn_trajectory(basis: FreePODBasis, problem) -> tuple[np.ndarray, np.ndarray]:
    """§3.2 reduced CN on ``problem`` (a certification.free_problem.FreeProblem):
    returns (alphas (n_T+1, r), reconstructed free trajectory (n_T+1, n_free))."""
    V = basis.modes
    if V.shape[0] != problem.fs.n_free:
        raise ValueError(f"basis n_free={V.shape[0]} != problem n_free={problem.fs.n_free}")
    r = V.shape[1]
    AV = np.column_stack([problem.free_ops.apply_A(problem.mu, V[:, j]) for j in range(r)])
    A_r = V.T @ AV
    F_lu = sla.lu_factor(np.eye(r) / problem.dt + 0.5 * A_r)
    B = np.eye(r) / problem.dt - 0.5 * A_r
    f_r = problem.forcing.f_I @ V  # row n = (V^T f_I^{n+1/2})^T
    a = V.T @ problem.free_ops.apply_M(problem.z0_target)
    alphas = [a]
    for n in range(problem.n_steps):
        a = sla.lu_solve(F_lu, B @ a + f_r[n])
        alphas.append(a)
    alphas = np.array(alphas)
    return alphas, alphas @ V.T


def free_pod_predictor(basis: FreePODBasis):
    """Predictor for pipeline.free_workflow.run_free_workflow: raw coefficients
    P(V a^n) + b^n with the problem's own lifting."""

    def predict(problem) -> np.ndarray:
        _, z = reduced_cn_trajectory(basis, problem)
        return problem.lift(z)

    predict.basis_type = basis.basis_type  # type: ignore[attr-defined]
    predict.rank = basis.modes.shape[1]  # type: ignore[attr-defined]
    return predict
