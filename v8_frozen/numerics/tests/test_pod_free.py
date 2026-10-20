"""Regression test for the repaired free-space POD ROM --
CGMY_V4_REMAINING_WORK.md item 3's decisive test: full-free-rank time
stepping must reproduce the constrained CN trajectory to near machine
precision (the legacy src/learning/pod.py baseline failed this: ~9.46
dynamics error at full rank despite ~4.74e-15 snapshot-projection error --
see scripts/pod_free_baseline.py and reports/v4_overnight_log.md for the
full account and the real 800-trajectory numbers; this test uses a small,
fast, synthetic snapshot set, not the production dataset)."""

from __future__ import annotations

import numpy as np

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.constrained import FreeOperatorCache, free_space_for_mesh, solve_free_cn_trajectory_direct
from learning.pod_free import FreePODGalerkinROM, compute_free_pod_basis, oracle_mass_projection_error

X_MIN, X_MAX = -3.394829814011908, 12.605170185988092
PAYOFF = EuropeanPutPayoff(strike=100.0)
MUS = [
    CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.5),
    CGMYParameters(C=0.6, G=2.0, M=6.0, Y=1.6, r=0.03, T=0.3),
    CGMYParameters(C=0.15, G=4.0, M=3.0, Y=1.2, r=0.08, T=0.7),
]


def _solve(mesh, mu):
    # n_steps=25 -> 26 rows/trajectory x 3 trajectories = 78 snapshots,
    # comfortably >= n_free (62) so "full rank" below is actually n_free,
    # not capped by too few snapshot rows.
    return solve_cgmy(mu, PAYOFF, mesh, TimeGrid(n_steps=25), mode="toeplitz", store_forcing=True)


def test_free_pod_full_rank_recovers_exact_cn_system_to_machine_precision():
    mesh = Mesh(N=64, p=3, x_min=X_MIN, x_max=X_MAX)
    ops = OperatorCache(mesh, X_MIN, X_MAX)
    trajs = [_solve(mesh, mu) for mu in MUS]
    mesh_info = trajs[0].mesh_information
    fs = free_space_for_mesh(mesh_info)
    free_ops = FreeOperatorCache(ops, fs)

    snapshots_free = np.concatenate([t.coefficients[:, fs.free_idx] for t in trajs], axis=0)
    full_rank = fs.n_free  # 62, <= n_snapshots here (48 rows)

    basis = compute_free_pod_basis(snapshots_free, free_ops, full_rank)

    # Mass-orthonormality actually achieved.
    VtMV = basis.modes.T @ np.array([free_ops.apply_M(basis.modes[:, j]) for j in range(full_rank)]).T
    assert np.max(np.abs(VtMV - np.eye(full_rank))) < 1e-10

    # Projection floor at full rank is ~machine precision (basis spans everything).
    proj_err = oracle_mass_projection_error(snapshots_free[0], basis, free_ops)
    assert proj_err < 1e-10

    mu, traj = MUS[0], trajs[0]
    z0 = traj.coefficients[0, fs.free_idx]

    z_direct = solve_free_cn_trajectory_direct(mu, PAYOFF, mesh_info, traj.time_grid, ops, fs, free_ops, z0)

    rom = FreePODGalerkinROM(basis)
    z_rom = rom.solve_reduced_trajectory(mu, PAYOFF, mesh_info, traj.time_grid, fs, ops, free_ops, z0)

    max_err = np.max(np.abs(z_rom - z_direct))
    assert max_err < 1e-9, (
        f"full-rank free POD ROM should reproduce the direct free CN solve to near "
        f"machine precision, got max abs error {max_err:.3e}"
    )


def test_free_pod_rank_sweep_error_decreases_with_rank():
    """Sanity check for scripts/pod_free_baseline.py's rank sweep: a
    LOWER-rank basis should never do better than a higher-rank one on
    (any monotone function of) the SAME snapshot set (nested subspaces,
    since compute_free_pod_basis truncates the same SVD)."""
    mesh = Mesh(N=64, p=3, x_min=X_MIN, x_max=X_MAX)
    ops = OperatorCache(mesh, X_MIN, X_MAX)
    trajs = [_solve(mesh, mu) for mu in MUS]
    fs = free_space_for_mesh(trajs[0].mesh_information)
    free_ops = FreeOperatorCache(ops, fs)
    snapshots_free = np.concatenate([t.coefficients[:, fs.free_idx] for t in trajs], axis=0)

    errs = []
    for rank in (4, 8, 16, 32):
        basis = compute_free_pod_basis(snapshots_free, free_ops, rank)
        errs.append(oracle_mass_projection_error(snapshots_free[0], basis, free_ops))

    assert all(errs[i] >= errs[i + 1] - 1e-12 for i in range(len(errs) - 1)), errs
