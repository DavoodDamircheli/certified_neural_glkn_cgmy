"""Write a trajectory to HDF5, read it back, and confirm it still produces a
near-zero residual through the Stage B code. This is the check that catches
storage/ordering bugs (e.g. a transposed coefficient array, a shuffled
bc_dofs list, a wrong dt) that would otherwise look exactly like neural
approximation error much later in the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid
from certification.residuals import compute_all_residuals, residual_norm
from data.dataset import generate_one_sample, read_trajectory, write_manifest, write_sample

MU = CGMYParameters(C=1.0, G=5.0, M=5.0, Y=0.5, r=0.05, T=0.5)
PAYOFF = EuropeanPutPayoff(strike=100.0)


def test_roundtrip_preserves_near_zero_residual(tmp_path: Path):
    mesh = Mesh(N=32, p=3)
    tg = TimeGrid(n_steps=20)

    result = generate_one_sample(MU, PAYOFF, mesh, tg)
    assert result.ok

    path = tmp_path / "roundtrip.h5"
    write_sample(path, 0, split="train_interior", result=result)
    write_manifest(path, {"train_interior": 1}, n_failures=0, config_snapshot="test", seed=0)

    traj_original = result.traj
    traj_loaded = read_trajectory(path, 0)

    # Storage must be exact (not just "close"): these are the same floats.
    np.testing.assert_array_equal(traj_loaded.coefficients, traj_original.coefficients)
    np.testing.assert_array_equal(traj_loaded.initial_coefficients, traj_original.initial_coefficients)
    np.testing.assert_array_equal(traj_loaded.time_grid, traj_original.time_grid)
    assert traj_loaded.solver_metadata.dt == traj_original.solver_metadata.dt
    assert traj_loaded.solver_metadata.bc_dofs == traj_original.solver_metadata.bc_dofs
    assert traj_loaded.solver_metadata.scheme_per_step == traj_original.solver_metadata.scheme_per_step
    assert traj_loaded.parameter_vector == traj_original.parameter_vector

    ops = OperatorCache(
        Mesh(N=traj_loaded.mesh_information.N, p=traj_loaded.mesh_information.p),
        traj_loaded.mesh_information.x_min, traj_loaded.mesh_information.x_max,
    )

    residuals_original = compute_all_residuals(traj_original, MU, ops)
    residuals_loaded = compute_all_residuals(traj_loaded, MU, ops)

    max_norm_original = max(
        residual_norm(residuals_original[n], ops, traj_original.solver_metadata.bc_dofs)
        for n in range(traj_original.solver_metadata.n_steps)
    )
    max_norm_loaded = max(
        residual_norm(residuals_loaded[n], ops, traj_loaded.solver_metadata.bc_dofs)
        for n in range(traj_loaded.solver_metadata.n_steps)
    )

    print(f"\nmax residual (in-memory trajectory)   = {max_norm_original:.6e}")
    print(f"max residual (round-tripped trajectory) = {max_norm_loaded:.6e}")

    assert max_norm_loaded < 1e-6, (
        f"round-tripped trajectory's residual ({max_norm_loaded:.3e}) is not at "
        f"solver tolerance -- this indicates a storage/ordering bug, not a "
        f"solver issue (the in-memory trajectory's residual is {max_norm_original:.3e})"
    )
    assert abs(max_norm_loaded - max_norm_original) < 1e-9, (
        "round-tripped residual must match the in-memory residual almost exactly"
    )


def test_roundtrip_preserves_forcing_none_vs_stored(tmp_path: Path):
    """forcing_vectors=None (not stored) must round-trip as None, not as a
    fabricated zero array -- storage must not silently paper over the
    'missing forcing must raise' invariant from Stage B."""
    mesh = Mesh(N=32, p=3)
    tg = TimeGrid(n_steps=10)
    result = generate_one_sample(MU, PAYOFF, mesh, tg)
    assert result.ok
    assert result.traj.forcing_vectors is not None  # sanity: normally populated

    import dataclasses
    result_no_forcing = dataclasses.replace(result, traj=dataclasses.replace(result.traj, forcing_vectors=None))

    path = tmp_path / "no_forcing.h5"
    write_sample(path, 0, split="train_interior", result=result_no_forcing)
    traj_loaded = read_trajectory(path, 0)

    assert traj_loaded.forcing_vectors is None


def test_write_manifest_records_git_sha(tmp_path: Path):
    """Task 5 (working prompt): the dataset manifest must record the git
    commit at write time, the same way common.checkpoint.save_checkpoint
    already does for model checkpoints -- previously missing."""
    path = tmp_path / "manifest_only.h5"
    write_sample(path, 0, split="train_interior", result=generate_one_sample(MU, PAYOFF, Mesh(N=32, p=3), TimeGrid(n_steps=5)))
    write_manifest(path, {"train_interior": 1}, n_failures=0, config_snapshot="test", seed=0)
    with h5py.File(path, "r") as f:
        assert "git_sha" in f.attrs
        assert isinstance(f.attrs["git_dirty"], (bool, np.bool_))
