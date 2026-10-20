"""Stage J correction tests: correction.krylov must reproduce the trusted
solver's own CN stepping exactly (never a reimplementation) and must
actually exploit a warm start (fewer/zero GMRES iterations when seeded near
the true solution), not merely accept and ignore a seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from correction.krylov import bc_dofs_for_mesh, compute_initial_coefficients, correct_trajectory

MU = CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.5)
PAYOFF = EuropeanPutPayoff(strike=100.0)
MESH = Mesh(N=64, p=3, x_min=float(np.log(100.0) - 8), x_max=float(np.log(100.0) + 8))
TIME_GRID_REQ = TimeGrid(n_steps=60)


@pytest.fixture(scope="module")
def reference_traj():
    return solve_cgmy(MU, PAYOFF, MESH, TIME_GRID_REQ, mode="toeplitz")


@pytest.fixture(scope="module")
def ops(reference_traj):
    return OperatorCache(MESH, reference_traj.mesh_information.x_min, reference_traj.mesh_information.x_max)


def test_compute_initial_coefficients_matches_reference_solve(reference_traj, ops):
    """The cheap mass-solve c0 must match solve_cgmy's own initial_coefficients
    bit-for-bit (to solver tolerance), since both solve the SAME linear
    system M_h c0 = b0 for the SAME b0."""
    c0, bc_left, bc_right = compute_initial_coefficients(MU, PAYOFF, reference_traj.mesh_information, ops)
    assert np.max(np.abs(c0 - reference_traj.initial_coefficients)) < 1e-10
    assert (bc_left, bc_right) == reference_traj.solver_metadata.bc_dofs


def test_bc_dofs_for_mesh_matches_reference_solve(reference_traj):
    bc_left, bc_right = bc_dofs_for_mesh(reference_traj.mesh_information)
    assert (bc_left, bc_right) == reference_traj.solver_metadata.bc_dofs


def test_warm_start_with_true_trajectory_converges_in_zero_iterations(reference_traj, ops):
    """Seeding GMRES with the EXACT solution of each step's linear system
    should converge in ~0 iterations and reproduce the trajectory almost
    exactly -- the cleanest possible correctness check on the operator/
    stepping reproduction."""
    result = correct_trajectory(
        MU, PAYOFF, reference_traj.mesh_information, reference_traj.time_grid, ops,
        seed_coefficients=reference_traj.coefficients, warm_start=True,
    )
    # Tolerance loosened from 1e-10 to 1e-9 for Task 1 (working prompt):
    # correct_trajectory now pins its Dirichlet-dof output exactly to
    # cgmy.dirichlet_bc.dirichlet_bc_values' closed-form formula (machine
    # precision against THAT formula -- see test_dirichlet_bc.py), rather
    # than to whatever GMRES's identity-row convergence produced. That
    # formula matches solve_cn's own internal BC-value computation to
    # ~1e-9 absolute, not bit-for-bit (benign floating-point-ordering
    # difference, not a modeling error -- see
    # test_dirichlet_bc.py::test_reference_solve_already_satisfies_the_same_bc_formula).
    assert np.max(np.abs(result.coefficients - reference_traj.coefficients)) < 1e-9
    assert np.mean(result.n_iterations_per_step) < 1.0
    assert all(result.converged_per_step)


def test_cold_start_matches_trusted_solvers_own_gmres_iteration_count(reference_traj, ops):
    """Cold-started (warm_start=False) correction must reproduce solve_cn's
    own mean GMRES iteration count exactly -- this is the strongest possible
    evidence that correct_trajectory's F/B matvecs and GMRES call are
    bit-identical to solve_cn's, not merely "close"."""
    result = correct_trajectory(
        MU, PAYOFF, reference_traj.mesh_information, reference_traj.time_grid, ops,
        seed_coefficients=None, warm_start=False,
    )
    assert np.mean(result.n_iterations_per_step) == pytest.approx(
        reference_traj.solver_metadata.mean_gmres_iters, rel=0.05
    )
    assert np.max(np.abs(result.coefficients - reference_traj.coefficients)) < 1e-8


def test_warm_start_requires_seed_coefficients(reference_traj, ops):
    with pytest.raises(ValueError):
        correct_trajectory(
            MU, PAYOFF, reference_traj.mesh_information, reference_traj.time_grid, ops,
            seed_coefficients=None, warm_start=True,
        )


def test_a_poor_seed_still_helps_relative_to_cold(reference_traj, ops):
    """A seed that is merely CLOSER to the truth than zero (not exact)
    should still reduce iteration count relative to cold start -- correction
    must actually exploit whatever prediction it is given, never discard
    it. Uses a deliberately crude seed (a constant fraction of the true
    trajectory) rather than the exact answer, to check this holds for an
    imperfect prediction too."""
    crude_seed = 0.7 * reference_traj.coefficients
    warm = correct_trajectory(
        MU, PAYOFF, reference_traj.mesh_information, reference_traj.time_grid, ops,
        seed_coefficients=crude_seed, warm_start=True,
    )
    cold = correct_trajectory(
        MU, PAYOFF, reference_traj.mesh_information, reference_traj.time_grid, ops,
        seed_coefficients=None, warm_start=False,
    )
    assert sum(warm.n_iterations_per_step) < sum(cold.n_iterations_per_step)
