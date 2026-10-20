"""Task 1 (working prompt): the time-dependent Dirichlet BC must be enforced
on any coefficient trajectory NOT produced by solve_cn itself (a neural
prediction, a corrected trajectory) -- structurally, not by a loss term or
tolerance -- since certification.residuals deliberately zeroes the residual
rows at those dofs and so can never catch a violation there on its own.

E_boundary = max over n, j in I_BC of |c[n,j] - c_BC[n,j]| must be exactly
(not merely approximately) zero on anything project_bc has touched, and the
raw, unprojected network output must NOT already satisfy this by accident
(confirming the test is actually exercising the fix, not a vacuous check).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cgmy.dirichlet_bc import bc_dofs_for_mesh, boundary_error, dirichlet_bc_values, project_bc
from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from data.sampling import load_parameter_box
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import stats_from_box
from pipeline.predict_certify_correct import build_mesh_info, predict_certify_correct

torch.set_default_dtype(torch.float64)

MU = CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.4)
PAYOFF = EuropeanPutPayoff(strike=100.0)
MESH = Mesh(N=32, p=3, x_min=float(np.log(100) - 6), x_max=float(np.log(100) + 6))
TIME_GRID_REQ = TimeGrid(n_steps=10)


@pytest.fixture(scope="module")
def reference_traj():
    return solve_cgmy(MU, PAYOFF, MESH, TIME_GRID_REQ, mode="toeplitz")


def test_reference_solve_already_satisfies_the_same_bc_formula(reference_traj):
    """Consistency check that dirichlet_bc_values is not just self-consistent
    within this module but actually matches what the trusted solver did --
    to floating-point precision, not bit-for-bit: dirichlet_bc_values is a
    Python reimplementation of solve_cn's own overwrite step (pre-existing
    in correction.krylov before this module; not reimplemented here), and
    the two evaluate exp/sqrt/multiply in whatever order each happens to,
    so a ~1e-9 absolute (~1e-11 relative, on O(50-60)-magnitude values)
    gap is benign floating-point-ordering noise, not a modeling error."""
    mesh_info = reference_traj.mesh_information
    e = boundary_error(reference_traj.coefficients, MU, PAYOFF, mesh_info, reference_traj.time_grid)
    assert e < 1e-8, f"trusted solver's own trajectory violates dirichlet_bc_values by {e:.3e}"


def test_project_bc_is_exact_and_leaves_interior_untouched(reference_traj):
    mesh_info = reference_traj.mesh_information
    time_grid = reference_traj.time_grid
    bc_left, bc_right = bc_dofs_for_mesh(mesh_info)
    interior = [j for j in range(mesh_info.n_dof) if j not in bc_left and j not in bc_right]

    rng = np.random.default_rng(0)
    corrupted = reference_traj.coefficients.copy()
    corrupted[:, bc_left] += rng.normal(scale=50.0, size=(corrupted.shape[0], len(bc_left)))
    corrupted[:, bc_right] += rng.normal(scale=50.0, size=(corrupted.shape[0], len(bc_right)))

    e_before = boundary_error(corrupted, MU, PAYOFF, mesh_info, time_grid)
    assert e_before > 1.0, "corruption should be large enough to be a meaningful regression test"

    fixed = project_bc(corrupted, MU, PAYOFF, mesh_info, time_grid)
    e_after = boundary_error(fixed, MU, PAYOFF, mesh_info, time_grid)
    assert e_after == 0.0, f"E_boundary must be exactly zero after projection, got {e_after:.3e}"

    # project_bc must not touch interior dofs.
    np.testing.assert_array_equal(fixed[:, interior], corrupted[:, interior])
    # project_bc must not mutate its input.
    assert not np.array_equal(corrupted, fixed)


def test_dirichlet_bc_values_right_boundary_is_always_zero():
    mesh_info = build_mesh_info(MESH)
    for tau in (0.0, 0.1, 0.4):
        _, c_R = dirichlet_bc_values(MU, PAYOFF, mesh_info, tau)
        assert c_R == 0.0


def test_dirichlet_bc_values_decays_with_tau_via_discounting():
    mesh_info = build_mesh_info(MESH)
    c_L_0, _ = dirichlet_bc_values(MU, PAYOFF, mesh_info, 0.0)
    c_L_T, _ = dirichlet_bc_values(MU, PAYOFF, mesh_info, MU.T)
    expected_ratio = np.exp(-MU.r * MU.T)
    assert c_L_T / c_L_0 == pytest.approx(expected_ratio)


@pytest.fixture(scope="module")
def pipeline_fixture(reference_traj):
    box = load_parameter_box("configs/base.yaml")
    stats = stats_from_box(box)
    ops = OperatorCache(MESH, MESH.x_min, MESH.x_max)
    time_grid = np.linspace(0.0, MU.T, TIME_GRID_REQ.n_steps + 1)

    torch.manual_seed(0)
    model = MLPCoeffNet(mu_dim=6, psi_dim=ops.n_dof, hidden_dims=(16, 16))
    for p in model.parameters():
        p.data = torch.randn_like(p) * 3.0  # deliberately large/unusual untrained weights
    model.eval()

    return {"box": box, "stats": stats, "ops": ops, "time_grid": time_grid, "model": model}


def test_pipeline_E_boundary_is_machine_precision(pipeline_fixture):
    """assert E_boundary <= 1e-12 (Task 1's required test), on the actual
    Stage J online workflow, with an untrained (deliberately large-weight)
    model -- i.e. a raw prediction that is NOT expected to already satisfy
    the BC by accident."""
    trace = predict_certify_correct(
        MU, PAYOFF, MESH, pipeline_fixture["time_grid"],
        pipeline_fixture["model"], pipeline_fixture["stats"], pipeline_fixture["ops"],
        tolerance=1e6, quick_pcg_iterations=3, max_pcg_iterations=50,
    )
    assert trace.E_boundary_raw > 1e-6, (
        "raw (untrained) network output should violate the Dirichlet BC by a "
        "nontrivial amount -- otherwise this test isn't exercising the fix"
    )
    assert trace.E_boundary <= 1e-12, f"E_boundary={trace.E_boundary:.3e} exceeds 1e-12"
