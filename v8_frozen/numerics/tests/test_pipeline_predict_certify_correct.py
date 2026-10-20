"""Stage J pipeline tests: pipeline.predict_certify_correct's six-step
decision workflow, on a small mesh/time grid with a random (untrained)
model -- an interface/logic-conformance test (branch selection, decision
trace self-consistency, correction actually running/not running as
requested), not an accuracy test. See tests/test_correction_krylov.py for
the numerical-fidelity checks against the trusted solver, and
tests/test_model_interface.py for this project's established
untrained-model-on-a-small-mesh testing convention, followed here too.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.residuals import compute_all_residuals
from data.sampling import load_parameter_box
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import stats_from_box
from pipeline.predict_certify_correct import (
    build_mesh_info,
    build_skeleton_trajectory,
    predict_certify_correct,
)

torch.set_default_dtype(torch.float64)

FIXED_MESH = Mesh(N=32, p=3, x_min=float(np.log(100) - 6), x_max=float(np.log(100) + 6))
TIME_GRID_REQ = TimeGrid(n_steps=10)

# A fixed, known-SPD interior parameter point (not a random sample), matching
# tests/test_operator_consistency.py's convention of a hardcoded MU rather
# than a Sobol draw -- avoids picking a not-SPD corner (see exp05_ood.py's
# findings: high M / high Y corners can lose SPD-ness) and keeps this test
# deterministic across scipy/qmc versions.
MU = CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.4)


@pytest.fixture(scope="module")
def fixture_data():
    box = load_parameter_box("configs/base.yaml")
    stats = stats_from_box(box)
    ops = OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)

    mu = MU
    payoff = EuropeanPutPayoff(strike=100.0)
    traj = solve_cgmy(mu, payoff, FIXED_MESH, TIME_GRID_REQ, mode="toeplitz")
    time_grid = np.linspace(0.0, mu.T, TIME_GRID_REQ.n_steps + 1)

    torch.manual_seed(0)
    model = MLPCoeffNet(mu_dim=6, psi_dim=ops.n_dof, hidden_dims=(16, 16))
    model.eval()

    return {"box": box, "stats": stats, "ops": ops, "mu": mu, "payoff": payoff,
            "traj": traj, "time_grid": time_grid, "model": model}


def test_build_mesh_info_matches_a_real_solve(fixture_data):
    mesh_info = build_mesh_info(FIXED_MESH)
    ref = fixture_data["traj"].mesh_information
    assert mesh_info.N == ref.N and mesh_info.p == ref.p and mesh_info.n_dof == ref.n_dof
    assert mesh_info.h == pytest.approx(ref.h)
    assert mesh_info.x_min == pytest.approx(ref.x_min)
    assert mesh_info.x_max == pytest.approx(ref.x_max)
    np.testing.assert_allclose(mesh_info.nodes, ref.nodes)


def test_skeleton_trajectory_forcing_matches_real_solve(fixture_data):
    """boundary_forcing_vector is confirmed always-zero; the skeleton
    trajectory's forcing must match a real solve's stored forcing exactly
    (both zero), not merely be "close"."""
    traj = fixture_data["traj"]
    mesh_info = build_mesh_info(FIXED_MESH)
    skeleton = build_skeleton_trajectory(
        fixture_data["mu"], fixture_data["payoff"], mesh_info, fixture_data["time_grid"], traj.coefficients
    )
    np.testing.assert_array_equal(skeleton.forcing_vectors, traj.forcing_vectors)
    assert skeleton.solver_metadata.bc_dofs == traj.solver_metadata.bc_dofs
    assert skeleton.solver_metadata.dt == pytest.approx(traj.solver_metadata.dt)

    residuals = compute_all_residuals(skeleton, fixture_data["mu"], fixture_data["ops"])
    assert residuals.shape == (traj.solver_metadata.n_steps, mesh_info.n_dof)


def test_loose_tolerance_accepts_immediately(fixture_data):
    """Task 2 (working prompt): a trivially loose tolerance must still
    accept via the GUARANTEED certificate (accepted_guar_quick), never via
    eta_mass_indicator alone -- there is no "accepted_mass" branch anymore."""
    trace = predict_certify_correct(
        fixture_data["mu"], fixture_data["payoff"], FIXED_MESH, fixture_data["time_grid"],
        fixture_data["model"], fixture_data["stats"], fixture_data["ops"],
        tolerance=1e6, quick_pcg_iterations=3, max_pcg_iterations=50,
    )
    assert trace.branch == "accepted_guar_quick"
    assert trace.accepted
    assert not trace.mass_indicator_early_reject
    assert not np.isnan(trace.eta_guar_quick)
    assert trace.eta_final == trace.eta_guar_quick
    assert not trace.correction_ran
    assert trace.t_total == pytest.approx(
        trace.t_predict + trace.t_residual + trace.t_mass_indicator + trace.t_riesz_certificate, rel=1e-6
    )


def test_tight_tolerance_forces_correction_and_recovers_true_trajectory(fixture_data):
    trace = predict_certify_correct(
        fixture_data["mu"], fixture_data["payoff"], FIXED_MESH, fixture_data["time_grid"],
        fixture_data["model"], fixture_data["stats"], fixture_data["ops"],
        tolerance=1e-6, quick_pcg_iterations=3, max_pcg_iterations=50, keep_coefficients=True,
    )
    assert trace.correction_ran
    assert trace.branch in (
        "accepted_after_correction_guar", "corrected_but_not_certified",
    )
    # Task 2 (working prompt): eta_guar_corrected is ALWAYS computed after
    # correction (never skipped by eta_mass_indicator_corrected alone).
    assert not np.isnan(trace.eta_guar_corrected)
    # Correction must converge close to the true trajectory REGARDLESS of the
    # (here, untrained/poor) prediction quality -- warm-starting changes
    # iteration count, not the fixed point GMRES converges to.
    diff = np.max(np.abs(trace.final_coefficients - fixture_data["traj"].coefficients))
    assert diff < 1e-4
    assert trace.correction_n_iterations_per_step is not None
    assert trace.t_correction > 0.0


def test_run_correction_false_skips_the_expensive_step(fixture_data):
    trace = predict_certify_correct(
        fixture_data["mu"], fixture_data["payoff"], FIXED_MESH, fixture_data["time_grid"],
        fixture_data["model"], fixture_data["stats"], fixture_data["ops"],
        tolerance=1e-6, quick_pcg_iterations=3, max_pcg_iterations=50, run_correction=False,
    )
    assert trace.branch == "needs_correction_not_run"
    assert not trace.correction_ran
    assert trace.t_correction == 0.0
    assert not trace.accepted


def test_mass_indicator_reject_threshold_skips_quick_guar_but_never_accepts(fixture_data):
    """Task 2 (working prompt): a tight mass_indicator_reject_threshold must
    skip step 4 (eta_guar quick) entirely -- eta_guar_quick stays NaN, the
    query is routed toward correction, and it is NEVER accepted off
    eta_mass_indicator alone, however loose ``tolerance`` is."""
    trace = predict_certify_correct(
        fixture_data["mu"], fixture_data["payoff"], FIXED_MESH, fixture_data["time_grid"],
        fixture_data["model"], fixture_data["stats"], fixture_data["ops"],
        tolerance=1e6, quick_pcg_iterations=3, max_pcg_iterations=50,
        run_correction=False, mass_indicator_reject_threshold=0.0,
    )
    assert trace.mass_indicator_early_reject
    assert np.isnan(trace.eta_guar_quick)
    assert trace.branch == "needs_correction_not_run"
    assert not trace.accepted
    assert np.isnan(trace.eta_final)
    assert trace.t_riesz_certificate == 0.0

    # Same setup but WITH correction: still must never accept off the
    # indicator alone -- the final verdict always comes from eta_guar.
    trace2 = predict_certify_correct(
        fixture_data["mu"], fixture_data["payoff"], FIXED_MESH, fixture_data["time_grid"],
        fixture_data["model"], fixture_data["stats"], fixture_data["ops"],
        tolerance=1e6, quick_pcg_iterations=3, max_pcg_iterations=50,
        mass_indicator_reject_threshold=0.0,
    )
    assert trace2.mass_indicator_early_reject
    assert np.isnan(trace2.eta_guar_quick)
    assert trace2.correction_ran
    assert not np.isnan(trace2.eta_guar_corrected)
    assert trace2.eta_final == trace2.eta_guar_corrected
    assert trace2.branch in ("accepted_after_correction_guar", "corrected_but_not_certified")
