"""Every model in learning/ -- MLPCoeffNet, DeepONetCoeffNet, PODGalerkinROM,
ProjectedFNO -- must: return coefficients that reconstruct into V_h (right
shape, right dof ordering), satisfy the exact initial condition, and be
directly usable by certification.residuals.compute_all_residuals with no
shape or ordering fixups. This is an interface-conformance test, not an
accuracy test: models here are untrained/randomly initialized (except
PODGalerkinROM, whose "training" is just building a basis from a couple of
solved trajectories) -- large residuals are expected and fine; a shape
mismatch or an exception is not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from cgmy.operators import OperatorCache
from cgmy.reference_solver import EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.residuals import compute_all_residuals
from data import payoff_family
from data.sampling import load_parameter_box, sample_sobol
from learning.deeponet_coeff import DeepONetCoeffNet
from learning.fno import FNO1d
from learning.interface import wrap_as_trajectory
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import normalize_mu, stats_from_box
from learning.pod import PODGalerkinROM, compute_pod_basis
from learning.projected_fno import ProjectedFNO

torch.set_default_dtype(torch.float64)

FIXED_MESH = Mesh(N=32, p=3, x_min=np.log(100) - 6, x_max=np.log(100) + 6)


@pytest.fixture(scope="module")
def fixture_data():
    box = load_parameter_box("configs/base.yaml")
    stats = stats_from_box(box)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = sample_sobol(box, 4, seed=3)

    ops = OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)
    trajs = []
    for mu, payoff in samples:
        traj = solve_cgmy(mu, payoff, FIXED_MESH, TimeGrid(n_steps=10), mode="toeplitz")
        trajs.append((mu, payoff, traj))

    mu, payoff, traj = trajs[-1]  # held-out "test" point
    psi = payoff_family.european_put(payoff.strike, ops)
    mu_tilde = normalize_mu(mu, stats)
    return {
        "box": box, "stats": stats, "ops": ops, "trajs": trajs,
        "mu": mu, "payoff": payoff, "traj": traj, "psi": psi, "mu_tilde": mu_tilde,
    }


def _check_model(name, predicted, data):
    traj = data["traj"]
    n_dof = traj.mesh_information.n_dof

    assert predicted.shape == traj.coefficients.shape, (
        f"{name}: predicted shape {predicted.shape} != reference shape {traj.coefficients.shape}"
    )
    np.testing.assert_array_equal(
        predicted[0], traj.initial_coefficients,
        err_msg=f"{name}: predicted[0] != c0 exactly",
    )

    traj_pred = wrap_as_trajectory(traj, predicted)
    residuals = compute_all_residuals(traj_pred, data["mu"], data["ops"])
    assert residuals.shape == (traj.solver_metadata.n_steps, n_dof), (
        f"{name}: compute_all_residuals returned shape {residuals.shape}, "
        f"expected ({traj.solver_metadata.n_steps}, {n_dof})"
    )
    assert np.all(np.isfinite(residuals)), f"{name}: residuals contain NaN/inf"


def test_mlp_interface(fixture_data):
    torch.manual_seed(0)
    net = MLPCoeffNet(mu_dim=6, psi_dim=fixture_data["ops"].n_dof, hidden_dims=(16, 16))
    pred = net.predict_trajectory(
        fixture_data["mu_tilde"], fixture_data["psi"],
        fixture_data["traj"].initial_coefficients,
        fixture_data["traj"].time_grid, fixture_data["mu"].T,
    )
    _check_model("MLPCoeffNet", pred, fixture_data)


def test_deeponet_interface(fixture_data):
    torch.manual_seed(0)
    net = DeepONetCoeffNet(mu_dim=6, psi_dim=fixture_data["ops"].n_dof, latent_dim=16,
                            branch_hidden=(16,), trunk_hidden=(16,))
    pred = net.predict_trajectory(
        fixture_data["mu_tilde"], fixture_data["psi"],
        fixture_data["traj"].initial_coefficients,
        fixture_data["traj"].time_grid, fixture_data["mu"].T,
    )
    _check_model("DeepONetCoeffNet", pred, fixture_data)


def test_pod_galerkin_interface(fixture_data):
    trajs = fixture_data["trajs"]
    snapshots = np.concatenate([t.coefficients for _, _, t in trajs[:3]], axis=0)
    basis = compute_pod_basis(snapshots, n_modes=5)
    rom = PODGalerkinROM(basis, fixture_data["ops"], stats=fixture_data["stats"])
    pred = rom.predict_trajectory(
        fixture_data["mu_tilde"], fixture_data["psi"],
        fixture_data["traj"].initial_coefficients,
        fixture_data["traj"].time_grid, fixture_data["mu"].T,
    )
    _check_model("PODGalerkinROM", pred, fixture_data)


def test_projected_fno_interface(fixture_data):
    torch.manual_seed(0)
    fno = FNO1d(in_channels=1 + 1 + 6, out_channels=1, width=8, n_modes=4, n_layers=2)
    net = ProjectedFNO(fno, FIXED_MESH, fixture_data["ops"], mu_dim=6)
    pred = net.predict_trajectory(
        fixture_data["mu_tilde"], fixture_data["psi"],
        fixture_data["traj"].initial_coefficients,
        fixture_data["traj"].time_grid, fixture_data["mu"].T,
    )
    _check_model("ProjectedFNO", pred, fixture_data)

    # "report both raw and projected": raw grid output must NOT be
    # mistakenly usable as V_h coefficients (different length in general).
    raw = net.raw_grid_trajectory(fixture_data["mu_tilde"], fixture_data["psi"],
                                   fixture_data["traj"].time_grid, fixture_data["mu"].T)
    assert raw.shape == (fixture_data["traj"].time_grid.shape[0], FIXED_MESH.N + 1)
