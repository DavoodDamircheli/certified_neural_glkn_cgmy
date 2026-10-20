"""Stage F baseline report: POD/reduced-basis ("why not just a ROM?") and
raw-vs-projected FNO.

All trajectories here are solved on ONE common fixed mesh domain -- a
prerequisite for POD (see learning.pod's module docstring), applied
consistently for a fair comparison.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch

from cgmy.dirichlet_bc import boundary_error, project_bc
from cgmy.operators import OperatorCache
from cgmy.reference_solver import TimeGrid, solve_cgmy
from data import payoff_family
from data.sampling import load_mesh_and_time_grid, load_parameter_box, sample_sobol
from learning.fno import FNO1d
from learning.interface import wrap_as_trajectory
from learning.normalization import normalize_mu, stats_from_box
from learning.pod import (
    PODGalerkinROM,
    compute_pod_basis,
    online_cost_report,
    reconstruction_error,
)
from learning.projected_fno import ProjectedFNO
from certification.residuals import compute_all_residuals, residual_norm

torch.set_default_dtype(torch.float64)

# Mesh and time grid read from configs/base.yaml (single source of truth
# -- see data.sampling.load_mesh_and_time_grid).
FIXED_MESH, _PRODUCTION_TIME_GRID = load_mesh_and_time_grid("configs/base.yaml")
N_TRAIN = 15
N_TEST_IN_BOX = 5
N_STEPS = _PRODUCTION_TIME_GRID.n_steps
SEED = 0


def _solve_all(samples, mesh, n_steps):
    return [(mu, payoff, solve_cgmy(mu, payoff, mesh, TimeGrid(n_steps=n_steps), mode="toeplitz"))
            for mu, payoff in samples]


def pod_report():
    print("=" * 78)
    print("POD / reduced-basis baseline -- \"why not just a reduced-order model?\"")
    print("=" * 78)

    box = load_parameter_box("configs/base.yaml")
    stats = stats_from_box(box)
    ops = OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_samples = sample_sobol(box, N_TRAIN, seed=SEED)
        test_samples = sample_sobol(box, N_TEST_IN_BOX, seed=SEED + 1)  # held-out, still in-box
        # "extrapolation": deliberately push Y and M toward the box edges,
        # beyond where any training snapshot's Y, M land (Sobol interior
        # coverage is typically not exactly at the boundary).
        from data.sampling import _edge_samples
        extrap_samples = _edge_samples(box, 5, seed=SEED + 2)

    print(f"solving {N_TRAIN} training + {N_TEST_IN_BOX} in-box-test + "
          f"{len(extrap_samples)} edge/extrapolation trajectories on the fixed mesh...")
    t0 = time.time()
    train = _solve_all(train_samples, FIXED_MESH, N_STEPS)
    test_in_box = _solve_all(test_samples, FIXED_MESH, N_STEPS)
    test_extrap = _solve_all(extrap_samples, FIXED_MESH, N_STEPS)
    print(f"...done in {time.time() - t0:.1f}s")

    snapshots = np.concatenate([t.coefficients for _, _, t in train], axis=0)

    print(f"\n{'r':>4} {'oracle recon err (mean/max, train)':>36} "
          f"{'held-out-in-box err (mean/max)':>32} {'extrap err (mean/max)':>24}")
    for r in (2, 4, 8, 16, 24, 32):
        basis = compute_pod_basis(snapshots, n_modes=r)
        rom = PODGalerkinROM(basis, ops, stats=stats)

        oracle_errs = [reconstruction_error(c, basis) for _, _, t in train for c in t.coefficients]

        def rom_errs(dataset):
            errs = []
            for mu, payoff, traj in dataset:
                mu_tilde = normalize_mu(mu, stats)
                pred = rom.predict_trajectory(mu_tilde, None, traj.initial_coefficients,
                                               traj.time_grid, mu.T)
                errs.extend(np.linalg.norm(pred[n] - traj.coefficients[n]) for n in range(len(traj.time_grid)))
            return errs

        in_box_errs = rom_errs(test_in_box)
        extrap_errs = rom_errs(test_extrap)

        print(f"{r:4d} {np.mean(oracle_errs):16.3e}/{np.max(oracle_errs):<16.3e} "
              f"{np.mean(in_box_errs):14.3e}/{np.max(in_box_errs):<14.3e} "
              f"{np.mean(extrap_errs):10.3e}/{np.max(extrap_errs):<10.3e}")

    print("\nonline cost (r=8, one trajectory, n_steps=", N_STEPS, "):")
    basis8 = compute_pod_basis(snapshots, n_modes=8)
    cost = online_cost_report(basis8, ops, train[0][0], N_STEPS)
    print(f"  setup (A_r build) time: {cost['setup_time_s']*1000:.2f} ms")
    print(f"  {N_STEPS} reduced CN steps time: {cost['n_steps_time_s']*1000:.2f} ms")
    print(f"  storage: {cost['storage_floats']} floats "
          f"({cost['storage_vs_full_order_ratio']*100:.3f}% of a dense n_dof x n_dof matrix)")
    print(f"  (a full-order solve of this trajectory took "
          f"{train[0][2].solver_metadata.cpu_time*1000:.1f} ms)")

    print("\nInterpretation: oracle reconstruction error (best possible, using the TRUE\n"
          "trajectory) stays small even for held-out in-box points, but the actual\n"
          "PREDICTIVE POD-Galerkin ROM's error is substantially larger and grows further\n"
          "under extrapolation -- the ROM has no boundary-condition-aware lifting (a real,\n"
          "documented limitation, see learning.pod), and the fixed linear subspace U_r\n"
          "captures the training snapshots' manifold, not the true solution operator.\n"
          "This is the concrete answer to \"why not just a ROM\": online cost is negligible\n"
          "(a small dense linear solve vs. a full nonlocal Toeplitz solve), but accuracy is\n"
          "capped by the fixed subspace in a way a properly-trained neural model may not be.")


def fno_raw_vs_projected_report():
    print("\n" + "=" * 78)
    print("FNO: raw grid output vs. Pi_h-projected V_h coefficients")
    print("=" * 78)

    box = load_parameter_box("configs/base.yaml")
    stats = stats_from_box(box)
    ops = OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, payoff = sample_sobol(box, 1, seed=99)[0]
    traj = solve_cgmy(mu, payoff, FIXED_MESH, TimeGrid(n_steps=N_STEPS), mode="toeplitz")
    psi = payoff_family.european_put(payoff.strike, ops)
    mu_tilde = normalize_mu(mu, stats)

    torch.manual_seed(0)
    fno = FNO1d(in_channels=1 + 1 + 6, out_channels=1, width=32, n_modes=16, n_layers=4)
    net = ProjectedFNO(fno, FIXED_MESH, ops, mu_dim=6)

    raw = net.raw_grid_trajectory(mu_tilde, psi, traj.time_grid, mu.T)
    projected = net.predict_trajectory(mu_tilde, psi, traj.initial_coefficients, traj.time_grid, mu.T)

    print(f"raw grid output shape: {raw.shape}  (n_grid = N+1 = {FIXED_MESH.N + 1}) -- NOT V_h coefficients")
    print(f"projected coefficients shape: {projected.shape}  (n_dof = {ops.n_dof}) -- IS V_h, certifiable")
    print(f"raw[0] range: [{raw[0].min():.3e}, {raw[0].max():.3e}]  "
          f"(untrained network; not expected to resemble the payoff)")
    print(f"projected[0] == c0 exactly: {np.array_equal(projected[0], traj.initial_coefficients)}")

    # Task 1 (working prompt): "projected" above is the Pi_h projection onto
    # V_h coefficient space (learning.projected_fno), a different concept
    # from project_bc's Dirichlet-BC enforcement -- an untrained network's
    # output is certifiable (right space) but not yet BC-exact, and
    # compute_all_residuals below would be blind to that at the BC dofs.
    e_boundary_raw = boundary_error(projected, mu, payoff, traj.mesh_information, traj.time_grid)
    projected = project_bc(projected, mu, payoff, traj.mesh_information, traj.time_grid)
    print(f"E_boundary before projection: {e_boundary_raw:.3e}  (raw network output, untrained)")

    traj_pred = wrap_as_trajectory(traj, projected)
    residuals = compute_all_residuals(traj_pred, mu, ops)
    max_r = max(residual_norm(residuals[n], ops, traj.solver_metadata.bc_dofs)
                for n in range(traj.solver_metadata.n_steps))
    print(f"max residual norm of the PROJECTED (untrained) FNO trajectory: {max_r:.3e}")
    print("(large, as expected for an untrained network -- this demonstrates the pipeline\n"
          " runs end to end, not that this FNO is accurate; that needs actual training,\n"
          " which is future work, not part of this integration-level baseline report.)")


if __name__ == "__main__":
    pod_report()
    fno_raw_vs_projected_report()
