"""GATE 3: train the MLP baseline on 5-20 trajectories to near-memorization.

This is an integration test of the plumbing (dataset -> normalization ->
network -> certification), not a research model. If the MLP cannot overfit
this tiny a set, the bug is in the plumbing (dataset indexing,
normalization, time handling, payoff representation, or trajectory
reconstruction) -- not model capacity. Debug there first; do not respond by
enlarging the network.

Reports, per trajectory (on the SAME trajectories the model was trained
on -- this is deliberately an overfitting check, not a generalization
test): coefficient error (Euclidean, logging only), L2 error (mass-
weighted, the actual training objective's numerator), option-price error,
residual norm of the predicted trajectory, and eta_mass_indicator
(heuristic, not a certificate) / eta_sharp / eta_guar evaluated on that
residual.
"""

from __future__ import annotations

import dataclasses
import time
import warnings

import numpy as np
import torch
import yaml

from cgmy.operators import OperatorCache
from cgmy.reference_solver import EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from cgmy_bspline.projection import eval_solution
from certification.constants import compute_stability_constants
from certification.guaranteed_certificate import eta_guar, eta_sharp
from certification.mass_indicator import eta_mass_indicator as compute_eta_mass_indicator
from certification.residuals import compute_all_residuals, residual_norm
from data import payoff_family
from data.sampling import load_parameter_box, sample_sobol
from learning.losses import euclidean_coefficient_error, l2_error, mass_weighted_loss
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import normalize_mu, normalize_psi, normalize_tau, stats_from_box

N_TRAJECTORIES = 10  # in [5, 20] per the task
MESH = Mesh(N=64, p=3)
TIME_GRID = TimeGrid(n_steps=100)
MODE = "toeplitz"
N_EPOCHS = 20000
LR = 2e-3
SEED = 0

torch.set_default_dtype(torch.float64)


def main() -> None:
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    max_pcg_iterations = cfg["certification"]["max_pcg_iterations"]
    preconditioner = cfg["certification"]["preconditioner"]

    box = load_parameter_box("configs/base.yaml")
    stats = stats_from_box(box)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = sample_sobol(box, N_TRAJECTORIES, seed=SEED)

    print(f"Gate 3: overfitting to {N_TRAJECTORIES} trajectories "
          f"(mesh N={MESH.N} p={MESH.p}, n_steps={TIME_GRID.n_steps}, mode={MODE})")

    trajectories = []
    ops_per_traj = []
    psi_per_traj = []
    t0 = time.time()
    for i, (mu, payoff) in enumerate(samples):
        traj = solve_cgmy(mu, payoff, MESH, TIME_GRID, mode=MODE)
        ops = OperatorCache(
            Mesh(N=traj.mesh_information.N, p=traj.mesh_information.p),
            traj.mesh_information.x_min, traj.mesh_information.x_max,
        )
        psi = payoff_family.european_put(payoff.strike, ops)
        trajectories.append(traj)
        ops_per_traj.append(ops)
        psi_per_traj.append(psi)
        print(f"  [{i}] mu=(C={mu.C:.3f},G={mu.G:.3f},M={mu.M:.3f},Y={mu.Y:.3f},"
              f"r={mu.r:.3f},T={mu.T:.3f})  n_dof={traj.mesh_information.n_dof}")
    print(f"solved {N_TRAJECTORIES} trajectories in {time.time() - t0:.1f}s")

    n_dof = trajectories[0].mesh_information.n_dof
    assert all(t.mesh_information.n_dof == n_dof for t in trajectories), (
        "all trajectories must share n_dof (fixed N, p) for one network output size"
    )

    # M_h is invariant to mesh scale/position (h^{-1/2} basis normalization
    # cancels it out -- verified numerically), so any one OperatorCache's
    # M_h is valid as the SHARED mass matrix for the batched training loss.
    mass_matrix_dense = torch.from_numpy(
        np.array([ops_per_traj[0].apply_M(e) for e in np.eye(n_dof)]).T
    )

    # ── Build training tensors: every (trajectory, time-level) pair ────────
    tau_list, tau_norm_list, mu_tilde_list, psi_list, c0_list, target_list = [], [], [], [], [], []
    for traj, psi, (mu, payoff) in zip(trajectories, psi_per_traj, samples):
        mu_tilde = normalize_mu(mu, stats)
        psi_scaled = normalize_psi(psi, payoff.strike)
        for n, tau in enumerate(traj.time_grid):
            tau_list.append([tau])
            tau_norm_list.append([normalize_tau(tau, mu.T)])
            mu_tilde_list.append(mu_tilde)
            psi_list.append(psi_scaled)
            c0_list.append(traj.initial_coefficients)
            target_list.append(traj.coefficients[n])

    tau_t = torch.tensor(np.array(tau_list))
    tau_norm_t = torch.tensor(np.array(tau_norm_list))
    mu_tilde_t = torch.tensor(np.array(mu_tilde_list))
    psi_t = torch.tensor(np.array(psi_list))
    c0_t = torch.tensor(np.array(c0_list))
    target_t = torch.tensor(np.array(target_list))
    print(f"training set: {tau_t.shape[0]} (trajectory, time-level) pairs")

    # ── Train ────────────────────────────────────────────────────────────
    torch.manual_seed(SEED)
    net = MLPCoeffNet(mu_dim=6, psi_dim=n_dof, hidden_dims=(128, 128, 128))
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 1e-3)

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = net(tau_t, tau_norm_t, mu_tilde_t, psi_t, c0_t)
        loss = mass_weighted_loss(pred, target_t, mass_matrix_dense)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 2000 == 0 or epoch == N_EPOCHS - 1:
            print(f"  epoch {epoch:5d}  L_data={loss.item():.6e}  lr={scheduler.get_last_lr()[0]:.2e}")
    print(f"training took {time.time() - t0:.1f}s")

    # ── Evaluate on the SAME trajectories (overfitting check) ──────────────
    print("\n" + "=" * 100)
    print("GATE 3 REPORT (per trajectory, evaluated on its own training data)")
    print("=" * 100)
    header = (f"{'i':>2} {'coef_err(mean/max)':>22} {'L2_err(mean/max)':>20} "
              f"{'price_err':>10} {'max||r||':>10} {'eta_mass_ind':>12} {'eta_sharp':>10} {'eta_guar':>10}")
    print(header)
    print("-" * len(header))

    net.eval()
    all_price_err = []
    all_l2_err = []
    with torch.no_grad():
        for i, (traj, ops, psi, (mu, payoff)) in enumerate(
            zip(trajectories, ops_per_traj, psi_per_traj, samples)
        ):
            mu_tilde = normalize_mu(mu, stats)
            Nt1 = traj.time_grid.shape[0]
            tau_i = torch.tensor(traj.time_grid.reshape(-1, 1))
            tau_norm_i = torch.tensor(
                np.array([[normalize_tau(t, mu.T)] for t in traj.time_grid])
            )
            mu_tilde_i = torch.tensor(np.tile(mu_tilde, (Nt1, 1)))
            psi_i = torch.tensor(np.tile(normalize_psi(psi, payoff.strike), (Nt1, 1)))
            c0_i = torch.tensor(np.tile(traj.initial_coefficients, (Nt1, 1)))
            target_i = torch.tensor(traj.coefficients)

            pred_i = net(tau_i, tau_norm_i, mu_tilde_i, psi_i, c0_i)

            coef_err = euclidean_coefficient_error(pred_i, target_i).numpy()
            l2_err = l2_error(pred_i, target_i, mass_matrix_dense).numpy()

            pred_coeffs = pred_i.numpy()
            traj_pred = dataclasses.replace(traj, coefficients=pred_coeffs)

            grid_info = {"N": traj.mesh_information.N, "h": traj.mesh_information.h,
                         "x_min": traj.mesh_information.x_min}
            x0 = np.array([np.log(payoff.strike)])
            price_true = eval_solution(traj.coefficients[-1], grid_info, traj.mesh_information.p, x0)[0]
            price_pred = eval_solution(pred_coeffs[-1], grid_info, traj.mesh_information.p, x0)[0]
            price_err = abs(price_pred - price_true)

            const = compute_stability_constants(mu, ops)
            if not const.is_spd:
                print(f"{i:2d}  S_h(mu) not SPD -- skipping certificates for this trajectory")
                continue

            residuals = compute_all_residuals(traj_pred, mu, ops)
            max_r = max(
                residual_norm(residuals[n], ops, traj.solver_metadata.bc_dofs)
                for n in range(traj.solver_metadata.n_steps)
            )
            dt = traj.solver_metadata.dt
            em = compute_eta_mass_indicator(residuals, ops, dt, const.alpha_certified)
            es = eta_sharp(residuals, mu, ops, dt)
            eg = eta_guar(residuals, mu, ops, dt, const.alpha_certified,
                          max_pcg_iterations=max_pcg_iterations, preconditioner=preconditioner)

            all_price_err.append(price_err)
            all_l2_err.append(l2_err.mean())

            print(f"{i:2d} {coef_err.mean():10.3e}/{coef_err.max():9.3e} "
                  f"{l2_err.mean():9.3e}/{l2_err.max():9.3e} "
                  f"{price_err:10.3e} {max_r:10.3e} {em:12.3e} {es.eta:10.3e} {eg.eta:10.3e}")

    print("=" * 100)
    print(f"mean option-price error across trajectories: {np.mean(all_price_err):.4e}")
    print(f"mean L2 error (mean over time steps, then over trajectories): {np.mean(all_l2_err):.4e}")
    print("\nGate 3 is a sanity check on the plumbing, not a fixed pass/fail threshold: "
          "these numbers should look like near-memorization (small, not exactly zero) "
          "given the network structurally reproduces c_h^0 exactly at tau=0.")


if __name__ == "__main__":
    main()
