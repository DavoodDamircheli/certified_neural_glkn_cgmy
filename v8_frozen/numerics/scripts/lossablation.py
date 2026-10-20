"""Stage G: residual-based training ablation.

Three training regimes -- data-only, residual-only, hybrid (L_data +
lambda_R * L_res) swept over lambda_R -- with IDENTICAL dataset,
architecture, training budget, and optimizer schedule, differing only in
the loss function. Compares prediction accuracy, certificate magnitude,
training time, robustness (train->stress degradation), and generalization
(train->test gap). Writes results/lossablation/*.csv and
tables/lossablation.tex.
"""

from __future__ import annotations

import csv
import dataclasses
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from cgmy.dirichlet_bc import boundary_error, project_bc
from cgmy.operators import OperatorCache
from cgmy.reference_solver import EuropeanPutPayoff, TimeGrid, solve_cgmy
from certification.constants import compute_stability_constants
from certification.guaranteed_certificate import eta_sharp
from certification.mass_indicator import eta_mass_indicator as compute_eta_mass_indicator
from certification.residuals import compute_all_residuals, residual_norm
from data import payoff_family
from data.sampling import (
    ParameterBox,
    _edge_samples,
    load_mesh_and_time_grid,
    load_parameter_box,
    sample_sobol,
)
from learning.interface import wrap_as_trajectory
from learning.losses import l2_error, mass_weighted_loss, residual_loss, torch_cn_residual
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import normalize_mu, normalize_psi, normalize_tau, stats_from_box

torch.set_default_dtype(torch.float64)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "lossablation"
TABLES_DIR = REPO_ROOT / "tables"

# Spatial mesh read from configs/base.yaml (single source of truth --
# see data.sampling.load_mesh_and_time_grid). The production TimeGrid is
# discarded here: N_STEPS below is a deliberately SMALLER, ablation-scale
# step count (Stage G wall-clock reasons), not the production convention.
FIXED_MESH, _ = load_mesh_and_time_grid("configs/base.yaml")
N_TRAIN, N_TEST, N_STRESS = 10, 5, 5
N_STEPS = 40
N_EPOCHS = 3000
LR = 2e-3
SEED = 0

REGIMES: list[tuple[str, float | None]] = [
    ("data_only", 0.0),
    ("residual_only", None),  # None = no data term at all
    # NOTE: at initialization, L_res is ~3.4e6x the magnitude of L_data
    # (measured directly -- L_res is an unnormalized sum of squared
    # physical-unit residuals through M_h^{-1}, whose eigenvalues are large
    # since lambda_min(M_h) ~ 0.02; L_data is a normalized ratio, O(1) or
    # less by construction). A lambda_R sweep must be chosen relative to
    # THIS ratio, not to L_data's own O(1) scale, or every "sweep" value
    # ends up in the same (residual-dominated) regime -- which is exactly
    # what happened on the first attempt at this sweep (0.01-10.0 all
    # produced near-identical results indistinguishable from
    # residual_only). These values bracket the ~3e-7 balance point.
    ("hybrid_lambda_1e-8", 1e-8),
    ("hybrid_lambda_1e-7", 1e-7),
    ("hybrid_lambda_3e-7", 3e-7),
    ("hybrid_lambda_1e-6", 1e-6),
    ("hybrid_lambda_1e-5", 1e-5),
]


def _solve_set(samples, mesh, n_steps):
    return [(mu, payoff, solve_cgmy(mu, payoff, mesh, TimeGrid(n_steps=n_steps), mode="toeplitz"))
            for mu, payoff in samples]


def _build_dataset(trajs, ops, stats):
    """Flatten (trajectory, time-level) pairs into training tensors, keeping
    track of each trajectory's contiguous block (start, end, mu, dt,
    bc_dofs) so residual pairs can be sliced out per trajectory."""
    tau_list, tau_norm_list, mu_tilde_list, psi_list, c0_list, target_list = [], [], [], [], [], []
    blocks = []
    cursor = 0
    for mu, payoff, traj in trajs:
        mu_tilde = normalize_mu(mu, stats)
        psi = payoff_family.european_put(payoff.strike, ops)
        psi_scaled = normalize_psi(psi, payoff.strike)
        n1 = len(traj.time_grid)
        for n, tau in enumerate(traj.time_grid):
            tau_list.append([tau])
            tau_norm_list.append([normalize_tau(tau, mu.T)])
            mu_tilde_list.append(mu_tilde)
            psi_list.append(psi_scaled)
            c0_list.append(traj.initial_coefficients)
            target_list.append(traj.coefficients[n])
        blocks.append({
            "start": cursor, "end": cursor + n1, "mu": mu, "traj": traj,
            "dt": traj.solver_metadata.dt, "bc_dofs": traj.solver_metadata.bc_dofs,
        })
        cursor += n1

    tensors = {
        "tau": torch.tensor(np.array(tau_list)),
        "tau_norm": torch.tensor(np.array(tau_norm_list)),
        "mu_tilde": torch.tensor(np.array(mu_tilde_list)),
        "psi": torch.tensor(np.array(psi_list)),
        "c0": torch.tensor(np.array(c0_list)),
        "target": torch.tensor(np.array(target_list)),
    }
    return tensors, blocks


def _residual_terms(pred: torch.Tensor, blocks, ops):
    """Slice adjacent (n, n+1) pairs out of pred (one contiguous block per
    trajectory) and stack them into batched residual inputs."""
    c_n_list, c_np1_list, fbar_list, mus, bc_dofs = [], [], [], [], None
    for b in blocks:
        block_pred = pred[b["start"]:b["end"]]
        traj = b["traj"]
        for n in range(len(traj.time_grid) - 1):
            c_n_list.append(block_pred[n])
            c_np1_list.append(block_pred[n + 1])
            fbar_list.append(torch.tensor(traj.forcing_vectors[n]))
            mus.append(b["mu"])
        bc_dofs = b["bc_dofs"]  # same mesh/discretization for every block (fixed mesh)
    c_n = torch.stack(c_n_list)
    c_np1 = torch.stack(c_np1_list)
    fbar = torch.stack(fbar_list)
    return c_n, c_np1, fbar, mus, bc_dofs, blocks[0]["dt"]


def train_one_regime(name, lambda_r, train_tensors, blocks, ops, mass_matrix_dense, n_dof):
    torch.manual_seed(SEED)
    net = MLPCoeffNet(mu_dim=6, psi_dim=n_dof, hidden_dims=(128, 128, 128))
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 1e-3)

    use_data = name != "residual_only"
    use_residual = name == "residual_only" or (lambda_r is not None and lambda_r > 0)

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = net(train_tensors["tau"], train_tensors["tau_norm"], train_tensors["mu_tilde"],
                   train_tensors["psi"], train_tensors["c0"])

        loss = torch.tensor(0.0)
        if use_data:
            loss = loss + mass_weighted_loss(pred, train_tensors["target"], mass_matrix_dense)
        if use_residual:
            c_n, c_np1, fbar, mus, bc_dofs, dt = _residual_terms(pred, blocks, ops)
            r = torch_cn_residual(c_n, c_np1, fbar, dt, mus, ops, bc_dofs)
            l_res = residual_loss(r, ops)
            weight = lambda_r if (lambda_r is not None and lambda_r > 0) else 1.0
            loss = loss + weight * l_res

        loss.backward()
        optimizer.step()
        scheduler.step()
    train_time = time.time() - t0
    return net, train_time


def evaluate(net, dataset_trajs, ops, stats, max_pcg_iterations=200, preconditioner="none"):
    """Per-trajectory: L2 error, coefficient error, max residual,
    eta_mass_indicator (heuristic, not a certificate), eta_sharp."""
    rows = []
    for mu, payoff, traj in dataset_trajs:
        mu_tilde = normalize_mu(mu, stats)
        psi = payoff_family.european_put(payoff.strike, ops)
        pred_raw = net.predict_trajectory(mu_tilde, psi, traj.initial_coefficients, traj.time_grid, mu.T)
        # Task 1 (working prompt): project onto the exact Dirichlet BC before
        # this prediction feeds the residual/certificate machinery below --
        # certification.residuals zeroes the residual rows at those dofs, so
        # an unprojected prediction's violations there would never be caught.
        e_boundary = boundary_error(pred_raw, mu, traj.payoff, traj.mesh_information, traj.time_grid)
        pred = project_bc(pred_raw, mu, traj.payoff, traj.mesh_information, traj.time_grid)

        diff = pred - traj.coefficients
        l2_err = np.sqrt(np.array([d @ ops.apply_M(d) for d in diff]))

        traj_pred = wrap_as_trajectory(traj, pred)
        const = compute_stability_constants(mu, ops)
        if not const.is_spd:
            rows.append({"l2_mean": l2_err.mean(), "l2_max": l2_err.max(),
                         "max_residual": np.nan, "eta_mass_indicator": np.nan, "eta_sharp": np.nan,
                         "E_boundary_raw": e_boundary})
            continue

        residuals = compute_all_residuals(traj_pred, mu, ops)
        max_r = max(residual_norm(residuals[n], ops, traj.solver_metadata.bc_dofs)
                    for n in range(traj.solver_metadata.n_steps))
        dt = traj.solver_metadata.dt
        em = compute_eta_mass_indicator(residuals, ops, dt, const.alpha_certified)
        es = eta_sharp(residuals, mu, ops, dt)

        rows.append({"l2_mean": l2_err.mean(), "l2_max": l2_err.max(),
                     "max_residual": max_r, "eta_mass_indicator": em, "eta_sharp": es.eta,
                     "E_boundary_raw": e_boundary})
    return rows


def _agg(rows, key):
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    box = load_parameter_box("configs/base.yaml")
    stats = stats_from_box(box)
    ops = OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_samples = sample_sobol(box, N_TRAIN, seed=SEED)
        test_samples = sample_sobol(box, N_TEST, seed=SEED + 1)
        stress_samples = _edge_samples(box, N_STRESS, seed=SEED + 2)

    print(f"solving {N_TRAIN} train + {N_TEST} test + {N_STRESS} stress trajectories "
          f"on the fixed mesh (N={FIXED_MESH.N}, n_steps={N_STEPS})...")
    t0 = time.time()
    train_trajs = _solve_set(train_samples, FIXED_MESH, N_STEPS)
    test_trajs = _solve_set(test_samples, FIXED_MESH, N_STEPS)
    stress_trajs = _solve_set(stress_samples, FIXED_MESH, N_STEPS)
    print(f"...done in {time.time() - t0:.1f}s")

    n_dof = ops.n_dof
    mass_matrix_dense = torch.tensor(np.array([ops.apply_M(e) for e in np.eye(n_dof)]).T)
    train_tensors, blocks = _build_dataset(train_trajs, ops, stats)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for name, lambda_r in REGIMES:
        print(f"\n--- training regime: {name} ---")
        net, train_time = train_one_regime(name, lambda_r, train_tensors, blocks, ops, mass_matrix_dense, n_dof)

        train_rows = evaluate(net, train_trajs, ops, stats)
        test_rows = evaluate(net, test_trajs, ops, stats)
        stress_rows = evaluate(net, stress_trajs, ops, stats)

        result = {
            "regime": name,
            "lambda_R": lambda_r if lambda_r is not None else "n/a (no data term)",
            "train_time_s": train_time,
            "train_l2_mean": _agg(train_rows, "l2_mean"),
            "test_l2_mean": _agg(test_rows, "l2_mean"),
            "stress_l2_mean": _agg(stress_rows, "l2_mean"),
            "train_max_residual": _agg(train_rows, "max_residual"),
            "test_max_residual": _agg(test_rows, "max_residual"),
            "stress_max_residual": _agg(stress_rows, "max_residual"),
            "test_eta_mass_indicator": _agg(test_rows, "eta_mass_indicator"),
            "test_eta_sharp": _agg(test_rows, "eta_sharp"),
            "stress_eta_sharp": _agg(stress_rows, "eta_sharp"),
            "test_E_boundary_raw": _agg(test_rows, "E_boundary_raw"),
            "generalization_gap": _agg(test_rows, "l2_mean") - _agg(train_rows, "l2_mean"),
            "robustness_ratio_stress_over_train": _agg(stress_rows, "l2_mean") / max(_agg(train_rows, "l2_mean"), 1e-12),
        }
        all_results.append(result)
        print(f"  train_time={train_time:.1f}s  train_L2={result['train_l2_mean']:.3e}  "
              f"test_L2={result['test_l2_mean']:.3e}  stress_L2={result['stress_l2_mean']:.3e}  "
              f"test_eta_sharp={result['test_eta_sharp']:.3e}")

    csv_path = RESULTS_DIR / "ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nwrote {csv_path}")

    tex_path = TABLES_DIR / "lossablation.tex"
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"regime & $\lambda_R$ & train $s$ & train $L_2$ & test $L_2$ & stress $L_2$ & test $\eta_{sharp}$ & gen. gap \\",
        r"\midrule",
    ]
    for r in all_results:
        lam = r["lambda_R"] if isinstance(r["lambda_R"], str) else f"{r['lambda_R']:.3g}"
        lines.append(
            f"{r['regime']} & {lam} & {r['train_time_s']:.1f} & {r['train_l2_mean']:.3e} & "
            f"{r['test_l2_mean']:.3e} & {r['stress_l2_mean']:.3e} & {r['test_eta_sharp']:.3e} & "
            f"{r['generalization_gap']:.3e} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex_path}")

    print("\n" + "=" * 100)
    print("STAGE G ABLATION SUMMARY")
    print("=" * 100)
    header = (f"{'regime':22s} {'lambda_R':>10} {'train_s':>8} {'train_L2':>10} {'test_L2':>10} "
              f"{'stress_L2':>10} {'gen_gap':>10} {'robust_ratio':>12} {'test_eta_sharp':>14}")
    print(header)
    for r in all_results:
        lam = r["lambda_R"] if isinstance(r["lambda_R"], str) else f"{r['lambda_R']:.3g}"
        print(f"{r['regime']:22s} {lam:>10} {r['train_time_s']:8.1f} {r['train_l2_mean']:10.3e} "
              f"{r['test_l2_mean']:10.3e} {r['stress_l2_mean']:10.3e} {r['generalization_gap']:10.3e} "
              f"{r['robustness_ratio_stress_over_train']:12.3e} {r['test_eta_sharp']:14.3e}")


if __name__ == "__main__":
    main()
