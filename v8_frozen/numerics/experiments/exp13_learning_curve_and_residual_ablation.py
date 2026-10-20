"""exp13 -- Task 12 (working prompt): learning curve and residual-loss
ablation. Two independent deliverables, reported in one script since the
task groups them as one item, but using two different (deliberately
chosen) model/data setups:

Part 1 -- learning curve (main model, real production data): trains the
main model (exp_common.FINAL_MODEL_NAME = DeepONet) on tranches of the
REAL data/production_dataset.h5 (the same dataset checkpoints/
deeponet_final.pt and experiments/exp12_seed_variance.py use) at
N_train in {100, 200, 400, 800}. N_train=1600 (the task's fifth point) is
DROPPED: the production dataset only has 800 solved trajectories, so a
1600 point would need 800 NEW reference solves generated for this one
ablation point alone -- a data-generation step out of proportion to this
task's scope, so it is dropped for the same "wall-clock reasons" this
project has flagged at every other reduced-scale run, rather than done
silently or fabricated. Reports, per N_train: validation loss (the SAME
mass_weighted_loss functional actually trained on, evaluated out-of-
sample -- consistent with what checkpoints/*.pt's own "val_l2_error"
field already means throughout this project, not a different ad hoc
definition), test relative L2, test relative energy error, mean
guaranteed certificate (eta_guar, SPD queries only), training time.
Optionally fits E(N) ~= C * N^{-beta} via log-log least squares (task:
"optionally").

Part 2 -- residual-loss ablation (lambda_R sweep, MLPCoeffNet, ablation-
scale on-the-fly data): sweeps lambda_R in {0, 1e-9, 1e-8, 1e-7, 1e-6,
1e-5} -- the task's EXACT literal set, not Stage G's own
(scripts/lossablation.py's {0.0, None, 1e-8, 1e-7, 3e-7, 1e-6, 1e-5}
regimes and value choices, which answer a related but different
question and stay as they are, unmodified, alongside their own already-
published exp07_loss_ablation.py write-up). Reuses Stage G's VALIDATED
differentiable-residual training path (learning.losses.torch_cn_residual/
residual_loss, MLPCoeffNet, on-the-fly ablation-scale solves at
N_STEPS=40) rather than building and separately validating a new
differentiable-residual path for DeepONet -- an unnecessary, unvalidated
extra piece of code this task does not require. Reports, per lambda_R:
prediction error (relative L2), energy error (relative energy), residual
norm (max residual over the test set), sharp certificate (eta_sharp),
guaranteed certificate (eta_guar), effectivity (eta_guar / E_i, exp02's
own established definition) -- the task's exact metric list, going
beyond Stage G's own lossablation.py (which did not compute energy
error, eta_guar, or effectivity at all).

Task 12's own required honesty check: "residual regularization improved
predictive accuracy in the tested setting but did NOT monotonically
improve the certificate. Look for a Pareto tradeoff; do not claim it
improves certification." This is checked directly against THIS run's own
numbers below (see the printed verdict), not assumed or copy-pasted from
Stage G's separate, differently-scoped finding.

Outputs: results/exp13_learning_curve.h5, results/exp13_lambda_r_sweep.h5,
figures/exp13_learning_curve.png, figures/exp13_lambda_r_sweep.png,
tables/exp13_learning_curve.tex, tables/exp13_lambda_r_sweep.tex.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cgmy.reference_solver import TimeGrid, solve_cgmy
from common import io as common_io
from data.dataset import read_trajectory
from data import payoff_family
from data.sampling import sample_sobol
from exp_common import (
    FIGURES_DIR,
    FIXED_MESH,
    MAX_PCG_ITERATIONS,
    PRECONDITIONER,
    REPO_ROOT,
    RESULTS_DIR,
    TABLES_DIR,
    compute_certificates,
    fmt,
    get_box_and_stats,
    get_ops,
    predict,
    relative_energy_error,
    relative_l2_error,
    true_error_Ei,
    write_h5,
    write_latex_table,
)
from learning.deeponet_coeff import DeepONetCoeffNet
from learning.losses import mass_weighted_loss, residual_loss, torch_cn_residual
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import normalize_mu, normalize_psi, normalize_tau
from certification.constants import compute_stability_constants
from certification.guaranteed_certificate import eta_guar as compute_eta_guar
from certification.guaranteed_certificate import eta_sharp as compute_eta_sharp
from certification.residuals import compute_all_residuals, residual_norm
from cgmy.dirichlet_bc import project_bc
from learning.interface import wrap_as_trajectory

torch.set_default_dtype(torch.float64)

# ── Part 1: learning curve ──────────────────────────────────────────────
DATASET_PATH = REPO_ROOT / "data" / "production_dataset.h5"
N_TRAIN_CURVE = [100, 200, 400, 800]  # 1600 dropped -- see module docstring
N_EPOCHS_CURVE = 3000  # matches scripts/train_final_models.py's convention
LR_CURVE = 2e-3
SEED_CURVE = 0
N_HELDOUT = 40
HELDOUT_SEED = 131131  # disjoint from every other experiment's seed in this project

DEEPONET_KWARGS = {"mu_dim": 6, "latent_dim": 64, "branch_hidden": (128, 128), "trunk_hidden": (64, 64)}

# ── Part 2: residual-loss ablation ──────────────────────────────────────
LAMBDA_R_VALUES = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # task's literal set
N_TRAIN_ABL = 40
N_TEST_ABL = 25
N_STEPS_ABL = 40  # ablation-scale time grid, matches scripts/lossablation.py
N_EPOCHS_ABL = 3000
LR_ABL = 2e-3
SEED_ABL = 0
ABL_SEED_BASE = 141141  # disjoint from every other experiment's seed in this project


# ── shared helpers ───────────────────────────────────────────────────────

def load_train_samples(dataset_path: Path, n_samples: int):
    ids = common_io.list_groups(dataset_path, parent="samples")
    trajs = []
    for sid in sorted(ids, key=int):
        if int(sid) >= n_samples:
            continue
        traj = read_trajectory(dataset_path, int(sid))
        trajs.append((traj.parameter_vector, traj.payoff, traj))
    return trajs


def build_tensors(trajs, ops, stats):
    tau_l, tau_norm_l, mu_tilde_l, psi_l, c0_l, target_l = [], [], [], [], [], []
    for mu, payoff, traj in trajs:
        mu_tilde = normalize_mu(mu, stats)
        psi = normalize_psi(payoff_family.european_put(payoff.strike, ops), payoff.strike)
        for n, tau in enumerate(traj.time_grid):
            tau_l.append([tau]); tau_norm_l.append([normalize_tau(tau, mu.T)])
            mu_tilde_l.append(mu_tilde); psi_l.append(psi)
            c0_l.append(traj.initial_coefficients); target_l.append(traj.coefficients[n])
    return {
        "tau": torch.tensor(np.array(tau_l)), "tau_norm": torch.tensor(np.array(tau_norm_l)),
        "mu_tilde": torch.tensor(np.array(mu_tilde_l)), "psi": torch.tensor(np.array(psi_l)),
        "c0": torch.tensor(np.array(c0_l)), "target": torch.tensor(np.array(target_l)),
    }


# ── Part 1: learning curve ──────────────────────────────────────────────

def train_deeponet(tensors, mass_matrix_dense, n_dof, seed, n_epochs, lr):
    kwargs = dict(DEEPONET_KWARGS, psi_dim=n_dof)
    torch.manual_seed(seed)
    net = DeepONetCoeffNet(**kwargs)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 1e-3)
    t0 = time.time()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        pred = net(tensors["tau"], tensors["tau_norm"], tensors["mu_tilde"], tensors["psi"], tensors["c0"])
        loss = mass_weighted_loss(pred, tensors["target"], mass_matrix_dense)
        loss.backward()
        optimizer.step()
        scheduler.step()
    net.eval()
    return net, time.time() - t0, float(loss.item())


def run_learning_curve(ops, box, stats):
    n_dof = ops.n_dof
    mass_matrix_dense = torch.tensor(np.array([ops.apply_M(e) for e in np.eye(n_dof)]).T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        heldout_samples = sample_sobol(box, N_HELDOUT, seed=HELDOUT_SEED)

    print(f"exp13 (learning curve): solving {N_HELDOUT} held-out trajectories "
          f"(shared across every N_train)...")
    from exp_common import solve_reference
    heldout_trajs = [(mu, payoff, solve_reference(mu)) for mu, payoff in heldout_samples]
    heldout_tensors = build_tensors(heldout_trajs, ops, stats)

    results = []
    for n_train in N_TRAIN_CURVE:
        print(f"\nexp13 (learning curve): N_train={n_train} -- loading + training "
              f"({N_EPOCHS_CURVE} epochs)...")
        train_trajs = load_train_samples(DATASET_PATH, n_train)
        tensors = build_tensors(train_trajs, ops, stats)
        net, t_train, final_train_loss = train_deeponet(
            tensors, mass_matrix_dense, n_dof, SEED_CURVE, N_EPOCHS_CURVE, LR_CURVE
        )

        with torch.no_grad():
            pred_val = net(heldout_tensors["tau"], heldout_tensors["tau_norm"], heldout_tensors["mu_tilde"],
                            heldout_tensors["psi"], heldout_tensors["c0"])
            val_loss = float(mass_weighted_loss(pred_val, heldout_tensors["target"], mass_matrix_dense).item())

        rel_l2_list, rel_energy_list, eta_guar_list = [], [], []
        for mu, payoff, traj in heldout_trajs:
            pred = predict(net, mu, ops, stats, traj.initial_coefficients, traj.time_grid)
            rel_l2_list.append(relative_l2_error(pred, traj.coefficients, ops))
            rel_energy_list.append(relative_energy_error(pred, traj.coefficients, mu, ops))
            cert = compute_certificates(traj, pred, mu, ops)
            if cert.is_spd:
                eta_guar_list.append(cert.eta_guar)

        result = {
            "n_train": n_train, "train_time_s": t_train, "final_train_loss": final_train_loss,
            "val_loss": val_loss,
            "test_rel_l2_mean": float(np.mean(rel_l2_list)),
            "test_rel_energy_mean": float(np.mean(rel_energy_list)),
            "test_eta_guar_mean": float(np.mean(eta_guar_list)) if eta_guar_list else float("nan"),
        }
        results.append(result)
        print(f"  train_time={t_train:.1f}s  val_loss={val_loss:.4e}  "
              f"test_rel_l2={result['test_rel_l2_mean']:.4e}  "
              f"test_rel_energy={result['test_rel_energy_mean']:.4e}  "
              f"test_eta_guar={result['test_eta_guar_mean']:.4e}")

    # ── optional power-law fit E(N) ~= C * N^{-beta} ────────────────────
    Ns = np.array([r["n_train"] for r in results], dtype=float)
    Es = np.array([r["test_rel_l2_mean"] for r in results])
    beta, log_C = np.nan, np.nan
    if np.all(Es > 0):
        slope, intercept = np.polyfit(np.log(Ns), np.log(Es), 1)
        beta, log_C = -slope, intercept
        print(f"\nexp13 (learning curve): fitted E(N) ~= C * N^-beta over "
              f"{len(Ns)} points: beta={beta:.3f}, C={np.exp(log_C):.3e} "
              f"(only {len(Ns)} points -- a rough fit, not a tight one)")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    write_h5(
        RESULTS_DIR / "exp13_learning_curve.h5",
        {
            "n_train": Ns,
            "train_time_s": np.array([r["train_time_s"] for r in results]),
            "val_loss": np.array([r["val_loss"] for r in results]),
            "test_rel_l2_mean": Es,
            "test_rel_energy_mean": np.array([r["test_rel_energy_mean"] for r in results]),
            "test_eta_guar_mean": np.array([r["test_eta_guar_mean"] for r in results]),
        },
        attrs={"n_heldout": N_HELDOUT, "heldout_seed": HELDOUT_SEED, "n_epochs": N_EPOCHS_CURVE,
               "seed": SEED_CURVE, "beta_fit": beta, "C_fit": float(np.exp(log_C)) if log_C == log_C else float("nan"),
               "dropped_n_train_1600": "production_dataset.h5 only has 800 trajectories"},
    )

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(Ns, Es, "o-", label="test relative $L_2$")
    axes[0].plot(Ns, [r["val_loss"] for r in results], "s-", label="validation loss")
    if beta == beta:
        fit_line = np.exp(log_C) * Ns ** (-beta)
        axes[0].plot(Ns, fit_line, "k--", alpha=0.6, label=f"fit: $C N^{{-{beta:.2f}}}$")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("$N_{train}$"); axes[0].set_ylabel("error / loss")
    axes[0].set_title("exp13: learning curve"); axes[0].legend(fontsize=8)

    axes[1].plot(Ns, [r["test_eta_guar_mean"] for r in results], "^-", color="C2")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("$N_{train}$"); axes[1].set_ylabel(r"mean $\eta_{guar}$")
    axes[1].set_title("exp13: guaranteed certificate vs. $N_{train}$")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp13_learning_curve.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    header = ["$N_{train}$", "val. loss", "test rel $L_2$", "test rel energy", "mean $\\eta_{guar}$", "train time (s)"]
    rows = [[str(r["n_train"]), fmt(r["val_loss"]), fmt(r["test_rel_l2_mean"]), fmt(r["test_rel_energy_mean"]),
              fmt(r["test_eta_guar_mean"]), f"{r['train_time_s']:.1f}"] for r in results]
    write_latex_table(TABLES_DIR / "exp13_learning_curve.tex", header, rows)
    print(f"wrote {TABLES_DIR / 'exp13_learning_curve.tex'}")
    return results


# ── Part 2: residual-loss ablation ──────────────────────────────────────

def _build_ablation_dataset(trajs, ops, stats):
    tau_list, tau_norm_list, mu_tilde_list, psi_list, c0_list, target_list = [], [], [], [], [], []
    blocks = []
    cursor = 0
    for mu, payoff, traj in trajs:
        mu_tilde = normalize_mu(mu, stats)
        psi = normalize_psi(payoff_family.european_put(payoff.strike, ops), payoff.strike)
        n1 = len(traj.time_grid)
        for n, tau in enumerate(traj.time_grid):
            tau_list.append([tau]); tau_norm_list.append([normalize_tau(tau, mu.T)])
            mu_tilde_list.append(mu_tilde); psi_list.append(psi)
            c0_list.append(traj.initial_coefficients); target_list.append(traj.coefficients[n])
        blocks.append({"start": cursor, "end": cursor + n1, "mu": mu, "traj": traj,
                        "dt": traj.solver_metadata.dt, "bc_dofs": traj.solver_metadata.bc_dofs})
        cursor += n1
    tensors = {
        "tau": torch.tensor(np.array(tau_list)), "tau_norm": torch.tensor(np.array(tau_norm_list)),
        "mu_tilde": torch.tensor(np.array(mu_tilde_list)), "psi": torch.tensor(np.array(psi_list)),
        "c0": torch.tensor(np.array(c0_list)), "target": torch.tensor(np.array(target_list)),
    }
    return tensors, blocks


def _ablation_residual_terms(pred, blocks, ops):
    c_n_list, c_np1_list, fbar_list, mus, bc_dofs = [], [], [], [], None
    for b in blocks:
        block_pred = pred[b["start"]:b["end"]]
        traj = b["traj"]
        for n in range(len(traj.time_grid) - 1):
            c_n_list.append(block_pred[n]); c_np1_list.append(block_pred[n + 1])
            fbar_list.append(torch.tensor(traj.forcing_vectors[n])); mus.append(b["mu"])
        bc_dofs = b["bc_dofs"]
    c_n = torch.stack(c_n_list); c_np1 = torch.stack(c_np1_list); fbar = torch.stack(fbar_list)
    return c_n, c_np1, fbar, mus, bc_dofs, blocks[0]["dt"]


def train_mlp_hybrid(lambda_r, tensors, blocks, ops, mass_matrix_dense, n_dof, seed, n_epochs, lr):
    torch.manual_seed(seed)
    net = MLPCoeffNet(mu_dim=6, psi_dim=n_dof, hidden_dims=(128, 128, 128))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 1e-3)
    t0 = time.time()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        pred = net(tensors["tau"], tensors["tau_norm"], tensors["mu_tilde"], tensors["psi"], tensors["c0"])
        loss = mass_weighted_loss(pred, tensors["target"], mass_matrix_dense)
        if lambda_r > 0:
            c_n, c_np1, fbar, mus, bc_dofs, dt = _ablation_residual_terms(pred, blocks, ops)
            r = torch_cn_residual(c_n, c_np1, fbar, dt, mus, ops, bc_dofs)
            loss = loss + lambda_r * residual_loss(r, ops)
        loss.backward()
        optimizer.step()
        scheduler.step()
    net.eval()
    return net, time.time() - t0


def run_lambda_r_sweep(ops, box, stats):
    n_dof = ops.n_dof
    mass_matrix_dense = torch.tensor(np.array([ops.apply_M(e) for e in np.eye(n_dof)]).T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_samples = sample_sobol(box, N_TRAIN_ABL, seed=ABL_SEED_BASE)
        test_samples = sample_sobol(box, N_TEST_ABL, seed=ABL_SEED_BASE + 1)

    print(f"\nexp13 (lambda_R sweep): solving {N_TRAIN_ABL} train + {N_TEST_ABL} test "
          f"trajectories (ablation-scale, N_STEPS={N_STEPS_ABL}, shared across every lambda_R)...")
    t0 = time.time()
    train_trajs = [(mu, payoff, solve_cgmy(mu, payoff, FIXED_MESH, TimeGrid(n_steps=N_STEPS_ABL), mode="toeplitz"))
                   for mu, payoff in train_samples]
    test_trajs = [(mu, payoff, solve_cgmy(mu, payoff, FIXED_MESH, TimeGrid(n_steps=N_STEPS_ABL), mode="toeplitz"))
                  for mu, payoff in test_samples]
    print(f"  ...done in {time.time() - t0:.1f}s")

    tensors, blocks = _build_ablation_dataset(train_trajs, ops, stats)

    results = []
    for lambda_r in LAMBDA_R_VALUES:
        print(f"\nexp13 (lambda_R sweep): lambda_R={lambda_r:.1e} -- training "
              f"({N_EPOCHS_ABL} epochs)...")
        net, t_train = train_mlp_hybrid(
            lambda_r, tensors, blocks, ops, mass_matrix_dense, n_dof, SEED_ABL, N_EPOCHS_ABL, LR_ABL
        )

        rel_l2_list, rel_energy_list, max_res_list = [], [], []
        eta_sharp_list, eta_guar_list, effectivity_list = [], [], []
        n_not_spd = 0
        for mu, payoff, traj in test_trajs:
            mu_tilde = normalize_mu(mu, stats)
            psi = payoff_family.european_put(payoff.strike, ops)
            pred_raw = net.predict_trajectory(mu_tilde, psi, traj.initial_coefficients, traj.time_grid, mu.T)
            pred = project_bc(pred_raw, mu, traj.payoff, traj.mesh_information, traj.time_grid)

            rel_l2_list.append(relative_l2_error(pred, traj.coefficients, ops))
            rel_energy_list.append(relative_energy_error(pred, traj.coefficients, mu, ops))

            const = compute_stability_constants(mu, ops)
            if not const.is_spd:
                n_not_spd += 1
                continue
            traj_pred = wrap_as_trajectory(traj, pred)
            residuals = compute_all_residuals(traj_pred, mu, ops)
            dt = traj.solver_metadata.dt
            max_res_list.append(max(residual_norm(residuals[n], ops, traj.solver_metadata.bc_dofs)
                                     for n in range(traj.solver_metadata.n_steps)))
            es = compute_eta_sharp(residuals, mu, ops, dt)
            eg = compute_eta_guar(residuals, mu, ops, dt, const.alpha_certified,
                                   max_pcg_iterations=MAX_PCG_ITERATIONS, preconditioner=PRECONDITIONER)
            E_i = true_error_Ei(pred, traj.coefficients, ops)
            eta_sharp_list.append(es.eta)
            eta_guar_list.append(eg.eta)
            if E_i > 0:
                effectivity_list.append(eg.eta / E_i)

        result = {
            "lambda_R": lambda_r, "train_time_s": t_train, "n_not_spd": n_not_spd,
            "rel_l2_mean": float(np.mean(rel_l2_list)),
            "rel_energy_mean": float(np.mean(rel_energy_list)),
            "max_residual_mean": float(np.mean(max_res_list)) if max_res_list else float("nan"),
            "eta_sharp_mean": float(np.mean(eta_sharp_list)) if eta_sharp_list else float("nan"),
            "eta_guar_mean": float(np.mean(eta_guar_list)) if eta_guar_list else float("nan"),
            "effectivity_mean": float(np.mean(effectivity_list)) if effectivity_list else float("nan"),
        }
        results.append(result)
        print(f"  train_time={t_train:.1f}s  rel_l2={result['rel_l2_mean']:.4e}  "
              f"rel_energy={result['rel_energy_mean']:.4e}  max_residual={result['max_residual_mean']:.4e}  "
              f"eta_sharp={result['eta_sharp_mean']:.4e}  eta_guar={result['eta_guar_mean']:.4e}  "
              f"effectivity={result['effectivity_mean']:.3f}  not_spd={n_not_spd}/{N_TEST_ABL}")

    # ── Honesty check: does the certificate improve MONOTONICALLY with lambda_R? ──
    eta_guar_arr = np.array([r["eta_guar_mean"] for r in results])
    rel_l2_arr = np.array([r["rel_l2_mean"] for r in results])
    cert_monotonic_improving = bool(np.all(np.diff(eta_guar_arr) <= 1e-9))
    accuracy_improved = bool(rel_l2_arr[-1] < rel_l2_arr[0])
    print("\n" + "=" * 78)
    print("exp13 (lambda_R sweep): Task 12's required honesty check")
    print("=" * 78)
    print(f"  prediction accuracy improved from lambda_R=0 to the largest lambda_R tested: "
          f"{accuracy_improved} (rel_l2: {rel_l2_arr[0]:.3e} -> {rel_l2_arr[-1]:.3e})")
    print(f"  guaranteed certificate (eta_guar) improves MONOTONICALLY with lambda_R: "
          f"{cert_monotonic_improving}")
    if cert_monotonic_improving:
        print("  NOTE: this run's own data shows monotonic certificate improvement -- report "
              "exactly that, not the non-monotonic finding, since the task requires reporting "
              "what THIS run's numbers actually show, not a pre-supposed conclusion.")
    else:
        print("  CONFIRMED (this run): residual regularization does NOT monotonically improve "
              "the certificate, even where it improves raw predictive accuracy -- do not claim "
              "'residual regularization improves certification.' Report a Pareto tradeoff instead.")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    write_h5(
        RESULTS_DIR / "exp13_lambda_r_sweep.h5",
        {
            "lambda_R": np.array([r["lambda_R"] for r in results]),
            "rel_l2_mean": rel_l2_arr,
            "rel_energy_mean": np.array([r["rel_energy_mean"] for r in results]),
            "max_residual_mean": np.array([r["max_residual_mean"] for r in results]),
            "eta_sharp_mean": np.array([r["eta_sharp_mean"] for r in results]),
            "eta_guar_mean": eta_guar_arr,
            "effectivity_mean": np.array([r["effectivity_mean"] for r in results]),
            "n_not_spd": np.array([r["n_not_spd"] for r in results]),
        },
        attrs={"n_train": N_TRAIN_ABL, "n_test": N_TEST_ABL, "n_steps": N_STEPS_ABL,
               "n_epochs": N_EPOCHS_ABL, "seed": SEED_ABL,
               "cert_monotonic_improving": cert_monotonic_improving, "accuracy_improved": accuracy_improved},
    )

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lam_plot = [v if v > 0 else 1e-10 for v in LAMBDA_R_VALUES]  # log-scale placeholder for 0
    axes[0].plot(lam_plot, rel_l2_arr, "o-", label="prediction error (rel $L_2$)")
    axes[0].plot(lam_plot, [r["rel_energy_mean"] for r in results], "s-", label="energy error")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\lambda_R$ (0 plotted at $10^{-10}$)"); axes[0].set_ylabel("error")
    axes[0].set_title("exp13: prediction/energy error vs. $\\lambda_R$"); axes[0].legend(fontsize=8)

    axes[1].plot(lam_plot, [r["eta_sharp_mean"] for r in results], "^-", label=r"$\eta_{sharp}$")
    axes[1].plot(lam_plot, eta_guar_arr, "v-", label=r"$\eta_{guar}$")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\lambda_R$ (0 plotted at $10^{-10}$)"); axes[1].set_ylabel("certificate")
    axes[1].set_title("exp13: certificates vs. $\\lambda_R$"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp13_lambda_r_sweep.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    header = ["$\\lambda_R$", "pred. err.", "energy err.", "residual norm", "$\\eta_{sharp}$",
               "$\\eta_{guar}$", "effectivity"]
    rows = [[fmt(r["lambda_R"]), fmt(r["rel_l2_mean"]), fmt(r["rel_energy_mean"]),
              fmt(r["max_residual_mean"]), fmt(r["eta_sharp_mean"]), fmt(r["eta_guar_mean"]),
              f"{r['effectivity_mean']:.3f}"] for r in results]
    write_latex_table(TABLES_DIR / "exp13_lambda_r_sweep.tex", header, rows)
    print(f"wrote {TABLES_DIR / 'exp13_lambda_r_sweep.tex'}")
    return results


def main() -> None:
    ops = get_ops()
    box, stats = get_box_and_stats()
    run_learning_curve(ops, box, stats)
    run_lambda_r_sweep(ops, box, stats)


if __name__ == "__main__":
    main()
