"""Stage H item 2: train final versions of all four models on the production
dataset (from scripts/generate_production_dataset.py), saving checkpoints
with full reproducibility metadata (config, git SHA + dirty flag, dataset
manifest hash, RNG seed) via common.checkpoint.

POD-Galerkin has no gradient-trained weights; its "checkpoint" is the POD
basis itself (modes, singular values, mean) plus the same metadata fields,
saved separately (it is not an nn.Module and has no state_dict()).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from cgmy.operators import OperatorCache
from cgmy.reference_solver import Mesh, TimeGrid
from common.checkpoint import (
    build_data_metadata,
    build_discretization_metadata,
    dataset_manifest_hash,
    get_git_sha_and_dirty,
    save_checkpoint,
)
from data import payoff_family
from data.sampling import load_parameter_box, sample_sobol
from cgmy_bspline.projection import eval_solution
from learning.deeponet_coeff import DeepONetCoeffNet
from learning.fno import FNO1d
from learning.losses import mass_weighted_loss
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import normalize_mu, normalize_psi, normalize_tau, stats_from_box
from learning.pod import PODGalerkinROM, compute_pod_basis
from learning.projected_fno import ProjectedFNO
from generate_production_dataset import FIXED_MESH, TIME_GRID, load_all_train_samples, build_val_set

torch.set_default_dtype(torch.float64)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
N_EPOCHS = 3000
LR = 2e-3
SEED = 0


def _build_tensors(trajs, ops, stats):
    tau_list, tau_norm_list, mu_tilde_list, psi_list, c0_list, target_list = [], [], [], [], [], []
    for mu, payoff, traj in trajs:
        mu_tilde = normalize_mu(mu, stats)
        psi = normalize_psi(payoff_family.european_put(payoff.strike, ops), payoff.strike)
        for n, tau in enumerate(traj.time_grid):
            tau_list.append([tau])
            tau_norm_list.append([normalize_tau(tau, mu.T)])
            mu_tilde_list.append(mu_tilde)
            psi_list.append(psi)
            c0_list.append(traj.initial_coefficients)
            target_list.append(traj.coefficients[n])
    return {
        "tau": torch.tensor(np.array(tau_list)),
        "tau_norm": torch.tensor(np.array(tau_norm_list)),
        "mu_tilde": torch.tensor(np.array(mu_tilde_list)),
        "psi": torch.tensor(np.array(psi_list)),
        "c0": torch.tensor(np.array(c0_list)),
        "target": torch.tensor(np.array(target_list)),
    }


def train_gradient_model(model_ctor, model_kwargs, tensors, mass_matrix_dense, seed=SEED):
    torch.manual_seed(seed)
    net = model_ctor(**model_kwargs)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 1e-3)

    t0 = time.time()
    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = net(tensors["tau"], tensors["tau_norm"], tensors["mu_tilde"], tensors["psi"], tensors["c0"])
        loss = mass_weighted_loss(pred, tensors["target"], mass_matrix_dense)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return net, time.time() - t0, loss.item()


def train_fno(train_trajs, ops, mesh, stats, mu_dim=6, seed=SEED, width=32, n_modes=16, n_layers=4):
    """ProjectedFNO's predict_trajectory uses a NumPy (non-differentiable)
    Pi_h projection, so it cannot share MLP/DeepONet's coefficient-space
    training loop. Trains directly on the raw FNO's grid output instead,
    against a grid-space SHIFTED target (V_grid(tau) - V_grid(0)) -- the
    same shift-trick predict_trajectory uses for its exact-IC property, so
    matching this target in grid space controls the coefficient-space error
    after Pi_h (a bounded linear projection) at inference. Verified
    differentiable and correctly decreasing on a single-trajectory overfit
    check before being used here (see Stage H report)."""
    grid_nodes = np.linspace(mesh.x_min, mesh.x_max, mesh.N + 1)
    grid_info = {"N": mesh.N, "h": ops.h, "x_min": mesh.x_min}

    tau_norm_list, mu_tilde_list, psi_grid_list, target_list = [], [], [], []
    for mu, payoff, traj in train_trajs:
        mu_tilde = normalize_mu(mu, stats)
        psi = payoff_family.european_put(payoff.strike, ops)
        psi_grid = eval_solution(psi, grid_info, mesh.p, grid_nodes)
        V_true = np.array([eval_solution(traj.coefficients[n], grid_info, mesh.p, grid_nodes)
                            for n in range(len(traj.time_grid))])
        V_shift_true = V_true - V_true[0]
        for n, tau in enumerate(traj.time_grid):
            tau_norm_list.append([normalize_tau(tau, mu.T)])
            mu_tilde_list.append(mu_tilde)
            psi_grid_list.append(psi_grid)
            target_list.append(V_shift_true[n])

    tau_norm_t = torch.tensor(np.array(tau_norm_list))
    mu_tilde_t = torch.tensor(np.array(mu_tilde_list))
    psi_grid_t = torch.tensor(np.array(psi_grid_list))
    target_t = torch.tensor(np.array(target_list))

    torch.manual_seed(seed)
    fno = FNO1d(in_channels=1 + 1 + mu_dim, out_channels=1, width=width, n_modes=n_modes, n_layers=n_layers)
    net = ProjectedFNO(fno, mesh, ops, mu_dim=mu_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # Per-trajectory raw(0) values are needed to shift each row correctly;
    # since rows are flattened across trajectories, shift per-trajectory
    # blocks rather than globally. Every trajectory here shares the same
    # TimeGrid(n_steps=...), so block sizes are uniform.
    block_sizes = {len(traj.time_grid) for _, _, traj in train_trajs}
    if len(block_sizes) != 1:
        raise ValueError(
            f"train_fno's per-trajectory shift assumes every trajectory has the same "
            f"number of time steps; got sizes {block_sizes}."
        )
    steps_per_traj = block_sizes.pop()
    n_traj = len(train_trajs)
    n_grid = psi_grid_t.shape[1]

    # Mini-batched over TRAJECTORIES (not individual rows): each mini-batch
    # must contain every sampled trajectory's COMPLETE block (including its
    # own tau=0 row) for the per-trajectory shift trick to stay correct --
    # random individual-row batching would mix rows from trajectories whose
    # tau=0 output isn't in the same batch, breaking the shift.
    #
    # This replaces two earlier, separate problems found while profiling
    # the original full-batch version (which trained on all n_traj
    # trajectories in one forward/backward call per "epoch"):
    #   1. A per-block PYTHON LOOP inside that one big call caused
    #      unbounded-looking RSS growth (~14GB) -- likely PyTorch autograd
    #      reference cycles outpacing the cyclic GC. (Already fixed once,
    #      by vectorizing the shift via reshape -- see git history.)
    #   2. The deeper issue that fix did not address: doing only ONE
    #      gradient update per full pass over ~49k rows is extremely
    #      update-inefficient (3000 total gradient steps for the whole
    #      training run), which is why it took 6.6 hours even after fix
    #      #1. Measured directly: full-batch cost ~7.8s/epoch; mini-batched
    #      (trajectory groups of ~8, ~488 rows/batch) cost ~4.8s/epoch for
    #      an entire pass over all trajectories (~95 gradient steps instead
    #      of 1) -- i.e. ~95x more gradient updates per second of wall time.
    #      N_FNO_EPOCHS below is chosen so total gradient steps (n_epochs *
    #      n_traj/traj_batch_size) comfortably exceeds the original 3000.
    traj_batch_size = 8
    n_fno_epochs = 60  # ~60 * ceil(n_traj/8) total gradient steps, e.g. ~7500 for n_traj=800
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_fno_epochs, eta_min=LR * 1e-3)

    tau_norm_blocks = tau_norm_t.view(n_traj, steps_per_traj, 1)
    mu_tilde_blocks = mu_tilde_t.view(n_traj, steps_per_traj, mu_dim)
    psi_grid_blocks = psi_grid_t.view(n_traj, steps_per_traj, n_grid)
    target_blocks = target_t.view(n_traj, steps_per_traj, n_grid)

    rng = np.random.default_rng(seed)
    t0 = time.time()
    for _ in range(n_fno_epochs):
        perm = torch.as_tensor(rng.permutation(n_traj), dtype=torch.long)
        for start in range(0, n_traj, traj_batch_size):
            idx = perm[start:start + traj_batch_size]
            n_batch_traj = len(idx)

            optimizer.zero_grad()
            raw = net.raw_grid_output(
                tau_norm_blocks[idx].reshape(-1, 1),
                mu_tilde_blocks[idx].reshape(-1, mu_dim),
                psi_grid_blocks[idx].reshape(-1, n_grid),
            )
            raw_b = raw.view(n_batch_traj, steps_per_traj, n_grid)
            shifted = (raw_b - raw_b[:, :1, :]).reshape(-1, n_grid)
            target_mb = target_blocks[idx].reshape(-1, n_grid)
            loss = ((shifted - target_mb) ** 2).mean()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return net, time.time() - t0, loss.item()

    t0 = time.time()
    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        raw = net.raw_grid_output(tau_norm_t, mu_tilde_t, psi_grid_t)
        raw_blocks = raw.view(n_traj, steps_per_traj, n_grid)
        shifted = (raw_blocks - raw_blocks[:, :1, :]).reshape(n_traj * steps_per_traj, n_grid)
        loss = ((shifted - target_t) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return net, time.time() - t0, loss.item()


def evaluate_model(model, eval_trajs, ops, stats):
    l2_errs = []
    for mu, payoff, traj in eval_trajs:
        mu_tilde = normalize_mu(mu, stats)
        psi = payoff_family.european_put(payoff.strike, ops)
        pred = model.predict_trajectory(mu_tilde, psi, traj.initial_coefficients, traj.time_grid, mu.T)
        diff = pred - traj.coefficients
        l2_errs.extend(np.sqrt(d @ ops.apply_M(d)) for d in diff)
    return float(np.mean(l2_errs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=str(REPO_ROOT / "data" / "production_dataset.h5"))
    parser.add_argument("--n-train", type=int, required=True,
                         help="How many samples from --dataset to use (the tranche size the "
                              "learning curve chose).")
    parser.add_argument("--n-val", type=int, default=30)
    args = parser.parse_args()

    box = load_parameter_box(CONFIG_PATH)
    stats = stats_from_box(box)
    ops = OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)
    n_dof = ops.n_dof
    config_snapshot = CONFIG_PATH.read_text()
    # Task 5 (working prompt): the REALIZED mesh/time-grid/box actually used
    # here, recorded explicitly and checked at load time -- never inferred
    # from config_snapshot alone (see common.checkpoint's module docstring
    # for why that already went stale once in this project's history).
    discretization = build_discretization_metadata(FIXED_MESH, TIME_GRID)
    data_box = build_data_metadata(box, box.strike)

    sha, dirty = get_git_sha_and_dirty()
    print(f"git SHA: {sha}  (dirty: {dirty})")
    if dirty:
        print("WARNING: working tree has uncommitted changes -- this SHA does not fully "
              "determine the code that produced these checkpoints. Commit before treating "
              "these checkpoints as reproducible from the SHA alone.")

    print(f"loading {args.n_train} training trajectories from {args.dataset}...")
    train_trajs = load_all_train_samples(Path(args.dataset), args.n_train)
    print(f"building {args.n_val} validation trajectories...")
    val_trajs = build_val_set(args.n_val)

    tensors = _build_tensors(train_trajs, ops, stats)
    mass_matrix_dense = torch.tensor(np.array([ops.apply_M(e) for e in np.eye(n_dof)]).T)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    # --- MLP ---
    print("\ntraining MLPCoeffNet...")
    mlp_kwargs = {"mu_dim": 6, "psi_dim": n_dof, "hidden_dims": (128, 128, 128)}
    mlp, mlp_time, mlp_loss = train_gradient_model(MLPCoeffNet, mlp_kwargs, tensors, mass_matrix_dense)
    mlp_val_err = evaluate_model(mlp, val_trajs, ops, stats)
    meta = save_checkpoint(
        CHECKPOINTS_DIR / "mlp_final.pt", mlp, "MLPCoeffNet", mlp_kwargs,
        config_snapshot, SEED, args.dataset,
        discretization=discretization, data_box=data_box,
        extra={"n_train": args.n_train, "train_time_s": mlp_time, "final_train_loss": mlp_loss,
               "val_l2_error": mlp_val_err},
    )
    print(f"  train_time={mlp_time:.1f}s  val_L2={mlp_val_err:.4e}  -> checkpoints/mlp_final.pt")
    summary.append(("MLPCoeffNet", mlp_val_err, mlp_time))

    # --- DeepONet ---
    print("\ntraining DeepONetCoeffNet...")
    deeponet_kwargs = {"mu_dim": 6, "psi_dim": n_dof, "latent_dim": 64,
                        "branch_hidden": (128, 128), "trunk_hidden": (64, 64)}
    deeponet, deeponet_time, deeponet_loss = train_gradient_model(
        DeepONetCoeffNet, deeponet_kwargs, tensors, mass_matrix_dense
    )
    deeponet_val_err = evaluate_model(deeponet, val_trajs, ops, stats)
    save_checkpoint(
        CHECKPOINTS_DIR / "deeponet_final.pt", deeponet, "DeepONetCoeffNet", deeponet_kwargs,
        config_snapshot, SEED, args.dataset,
        discretization=discretization, data_box=data_box,
        extra={"n_train": args.n_train, "train_time_s": deeponet_time, "final_train_loss": deeponet_loss,
               "val_l2_error": deeponet_val_err},
    )
    print(f"  train_time={deeponet_time:.1f}s  val_L2={deeponet_val_err:.4e}  -> checkpoints/deeponet_final.pt")
    summary.append(("DeepONetCoeffNet", deeponet_val_err, deeponet_time))

    # --- POD-Galerkin ROM ---
    print("\nbuilding POD-Galerkin ROM...")
    t0 = time.time()
    snapshots = np.concatenate([t.coefficients for _, _, t in train_trajs], axis=0)
    n_modes = min(32, snapshots.shape[0] - 1)
    basis = compute_pod_basis(snapshots, n_modes=n_modes)
    pod_time = time.time() - t0
    rom = PODGalerkinROM(basis, ops, stats=stats)
    pod_val_err = evaluate_model(rom, val_trajs, ops, stats)
    pod_ckpt_path = CHECKPOINTS_DIR / "pod_final.npz"
    np.savez(
        pod_ckpt_path, modes=basis.modes, singular_values=basis.singular_values, mean=basis.mean,
        n_modes=n_modes, git_sha=sha, git_dirty=dirty, seed=SEED, n_train=args.n_train,
        val_l2_error=pod_val_err, train_time_s=pod_time,
        # Task 5 (working prompt): explicit, checked discretization/data
        # provenance -- see common.checkpoint's module docstring.
        discretization_json=json.dumps(discretization), data_box_json=json.dumps(data_box),
        dataset_manifest_hash=dataset_manifest_hash(args.dataset), dataset_path=str(args.dataset),
        config_snapshot=config_snapshot,
    )
    print(f"  n_modes={n_modes}  build_time={pod_time:.2f}s  val_L2={pod_val_err:.4e}  -> checkpoints/pod_final.npz")
    summary.append(("PODGalerkinROM", pod_val_err, pod_time))

    # --- Projected FNO ---
    print("\ntraining ProjectedFNO (grid-space shifted-target loss -- see module docstring)...")
    fno_net, fno_time, fno_loss = train_fno(train_trajs, ops, FIXED_MESH, stats)
    fno_val_err = evaluate_model(fno_net, val_trajs, ops, stats)
    fno_kwargs = {"mu_dim": 6}
    torch.save({
        "fno_state_dict": fno_net.fno.state_dict(),
        "fno_init_kwargs": {"in_channels": 1 + 1 + 6, "out_channels": 1, "width": 32, "n_modes": 16, "n_layers": 4},
        "mu_dim": 6,
        "mesh": {"N": FIXED_MESH.N, "p": FIXED_MESH.p, "x_min": FIXED_MESH.x_min, "x_max": FIXED_MESH.x_max},
        # Task 5 (working prompt): explicit, checked discretization/data
        # provenance (see common.checkpoint's module docstring) -- "mesh"
        # above is kept for backward compatibility with existing loaders,
        # "discretization" is the one load-time compatibility checks use.
        "discretization": discretization, "data_box": data_box,
        "config_snapshot": config_snapshot, "git_sha": sha, "git_dirty": dirty,
        "dataset_manifest_hash": dataset_manifest_hash(args.dataset), "dataset_path": args.dataset,
        "seed": SEED, "n_train": args.n_train, "train_time_s": fno_time,
        "final_train_loss": fno_loss, "val_l2_error": fno_val_err,
    }, CHECKPOINTS_DIR / "projected_fno_final.pt")
    print(f"  train_time={fno_time:.1f}s  val_L2={fno_val_err:.4e}  -> checkpoints/projected_fno_final.pt")
    summary.append(("ProjectedFNO", fno_val_err, fno_time))

    print("\n" + "=" * 70)
    print("FINAL MODELS SUMMARY")
    print("=" * 70)
    for name, err, t in summary:
        print(f"{name:20s}  val_L2={err:10.4e}  time={t:8.1f}s")
    print(f"\ngit SHA embedded in all checkpoints: {sha} (dirty={dirty})")


if __name__ == "__main__":
    main()
