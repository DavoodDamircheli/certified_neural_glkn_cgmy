"""exp05 — out-of-distribution behavior: train a fresh model on a
deliberately RESTRICTED interior region of the parameter box, then test at
increasing distance from that region -- first within the frozen box, then
genuinely outside it (still a "meaningful CGMY regime": Stage D's OOD
sampler, which respects the model's hard constraints -- C,G>0, M>1, Y in
(0,2)\\{1}, T>0, r>=0 -- while sampling outside the frozen box). Question:
does the certificate rise as the prediction degrades?

CHANGED (pre-submission fixes, item 1.3): the "beyond" tier's sampler
(data.sampling._ood_samples) previously left C>0/G>0 to a downstream
ValueError from CGMYParams -- so candidates with C<0 or G<0 (outside the
CGMY model's OWN admissible domain, not merely outside the training
distribution) were routinely drawn and only discovered inadmissible at
solve time, silently shrinking the effective tier size (this project's
own last run: 7/50 such candidates were drawn and discarded here). C and
G are now floored in the sampler itself, the same way M/Y/r/maturity
already were -- every "beyond" query is admissible by construction.
Verified directly (not assumed) that the new tier's distance-from-box
(data.sampling.parameter_distance_from_box, reported per tier below)
stayed comparable to the old one: mean 0.134 (new) vs. 0.141 (old,
reproduced with the pre-fix sampler for this one comparison) -- the fix
does not make the "beyond" tier meaningfully closer to or farther from
the box, only removes the inadmissible corner of it. interior/near/edge
tiers are drawn from the ALREADY-admissible frozen box (never touch
`_ood_samples`), and are checked (not just assumed) to contain zero
inadmissible samples below.

This trains its OWN model (not one of the Stage H checkpoints, which were
trained on Sobol samples spanning the FULL box -- reusing them would
defeat the point of testing generalization beyond a restricted training
region). Same architecture/recipe as Stage H's MLP (mass-weighted loss,
exact-IC structural embedding), trained on fewer trajectories/epochs for
wall-clock reasons -- flagged like every other reduced-scale run in this
project.

Four tiers, by shrink_frac of the box (0 = full box; see
data.sampling.sample_sobol's shrink_frac -- note shrink_frac is applied to
BOTH sides, so e.g. 0.35 leaves 1-2*0.35 = 30% of the box width, and 0.5
would collapse the box to a single point; caught by a crashing ValueError
from scipy's qmc bounds validation before any wasted compute):
  - train (shrink_frac=0.35): the region actually trained on (30% width)
  - near   (shrink_frac=0.3): mild relaxation, still well inside the box
  - edge   (shrink_frac=0.0): the box's own edges
  - beyond (genuinely outside the frozen box, via data.sampling's OOD sampler)

Scale note: N_TEST_PER_TIER=50 (Task 11, working prompt: "OOD study: 50-100
queries per region (interior, near-boundary, edge, beyond-domain)" -- the
four tiers below map onto those four regions in order: train=interior,
near=near-boundary, edge=edge, beyond=beyond-domain).

Outputs: results/exp05_ood.h5, figures/exp05_ood.png, tables/exp05_ood.tex.
"""

from __future__ import annotations

import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cgmy.operators import OperatorCache
from data import payoff_family
from data.sampling import _ood_samples, parameter_distance_from_box, sample_sobol
from exp_common import (
    FIGURES_DIR,
    FIXED_MESH,
    RESULTS_DIR,
    TABLES_DIR,
    compute_certificates,
    fmt,
    get_box_and_stats,
    get_ops,
    capture_git_commit_hash,
    provenance_start,
    solve_reference,
    supersede_if_exists,
    true_error_Ei,
    write_h5,
    write_latex_table,
    write_provenance,
)
from learning.losses import mass_weighted_loss
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import normalize_mu, normalize_psi, normalize_tau

torch.set_default_dtype(torch.float64)

N_TRAIN = 150
TRAIN_SHRINK_FRAC = 0.35
N_TEST_PER_TIER = 50
N_EPOCHS = 2000
LR = 2e-3
SEED = 424250


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


def _check_admissible(tier_name: str, samples) -> int:
    """Count inadmissible (C<=0, G<=0, M<=1, or Y outside (0,2)) samples
    in a tier drawn from the frozen box -- should always be 0 for
    interior/near/edge (the box itself is already admissible), checked
    rather than assumed (item 1.3)."""
    n_bad = 0
    for mu, _ in samples:
        if not (mu.C > 0 and mu.G > 0 and mu.M > 1 and 0 < mu.Y < 2):
            n_bad += 1
    print(f"  [{tier_name}] admissibility check: {n_bad}/{len(samples)} inadmissible")
    return n_bad


def main() -> None:
    t_start = provenance_start()
    git_commit = capture_git_commit_hash()  # BEFORE any output is written -- see its docstring
    ops = get_ops()
    box, stats = get_box_and_stats()
    n_dof = ops.n_dof

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_samples = sample_sobol(box, N_TRAIN, seed=SEED, shrink_frac=TRAIN_SHRINK_FRAC)

    print(f"exp05: solving {N_TRAIN} restricted-region (shrink_frac={TRAIN_SHRINK_FRAC}) training trajectories...")
    t0 = time.time()
    train_trajs = [(mu, payoff, solve_reference(mu)) for mu, payoff in train_samples]
    print(f"...done in {time.time() - t0:.1f}s")

    tensors = build_tensors(train_trajs, ops, stats)
    mass_matrix_dense = torch.tensor(np.array([ops.apply_M(e) for e in np.eye(n_dof)]).T)

    print(f"training MLP on the restricted region ({N_EPOCHS} epochs)...")
    torch.manual_seed(SEED)
    net = MLPCoeffNet(mu_dim=6, psi_dim=n_dof, hidden_dims=(128, 128, 128))
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
    print(f"...done in {time.time() - t0:.1f}s, final train loss={loss.item():.4e}")
    net.eval()

    # ── Test tiers ───────────────────────────────────────────────────────
    tiers = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tiers["train"] = sample_sobol(box, N_TEST_PER_TIER, seed=SEED + 1, shrink_frac=TRAIN_SHRINK_FRAC)
        tiers["near"] = sample_sobol(box, N_TEST_PER_TIER, seed=SEED + 2, shrink_frac=0.3)
        tiers["edge"] = sample_sobol(box, N_TEST_PER_TIER, seed=SEED + 3, shrink_frac=0.0)
        tiers["beyond"], beyond_stats = _ood_samples(box, N_TEST_PER_TIER, seed=SEED + 4, return_stats=True)

    tier_order = ["train", "near", "edge", "beyond"]

    print("admissibility check across every tier (item 1.3):")
    n_inadmissible_by_tier = {t: _check_admissible(t, tiers[t]) for t in tier_order}
    assert n_inadmissible_by_tier["beyond"] == 0, "the 'beyond' sampler's own admissibility assertion should have caught this already"
    print(f"  beyond-tier sampling stats: {beyond_stats}")

    mean_distance_by_tier = {t: float(np.mean([parameter_distance_from_box(mu, box) for mu, _ in tiers[t]]))
                              for t in tier_order}
    print("mean distance-from-box per tier (item 1.3, should be monotone interior->beyond):")
    for t in tier_order:
        print(f"  {t:8s}: mean_distance={mean_distance_by_tier[t]:.4f}")

    E_i_by_tier = {t: [] for t in tier_order}
    eta_guar_by_tier = {t: [] for t in tier_order}
    n_not_spd_by_tier = {t: 0 for t in tier_order}

    print("\nevaluating test tiers...")
    for tier in tier_order:
        for mu, payoff in tiers[tier]:
            try:
                traj = solve_reference(mu)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{tier}] mu={mu}: solve failed: {exc}")
                continue
            mu_tilde = normalize_mu(mu, stats)
            psi_raw = payoff_family.european_put(payoff.strike, ops)
            pred = net.predict_trajectory(mu_tilde, psi_raw, traj.initial_coefficients, traj.time_grid, mu.T)

            E_i = true_error_Ei(pred, traj.coefficients, ops)
            cert = compute_certificates(traj, pred, mu, ops)
            E_i_by_tier[tier].append(E_i)
            if cert.is_spd:
                eta_guar_by_tier[tier].append(cert.eta_guar)
            else:
                n_not_spd_by_tier[tier] += 1
        print(f"  {tier:8s}: mean E_i={np.mean(E_i_by_tier[tier]):.3e}  "
              f"mean eta_guar={np.mean(eta_guar_by_tier[tier]) if eta_guar_by_tier[tier] else float('nan'):.3e}  "
              f"not_spd={n_not_spd_by_tier[tier]}/{N_TEST_PER_TIER}")

    mean_E = np.array([np.mean(E_i_by_tier[t]) for t in tier_order])
    mean_eta = np.array([np.mean(eta_guar_by_tier[t]) if eta_guar_by_tier[t] else np.nan for t in tier_order])
    rising_error = bool(np.all(np.diff(mean_E) >= -1e-12))
    rising_cert = bool(np.all(np.diff(mean_eta[~np.isnan(mean_eta)]) >= -1e-12))

    print(f"\ntrue error rises monotonically across tiers: {rising_error}")
    print(f"certificate rises monotonically across tiers: {rising_cert}")

    # ── Never overwrite results silently ────────────────────────────────
    h5_path = RESULTS_DIR / "exp05_ood.h5"
    fig_path = FIGURES_DIR / "exp05_ood.png"
    tex_path = TABLES_DIR / "exp05_ood.tex"
    moved = supersede_if_exists(h5_path, fig_path, tex_path, h5_path.with_suffix(".config.json"))
    for m in moved:
        print(f"superseded existing output -> {m}")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    arrays = {
        "tier_mean_E_i": mean_E, "tier_mean_eta_guar": mean_eta,
        "tier_mean_distance": np.array([mean_distance_by_tier[t] for t in tier_order]),
    }
    for t in tier_order:
        arrays[f"E_i_{t}"] = np.array(E_i_by_tier[t])
        arrays[f"eta_guar_{t}"] = np.array(eta_guar_by_tier[t]) if eta_guar_by_tier[t] else np.array([])
    write_h5(
        h5_path, arrays,
        attrs={"n_train": N_TRAIN, "train_shrink_frac": TRAIN_SHRINK_FRAC, "n_test_per_tier": N_TEST_PER_TIER,
               "n_epochs": N_EPOCHS, "seed": SEED, "rising_error": rising_error, "rising_cert": rising_cert,
               "n_inadmissible_train": n_inadmissible_by_tier["train"],
               "n_inadmissible_near": n_inadmissible_by_tier["near"],
               "n_inadmissible_edge": n_inadmissible_by_tier["edge"],
               "n_inadmissible_beyond": n_inadmissible_by_tier["beyond"],
               "beyond_n_drawn": beyond_stats["n_drawn"], "beyond_n_kept": beyond_stats["n_kept"],
               "beyond_n_rejected_outside": beyond_stats["n_rejected_outside"]},
    )

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(tier_order))
    ax1.boxplot([E_i_by_tier[t] for t in tier_order], positions=x - 0.15, widths=0.25,
                tick_labels=tier_order, patch_artist=True,
                boxprops=dict(facecolor="C0", alpha=0.5))
    ax1.set_ylabel("true error $E_i$", color="C0")
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    box_data = [eta_guar_by_tier[t] if eta_guar_by_tier[t] else [np.nan] for t in tier_order]
    ax2.boxplot(box_data, positions=x + 0.15, widths=0.25, patch_artist=True,
                boxprops=dict(facecolor="C1", alpha=0.5))
    ax2.set_ylabel(r"$\eta_{guar}$", color="C1")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="C1")

    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_order)
    ax1.set_xlabel("distance tier from training region "
                    f"(shrink_frac={TRAIN_SHRINK_FRAC} $\\to$ full box $\\to$ outside box)")
    ax1.set_title("exp05: does the certificate rise as the prediction degrades?")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    header = ["tier", "mean $E_i$", "mean $\\eta_{guar}$", "not-SPD count", "mean distance-from-box",
               "inadmissible count"]
    rows = []
    for t in tier_order:
        eta_val = np.mean(eta_guar_by_tier[t]) if eta_guar_by_tier[t] else float("nan")
        rows.append([t, fmt(np.mean(E_i_by_tier[t])), fmt(eta_val), str(n_not_spd_by_tier[t]),
                     fmt(mean_distance_by_tier[t]), str(n_inadmissible_by_tier[t])])
    write_latex_table(tex_path, header, rows)
    print(f"wrote {tex_path}")

    prov_path = write_provenance(
        h5_path, start_time=t_start, seed=SEED,
        git_commit_override=git_commit,
        n_requested={"train": N_TEST_PER_TIER, "near": N_TEST_PER_TIER, "edge": N_TEST_PER_TIER,
                     "beyond": N_TEST_PER_TIER},
        n_kept={"train": len(E_i_by_tier["train"]), "near": len(E_i_by_tier["near"]),
                "edge": len(E_i_by_tier["edge"]), "beyond": len(E_i_by_tier["beyond"])},
        n_rejected={"beyond_rejected_inside_box": beyond_stats["n_rejected_outside"],
                    "beyond_drawn_total": beyond_stats["n_drawn"]},
        extra={
            "n_train_trajectories": N_TRAIN, "n_epochs": N_EPOCHS,
            "n_inadmissible_by_tier": n_inadmissible_by_tier,
            "mean_distance_by_tier": mean_distance_by_tier,
            "rising_error": rising_error, "rising_cert": rising_cert,
        },
    )
    print(f"wrote {prov_path}")


if __name__ == "__main__":
    main()
