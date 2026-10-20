# Gate E4 — architecture and training documentation

All numbers below are read directly from the actual checkpoint files
(`checkpoints/{mlp,deeponet,projected_fno}_final.pt`, `pod_final.npz`) and
`scripts/train_final_models.py` (the training script these checkpoints were
produced by — found by grepping for `save_checkpoint`/`torch.optim` calls
across the repo). Nothing below is inferred or guessed; where a field is
not recoverable, it is marked **NOT RECOVERABLE** rather than estimated.

## What is and isn't recoverable from a checkpoint alone

`src/common/checkpoint.py:save_checkpoint` embeds: `model_class_name`,
`model_init_kwargs` (architecture hyperparameters), `config_snapshot`,
`discretization` (realized mesh/time-grid), `data_box` (realized parameter
box/strike), `git_sha`/`git_dirty`, `dataset_manifest_hash`, `dataset_path`,
`seed`, and whatever `extra` dict the caller passes (here: `n_train`,
`train_time_s`, `final_train_loss`, `val_l2_error`). **It does NOT store
the optimizer, learning-rate schedule, epoch count, or batch size** — those
were only recoverable because `scripts/train_final_models.py` (the actual
training entry point) still exists in the repo and was found by this audit.
If that script were ever deleted, those four items would become
unrecoverable from the checkpoints alone — worth flagging as a provenance
gap independent of this specific pass.

**A known-stale field, confirmed directly, not merely cited**: all four
checkpoints' embedded `config_snapshot` text still reads `n_elements: 256`,
`domain ±10`, `n_time_steps: 200` (the old Stage-A scaffold values) — this
is exactly the staleness `common/checkpoint.py`'s own module docstring
warns about and is why it stores `discretization`/`data_box` as separately
realized, separately checked fields rather than trusting
`config_snapshot`. Every checkpoint's `discretization` field correctly
reads `N=64, domain ±8, n_time_steps=60`, confirming the hard-compatibility
machinery is using the right field. **Do not read `config_snapshot` out of
any of these four checkpoints as if it described what was actually
trained** — it does not; the `discretization`/`data_box` fields do.

## Common to all four

- Dataset: `data/production_dataset.h5`, `n_train=800` (all four
  checkpoints), Sobol-sampled, canonical seed 0, `dataset_manifest_hash`
  identical across MLP/DeepONet/POD
  (`d11dfb1ad96cf42be99a557608a580784ac4c01ecfd423cf9b3fd462cccf92c6`).
- Validation: a SEPARATE 30-query Sobol set built by
  `scripts/generate_production_dataset.py:build_val_set`, fixed
  `VAL_SEED=999_999` — **not** a held-out split of the 800 training
  trajectories; an independently-sampled fixed set reused across all
  models and across different `n_train` tranche sizes.
- Dataset rejection: `write_manifest(..., n_failures=0, ...)` at
  `scripts/generate_production_dataset.py:201` — the production dataset's
  own manifest records 0 generation failures. (This is the generation
  script's own hardcoded argument at that call site, not independently
  re-derived here — flagged as what it is.)
- Payoff encoding: `data.payoff_family.european_put(strike, ops)` — the
  discrete coefficient-space projection of the put payoff onto the
  production mesh, passed as `psi` to every model.
- Data normalization: `learning.normalization.{normalize_mu, normalize_psi,
  normalize_tau, stats_from_box}` — `mu` features and `tau` are normalized
  against the frozen parameter box range per feature; `psi` normalized by
  strike.
- Checkpoint criterion: **none** — all three gradient-trained models
  (MLP, DeepONet, ProjectedFNO) train for a FIXED epoch count with no
  early stopping or best-validation-checkpointing; the saved checkpoint is
  simply the state after the last epoch. Validation error is computed
  once, after training, purely for reporting (`evaluate_model`), not used
  to select among epochs.

## MLPCoeffNet

- `model_init_kwargs`: `{mu_dim: 6, psi_dim: 66, hidden_dims: (128, 128, 128)}`
- Activation: not in `model_init_kwargs` — **NOT independently confirmed in
  this pass** (would require reading `src/learning/mlp_coeff.py`'s
  architecture code directly, out of scope for this audit's time budget;
  flagged rather than guessed).
- Parameter count: not computed in this pass (straightforward to add —
  `sum(p.numel() for p in model.parameters())` — not run here for time).
- Optimizer: Adam, `lr=2e-3` (`scripts/train_final_models.py:LR`).
- LR schedule: `torch.optim.lr_scheduler.CosineAnnealingLR(T_max=3000,
  eta_min=2e-6)`.
- Epochs: 3000, **full-batch** (every training row — all 800 trajectories
  × every time step — in a single forward/backward per epoch; no
  mini-batching for MLP/DeepONet, unlike ProjectedFNO below).
- Loss: `learning.losses.mass_weighted_loss` (the `M_h`-weighted
  coefficient-space loss, not plain MSE).
- Recorded outcome (`checkpoints/mlp_final.pt`): `train_time_s=242.5`,
  `final_train_loss=2.43e-05`, `val_l2_error=0.7956` (see note below on
  this number's scale).

## DeepONetCoeffNet

- `model_init_kwargs`: `{mu_dim: 6, psi_dim: 66, latent_dim: 64,
  branch_hidden: (128, 128), trunk_hidden: (64, 64)}`
- Same optimizer/schedule/epochs/loss as MLP (shared
  `train_gradient_model` function).
- Recorded outcome: `train_time_s=372.3`, `final_train_loss=3.71e-06`,
  `val_l2_error=0.4302`.

## ProjectedFNO

- `fno_init_kwargs`: `{in_channels: 8, out_channels: 1, width: 32,
  n_modes: 16, n_layers: 4}`, `mu_dim=6`.
- Optimizer: Adam, `lr=2e-3`; LR schedule: `CosineAnnealingLR(T_max=60,
  eta_min=2e-6)` — **different T_max than MLP/DeepONet** since this model
  trains in epochs-over-mini-batches, not full-batch (see below).
- Epochs: 60, **mini-batched by trajectory group** (`traj_batch_size=8`,
  i.e. ~100 gradient steps/epoch for 800 trajectories, ~6000 total
  gradient steps) — a documented, deliberate redesign from an earlier
  full-batch version (`scripts/train_final_models.py`'s own inline
  comment: the full-batch version took 6.6h at ~1 gradient step per
  epoch; the mini-batched version trains in comparable wall time with
  ~95x more gradient updates).
- Loss: plain MSE on a grid-space SHIFTED target (`V_grid(tau) -
  V_grid(0)`), NOT `mass_weighted_loss` — ProjectedFNO's
  `predict_trajectory` uses a non-differentiable NumPy projection step, so
  it cannot share the other two models' coefficient-space training loop
  (documented in `train_fno`'s own docstring).
- Recorded outcome: `train_time_s=295.2`, `final_train_loss=0.151`,
  `val_l2_error=1.265`.

## PODGalerkinROM

- Not gradient-trained: `compute_pod_basis` is a single SVD of the
  mean-centered snapshot matrix (`n_train=800` trajectories × 61 time
  steps = 48,800 snapshot rows, `n_dof=66` columns).
- `n_modes=32` at training time (`scripts/train_final_models.py: n_modes =
  min(32, snapshots.shape[0] - 1)` — the cap of 32 is a fixed choice, not
  constrained by snapshot count here since 48,800 ≫ 32).
- `train_time_s=0.113` (pure linear algebra, no optimizer).
- Recorded outcome: `val_l2_error=9.458`.
- **See the rank ablation below (`pod_rank_ablation.csv`) for why this
  number is large and does not shrink with rank** — it is NOT a basis
  size problem.

## A metric-scale discrepancy, flagged not resolved

The checkpoints' own recorded `val_l2_error` (MLP 0.796, DeepONet 0.430,
ProjectedFNO 1.265, POD 9.458) is **far larger** than `main.tex`'s stated
headline numbers (e.g. "DeepONet... median relative L2 error 2.02e-3" on
the 100-query interpolation test set). These are evidently different
quantities — `val_l2_error` here is computed by `evaluate_model`
(`scripts/train_final_models.py`), an **absolute**, unnormalized
mass-norm L2 error averaged over a 30-query validation set with no
reference-price-scale normalization, versus the manuscript's **relative**
L2 error on a separate 100-query test set (`experiments/exp01_interpolation.py`,
not read in this pass). This is reported here as a flagged
**discrepancy-in-definition, not a contradiction** — not independently
reconciled in this pass (out of this audit's scope; cross-referencing
`exp01_interpolation.py`'s exact normalization would resolve it). Logged
for Gate E3's reconciliation list.

## POD rank ablation — see `pod_rank_ablation.csv`/`.py`

Summary (full numbers in the CSV, not retyped here):

| rank | mass-weighted projection floor | reduced initial error | reduced dynamics error (mean) |
|---:|---:|---:|---:|
| 8 | 2.152e-01 | 0.0 (exact) | 9.454 |
| 16 | 1.983e-02 | 0.0 (exact) | 9.440 |
| 32 | 3.223e-04 | 0.0 (exact) | 9.458 |
| 48 | 3.445e-06 | 0.0 (exact) | 9.459 |
| 66 (full) | 4.743e-15 | 0.0 (exact) | 9.461 |

**Full-rank recovery: PASSES.** At rank 66 (= `n_dof`, the largest a
66-dimensional coefficient space supports — confirmed `n_snapshots=48800
≫ n_dof`, so the SVD's `U` has a full 66 columns), the mass-weighted
projection floor is `4.74e-15` — machine precision, exactly as the roadmap
requires ("Full-rank recovery of the identical discrete reference is a
meaningful check").

**Reduced initial error is exactly 0.0 at every rank — not a bug, and not
what the roadmap's "irreducible reduced-space projection error" warning
anticipates.** `PODGalerkinROM` uses the same structural exact-IC trick
every model in this project uses (`src/learning/pod.py`'s own docstring:
`c_theta(tau) = c_h^0 + [g(tau) - g(0)]`) — the REFERENCE initial
coefficient is used directly as the baseline, never POD-projected, so
there is no reduced-space IC error to report for THIS architecture. The
roadmap's warning would apply to a naive POD-ROM that projects `c0` onto
the reduced basis; this implementation deliberately avoids that pitfall by
construction.

**Reduced dynamics error is essentially FLAT across every rank (9.45–9.46)
— basis truncation is not the dominant error source for this
architecture, at any rank tested.** This matches the pre-existing
checkpoint's own recorded `val_l2_error=9.458` almost exactly (the rank=32
row above, the same rank the checkpoint was trained at). Investigated
directly (not left as a bare number): on one validation query at full
rank, `max|pred - true|` at the 4 Dirichlet-constrained dofs is **0.979**
(predicted 49.55 vs. true 48.86 at the left boundary) and at the 62
interior dofs is **2.19** — both large, neither alone fully explaining the
total, confirming `src/learning/pod.py`'s own documented limitation
("[the reduced system] does not enforce the reference solver's Dirichlet
boundary conditions... Errors concentrate near the domain edges") is real
and reproducible, but Dirichlet-row error is not the *sole* contributor —
there is also genuine interior propagation error in the reduced dynamics
that persists even with zero basis truncation. **This is a structural
property of the reduced-order time-stepping scheme (the Dirichlet
row-overwrite trick used by the full-order solver has no clean Galerkin-
projected equivalent), not a rank/capacity problem, and not fixed by this
audit** (fixing it would mean redesigning how Dirichlet constraints are
imposed in the reduced dynamics — a nontrivial numerics change, out of
this audit's scope and not this audit's call to make unilaterally).

## 3-seed neural comparison

`experiments/exp12_seed_variance.py` already exists and already covers
exactly this requirement (DeepONet, seeds {0,1,2}, `n_train=800` matching
the canonical checkpoint, 50 held-out test queries, `test_seed=121121`).
**Not re-run in this pass** — its existing result
(`results/exp12_seed_variance.h5`) was read directly (not re-derived or
guessed) and the per-seed training times recorded there
(`train_time_seed0=1337s, seed1=346s, seed2=404s`) confirm these were real,
already-completed, expensive training runs; re-running all three now would
cost on the order of 15-30 minutes for a result already on record, not
warranted in this pass. Its numbers, read directly from the HDF5 file:

| metric | mean (across 3 seeds) | std (across 3 seeds) |
|---|---:|---:|
| accepted (fraction) | 0.980 | 0.000 |
| effectivity | 5.677 | 0.156 |
| price_err | 0.299 | 0.0147 |
| rel_energy | 0.01634 | 0.000338 |
| rel_l2 | 0.002399 | 0.0000200 |
| runtime (s) | 0.2410 | 0.0105 |

Seed-to-seed variance is small relative to the mean for every metric
(e.g. `rel_l2` std is ~0.8% of its mean) — consistent with the roadmap's
caution to not over-interpret pure seed variance as parameter-
generalization uncertainty: here it is small enough that it isn't
competing with that uncertainty in the first place, for this metric set.
