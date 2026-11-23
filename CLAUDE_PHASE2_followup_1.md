# Follow-up to CLAUDE_PHASE2.md after Phase 2.1 (paste this to Claude Code as the next prompt)

Read this file, then continue with the tasks in `CLAUDE_PHASE2.md` starting at 2.2. The rules of
that brief still apply (no edits under `manuscript_v9/`, no edits under `v8_frozen/`, results as
JSON with provenance, deviations in `results/DEVIATIONS.md`, commit per item).

## 1. Restore `v8_frozen/` from git before anything else

`v8_frozen/` is tracked in the repository history: commit `2dd5f4f` ("Freeze v8 workspace") and the
tag `v8-frozen` contain it. Only the working-tree copy was left out of the zip, and your baseline
commit recorded it as deleted. Run:

```
git checkout v8-frozen -- v8_frozen
git commit -m "Restore v8_frozen from tag v8-frozen"
python3 -m pytest tests -q      # the 4 v8 regression tests must now run and pass, not skip
```

Do not reconstruct it from anywhere else. The 80 MB `data/production_dataset.h5` is not in git;
if it is needed for 2.5, ask for it (sha256 in `v8_frozen/production_dataset.sha256`). It is
only used for the DeepONet decision.

After the restore, these paths exist and are what 2.5 and 3.1 need:
- `v8_frozen/numerics/validation/cos_pricer.py` (COS and Carr-Madan reference prices),
- `v8_frozen/models/deeponet_final.pt`, `v8_frozen/models/checkpoint_metadata.json`,
  `v8_frozen/models/training_code_at_checkpoint_commit_13bf0d5/`,
- `v8_frozen/manuscript_v8/tables/tab_corner.tex` (the two corner queries),
- `v8_frozen/query_sets/*.csv`, `v8_frozen/environment/*`.

## 2. Phase 2.1 review: accepted as is

All four deviations are accepted; keep them as recorded in `results/DEVIATIONS.md`:
- Q = 23 at N = 64 (set by the operator-norm bound, which is the quantity the certificate uses);
- storing the Toeplitz columns of the EIM basis functions;
- the symmetrised symbol family (the unpaired alias at theta = -pi for |k| <= K contributes an
  imaginary part of order (2K)^(Y-6) ~ 1e-10, inside eps_K; add that one sentence to DEVIATIONS.md);
- the compression argument (grid maximum alone bounds ||A_I - A_Q||_2 for the assembled
  matrices). It has been added to the paper as Remark 2.9; keep both the margin and eps_K in the
  code so that alpha_lb and eps_Q stay valid for the exact Galerkin matrix as well. No code change.

## 3. Specific instructions for 2.2-2.4

### 2.2 references and POD-Galerkin
- Generate the reference trajectories first and only once: 800 Sobol queries (seed 0, with T) and
  the 4000-query set (seed 0 extended, the first 800 identical to the 800 set) at N = 512,
  N_T = 240, with `reference_trajectory(q, N, N_T, solver="gmres")` in a process pool. Store as
  `data/references_N512_NT240_seed0_800.npz` / `..._4000.npz` (float64, query ids, mu, T) and write
  sha256 of each file into `results/MANIFEST.json` before building any POD basis. 2.3, 2.4 and
  E1 must read these files, never regenerate.
- Also generate the validation30 trajectories at the same (N, N_T) and store them the same way.
- The reduced forcing with the affine operator needs the coupling vectors of every stored EIM
  column: `V^T T_q e_bd` for the two left boundary directions (raw indices -(p-1), -1), not only
  `uA` from `direct.free_forcing`. Build them from the stored Toeplitz columns (lags 1..p-1
  shifted), so that the reduced right-hand side is affine in the same scalars ell_q(nu) and in
  b_L(tau_n). Test: the affine reduced forcing equals `V^T forcing.F` built with the full
  operator to eps_Q level.
- Full-order certificate of a POD-Galerkin candidate built on A_Q: call
  `certify(..., allowance=eps_Q*||cbar_n||_2/sqrt(spec.alpha_lb), allowance_X=eps_Q*||cbar_n||_2/sqrt(spec.x_min))`
  with cbar_n = V (alpha^{n+1}+alpha^n)/2 (the free vector). Record both bounds and the size of the
  allowance relative to the residual term in `results/E2_2_pod.json`.
- Done-when checks as in the brief: r = 62 vs direct to 1e-10 (with A_I) and to eps_Q level (with
  A_Q); r = 48 acceptance pattern on validation30 at eps = 1e-3.

### 2.3 reduced online certificate
- Cells: 2 x 2 in (log r, log C). beta-bar per cell = 0.99 x min of `spectral_data(q, 512).beta`
  over the 5 x 5 (r, C) grid of the cell combined with the 27 points of the 3 x 3 x 3
  (min, mid, max) grid in (G, M, Y). r-bar = r_min of the cell.
- Offline Gram blocks by banded solves (`scipy.linalg.solve_banded` with `banded_from_toeplitz`).
- Tests: `eta^{X-bar}` online == full-order `rho^T X-bar^{-1} rho` to 1e-10 relative on 20 queries;
  online wall time at r = 48, Q = 29 (your Q at N = 512) for N in {256, 1024, 4096} must be flat;
  report numpy time and, if you use numba/torch, compiled time, both single-threaded.

### 2.4 POD-NN
- Use the stored 800 and 4000 trajectory files. r = 48, alpha^0 fixed to the projection of z_h^0.
- `torch.set_num_threads(1)` for every timed online evaluation; training may use all threads.
- Before training with lambda_R > 0, verify once that the torch implementation of eta^{X-bar}
  agrees with the numpy one of 2.3 to 1e-10 on 5 random reduced trajectories.
- Log per-epoch train/validation losses, the selected epoch, seeds, wall time and hardware to
  `results/training_log.json`; checkpoints to `checkpoints/podnn_r48_n{800,4000}_lamR{...}_seed{0,1,2}.pt`.
- Since 3.1 has not run yet, use eps = 1e-3 (the v8 tolerance) as the provisional acceptance
  target for the 2.4 done-when check and say so in the JSON; E1 repeats it at eps_N.

### 2.5 DeepONet decision
- First certify the old checkpoint as a candidate of the new system at N = 64, N_T = 60, on
  bench120 (two-stage mode; status counts, median/max bound, median true error vs a dense LU
  reference) -> `results/deeponet_old_vs_new.json`. It needs `data/production_dataset.h5` only if
  the loader insists on it; otherwise skip the dataset.
- Then decide (max 3 days): retrain on the new N = 64 references only if
  `v8_frozen/models/training_code_at_checkpoint_commit_13bf0d5/` runs unchanged against the new
  data; otherwise drop it and record the decision.

## 4. When Phase 2 is complete
Zip as described in the brief (`phase2_results_<date>.zip`: cgmy_cert, tests, scripts, results,
figures_v9, tables_v9, checkpoints/*.json, DEVIATIONS.md, NOTES_FOR_PAPER.md, MANIFEST.json;
trajectory files and checkpoints > 50 MB excluded, hashes listed) and write
`results/PHASE2_REPORT.md` with the plan-items-vs-status table, the week-4 DeepONet decision, and
the exact commands that regenerate F4 and every JSON.
