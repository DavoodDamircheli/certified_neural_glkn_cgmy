# Follow-up 2 after Phase 2.2–2.3 (paste to Claude Code as the next prompt)

Read this file, then act on it in the order given. The rules of `CLAUDE_PHASE2.md` remain in force
(no edits under `manuscript_v9/` or `v8_frozen/`, results as JSON with provenance, deviations in
`results/DEVIATIONS.md`, one commit per item). Do not interrupt the running 2.4 sweep; items 1–3
below do not depend on it, item 4 is for after it finishes.

## 1. `v8_frozen/` arrives as a zip, not from git

The file `v8_frozen.zip` (sha256 `c93244ed28d339fef12658dd563ab058b33b2e113d53f60e9c130b63da46cd65`)
has been placed in the repository root by the user. Do:

```
sha256sum v8_frozen.zip            # must match the value above
unzip -q v8_frozen.zip             # creates v8_frozen/ at the repo root
python3 -m pytest tests -q         # the 4 v8 regression tests must now run and pass (not skip)
git add v8_frozen && git commit -m "Add v8_frozen (handoff 2026-10-19, repo 568e1b5), from zip"
```

`v8_frozen/` is read-only reference material, as before. Then do 2.5 (the DeepONet decision) as
specified in `CLAUDE_PHASE2_followup_1.md`, section 3. `data/production_dataset.h5` is still not
included; ask for it only if the checkpoint loader cannot run without it.

## 2. Review of 2.2–2.3: what is accepted, what to change

Accepted as is (keep in `DEVIATIONS.md`):
- the allowance uses the raw coefficient vector including the two prescribed entries (correct;
  the paper's Theorem 5.2 has been changed accordingly);
- the R-factor form of the reduced residual norm with an explicit rounding term, replacing the
  Gram form (this is the known cancellation problem; the paper now cites Casenave–Ern–Lelièvre 2014
  and Buhr et al. 2014 for it; keep the strict 1e-10 test as xfail with the extended-precision
  evidence attached);
- the β₀ fix (exact Pythagoras form);
- the r = 48 sanity check not reproducing the v8 acceptance pattern at eps = 1e-3. This is a
  result, not a failure: the relevant tolerance is eps_N from 3.1, and effectivity 3–10 at N = 512
  was already seen in E0.3. No action.

Change these three things:

### 2a. The full-order certificate evaluates the residual with the exact operator A_I, not A_Q
For every candidate (POD-Galerkin, POD-NN, DeepONet) the *full-order* certificate (`certify` in
banded / exact / two-stage mode) must compute the CN residual with `fs.A` (the exact assembly,
two FFTs per step) and therefore carries **no EIM allowance**. A_Q is used only (i) inside the
reduced solve and (ii) inside the N-independent online certificate, where the allowance is
unavoidable. Re-run the 2.2 full-order numbers this way and store both variants side by side
(`with_AI` and `with_AQ_allowance`) in `results/E2_2_pod.json`, so the paper can show how much the
allowance costs. Expected: the `with_AI` bound is tighter from r ≈ 40 upward.

### 2b. Make the per-query symbol quantities N-independent and cheap
The N-independence claim of the paper is about the whole online certificate, so eps_Q(mu) and the
optional tightening factor must not cost O(N K). Two changes:
- The factor c_mu = (inf_theta s_mu / x-bar)^{-1} is **not needed for validity**: with the cell
  construction x-bar <= x_mu <= s_mu pointwise, so rho^T S^{-1} rho <= rho^T X-bar^{-1} rho
  directly (c_mu <= 1). Make it optional (`tighten=True/False`), default off in the timed online
  path; report both bounds in E1.
- Evaluate eps_Q(mu) (and c_mu when requested) on the N-independent graded grid used by
  `spectral_data` (uniform 2^14 points plus the geometric refinement on the scale min(G,M) h near
  theta = 0, or a coarser version of it) with the cell-wise derivative margin of
  `spectral._grid_extreme`. The continuous bound sup_theta |e_nu| >= max over the assembly grid, so
  it is valid for the assembled matrices too. Then reduce the number of points until the total
  online symbol cost is < 5 ms at every N (the derivative margin keeps it rigorous whatever the
  grid); record the grid size and the resulting loss of tightness in `results/E2_3_reduced.json`.
  Keep your adaptive alias truncation if it is rigorous (state the bound it uses).
- Report the new timing table (certificate evaluation / eps_Q / c_mu, one thread, N = 256, 1024,
  4096); all three rows should be flat in N.

### 2c. Cells in (C, Y, min(G,M)), not (r, C)
beta_mu = C * beta~(G, M, Y) (the fractional part of s_mu − r m is proportional to C), so the r-split
is useless and the C-split alone leaves the (G, M, Y) variation uncovered — hence effectivity 51 vs
18. Replace the cells by a partition of (log C, Y, log min(G,M)): start with 2 × 2 × 2, then 3 × 3 × 2
if storage allows (each cell stores the offline tensors; report MB per cell). Use r-bar = r_min of
the box for all cells. Report, on validation30 with the POD-Galerkin r = 48 candidate: median and
max of eta^{X-bar} / eta^{X} (cell norm vs the query's own X_I) and eta^{X-bar} / eta^{S}, for
each partition. The goal is a median cell effectivity within a factor ~2 of the own-norm one.
If the POD-NN loss used the old cells, keep the sweep running (its loss is still a valid majorant)
and note in `training_log.json` which cell partition the loss used; the *evaluation* in 2.4 and E1
uses the new cells.

### 2d. EIM tolerance as a function of the target
Because the online certificate keeps the allowance, Q must satisfy
eps_Q ||c-bar||_2 sqrt(T / alpha_lb) <= eps / 10 for the target eps. With the current Q = 29 at
N = 512 the allowance is ~1e-4 in the M_I norm, which is where the r >= 56 limitation comes from.
Add to `eim.py` a function that returns the smallest stored Q meeting this inequality for a given
(mu, T, eps), build the EIM at N = 512 up to the 1e-11 relative test tolerance (expect Q ≈ 35–40)
and store all intermediate Q, so that E1 can report bounds at the Q matched to eps_N. Report Q vs
tolerance down to 1e-11 for F4.

## 3. Things to record for the paper (append to `results/NOTES_FOR_PAPER.md`)
- the float64 requirement for the network output (bound ~1e-3 from float32 rounding alone);
- the parameter count 3.1 M (the 0.5 M in the plan was wrong; r N_T = 11 520 outputs);
- the allowance size estimate and the "full-order certificate uses A_I" rule;
- the measured effectivity of the cell norm before and after 2c.

## 4. After the 2.4 sweep finishes
Evaluate every checkpoint on validation30 at N = 512 with (i) the reduced online certificate (new
cells, Q matched to eps = 1e-3 by 2d, allowance included), (ii) the full-order banded certificate
with A_I, (iii) the full-order exact certificate; report beta_0, the three bounds, the true error
vs the stored reference, effectivity, and acceptance at eps in 1e-3 × {0.1, 0.3, 1, 3, 10}. Then the
2.5 decision, then the Phase 2 zip and `results/PHASE2_REPORT.md` as described in
`CLAUDE_PHASE2.md`.
