# Follow-up 3 (paste to Claude Code as the next prompt; items 1–3 can run while the POD-NN sweep continues)

Rules of `CLAUDE_PHASE2.md` unchanged. Sections 1–3 of follow-up 2 are accepted, including the
decision to drop the DeepONet retrain (record the two reasons verbatim in `DEVIATIONS.md`; keep
`results/deeponet_old_vs_new.json`, it goes into the paper as one sentence).

## 1. Find and remove the O(N) work left in the online certificate (before any Phase 3 timing)

The reduced certificate's quadratic form is O(Q² r²), yet the measured evaluation grows with N
(0.55 / 1.95 / 2.99 ms and 0.6 / 5.4 / 9.1 ms at N = 256 / 1024 / 4096). Something in the online
path still touches length-N_h arrays. Profile one call at N = 4096 (`cProfile`, one thread) and
move every N-dependent piece offline. The usual suspects:
- `alpha^0 = V^T M_I z_h^0` and `beta_0 = ||(I - V V^T M_I) z_h^0||_{M_I}`: z_h^0 depends only on
  (K, N), not on the query (K = 100 is fixed; the prescribed value at tau = 0 is K sqrt(h)), so
  both are offline constants per (N, basis) — store them with the basis;
- the reduced forcing: the coupling vectors `V^T T_q e_bd`, `V^T M_I e_bd` (two boundary
  directions, each EIM column) must be stored offline as r-vectors; online only the scalars
  ell_q(nu), b_L(tau_n) enter;
- reading or slicing the stored Toeplitz columns, or forming any length-N_h vector, inside the
  timed call.
Target: the timed online call (certificate + eps_Q) is flat in N to within noise, < 5 ms at one
thread when the machine is idle. Report the profile before/after in `results/E2_3_reduced.json`.

## 2. Make beta-bar rigorous over (C, G, M); grid only in Y

beta_mu = C beta~(G, M, Y), and beta~ is nonincreasing in G and in M because I_a(xi) is decreasing
in a (the factor e^{-a x}; Lemma 2.6 of the paper). Hence on a cell the minimum over (C, G, M) is
attained exactly at the corner (C_min, G_max, M_max); only the one-dimensional minimum over Y is
taken on a grid (use >= 33 points) with the 0.99 factor. Restructure `beta_bar` accordingly, verify
numerically that the old grid minima were never below the corner value times 0.99 (they should
not be), and record the statement in `NOTES_FOR_PAPER.md` (the paper's Remark 4.3 already says
this). The 3 × 3 × 2 partition stays.

## 3. Phase 3.1 now (it only needs direct solves and the COS pricer; timing-insensitive)

Run 3.1 exactly as in `CLAUDE_PHASE2.md`: ref30 + the two v8 corner queries
(`v8_frozen/manuscript_v8/tables/tab_corner.tex`), N in {128, 256, 512, 1024}, N_T in {60, 240},
five spots 0.8K..1.2K, `v8_frozen/numerics/validation/cos_pricer.py` at the extended settings
(8192 modes, range parameter 30; confirm against Carr–Madan to 1e-12 on 10 rows). Production N
= smallest with max |V_h − V_COS| <= 1 cent over queries and spots; eps_N = 0.01 / max_s beta_I(x_s)
(`direct.spot_price_factor`), also 0.1 cent at N = 1024. Outputs: `results/E3_1_calibration.json`,
`tables_v9/tab_budget_v9.tex`, `results/PRODUCTION_CONFIG.json` (N, N_T, eps_N, the sweep
eps_N × {0.1, 0.3, 1, 3, 10}). If the production N differs from 512 or N_T from 240, say so
prominently in the report: the stored references and bases would then have to be regenerated
before E1 (do not start that without asking).

## 4. After the sweep: section 4 of follow-up 2, plus one optional variant

Evaluate every checkpoint as described there (reduced certificate with the 3 × 3 × 2 cells and the
eps-matched Q; full-order banded with A_I; full-order exact), with acceptance at eps_N × {0.1, 0.3,
1, 3, 10} from `PRODUCTION_CONFIG.json` as well as at 1e-3.

Optional, capped at one day of compute and only if the best median full-order bound is more than
10 × eps_N: train one more variant whose output is the space–time POD representation of the
coefficient matrix (second POD over time of the 800/4000 reduced trajectories, keep r_t temporal
modes such that the projection error is < eps_N / 10, expect r_t ≈ 20–30; outputs r × r_t instead
of r × N_T), same MLP widths, lambda_R = best of the sweep, 3 seeds, 4000 trajectories. Report it
as a separate row. If the cap is hit, stop and report what was reached.

## 5. Then: Phase 2 zip and report

`results/PHASE2_REPORT.md` with the items-vs-status table, the week-4 decision (DeepONet dropped,
reasons), the week-8 decision (production N, N_T, eps_N from 3.1), the POD-NN result at eps_N and
at 1e-3, every deviation, and the regeneration commands. Zip as in `CLAUDE_PHASE2.md`
(`phase2_results_<date>.zip`, large files excluded with hashes in `MANIFEST.json`). Do not start
E1–E4 before the zip is handed over.
