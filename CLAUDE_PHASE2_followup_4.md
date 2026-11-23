# Follow-up 4: production resolution is N = 1024 (paste to Claude Code as the next prompt)

Rules of `CLAUDE_PHASE2.md` unchanged. Items 1–3 of follow-up 3 are accepted (COS agreement at
1.5e-12 / 2.6e-12 recorded as is; the SVD-compressed factor with its rigorous discarded-part term
is fine — but state in `E2_3_reduced.json` whether the quoted "bound increase of at most 8e-4" is
relative or absolute; if absolute, it is 70 % of eps_N and must be reduced by keeping more rows).

## Decision

The calibration rule of the plan is binding: **production configuration N = 1024, N_T = 240,
eps_N = 1.18e-3**, as written in `results/PRODUCTION_CONFIG.json`. The two corner queries are
inside the parameter box and the paper's claim is a one-cent discrete price over the whole box, so
N = 512 cannot be argued away. Nothing in `PRODUCTION_CONFIG.json` changes from now on.

Consequences, in this order:

## 1. Stop the N = 512 POD-NN sweep now; keep it as a pilot

Kill the running sweep. Keep every log and checkpoint produced so far under
`results/pilot_N512/` and `checkpoints/pilot_N512/`, and summarise in `results/pilot_N512.json`:
per run, the last epoch reached, the training/validation losses and the validation median
reduced-certificate bound at that epoch. This is the pilot that justifies the trimmed design of
item 3 and it goes into the paper as one sentence. Do not evaluate the pilot checkpoints further.

## 2. Regenerate everything at N = 1024, N_T = 240 (same seeds, same query sets)

- references: 800 (seed 0), 4000 (seed 0 extended), validation30, bench120 and ref30, with
  `reference_trajectory(q, 1024, 240, solver="gmres")`; files `data/references_N1024_NT240_*.npz`,
  sha256 into `results/MANIFEST.json`;
- mass-weighted POD from the 800 set and from the 4000 set (ranks 8..62; also store the singular
  values for the data-size study);
- EIM: the stored N = 1024 columns (Q = 29 at 1e-9) must be extended to the 1e-11 tolerance like
  the N = 512 ones (expect Q ≈ 37–42) so that the eps-matched Q of 2d exists at 1024; record the
  eps_N-matched Q on validation30 (eps_N = 1.18e-3 and eps_N/10);
- cell comparison norms (3 × 3 × 2 in (log C, Y, log min(G,M)), corner rule for beta-bar, grid in Y)
  and the offline reduced-certificate tensors for the two Q values that 2d selects at eps_N and
  eps_N/10, with the SVD-compressed factor; report MB per cell;
- re-run the 2.2 and 2.3 done-when checks at N = 1024 (r = 62 vs direct with A_I to 1e-10;
  reduced online == full-order to the floor; flat timing N = 256 / 1024 / 4096 with the 1024
  tensors used at every N where applicable — the tensors are N-specific, so time the N = 1024 set
  and state that the cost does not depend on N by construction), and the POD-Galerkin
  acceptance table on validation30 at eps_N × {0.1, 0.3, 1, 3, 10}: both certificates
  (full-order with A_I, reduced online) — this is already most of E1's POD-Galerkin column.
Everything at N = 512 stays on disk and in git as the 2.2/2.3 development record; the paper's
numbers come from N = 1024.

## 3. POD-NN sweep at N = 1024, trimmed

Same architecture (float64, r = 48, alpha^0 fixed), same optimiser and schedule, validation-based
checkpoint selection on validation30. Design, to fit in about half the time of the full sweep:
- 800 and 4000 trajectories × lambda_R in {0, 1e-2, 1e-1, 1}, seed 0 only (8 runs);
- then the best (data size, lambda_R) by validation median full-order bound, seeds 1 and 2 (2 runs);
- the optional space–time variant of follow-up 3 §4 (r_t temporal modes with projection error
  < eps_N/10) at the best (data size, lambda_R), seeds 0, 1, 2 — run it in any case this time, since
  the pilot already shows the bound is far above eps_N; cap it at one day.
Log everything to `results/training_log.json` (one entry per run, including the pilot's design)
and evaluate every checkpoint as in follow-up 2 §4 with the N = 1024 tensors and eps_N.

## 4. Then the Phase 2 report and zip

As in follow-up 3 §5. The report must state, in its first lines, the week-4 decision (DeepONet
dropped), the week-8 decision (N = 1024, N_T = 240, eps_N = 1.18e-3; the two corner queries decide
it; 60 steps insufficient at 1024: 4.6 cents), and the week-10 answer (does any POD-NN pass at
eps_N, and the best median bound otherwise), with the table of the two calibration
resolutions as measured. Hand over the zip before starting E1–E4.
