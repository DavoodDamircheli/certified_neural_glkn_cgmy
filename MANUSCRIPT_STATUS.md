# Manuscript status (v9), 2026-10-25

`manuscript_v9/main_v9.tex` + `theory/*.tex` + `../tables_v9/` + `../figures_v9/` (PDF figures with PNG
fallback); 36 pages; compiles with pdflatex x3, no overfull boxes; `cover_letter.tex` (2 pages).
One `\todo` left: the archive URL in Section 5.10 (plan 4.4), to be filled in when the archive is deposited.

## Verification state
- Number check (`scripts/check_numbers.py`, `results/paper_claims.json`): 294 claims, 294 pass at commit
  2d2dd74 of the user's repository (280 from result files, 14 source-backed: test thresholds, fixtures,
  v8 records, NOTES/DEVIATIONS). After the editorial pass two claim texts need re-pointing
  (`s51_calib_spots`, `th_eim_Q37`); brief `CLAUDE_PHASE4_followup_3.md`.
- Archive check: clean clone rebuilds 9 tables byte-identically and 10 figures pixel-identically;
  `run_E0.py` end to end 32 min on one core; 67 tests. Tag `v9-archive` (annotated) on the user's machine.
- E0.3/E0.4 rerun with deterministic (crc32) seeds; Table 4 and F3 are the rerun values (plain CG 42-1915,
  PCG 8-12).

## Content by section
1. Introduction: contributions (five), COS on page 1, RB/SCM/EIM/Toeplitz literature, notation table (Table 1).
2. CGMY spline Galerkin via symbols: periodisation and tail lemmas, Toeplitz sections (blocks $T_G^-$, $T_M^+$),
   Levy-Khintchine lemma, spectral theorem, compression remark, Remark 2.3 (c_1 = 5.1e-5 explicit, 1.4e-4 sharp).
3. Residual certification: CN theorem, bracket with alpha_lb, comparison norm, two-stage and three-stage rules
   (status names unified: reduced accept / cheap reject / banded accept / exact accept / exact reject /
   corrected accept), efficiency, online rule figure with the reduced stage, guaranteed acceptance.
4. Affine structure and candidates: EIM on the symbol, error-transfer theorem (allowance 6.8e-5 median to
   4.3e-4 max at N = 512, Q = 29; max 3.9e-6 at Q = 37), DeepONet (old checkpoint, rejected), POD-Galerkin with
   A_Q, reduced online certificate (R-factor, SVD-compressed, rho retained rows), POD-NN, cost model.
5. Numerics: E0.1-E0.5; calibration (N = 1024, N_T = 240, eps_N = 1.18e-3); reduced model and candidates
   (Table 7, Figs 5-6; reduced effectivity 230-420 at r >= 96); benchmark (Tables 10-11, Fig 7; 337 ms cold);
   OOD (Table 12, Fig 8; 193 of 210, median effectivity 6.3); multi-query (Fig 9); reproducibility.
6. Interpretation. Appendix A: A.1 via Levy-Khintchine; A.2-A.6 kept from v8 with the stale cross-reference
   and the principal-period sentence removed; VI theorem.

## Notation decisions
Strike $\mathcal K$ (the letter $K$ is the alias truncation); Greek $\Gamma_{\rm opt}$; mass matrix $M_h$ and
tempering $M$ both kept; Toeplitz blocks $T_G^-$, $T_M^+$; estimators $\eta^S$, $\eta^X$, $\eta^{\bar X}$.

## Remaining
- Archive URL (Section 5.10) and removal of the `\todo` macro at submission.
- Cover letter: one bracketed line about the earlier version's circulation status to be settled by the author.
- Fold in the final3 archive (claims re-pointed, tag moved) when it comes back.
