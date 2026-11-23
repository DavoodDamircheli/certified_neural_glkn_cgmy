# Claude Code brief: Phase 4 follow-up 4 (SIAM format: main article + supplement)

Same ground rules: no manuscript edits, no reruns. The manuscript has been converted to the SIAM
article class for SISC (`siamart251216.cls`) and split into a main article (26 pages, the journal's
hard limit) and Supplementary Materials (11 pages). Nothing was removed: Appendix A, the
verification studies of the old Sections 5.2–5.5, the DeepONet paragraph, and the tables of POD-NN
checkpoints, training details and cost-model constants now live in the supplement, which is refereed
with the paper. This brief folds the new layout into the repository and re-points the number check.

## 0. Pull the bundle

`manuscript_v9_bundle.zip` is attached; unpack it over the repository after stripping the top-level
`cgmy_v9/` folder and commit it as one commit. New layout of `manuscript_v9/`:

- `main_v9.tex` — the SIAM main article (`\documentclass[hidelinks]{siamart251216}`);
- `supplement_v9.tex` — the supplement (`[supplement,hidelinks]`); the two files cross-reference each
  other through `\externaldocument` (xr-hyper), so each must be compiled after the other's `.aux`
  exists: `pdflatex main_v9; pdflatex supplement_v9; pdflatex main_v9; pdflatex supplement_v9;
  pdflatex main_v9`;
- `references_v9.tex` — the shared `thebibliography`, input by both;
- `siamart251216.cls`, `siamplain.bst` — the SIAM class files (SIAM's own, unmodified);
- `theory/*.tex` — unchanged statements; a few displays were reflowed for the narrower text width;
- `cover_letter.tex`;
- `amsart_version/main_v9_amsart.tex` — the previous single-file version, kept for the record
  (not compiled, not checked).

Tables and figures are unchanged. `tab_assembly_v9`, `tab_pcg_v9`, `tab_podnn_v9`, `tab_training_v9`,
`tab_cost_v9`, `F2`, `F3` are now input by the supplement; the parameter-box table and the populations
table of the main text were folded into prose (equation (2.10) and Section 5.1).

## 1. Number check over both files

`scripts/check_numbers.py` in the bundle has a one-line change: `tex_corpus()` now reads
`main_v9.tex`, `supplement_v9.tex` and `theory/*.tex`. Keep it. With that corpus, 282 of the 296 claims
pass unchanged; 14 report "text not found" because the sentence was reworded (the values did not
change). Re-point them:

| id | new wording (main text) |
|---|---|
| s51_audit48 | "48 audit queries of the earlier version with perturbed initial vectors" (Section 5.1) |
| s51_bench120 | "120 Sobol benchmark queries" (5.1) |
| s51_e4_1e4 | "$10^4$ Sobol queries for the multi-query study" (5.1) |
| s51_ood200, s51_ood64 | "200 Sobol queries outside the box plus 64 extended vertices" (5.1) |
| s57_sel_bound, s57_sel_err, s57_sel0_bound, s57_sel0_err | "(bound $1.26$, error $0.46$, against bound $12.2$, error $0.082$ when selecting by the true error" (5.4) |
| s58_cold_0.30, s58_init_proj_90 | "without reusable setup the ratio rises to $0.30$ because the query-independent initial projection, $90$~ms, is rebuilt inside the timer" (5.5) |
| s6_0.03 | "at $0.03$ of the direct time from rank $80$ on" (Section 6) |
| s6_N1024 | the conclusion no longer names $N=1024$; re-point to "$N=1024$, $N_T=240$, $\varepsilon_N=1.18\times10^{-3}$" in 5.1 or drop as a duplicate of s51 |
| s44_online_3e-15 | the "$3\times10^{-15}$ for $O(1)$ residuals" statement was removed from 4.3; drop the claim |

The verification summary paragraph (Section 5.2 of the main text) quotes numbers that also appear
in the supplement's full account (SM8); the existing claims match the supplement's sentences, and the
summary's own phrases ("$8$--$12$ iterations for $N=64$--$4096$ against $42$--$1915$ unpreconditioned",
"median $2.05$ at $N=1024$", "$\lambda_{\min}(S_I)/\smin(\mu)\in[1.003,17.8]$") are covered by the
same sources; add claims for them if any number of that paragraph is not yet covered.

## 2. Build, archive, report

- `README.md` of the archive: update the "paper table/figure -> result file -> script" mapping to the
  new numbering (main text: Table 1 notation, Table 2 budget, Table 3 POD ranks, Table 4 benchmark,
  Table 5 OOD; Figures 1 online rule, 2 EIM, 3 acceptance, 4 benchmark, 5 OOD, 6 online time and
  7 amortisation (one two-panel float); supplement: SM8 verification with the assembly table, F2, F3,
  PCG table; SM9 DeepONet; SM10 POD-NN and training tables; SM11 cost table). Read the compiled PDFs
  for the final numbers rather than trusting this list.
- `make_paper_claims.py && check_numbers.py` with exit code 0; `bash build.sh` (tables and figures
  unchanged, so byte-identical); manifest; build-only clean-clone check; annotated tag `v9-archive`
  moved; `phase4_archive_<date>_final4.zip`; two-line addendum to `PHASE4_REPORT.md`.
