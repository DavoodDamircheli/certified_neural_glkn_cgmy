# Claude Code brief: Phase 4 follow-up 3 (editorial pass folded in; final archive)

Same ground rules: no manuscript edits, no reruns. The paper side did the editorial pass of plan 4.3
(notation table, unified status names, strike renamed to $\mathcal K$ so that $K$ is only the alias
truncation, Toeplitz blocks $T_G^-$, $T_M^+$, dangling sentences and v8-era remarks removed, displays
reflowed). Two table generators were touched on the paper side so that the regenerated tables match
the manuscript; this brief folds those in, re-points the claims whose text changed, and refreshes the
archive.

## 0. Pull the bundle

`manuscript_v9_bundle.zip` is attached; unpack it over the repository after stripping the top-level
`cgmy_v9/` folder and commit it as one commit. It carries `manuscript_v9/` (including the new
`cover_letter.tex`), `tables_v9/`, `figures_v9/`, and three scripts:

- `scripts/run_E0.py`: `plot_e01` now wraps the tabular of `tab_assembly_v9.tex` in
  `\resizebox{\textwidth}{!}{...}` (the table overflowed the text width) and its caption says "The
  principal-period rule of the earlier version is the alias sum with $K=0$" instead of "(v8)";
- `scripts/run_E3_1.py`: `do_table` caption of `tab_budget_v9.tex` without "(Phase 3.1)", "30
  price-diagnostic queries and the two box corners", spots in $\mathcal K$, and column heads "30
  queries / corner A / corner B";
- `scripts/make_tables_phase2.py` unchanged since follow-up 2 (included for completeness).

The two tables in the bundle were regenerated from these scripts; `bash build.sh` must reproduce them
byte-identically. Nothing else in `tables_v9/` or `figures_v9/` changed.

## 1. Re-point the claims whose text changed

Running `check_numbers.py` against the bundle gives exactly two "text not found" claims (plus
`s43_deeponet_params`, which only needs torch):

- `s51_calib_spots`: the text is now "five spots $0.8\mathcal K,\dots,1.2\mathcal K$" (Table of
  populations and Section 5.5);
- `th_eim_Q37`: the text is now "its maximum drops to $3.9\times10^{-6}$ with $Q=37$"
  (`theory/thm_eim_cert.tex`).

Also check these spots, whose wording changed without a number changing: the status names in
`theory/prop_twostage.tex` and `theory/prop_threestage.tex` (now `banded accept`, `cheap reject`,
`exact accept`, `exact reject`, `reduced accept`), the online rule figure in Section 3 (a reduced-stage
step was added, so step numbers moved), and "a gap of $46$--$1000$" in Section 5.7 (was 47). If any
claim quoted those passages, re-point it.

## 2. Re-run and finish

`make_paper_claims.py && check_numbers.py` with exit code 0; `bash build.sh` with byte-identical tables
and pixel-identical figures; manifest; build-only clean-clone check; annotated tag `v9-archive` moved
to the new head; `phase4_archive_<date>_final3.zip`; two-line addendum to `results/PHASE4_REPORT.md`
with the commit hash and the claim counts. The manuscript still carries one `\todo` (the archive URL in
Section 5.10), which the paper side fills in once the archive is deposited; it is not a number and
needs no claim.
