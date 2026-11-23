# Claude Code brief: Phase 4 follow-up 2 (last three numbers, Figure 6 legend)

Same ground rules (no manuscript edits, no reruns of experiments). The three remaining FAILs and the two
points outside the claims have been handled on the paper side; this brief closes them on the repository
side.

## 0. Pull the bundle

`manuscript_v9_bundle.zip` is attached; unpack it over the repository after stripping the top-level
`cgmy_v9/` folder and commit it as one commit. Text changes since 224a44b (nothing else in the bundle
changed):

- Section 5.7: "raises the reduced effectivity to $230$--$420$" (was 120–140);
- Section 5.7: "bounds $0.054$--$0.058$ over three seeds" (was 0.055);
- Section 3 (after the algorithm figure): "at most $12$ iterations in all reported runs,
  Table~\ref{tab:pcg-v9}" (was 15, the GMRES maximum);
- Remark 4.3 (`theory/thm_eim_cert.tex`): the allowance sizes are now quoted with two digits as recorded in
  `NOTES_FOR_PAPER.md` item 10: "$6.8\times10^{-5}$ (median) to $4.3\times10^{-4}$ (maximum) … drops to
  $3.9\times10^{-6}$ with $Q=37$".

Re-point the four claims (two effectivity values at r = 96/128 from `E1.json`, the bound minimum over the
space-time seeds, the PCG maximum from `E0.json` E0.4, and the three allowance numbers) to the new text.

## 1. Allowance sizes: a result file if it costs nothing

The three allowance numbers of Remark 4.3 are recorded only in `NOTES_FOR_PAPER.md` item 10. If the Phase 2
run that produced them (`run_reduced_cert.py`, item 2.3, N = 512, Q = 29 and Q = 37, validation queries)
left the per-query allowance values in `E2_3_reduced_cert.json` under another name, or in its log
`results/run_reduced_cert_v2.log`, register the claims against that file (median and maximum over the
validation queries at Q = 29; median at Q = 37). If the only trace is the note, leave the claims reading the
note and keep them listed under "Source-backed" with the note as the source; do not recompute them.

## 2. Figure 6 legend (F6a)

`figures_v9/F6a_benchmark` prints 370 ms for the cold direct solve (the E1 timing value on
validation30[1:11]), while the benchmark of Section 5.8 and Table 9 are E2 (bench120, cold 337 ms). Change
`run_E2.do_figs` so the legend reads the cold direct median from `E2.json`
(`bench/per_setup/cold/summary/direct/median_time_ms`, 337.1 ms) and the reusable value from the same file,
then regenerate F6a (PNG + PDF) through `build.sh`. No other change to the figure. Add a claim that the
legend value equals the E2 value (read the number from the figure script's JSON path, not from the PNG).

## 3. Re-run and finish

`make_paper_claims.py && check_numbers.py`: expected 0 FAIL of any kind, exit code 0. Then, as before:
manifest, build-only clean-clone check, annotated tag `v9-archive` moved to the new head,
`phase4_archive_<date>_final2.zip`, and a two-line addendum to `results/PHASE4_REPORT.md` with the commit
hash and the claim counts. If anything still disagrees, `NUMBER_CHECK.md` gets the path and value and the
manuscript stays untouched.
