# Claude Code brief: Phase 4 (archive, number check, final packaging). No E5.

Ground rules of `CLAUDE_PHASE2.md` still apply, with one change: `manuscript_v9/` is now present in
the repository (copied from the paper-editing side) and you may **read** it, but you still do not
edit it. No experiment is rerun and no network is retrained; everything below is packaging and
verification. Decision recorded: E5 (American put) is not done; Theorem A.7 stays as outlook.

## 1. Reproducibility archive (plan 4.4)

1. `build.sh` (and a `Makefile` target `all`) that regenerates **every** table in `tables_v9/` and
   every figure in `figures_v9/` from `results/*.json` only — no solver call, no data file. Include
   `scripts/make_tables_phase2.py` (already in the repo) and whatever scripts made the other tables
   and figures; where a figure script currently recomputes anything, split it into a compute step
   (not part of `build.sh`) and a plot step that reads JSON.
2. Vector versions of every figure: re-run the plot steps with `savefig("... .pdf")` next to the
   PNGs (same names, `.pdf`), fonts embedded, no change of content. F5a–F8 and F2–F4.
3. Clean-clone test: `git clone` into a temporary directory, `pip install -r requirements.txt` in a
   fresh venv (write `requirements.txt` with the pinned versions actually used), `bash build.sh`,
   and `diff` the regenerated tables against the committed ones (byte-identical) and the figures
   against the committed PNGs (identical pixel arrays via numpy). Record the result in
   `results/ARCHIVE_CHECK.json` with the clone's commit hash.
4. `python3 scripts/run_E0.py` end to end in the clean clone, single process, with
   `CGMY_FULL_TESTS=0 python3 -m pytest tests -q` afterwards; record wall times in
   `ARCHIVE_CHECK.json`. The plan's target is < 1 h on a laptop; report what it takes on the
   workstation with `--nproc 1` and estimate the laptop time from the CPU ratio.
5. `MANIFEST.json`: SHA-256 of every file in the archive (code, tests, scripts, results, figures,
   tables, fixtures) plus the data files and checkpoints that are excluded from the archive, with
   sizes and the commands that regenerate each of them.
6. `README.md` of the archive: layout, environment (Python/numpy/scipy/torch/mpmath versions, CPU
   model, thread pinning), the one-line commands for `build.sh`, the tests, and each
   `run_E*.py`, and the mapping "paper table/figure -> result file -> script" (one row per item).

## 2. Number check against the manuscript (plan 4.5)

Write `scripts/check_numbers.py` and a claims file `results/paper_claims.json`:
- each claim = {"text": the number as it appears in `manuscript_v9/main_v9.tex` (and in
  `manuscript_v9/theory/*.tex`), "json": result file, "path": JSON path, "transform": optional
  (e.g. "percent", "ratio", "median", "round:2"), "tol": relative tolerance};
- cover every numerical statement in Sections 2.5, 2.7, 4.1, 4.3, 4.4, 5.1–5.10, 6 and the abstract
  (expect 120–180 claims; the tables are checked by regeneration, so only the prose numbers need
  claims; for tables that were hand-transcribed — `tab_ood_v9.tex` — add a regeneration step from
  `E3.json` to `build.sh` instead);
- the script greps the tex for each `text`, evaluates the JSON path, and prints PASS/FAIL with the
  two values; exit code 1 on any FAIL;
- write the FAIL list to `results/NUMBER_CHECK.md` with, for each, what the JSON says. Do not fix the
  manuscript; the list is handed back.

## 3. Final packaging

- Tag the repository `v9-archive` after items 1–2 pass.
- `phase4_archive_<date>.zip`: the archive contents of item 1 (code, tests, scripts, results,
  figures PNG+PDF, tables, fixtures, README, MANIFEST, build.sh; **no** data files, no checkpoints),
  plus `results/ARCHIVE_CHECK.json` and `results/NUMBER_CHECK.md`. Keep it under 100 MB.
- `results/PHASE4_REPORT.md`: items vs status, the archive-check wall times, the number-check
  summary (claims checked / passed / failed), and anything you found inconsistent between the
  manuscript and the result files while writing the claims (this is the most useful part).
