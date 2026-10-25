# Phase 4 report: reproducibility archive and number check (v9)

Date 2026-10-25.  Brief: `CLAUDE_PHASE4.md`, carried out against the manuscript of `manuscript_v9_bundle.zip`
(sha256 0298b69d..., committed in e6ed506; `manuscript_v9/` read-only throughout).  No experiment was rerun except
E0.3/E0.4 (requested, see 1.4).  No network was retrained.  E5 was not done.  K = 100 everywhere.

## Items vs status

| Item | What | Status | Where |
|---|---|---|---|
| 1.1 | `build.sh` / `make all`: every table and figure from `results/*.json` | done: 10 table files (the 9 the manuscript inputs + `tab_pod_ranks_E1_v9.tex`), 10 figures (PNG + PDF), 4.5 s; no solver call, no data file | `build.sh`, `scripts/build_outputs.py` |
| 1.2 | PDF versions of the figures | done; written with fixed metadata, so they are byte-identical across builds | `cgmy_cert/figio.py` |
| 1.3 | clean-clone test (byte-identical tables, pixel-identical figures) | done at d971756 (full) and 3fff942 (build + tests, after the manuscript bundle and the seed fix) | `results/ARCHIVE_CHECK.json` |
| 1.4 | `run_E0.py --nproc 1` end to end + quick tests, wall times | done: 32.2 min on the workstation, laptop estimate 46-54 min (target < 1 h); 51 passed, 16 skipped | `results/ARCHIVE_CHECK.json` |
| 1.4a | non-deterministic seeding found (Python `hash()`), fixed (`zlib.crc32`); E0.3/E0.4 rerun with the fixed seeds | done (87447a5); a second E0.4 run under another `PYTHONHASHSEED` reproduces `tab_pcg_v9.tex` byte-identically | `results/DEVIATIONS.md` (Phase 4) |
| 1.5 | MANIFEST: SHA-256 of every archive file + excluded data/checkpoints with regeneration commands | done: 196 archive files (22.3 MB) and 116 excluded data files and checkpoints (27.1 GB), each with a regeneration command | `results/MANIFEST.json` -> archive, excluded |
| 1.6 | archive README: layout, environment, commands, paper item -> result file -> script | done, with the table and figure numbers of the current manuscript | `README.md` |
| 2 | claims file, `check_numbers.py`, `NUMBER_CHECK.md` | done: 264 claims, 230 pass, 34 fail (15 substantive, 14 without a result file, 5 rounding-level); the manuscript is not changed | `results/paper_claims.json`, `results/NUMBER_CHECK.md` |
| 3 | tag `v9-archive`, `phase4_archive_<date>.zip`, this report | done: tag `v9-archive`; `phase4_archive_20261025.zip`, 5.6 MB, 197 files (figures 10 PNG + 10 PDF), no data, no checkpoints | repository root |

The tag is placed although `check_numbers.py` exits 1. Its FAILs are statements in the manuscript, which this phase
must not edit. The checking machinery itself passes: every claim resolves, and every quoted text is found verbatim.
After the manuscript is corrected, `python3 scripts/check_numbers.py` should be rerun, and the tag moved if the
archive changes.

## Archive check (wall times)

| Step | Full check at d971756 (2026-10-23) | Recheck at 3fff942 (2026-10-25) |
|---|---|---|
| fresh venv + `pip install -r requirements.txt` | 0.3 min | 0.3 min |
| `bash build.sh` | 0.1 min; 7 tables byte-identical, 10 figures pixel-identical | 0.1 min; 10 tables byte-identical, 10 figures pixel-identical |
| `run_E0.py --nproc 1` | 32.2 min (laptop estimate 46-54 min) | not repeated (E0.3/E0.4 were rerun in the working tree with the fixed seeds; E0.1, E0.2 and E0.5 are unchanged) |
| quick tests | 2.6 min, 51 passed, 16 skipped (data-dependent tests skip without `data/`) | 2.6 min, 51 passed, 16 skipped |

Host: Intel Core Ultra 9 285K, one thread, Python 3.10.12, numpy 1.26.4, scipy 1.15.3, torch 2.14.0+cpu.

## Number check summary

264 claims from the abstract, Sections 2.5, 4.1, 4.3, 4.4, 4.5, 5.1-5.10 (including the reproducibility paragraph),
6 and `theory/*.tex`.  Section 2.7 states no numbers.  Table bodies are not claimed: all table files are regenerated
byte-identically from the result files by `build.sh`.  This includes `tab_ood_v9.tex` (the hand-transcribed table
the brief names), which `run_E3.do_table_ood` now builds from `E3.json`.

* **passed 230;**
* **failed 34:** 15 substantive, 14 numbers that no result file holds, 5 rounding-level (off by at most one unit of
  the last digit).

Each claim names its result file and path. Regenerate with
`python3 scripts/make_paper_claims.py && python3 scripts/check_numbers.py`.

## Manuscript vs result files: what is inconsistent

### A. Statements the result files contradict

1. **Abstract and Section 6: "no false accept occurs in $10^4$ in-distribution ... queries"** (6: "no false
   accept in $10^4$ in-box"). The $10^4$-query run (E4) has no reference solutions, and Section 5.10 says so itself.
   False accepts were checked on 150 in-distribution queries (validation30 in E1, bench120 in E2) and on the 264 OOD
   queries (E3). All of those show none.
2. **Abstract: "264 out-of-distribution queries, on 92% of which the reduced model is still accepted".** As written,
   this says 92% of 264. In fact r = 80 is accepted on 193/264 = 73% (r = 64: 181/264 = 69%). The 92% is 193/210,
   the share of the queries on which the model is within one cent, which is how Sections 5.9 and 6 state it.
3. **Abstract "fail ... by a factor of at least 47"; Section 6 "comes within a factor 47".** The median exact bounds
   of the three space-time POD-NN seeds are 49.1, 47.2 and 46.0 eps_N; the median over seeds is 47.2 and the pooled
   median 47.5. "At least 47" fails for seed 2.
4. **Section 2.5: principal-period error "0.33--0.41 at every resolution".** E0.1 gives 0.308-0.410: per N,
   64: 0.35-0.41, 128: 0.33-0.41, 256: 0.31-0.40, 512: 0.31-0.40. Phase 1 reported 0.31-0.41.
5. **Section 5.2: "$\varepsilon_K$ is within a factor 1.1--4 of the observed error at every K and N".** True at
   K = 100 (1.14-3.69). At K = 20 the factor is 1.30-4.43, and at K = 5 it is 2.07-8.58.
6. **Section 4.4: "The 800 trajectories supply 48,800 snapshots".** At N_T = 240 the count is
   800 x 241 = 192,800; 48,800 = 800 x 61 is the old N_T = 60 count.
7. **Section 5.7: "five times more training data does not enlarge the space noticeably (singular values within
   10%)".** The normalised singular values sigma_r/sigma_1 of the 800 and 4000 bases agree within 9% up to r = 48,
   but differ by 17-49% at r = 56-128 (`N1024.json` podext).
8. **Section 5.7: "The exact certificate has effectivity 9--15 for $r\le80$".** At r = 8 it is 6.8 (Table 6 row 8);
   the range 9.5-14.9 holds for 16 <= r <= 80.
9. **Section 5.7: best standard POD-NN "median exact bound of 1.25".** The value is 1.264 (median over seeds; seed
   medians 1.236, 1.264, 1.265; pooled 1.241). Ten lines later the same section says 1.26 (consistent with
   `E1.json`).
10. **Section 5.8: direct solve "370 ms cold".** On the benchmark (E2, bench120, cold setup) it is 337 ms. 369.9 ms
    is the E1 timing on validation30[1:11].
11. **Section 5.8: with the fine spectral data the certified solve would be "2.87 times" the uncertified one.**
    `E2_dryrun_fine_spectral.json` gives a geometric mean of 2.92 (CI 2.88-2.96) and a ratio of medians of 2.91.
    The 2.87 appears only in `PHASE3_REPORT.md` and `NOTES_FOR_PAPER.md` item 29, so those two are wrong too.
12. **Section 5.9: "17 false rejects from the effectivity of about ten".** The median effectivity of the exact
    certificate for r = 80 on the 264 OOD queries is 6.3.
13. **Section 5.7 timings: "the reduced solve takes 0.75--2.0 ms".** `E1.json` gives 0.73-1.97 ms. Section 4.4
    quotes the other run (`N1024.json`: 0.72-1.92 ms, "0.7--1.9"); see C.2.
14. **Reproducibility paragraph: "66 tests".** The suite collects 67 (51 passed + 16 skipped without `data/`; all 67
    run with the data).
15. **Remark rem:coercivity: "the explicit constant $c_1=5.1\times10^{-5}$".**
    `results/coercivity_c1_provable.json` (Phase 1, the provable constant of the explicit argument) gives
    3.06e-5 at (C, G, M, Y, r) = (0.05, 5, 9, 1.8, 0.01). Either the remark or the file uses a different form of
    (eq:c1-explicit); this should be checked against Appendix app:proof-uniform.

### B. Rounding-level (one unit in the last digit)

* 4.4 certificate time at r = 64 is "4.6 ms" (`N1024.json` 4.55; `E1.json` 4.61).
* 5.6 "the discretisation error at N = 1024 is 0.24 cent": it is 0.235, so "below 0.24" (stated two lines earlier)
  is right.
* 5.7 space-time error "0.015": 0.0159 (seed median) or 0.0155 (pooled).
* 5.7 "2.9--8.2 ms": `E1.json` gives 8.28 at r = 128.
* 6 "5--6% of the direct solve at every resolution": 4.45-5.60%; 5.8 states it correctly as 4.5-5.6%.

### C. Two result files, two numbers (the manuscript mixes them)

1. **Table 6 and the 5.7 text vs Figures 5 and 6.** Table 6 (`tab_pod_ranks_v9.tex`, generated by
   `make_tables_phase2.py`) and the r >= 96 sentence of 5.7 read `N1024.json` (Phase 2, follow-up 5). Figure 5
   (F5a), Figure 6 (F5c) and Section 5.9's validation points read `E1.json` (Phase 3 E1). The two agree to <= 1e-3
   relative for r <= 80. At r = 96 and 128 the true errors are 4.3e-8 / 2.3e-8 (`N1024.json`) against 3.6e-8 /
   9.9e-9 (`E1.json`); the reduced effectivities are 143 / 119 against 231 / 417; and the exact effectivity at
   r = 128 is 4.6 against 4.3. At these ranks the error is at the floor of the references (`E1.json`
   max_reference_bound 2e-9). The "120--140" of 5.7 is the `N1024.json` value. Use one file for the table and the
   figure.
2. **Online timings in 4.4 vs 5.7.** 4.4 ("2.9, 3.8, 4.6, 5.5, 6.7, 8.2 ms"; solve "0.7--1.9") quotes `N1024.json`
   timing_ext (idle, 2026-10-21). 5.7 ("2.9--8.2", "0.75--2.0", 154 ms) and Figure 6 quote `E1.json` timing (idle,
   Phase 3: 2.90, 3.87, 4.61, 5.63, 6.79, 8.28 ms). They differ by <= 3%, but the paper should use one of them
   (E1, which the figure shows).
3. **4.1: "about 1.6 ms" for eps_Q on the 161 nodes.** This is the Phase 2 measurement taken under load
   (`E2_3_reduced_cert.json` timing_v2). On the idle machine at N = 1024, Q = 36 it is 1.33 ms (`N1024.json`
   checks/timing_ms); together with ell it is the 2.7 ms of Table 10.

### D. Numbers no result file holds

* 2.5 the L-aliasing below 1e-18 on the box (analytic; E3 measures the change on doubling L, 5.5e-16 max);
* 4.1 the allowance "of the order 1e-8 ||c|| for Q ~ 25" (production uses Q = 36/37);
* 4.3 the DeepONet's 80,002 parameters;
* 4.4 "the residual is 1e-8 of the terms" (Gram form) and the float32 bound "of the order 1e-3 at N = 512" (only
  in `NOTES_FOR_PAPER.md` item 8);
* 5.4 the LU/GMRES agreement to 1e-12 (a test, not a result file; E0.5 has 1.5e-12 for the three Y < 1 queries) and
  the PCG bracket widths below 1e-20 and 1e-18 (E0.4 stores iteration counts only);
* 5.6 "price gaps above two dollars at N = 64" (the calibration starts at N = 128, where the corners are at 0.70 and
  0.78 dollars);
* 5.10 "the POD used four BLAS threads" (not recorded);
* the 459-coordinate regression and the 459-point audit (v8, `v8_frozen/`);
* thm_eim_cert "it reaches 1e-4 at N = 512" (an estimate).

### E. Statements about the method that differ from what was run

1. **Section 4.3 training objective.** (4.17)-(4.18) define L_res with the exact S(mu)^{-1}, squared and averaged
   over steps, and the hybrid loss L_data + lambda_R L_res. The POD-NN was trained with the relative least squares
   in reduced coordinates + lambda_R x mean(eta^{X-bar}), i.e. the reduced online bound in the cell norm, not
   squared (`scripts/train_podnn.py`, `tab_training_v9.tex` caption). Section 4.4's POD-NN paragraph describes this
   correctly; 4.3 should say that (4.17) is the motivation and 4.4 the objective used.
2. **Section 4.4 reduced online certificate.** The paragraph first states the Gram-block form ("(Q+2)^2 Gram blocks
   ... costs O(Q^2 r^2)") and then says that form is unusable and the R-factor form is used. The cost model of 4.5
   (rho (Q+2)(2r+1)) is the R-factor one. The O(Q^2 r^2) sentence describes a method that is not run.
3. **Reproducibility paragraph.** It says the archive contains "the trained checkpoints". By the Phase 4 brief the
   zip excludes them: they are listed with SHA-256 and regeneration commands in `MANIFEST.json`. The
   `\todo{Phase 4.4: archive URL and the end-to-end regeneration check}` can now cite `ARCHIVE_CHECK.json`
   (clean clone, byte/pixel-identical rebuild, E0 in 32 min).
4. **5.8: "(90 ms, Table tab:cost)".** The 90 ms initial projection is in `cost_model.json` remarks, not in Table 10.
5. **Populations (Table 2) and the Figure 3 caption say 288 verification queries; 5.3 says 289 x 3 pairs.** 289 = 288
   + the starred worst audit vertex. Consistent once explained, but not explained.
6. **Table 4 (tab:pcg-v9).** The bundle's copy had the hash-seeded iteration counts. The archive's table
   (87447a5) differs in plain CG by up to 6 iterations (N = 4096 maximum 1909 -> 1915; medians by <= 2) and by
   one in the N = 64 PCG maximum (9 -> 8); the text's "5--9" GMRES range is unchanged.

## Deliverables

* `results/ARCHIVE_CHECK.json`: the clean-clone check (full run at d971756; the recheck at 3fff942 under
  `recheck`);
* `results/MANIFEST.json`: archive and excluded sections;
* `results/paper_claims.json`, `results/NUMBER_CHECK.md` / `.json`; `scripts/make_paper_claims.py`,
  `scripts/check_numbers.py`;
* `phase4_archive_20261025.zip` (5.6 MB, 197 files), built by `scripts/make_phase4_zip.py` from the manifest's file list.
  The data files (data/) and checkpoints are not included;
* git tag `v9-archive`. Nothing was pushed.

## Addendum: follow-up 1 (number check closed against the corrected manuscript)

* **Manuscript checked:** commit `fd561a3` ("paper-side corrections after NUMBER_CHECK", the bundle unpacked
  unchanged). Claims and checker: `830aede`. Clean-clone build-only check: `830aede`, recorded in
  `ARCHIVE_CHECK.json -> build_only_check`. It gives 9 tables byte-identical and 10 figures pixel-identical; venv + pip
  0.3 min, build 4.4 s. `run_E0.py` and the tests were not rerun there, and the earlier wall times are kept (32 min, 2.6 min).
  Quick tests run locally once: 63 passed, 3 skipped, 1 xpassed (67).
* **Claims checked: 291. Passed: 288**, of which 274 come from result files and 14 are source-backed. Failed: 3
  (2 substantive, 1 rounding-level). There are no "text not found" and no "no source" claims.
* **The source-backed claims** are listed in their own section of `NUMBER_CHECK.md`. Their evidence:
  * the query fixture (459, twice);
  * the test suite (LU/GMRES 1e-12; bracket widths 1e-20 / 1e-18, three claims);
  * the v8 DeepONet checkpoint (80,002 parameters, counted from `model_state_dict`, which loads with `weights_only=True`);
  * the frozen v8 `tab_corner.tex` (N = 64 corner gaps 2.22 and 2.07 dollars);
  * `cgmy_cert.params` (aliasing exponent 48 >= 46);
  * `DEVIATIONS.md` (Gram residual 1e-8);
  * `NOTES_FOR_PAPER.md` (float32 1e-3 and the three allowance sizes of Remark 4.3).
* **Still disagreeing with a result file** (the manuscript is unchanged; paths and values are also in `NUMBER_CHECK.md`):
  1. Section 5.7, "raises the reduced effectivity to $120$--$140$" at r >= 96. Table 6 is now built from `E1.json`,
     where `eval/summary/pod_galerkin_r96_n800/reduced/median_effectivity` = 230.6 and the r128 value = 416.8.
     120-140 are the Phase 2 numbers (`N1024.json table_ext/800/summary/{96,128}/reduced/median_effectivity` = 143, 119),
     which the table no longer shows. Suggested: "to $230$--$420$". The "$4\times10^{-8}$" in the same sentence
     still holds at r = 96 (3.6e-8), but at r = 128 E1 gives 9.9e-9.
  2. Section 5.7, "bounds $0.055$--$0.058$": the smallest per-seed median is
     `E2_4_podnn_N1024.json results/podnn_N1024_r48_n4000_lamR1_seed2_spacetime/summary/exact_AI/median` = 0.05447,
     which rounds to 0.054. Suggested: "$0.054$--$0.058$".
* **Remark 4.3 (theory/thm_eim_cert.tex):** no result file holds the allowance sizes $7\times10^{-5}$ / $4\times10^{-4}$
  (Q = 29) and $4\times10^{-6}$ (Q = 37). They are not in `E2_3_reduced_cert.json`, `eim.json` or any other `results/*.json`. Their
  only record is `NOTES_FOR_PAPER.md` item 10 (6.8e-5, 4.3e-4, 3.9e-6), against which they are registered. Reword
  them, or regenerate them into a result file.
* **Where the brief and the repository differ:**
  * `tab_pod_ranks_E1_v9.tex` was not byte-identical to the new `tab_pod_ranks_v9.tex`: the numbers were the same but
    the layout and label differed. It is deleted as asked; `make_tables_phase2.py` is now the only writer of Table 6.
  * `coercivity_constant_provable.py` reproduces the committed `c1_provable` to 1 ulp (5.107198991695751e-05 against
    5.1071989916957514e-05); the committed JSON is kept.
  * The suggested v8 regex ("by more than \$2 at two box corners") does not occur in v8's `main.tex`, which says
    "price gaps exceeding \$2". The claim reads the generated v8 table instead. Its run directory is not in
    `v8_frozen/runs`; only the hash of its `cos/prices.csv` is in `evidence_v6.json`.
  * The Gram-form ratio and the float32 bound have no JSON record, so they are read from `DEVIATIONS.md` and
    `NOTES_FOR_PAPER.md`.
* **How some numbers are read:**
  * "27~GB" is binary (29.0e9 bytes = 27.05 GiB; 29 GB in decimal).
  * "effectivity $9$--$15$" is checked as a range that encloses the medians 9.53-14.90; rounded endpoints would read 10-15.
  * Section 3 was outside the Phase 4 claim scope. Its PCG sentence changed with the E0.4 rerun, so three claims were
    added: 8-12 and 42-1915 iterations, and the 1e-18 bracket width.
* **Not covered by claims:**
  * The sentence after the algorithm figure (`alg:pcc`) says PCG takes "at most $15$ iterations in all reported runs". `tab_pcg` gives at most 12 for PCG; 15 is
    the GMRES maximum.
  * Figure 6's legend shows the E1 cold direct solve (370 ms, validation30[1:11]), while Section 5.8 quotes E2's 337 ms
    on bench120.
  * The reproducibility paragraph still carries `\todo{Phase 4.4: archive URL ...}`.
* **Deliverables:**
  * `results/MANIFEST.json`: 196 archive files, 22.4 MB; 116 excluded files, 27.1 GB.
  * `phase4_archive_20261025_final.zip`: about 5.6 MB, 197 files, built by `scripts/make_phase4_zip.py --suffix _final`.
    No data files and no checkpoints.
  * Tag `v9-archive`, moved with `git tag -f` to the commit that adds this addendum. Nothing was pushed.

**Addendum, follow-up 2.** The manuscript was checked at `e8cc432`; claims, code and the build-only clean clone at `7a32779` (9 tables byte-identical, 10 figures pixel-identical). 294 claims: 294 pass (280 from result files, 14 source-backed), 0 fail, and `check_numbers.py` exits with code 0.
The Figure 6 (F5c, not F6a) legend now reads E2.json (337 ms cold, 259 ms reusable), checked by two claims. The Remark 4.3 allowance sizes have no result-file record and stay note-backed (NOTES_FOR_PAPER item 10); details are in DEVIATIONS.md.
