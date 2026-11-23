# Claude Code brief: Phase 4 follow-up 1 (close the number check, final archive)

Ground rules of `CLAUDE_PHASE4.md` still apply: you read `manuscript_v9/` but do not edit it; nothing is
rerun or retrained. The 34 number-check items of `results/NUMBER_CHECK.md` have been handled on the
paper side (15 substantive and 5 rounding items corrected in the text; of the 14 "no result file"
items two were removed from the text, two were reworded with sources, and the remaining ten are
facts that live in tests, fixtures or v8 records and only need to be registered). This brief makes
`check_numbers.py` report zero FAILs against the corrected manuscript and refreshes the archive.

## 0. Pull the paper-side bundle

`manuscript_v9_bundle.zip` is attached. Unpack it over the repository **after stripping the top-level
`cgmy_v9/` folder** (`unzip -o manuscript_v9_bundle.zip && cp -r cgmy_v9/* . && rm -r cgmy_v9`). It
carries:

- `manuscript_v9/` (main_v9.tex + theory/*.tex, the corrected text; figures are now included as the
  `.pdf` versions you produced, with the `.png` as fallback);
- `tables_v9/` and `figures_v9/` — identical to your archive except `tab_pod_ranks_v9.tex`, which is
  now generated from `E1.json` (see item 1);
- `scripts/make_tables_phase2.py` — the E1-based version (reads
  `E1.json["eval"]["summary"]["pod_galerkin_r{r}_n800"]`, eps key `"0.001185"`); it supersedes the
  N1024-based one in the archive;
- `scripts/coercivity_constant_provable.py` and `results/coercivity_c1_provable.json` — the
  explicit coercivity constant of Appendix A.1 with the bound actually stated there,
  `c1 = min_box min{ r/(1+2^{1+Y/2} a_*^Y), C kappa_Y/2^{Y/2} }`, `a_* = max(G_max, M_max) = 9`,
  giving `c1_provable = 5.107e-5` (low-frequency branch, Y = 1.8). Your archive's
  `coercivity_c1_provable.json` (3.06e-5) evaluated a looser bound and is replaced.

Commit the bundle as one commit ("paper-side corrections after NUMBER_CHECK") before touching anything
else, so the manuscript state that the claims are checked against is a single commit hash.

## 1. Build script: one writer per table

`scripts/build_outputs.py` writes `tables_v9/tab_pod_ranks_v9.tex` twice (`run_E1.do_figs`, then
`make_tables_phase2.py`), and the archive also carries `tab_pod_ranks_E1_v9.tex`, which is byte-identical
to the new `tab_pod_ranks_v9.tex`. Make `make_tables_phase2.py` (E1-based) the only writer of
`tab_pod_ranks_v9.tex`; drop the table write from `run_E1.do_figs` (keep its figures) and delete
`tab_pod_ranks_E1_v9.tex`. Run `python3 scripts/coercivity_constant_provable.py` and confirm it reproduces
the committed JSON (5.1071989916957514e-05). Then `bash build.sh` and check that every table in
`tables_v9/` is byte-identical to the committed one and every figure pixel-identical (the
`archive_check.py` diff step, without the clean clone and without `run_E0.py`).

## 2. Update the claims to the corrected text

Every claim whose `text` no longer occurs verbatim in the manuscript must be re-pointed; the checker's
"text not found" status will show them. The corrected statements and their sources are:

| id (old) | corrected text in the manuscript | source |
|---|---|---|
| abs_92pct, s59 | abstract: "accepts 193 of the 210 queries on which the reduced model is still within one cent"; 5.9: "on 210 of the 264 queries, and the certificate accepts 193 of them (92\%)" | E3.json works / certified for podg80 (as in your note) |
| abs_factor47, s6_factor47 | "by a factor of at least $46$" / "comes within a factor $46$" | min over the three space-time seeds of median exact bound / eps_N (45.98), `rel: ge` with `tol: round` |
| s25_pp_lo | "$0.31$--$0.41$" | E0.json pp min/max |
| s44_snapshots | "192,800 snapshots" | 800 (N_T+1) |
| s52_factor_hi | "within a factor $1.1$--$3.7$ of the observed error at $K=100$" (and the sentence now quotes $4.4$ at $K=20$ and $8.6$ at $K=5$) | E0.json eps_K/K per K |
| s57_sv10 | "within $10\%$ up to $r=48$ and within a factor $1.5$ at $r=128$" | N1024.json sv_ratio 4000 vs 800 |
| s57_effS_lo | "effectivity $9$--$15$ for $16\le r\le80$ ($6.8$ at $r=8$)" | N1024.json / E1.json medians |
| s57_best_bound_1.25 | "a median exact bound of $1.26$" | E1.json selection, median of the per-seed medians |
| s57_solve_lo | "$0.7$--$2.0$~ms" (both in 4.4 and 5.7) | E1.json timing/per_rank reduced_solve (0.733–1.971) |
| s58_direct_cold_370 | "$337$~ms cold" | E2.json cold direct median |
| s58_fine_2.87 | "$2.92$ times" | E2_dryrun_fine_spectral.json geo_mean_ratio |
| s59_eff_ten | "effectivity of the exact certificate (median $6.3$)" | E3.json podg80 exact bound / true_err median |
| srep_tests66 | "67 tests" | ARCHIVE_CHECK.json |
| th_c1_explicit | "$c_1=5.1\times10^{-5}$" | coercivity_c1_provable.json: c1_provable (new file) |
| s44_cert_ms_r64, s57_cert_hi | 4.4 and 5.7 both quote E1 now: "$2.9$, $3.9$, $4.6$, $5.6$, $6.8$, $8.3$~ms" and "$2.9$--$8.3$~ms" | E1.json timing/per_rank reduced_certificate |
| s56_disc_0.24 | "$0.235$ cent" | E3_1_calibration.json |
| s57_st_err | "errors $0.014$--$0.016$ and bounds $0.055$--$0.058$ over three seeds" | E2_4_podnn_N1024.json space-time seeds: min/max over seeds of the per-seed medians (four claims, `tol: round`) |
| s6_frac_lo | "about $5\%$ of the direct solve at every resolution" | E2.json refine per_N (4.45–5.60%): `tol: abs:1` |

Two "no result file" items were removed from the text and their claims are dropped: `s41_allow_Q25`
(the $10^{-8}$ for $Q\approx25$ sentence) and `th_eim_1e-4`, which now reads "$7\times10^{-5}$ (median) to
$4\times10^{-4}$ (maximum) in the $M_I$ norm at $N=512$ with $Q=29$ … below $4\times10^{-6}$ with
$Q=37$" (Remark after Theorem 4.2, `theory/thm_eim_cert.tex`). Find the result file that holds these
allowance sizes (E2_3_reduced_cert.json or eim.json, Phase 2 item 2.3) and register them; if no file
holds them, say so in `NUMBER_CHECK.md` and I will reword.

Two were reworded with a computable source:

- `abs_no_fa_1e4`, `s6_no_fa_1e4`: now "No false accept occurs on any query with a reference solution
  (150 in the parameter box, 264 outside it)" and "no false accept on any of the $150$ in-box and $264$
  out-of-box queries". Claims: 150 = validation30 (E1.json) + bench120 (E2.json) query counts; 264 =
  E3.json n_queries; and the false-accept counts in E1, E2 and E3 are all zero (three `rel: eq` claims
  with value 0).
- `s25_alias_1e-18`: now "with $L=4N_h$ the exponent is at least $46$ on the box". Claim: expr
  `3 * BOX['G'][0] * (X_MAX - X_MIN)` from `cgmy_cert.params` (= 48; the text says at least 46 because
  $L-N_h$ is measured in free-section units), `rel: ge`. The checker may import the package for this.

## 3. Register the ten source-backed facts

Extend `check_numbers.py` with two source kinds, so that no claim is left with `"json": null`:

- a `"file"` source relative to the repository root (not `results/`), for JSON files elsewhere:
  `tests/fixtures/queries.json`, path `populations/audit459/@len` → 459 (`srep_459`, `s6_audit459`);
- a `"regex"` source: `{"file": <path>, "regex": <pattern with one capture group>}` whose capture is
  the number. Use it for:
  - `s54_lu_gmres` (1e-12): `tests/test_direct_certificate.py`, `assert d <= (1e-12) \* scale`;
  - `s54_bracket_1e-20`, `s54_bracket_1e-18`: `tests/test_spectral_riesz.py`,
    `\(1e-20 if N <= 1024 else 1e-18\)` (two claims, two capture groups or two patterns);
  - `s44_resid_1e-8` (Gram form, residual $10^{-8}$ of its terms): the number in
    `results/DEVIATIONS.md` or `results/NOTES_FOR_PAPER.md` (Phase 2, R-factor decision); if
    `E2_3_reduced_cert.json` or `scripts/check_reduced_cert_longdouble.py` recorded the cancellation
    ratio, prefer that as a JSON source;
  - `s44_float32_1e-3`: `results/NOTES_FOR_PAPER.md` item 8 ("~1e-3 from float32 rounding alone");
  - `s43_deeponet_params` (80,002): count the parameters of the v8 checkpoint directly —
    `sum(v.numel() for v in torch.load('v8_frozen/models/deeponet_final.pt', map_location='cpu')
    [state-dict key].values())` — as an `expr` claim; fall back to the regex on
    `v8_frozen/manuscript_v8/main.tex` ("80,002 trainable parameters") only if the checkpoint does not
    load;
  - `s56_corners_2dollar` ("price gaps above two dollars at $N=64$"): find the v8 run that recorded the
    N = 64 price gaps at the two corners (C = 0.05, Y = 1.1, T = 0.1; `v8_frozen/runs/*` price or
    calibration CSVs) and register the maximum gap with `rel: ge`; if no run file holds it, use the regex
    on `v8_frozen/manuscript_v8/main.tex` ("by more than \$2 at two box corners") and say so in the
    note.

Mark these claims with `"kind": "source-backed"` in the claims file and list them in a separate
"Source-backed (tests, fixtures, v8 records)" section of `NUMBER_CHECK.md`, so a reader sees which
numbers rest on which kind of evidence.

## 4. Re-run and report

`python3 scripts/make_paper_claims.py && python3 scripts/check_numbers.py` — expected: 0 FAIL, 0 "text not
found", 0 "no source". Also `python3 -m pytest tests -q` (quick mode) once, since `coercivity_c1_provable.json`
changed. Then:

- `python3 scripts/make_manifest.py`; `git tag -f v9-archive` on the new head;
- `python3 scripts/archive_check.py` in **build-only mode** (clean clone + `build.sh` + diff; do not rerun
  `run_E0.py`, keep the earlier wall times in `ARCHIVE_CHECK.json` and add the new commit hash);
- `phase4_archive_<date>_final.zip` as before (< 100 MB, no data, no checkpoints), including the updated
  `NUMBER_CHECK.md`, `paper_claims.json`, `ARCHIVE_CHECK.json`;
- append a short section to `results/PHASE4_REPORT.md`: claims checked / passed / source-backed, the commit
  hash the check ran against, and anything in the corrected text that still disagrees with a result file.

If a corrected statement in the manuscript does not match the result file you find (for example the
allowance sizes of Remark 4.3 or the two-dollar corners), do not edit the manuscript: put the exact
JSON value and path in `NUMBER_CHECK.md` and in the report, and I will change the text.
