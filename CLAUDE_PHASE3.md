# Claude Code brief: Phase 3 (E1–E4) at the production configuration

Read this file, then `CLAUDE_PHASE2.md` (its ground rules 1–9 still apply: no edits under
`manuscript_v9/` or `v8_frozen/`, results as self-describing JSON with provenance, never a verdict
at a single tolerance, deviations in `results/DEVIATIONS.md`, one commit per item, nothing pushed),
`results/PHASE2_REPORT.md` and `results/PRODUCTION_CONFIG.json`. Phase 2 is accepted as handed over.

**Fixed for every experiment below:** N = 1024, N_T = 240, ε_N = 1.18e-3, sweep ε_N × {0.1, 0.3,
1, 3, 10}, K = 100, Q matched to ε by the 2d rule, the 3 × 3 × 2 cells, the three-stage rule
(`certify_three_stage`). "True error" always means max_n ‖z^n − z_ref^n‖_{M_I} against the stored
(or freshly computed) GMRES reference, with the reference's own residual-derived allowance
reported next to it.

**Timing protocol (all wall-clock numbers that may enter the paper):** idle machine (no training,
no other jobs; check with `uptime` before and after and store the load averages), one thread
(`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(1)`,
`taskset -c <one performance core>`), one warm-up query outside the population, 5 repetitions per
query in a reproducibly shuffled method order, report medians of per-query medians and
geometric-mean ratios with 95 % query-bootstrap intervals (resample queries, 10 000 resamples).
Record CPU model, frequency governor, Python/numpy/scipy/torch versions, git SHA.

## 0. Cost model (plan 2.6) — do this first, it is cheap

`results/cost_model.json` and `tables_v9/tab_cost_v9.tex`: for N ∈ {256, 512, 1024, 2048, 4096}
and r ∈ {16, 32, 48, 64, 80, 128}, the measured component times (one thread) and the operation
counts they correspond to:
- direct solve: N_T × mean GMRES iterations × (FFT pair of length 2N_h + banded products);
- full-order certificate: residual (one FFT pair per step), banded η^X (one banded Cholesky + N_T
  banded solves), exact η^S (N_T × mean PCG iterations × FFT pair) — each separately;
- reduced solve: (Q+2) r² assembly + N_T r² march;
- reduced online certificate: ρ (Q+2)(2r+1) with the retained row count ρ, plus ε_Q and ℓ(ν);
- symbol work per query: assembly O(LK), `spectral_data`, ε_Q on the 161-point set.
Fit each measured time to its operation count (one constant per component) and report the fit
residual, so the paper's cost formulas carry measured constants. The reduced-model rows need
tensors only at N = 1024; for the other N state that the reduced cost is N-independent by
construction (ρ saturates at ≈ 350–420 rows) and show ρ(N) from the existing data.

## 1. E1 — reduced model and candidates on validation30 (plan 3.2)

Most of it exists from Phase 2; consolidate into `results/E1.json` and produce the figures.
Candidates: POD-Galerkin r ∈ {8, 16, 24, 32, 40, 48, 56, 62, 64, 80, 96, 128} (800 and 4000 bases),
the standard-output POD-NN (best checkpoint per λ_R, all three seeds where trained) and the
space–time POD-NN (3 seeds). Per candidate and query: β₀, η^{X̄} (reduced), η^X, η^S, true error,
effectivity of each, three-stage status and the stage at which it stopped, at every ε of the sweep.
**Report both the bound-selected and the error-selected POD-NN checkpoint** for each setting
(Phase 2 found that selection by bound picks a less accurate network than λ_R = 0; the paper
must show both, nothing is retrained).
Figures (`figures_v9/`): F5a acceptance vs ε per candidate for the three certificates (already
drafted; update), F5b effectivity histograms per certificate (POD-Galerkin r = 56, 64, 80 and the
space–time POD-NN), F5c online time vs r (solve, reduced certificate, three-stage total) with the
uncertified direct-solve time at N = 1024 as a horizontal line. Table `tab_pod_ranks_v9.tex`
(the rank × {true error, three accept counts, effectivities} table of follow-up 5).

## 2. E2 — matched benchmark on bench120 (plan 3.3)

Methods, all at the production configuration, cold and reusable setup as in v8 (cold = per-query
symbol work, assembly, spectral data and factorisations inside the timer; reusable = only
query-independent data cached: mesh, POD bases, offline tensors, EIM columns):
1. direct: circulant-preconditioned GMRES, uncertified — **the baseline of every ratio**;
2. direct + banded certificate η^X (to show the certificate fraction);
3. POD-Galerkin r = 64 with the three-stage rule, and correction fallback when rejected;
4. POD-Galerkin r = 80 with the three-stage rule (expected: reduced accept on all);
5. space–time POD-NN (best seed by validation) with the three-stage rule and correction fallback.
Correction fallback = full-order CN march from z_h^0 warm-started by the candidate (GMRES, same
tolerance as the direct solve), then the banded certificate on the corrected trajectory; report
correction counts and the time split (candidate / certificate / correction / recertify).
Report at ε_N × {0.3, 1, 3}: acceptance counts by status, correction counts, false accepts
(must be 0), geometric-mean time ratio to method 1 with bootstrap CI, and for method 2 the
certificate fraction of the solve time. `results/E2.json`, table `tab_benchmark_v9.tex`, figure
F6a (ratios with CIs, one panel per tolerance).

**Refinement (F6b):** N ∈ {64, 128, …, 4096} on 20 bench queries: direct-solve time,
full-order banded certificate time, their ratio, and the reduced online certificate time at
N = 1024 drawn as a horizontal line with the note that it does not depend on N. Use N_T = 240
throughout; this replaces v8 Table 8 / Fig. 3.

## 3. E3 — out-of-distribution detection (plan 3.4)

Queries: 200 Sobol (seed 7) in the union of the shells C ∈ [1, 2], Y ∈ [0.9, 1.1) ∪ (1.8, 1.95],
G ∈ [0.5, 1), M ∈ [9, 15] (each query outside the box in at least one coordinate, the others drawn
from the box), excluding |Y − 1| < 0.02 (Γ(−Y) is singular at Y = 1); plus the 64 vertices of the
extended box with Y = 1 replaced by 0.98 / 1.02. T and r from the box.
Rules specific to OOD queries, record them in the JSON:
- the reduced (cell) stage applies only if the query lies in one of the 18 cells; otherwise the
  three-stage rule starts at the banded stage with the query's own `spectral_data`;
- ε_Q is evaluated at the query (it is rigorous anywhere; expect it to grow outside the EIM
  training box) and enters the reduced-model allowance;
- for G < 1 check that L = 4 N_h keeps the Toeplitz aliasing below 1e-10 (|t_m| ~ e^{−Gh m});
  raise L for the affected queries if not and say so;
- references by direct GMRES at the query (N = 1024, N_T = 240), true error as above.
Candidates: POD-Galerkin r = 64 and r = 80, space–time POD-NN (3 seeds). Proxy: seed-ensemble
disagreement of the space–time POD-NN, max_n ‖z_seed_i − z_mean‖_{M_I}, thresholded at ε_N.
Output: false accepts per certificate (must be 0 by theorem), false rejects, proxy precision and
recall for "true error > ε_N", ε_Q and α_lb distributions in vs out of the box. Figure F7: log–log
scatter of true error vs certified bound (three certificates, colour by in/out of box), with the
lines ε_N and bound = error. `results/E3.json`.

## 4. E4 — multi-query demonstration (plan 3.5)

10⁴ Sobol queries (seed 4) in the box at the production configuration, timing protocol above
(no repetitions needed; one pass, shuffled, idle machine). Methods: (i) direct (uncertified);
(ii) certified POD-Galerkin r = 80 (three-stage, correction fallback); (iii) certified POD-Galerkin
r = 64 (same); (iv) certified space–time POD-NN with correction fallback. Report fraction accepted
per stage, fraction corrected, total online time, and the offline cost C₀ split into reference
generation, POD, EIM, cell tensors and training (from the logs; training wall time on 24 threads,
stated as such). Figure F8: cumulative wall time vs number of queries with C₀ as intercept and
the break-even point per method; give the formula C₀/Q + C₁ fitted. `results/E4.json`.

## 5. Not now

Do not start E5 (American put); that decision is the user's. Do not retrain any network.

## 6. Deliverable

`results/PHASE3_REPORT.md` (items vs status, the numbers that answer each plan done-when, every
deviation, regeneration commands), and a zip `phase3_results_<date>.zip` containing code, tests,
scripts, results, figures_v9, tables_v9, DEVIATIONS.md, NOTES_FOR_PAPER.md, MANIFEST.json — **no
checkpoints (*.pt) and no data files**, hashes in MANIFEST.json, so that the zip stays below
100 MB for upload.
