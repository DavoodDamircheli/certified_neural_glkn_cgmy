# Phase 3 report (v9 workspace, 2026-10-21/23): E1-E4 at the production configuration

Production configuration (frozen, `results/PRODUCTION_CONFIG.json`): N = 1024, N_T = 240, K = 100, eps_N = 1.18e-3,
sweep eps_N x {0.1, 0.3, 1, 3, 10}; 3 x 3 x 2 cells; Q matched to eps (Q = 37 for eps < eps_N, 36 otherwise); three-stage
rule. All wall-clock numbers: one performance core (core 3), one thread, idle machine (load averages stored), warm-up,
5 repetitions, shuffled order, medians of per-query medians, geometric-mean ratios with 10 000-resample query
bootstrap CIs. E5 not started; no network retrained.

## Headline numbers

* **Certificates never accept wrongly.** No false accepts in any experiment, under any certificate, at any eps:
  - E1: validation30, 37 candidates;
  - E2: bench120, 5 methods;
  - E3: 264 out-of-box queries;
  - E4: 10^4 queries, where no true errors are available (see deviations).
* **Full-order certificate: 6.0% of the direct solve** (residual + banded solves), **7.2% including the per-query
  spectral data**. Direct + certificate takes 1.06x the uncertified solve (E2, bench120). Before the fast online
  spectral mode it took 2.87x. The ratio is flat at ~5% from N = 64 to 4096 (F6b).
* **Certified POD-Galerkin r = 80: 0.032x the uncertified direct solve** (8.3 ms vs 259 ms, bench120, reduced accept
  on 120/120 at every eps). On 10^4 queries: 9.0 ms per query, 99.77% settled at the N-independent reduced stage,
  break-even with the direct solver after ~6 100 queries (C0 ~ 1530 s).
* **No neural candidate passes at eps_N.** The best, the space-time POD-NN, has a median exact bound of 0.055
  (47 x eps_N; true error 0.015). It is cheaply rejected and corrected on every query, at 1.07x the direct cost, and
  never breaks even.
* **Out of distribution** (E3): POD-Galerkin still works on 196 / 210 (r = 64 / 80) of 264 out-of-box queries, and
  the certificate finds 181 / 193 of them. The naive "reject everything outside the box" rule has precision
  0.26 / 0.20 against the certificate's 0.82 / 0.76.

Tests: full suite `CGMY_FULL_TESTS=1 python3 -m pytest tests`: **66 passed, 1 xfailed** (the documented strict 1e-10
equality at the eps*kappa floor), 59 min (`results/fulltest_phase3.log`).

## Items vs status

| Item | Status | Outputs |
|---|---|---|
| 0 cost model (plan 2.6) | done; one-constant fits plus two-constant fits (per-call overhead) where needed | `cost_model.json`, `tab_cost_v9.tex` |
| E1 (plan 3.2) | done: 37 candidates, Q matched to eps, three-stage statuses, bound- vs error-selected POD-NN | `E1.json`, F5a, F5b, F5c, `tab_pod_ranks_v9.tex` |
| E2 (plan 3.3) | done: bench120 x 5 methods x {cold, reusable} x eps_N {0.3, 1, 3}; refinement N = 64..4096 | `E2.json`, `E2_dryrun_fine_spectral.json`, F6a, F6b, `tab_benchmark_v9.tex` |
| E3 (plan 3.4) | done: 264 OOD queries, 5 candidates, works/certified per shell, naive-rule comparison | `E3.json`, F7 |
| E4 (plan 3.5) | done: 10^4 queries x 4 methods, offline cost, break-even | `E4.json`, `E4_progress.json`, F8 |
| follow-up 1 (fast spectral) | done: `spectral_data(mode="fast")`, `theta_bar(mode="fast")`, validated on spectral288; cold split | `spectral_fast.json` |
| E5 | not started (user decision) | |

## Numbers per plan done-when

* **E1:** acceptance at eps_N under reduced / banded / exact (POD-Galerkin, basis 800): r = 48: 0/0/0; r = 56: 0/9/28;
  r = 62: 24/30/30; r = 64: 26/30/30; r >= 80: 30/30/30. Exact effectivity 9-15 for r <= 80.
  POD-NN, 4000 trajectories, two selections:
  - bound-selected (lambda_R = 1): median exact bound 1.26, true error 0.46;
  - error-selected (lambda_R = 0): bound 12.2, true error 0.082;
  - space-time (lambda_R = 1): bound 0.055, true error 0.015.
  Online (idle): reduced certificate 2.9-8.3 ms for r = 8-128. Three-stage total at eps_N: 5-6 ms (r <= 48), 154 ms
  (r = 56), 4.4-8.3 ms (r >= 62). Uncertified direct solve: 370 ms cold, 167 ms warm.
* **E2:** reusable-setup ratios to direct:
  - direct + banded: 1.06;
  - POD-G r = 64: 0.55 / 0.041 / 0.026 at 0.3 / 1 / 3 eps_N;
  - POD-G r = 80: 0.032;
  - space-time POD-NN: 1.07.
  Cold: direct + banded 1.04, POD-G r = 80 0.30 (z_h^0 rebuilt inside the timer). Corrections: POD-NN 120/120;
  POD-G r = 64 4/120 at 0.3 eps_N. False accepts: 0.
* **E3:** 0 false accepts; all 264 certifiable; no aliasing adjustment needed. eps_Q out of box: median 2.0e-7
  (in box 1.7e-9). Per-shell counts in `E3.json` -> breakdown and NOTES item 27.
* **E4:** see headline; status fractions in `E4.json` -> methods.

## Deviations (full text in `results/DEVIATIONS.md`, section "Phase 3")

* **Cost model:** the one-constant fits miss the per-step interpreter overhead (direct 156%, reduced solve 638% max
  deviation); two-constant fits are added (11% and 4%).
* **E2's first run was discarded** after two bugs: a JSON lookup inside the timer, and an npz array re-read in the
  error loop. Its fine-spectral numbers are kept as `E2_dryrun_fine_spectral.json`.
* **Fast online spectral mode:**
  - it uses a sin^{2p}-weighted alias tail, which is rigorous with K' = 10;
  - alpha_lb and Theta-bar fall on the "unsafe" side of the brief's test for some queries, but both remain rigorous
    (the fine mode is looser there);
  - eta^X changes by +0.002%.
* **K = 100 kept in the assembly** by decision; the adaptive-alias assembly (36 vs 107 ms) is recorded as a remark.
* **POD-NN error-selection is across lambda_R:** only the bound-selected checkpoint of each run was saved.
* **E3 rules:**
  - the reduced stage is used only inside the box: the cell beta-bar is exact by monotonicity only up to the box
    maxima;
  - the ensemble proxy is uninformative, because all space-time POD-NN errors exceed eps_N;
  - no extended-box vertex has Y = 1, so the brief's Y-substitution was not triggered.
* **E4:** no true errors (no references for 10^4 queries); C0 partly estimated (references, cell beta-bar); training is
  the recorded parallel wall time.
* **About a day of idle machine time was lost** to a stuck helper (its `pgrep` pattern matched itself). No result was
  affected.

## Regeneration commands (one thread: `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; idle machine for timing)

| Output | Command |
|---|---|
| `cost_model.json`, `tab_cost_v9.tex` | `python3 scripts/run_cost_model.py` |
| `spectral_fast.json` (validation, cold split) | `python3 scripts/validate_spectral_fast.py ratios etax --nproc 14` ; `python3 scripts/validate_spectral_fast.py split` |
| `E1.json`, F5a-c, `tab_pod_ranks_v9.tex` | `python3 scripts/run_E1.py qeps tensors eval select --nproc 20` ; `python3 scripts/run_E1.py timing figs` ; `python3 scripts/fig_F5a.py` |
| `E2.json`, F6a, F6b, `tab_benchmark_v9.tex` | `python3 scripts/run_E2.py bench refine figs` |
| `E3.json`, F7 | `python3 scripts/run_E3.py run summary fig breakdown --nproc 19` |
| `E4.json`, F8 | `python3 scripts/run_E4.py run summary` |
| zip | `python3 scripts/make_phase3_zip.py` |
