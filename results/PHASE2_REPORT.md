# Phase 2 report (v9 workspace, 2026-10-20/21)

## The three decision points

* **Week 4: DeepONet dropped.** The v8 checkpoint, certified as a candidate of the new exact-assembly system
  (N = 64, N_T = 60, bench120, full-order with A_I), is rejected on 120/120 queries at every eps of
  1e-3 x {0.1, 0.3, 1, 3, 10}. Median true error 0.78 (relative 3.1e-3), median exact bound 3.4, no false accepts
  (`results/deeponet_old_vs_new.json`). It was not retrained, for two reasons:
  1. the v8 training script regenerates its validation set with the principal-period assembly, which rule 2
     forbids;
  2. it reads v8's HDF5 schema and cannot read the new npz references unchanged.
* **Week 8: production configuration N = 1024, N_T = 240, eps_N = 1.18e-3** (= 1 cent / max spot factor 8.44),
  sweep eps_N x {0.1, 0.3, 1, 3, 10}, frozen in `results/PRODUCTION_CONFIG.json`. The two v8 corner queries
  (C = 0.05, Y = 1.1, T = 0.1) decide N. With 60 time steps, the N = 1024 error is 4.6 cents.
  Max |V_h - V_COS| over ref30 + corners, spots 0.8K..1.2K, N_T = 240 (`results/E3_1_calibration.json`,
  `tables_v9/tab_budget_v9.tex`):

  | N | ref30 | Corner A | Corner B | max beta_I | eps (1 cent) |
  |---|---|---|---|---|---|
  | 512 | 0.11 cent | 2.4 cent | **3.2 cent** | 6.07 | 1.65e-3 |
  | 1024 | 0.16 cent | 0.14 cent | **0.24 cent** | 8.44 | **1.18e-3** |

  COS (8192 modes, range 30) agrees with Carr-Madan on 10 rows to <= 2.6e-12 absolute (5.6e-11 relative).
* **Week 10: no POD-NN passes at eps_N (nor at 1e-3).** Best median exact full-order bound on validation30 at
  N = 1024 (`results/E2_4_podnn_N1024.json`):

  | candidate | true error | exact | banded | reduced online |
  |---|---|---|---|---|
  | POD-NN, standard output (4000 trajectories, lambda_R = 1), seeds 0-2 | 0.46-0.47 | 1.24-1.27 | 2.04-2.08 | 4.8-4.9 |
  | **POD-NN, space-time output (r_t = 23)**, seeds 0-2 | 0.014-0.016 | **0.054-0.058** | 0.082-0.089 | 0.18-0.19 |
  | projection of the references (same basis) | 3.3e-4 | 3.2e-3 | 6.2e-3 | 1.3e-2 |
  | POD-Galerkin r = 48 | 3.5e-4 | 3.8e-3 | 8.2e-3 | 1.5e-2 |
  | N = 512 pilot (6 runs, stopped at ~1200 epochs; old cells) | | | | 5.8-7.7 |

  The best neural candidate is 47 x eps_N. POD-Galerkin reaches eps_N at r = 56-64 (exact certificate 28/30 at
  r = 56, all three certificates >= 26/30 at r = 64, 30/30 at r >= 80).

## Plan items vs status

| Item | Status | Where |
|---|---|---|
| 2.1 EIM | done; Q = 23/26/27/29/29 (1e-9, N = 64..1024), extended to 1e-11 (Q = 37 at N = 512 and 1024); eps_Q on an N-independent 161-node set | `cgmy_cert/eim.py`, `results/eim.json`, F4 |
| 2.2 references, POD-Galerkin | done at N = 512 (development) and N = 1024 (production): 4000 / 800 / validation30 / bench120 / ref30 references; POD from 800 and 4000 (r <= 128 at 1024); affine reduced model | `cgmy_cert/pod.py`, `results/E2_2_pod.json`, `results/N1024.json` |
| 2.3 reduced online certificate | done: R-factor form (the Gram form is unstable), SVD-compressed with a rigorous tail, cells 3 x 3 x 2 in (C, Y, min(G,M)) with corner-rule beta-bar; online certificate 3.8 ms at r = 48 (idle, one thread), N-independent by construction | `cgmy_cert/reduced_cert.py`, `results/E2_3_*.json`, `results/N1024.json` |
| 2.3+ three-stage rule | done (follow-up 5): reduced -> banded -> exact with a rigorous cheap rejection | `cgmy_cert/three_stage.py`, F5a |
| 2.4 POD-NN | done; done-when **not met** (week-10 answer above); pilot at N = 512 archived | `cgmy_cert/podnn.py`, `scripts/train_podnn.py`, `results/training_log.json`, `results/pilot_N512.json` |
| 2.5 DeepONet | decided: dropped (week 4) | `results/deeponet_old_vs_new.json` |
| 2.6 cost model (`cost_model.json`) | **not done**: none of the follow-ups requested it. The measured component timings it would be built from are in `E2_3_reduced_cert.json` (timing_v3) and `N1024.json` (checks.timing_ms, timing_ext) | open |
| 3.1 calibration | done (ahead of Phase 3, follow-up 3) | `results/E3_1_calibration.json`, `PRODUCTION_CONFIG.json` |
| v8_frozen restore | done from the zip (sha256 verified); the 4 v8 regression tests run and pass | commit 1b58600 |

Tests: `python3 -m pytest tests -q` green (xfail: the strict 1e-10 equality on POD-Galerkin residuals, see
deviations). Full suite `CGMY_FULL_TESTS=1 python3 -m pytest tests`: **66 passed, 1 xfailed** (the documented strict
equality), 58 min (`results/fulltest_phase2.log`).

## Key numbers at the production resolution (N = 1024; `results/N1024.json`)

* EIM: operational Q = 36 at eps_N, 37 at eps_N / 10.
* Cells: X-bar <= S on 400 Sobol queries (min c_mu = 1.00016; the margin vanishes at theta = 0 by construction).
  Cell vs own-norm effectivity: median 1.87.
* Checks: full basis vs direct 1.6e-13. Online vs full-order: 2.7e-15 (perturbed candidates) and 2.0e-10
  (POD-Galerkin, the eps x kappa floor). The compressed bound is <= 1.0005 x the full one.
* POD-Galerkin on validation30 (both bases, r = 8..128): exact effectivity 9-15 for r <= 80. Accepted at eps_N
  under reduced / banded / exact: r = 56: 0/9/28; r = 62: 24/30/30; r = 64: 26/30/30; r >= 80: 30/30/30. The
  three-stage rule settles r <= 48 by cheap rejection and r >= 80 by reduced acceptance. No false accepts under
  any certificate, rank or eps.
* Online certificate time (idle, one thread, Q = 36): 2.9 ms (r = 8) to 8.2 ms (r = 128); crosses 5 ms at
  r ~ 74, never reaches 50 ms. Retained rows rho = 185-1022. Offline tensors (18 cells): 9 MB (r = 8) to 688 MB
  (r = 128).

## Deviations (full text and numbers in `results/DEVIATIONS.md`)

2.1: Q = max(Q_tol, Q_op); stored basis-function columns; symmetrised family; N-independent eps_Q (161 nodes; up to
2.1x looser than the L-grid version). 2.2: allowance on the raw coefficient vector; the r = 48 sanity check does
not reproduce the v8 acceptance pattern (a result). 2.3: R-factor form instead of the Gram tensor; 1e-10 equality
floor (strict test xfail, extended-precision evidence attached); beta_0 by Pythagoras; c_mu optional; cells
(C, Y, min(G,M)); beta-bar exact by corners; SVD compression (the O(N) work was the R-factor's row count);
compression is Q-specific. 2.4: 3.09 M parameters (not 0.5 M); float64 required; selection on the rigorous
reduced bound; trimmed design at N = 1024; space-time variant. 2.5: DeepONet dropped. 3.1: N = 1024, not 512;
COS vs Carr-Madan at 2.6e-12 rather than 1e-12 on 2 of 10 rows. Item 2.6 not done.

## Regeneration commands

| Output | Command |
|---|---|
| EIM, `eim.json`, F4 | `python3 scripts/run_eim.py build select validate figure retol qtarget --nproc 20` |
| references (N = 512 / 1024) | `python3 scripts/gen_references.py --N 512` ; `python3 scripts/gen_references.py --N 1024 --extra bench120 ref30` |
| `E2_2_pod.json` (N = 512) | `OMP_NUM_THREADS=1 python3 scripts/run_pod.py basis validate` |
| `E2_3_reduced_cert.json` | `OMP_NUM_THREADS=1 python3 scripts/run_reduced_cert.py cells build check graded timing timing_v3` ; `python3 scripts/check_reduced_cert_longdouble.py` |
| `E2_3_cells.json` (N = 512) | `OMP_NUM_THREADS=1 python3 scripts/run_cells.py table partitions validity effect corners --build CYm:3x3x2` (and `--Q 29 37`) |
| `deeponet_old_vs_new.json` | `OMP_NUM_THREADS=1 python3 scripts/run_deeponet_old.py` |
| `E3_1_calibration.json`, `tab_budget_v9.tex`, `PRODUCTION_CONFIG.json` | `OMP_NUM_THREADS=1 python3 scripts/run_E3_1.py cos cm galerkin summary table --nproc 8` |
| `N1024.json` | `OMP_NUM_THREADS=1 python3 scripts/run_N1024.py pod eimq cells tensors checks table podext tensors_ext table_ext timing_ext --nproc 20` (timing steps single-threaded on an idle machine) |
| POD-NN sweep, `training_log.json`, `E2_4_podnn_N1024.json` | `CGMY_N=1024 python3 scripts/train_podnn.py prep` ; `... check` ; `... tbasis --n 800` / `--n 4000` ; `python3 scripts/sweep_N1024.py` (runs `scripts/eval_podnn.py`) |
| pilot summary | `results/pilot_N512.json` (archived logs; the pilot is not rerun) |
| F5a | `python3 scripts/fig_F5a.py` |
| zip | `python3 scripts/make_phase2_zip.py` |
