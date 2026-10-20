# Claude Code brief: Phase 2 (affine structure, reduced model, neural candidates) and Phase 3 experiments

You are working in the repository `cgmy_v9` (this directory). Read this file completely, then
`README.md`, `PHASE1_REPORT.md`, `step_by_step_workplan_v9.md` (project plan; the authoritative
"done-when" list) and the docstrings of `cgmy_cert/*.py`, before writing code.

## Ground rules

1. **Do not edit anything under `manuscript_v9/`** (not `main_v9.tex`, not `theory/`, not the PDF).
   The paper is edited elsewhere. Your outputs are code, tests, `results/*.json`, `figures_v9/*.png`,
   `tables_v9/*.tex` and a zip. If a figure caption or a table header should change, write the
   suggestion into `results/NOTES_FOR_PAPER.md`.
2. **Do not touch `v8_frozen/`** (read-only reference) and do not rerun anything with the
   principal-period matrices ("What not to do" in the plan). Everything is built on the exact
   assembly `cgmy_cert.symbols.free_matrices(q, N, K=100)`.
3. Keep the conventions frozen in `cgmy_cert/params.py` and the docstrings of `direct.py`
   (p = 3, Omega = (log 100 - 8, log 100 + 8), free indices k = 0..N-3, raw Toeplitz lifting,
   Omega-quadrature initial projection, tau_n = n T/N_T, pure CN, K = 100, L a multiple of N_h).
4. Every number that may end up in the paper goes into a JSON under `results/` with the query ids,
   N, N_T, eps, seeds, wall times, hardware (`platform`, CPU model, thread count) and git SHA.
   Never report a verdict at a single tolerance; always the sweep eps_N x {0.1, 0.3, 1, 3, 10}.
5. All paper timings come from Phase 3 (E2) on an idle machine with the protocol below; never
   quote numpy prototype timings. Pin one thread (`OMP_NUM_THREADS=1`, `taskset`), one warm-up
   query outside the population, 5 repetitions, shuffled method order, medians of per-query medians.
6. Tests first: every module gets a `tests/test_<module>.py` reproducing the plan's done-when
   criteria. `python3 -m pytest tests -q` must stay green (quick mode) and
   `CGMY_FULL_TESTS=1` must pass before you declare a phase done.
7. Commit after every completed item with a message `Phase 2.x: ...` / `Phase 3.x: ...`.
8. When something in the plan cannot be met, do not weaken the test silently: record the fact and
   the best achieved value in `results/DEVIATIONS.md`, keep the strict test marked `xfail` with the
   reason, and continue. A negative result is a result.
9. Environment: Python 3.10, numpy 1.26, scipy 1.15, torch 2.14 (CPU, 24 threads available for
   training only; set `torch.set_num_threads(1)` for all timed online work), h5py, mpmath 1.3.
   Install missing packages into the project venv, never system-wide.

## What already exists (Phase 1, all tests green)

| module | key API |
|---|---|
| `params.py` | `Query(C,G,M,Y,r,T,K,query_id)` with `.omega_S`, `.lambda0`, `.gammaY`; `Mesh(N)` (`h`, `n_free`, `n_raw`, `free_slice`); `BOX`; `box_vertices()`, `sobol_queries(n, seed)`; `load_queries_json` |
| `symbols.py` | `symbols(q, N, K, L, theta=None)` -> `Symbols(theta, a, s, m, d1, g1, phi, s_prime, eps_K)`; `fractional_alias(theta, q, h, K, with_derivative)` -> `(Phi_nu, Phi_nu')` (the EIM family); `free_matrices(q, N, K)` -> `FreeSystem(A: ToeplitzOp, S: ToeplitzOp, M_entries, K1_entries, D_entries, eps_K, sym, mesh)`; `ToeplitzOp.matvec/matmat/dense/entry(m)`; `banded_from_toeplitz`, `_dense_banded`; `legacy_principal_period_free` (tests only) |
| `spectral.py` | `spectral_data(q, N, K)` -> `SpectralData(s_min, alpha_lb, eps_K, r_mmin, beta, Theta, x_min, X_entries)`; `comparison_norm_wide(q, N, d)` (not needed: the Theta > 4 trigger refers to the **median** and is not hit at N = 512) |
| `riesz.py` | `circulant_preconditioner(sym_vals=fs.sym.s, n, kind="symbol")`; `pcg_bracket(S, rho, alpha_lb, precond, tol, maxit, budget)` -> `Bracket(l, u, v, iters, width)` |
| `direct.py` | `initial_free_vector(q, mesh)` (z_h^0), `free_forcing(fs, T, N_T)` -> `FreeForcing(F, tau, dt, bL, uM, uA)`, `cn_march(fs, z0, T, N_T, solver="lu"|"gmres", forcing)` -> `Trajectory(Z, iters)`, `reference_trajectory(q, N, N_T, solver)` -> `(fs, forcing, traj)`, `mass_norm(fs, v)`, `free_basis_eval(x, mesh)`, `spot_price_factor(fs, x)` (beta_I(x)) |
| `certificate.py` | `certify(fs, forcing, Z, z_h0, eps, mode="banded"|"exact"|"two-stage", spec, allowance=None)` -> `Certificate(status, beta0, etaX, etaX_aug, etaS_upper, etaS_lower, etaS_aug, sqrt_Theta, pcg_iters, time_*)`; `cn_residuals(fs, Z, forcing)`; `banded_dual_norms(entries, n, bw, R)` |
| `pod.py` | `pod_basis(snapshots, fs, r_max)` (mass-orthonormal), `pod_galerkin(fs, forcing, V, z0)` (full operator, no EIM yet) |
| `highprec.py` | mpmath exact symbol / Toeplitz entries (fixtures only) |
| `tests/fixtures/queries.json` | populations `audit459, bench120, ref30, validation30, scale30, price_diag8, spectral288` |
| `results/E0.json` | E0.1-E0.5 numbers; `results/coercivity_c1.json` |
| `v8_frozen/numerics/validation/cos_pricer.py` | the COS reference pricer (and Carr-Madan) used by v8; reuse it for 3.1 |
| `data/production_dataset.h5` | v8 training set (old discretisation; **do not train on it**, only for the DeepONet decision) |

The `allowance` argument of `certify` is the per-step term d_n of Theorem 5.2 / Remark 4.4:
`allowance[n] = eps_Q(mu) * ||cbar^{n+1/2}||_2 / sqrt(alpha)` with `alpha = spec.alpha_lb` for the
exact mode and `alpha = spec.x_min` for the banded mode; `certify` then uses
`(sqrt(u_n) + allowance[n])^2` per step; `allowance_X` is the banded-mode array (defaults to `allowance`).

## Phase 2 tasks (plan 2.1-2.6; weeks 4-7)

### 2.1 `cgmy_cert/eim.py` - EIM of the symbol family
* Family: `Phi_nu(theta)` on the L-grid of `symbols()` (L = N_h * max(4, ceil(256/N_h))), real and
  imaginary parts stacked, `nu = (G, M, Y)`. Use `symbols.fractional_alias(theta, q, h, K=100)`.
* Training set: tensor grid 16 x 16 x 16 in (log G, log M, Y). Greedy EIM with magic points
  (standard: pick the snapshot with the largest residual sup-norm, then the point where it is
  largest; interpolation matrix is lower-triangular with unit diagonal). Stop at Q when the test
  sup-error on 1000 Sobol nu (seed 2026) is <= 1e-9 relative to max|Phi|; expect Q ~ 25-30 at N = 512.
  Store Q as a function of the tolerance {1e-4, ..., 1e-10} for Figure F4 (EIM singular values and Q vs tolerance).
* Store (npz per N): magic points, the Q x Q matrix, the Q first columns `T(Phi_{nu_q})` for every
  production N (64, 128, 256, 512, 1024), and the training-grid definition.
* Online: `ell(nu)` (triangular solve at the magic points);
  `A_Q(mu) = -(r - omega_S) D_I + lambda_0 M_I - C Gamma(-Y) h^{-Y} sum_q ell_q T_q`
  as a `ToeplitzOp` (sum the first columns, then build one operator);
  `eps_Q(mu) = |C Gamma(-Y)| h^{-Y} ( max_grid |Phi_nu - I_Q Phi_nu| + (pi/L) max|e'| )` with the exact
  symbol and its derivative at the query (`fractional_alias(..., with_derivative=True)`), cost O(LK).
* Done when (tests): on 1000 Sobol test queries at N = 64 and 512,
  `||A_I(mu) - A_Q(mu)||_2 <= eps_Q(mu) <= 1e-8 ||A_I||_2`, and eps_Q is never exceeded by the true
  operator-norm error (dense 2-norm at N = 64; at N = 512 use the symbol sup-norm of the difference
  on a 4x finer grid as the "true" error, plus the dense norm on 50 queries).

### 2.2 `cgmy_cert/pod.py` - POD-Galerkin with the affine operator
* Snapshots: *new* exact-assembly references (`reference_trajectory(..., solver="gmres")`) for 800
  Sobol queries (seed 0, with T) and a 4000-trajectory set (seed 0 extended) for the data-size study,
  at the production N (fix N = 512, N_T = 240 provisionally; 3.1 confirms). Store trajectories as
  float64 npz (`data/references_N{N}_NT{NT}_seed0_{n}.npz`) with sha256 in the manifest.
* Ranks r in {8, 16, 24, 32, 40, 48, 56, 62}. Reduced operator `V^T A_Q V = sum_q ell_q (V^T T_q V)`
  (Q + 2 precomputed r x r blocks), reduced forcing via `V^T M_I e_bdry`, `V^T T_q e_bdry` for the
  two left boundary directions (see `direct.free_forcing`: `uM`, `uA` are exactly those couplings;
  for the affine version you need `uA` per EIM term), `alpha^0 = V^T M_I z_h^0`, reconstruct `V alpha^n`.
* Certificate: full-order `eta^X` with the allowance `eps_Q ||V alpha_bar||_2 / sqrt(x_min)`
  (Theorem 5.2) and the reduced online `eta^{X-bar}` (2.3).
* Done when: at r = 62 the reduced solution equals the direct solution to 1e-10 (with A_I, not A_Q;
  with A_Q to eps_Q level); at r = 48 the full-order certificate reproduces the direct-acceptance
  pattern on the 30 validation queries at eps = 1e-3 (report it; it is a sanity check, not a result).

### 2.3 Reduced online certificate (N-independent)
* Cells: 2 x 2 in (log r, log C); per cell `X-bar = r-bar M_I + beta-bar K1_I`, `beta-bar` = 0.99 x
  min of `spectral_data(q, N).beta` over a 5 x 5 grid of the cell (r, C at corners/midpoints; G, M, Y
  at their 3 x 3 x 3 extreme/mid values - take the min over all).  `r-bar` = the cell's r_min.
* Offline per cell: `G_{qq'} = R_q^T X-bar^{-1} R_{q'}` for `R_q in {M_I V, T_q V (q=1..Q), D_I V,
  f-components}` (banded solves, `scipy.linalg.solve_banded`), stored as a dense tensor.
* Online: `eta^{X-bar}` from the quadratic form with `C = sum_n w^n (w^n)^T`; rigorous conversion
  `(min_theta s_mu / x-bar)^{-1}` at the query (one graded-grid evaluation like `spectral_data`).
* Done when: `eta^{X-bar}` online equals the full-order `rho^T X-bar^{-1} rho` to 1e-10 (relative);
  online wall time at r = 48, Q = 25 is < 5 ms compiled (numba or torch.compile is fine) / < 50 ms
  numpy, independent of N (test N = 256, 1024, 4096).

### 2.4 `cgmy_cert/podnn.py` - the POD-NN candidate (torch)
* Input (mu, T) standardised (6 scalars); output r x N_T POD coefficients, r = 48, with alpha^0
  fixed to the projection of z_h^0 (so beta_0 is the projection error only).
* MLP 6 -> 256 -> 256 -> 256 -> r N_T, SiLU, ~0.5 M parameters. Loss = relative least squares in the
  reduced coordinates + lambda_R * eta^{X-bar} (the offline tensors make it differentiable and
  N-independent). lambda_R in {0, 1e-2, 1e-1, 1}.
* Training: 800 and 4000 trajectories; Adam 1e-3 cosine to 1e-5, 2000 epochs, batch 64,
  validation-based checkpoint selection on validation30, 3 seeds {0, 1, 2}. Log wall time, hardware,
  hyper-parameters, per-epoch losses to `results/training_log.json`. The training script
  `scripts/train_podnn.py` is part of the archive; checkpoints under `checkpoints/`.
* Done when: median certified bound on validation30 at N = 512 is below the cent tolerance eps_N
  (from 3.1) for at least one lambda_R; if not, report the best achieved and move on.

### 2.5 DeepONet (decision, max 3 days)
* The v8 checkpoint `v8_frozen/models/deeponet_final.pt` and the training code at
  `v8_frozen/models/training_code_at_checkpoint_commit_13bf0d5/` exist. First, certify the *old*
  checkpoint's prediction as a candidate of the *new* system at N = 64 on bench120 (it was trained
  on principal-period data, so expect large residuals) - one JSON, `results/deeponet_old_vs_new.json`.
* Retrain on the new N = 64 references only if the training script runs unchanged by the end of
  week 4; otherwise drop it and record the decision in `results/DEVIATIONS.md`.

### 2.6 Cost model numbers
Produce `results/cost_model.json`: measured flop-proxy counts (number of FFTs of size 2 N_h, banded
solves, r x r operations) for direct solve, full-order certificate, reduced solve and reduced
certificate at N in {64, ..., 4096}, so that the table in Section 4.4 of the paper can be filled.

## Phase 3 tasks (plan 3.1-3.5; weeks 8-10) - run after Phase 2 is green

### 3.1 Resolution and tolerance calibration (first; everything depends on it)
ref30 + the two corner queries of v8 (`v8_frozen/manuscript_v8/tables/tab_corner.tex` gives them):
discrete price vs COS at N in {128, 256, 512, 1024} with 60 and 240 CN steps, five spots
0.8K..1.2K. Production N = smallest with max |V_h - V_COS| <= 1 cent (expect 512, 240 steps).
`eps_N = tau_$ / max_s beta_I(x_s)` with tau_$ = 1 cent (and 0.1 cent at N = 1024); sweep
eps in eps_N x {0.1, 0.3, 1, 3, 10}. Output `results/E3_1_calibration.json` + `tables_v9/tab_budget_v9.tex`.
**Write the chosen (N, N_T, eps_N) into `results/PRODUCTION_CONFIG.json`; no later experiment changes it.**

### 3.2 E1 - reduced model and candidates on validation30
POD-Galerkin r = 8..62 and POD-NN (3 seeds, best lambda_R) at production N: beta_0, eta^X, eta^S,
true error vs dense direct (LU at N <= 1024), effectivity, acceptance at each eps of the sweep,
online time split (solve / certificate) in full-order and reduced modes.
Figures: F5a acceptance vs eps per candidate; F5b effectivity histograms; F5c online time vs r
with the direct-solve line. `results/E1.json`.

### 3.3 E2 - matched benchmark at production resolution
bench120; methods: direct (circulant-GMRES, exact assembly; with and without certificate),
POD-Galerkin r in {32, 48}, POD-NN, (DeepONet if kept); cold and reusable setup; timing protocol
of rule 5 on an idle machine. Report at eps_N x {0.3, 1, 3}: acceptance counts, correction counts,
geometric-mean time ratio to the *uncertified* direct solve with query-bootstrap 95% CI, and the
certificate fraction of the solve time. Refinement: N = 64..4096 for direct and full-order
certificate (certificate/solve ratio vs N) and the reduced online time (flat line).
Figures F6 (benchmark ratios + certificate fraction vs N), table T6. `results/E2.json`.

### 3.4 E3 - out-of-distribution detection
200 queries outside the box: C in [1, 2], Y in [0.9, 1.1) and (1.8, 1.95], G in [0.5, 1), M in [9, 15]
(Sobol seed 7 in the union of these shells), plus the 64 vertices of the extended box. Candidates:
POD-Galerkin r = 48, POD-NN. For each: true error, certified bound, verdict at eps_N; naive proxy =
seed-ensemble disagreement of POD-NN. Figure F7 (log-log scatter true error vs bound with the
eps_N lines); table: false accepts (must be 0), false rejects, proxy precision/recall. `results/E3.json`.
Note for G < 1: the symbol decay scale Gh is smaller; `symbols()` keeps L = 4 N_h, check that the
aliasing error (|t_m| ~ e^{-Gh m}) stays below 1e-10 at the production N, else raise L for those queries.

### 3.5 E4 - multi-query demonstration
10^4 Sobol queries (seed 4) at production N: wall-clock of (i) direct, (ii) certified POD-NN with
correction fallback (full-order CN warm start from the candidate, GMRES, recertify),
(iii) certified POD-Galerkin. Fraction corrected, total time, amortisation C_0/Q + C_1 with the
offline cost (data generation + POD + training). Figure F8 (cumulative time vs number of queries,
offline cost as intercept). `results/E4.json`.

### 3.6 E5 (optional, only if everything above is green by week 10)
American put: backward-Euler obstacle solver (primal-dual active set) on the same spline space;
POD-NN on 800 American trajectories; certificate from Theorem 6.1 of the note with the banded norm.

## Deliverable

`zip -r phase2_3_results_<date>.zip cgmy_cert tests scripts results figures_v9 tables_v9
checkpoints/*.json results/DEVIATIONS.md results/NOTES_FOR_PAPER.md` (exclude trajectories and
checkpoints larger than 50 MB; list their sha256 in `results/MANIFEST.json`). The zip is uploaded
to the paper-editing session; make every JSON self-describing (units, populations, seeds).

## Reporting back
At the end, write `results/PHASE2_3_REPORT.md`: one table of plan items vs status, the numbers that
answer the three decision points (week 4: DeepONet kept?; week 8: production N; week 10: does
POD-NN pass at eps_N?), every deviation, and the exact commands that regenerate each figure.
