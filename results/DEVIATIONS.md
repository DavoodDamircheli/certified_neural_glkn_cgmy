# Deviations from the plan / brief (Phase 2-3)

Each entry: what the plan says, what was done, the best achieved value, and where it is recorded.

## 2.1 EIM rank selection: Q = max(Q_tol, Q_op)

* **Plan:** stop at Q when the test sup-error on the 1000 Sobol nu (seed 2026) is <= 1e-9 relative
  to max|Phi_nu|. Done-when: `||A_I - A_Q||_2 <= eps_Q(mu) <= 1e-8 ||A_I||_2` on the 1000 test queries.
* **Finding:** at N = 64 the 1e-9 stopping rule gives Q_tol = 22, and with Q = 22 the second
  inequality fails on 2 of the 1000 test queries (max eps_Q/||A_I||_2 = 1.43e-8). The plan's two criteria
  are measured differently: sup-error relative to max|Phi_nu| on the grid, versus the operator-norm
  bound (grid term plus derivative margin) relative to ||A_I||_2, which also contains the drift and
  lambda_0 M_I parts.
* **Done:** the operational rank is Q = max(Q_tol, Q_op), where Q_op is the smallest Q whose max over
  the 1000 test queries of eps_Q/||A_I||_2 is <= 1e-8 (`scripts/run_eim.py select`). Result:

  | N | 64 | 128 | 256 | 512 | 1024 |
  |---|---|---|---|---|---|
  | Q_tol (1e-9 rule) | 22 | 26 | 27 | 29 | 29 |
  | Q_op (1e-8 operator bound) | 23 | 24 | 25 | 26 | 26 |
  | **Q used** | **23** | 26 | 27 | 29 | 29 |

  Only N = 64 changes (by one term). The strict test is unchanged; it passes with Q = 23
  (max eps_Q/||A_I||_2 = 1.65e-9). Numbers: `results/eim.json` -> `select`.

## 2.1 Stored basis: T(xi_q), not T(Phi_{nu_q})

* **Brief:** store "the Q first columns T(Phi_{nu_q})".
* **Done:** the stored first columns are those of the EIM basis functions xi_q (normalised greedy
  residuals). They span the same space as the Phi_{nu_q}, and they are what makes the interpolation
  matrix unit lower triangular, so ell(nu) is a triangular solve. The snapshot-basis matrix is
  ill-conditioned (the singular values of the training set fall to 1e-12 by index 40). The selected
  nu_q and the recursion coefficients are stored too, so T(Phi_{nu_q}) can be rebuilt if wanted.

## 2.1 Symmetrised family

* The K-truncated alias sum is not exactly conjugate-symmetric near theta = -pi, because the alias set
  {theta + 2 pi k, |k| <= K} is not symmetric there; the defect is O(eps_K). `ToeplitzOp` keeps the real
  part of the Toeplitz entries, so the assembled A_I is T of the *symmetrised* samples
  (Phi(theta) + conj Phi(-theta))/2 (tested in `tests/test_eim.py`). The EIM is built on, and eps_Q
  measured against, these symmetrised samples. The unpaired alias at theta = -pi for |k| <= K contributes
  an imaginary part of order (2K)^(Y-6) ~ 1e-10, which lies inside eps_K. A side effect is that every basis function has real
  Toeplitz entries exactly; without symmetrisation, greedy roundoff gave relative imaginary parts of
  9e-8.

## 2.1 EIM files are built for N <= 1024 only

* N = 2048 and 4096 are not production resolutions. They are needed only for the N-independence timing
  test of 2.3. `load_eim(N)` builds them on demand (`python3 scripts/run_eim.py build select --N 2048 4096`).

## 2.2 Allowance uses the raw coefficient vector, not the free vector

* **Follow-up brief:** allowance = eps_Q ||cbar_n||_2 / sqrt(alpha) with cbar_n = V (alpha^{n+1}+alpha^n)/2,
  i.e. the free vector.
* **Done:** the forcing of the A_Q system is also built with A_Q, through the boundary couplings u_q.
  The residual with A_I therefore differs from the one with A_Q by E^T (Abar_I - Abar_Q) cbar_raw, where
  cbar_raw also contains the p-1 prescribed left values (b_L^{n+1}+b_L^n)/2. With the free vector alone
  the forcing-difference term is not covered. `pod.allowance_vectors` uses ||cbar_raw||_2 (still bounded
  by eps_Q, since eps_Q bounds the symbol on the grid and L >= 2 n_raw). Measured cost on validation30:
  ||cbar_raw|| / ||cbar_free|| <= 1.0052. Both norms are reported in `results/E2_2_pod.json`.

## 2.2 r = 48 sanity check: the certificate does *not* reproduce the direct acceptance pattern at eps = 1e-3

* All 30 validation candidates at r = 48 have true error <= 1e-3 (median 1.2e-4, max 3.4e-4).
* Certified accepts at eps = 1e-3: 8/30 in exact mode (eta^S) and 0/30 in banded mode (eta^X).
  Median effectivity is 10.9 (eta^S) and 18 (eta^X). There are no false accepts at any rank or
  tolerance of the sweep.
* At r = 56 all 30 are accepted at eps = 1e-3. At r = 48, 28/30 (exact) and 23/30 (banded) are accepted
  at eps = 3e-3. As the brief says, this is a sanity check, not a result; E1 repeats it at eps_N.

## 2.2 Order of work

* Section 1 of the follow-up (restore `v8_frozen/` from the tag `v8-frozen`) could not be done first.
  This clone has neither the tag nor commit 2dd5f4f, no other local repository has them, and fetching
  from origin needs GitHub credentials. 2.2-2.4 do not touch `v8_frozen/`, so they went ahead; the restore
  and the 4 regression tests follow as soon as the tag is fetched.

## 2.3 R-factor form of the reduced certificate instead of the Gram tensor

* **Plan:** offline G_{qq'} = R_q^T X-bar^{-1} R_q', online quadratic form with C = sum_n w^n (w^n)^T.
* **Finding:** for good candidates the residual is ~1e-8 of its individual terms (kappa ~ 5e7 for
  POD-Galerkin at r = 48), and the Gram form loses eps * kappa^2 = O(1) relative accuracy. Measured on 3
  queries: per-step errors up to 100% and eta^2 errors of 0.5%. The Gram form can therefore
  *under*-estimate the bound.
* **Done:** offline banded Cholesky X-bar = L L^T, W = L^{-1} F, economy QR W = Q_W R, storing R
  (m x (Q+2)(r+1), m = min(n, (Q+2)(r+1))). Online: u_n = ||R_M x_n + (sum_k theta_k R_k) y_n||^2.
  The error is eps * kappa, and a rigorous dot-product rounding term is added to sqrt(u_n) (it is at most
  1.4e-3 of sqrt(u_n)). Online cost 0.55 / 1.95 / 2.99 ms (numpy, one thread) at N = 256 / 1024 / 4096: it
  grows with m until m saturates at (Q+2)(r+1) = 1519, so it is bounded independently of N.

## 2.3 Equality to 1e-10: met for O(1) residuals; floor ~1.4e-10 for POD-Galerkin candidates

* Online vs full-order (rho^T X-bar^{-1} rho)^{1/2} on validation30[:20]: perturbed candidates <= 1.2e-15;
  POD-Galerkin candidates <= 1.44e-10 (per step <= 2.3e-7). Against an extended-precision (long double)
  reference for the worst query, the online value is off by 1.1e-10 and the double-precision full-order
  value by 3.3e-11. This is the eps * kappa floor. The strict 1e-10 test for POD-Galerkin candidates is
  kept as `xfail`; a 3e-10 test passes. The bound itself is safe because of the rounding term.

## 2.3 Online time: the quadratic form is N-independent, the per-query symbol work is not (eps_Q)

* Single thread, r = 48, Q = 29 (median over 10 queries; `results/E2_3_reduced_cert.json` -> timing):

  | N | form | reduced solve | ell(nu) | eps_Q | conversion c_mu |
  |---|---|---|---|---|---|
  | 256 | 0.55 ms | 0.98 ms | 1.09 ms | 29 ms | 58 ms |
  | 1024 | 1.95 ms | 0.98 ms | 1.10 ms | 123 ms | 103 ms |
  | 4096 | 2.99 ms | 0.98 ms | 1.14 ms | 530 ms | 199 ms |

* The plan's < 5 ms (compiled) / < 50 ms (numpy) holds for the certificate's quadratic form and for the
  reduced solve. It does not hold for the two candidate-independent per-query quantities. eps_Q needs the
  exact symbol on the L-grid (O(N K); it is also what makes the compression argument of Remark 2.9 apply).
  The conversion factor c_mu = min (s_mu - eps)/x-bar needs one graded-grid symbol evaluation. Both are
  computed once per query, whatever the candidate. At N = 512 they total about 0.2 s, which is the cost of
  one direct GMRES solve (0.19 s). This changes the cost story and must be in the paper's cost model (2.6).
* c_mu uses an adaptive alias count K' in {10, 20, 40, 100}: the smallest K' whose tail bound eps_{K'} is
  <= 2% of min x-bar. It is rigorous for the exact symbol, loses <= 1.1% of c_mu, and is 3-8x cheaper than
  K = 100. A fixed K' = 10 is not robust (c_mu < 0 at N = 4096).
* No compiled (numba/torch.compile) timing: the form already meets 5 ms in numpy.

## 2.3 Cells: beta-bar does not depend on r

* beta_mu = 0.99 min (s - r m)/g1 does not depend on r, because the lambda_0 m part of s contains r m.
  The r0 and r1 cells therefore have identical beta-bar (2.07 for C <= 0.224, 9.15 for C >= 0.224), and
  splitting in r only changes r-bar. Within a cell beta ranges from 2.1 to 25.6 (median), driven by
  (G, M, Y). With this X-bar the reduced certificate has median effectivity 51 on POD-Galerkin r = 48 (the
  full-order eta^X with the query's own beta has 18). Cells in (C, Y) or (C, M) would tighten it.
  The plan fixes the (r, C) cells, so this is reported, not changed.

## 2.4 POD-NN: parameter count, precision, selection metric

* **Parameter count.** The specified MLP 6 -> 256 -> 256 -> 256 -> r N_T (r = 48, N_T = 240) has 3.09 M
  parameters, not "~0.5 M": the 11520-wide output layer alone has 2.96 M. The architecture is kept as
  specified.
* **float64.** float32 is unusable for a certified candidate. With |alpha| ~ 1e2 the quantisation noise
  (~1e-5) is amplified by 1/dt and by the operator in the residual to ~1e-3 in eta, the target tolerance.
* **Checkpoint selection** every 10 epochs on the median *rigorous* reduced bound eta_aug on validation30
  (beta_0, allowance, 1/c_mu and the rounding term included); the relative-LS validation loss is logged too.
* **Reference points for the done-when (reduced certificate, validation30, r = 48):** the exact projection
  of the references has median eta_aug 5.4e-3 (0/30 <= 1e-3), and POD-Galerkin 6.4e-3 (0/30). A POD-NN
  can pass eps = 1e-3 with this certificate only if the residual term drives it below the projection's
  residual.
* **beta_0 fix (2.3 code).** beta_0 was computed as sqrt(||z0||^2 - 2 a.Vt M z0 + |a|^2), which cancels
  (||z0||_M = 254, beta_0 = 2.2e-5). It is now computed by Pythagoras from the full-order projection error.
  The redcert files were rebuilt and the E2_3 check rerun (MANIFEST updated).
* **Compute.** n = 4000 with lambda_R > 0 takes 17.5 s/epoch (4 threads, 6 runs in parallel), i.e. ~10 h per
  2000-epoch run; the full sweep (2 x 4 x 3 = 24 runs) takes about a day.

## 2.5 DeepONet: not kept (week-4 decision)

* The old checkpoint, certified as a candidate of the new system (N = 64, N_T = 60, bench120, two-stage with
  A_I): 120/120 rejected at every eps in 1e-3 x {0.1, 0.3, 1, 3, 10}; median true error 0.78 (relative
  3.1e-3, max 3.3); median eta^S 3.4 (effectivity 4.2); no false accepts (`results/deeponet_old_vs_new.json`).
  production_dataset.h5 was not needed: the network inputs come from the v8 code, and c^0 equals the v9
  vector to 8.5e-14.
* Retraining is dropped. The v8 training script cannot run unchanged against the new references: its
  validation set is regenerated with the principal-period assembly (forbidden by rule 2), and its loader
  reads v8's HDF5 schema, not the new npz files.


# Follow-up 2 (review of 2.2-2.3): changes 2a-2d

The earlier entries stand as accepted by the review: raw-vector allowance, R-factor form with rounding
term and the 1e-10 xfail (extended-precision evidence now in `results/E2_3_reduced_cert.json` ->
check.extended_precision_evidence, script `scripts/check_reduced_cert_longdouble.py`), the beta_0 fix, and the
r = 48 sanity check. The 2.3 timing table above is superseded by 2b.

## 2a. The full-order certificate uses A_I (no allowance)

`results/E2_2_pod.json` now stores, for every rank and query, `variants.with_AI` (residual with A_I and f_I,
no allowance: the full-order certificate proper) and `variants.with_AQ_allowance` (the 2.2 numbers).
Validation30, median eta^S: identical up to r = 32; r = 40: 9.850e-03 vs 9.891e-03;
r = 48: 1.369e-03 vs 1.384e-03;
r = 56: 1.994e-04 vs 2.331e-04;
r = 62: 6.625e-05 vs 9.875e-05. Acceptance counts at the eps sweep
do not change on this population. From now on A_Q appears only in the reduced solve and in the
N-independent online certificate.

## 2b. N-independent per-query symbol work

* **eps_Q on an N-independent node set** (`EIM.eps_Q_graded`, `eim.GradedNodes`). The nodes are symmetric:
  a uniform 96-point grid plus 32 geometric points on the scale h * min_box min(G, M) = h near 0, plus the
  one-sided limits at +-pi, 161 nodes in all. The EIM basis is precomputed on them, along with
  theta_k, |N^(theta_k)|^2 and its derivative (query-independent). Online, (delta +- i theta_k)^Y is
  evaluated in real arithmetic with all K = 100 aliases. Bound: sup_theta |e| <= max over cells of
  (max endpoint |e| + half-width x max endpoint |e'|) (spectral._grid_extreme's rule), and separately the
  L-grid's convention value at theta = -pi.
  Alias truncation is *not* used for eps_Q: the tail at any useful K' is far larger than e itself
  (~1e-9 relative), so it cannot be rigorous.
* Rigour check, 200 queries per N: graded eps_Q / (scale * max_L-grid |e|) >= 1.050 at every N in 64..4096
  (the right-hand side bounds ||A_I - A_Q||_2 by the compression argument). Loss of tightness against the
  old L-grid eps_Q (median / max): N=64: 1.02 / 1.31; N=256: 1.20 / 1.54; N=512: 1.55 / 1.78; N=1024: 1.52 / 1.90; N=4096: 1.75 / 2.08.
* **c_mu is optional** (`ReducedCertificate.certify(..., tighten=False)` by default). The cells are built so that
  x-bar <= x_mu <= s_mu; on 400 Sobol queries c_mu >= 1.0057 for every partition
  (`results/E2_3_cells.json` -> validity). With tighten=True, c_mu uses the adaptive alias truncation
  K' in {10, 20, 40, 100}, rigorous via the Lemma 1.2 tail bound eps_K'(all |k| > K'):
  s_mu >= s_(mu,K') - eps_K'. It costs 0.2-0.4 s, so it is not in the timed path.
* **Timing** (one thread, r = 48, Q = 29, CYm:3x3x2 cells, `timing_v2`; measured while the 2.4 sweep
  occupied ~24 threads):

  | N | cert. form | ell(nu) | eps_Q (graded) | total online cert. | c_mu (optional) |
  |---|---|---|---|---|---|
  | 256 | 0.63 ms | 1.17 ms | 1.62 ms | 3.42 ms | 209 ms |
  | 1024 | 5.42 ms | 1.19 ms | 1.62 ms | 8.22 ms | 410 ms |
  | 4096 | 9.07 ms | 1.24 ms | 1.60 ms | 11.91 ms | 399 ms |

  ell and eps_Q are flat in N. The form costs O(m r N_T) with m = min(n_free, (Q+2)(r+1)), so it grows with N
  until m saturates at 1519 (N >= ~1520). It is bounded independently of N but not flat below that. On a
  less loaded machine this afternoon the same form took 0.55 / 1.95 / 2.99 ms (table above); the slowdown
  here is contention. Reducing m via a smaller Gram matrix would reintroduce the cancellation problem.
  The < 5 ms target is met for N <= 1024 under contention for ell + eps_Q, and for the full total only
  at N = 256. Phase 3 (E2) repeats this on an idle machine.

## 2c. Cells in (log C, Y, log min(G, M))

* beta~(G, M, Y) = beta_mu / C was tabulated on a 9 x 10 x 13 grid (N = 512). It is monotone on the table:
  nonincreasing in G and in M, nondecreasing in Y. The cell minimum is therefore attained at a corner,
  and the cell boundaries are table nodes. This monotonicity is observed, not proven. beta~ ranges from 41
  to 2417 and is driven by Y; the min(G, M) split changes beta-bar by only ~3%.
* beta-bar(cell) = 0.99 C_lo min_cell beta~, r-bar = r_min of the box (0.01) for all cells.
* Validation30, POD-Galerkin r = 48, residual parts (A_I), eta^{X-bar} / eta^X (own X_I) and / eta^S,
  median (max):
  * rC:2x2: 3.52 (10.06) / 5.72 (13.48)
  * CYm:2x2x2: 2.61 (5.10) / 4.29 (7.05)
  * CYm:3x3x2: 1.82 (3.17) / 3.24 (5.10)
  CYm:3x3x2 meets the target (median within ~2x of the own norm). Storage: 8.4 MB per cell at N = 512 with
  Q_max = 42 (151 MB for 18 cells; ~37 MB per cell at N >= 2048). Files:
  `data/redcert_N512_r48_n{800,4000}_CYm_3x3x2.npz` (sha256 in MANIFEST).
  A finer Y split would probably pay off more than the min(G, M) split.
* The running 2.4 sweep's loss uses the old rC:2x2 cells (its loss is still a valid majorant). Its
  evaluation (section 4) uses the new cells. This is recorded in `training_log.json` at collection.

## 2d. EIM tolerance matched to the target

* `EIM.Q_for_target(q, eps)`: the smallest stored Q with eps_Q(mu) ||cbar||_2 sqrt(T/alpha) <= eps/10, with
  defaults ||cbar|| = 1.1 ||c_raw^0||_2 and alpha = r min m (a rigorous lower bound of alpha_lb). This is a
  selection heuristic only: the certificate always uses the actual allowance.
* No rebuild was needed. The greedy had already stored every term down to a training error of 1e-13
  (q_max = 42 at N = 512). The 1e-11 test tolerance needs Q = 37 at N = 512 and 1024 (28 / 32 / 36 at
  N = 64 / 128 / 256). F4 now extends to 1e-11.
* Reduced tensors are built for Q_max = 42: the leading block columns of R are the R-factor of the leading
  columns, so one tensor serves every Q <= 42 (tested).
* Validation30 at N = 512: target-matched Q median / max = eps=0.0001: 32.0 / 38; eps=0.0003: 31.0 / 34; eps=0.001: 30.5 / 32; eps=0.003: 28.0 / 31; eps=0.01: 26.0 / 31.


# Follow-up 3

## 2.5 DeepONet retrain dropped: the two reasons (verbatim, as accepted)

1. "train_final_models.py -> generate_production_dataset.build_val_set regenerates the validation set with
   the v8 trusted solver generate_one_sample(mode='toeplitz'), i.e. the principal-period assembly; running it
   unchanged violates CLAUDE_PHASE2.md rule 2"
2. "load_all_train_samples reads v8's HDF5 schema (data.dataset.read_trajectory, raw 66-dof coefficients incl.
   boundary rows); the new references are npz free-coefficient files, so the script cannot read them unchanged"

## Item 1. The remaining N-dependence of the online certificate, and its removal

* **Profile** (cProfile, one thread, N = 4096, Q = 29, r = 48; `results/E2_3_reduced_cert.json` -> timing_v3).
  Before: per_step 14.8 ms, tensordot 6.3 ms and the eps_Q kernel 6.1 ms per call.
  The first two scale with m = min(n_free, (Q+2)(r+1)), the row count of the stable R-factor (254 / 510 / 1022 /
  1519 at N = 256 / 512 / 1024 / 4096). No length-N_h array is touched online: alpha^0, beta_0 and the boundary
  couplings V^T T_q e_bd, V^T M_I e_bd were already offline constants (the boundary couplings are the (r+1)-th
  column of every block of R).
* **Why not the plan's O(Q^2 r^2) Gram form:** it is numerically unusable for good candidates (2.3 entry
  above, eps * kappa^2 ~ O(1)).
* **Fix: rigorous SVD compression of R (offline, per cell, per operational Q).** R = U S W^T; keep the
  k rows S_k W_k^T = U_k^T R with sigma_(k+1) <= 1e-12 sigma_1; online ||R w|| <= ||U_k^T R w|| + sigma_(k+1) ||w||
  with ||w|| <= ||x_n|| + ||theta|| ||y_n||; the tail term is added to the rounding term. It is still an
  orthogonal transform, hence equally stable. k ~ 350-420 at every N >= 512 (Q = 29), so the online cost is
  O(min(n_free, k)).
* **After:** per_step 2.1 ms, tensordot 1.8 ms, eps_Q kernel 4.8 ms. Timed call (ell + eps_Q on the 161 fixed nodes
  + certificate), one thread, measured while the 2.4 sweep occupied ~24 threads (not idle):

  | N | rows before | rows after | before | after | max bound inflation |
  |---|---|---|---|---|---|
  | 256 | 254 | 254-254 | 10.4 ms | 10.4 ms | 0.0e+00 |
  | 512 | 510 | 349-404 | 13.1 ms | 11.6 ms | 3.6e-04 |
  | 1024 | 1022 | 366-419 | 23.1 ms | 12.2 ms | 4.1e-04 |
  | 4096 | 1519 | 404-416 | 32.4 ms | 11.5 ms | 7.6e-04 |

  The call is flat in N to within noise. The absolute values are inflated ~3-4x by the sweep (the same
  eps_Q kernel took 1.6 ms on a less loaded machine). The < 5 ms idle target must be confirmed in Phase 3
  (E2) on an idle machine.
* Compression is Q-specific: a tensor that serves every Q <= Q_max cannot be compressed (with Q_max = 42 at
  N = 512, k = m = 510). Stored: the general uncompressed Q_max = 42 tensor (evaluation at any Q), plus compressed
  `..._Q29c.npz` (k = 345-398, 4.6 MB/cell) and `..._Q37c.npz` (k = 481-506, 7.4 MB/cell). The tensor for the
  eps_N-matched Q is built once 3.1 fixes eps_N.

## Item 2. beta-bar exact in (C, G, M), grid only in Y

* beta_mu = C beta~(G, M, Y), and beta~ is nonincreasing in G and in M (Lemma 2.6; Remark 4.3 of the paper).
  On a CYm cell the infimum over (C, G, M) is therefore attained at C = C_min and at the largest admissible
  (G, M) with min(G, M) in the cell: (m_hi, M_max) and (G_max, m_hi) for the lower min(G, M) cell, and
  (G_max, M_max) for the upper one (`reduced_cert.cell_gm_corners`). Only the minimum over Y is a grid minimum
  (33 points, endpoints included), with the 0.99 factor (`reduced_cert.beta_bar_rigorous`).
* Check: in all 18 cells the old table-based minimum equals the corner value (old / corner = 1.000 to 5 digits;
  never below 0.99 x corner). beta-bar is unchanged. The minimum sits at the cell's lowest Y and at (sqrt 5, 9)
  or (5, 9). Tensors were rebuilt with the corner rule; the file names are unchanged.

## Item 3 (Phase 3.1): PRODUCTION N = 1024, NOT 512  <-- decision needed before E1

* Max |V_h(T,S) - V_COS| over ref30 + the two v8 corners, spots 0.8K..1.2K (`results/E3_1_calibration.json`,
  `tables_v9/tab_budget_v9.tex`), in cents (N_T = 240): N = 128: 78; 256: 20; **512: 3.2** (ref30 0.11,
  Corner A 2.4, Corner B 3.2); **1024: 0.24** (ref30 0.16, A 0.14, B 0.24). At N = 512 ref30 alone is within
  1 cent; the corner queries (C = 0.05, Y = 1.1, T = 0.1) are not.
* **Production: N = 1024, N_T = 240, eps_N = 0.01 / max_s beta_I = 0.01 / 8.441 = 1.1846e-3** (sweep
  eps_N x {0.1, 0.3, 1, 3, 10}), written to `results/PRODUCTION_CONFIG.json`. N_T = 60 does not suffice at
  N = 1024: ref30 error 4.6 cents, larger than at N = 512. Pure CN damps the kink's high frequencies poorly
  when lambda dt is large.
* The 0.1-cent tolerance at N = 1024 gives eps = 1.18e-4. This is a certificate tolerance only: the
  discretisation error at N = 1024 is 0.24 cents.
* **Consequence (not started, as instructed):** the stored references (`data/references_N512_NT240_*`), the
  POD bases, the reduced tensors and the POD-NN sweep are all at N = 512. E1 at the production resolution
  needs them regenerated at N = 1024 (references ~4800 trajectories, POD, EIM/redcert at N = 1024, and a POD-NN
  retrain). This awaits your decision.
* COS (8192 modes, L = 30) vs Carr-Madan on 10 rows: 8 rows <= 6e-13, one 1.5e-12, and Corner B 2.6e-12 after
  extending Carr-Madan's cutoff (at the default max_v = 400 it was cutoff-limited, diff 3.4e-8). The 1e-12
  target is not met strictly on 2 of the 10 rows: the max is 2.6e-12 absolute, 5.6e-11 relative.


# Follow-up 4: production resolution N = 1024

* The "bound increase of at most 8e-4" quoted for the SVD compression is **relative**:
  eta_aug(compressed) / eta_aug(uncompressed) - 1 (`E2_3_reduced_cert.json` -> timing_v3.bound_inflation_definition).
  At N = 1024 the compressed bound over the full-order value is <= 1.0005 on 20 validation queries.

## Item 1. N = 512 POD-NN sweep stopped and archived as the pilot

`results/pilot_N512.json`, logs in `results/pilot_N512/logs/`, checkpoints in `checkpoints/pilot_N512/` (143 MB, not in
git). Six of the 24 planned runs (n = 4000, lambda_R in (1, 0.1), seeds 0-2) had started and reached epochs
1200-1240 of 2000. Best validation median reduced bound (old rC cells): 5.8-7.7, against 5.4e-3 for the
projection of the references. Not evaluated further.

## Item 2. Everything regenerated at N = 1024, N_T = 240 (`results/N1024.json`, `scripts/run_N1024.py`)

* References: 4000 (+ its first 800), validation30, bench120, ref30; GMRES <= 11 iterations per step; sha256 in
  MANIFEST. POD from 800 and 4000 (sigma_48/sigma_1 = 2.2e-7 for both).
* EIM at N = 1024: no rebuild was needed (44 stored terms, Q = 37 at the 1e-11 test tolerance). eps-matched
  operational Q on validation30 (max over queries): **Q = 36 at eps_N** (median 33) and **Q = 37 at eps_N/10**
  (median 35).
* Cells (3 x 3 x 2, corner rule, 33-point Y grid) recomputed at N = 1024. On 400 Sobol queries
  c_mu >= 1.00016: X-bar <= S holds, but with only 0.016% margin at the worst query.
  Cell vs own-norm effectivity (POD-Galerkin r = 48): median 1.87, max 3.38.
* Compressed tensors (svd_tol 1e-12): r = 48 for both bases at Q = 36 and 37 (k = 501-606 of 1022 rows, 7.1-8.8 MB
  per cell, ~140 MB for 18 cells), and ranks 8-62 (basis 800) at Q = 36.
* Done-when checks:
  - **Full basis vs direct:** r = n_free = 1022 (the analogue of r = 62 = n_free at N = 64; at N = 1024 a
    62-dimensional basis cannot reproduce the direct solution). Max relative error
    1.6e-13 with A_I.
  - **Online vs full-order** (20 validation queries, Q = 36). Uncompressed: 2.7e-15 (perturbed) and
    2.0e-10 (POD-Galerkin: the eps*kappa floor). Compressed: the rigorous per-step bracket holds to the
    floor; the compressed main part exceeds the double-precision full-order value by at most
    1.5e-07 per step, which equals the uncompressed per-step floor of
    1.6e-07.
  - **Timing, idle machine, one thread** (N = 1024 tensors, 10 validation queries): Q = 36: ell 1.27 + eps_Q
    1.33 + form 1.06 = **3.74 ms**; Q = 37: 3.84 ms; reduced solve
    0.94 ms. The tensors are N-specific; the cost is O((Q+2) k r + k r N_T) and does not
    depend on N by construction (k is bounded by the compressed rank, measured flat for N = 512-4096 in
    follow-up 3).
* POD-Galerkin on validation30 (basis 800, reduced solve at Q = 36). Median bound and, in brackets, the number
  accepted out of 30 at eps_N = 1.18e-3 (full sweep in `N1024.json` -> pod_galerkin_table; no false accepts
  anywhere):

  | r | true err | exact (A_I) | banded (A_I) | reduced online | reduced tightened |
  |---|---|---|---|---|---|
  | 8 | 1.19e+00 | 8.42e+00 (0) | 1.33e+01 (0) | 2.94e+01 (0) | 1.70e+01 (0) |
  | 16 | 1.39e-01 | 2.39e+00 (0) | 3.57e+00 (0) | 7.61e+00 (0) | 4.28e+00 (0) |
  | 24 | 2.98e-02 | 4.72e-01 (0) | 7.94e-01 (0) | 1.57e+00 (0) | 8.76e-01 (0) |
  | 32 | 7.11e-03 | 8.33e-02 (0) | 1.57e-01 (0) | 3.07e-01 (0) | 1.69e-01 (0) |
  | 40 | 1.29e-03 | 1.69e-02 (0) | 3.41e-02 (0) | 6.39e-02 (0) | 3.76e-02 (0) |
  | 48 | 3.12e-04 | 3.57e-03 (0) | 6.96e-03 (0) | 1.38e-02 (0) | 8.21e-03 (0) |
  | 56 | 6.00e-05 | 7.23e-04 (28) | 1.42e-03 (9) | 2.72e-03 (0) | 1.59e-03 (6) |
  | 62 | 1.90e-05 | 2.25e-04 (30) | 4.28e-04 (30) | 8.42e-04 (24) | 4.67e-04 (29) |

## Item 3. Trimmed POD-NN sweep at N = 1024 (running; `scripts/sweep_N1024.py`)

* Space-time temporal POD (alpha^n - alpha^0, n = 1..N_T): r_t = 22 (800 set) and 23 (4000 set), max projection
  error < eps_N/10 = 1.18e-4.
* Single-run speed (idle, 3 threads, n = 4000, lambda_R > 0): 4.2 s/epoch. Eight runs in parallel took about
  5 h per phase.

# Follow-up 5

* **The 0.016% worst margin of X-bar <= S at N = 1024 is expected, not a near-failure.** For a query with
  r ~ r_min = r-bar, both symbols tend to r m(0) as theta -> 0 (g1 -> 0, and s_mu - r m -> 0 there), so
  min_theta s_mu / x-bar -> 1 by construction. The relative margin vanishes at theta = 0 whatever the cell;
  validity rests on s_mu >= x-bar pointwise, which holds with c_mu >= 1.00016 on the sample.
* **Three-stage rule** (`cgmy_cert/three_stage.py`). Statuses: reduced_accept, reject_cheap (reduced-stage
  lower bound > eps), banded_accept, reject_banded (Prop. 4.3 with the query's own X_I and Theta_mu: an added
  status, since the banded stage can also reject), exact_accept, exact_reject. The reduced-stage lower bound
  subtracts the EIM allowance and the rounding/compression term before dividing by sqrt(Theta-bar_mu), so it
  bounds what the exact stage (residual with A_I) can certify. Theta-bar_mu is an upper bound (s_K' + eps_K',
  adaptive K', graded grid, derivative margin). Validity check on 10 candidates (POD-Galerkin and perturbed,
  N = 1024): lower bound <= exact bound in every case, and close to it (2.04e-3 vs 2.19e-3).

# Phase 3

## Item 0: cost model (`results/cost_model.json`, `tables_v9/tab_cost_v9.tex`)

* The brief asks for one constant per component. Those fits are reported, but for the direct solve (max relative
  deviation 156%), exact eta^S (115%) and the reduced solve (638%) a single constant times the operation count
  cannot fit. Each of the N_T steps and each GMRES/PCG iteration pays a fixed interpreter overhead (scipy
  gmres / Python loop), which dominates at small N or small r. A two-constant fit t = c' ops + d calls is
  reported as well: direct solve 11% (d = 37 us per iteration or step), exact 21% (d = 12 us), reduced solve 4%
  (d = 3.3 us per step), reduced certificate 4%. Assembly, spectral data, banded eta^X and eps_Q fit to within
  16% with one constant.
* Cost facts that matter for E2. spectral_data (graded grid, all spectral quantities) costs 477 ms per query at
  every N, about 3x the direct GMRES solve at N = 1024 (147 ms) and more than assembly (106 ms). The full-order
  banded certificate is therefore dominated by its symbol work, not by the banded solves (3.2 ms). The reduced
  online certificate (eps_Q incl. ell 2.73 ms + form 0.3-5.5 ms) avoids it.
* The reduced rows are measured at N = 1024 only, since the tensors are N-specific. Their cost depends on
  (Q, r, rho, N_T), and rho saturates for N >= 512 (`rho_vs_N`, Q = 29: 349-404 / 366-419 / 404-416 rows at
  N = 512 / 1024 / 4096).

## E2: two bugs found in the first run (that run is discarded)

* **Q_used() read E1.json (1.6 MB) inside every timed certified call**, adding about 8 ms to each reduced-path
  timing: 13-15 ms measured instead of 6.8-8.4 ms. Fixed by caching the lookup.
* **The true-error loop re-read the whole 236 MB reference array from the npz on every access** (NpzFile is lazy):
  about 318 000 reads, which turned ~50 min into 15+ h. The timings themselves were not affected, but the run never
  finished in reasonable time. Fixed by loading the array once. E1, E3 and E4 were checked for the same pattern:
  none.

## Phase 3 follow-up 1: fast (online) spectral data and Theta-bar

* `spectral_data(q, N, mode="fast")` and `three_stage.theta_bar(..., mode="fast")`:
  - grid: coarse graded grid (512 uniform + 250 geometric points per side on the scale min(G, M) h; 1012 nodes);
  - aliases: adaptive count K' = 10 upward until the weighted tail is <= 2% of s - r m on the grid;
  - rigour: the dropped aliases carry |N^(theta_k)|^2 = sin^{2p}(theta/2) (2/theta_k)^{2p}, so
    |s_exact - s_K'| <= sin^{2p}(theta/2) eps_K' (Lemma 1.2 bounds the sum with sin <= 1). alpha_lb and beta use the
    lower envelope, Theta and Theta-bar the upper one, all with the cell-wise derivative margins.
  The weighted tail vanishes at theta = 0 to order 2p, so beta (a ratio with g1 ~ theta^2) is unaffected by it.
  Cost: ~3.5 ms (one thread) instead of 480-670 ms.
* **Validation on spectral288 at N = 1024** (`results/spectral_fast.json`):

  | fast / fine | median | min | max |
  |---|---|---|---|
  | beta | 0.99995 | 0.99992 | 0.99996 (all <= 1) |
  | Theta | 1.0005 | 1.00005 | 1.0022 (all >= 1) |
  | alpha_lb | 1.00001 | 1.0000001 | 1.0082 |
  | Theta-bar | 1.0004 | 0.9977 | 1.0021 |

  K' = 10 for all 288 queries. alpha_lb and Theta-bar fall on the "unsafe" side of the brief's test for some
  queries, but both remain rigorous: the fine mode adds or subtracts the *unweighted* eps_K everywhere, which is
  looser than the exact sin^{2p} weighting. For alpha_lb the minimum is at theta = 0, where s_mu = r exactly
  (alpha_fast <= r holds for all 288). The coarse grid broke no bound: beta_fast <= beta_fine and
  Theta_fast >= Theta_fine on every query. eta^X on validation30 POD-Galerkin (r = 48, 56, 64, 80) changes by
  +0.002% (ratios 1.00002-1.00003), not the few percent expected.
* The fine mode stays the default (E0, tests, E3's verification-grade certificates). The online path (E2's
  direct + banded, the three-stage banded stage, Theta-bar, correction + recertify, E4) uses the fast mode. E1's
  certificates and verdicts were computed with the fine mode; the Theta-bar difference (+-0.2%) cannot change any
  E1 status at the margins observed. E1's timing step is rerun with the fast mode.
* **Adaptive alias count in the assembly (follow-up item 3): measured, not adopted.** Assembling A_I with K' < 100
  changes the discrete operator itself, while K = 100 is fixed for every Phase 3 experiment (references and true
  errors are defined against the K = 100 system). The potential saving is reported in `spectral_fast.json` -> split.
  The decision is the user's.

## E1 timing rerun, E4

* E1's timing step was rerun with the fast online path (`E1.json` -> timing, F5c). The first numbers (fine spectral
  data and Theta-bar) are superseded. E1's verdicts are unchanged (they were computed with the fine mode, which
  differs from the fast one by +-0.2% at most).
* E4 has no references for the 10^4 queries: the brief asks for acceptance fractions, correction fractions, online time
  and the offline cost, not true errors. False accepts are checked on bench120 (E2), validation30 (E1) and the 264 OOD
  queries (E3): zero everywhere.
* E4's offline cost C0 uses the recorded logs where they exist (POD with 4 BLAS threads, EIM build, tensor build,
  training wall times) and estimates where they don't. References: n x the measured single-thread direct solve
  (generated with 22 processes). Cell beta-bar: 1188 spectral_data evaluations x the cost-model constant (not logged
  when it ran). Training is the best run's recorded wall time on 3 threads in an 8-run sweep, stated as such.
* A helper meant to archive the E2 dry run waited on a `pgrep -f` pattern that also matched its own command line, so
  it never returned. The cold-split job waiting on its output sat idle for about a day. No result was affected.

# Phase 4

* **Clean-clone check** (`results/ARCHIVE_CHECK.json`): in a fresh clone and venv (`requirements.txt`), `build.sh`
  regenerates all 7 tables byte-identically and all 10 figures pixel-identically. `run_E0.py --nproc 1` takes 32 min
  (laptop estimate 46-54 min), and the quick tests give 51 passed, 16 skipped (the data-dependent tests skip
  without `data/`).
* **Non-deterministic seeding found and fixed.** E0.4 (and the perturbed-z0 candidate of E0.3) seeded their random
  vectors with Python's `hash(query_id)`, which is randomised per process. The clean-clone rerun therefore gave
  `tab_pcg_v9.tex` iteration counts differing by +-1. The seeding is now `zlib.crc32(query_id)`; the method-order
  shuffles in `run_reduced_cert.py` were fixed the same way. All other E0 numbers reproduce (to 1e-9 or better).
  **E0.3 and E0.4 were rerun with the fixed seeds (2026-10-25, at the user's request; 254 s + 20 s with 20 processes).**
  The regenerated `E0.json`, `tab_pcg_v9.tex` and F3 are the archive's numbers. A second E0.4 run with a different
  `PYTHONHASHSEED` reproduces `tab_pcg_v9.tex` byte-identically. Changes relative to the earlier table (and to the
  manuscript bundle's copy): N = 64 PCG 8/9 -> 8/8; N = 512 CG 226/291 -> 224/292; N = 2048 CG 646 -> 648;
  N = 4096 CG 1078/1909 -> 1080/1915. The sqrtTheta statistics (symbol only) and the POD-Galerkin effectivity ratios
  are unchanged to 1e-9.
* **F2/F3 PNGs regenerated.** Same data; the Phase 1 files had been rendered on another machine with a different
  default font. The committed versions are now the reproducible ones (matplotlib's bundled DejaVu Sans).
* **The `manuscript_v9/` in the repository is the Phase 1 skeleton**, not the paper-editing copy the brief assumes, and
  `scripts/make_tables_phase2.py` does not exist. Item 2 (claims) and item 3 (tag, zip) wait for the current
  manuscript.

# Phase 4 follow-up 1

* **Paper-side bundle** committed as fd561a3 ("paper-side corrections after NUMBER_CHECK"); the number check runs
  against that manuscript state.
* **One writer per table.** `run_E1.do_figs` no longer writes a table (it keeps F5b/F5c); `tab_pod_ranks_v9.tex` is
  written only by `scripts/make_tables_phase2.py` (E1-based). `tab_pod_ranks_E1_v9.tex` is deleted. Contrary to the
  brief, it was not byte-identical to the new `tab_pod_ranks_v9.tex`: same E1 numbers, different caption, label
  (`tab:pod_ranks`), column order and number format. The manuscript never input it.
* **`coercivity_constant_provable.py` reproduces the committed JSON to 1 ulp, not byte for byte.** Rerun here (numpy
  1.26.4): c1_provable = 5.107198991695751e-05 against the committed 5.1071989916957514e-05 (relative 2e-16), and
  high_branch_min differs in the last digit. The difference is float `pow` rounding. The committed (bundle) JSON is
  kept unchanged.

# Phase 4 follow-up 2

* **Bundle 2** committed unchanged as e8cc432. It also carried `tables_v9/tab_pod_ranks_E1_v9.tex` (identical to the
  file deleted in 770b154). Nothing writes it and the manuscript does not input it, so it is deleted again in the next
  commit.
* **The 370 ms legend is in F5c, not F6a.** Manuscript Figure 6 (`fig:online-time`) is `F5c_online_time`, drawn by
  `run_E1.do_figs`. F6a (Figure 7) prints no direct-solve time. The requested change is therefore made in
  `run_E1.do_figs`: the dashed direct-solve line and its legend now read the E2 bench120 medians from `E2.json` (337 ms
  cold, 259 ms reusable setup) through `run_E1.direct_legend_ms`, instead of the E1 timing value (370 ms,
  validation30[1:11], still stored in `E1.json timing/direct_ms`). F6a is unchanged; F5c is regenerated through
  `build.sh`. The claims `fig6_legend_cold` / `fig6_legend_reusable` check that the legend values equal the E2 paths.
* **Remark 4.3 allowance sizes:** no result-file record exists (`E2_3_reduced_cert.json` has no allowance field and
  holds only Q = 29; `run_reduced_cert_v2.log` does not print them). The claims keep `NOTES_FOR_PAPER.md` item 10 as
  their source. The text's Q = 37 value (3.9e-6) is the maximum in that note, not the median (4.0e-7).

# Phase 4 follow-up 3

* **Bundle 3** committed unchanged as 374f65e, including the paper-side edits of `run_E0.plot_e01` and
  `run_E3_1.do_table` and the two tables they write; `bash build.sh` reproduces both tables byte-identically and leaves
  every other table and figure unchanged. The bundle's copy of the brief (`CLAUDE_PHASE4_followup_3.md`, identical to
  the one supplied) stays untracked like the earlier briefs; `MANUSCRIPT_STATUS.md` is committed.
* **Claims:** `s51_calib_spots` and `th_eim_Q37` re-pointed to the new wording (values unchanged). No claim quoted the
  renamed status names in `prop_twostage.tex` / `prop_threestage.tex` or the online-rule figure, so none needed
  re-pointing. The Section 5.7 sentence "a gap of $46$--$1000$" (was 47) had no claim; two are added (`s57_gap_46`,
  `s57_gap_1000`, from the same paths as `abs_factor46` and `s57_best_1e3`). 296 claims, 296 pass.
