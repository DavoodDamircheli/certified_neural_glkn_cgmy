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
