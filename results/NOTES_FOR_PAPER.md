# Notes for the paper (suggestions only; manuscript_v9/ is edited elsewhere)

> Record kept as written during Phases 2-4. Paths in it refer to the layout of that time: `tables_v9/` is now
> `tables/`, `figures_v9/` is `figures/`, `v8_frozen/` is the slimmed `legacy_v8/`, `manuscript_v12/` is `paper/`;
> `manuscript_v9/`, the `CLAUDE_*.md` briefs, the `PHASE*_REPORT.md` files and `MANUSCRIPT_STATUS.md` were removed in
> the Phase 5a cleanup (they remain in the git tag `pre-cleanup`).

## Section 5 / Lemma 5.1 (EIM error), from plan 2.1

1. **The derivative margin is not needed for the assembled operators.** A_I and A_Q are Toeplitz
   sections whose entries are the L-point DFT coefficients of the symbol samples (L >= 2 N_h). The
   section T_{N_h} is then a compression of the L x L circulant whose eigenvalues are the samples (up
   to the unitary diagonal (-1)^m). Hence
   `||A_I - A_Q||_2 <= |C Gamma(-Y)| h^{-Y} max_grid |Phi_nu - I_Q Phi_nu|` holds rigorously, with no
   derivative term. The term (pi/L) max|e'| is needed only to bound the sup-norm of the *continuous*
   symbol difference, which is relevant if the paper compares against the exact (non-sampled) Galerkin
   operator. Observed: dense ||A_I - A_Q||_2 / grid term <= 0.996 (N = 512, 50 queries) and <= 0.964
   (N = 64, 1000 queries). The derivative margin adds a median 8% (N = 64) and 14% (N = 512).
   Suggestion: state Lemma 5.1 with the grid term for the assembled operators, and keep the
   derivative margin as a remark for the continuous symbol.
2. Numbers for the text (1000 Sobol test queries, seed 2026; `results/eim.json`):
   Q = 23 / 26 / 27 / 29 / 29 at N = 64 / 128 / 256 / 512 / 1024. Max eps_Q/||A_I||_2 = 1.7e-9 (N = 64)
   and 5.7e-10 (N = 512). Effectivity ||A_I - A_Q||_2 / eps_Q <= 0.91 (N = 64) and <= 0.93 (N = 512),
   with zero violations. At N = 512 the 4x-finer-grid symbol sup error / eps_Q <= 0.97.
   Q is essentially flat between N = 512 and 1024 (Figure F4b): the family's dependence on h saturates.
3. **Figure F4 caption suggestion:** "(a) Relative sup-norm EIM error on 1000 Sobol test parameters
   (solid) and normalised singular values of the 16^3 training snapshots (dotted), N = 64-1024.
   (b) EIM rank Q needed for a given test tolerance. The curves for N = 512 and N = 1024 coincide down to
   1e-9." The operational Q includes the operator-norm check (one extra term at N = 64; see
   `results/DEVIATIONS.md`).
4. The family is additive, Phi_(G,M,Y) = P(G h, Y) + P'(M h, Y), so the training snapshots cost
   2 x 16 x 16 alias sums. One could also interpolate the two halves separately (two 2-parameter
   families). Not done; it would give a different Q, and the plan fixes the 3-parameter EIM.

## Section 5 / Theorem 5.2 (POD-Galerkin with A_Q), from plan 2.2 (`results/E2_2_pod.json`)

5. **The EIM allowance limits the bound at high rank.** At Q = 29 (N = 512) the allowance
   eps_Q ||cbar_raw||_2 / sqrt(x_min) is small compared with the residual term for r <= 40 (ratio <= 1e-2).
   It reaches 8e-2 at r = 48, 0.7 at r = 56 and 2.2 at r = 62. Consequently eta^X computed on A_Q (with
   the allowance) exceeds eta^X computed on A_I by up to 4% at r = 48 and by a factor of up to 2.8 at r = 62.
   If the paper certifies ranks >= 56 at small eps_N, use Q from the 1e-10 column of F4b (Q = 32), or say
   that the rank is EIM-limited. The ||c||_2 factor is the crude part: ||cbar_raw||_2 is about 300 in
   these units.
6. POD singular values (mass-weighted, N = 512, N_T = 240), sigma_r/sigma_1 at r = 8/16/24/32/40/48/56/62:
   800 trajectories: 1.3e-3, 1.2e-4, 1.9e-5, 3.4e-6, 6.2e-7, 1.1e-7, 1.9e-8, 5.7e-9; 4000 trajectories:
   about the same (1.4e-3 ... 8.7e-9). Five times more data does not enlarge the space noticeably.
7. Validation30, POD-Galerkin on A_Q, median true error (max_n mass norm): r = 32: 4.2e-3; r = 40: 6.8e-4;
   r = 48: 1.2e-4; r = 56: 1.7e-5; r = 62: 6.5e-6. Median effectivity eta^S/err is 11-16 and eta^X/err is 18-21.
   These are higher than the E0.3 values (6-9 at N = 512) because the basis is now trained on 800
   trajectories over the whole box, so the residual is smaller and less structured.

## From follow-up 2 (section 3)

8. **Network output must be float64.** With |alpha| ~ 1e2, float32 rounding of the predicted POD
   coefficients (~1e-5 absolute) is amplified by 1/dt (time difference) and by the operator
   (||A|| ~ h^{-Y}) in the residual. The certified bound comes out ~1e-3 from float32 rounding alone,
   before any learning error. The POD-NN is trained and evaluated in float64.
9. **Parameter count.** The POD-NN MLP 6 -> 256 -> 256 -> 256 -> r N_T (r = 48, N_T = 240, so 11 520
   outputs) has 3.09 M parameters; the output layer alone has 2.96 M. The plan's "~0.5 M" was wrong.
10. **Allowance size and the "full-order certificate uses A_I" rule.** The online certificate must
    carry the EIM allowance; its L2-in-time size is about eps_Q ||cbar_raw||_2 sqrt(T / alpha_lb), with
    ||c_raw^0||_2 = 256 at N = 512. Validation30, eps_Q on the N-independent nodes:
    Q = 29: median 6.8e-5, max 4.3e-4; Q = 37: median 4.0e-7, max 3.9e-6.
    With Q = 29 the allowance therefore limits the online certificate at the 1e-4 level (this is the
    r >= 56 effect of note 5); Q = 37 (1e-11 EIM tolerance) removes it for every eps >= 1e-4. The
    *full-order* certificate never needs the allowance: it evaluates the residual with the exact
    operator A_I (two FFTs per step). Measured cost of keeping the allowance there instead
    (`E2_2_pod.json`, median eta^S): 0.4% at r = 40, 17% at r = 56, factor 1.5 at r = 62.
11. **Cell norm effectivity before and after the new cells** (validation30, POD-Galerkin r = 48,
    eta^{X-bar} / eta^X with the query's own X_I, median (max)): (r, C) 2 x 2 cells: 3.5 (10.1);
    (C, Y, min(G,M)) 2 x 2 x 2: 2.6 (5.1); 3 x 3 x 2: 1.8 (3.2). Against eta^S: 5.7 / 4.3 / 3.2 (medians).
    beta_mu = C beta~(G, M, Y), with beta~ in [41, 2417] driven mostly by Y and monotone on the table
    (nonincreasing in G and M, nondecreasing in Y). The cell minimum therefore sits at a corner; this
    monotonicity is observed, not proven, and should be stated as such. The conversion factor c_mu is
    >= 1.006 on 400 Sobol queries for every partition (X-bar <= S), so it is optional; with tightening the
    bound improves by sqrt(c_mu) (median c_mu = 3.0 for the 3 x 3 x 2 cells).
12. **N-independent eps_Q.** 161 nodes, about 1.6 ms (+ 1.2 ms for ell) at every N. It dominates the
    assembled-matrix quantity by >= 1.05 and is median 1.0-1.75x (max 2.1x) the old L-grid eps_Q. A
    sentence for the paper: "eps_Q(mu) is evaluated on a fixed node set of 161 points with a cell-wise
    derivative margin; it bounds the continuous sup-norm of the EIM error and hence, by Remark 2.9, the
    assembled operator difference, at a cost independent of N."

13. **beta-bar per cell (Remark 4.3).** Since beta_mu = C beta~(G, M, Y) and beta~ is nonincreasing in G and in M
    (Lemma 2.6: I_a(xi) decreases in a), the cell infimum over (C, G, M) is attained exactly at
    (C_min, G_max, M_max). With the min(G, M) split, these are the corners (m_hi, M_max) / (G_max, m_hi), or
    (G_max, M_max). Only the one-dimensional minimum over Y is taken on a grid (33 points) with the factor 0.99.
    In all 18 cells of the 3 x 3 x 2 partition this reproduces the earlier 9 x 10 x 13 table minimum exactly.
14. **Online cost.** The R-factor form is compressed offline by an SVD with a rigorous tail term
    sigma_(k+1) ||w||, keeping k ~ 350-420 rows (Q = 29, r = 48) at every N >= 512. The online certificate
    therefore costs O((Q+2) k r + k r N_T), independent of N, with a bound inflation <= 8e-4 (measured).

## From follow-up 5

15. **Exact-certificate effectivity on POD-Galerkin at N = 1024 is about 10** (r = 48: true error 3.1e-4 in the
    mass norm, bound 3.6e-3). This is consistent with Theorem 3.4: efficiency holds in the energy norm, which
    weights the high-frequency part of the error far more than the mass norm used for the true error. It also
    matches E0.3 at N = 512. It is the honest reason why acceptance at eps_N needs r ~ 56-64 rather than 48.
    Median effectivities per certificate and rank: see the table below (from `results/N1024.json` -> table_ext).
16. **The reduced online certificate is about 4x looser than the exact one at r = 56**: cell factor ~1.9 times
    sqrt(Theta_mu) ~ 2. Acceptance at eps_N is 0 / 9 / 28 of 30 for reduced / banded / exact. This is what the
    three-stage rule (reduced -> banded -> exact) is for. Its cheap rejection uses Theta-bar_mu = sup s_mu / x-bar
    (Lemma 2.5) and is tight in practice: the reduced-stage lower bound was within ~7% of the exact bound on the
    POD-Galerkin candidates tested.
17. **Production configuration:** N = 1024, N_T = 240, eps_N = 1.18e-3 (max spot factor 8.44). The two corner
    queries fix N (2.4 / 3.2 cents at N = 512 vs 0.14 / 0.24 at N = 1024); 60 steps give 4.6 cents at N = 1024.

18. **Effectivity per certificate and rank, and the three-stage counts at eps_N** (validation30, N = 1024, Q = 36;
    medians; acceptance counts reduced / banded / exact; three-stage counts reduced_accept / reject_cheap /
    banded_accept / exact_accept / exact_reject; `results/N1024.json` -> table_ext). No false accepts anywhere, and the
    reduced-stage lower bound never exceeded the exact bound:

    | basis | r | true err | eff exact | eff banded | eff reduced | eff reduced tight. | accepted | three-stage |
    |---|---|---|---|---|---|---|---|---|
    | 800 | 8 | 1.2e+00 | 6.8 | 10.8 | 23.7 | 14.0 | 0/0/0 | 0/30/0/0/0 |
    | 800 | 16 | 1.4e-01 | 14.9 | 25.2 | 50.5 | 29.2 | 0/0/0 | 0/30/0/0/0 |
    | 800 | 24 | 3.0e-02 | 13.9 | 25.2 | 52.0 | 27.4 | 0/0/0 | 0/30/0/0/0 |
    | 800 | 32 | 7.1e-03 | 13.3 | 23.2 | 48.4 | 26.3 | 0/0/0 | 0/30/0/0/0 |
    | 800 | 40 | 1.3e-03 | 13.2 | 24.1 | 47.2 | 26.8 | 0/0/0 | 0/30/0/0/0 |
    | 800 | 48 | 3.1e-04 | 11.7 | 21.6 | 49.4 | 25.9 | 0/0/0 | 0/30/0/0/0 |
    | 800 | 56 | 6.0e-05 | 11.5 | 21.5 | 46.5 | 26.1 | 0/9/28 | 0/2/9/19/0 |
    | 800 | 62 | 1.9e-05 | 11.3 | 22.0 | 49.7 | 26.6 | 24/30/30 | 24/0/6/0/0 |
    | 800 | 64 | 1.7e-05 | 10.5 | 20.0 | 41.9 | 23.6 | 26/30/30 | 26/0/4/0/0 |
    | 800 | 80 | 6.6e-07 | 9.5 | 19.3 | 46.5 | 26.2 | 30/30/30 | 30/0/0/0/0 |
    | 800 | 96 | 4.3e-08 | 9.2 | 15.4 | 142.7 | 78.0 | 30/30/30 | 30/0/0/0/0 |
    | 800 | 128 | 2.3e-08 | 4.6 | 5.3 | 118.6 | 70.8 | 30/30/30 | 30/0/0/0/0 |
    | 4000 | 8 | 9.6e-01 | 8.6 | 13.2 | 29.8 | 16.6 | 0/0/0 | 0/30/0/0/0 |
    | 4000 | 16 | 1.7e-01 | 12.6 | 21.1 | 42.6 | 25.2 | 0/0/0 | 0/30/0/0/0 |
    | 4000 | 24 | 3.4e-02 | 12.5 | 25.3 | 50.0 | 27.6 | 0/0/0 | 0/30/0/0/0 |
    | 4000 | 32 | 7.2e-03 | 11.5 | 21.2 | 42.9 | 24.4 | 0/0/0 | 0/30/0/0/0 |
    | 4000 | 40 | 1.5e-03 | 11.3 | 21.7 | 46.6 | 24.2 | 0/0/0 | 0/30/0/0/0 |
    | 4000 | 48 | 3.5e-04 | 11.0 | 22.4 | 46.4 | 24.3 | 0/0/0 | 0/30/0/0/0 |
    | 4000 | 56 | 8.6e-05 | 9.9 | 18.2 | 40.1 | 21.5 | 0/4/22 | 0/6/4/18/2 |
    | 4000 | 62 | 2.8e-05 | 9.9 | 20.2 | 43.1 | 22.8 | 19/30/30 | 19/0/11/0/0 |
    | 4000 | 64 | 1.8e-05 | 9.7 | 20.0 | 43.2 | 23.8 | 26/30/30 | 26/0/4/0/0 |
    | 4000 | 80 | 8.7e-07 | 9.7 | 18.4 | 44.1 | 23.7 | 30/30/30 | 30/0/0/0/0 |
    | 4000 | 96 | 5.2e-08 | 8.6 | 15.0 | 136.9 | 74.1 | 30/30/30 | 30/0/0/0/0 |
    | 4000 | 128 | 2.3e-08 | 4.6 | 5.3 | 118.7 | 70.8 | 30/30/30 | 30/0/0/0/0 |

    The exact effectivity is 9-15 for r <= 80 (4.6 at r = 128, where the mass-norm error reaches ~2e-8). At
    r >= 96 the reduced effectivity jumps to 120-140 because the true error (4e-8) drops below the floor set by
    the EIM allowance at Q = 36. That floor is harmless for acceptance, since everything is accepted there.
    The three-stage rule settles every r <= 48 query by reject_cheap: the reduced lower bound is tight. Every
    r >= 80 query is settled by reduced_accept. The banded and exact stages are needed only in the transition
    band r = 56-64.
19. **Offline tensor size per (basis, Q = 36, r)**, 18 cells, compressed: r = 8: 9 MB, r = 16: 24 MB, r = 24: 44 MB, r = 32: 69 MB, r = 40: 101 MB, r = 48: 138 MB, r = 56: 182 MB, r = 62: 220 MB, r = 64: 233 MB, r = 80: 356 MB, r = 96: 498 MB, r = 128: 688 MB
    (the 4000 basis is the same to within 1%). At r = 128 the compressed factor keeps all 1022 rows (k = n_free).

## POD-NN at the production resolution (follow-up 4 item 3, follow-up 5 item 3; `results/E2_4_podnn_N1024.json`)

20. **No POD-NN passes at eps_N = 1.18e-3 (nor at 1e-3).** Validation30, N = 1024, r = 48, median bounds
    (exact full-order / banded / reduced online):
    - best standard output (4000 trajectories, lambda_R = 1, 3 seeds): exact 1.24-1.27, banded 2.04-2.08,
      reduced 4.8-4.9; true error 0.46-0.47; exact effectivity ~2.8;
    - space-time output (r_t = 23 temporal modes, same setting, 3 seeds): exact 0.054-0.058, banded 0.082-0.089,
      reduced 0.18-0.19; true error 0.014-0.016; exact effectivity ~3.7. This is the best neural candidate, still
      ~47 x eps_N;
    - reference candidates on the same basis: projection of the references, exact 3.2e-3 (true error 3.3e-4);
      POD-Galerkin, exact 3.8e-3 (true error 3.5e-4);
    - the N = 512 pilot (6 runs stopped at ~1200 of 2000 epochs) reached median reduced bounds of 5.8-7.7 with the
      old cells (indicative only: different N, cells and Q).
    Every POD-NN query is settled by the cheap reduced-stage rejection at every eps of the sweep; there are no
    false accepts.
21. **The residual term in the loss trades accuracy for temporal smoothness.** With lambda_R = 0 the network fits
    the coefficients well (true error 0.08, 4000 trajectories), but its trajectory is noisy in time: bound 12.2,
    effectivity ~150. With lambda_R in {1e-2, 1e-1, 1} the bound drops ~10x (1.26), the true error rises ~5x
    (0.46) and the effectivity falls to ~3. The bound penalises time-difference noise amplified by 1/dt and by
    the operator. The space-time representation removes that noise by construction and wins on both measures.
    Data: 4000 trajectories beat 800 by ~4x in the bound (1.26 vs 5.2).
22. Selected epochs are 1890-2000 of 2000, so the networks are not converged at the end of the cosine schedule.
    Longer training would help somewhat, but cannot close the 47-1000x gap.
23. **Online timing at N = 1024** (idle machine, one thread, Q = 36; `N1024.json` -> timing_ext). The
    certificate (ell + eps_Q + form) takes 2.9 / 3.8 / 4.6 / 5.5 / 6.7 / 8.2 ms at r = 8 / 48 / 64 / 80 / 96 /
    128, with retained rows rho = 185-206 / 502-582 / 639-738 / 786-898 / 939-1016 / 1022. It crosses 5 ms at
    r ~ 74 and stays below 50 ms up to r = 128. The reduced solve takes 0.7-1.9 ms. About 2.7 ms of the
    certificate is the rank-independent ell + eps_Q part.

## Phase 3

24. **Cost model** (`results/cost_model.json`, `tables_v9/tab_cost_v9.tex`; one thread, idle). At N = 1024:
    assembly 106 ms, spectral data 477 ms (N-independent: fixed graded grid), GMRES march 147 ms, residual 6 ms,
    banded eta^X 3 ms, exact eta^S 137 ms. Reduced, r = 64: solve 1.1 ms, eps_Q + ell 2.7 ms, certificate form 1.8 ms.
    The full-order banded certificate is dominated by the spectral data (477 ms), not by its banded solves. This is
    the cost the reduced certificate avoids. Fitted constants: one constant per component as asked, plus
    t = c' ops + d calls where the per-step/per-iteration interpreter overhead matters (direct 11%, reduced solve 4%).
25. **E1** (`results/E1.json`; validation30, N = 1024, Q matched to eps). 37 candidates (POD-Galerkin r = 8..128 on two
    bases, 13 POD-NN checkpoints): no false accepts under any certificate at any eps, and the reduced-stage lower bound
    never exceeds the exact bound. The stored references' own certified bound is <= 2.0e-9 (negligible).
    POD-NN, bound- vs error-selected per setting (4000 trajectories): bound-selected lambda_R = 1 (median exact bound
    1.26, true error 0.46); error-selected lambda_R = 0 (bound 12.2, true error 0.082). The space-time variant
    (lambda_R = 1 only) has bound 0.055 and true error 0.015. Only one checkpoint per run was saved (the
    bound-selected epoch), so the selection is across lambda_R.
26. **Online time vs rank** (F5c; idle, one thread, validation30[1:11], eps_N). Reduced solve 0.75-2.0 ms; reduced
    certificate 2.9-8.2 ms; three-stage total ~200 ms for r <= 48 (all cheap rejections, dominated by the
    Theta-bar evaluation), 717 ms at r = 56 (banded/exact stages with spectral data), and equal to the reduced
    certificate from r = 62 (4.5-8.3 ms). Uncertified direct solve: 369 ms cold (assembly + forcing + z0 + march),
    170 ms warm.
27. **E3, out of distribution** (`results/E3.json` -> summary, breakdown; 264 out-of-box queries: 200 Sobol in the
    shells plus 64 extended-box vertices; ground truth = true error <= eps_N against a GMRES reference at the query).
    No false accepts for any candidate, certificate or eps; all 264 are certifiable; no aliasing adjustment was needed
    (G < 1: Toeplitz change <= 5.5e-16 at L = 4 N_h). Outside the box the reduced stage is not used (the cell beta-bar
    is valid only up to G_max, M_max), so the three-stage verdicts equal the full-order ones.
    - **The reduced model still works on 196 (r = 64) / 210 (r = 80) of the 264 out-of-box queries, and the
      certificate finds 181 / 193 of them** (exact and three-stage alike: 92%). The space-time POD-NN works on none
      (median true error 0.9-1.1) and is rejected on all.
    - Per shell (r = 80; queries / works / certified / median true error / median eps_Q; shells overlap):
      C > 1: 122 / 104 / 95 / 3.7e-6 / 5.3e-8; Y < 1.1: 39 / 36 / 31 / 6.1e-6 / 1.4e-8; Y > 1.8: 36 / 18 / 15 /
      2.8e-3 / 6.1e-7; G < 1: 27 / 21 / 13 / 3.9e-4 / 3.1e-6; M > 9: 111 / 101 / 94 / 8.6e-6 / 2.5e-7;
      vertices: 64 / 34 / 31 / 2.0e-4 / 4.3e-6.
      For r = 64: C > 1: 122 / 96 / 93; Y < 1.1: 39 / 31 / 25; Y > 1.8: 36 / 15 / 15; G < 1: 27 / 17 / 13; M > 9:
      111 / 93 / 88; vertices: 64 / 33 / 27.
      Y > 1.8 (rougher symbol) and G < 1 (slower decay) are where the reduced model fails most often. These are also
      the shells with the largest eps_Q (the EIM extrapolates in Y and G). In-box validation30 has median
      eps_Q 1.7e-9.
    - **Naive rule "reject everything outside the box"** as a detector of "true error > eps_N": recall 1 but
      precision 0.26 (r = 64) / 0.20 (r = 80); it discards all 196 / 210 good queries. The certificate has recall 1
      (by theorem) and precision 0.82 / 0.76: it discards only 15 / 17 good queries. The ensemble proxy of the
      space-time POD-NN could not make this comparison, because all its errors exceed eps_N.
28. **Cold direct-solve split at N = 1024** (one thread, idle; `spectral_fast.json` -> split, `cost_model.json` ->
    remarks): assembly 107 ms (almost all of it the symbol on the L-grid with K = 100), forcing 0.6 ms, initial
    projection 90 ms (query-independent, cached in the reusable setup, inside the timer in the cold setup as in v8),
    preconditioner < 0.1 ms, CN march 160 ms. Full-order certificate components: spectral data 2.5 ms (fast online
    mode; 478 ms in the fine verification mode), residual ~6 ms, banded solves ~3 ms.
    *Remark (not a method):* with the adaptive alias count of the fast spectral mode, the assembly would take 36 ms
    instead of 107 ms. K = 100 is kept for every experiment, because the assembled operator defines the discrete
    problem.
29. **Fast online spectral data** (`results/spectral_fast.json`): coarse graded grid (1012 nodes), alias count
    K' = 10, sin^{2p}(theta/2)-weighted tail bound (the dropped aliases carry this factor exactly), rigorous with the
    same derivative margins. Fast / fine on spectral288 at N = 1024: beta 0.99992-0.99996, Theta 1.00005-1.0022,
    alpha_lb 1.0000-1.008 (fast is exact at theta = 0, where s_mu = r), Theta-bar 0.998-1.002. eta^X on POD-Galerkin
    changes by +0.002%. This turns the banded full-order certificate from ~490 ms (twice the solve) into ~12 ms
    (6.0% of the solve, 7.2% with the spectral data; see 30).
30. **E2, matched benchmark on bench120** (`results/E2.json`, `tables_v9/tab_benchmark_v9.tex`, F6a/F6b; N = 1024,
    one thread, idle, 5 repetitions, geometric-mean ratio to the uncertified direct solve with 95% query-bootstrap CI).
    Reusable setup: direct 259 ms; direct + banded certificate
    1.059x (the dry run with the fine spectral data gave 2.87x, i.e.
    752 ms vs 258 ms; after the fix 275 ms). Certificate fraction of the solve:
    **6.0% for the certificate proper** (residual + banded
    solves) and **7.2% including the per-query
    spectral data**. Cold: 3.0% /
    3.8% of a larger cold solve.
    POD-Galerkin r = 80 with the three-stage rule: 0.032x (reduced accept
    on 120/120 at every eps); r = 64: 0.041x at eps_N (102 reduced +
    18 banded accepts), 0.026x at 3 eps_N, and
    0.55x at 0.3 eps_N (most queries need the banded or exact
    stage; 4 corrected). Space-time POD-NN: 1.07x (cheap rejection
    + correction on 120/120). False accepts: 0 for every method and eps. In the cold setup the ratios of the reduced methods
    rise (r = 80: 0.30x) because z_h^0 (90 ms) is rebuilt per query
    inside the timer.
    Refinement (F6b, bench120[:20], N_T = 240): banded certificate / direct solve =
    0.052 (N = 64), 0.055 (N = 128), 0.045 (N = 256), 0.048 (N = 512), 0.050 (N = 1024), 0.056 (N = 2048), 0.054 (N = 4096). This is flat at ~5% from N = 64
    to 4096. The reduced online certificate (4.6 / 5.6 ms at r = 64 / 80) does not depend on N.
31. **E4, 10^4 queries** (`results/E4.json`, F8; Sobol seed 4, eps_N, one thread, idle, one pass, method order shuffled
    per query). Mean online time per query: direct 260 ms; certified POD-Galerkin r = 80: **9.0 ms** (99.77% reduced
    accept, 0.14% banded, 0.02% exact, 0.07% rejected and corrected); r = 64: 21.2 ms mean / 7.2 ms median (89.7% reduced,
    9.8% banded, 0.13% exact, 0.35% corrected); space-time POD-NN: 282 ms (100% cheap rejection and correction).
    Offline cost C0 (POD-Galerkin, basis 800) ~ 1530 s: references 208 s (single-thread estimate: 800 x direct),
    POD 797 s (4 BLAS threads), EIM 65 s, cell beta-bar 425 s (estimate: 1188 spectral_data evaluations x 0.48 s), tensors
    39 s. Fitted T(Q) = 1533 s + 9.0 ms Q (r = 80) and 1533 s + 21.2 ms Q (r = 64), against 260 ms Q for direct.
    **Break-even at ~6 100 (r = 80) and ~6 400 (r = 64) queries.** The space-time POD-NN never breaks even: its online
    cost exceeds the direct solve, and its C0 (~26 700 s) is dominated by training (best run 5.9 h wall, 3 threads in an
    8-run sweep on 24 threads; recorded, not single-thread).
32. **E1 timing with the fast online path** (supersedes item 26; `E1.json` -> timing): the three-stage total at eps_N is
    5.3-6.2 ms for r <= 48 (cheap rejection now costs ~3.5 ms instead of ~200 ms), 154 ms at r = 56 (banded/exact stages),
    and equal to the reduced certificate for r >= 62 (4.4-8.3 ms). The uncertified direct solve takes 370 ms cold and
    167 ms warm.

33. **Number check (Phase 4, item 2).** `results/NUMBER_CHECK.md` lists every number in the abstract, Sections 2.5,
    4.1, 4.3-4.5, 5.1-5.10, 6 and theory/*.tex that the result files do not support, with the value the result file
    gives. 264 claims: 230 pass, 15 substantive mismatches, 14 numbers that no result file holds, 5 rounding-level.
    Regenerate with `python3 scripts/make_paper_claims.py && python3 scripts/check_numbers.py`.
