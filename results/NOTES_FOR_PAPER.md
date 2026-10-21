# Notes for the paper (suggestions only; manuscript_v9/ is edited elsewhere)

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
