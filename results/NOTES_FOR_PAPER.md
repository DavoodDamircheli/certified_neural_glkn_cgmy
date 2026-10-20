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
