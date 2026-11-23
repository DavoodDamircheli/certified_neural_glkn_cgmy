# Follow-up 5 (paste to Claude Code; items 1–2 can run alongside the N = 1024 sweep, 3–4 after it)

Rules of `CLAUDE_PHASE2.md` unchanged. Items 1–2 of follow-up 4 are accepted as reported. The
0.016 % worst margin of X̄ ≤ S is expected and harmless: for a query with r ≈ r_min the two symbols
coincide at θ → 0 (both tend to r m(0) with g₁ → 0), so the relative margin vanishes there by
construction; note this in `DEVIATIONS.md` so nobody reads it as a near-failure.

## 1. Extend the POD rank range at N = 1024

The rank list 8..62 of the plan was inherited from N = 64, where 62 = N_h. At N = 1024
(N_h = 1022) the reduced model is meaningful up to r ≈ 128, and the acceptance table shows the
transition happens exactly where the list ends. Add r ∈ {64, 80, 96, 128} to the POD-Galerkin
study (both bases, 800 and 4000 trajectories) and report, on validation30 at ε_N × {0.1, 0.3, 1,
3, 10}: acceptance under the three certificates, true error, effectivity (bound / true error) per
certificate, online solve time and online certificate time (one thread, idle machine), and the
tensor size per (basis, Q, r). The reduced certificate's cost grows like Q²(2r+1)²; report where
it crosses 5 ms and 50 ms. Build the offline tensors only for the ranks you time.

## 2. A rigorous three-stage rule (reduced → banded full-order → exact), and its status counts

Proposition 4.3 of the paper generalises: for the reduced certificate the loss factor with respect
to the exact one is bounded by Θ̄_μ := sup_θ s_μ(θ) / x̄(θ) (Lemma 2.5), which is one symbol
evaluation at the query (graded grid, derivative margin, like Θ_μ). Implement
`certify_three_stage(fs, forcing, candidate, z_h0, eps, ...)` with statuses
`reduced_accept`, `reject_cheap` (when (β₀² + η_red²/Θ̄_μ)^{1/2} > ε), `banded_accept`,
`exact_accept`, `exact_reject`, escalating only when the cheaper bound neither accepts nor rejects
(the banded stage uses the query's own X_I and Θ_μ as in Prop. 4.3). Time each stage separately.
Use it for every candidate in E1 and report status counts at every ε of the sweep; this is the
table that shows what the N-independent certificate buys and what it costs.

## 3. After the sweep: evaluation, report, zip (follow-up 4 §4)

Add to the POD-NN evaluation: effectivity of each certificate against the true error, the pilot's
best bounds for comparison, and the space–time variant as a separate row. Then
`results/PHASE2_REPORT.md` and the Phase 2 zip. Hand over before E1–E4.

## 4. Notes for the paper (append to `NOTES_FOR_PAPER.md`)

- The effectivity of the exact certificate on POD-Galerkin candidates at N = 1024 is about 10
  (r = 48: true error 3.1e-4, bound 3.6e-3). This is consistent with Theorem 3.4 (efficiency is in
  the energy norm, which weights the high-frequency part of the error much more than the mass norm
  we compare to) and with E0.3 at N = 512; it is the honest reason why acceptance at ε_N needs
  r ≈ 56–64 rather than 48. Record the median effectivities per certificate and rank.
- The reduced online certificate is about 4× looser than the exact one at r = 56 (cell factor
  ≈ 1.9 × √Θ_μ ≈ 2): acceptance 0 / 9 / 28 of 30 for reduced / banded / exact. This is what the
  three-stage rule is for.
- Production configuration: N = 1024, N_T = 240, ε_N = 1.18e-3 (max spot factor 8.44); the two
  corner queries fix N (2.4 / 3.2 cents at N = 512 vs 0.14 / 0.24 at 1024); 60 steps give 4.6 cents
  at N = 1024.
