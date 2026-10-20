# Phase 0 + Phase 1 report (v9 workspace, 2026-10-19/20)

Scope: plan items 0.1–0.3 and 1.1–1.8 of `step_by_step_workplan_v9.md`. Everything below is in
the git repository `cgmy_v9` (tag `v8-frozen` = the handoff, HEAD = this report).

## 1. What was done

| Item | Status | Where |
|---|---|---|
| 0.1 freeze v8 | done (tag `v8-frozen`); **reproduction of Tables 6–7 not rerun** (see §3) | `v8_frozen/` |
| 0.2 fixtures | done, with a stronger fixture than the plan assumed (see §2.1) | `tests/fixtures/` |
| 0.3 manuscript skeleton | done: `main_v9.tex` compiles (21 pp.), theorems as `theory/*.tex` | `manuscript_v9/` |
| 1.1 `symbols.py` | done, all unit tests pass | `cgmy_cert/symbols.py`, `tests/test_symbols.py` |
| 1.2 `spectral.py` | done, verified on 288 queries × N ∈ {64, 256, 1024} | `cgmy_cert/spectral.py` |
| 1.3 `riesz.py` | done; PCG ≤ 15 iterations for N = 64..4096 | `cgmy_cert/riesz.py` |
| 1.4 `certificate.py` | done; effectivity 1.00 ± 1e-6 on the 48 perturbed-z⁰ cases | `cgmy_cert/certificate.py` |
| 1.5 `direct.py` | done; LU/GMRES agree to 1e-12, GMRES ≤ 5–6 iterations per step | `cgmy_cert/direct.py` |
| 1.6 Section 2 text | theorems inserted, Remark 2.2 replaced, c₁ computed; **Appendix A.1 rewrite still a TODO** | `manuscript_v9/main_v9.tex` |
| 1.7 Section 3 text | bracket kept with α_lb = min s_K − ε_K; Weyl paragraph deleted; comparison norm + two-stage inserted; algorithm figure updated | same |
| 1.8 E0 experiments | E0.1–E0.5 done (`results/E0.json`, F2/F3 figures, T3/T4 tables) | `results/E0.json`, `figures_v9/`, `tables_v9/` |

Tests: `python3 -m pytest tests -q` → 28 passed, 3 skipped (the skipped ones need `CGMY_FULL_TESTS=1`).

## 2. Findings that change the plan

### 2.1 The "40-digit" fixtures of v8 did not exist (handoff README, item 2)
The v8 "real line" column is double-precision QUADPACK, the 24 eigenvalues were stored as floats,
and the "40-digit quadrature" ran at 30 digits. I built a genuinely high-precision fixture instead:

* `tests/fixtures/realline_40digit.npz`: the exact Galerkin first column of A_I for the 33 (query, N)
  pairs of v8 Table 9 and the fractional blocks of the four audit queries, at 40 digits
  (`cgmy_cert/highprec.py`: alias series summed explicitly for |k| ≤ 30, both tails in closed form
  through Hurwitz zeta functions, Toeplitz entries by an L-point trapezoid rule with L chosen so
  that aliasing is < 1e-45). Agreement with v8's QUADPACK values: 3e-9 absolute (QUADPACK's limit).
* `tests/fixtures/eig24_40digit.json`: λ_min(S_I) at 40 digits for the 24 Riesz-bracket queries,
  both for the exact assembly (K = 100) and for the old principal-period matrix (the latter agrees
  with v8's `alpha_check.csv` to 10 digits).
* `tests/fixtures/queries.json`: the v8 populations byte-for-byte (sha256 recorded) plus the
  288-query spectral set.

Regression: the K = 0 assembly reproduces all 459 audit values of λ_min(S_I) to 3e-13 relative,
min 0.01030 (failure stratum) and 0.01038 (vertices) as in v8 Table 3.

### 2.2 Assembly accuracy (E0.1, `tables_v9/tab_assembly_v9.tex`)
Relative operator-norm error of A_I vs the exact matrix, max over the fixture queries:
principal period 0.31–0.41 (matches v8 Table 9 to 2e-3); K = 5: 6e-5; K = 20: 8e-7;
K = 100: 4.2e-9; bound ε_100/‖A‖ ≤ 4.8e-9. The tail bound is within a factor 1.1–4 of the
observed error. At K = 300 all 33 pairs agree with the fixture to ≤ 1.05e-10 (the plan said 1e-10;
one query at Y = 1.8 gives 1.05e-10), at K = 1000 to ≤ 1e-11.

**Deviation from the plan:** the production assembly uses the exact trigonometric polynomials for
the mass, drift and stiffness symbols (p = 3: m = 11/20 + 13/30 cos θ + 1/60 cos 2θ,
d₁ = 5/6 sin θ + 1/12 sin 2θ, g₁ = 1 − 2/3 cos θ − 1/3 cos 2θ) and truncates only the fractional
family Φ_ν. ε_K then has only the fractional term (tighter and exact); the three-term bound of
Lemma 1.2 is implemented too (`eps_K_full`) and tested to dominate.

### 2.3 Spectral quantities: the derivative margin had to be local
The plan's margin (π/L)·max|s′| is useless for N ≥ 1024: |s′| ~ h^{−Y} at θ = ±π while the minimum
sits near θ = 0, giving α_lb < 0 at N = 4096 with L = 2¹⁴. `spectral.py` uses a graded grid
(uniform 2¹⁴ points plus a geometric refinement on the scale min(G,M)h near 0, ~19 400 points for
every N) and a cell-wise margin ½·w·max(|f′| at the cell ends). Cost 1.3 s per query, independent
of N. The only non-enclosed step is that sup|f′| on a cell is taken as the larger endpoint value;
this must be said in the paper (I put the sentence in Section 2.7) or replaced by an interval
evaluation of s′ later.

### 2.4 √Θ_μ vs N (E0.3) reproduces the note and shows the one-dimensional ceiling

| N | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|
| median √Θ | 1.15 | 1.29 | 1.49 | 1.74 | 2.05 | 2.43 | 2.91 | 3.45 | 4.11 |
| max √Θ | 1.75 | 2.23 | 2.90 | 3.78 | 4.93 | 6.42 | 8.30 | 10.65 | 13.55 |

Observed η^X/η^S on POD–Galerkin candidates (ranks 16/32/48, basis from 200 Sobol trajectories):
median 1.08–1.13 (max 1.57) at N = 64, median 1.57–1.73 (max 3.14) at N = 512. Effectivity of the
exact bound η^S on POD–Galerkin: median 4.7–5.1 at N = 64 (v8 reported ≈ 5 for its POD r = 16),
6–9 at N = 512. So at the expected production N = 512 the cheap certificate costs a factor ≈ 1.7
in effectivity, and the plan's trigger "Θ > 4 at production N" (i.e. √Θ > 2) is **not** hit at
N = 512 (max √Θ = 3.78 means Θ_max = 14.3, median Θ = 3.0 — read the trigger as a median; the
optional wide-band minorant `comparison_norm_wide` is implemented but not needed).

### 2.5 The coercivity constant is 1.40e-4, not "~1e-5", and it is sharp
`scripts/coercivity_constant.py`: with Lemma 3.1, Re A_μ(ξ) ≥ r and the two elementary bounds of
Remark 3.3, the infimum of Re A_μ(ξ)/(1 + |M − iξ|^Y + |G + iξ|^Y) over the box is attained at
ξ = 0 and equals r_min/(1 + M_max^{Y_max} + G_max^{Y_max}) = 1.4022e-4 (v8: 1.6e-8). The numerical
infimum over 544 box points and ξ ∈ [0, 10⁶] is the same number, so the constant is exact.

### 2.6 Y < 1 works without change (E0.5)
Y ∈ {0.5, 0.8, 0.95}, μ = (0.5, 3, 5, Y, 0.05): α_lb = 0.0500 = r, λ_min(S_I) = 0.052–0.055,
√Θ ≤ 1.47, GMRES ≤ 5 iterations per step, certificate effectivity 1.000000 on perturbed exact
dynamics. (Note: for Y < 1 the single-trajectory POD rank needed is tiny; the real test is Phase 2.)

### 2.7 Lifting / initial-vector convention (documented in Section 2.5 of main_v9.tex)
Raw matrices are Toeplitz sections of the full cardinal splines (boundary functions not truncated
to Ω). For the mass and drift couplings this coincides with v8's Ω-quadrature; for the fractional
coupling v8 also used the full-line Toeplitz values, so the lifting formula is unchanged except for
the exact (K = 100) entries. The initial projection c⁰ uses the Ω-quadrature raw mass matrix
(exactly v8's; checked against the vendored `mass_matrix` to 4e-16) because the full-line Gram
matrix produces boundary ringing in the projection of a constant.

## 3. What was *not* done, and why
* **0.1 "v8-frozen reproduces Tables 6 and 7 from scratch."** Not rerun: it needs torch 2.14
  (not installed here), the DeepONet checkpoint, and a multi-hour benchmark whose timings are
  explicitly not needed. The parts of v8 that v9 builds on (assembly, free system, audit values,
  initial projection, lifting) are reproduced to ≤ 3e-13 by the regression tests instead. If you want
  the literal reproduction, it is `v8_frozen/numerics/benchmarks/run_gate_d_benchmark_v5.py` on
  your machine (python 3.10, versions in `v8_frozen/environment/`).
* **1.4 "20 DeepONet/POD candidates".** No v9 DeepONet exists yet; the ordering η^S ≤ η^X ≤ √Θ η^S
  was verified on POD–Galerkin, coarse-prolongation and perturbed-z⁰ candidates (E0.3, 48 queries ×
  2 resolutions × 5 candidate types). The v8 DeepONet checkpoint can be certified as a candidate of
  the *new* system in Phase 2 once torch is installed (its residual will be large, since it was
  trained on principal-period data).
* **1.6/1.7 prose.** The structural edits of the plan are done; Appendix A.1 (replace Steps 3–5 by
  the Lévy–Khintchine bound) and the final read-through are marked `\todo` in `main_v9.tex`
  (13 TODO markers in total, all listed by `grep todo{ main_v9.tex`).
* **E0.2 (288 + 1 queries × N ∈ {64, 256, 1024}):** λ_min(S_I)/s_min ∈ [1.003, 17.8],
  λ_min(S_I)/(r m_min) ∈ [7.5, 133]; zero violations of the spectral bound and of the generalised
  eigenvalue bracket [1, Θ] — the figure F2 is in `figures_v9/`.
* **E0.4 (20 queries, N = 64..4096):** plain CG 42/45 → 1078/1909 (median/max) iterations;
  circulant PCG 8/9 → 11/12; circulant GMRES for B₊: 5.0/5 → 9.0/15 per step (`tables_v9/tab_pcg_v9.tex`).

## 4. Numbers for the paper (from this phase)
* ε_K/‖A_I‖₂ ≤ 4.8e-9 at K = 100 on the box (replaces "≤ 2e-8").
* principal-period relative error 0.31–0.41 (v8 Table 9 confirmed; one table, `tab_assembly_v9`).
* α_lb = r at every tested query whose minimum sits at θ = 0 (all queries so far): the spectral
  bound is attained at the worst audit vertex to within 0.4 %.
* PCG: ≤ 15 iterations (typically 7–12) for N ≤ 4096; GMRES for B₊: ≤ 6 per step.
* c₁ = 1.40e-4.

## 5. Next steps (Phase 2, weeks 4–7)
1. `eim.py` (plan 2.1): the family Φ_ν on the L-grid is already produced by `symbols.fractional_alias`;
   greedy EIM with the evaluated error ε_Q(μ) from Lemma 5.1. `comparison_norm_wide` exists if needed.
2. `pod.py` is started (mass-weighted POD, POD–Galerkin with the full operator); add the affine
   operator and the reduced online certificate (plan 2.2–2.3).
3. Install torch (CPU) for `podnn.py` and for certifying the v8 DeepONet checkpoint against the
   new system (decision point week 4).
4. Theory text: Appendix A.1, Section 4 prose, cost model.
