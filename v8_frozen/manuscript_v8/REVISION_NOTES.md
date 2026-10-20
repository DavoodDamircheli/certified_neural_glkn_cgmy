# Editorial revision v8

Base: supplied CGMY_v7_Overleaf_bundle.zip. Date: 9 October 2026.

## Result

The manuscript is reduced from 51 to 23 pages, with approximately 53% fewer extracted PDF words. It has six main sections and one mathematical appendix, with no separate supplement. Body font and margins remain unchanged.

## Changes

1. Reorganised the paper into Introduction; CGMY model and constrained discretisation; Neural and reduced trajectory approximations; Residual certification and tolerance-controlled correction; Numerical evaluation; Interpretation and conclusion; and a single mathematical appendix.
2. Removed both historical appendices and all nine historical plot files from the new manuscript bundle. The two current plots are unchanged and renamed without revision prefixes.
3. Replaced the number-heavy abstract with a method-and-finding summary.
4. Removed internal revision terminology from manuscript prose and displayed captions, including version references, historical withdrawal language, gates, legacy labels and production-matrix terminology.
5. Consolidated discrete, exact-Galerkin and continuum scope statements, retaining the arithmetic and assembly qualifications essential to the claims.
6. Stated the central CN certificate and upper bracket for general initial discrepancy from the outset. Neural and POD candidates now share the augmented notation.
7. Moved explicit uniform constants, parameter stability, alternative time schemes, exponential shifts, derivative bounds and conditional Richardson estimates into the same mathematical appendix. Retained the central CN and bracket proofs in the main text.
8. Removed trivial standalone lifting, estimator-consistency and exact-correction lemmas; their conclusions remain in the text. Clarified that warm-starting individual correction solves does not remove the prescribed nonzero initial correction.
9. Consolidated architectures and training information around the evaluated DeepONet and mass-weighted POD methods. The absent training entry script remains a reproducibility limitation.
10. Retained all current core evidence: positivity, Riesz comparisons (resolved and unresolved distinguished), initial-error tests, matched costs, POD ranks, scaling, assembly differences, refinement, corner discrepancies and financial budgets.
11. Replaced repetitive amortisation and price-workload tables with summaries. Detailed Fourier reconciliation is replaced with the current independent-reference settings and agreement. Euclidean POD results are summarised; all original numerical tables remain as frozen source records.
12. Split wide assembly and scaling tables into panels and shortened benchmark labels; numerical cells were preserved. Corrected stale basis-identification language by consistently describing quadratic order-3 splines and the 62-free/66-stored coefficient distinction.

## Validation

- Compiled the final manuscript with repeated pdfLaTeX passes.
- All citation keys and source references resolve; no duplicate labels or overfull boxes.
- Retained table numerical tokens match frozen source values. Split-table identifiers are handled separately from results; omitted Euclidean rows remain in frozen source records.
- Both current figures are byte-identical to the input bundle.
- Removed wording checked in main source and included tables.
- Rendered and visually reviewed page layouts, including wide tables and numerical-to-conclusion flow.
- No solver or training run was performed, and no experimental values or timings were changed.
