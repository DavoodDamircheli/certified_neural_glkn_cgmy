# M6 audit 20261007T175335Z_m6_audit459

Floating-point evaluation of exact-arithmetic theory. The 139 known_failure rows are deliberately selected and are NOT a production failure rate.

## Positivity (gate passes / n)

| stratum | n | free S_I | legacy S_h | legacy-only rejections | free-only rejections | exceptions | min λ_min(S_I) | min λ_min(S_h) | max α−λ (free) | max cond(S_I) | κ_I median/max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sobol_main_box | 256 | 256 | 253 | 3 | 0 | 0 | 0.0158 | -0.0362 | -5.27e-15 | 439 | 1.45 / 2.55 |
| exact_vertex | 64 | 64 | 48 | 16 | 0 | 0 | 0.0104 | -1.8 | -2.51e-15 | 525 | 1.78 / 2.68 |
| known_failure | 139 | 139 | 4 | 135 | 0 | 0 | 0.0103 | -30.4 | -8.69e-16 | 564 | 1.52 / 2.68 |

## Fixed-residual sharp comparison (both gates pass): q_free / q_full

| stratum | n | min | median | max | Schur bound holds |
|---|---|---|---|---|---|
| sobol_main_box | 253 | 0.167 | 0.957 | 0.999 | 253/253 |
| exact_vertex | 48 | 0.467 | 0.951 | 0.998 | 48/48 |
| known_failure | 4 | 0.895 | 0.981 | 0.995 | 4/4 |

## Integrated free-workflow traces (declared subset: first 16 per stratum)

| delivered | legacy_only_rejections_processed | n | n_accepted_after_correction | n_accepted_direct | predictor | stratum |
|---|---|---|---|---|---|---|
| 16 | 0 | 16 | 16 | 0 | DeepONet | sobol_main_box |
| 16 | 0 | 16 | 0 | 16 | pod_r48 | sobol_main_box |
| 16 | 3 | 16 | 16 | 0 | DeepONet | exact_vertex |
| 16 | 3 | 16 | 0 | 16 | pod_r48 | exact_vertex |
| 16 | 16 | 16 | 16 | 0 | DeepONet | known_failure |
| 16 | 16 | 16 | 0 | 16 | pod_r48 | known_failure |

## Augmented-bound validation (true max-in-time M_I error vs dense direct reference)

| candidate | n | bound holds | invalid | effectivity median / max | residual-only would underestimate |
|---|---|---|---|---|---|
| homogeneous_error_from_perturbed_ic | 48 | 48 | 0 | 1 / 1 | 48 |
| pod_r16 | 48 | 48 | 0 | 5.37 / 10.6 | 0 |
| raw_deeponet | 48 | 48 | 0 | 4.98 / 26.3 | 0 |
