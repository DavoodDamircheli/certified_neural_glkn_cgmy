# M3 POD study 20261006T215800Z_m3_pod_study

Scope: ordinary floating-point evaluation of exact-arithmetic theory: the reliability theorems hold in exact arithmetic under their stated hypotheses (including alpha <= lambda_min(S_I)); every reported number is computed in IEEE double precision without validated/outward rounding. No machine-validated enclosure and no total continuum-price error bound is claimed.

All numbers below are rebuilt by scripts/v5_pod_summary.py from the raw CSVs in this directory.
Errors are vs the STORED reference (trusted GMRES solve, projected); its own augmented certificate is the reference allowance column. E_rel = max_n ||z_pod - z_ref||_{M_I} / max_n ||z_ref||_{M_I}; not comparable to the legacy time-averaged absolute rank-sweep error.

## Rank sweep (validation30, tolerance from configs/v5_free_workflow.yaml)

| basis | rank | proj (med/max) | init β₀ (med) | evol (med/max) | E_rel (med/max) | price err $ (med/max) | cand η_aug (med) | direct acc. | corrected acc. | other | t_total s (med) | ref allowance η / $ (max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mass-weighted | 8 | 3.319e-01 / 8.596e-01 | 3.319e-01 | 1.133e-01 / 6.763e-01 | 1.334e-03 / 4.170e-03 | 3.950e-02 / 9.497e-02 | 2.962e+00 | 0 | 30 | 0 | 3.911e-01 | 3.773e-06 / 6.580e-06 |
| mass-weighted | 16 | 2.351e-02 / 8.756e-02 | 2.351e-02 | 1.198e-02 / 6.477e-02 | 1.049e-04 / 4.372e-04 | 4.567e-04 / 5.054e-03 | 2.515e-01 | 0 | 30 | 0 | 3.788e-01 | 3.773e-06 / 6.580e-06 |
| mass-weighted | 32 | 3.341e-04 / 7.048e-04 | 1.444e-04 | 1.029e-04 / 4.722e-04 | 1.364e-06 / 3.396e-06 | 1.142e-06 / 4.191e-06 | 1.585e-03 | 0 | 30 | 0 | 3.588e-01 | 3.773e-06 / 6.580e-06 |
| mass-weighted | 48 | 4.162e-06 / 1.384e-05 | 7.646e-07 | 1.248e-06 / 5.586e-06 | 1.720e-08 / 5.569e-08 | 1.658e-08 / 3.858e-08 | 1.630e-05 | 30 | 0 | 0 | 7.520e-02 | 3.773e-06 / 6.580e-06 |
| mass-weighted | 62 | 2.583e-13 / 3.046e-13 | 2.019e-13 | 2.640e-07 / 1.526e-06 | 1.061e-09 / 6.130e-09 | 1.540e-08 / 3.804e-08 | 3.399e-11 | 30 | 0 | 0 | 7.548e-02 | 3.773e-06 / 6.580e-06 |
| Euclidean diag. | 8 | 2.863e-01 / 9.276e-01 | 2.863e-01 | 1.138e-01 / 6.032e-01 | 1.150e-03 / 4.446e-03 | 3.306e-02 / 1.180e-01 | 2.796e+00 | 0 | 30 | 0 | 3.903e-01 | 3.773e-06 / 6.580e-06 |
| Euclidean diag. | 16 | 2.304e-02 / 8.782e-02 | 1.973e-02 | 1.103e-02 / 6.737e-02 | 1.012e-04 / 4.374e-04 | 3.832e-04 / 6.088e-03 | 2.289e-01 | 0 | 30 | 0 | 3.784e-01 | 3.773e-06 / 6.580e-06 |
| Euclidean diag. | 32 | 3.409e-04 / 1.026e-03 | 1.419e-04 | 1.114e-04 / 7.282e-04 | 1.399e-06 / 5.027e-06 | 7.376e-07 / 4.334e-06 | 1.980e-03 | 4 | 26 | 0 | 3.606e-01 | 3.773e-06 / 6.580e-06 |
| Euclidean diag. | 48 | 3.981e-06 / 1.250e-05 | 7.759e-07 | 1.294e-06 / 6.165e-06 | 1.607e-08 / 5.096e-08 | 1.669e-08 / 3.860e-08 | 1.509e-05 | 30 | 0 | 0 | 7.530e-02 | 3.773e-06 / 6.580e-06 |
| Euclidean diag. | 62 | 1.900e-13 / 2.675e-13 | 1.193e-13 | 2.640e-07 / 1.526e-06 | 1.061e-09 / 6.130e-09 | 1.540e-08 / 3.804e-08 | 3.188e-11 | 30 | 0 | 0 | 7.549e-02 | 3.773e-06 / 6.580e-06 |

## Training-snapshot projection error in the M_I norm (what mass-weighted POD minimises)

| basis | rank | relative aggregate error | error on the (common) initial payoff vector z_h^0 |
|---|---|---|---|
| mass_weighted_pod | 8 | 1.068e-03 | 3.319e-01 |
| mass_weighted_pod | 16 | 9.952e-05 | 2.351e-02 |
| mass_weighted_pod | 32 | 1.173e-06 | 1.444e-04 |
| mass_weighted_pod | 48 | 1.231e-08 | 7.646e-07 |
| mass_weighted_pod | 62 | 7.461e-16 | 1.727e-13 |
| euclidean_svd_subspace_with_mass_orthonormalisation | 8 | 1.100e-03 | 2.863e-01 |
| euclidean_svd_subspace_with_mass_orthonormalisation | 16 | 1.021e-04 | 1.973e-02 |
| euclidean_svd_subspace_with_mass_orthonormalisation | 32 | 1.291e-06 | 1.419e-04 |
| euclidean_svd_subspace_with_mass_orthonormalisation | 48 | 1.242e-08 | 7.759e-07 |
| euclidean_svd_subspace_with_mass_orthonormalisation | 62 | 5.000e-16 | 8.070e-14 |

## Full-rank (62) recovery vs dense direct free march

| population | source | n | within residual allowance | max coef rel err | max M_I err | max err/allowance | max direct residual | max cond(F) |
|---|---|---|---|---|---|---|---|---|
| validation30 | sobol_seed999999 | 30 | 30 | 2.602e-14 | 4.357e-12 | 1.097e-02 | 3.791e-11 | 7.431e+00 |
| audit459 | exact_vertex | 8 | 8 | 5.062e-14 | 3.526e-12 | 9.875e-03 | 3.978e-11 | 7.466e+00 |
| audit459 | known_failure | 8 | 8 | 2.772e-14 | 3.005e-12 | 5.406e-03 | 3.264e-11 | 7.393e+00 |

## Status counts

- mass_weighted_pod r=8: {'accepted_after_correction': 30}
- mass_weighted_pod r=16: {'accepted_after_correction': 30}
- mass_weighted_pod r=32: {'accepted_after_correction': 30}
- mass_weighted_pod r=48: {'accepted_direct': 30}
- mass_weighted_pod r=62: {'accepted_direct': 30}
- euclidean_svd_subspace_with_mass_orthonormalisation r=8: {'accepted_after_correction': 30}
- euclidean_svd_subspace_with_mass_orthonormalisation r=16: {'accepted_after_correction': 30}
- euclidean_svd_subspace_with_mass_orthonormalisation r=32: {'accepted_direct': 4, 'accepted_after_correction': 26}
- euclidean_svd_subspace_with_mass_orthonormalisation r=48: {'accepted_direct': 30}
- euclidean_svd_subspace_with_mass_orthonormalisation r=62: {'accepted_direct': 30}
