# M6 Riesz-bracket validation 20261007T175335Z_m6_riesz_bracket

Reference: mpmath 40-digit Cholesky of the same assembled S (exactly symmetrised). comparison_uncertainty = 8 n eps cond2(S) q_ref. 'numerically_unresolved' = a margin within ±uncertainty (mostly converged brackets, see side counts). Floating-point evaluation of exact-arithmetic theory.

| residual source | residual norm range | preconditioner | n | pass | unresolved | violation | exception |
|---|---|---|---|---|---|---|---|
| corrected | 5.81e-07–1.44e-05 | banded | 576 | 216 | 360 | 0 | 0 |
| corrected | 5.81e-07–1.44e-05 | jacobi | 576 | 432 | 144 | 0 | 0 |
| corrected | 5.81e-07–1.44e-05 | mass | 576 | 499 | 77 | 0 | 0 |
| corrected | 5.81e-07–1.44e-05 | none | 576 | 432 | 144 | 0 | 0 |
| manufactured | 6.85–9.54 | jacobi | 120 | 104 | 16 | 0 | 0 |
| manufactured | 6.85–9.54 | none | 120 | 105 | 15 | 0 | 0 |
| pod_r16 | 0.0347–7.53 | banded | 576 | 201 | 375 | 0 | 0 |
| pod_r16 | 0.0347–7.53 | jacobi | 576 | 432 | 144 | 0 | 0 |
| pod_r16 | 0.0347–7.53 | mass | 576 | 504 | 72 | 0 | 0 |
| pod_r16 | 0.0347–7.53 | none | 576 | 432 | 144 | 0 | 0 |
| pod_r32 | 0.000555–0.129 | banded | 576 | 212 | 364 | 0 | 0 |
| pod_r32 | 0.000555–0.129 | jacobi | 576 | 432 | 144 | 0 | 0 |
| pod_r32 | 0.000555–0.129 | mass | 576 | 495 | 81 | 0 | 0 |
| pod_r32 | 0.000555–0.129 | none | 576 | 432 | 144 | 0 | 0 |
| raw_neural | 0.964–78.8 | banded | 576 | 190 | 386 | 0 | 0 |
| raw_neural | 0.964–78.8 | jacobi | 576 | 432 | 144 | 0 | 0 |
| raw_neural | 0.964–78.8 | mass | 576 | 499 | 77 | 0 | 0 |
| raw_neural | 0.964–78.8 | none | 576 | 432 | 144 | 0 | 0 |
| reference | 7.1e-07–1.49e-05 | banded | 576 | 216 | 360 | 0 | 0 |
| reference | 7.1e-07–1.49e-05 | jacobi | 576 | 432 | 144 | 0 | 0 |
| reference | 7.1e-07–1.49e-05 | mass | 576 | 504 | 72 | 0 | 0 |
| reference | 7.1e-07–1.49e-05 | none | 576 | 432 | 144 | 0 | 0 |

Unresolved by side: corrected: {'lower+upper': 720, 'lower': 5}; manufactured: {'lower+upper': 30, 'lower': 1}; pod_r16: {'lower+upper': 720, 'lower': 15}; pod_r32: {'lower+upper': 726, 'lower': 7}; raw_neural: {'lower+upper': 724, 'lower': 27}; reference: {'lower+upper': 720}

alpha hypothesis (alpha <= mpmath lambda_min of the same S): 24/24 hold; max alpha - lambda_min = -2.53e-15

## relative upper margin (u - q)/q by PCG budget, no preconditioner

| source | budget | n | median | max |
|---|---|---|---|---|
| corrected | 0 | 72 | 32.6 | 143 |
| corrected | 1 | 72 | 4.02 | 18.9 |
| corrected | 2 | 72 | 2.04 | 6.29 |
| corrected | 5 | 72 | 1.39 | 10.2 |
| corrected | 10 | 72 | 0.283 | 8.87 |
| corrected | 20 | 72 | 0.00264 | 0.88 |
| corrected | 50 | 72 | 1.92e-16 | 5.06e-14 |
| corrected | 200 | 72 | 6.25e-17 | 5.15e-14 |
| manufactured | 0 | 15 | 9.17 | 22.4 |
| manufactured | 1 | 15 | 43.9 | 421 |
| manufactured | 2 | 15 | 46.3 | 321 |
| manufactured | 5 | 15 | 75.5 | 452 |
| manufactured | 10 | 15 | 71.7 | 313 |
| manufactured | 20 | 15 | 17.5 | 3.04e+03 |
| manufactured | 50 | 15 | 0.842 | 1.05e+03 |
| manufactured | 200 | 15 | 1.59e-13 | 1.02e+06 |
| pod_r16 | 0 | 72 | 27.8 | 209 |
| pod_r16 | 1 | 72 | 7.38 | 24.8 |
| pod_r16 | 2 | 72 | 3.75 | 19.3 |
| pod_r16 | 5 | 72 | 0.681 | 5.38 |
| pod_r16 | 10 | 72 | 0.0483 | 1.18 |
| pod_r16 | 20 | 72 | 0.00139 | 0.218 |
| pod_r16 | 50 | 72 | 1.71e-16 | 1.14e-14 |
| pod_r16 | 200 | 72 | 8.71e-17 | 1.19e-14 |
| pod_r32 | 0 | 72 | 60.4 | 236 |
| pod_r32 | 1 | 72 | 7.9 | 36.4 |
| pod_r32 | 2 | 72 | 0.796 | 8.38 |
| pod_r32 | 5 | 72 | 0.00938 | 0.108 |
| pod_r32 | 10 | 72 | 0.00047 | 0.0175 |
| pod_r32 | 20 | 72 | 1.03e-05 | 0.00275 |
| pod_r32 | 50 | 72 | 0 | 8.77e-16 |
| pod_r32 | 200 | 72 | 0 | 8.77e-16 |
| raw_neural | 0 | 72 | 15 | 291 |
| raw_neural | 1 | 72 | 4.3 | 20.8 |
| raw_neural | 2 | 72 | 2.93 | 11.5 |
| raw_neural | 5 | 72 | 1.19 | 7.64 |
| raw_neural | 10 | 72 | 0.176 | 5.17 |
| raw_neural | 20 | 72 | 0.00269 | 0.892 |
| raw_neural | 50 | 72 | -6.78e-16 | 7.09e-14 |
| raw_neural | 200 | 72 | -7.25e-16 | 7.09e-14 |
| reference | 0 | 72 | 30.6 | 183 |
| reference | 1 | 72 | 3.8 | 20.6 |
| reference | 2 | 72 | 1.85 | 7.68 |
| reference | 5 | 72 | 1.15 | 3.71 |
| reference | 10 | 72 | 0.388 | 2.9 |
| reference | 20 | 72 | 0.0108 | 1.95 |
| reference | 50 | 72 | -3.49e-16 | 4.35e-14 |
| reference | 200 | 72 | -3.43e-16 | 4.37e-14 |
