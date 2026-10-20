# 20261007T175334Z_gate_d_v5_full

Two workloads, reported separately; never combined into one speed-up.

## trajectory

| mode | method | n | direct | corrected | fallback | over budget | uncertifiable | num. failure | no-cert delivered | verified | emp. checked | failed | unavailable | median s | delivered err med/max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reuse | free_cold_gmres | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.335 | 3.57e-07 / 1.79e-06 |
| reuse | free_direct_lu | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.271 | 2.75e-07 / 1.36e-06 |
| reuse | neural_direct_riesz | 120 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0.0782 | nan / nan |
| reuse | neural_free_workflow | 120 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.425 | 3.4e-07 / 1.91e-06 |
| reuse | neural_warm_gmres | 120 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.358 | 3.4e-07 / 1.91e-06 |
| reuse | pod32_free_workflow | 120 | 9 | 111 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.38 | 3.74e-07 / 0.000195 |
| reuse | pod48_free_workflow | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.0802 | 3.74e-06 / 2.44e-05 |
| reuse | reference_trusted | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.926 | 0 / 0 |

| mode | method | baseline | common/total | coverage m / b | median ratio | geomean ratio [CI] |
|---|---|---|---|---|---|---|
| reuse | free_cold_gmres | reference_trusted | 120/120 | 1 / 1 | 0.361 | 0.359 [0.358, 0.361] |
| reuse | free_direct_lu | reference_trusted | 120/120 | 1 / 1 | 0.293 | 0.29 [0.287, 0.293] |
| reuse | neural_direct_riesz | reference_trusted | 0/120 | 0 / 1 | nan | nan [nan, nan] |
| reuse | neural_free_workflow | reference_trusted | 120/120 | 1 / 1 | 0.46 | 0.458 [0.456, 0.459] |
| reuse | neural_warm_gmres | reference_trusted | 120/120 | 1 / 1 | 0.386 | 0.384 [0.383, 0.386] |
| reuse | pod32_free_workflow | reference_trusted | 120/120 | 1 / 1 | 0.41 | 0.363 [0.337, 0.387] |
| reuse | pod48_free_workflow | reference_trusted | 120/120 | 1 / 1 | 0.0866 | 0.0868 [0.0866, 0.0871] |
| reuse | free_cold_gmres | free_direct_lu | 120/120 | 1 / 1 | 1.24 | 1.24 [1.23, 1.25] |
| reuse | neural_direct_riesz | free_direct_lu | 0/120 | 0 / 1 | nan | nan [nan, nan] |
| reuse | neural_free_workflow | free_direct_lu | 120/120 | 1 / 1 | 1.58 | 1.58 [1.57, 1.59] |
| reuse | neural_warm_gmres | free_direct_lu | 120/120 | 1 / 1 | 1.33 | 1.33 [1.32, 1.33] |
| reuse | pod32_free_workflow | free_direct_lu | 120/120 | 1 / 1 | 1.39 | 1.25 [1.16, 1.34] |
| reuse | pod48_free_workflow | free_direct_lu | 120/120 | 1 / 1 | 0.295 | 0.299 [0.296, 0.303] |
| reuse | reference_trusted | free_direct_lu | 120/120 | 1 / 1 | 3.42 | 3.45 [3.42, 3.48] |

## terminal_price

| mode | method | n | direct | corrected | fallback | over budget | uncertifiable | num. failure | no-cert delivered | verified | emp. checked | failed | unavailable | median s | delivered err med/max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reuse | cos | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 9 | 111 | 0 | 0.124 | 0.0312 / 0.314 |
| reuse | free_cold_gmres | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.335 | 6.06e-08 / 3.51e-07 |
| reuse | free_direct_lu | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.271 | 5.34e-08 / 4.76e-07 |
| reuse | neural_direct_riesz | 120 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0.0782 | nan / nan |
| reuse | neural_free_workflow | 120 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.425 | 6.07e-08 / 5.23e-07 |
| reuse | neural_warm_gmres | 120 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.358 | 6.07e-08 / 5.23e-07 |
| reuse | pod32_free_workflow | 120 | 9 | 111 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.38 | 9.31e-08 / 1.86e-05 |
| reuse | pod48_free_workflow | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.0802 | 5.75e-08 / 5.02e-07 |
| reuse | reference_trusted | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.926 | 0 / 0 |

| mode | method | baseline | common/total | coverage m / b | median ratio | geomean ratio [CI] |
|---|---|---|---|---|---|---|
| reuse | free_cold_gmres | cos | 9/120 | 1 / 0.075 | 2.68 | 2.69 [2.66, 2.72] |
| reuse | free_direct_lu | cos | 9/120 | 1 / 0.075 | 2.24 | 2.23 [2.18, 2.27] |
| reuse | neural_direct_riesz | cos | 0/120 | 0 / 0.075 | nan | nan [nan, nan] |
| reuse | neural_free_workflow | cos | 9/120 | 1 / 0.075 | 3.43 | 3.43 [3.39, 3.46] |
| reuse | neural_warm_gmres | cos | 9/120 | 1 / 0.075 | 2.87 | 2.88 [2.85, 2.91] |
| reuse | pod32_free_workflow | cos | 9/120 | 1 / 0.075 | 3.1 | 3.08 [3.04, 3.12] |
| reuse | pod48_free_workflow | cos | 9/120 | 1 / 0.075 | 0.65 | 0.649 [0.648, 0.65] |
| reuse | reference_trusted | cos | 9/120 | 1 / 0.075 | 7.46 | 7.47 [7.43, 7.51] |

## reusable-setup amortisation (trajectory)

| method | C0 s | C1 s/query | Q=1 | Q=8 | Q=120 | Q=1000 | break-even vs reference_trusted | break-even vs free_direct_lu |
|---|---|---|---|---|---|---|---|---|
| free_cold_gmres | 0.568 | 0.332 | 0.9 | 0.403 | 0.337 | 0.333 | 1 | none |
| free_direct_lu | 0.575 | 0.268 | 0.844 | 0.34 | 0.273 | 0.269 | 1 | n/a |
| neural_direct_riesz | 0.568 | 0.0781 | 0.647 | 0.149 | 0.0829 | 0.0787 | none (verified 0/120 vs 120/120) | none (verified 0/120 vs 120/120) |
| neural_free_workflow | 0.564 | 0.423 | 0.987 | 0.493 | 0.428 | 0.423 | 1 | none |
| neural_warm_gmres | 0.566 | 0.355 | 0.921 | 0.426 | 0.36 | 0.356 | 1 | none |
| pod32_free_workflow | 0.566 | 0.355 | 0.921 | 0.426 | 0.36 | 0.355 | 1 | none |
| pod48_free_workflow | 0.568 | 0.0802 | 0.648 | 0.151 | 0.0849 | 0.0808 | 1 | 1 |
| reference_trusted | 0.565 | 0.924 | 1.49 | 0.994 | 0.928 | 0.924 | n/a | none |
