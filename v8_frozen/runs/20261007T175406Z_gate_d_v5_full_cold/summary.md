# 20261007T175406Z_gate_d_v5_full_cold

Two workloads, reported separately; never combined into one speed-up.

## trajectory

| mode | method | n | direct | corrected | fallback | over budget | uncertifiable | num. failure | no-cert delivered | verified | emp. checked | failed | unavailable | median s | delivered err med/max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | free_cold_gmres | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.905 | 3.57e-07 / 1.79e-06 |
| cold | free_direct_lu | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.843 | 2.75e-07 / 1.36e-06 |
| cold | neural_direct_riesz | 120 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0.649 | nan / nan |
| cold | neural_free_workflow | 120 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.995 | 3.4e-07 / 1.91e-06 |
| cold | neural_warm_gmres | 120 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.927 | 3.4e-07 / 1.91e-06 |
| cold | pod32_free_workflow | 120 | 9 | 111 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.95 | 3.74e-07 / 0.000195 |
| cold | pod48_free_workflow | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.651 | 3.74e-06 / 2.44e-05 |
| cold | reference_trusted | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 1.49 | 0 / 0 |

| mode | method | baseline | common/total | coverage m / b | median ratio | geomean ratio [CI] |
|---|---|---|---|---|---|---|
| cold | free_cold_gmres | reference_trusted | 120/120 | 1 / 1 | 0.608 | 0.607 [0.607, 0.608] |
| cold | free_direct_lu | reference_trusted | 120/120 | 1 / 1 | 0.565 | 0.564 [0.563, 0.566] |
| cold | neural_direct_riesz | reference_trusted | 0/120 | 0 / 1 | nan | nan [nan, nan] |
| cold | neural_free_workflow | reference_trusted | 120/120 | 1 / 1 | 0.669 | 0.668 [0.667, 0.669] |
| cold | neural_warm_gmres | reference_trusted | 120/120 | 1 / 1 | 0.623 | 0.623 [0.622, 0.624] |
| cold | pod32_free_workflow | reference_trusted | 120/120 | 1 / 1 | 0.637 | 0.62 [0.608, 0.63] |
| cold | pod48_free_workflow | reference_trusted | 120/120 | 1 / 1 | 0.438 | 0.438 [0.437, 0.439] |
| cold | free_cold_gmres | free_direct_lu | 120/120 | 1 / 1 | 1.08 | 1.08 [1.07, 1.08] |
| cold | neural_direct_riesz | free_direct_lu | 0/120 | 0 / 1 | nan | nan [nan, nan] |
| cold | neural_free_workflow | free_direct_lu | 120/120 | 1 / 1 | 1.18 | 1.18 [1.18, 1.19] |
| cold | neural_warm_gmres | free_direct_lu | 120/120 | 1 / 1 | 1.1 | 1.1 [1.1, 1.1] |
| cold | pod32_free_workflow | free_direct_lu | 120/120 | 1 / 1 | 1.13 | 1.1 [1.08, 1.11] |
| cold | pod48_free_workflow | free_direct_lu | 120/120 | 1 / 1 | 0.773 | 0.776 [0.774, 0.779] |
| cold | reference_trusted | free_direct_lu | 120/120 | 1 / 1 | 1.77 | 1.77 [1.77, 1.78] |

## terminal_price

| mode | method | n | direct | corrected | fallback | over budget | uncertifiable | num. failure | no-cert delivered | verified | emp. checked | failed | unavailable | median s | delivered err med/max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold | cos | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 9 | 111 | 0 | 0.124 | 0.0312 / 0.314 |
| cold | free_cold_gmres | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.917 | 6.06e-08 / 3.51e-07 |
| cold | free_direct_lu | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.854 | 5.34e-08 / 4.76e-07 |
| cold | neural_direct_riesz | 120 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0.649 | nan / nan |
| cold | neural_free_workflow | 120 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 1 | 6.07e-08 / 5.23e-07 |
| cold | neural_warm_gmres | 120 | 0 | 120 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.938 | 6.07e-08 / 5.23e-07 |
| cold | pod32_free_workflow | 120 | 9 | 111 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.956 | 9.31e-08 / 1.86e-05 |
| cold | pod48_free_workflow | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 0.657 | 5.75e-08 / 5.02e-07 |
| cold | reference_trusted | 120 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 0 | 1.5 | 0 / 0 |

| mode | method | baseline | common/total | coverage m / b | median ratio | geomean ratio [CI] |
|---|---|---|---|---|---|---|
| cold | free_cold_gmres | cos | 9/120 | 1 / 0.075 | 7.39 | 7.4 [7.36, 7.43] |
| cold | free_direct_lu | cos | 9/120 | 1 / 0.075 | 6.94 | 6.93 [6.87, 6.99] |
| cold | neural_direct_riesz | cos | 0/120 | 0 / 0.075 | nan | nan [nan, nan] |
| cold | neural_free_workflow | cos | 9/120 | 1 / 0.075 | 8.1 | 8.08 [8.03, 8.13] |
| cold | neural_warm_gmres | cos | 9/120 | 1 / 0.075 | 7.61 | 7.58 [7.54, 7.62] |
| cold | pod32_free_workflow | cos | 9/120 | 1 / 0.075 | 7.75 | 7.73 [7.67, 7.78] |
| cold | pod48_free_workflow | cos | 9/120 | 1 / 0.075 | 5.33 | 5.31 [5.28, 5.34] |
| cold | reference_trusted | cos | 9/120 | 1 / 0.075 | 12.1 | 12.1 [12.1, 12.1] |

