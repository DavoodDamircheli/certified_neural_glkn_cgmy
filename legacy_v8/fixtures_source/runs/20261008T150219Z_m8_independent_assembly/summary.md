# M8 independent assembly 20261008T150219Z_m8_independent_assembly

Production assembly DEFINES the discrete problem. convention_check = production (L=2n IFFT) vs exact quadrature of the same principal-period integral; galerkin_deviation = production vs real-line (exact Galerkin) coefficients, DIAGNOSTIC. Operator norms computed directly; nothing here enters a bound.

| comparison | N | q | entry A med/max | entry M max | entry D max | action max | ‖ΔA‖/‖A‖ med/max | ‖ΔS_I‖/‖S_I‖ max | Δλ_min(S_I) rel max | free gate agrees |
|---|---|---|---|---|---|---|---|---|---|---|
| convention_check | 64 | 10 | 0.000564 / 0.00103 | 4.04e-16 | 5.66e-16 | 0.00532 | 0.00695 / 0.0135 | 0.00231 | 4.05e-07 | 10/10 |
| convention_check | 128 | 10 | 0.00014 / 0.000259 | 4.04e-16 | 5.66e-16 | 0.00163 | 0.00326 / 0.00623 | 0.0012 | 6.02e-08 | 10/10 |
| convention_check | 256 | 10 | 3.33e-05 / 6.34e-05 | 3.03e-15 | 9.86e-15 | 0.000531 | 0.00155 / 0.00292 | 0.000608 | 8.29e-09 | 10/10 |
| convention_check | 512 | 3 | 1.13e-05 / 1.22e-05 | 7.87e-15 | 1.72e-14 | 0.000126 | 0.001 / 0.00111 | 0.000306 | 9.42e-10 | 3/3 |
| galerkin_deviation | 64 | 10 | 0.0926 / 0.108 | 4.04e-16 | 5.66e-16 | 0.253 | 0.382 / 0.412 | 0.408 | 0.000893 | 10/10 |
| galerkin_deviation | 128 | 10 | 0.0865 / 0.105 | 4.04e-16 | 5.66e-16 | 0.208 | 0.371 / 0.408 | 0.407 | 0.000436 | 10/10 |
| galerkin_deviation | 256 | 10 | 0.0843 / 0.102 | 3.03e-15 | 9.86e-15 | 0.199 | 0.358 / 0.405 | 0.404 | 0.000221 | 10/10 |
| galerkin_deviation | 512 | 3 | 0.0775 / 0.101 | 7.87e-15 | 1.72e-14 | 0.19 | 0.333 / 0.402 | 0.402 | 0.000106 | 3/3 |
