# Deviations from the plan / brief (Phase 2-3)

Each entry: what the plan says, what was done, the best achieved value, and where it is recorded.

## 2.1 EIM rank selection: Q = max(Q_tol, Q_op)

* **Plan:** stop at Q when the test sup-error on the 1000 Sobol nu (seed 2026) is <= 1e-9 relative
  to max|Phi_nu|. Done-when: `||A_I - A_Q||_2 <= eps_Q(mu) <= 1e-8 ||A_I||_2` on the 1000 test queries.
* **Finding:** at N = 64 the 1e-9 stopping rule gives Q_tol = 22, and with Q = 22 the second
  inequality fails on 2 of the 1000 test queries (max eps_Q/||A_I||_2 = 1.43e-8). The plan's two criteria
  are measured differently: sup-error relative to max|Phi_nu| on the grid, versus the operator-norm
  bound (grid term plus derivative margin) relative to ||A_I||_2, which also contains the drift and
  lambda_0 M_I parts.
* **Done:** the operational rank is Q = max(Q_tol, Q_op), where Q_op is the smallest Q whose max over
  the 1000 test queries of eps_Q/||A_I||_2 is <= 1e-8 (`scripts/run_eim.py select`). Result:

  | N | 64 | 128 | 256 | 512 | 1024 |
  |---|---|---|---|---|---|
  | Q_tol (1e-9 rule) | 22 | 26 | 27 | 29 | 29 |
  | Q_op (1e-8 operator bound) | 23 | 24 | 25 | 26 | 26 |
  | **Q used** | **23** | 26 | 27 | 29 | 29 |

  Only N = 64 changes (by one term). The strict test is unchanged; it passes with Q = 23
  (max eps_Q/||A_I||_2 = 1.65e-9). Numbers: `results/eim.json` -> `select`.

## 2.1 Stored basis: T(xi_q), not T(Phi_{nu_q})

* **Brief:** store "the Q first columns T(Phi_{nu_q})".
* **Done:** the stored first columns are those of the EIM basis functions xi_q (normalised greedy
  residuals). They span the same space as the Phi_{nu_q}, and they are what makes the interpolation
  matrix unit lower triangular, so ell(nu) is a triangular solve. The snapshot-basis matrix is
  ill-conditioned (the singular values of the training set fall to 1e-12 by index 40). The selected
  nu_q and the recursion coefficients are stored too, so T(Phi_{nu_q}) can be rebuilt if wanted.

## 2.1 Symmetrised family

* The K-truncated alias sum is not exactly conjugate-symmetric near theta = -pi, because the alias set
  {theta + 2 pi k, |k| <= K} is not symmetric there; the defect is O(eps_K). `ToeplitzOp` keeps the real
  part of the Toeplitz entries, so the assembled A_I is T of the *symmetrised* samples
  (Phi(theta) + conj Phi(-theta))/2 (tested in `tests/test_eim.py`). The EIM is built on, and eps_Q
  measured against, these symmetrised samples. A side effect is that every basis function has real
  Toeplitz entries exactly; without symmetrisation, greedy roundoff gave relative imaginary parts of
  9e-8.

## 2.1 EIM files are built for N <= 1024 only

* N = 2048 and 4096 are not production resolutions. They are needed only for the N-independence timing
  test of 2.3. `load_eim(N)` builds them on demand (`python3 scripts/run_eim.py build select --N 2048 4096`).
