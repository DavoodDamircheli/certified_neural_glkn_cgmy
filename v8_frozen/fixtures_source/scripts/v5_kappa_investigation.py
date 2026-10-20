"""M4 (roadmap §4.1): reproduce and investigate the kappa=1 identity failure of
tests/test_efficiency_norm_kappa.py::test_kappa_is_one_when_A_equals_S_even_with_large_cond_S.

Reproduces the matrix with the test's own seed and construction
(np.random.default_rng(12345), first draw, _random_spd(12, cond_boost=1e8)) and
records: condition numbers, asymmetry of the stored S, Cholesky reconstruction
error, triangular-solve residuals, symmetry/identity defects of the transformed
operator B = L^{-1} A L^{-T}, and higher-precision (mpmath, 60 digits)
references computed on the SAME ROUNDED double-precision matrix entries (the
stored A and S are converted exactly to mpf; no ideal matrix is regenerated).

    PYTHONPATH=src:experiments:tests python3 scripts/v5_kappa_investigation.py [out.json]
"""
from __future__ import annotations

import json
import sys

import mpmath as mp
import numpy as np
import scipy.linalg as sla

sys.path.insert(0, "tests")
from test_efficiency_norm_kappa import _FakeOps, _random_spd  # noqa: E402
from certification.constants import kappa_operator_norm  # noqa: E402

mp.mp.dps = 60
EPS = np.finfo(float).eps


def mp_kappa(A: np.ndarray, S: np.ndarray) -> mp.mpf:
    """||S^{-1/2} A S^{-1/2}||_2 in 60-digit arithmetic for exactly the given
    double entries; S must be symmetric (it is passed already symmetrized)."""
    n = A.shape[0]
    Am, Sm = mp.matrix(A.tolist()), mp.matrix(S.tolist())
    w, Q = mp.eigsy(Sm)
    Sih = Q * mp.diag([1 / mp.sqrt(x) for x in w]) * Q.T
    B = Sih * Am * Sih
    sv = mp.svd_r(B, compute_uv=False)
    return max(sv[i] for i in range(n))


def rel_resid(res: np.ndarray, bound: np.ndarray) -> float:
    """max |res|/bound over entries with bound > 0; entries with bound == 0 must
    have res == 0 exactly (else +inf is returned)."""
    zero = bound == 0
    if np.any(res[zero] != 0):
        return float("inf")
    return float(np.max(res[~zero] / bound[~zero]))


def main(out=None):
    rng = np.random.default_rng(12345)
    n = 12
    S = _random_spd(n, rng, cond_boost=1e8)
    A = S
    rec = {"n": n, "seed": 12345, "construction": "tests/test_efficiency_norm_kappa._random_spd(12, rng, cond_boost=1e8), first draw"}
    rec["cond2_S"] = float(np.linalg.cond(S))
    rec["S_asymmetry_max_abs"] = float(np.max(np.abs(S - S.T)))
    rec["S_asymmetry_rel"] = float(np.linalg.norm(S - S.T, 2) / np.linalg.norm(S, 2))
    kappa = kappa_operator_norm(None, _FakeOps(A=A, S=S, M=np.eye(n)))
    rec["kappa_production"] = kappa
    rec["kappa_minus_1"] = kappa - 1.0

    L = sla.cholesky(S, lower=True)  # reads the lower triangle only, as production does
    S_low = np.tril(S) + np.tril(S, -1).T  # the symmetric matrix Cholesky actually factors
    absL = np.abs(L)
    rec["chol_reconstruction_max_abs"] = float(np.max(np.abs(L @ L.T - S_low)))
    rec["chol_reconstruction_rel_to_LLt"] = rel_resid(np.abs(L @ L.T - S_low), absL @ absL.T)
    rec["chol_backward_bound_gamma_n1"] = (n + 1) * EPS / (1 - (n + 1) * EPS)
    X = sla.solve_triangular(L, A, lower=True)
    rec["trisolve1_residual_rel"] = rel_resid(np.abs(L @ X - A), absL @ np.abs(X))
    Bt = sla.solve_triangular(L, X.T, lower=True)
    rec["trisolve2_residual_rel"] = rel_resid(np.abs(L @ Bt - X.T), absL @ np.abs(Bt))
    rec["trisolve_backward_bound_gamma_n"] = n * EPS / (1 - n * EPS)
    B = Bt.T
    rec["B_symmetry_defect_2norm"] = float(np.linalg.norm(B - B.T, 2))
    rec["B_identity_defect_2norm"] = float(np.linalg.norm(B - np.eye(n), 2))

    # mpmath references on the same stored entries
    k_low = mp_kappa(A, S_low)  # S = what Cholesky factors; A = full stored matrix
    k_sym = mp_kappa(A, 0.5 * (A + A.T))  # manuscript definition S = sym(A) (exact in mp? see below)
    Am = mp.matrix(A.tolist())
    Ssym_mp = (Am + Am.T) / 2  # exact symmetric part in mp, no double rounding
    w, Q = mp.eigsy(Ssym_mp)
    Sih = Q * mp.diag([1 / mp.sqrt(x) for x in w]) * Q.T
    k_sym_exact = max(mp.svd_r(Sih * Am * Sih, compute_uv=False))
    rec["mp_kappa_A_vs_lower_triangle_S"] = float(k_low)
    rec["mp_kappa_minus_1_A_vs_lower_triangle_S"] = float(k_low - 1)
    rec["mp_kappa_A_vs_symA_exact"] = float(k_sym_exact)
    rec["mp_kappa_minus_1_A_vs_symA_exact"] = float(k_sym_exact - 1)
    rec["production_minus_mp_lower"] = float(mp.mpf(kappa) - k_low)
    rec["first_order_forward_bound_n_eps_cond"] = n * EPS * rec["cond2_S"]
    # symmetric input: the same test with S exactly symmetric
    S_s = 0.5 * (S + S.T)
    k_s = kappa_operator_norm(None, _FakeOps(A=S_s, S=S_s, M=np.eye(n)))
    rec["kappa_production_symmetrized_input"] = k_s
    rec["mp_kappa_symmetrized_input"] = float(mp_kappa(S_s, S_s))
    print(json.dumps(rec, indent=2))
    rec["thread_env"] = {k: __import__("os").environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    if out:
        with open(out, "w") as f:
            json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
