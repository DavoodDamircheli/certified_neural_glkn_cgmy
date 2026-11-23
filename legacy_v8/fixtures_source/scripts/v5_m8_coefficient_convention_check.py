"""M8 finding (2026-10-08): convention of the fractional connection coefficients.

Compares, for production mesh N=64 and selected queries, three definitions of
Lambda(m; delta) = (1/2pi) * int |hat N_3(tau)|^2 S(tau; delta) e^{i tau m} dtau:
  production  vendor/cgmy_bspline/connection_coefficients.py: IFFT with L = 2*n_dof samples
              of the integrand on the principal period [-pi, pi) (toeplitz.py: L = 2N)
  principal   exact quadrature over (-pi, pi] (audit/independent_assembly.independent_lambda)
  real_line   integral over R (exact Galerkin entry by Plancherel), double-precision QUADPACK
              (Fourier-weighted on [1, inf)), cross-checked against an mpmath period sum
and the resulting spectra of S_h, S_I and S_h - r M_h. DIAGNOSTIC, no code is changed.

    OMP_NUM_THREADS=1 PYTHONPATH=src:experiments:audit:vendor python3 scripts/v5_m8_coefficient_convention_check.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "experiments", "audit", "vendor"):
    sys.path.insert(0, str(REPO_ROOT / sub))

import independent_assembly as ia  # noqa: E402
from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, Mesh  # noqa: E402
from cgmy_bspline.connection_coefficients import tempered_coeffs_via_ifft  # noqa: E402
from certification.constants import dense_from_apply  # noqa: E402
from certification.constrained import free_space_for_mesh  # noqa: E402
from common import run_manifest as rm  # noqa: E402
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1, ResultGroupWriter  # noqa: E402
from pipeline.predict_certify_correct import build_mesh_info  # noqa: E402

warnings.simplefilter("ignore")
QUERIES = ("Q_audit_0000", "Q_audit_0001", "Q_audit_0401", "Q_audit_0139")
LAGS = (0, 1, 2, 5, 10, 30, 65)


def F(t, d, Y, sign):
    hat = 1.0 if abs(t) < 1e-12 else ((1 - np.exp(-1j * t)) / (1j * t)) ** 3
    S = (d - 1j * t) ** Y - d ** Y if sign == "plus" else (d + 1j * t) ** Y - d ** Y
    return abs(hat) ** 2 * S


def lam_real_line(m, d, Y, sign):
    a = quad(lambda t: (F(t, d, Y, sign) * np.exp(1j * t * m)).real, 0, 1, limit=200)[0]
    re = lambda t: F(t, d, Y, sign).real  # noqa: E731
    im = lambda t: F(t, d, Y, sign).imag  # noqa: E731
    if m == 0:
        b = quad(re, 1, np.inf, limit=500)[0]
    else:
        b = (quad(re, 1, np.inf, weight="cos", wvar=m, limlst=200)[0]
             - quad(im, 1, np.inf, weight="sin", wvar=m, limlst=200)[0])
    return (a + b) / np.pi


def lam_real_line_mp(m, d, Y, sign, K=20):
    mp.mp.dps = 20
    f = lambda t: abs(ia._hat_N3(t)) ** 2 * ia._S(t, d, Y, sign) * mp.e ** (1j * t * m)  # noqa: E731
    tot = sum(mp.quad(f, [-mp.pi + 2 * mp.pi * k, 2 * mp.pi * k, mp.pi + 2 * mp.pi * k]) for k in range(-K, K + 1))
    return float(mp.re(tot) / (2 * mp.pi))


def main():
    pops = rm.declare_populations()
    audit = {r["query_id"]: r for r in pops["audit459"]["rows"]}
    run_dir = rm.create_run_dir(rm.new_run_id("m8_coefficient_convention"))
    w = ResultGroupWriter(run_dir / "convention", METHOD_CONSTRAINED_FREE_CN_V1)
    mesh = Mesh(N=64, p=3, x_min=-3.394829814011908, x_max=12.605170185988092)
    mi = build_mesh_info(mesh)
    ops = OperatorCache(mesh, mesh.x_min, mesh.x_max)
    n, h, I = mi.n_dof, ops.h, free_space_for_mesh(mi).free_idx
    M = ops.M_h.toarray()
    for qid in QUERIES:
        q = audit[qid]
        mu = CGMYParameters(*(q[k] for k in "C G M Y r T".split()))
        om, lam0, gY = ops._drift_constants(mu)
        K = {}
        for side, d in (("minus", mu.G * h), ("plus", mu.M * h)):
            prod = tempered_coeffs_via_ifft(d, mu.Y, 3, side, 2 * n)
            lam = {m: lam_real_line(m, d, mu.Y, side) for m in range(-(n - 1), n)}
            for m in LAGS:
                w.append_rows("coefficients.csv", [dict(
                    key=f"{qid}|{side}|{m}", method_id=METHOD_CONSTRAINED_FREE_CN_V1, query_id=qid, side=side, lag=m,
                    delta=d, production_L2n=float(prod[m % (2 * n)]),
                    principal_period_exact=ia.independent_lambda(m, d, mu.Y, side)[0],
                    real_line_quadpack=lam[m],
                    real_line_mpmath_K20=lam_real_line_mp(m, d, mu.Y, side) if m in (0, 1, 10) else "")])
            idx = np.arange(n)
            K[side] = np.vectorize(lambda i, j: lam[i - j])(idx[:, None], idx[None, :]) * h ** (-mu.Y)
        A_p = dense_from_apply(lambda v: ops.apply_A(mu, v), n)
        A_r = -(mu.r - om) * ops.D_h.toarray() - mu.C * gY * (K["plus"] + K["minus"]) + lam0 * M
        for name, A in (("production", A_p), ("real_line_galerkin", A_r)):
            S = 0.5 * (A + A.T)
            w.append_rows("spectra.csv", [dict(
                key=f"{qid}|{name}", method_id=METHOD_CONSTRAINED_FREE_CN_V1, query_id=qid, source=q["source"],
                operator=name, lambda_min_S_h=float(np.linalg.eigvalsh(S)[0]),
                lambda_min_S_I=float(np.linalg.eigvalsh(S[np.ix_(I, I)])[0]),
                lambda_min_S_h_minus_rM=float(np.linalg.eigvalsh(S - mu.r * M)[0]),
                rel_opnorm_diff_vs_real_line=float(np.linalg.norm(A - A_r, 2) / np.linalg.norm(A_r, 2)))])
        print(qid, "done", flush=True)
    resolved = rm.resolve_config()
    m = rm.build_manifest(run_dir.name, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                          studies={"assembly_refinement": "coefficient-convention diagnostic"}, seeds={},
                          resolved_config=resolved, populations=pops, population_used=list(QUERIES))
    m["role"] = "diagnostic of a code-vs-manuscript definition conflict; no code changed"
    rm.write_json(run_dir / "manifest.json", m)
    print(run_dir)


if __name__ == "__main__":
    main()
