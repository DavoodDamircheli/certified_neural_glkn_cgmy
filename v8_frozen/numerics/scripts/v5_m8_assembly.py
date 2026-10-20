"""M8 §8.1: independent assembly vs the PRODUCTION operator at N=64/128/256 (+512 subset).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:experiments:audit:vendor \
        taskset -c 11-22 python3 scripts/v5_m8_assembly.py [--resume <run_dir>] [--workers 12]

Decision (2026-10-08): the production assembly DEFINES the discrete problem. Its fractional
blocks use Lambda(m) = IFFT of |hat N_3|^2 S(.;delta) sampled at L = 2*n_dof points of the
principal period [-pi, pi) (vendor connection_coefficients.py, toeplitz.py) -- a principal-
period FFT realisation of the symbol integral, NOT the exact real-line Galerkin entry.
This study therefore reports two different things, never conflated:

 1. convention_check -- does production implement ITS OWN convention correctly?
    production (L = 2n IFFT) vs exact quadrature of the SAME principal-period integral
    (audit/independent_assembly.independent_lambda, mpmath dps 30, never imports the audited
    routines), plus independently re-derived M_h, D_h and drift/lambda0. Differences here are
    the IFFT (Riemann-sum) discretisation of the stated convention.
 2. galerkin_deviation (DIAGNOSTIC) -- how far is the defining discrete operator from exact
    Galerkin assembly? production vs real-line coefficients (double-precision QUADPACK,
    Fourier-weighted on [1, inf); cross-checked by an mpmath period sum in
    scripts/v5_m8_coefficient_convention_check.py).

Comparator: OperatorCache.apply_A assembled densely (exact circulant-embedded Toeplitz action,
the operator every v5 residual/certificate/solve uses). Same domain (log K +- 8), N, order 3,
dof indexing, free index set for S_I. Per (query, N, comparison): relative max-entry error of
A_h (and M_h, D_h for 1), random-vector action errors (5 seeded vectors), relative operator
2-norm errors of A_h, S_h, S_I, and lambda_min/lambda_max of S_h, S_I on both sides.
Operator-norm differences are computed directly (no bound is inferred from eigenvalues), and
none of these discrepancies is used in any certificate or lower bound.

Population (declared): first 10 rows of scale30 at N=64,128,256; first 3 at N=512.
"""

from __future__ import annotations

import argparse
import multiprocessing as mpc
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "experiments", "audit", "vendor"):
    sys.path.insert(0, str(REPO_ROOT / sub))

POP_N = {64: 10, 128: 10, 256: 10, 512: 3}


def task(args):
    q, N = args
    import independent_assembly as ia
    import mpmath as mp

    from cgmy.operators import OperatorCache
    from cgmy.reference_solver import CGMYParameters, Mesh
    from certification.constants import dense_from_apply
    from certification.constrained import free_space_for_mesh
    from pipeline.predict_certify_correct import build_mesh_info

    x_min, x_max = -3.394829814011908, 12.605170185988092
    mesh = Mesh(N=N, p=3, x_min=x_min, x_max=x_max)
    mi = build_mesh_info(mesh)
    mu = CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
    t0 = time.perf_counter()
    ops = OperatorCache(mesh, x_min, x_max)
    A_p = dense_from_apply(lambda v: ops.apply_A(mu, v), mi.n_dof)
    M_p, D_p = ops.M_h.toarray(), ops.D_h.toarray()
    t_prod = time.perf_counter() - t0

    t0 = time.perf_counter()
    ia.assert_independent()
    h = (x_max - x_min) / N
    n = mi.n_dof
    M_i = ia.independent_mass_matrix(N, 3)
    D_i = ia.independent_derivative_matrix(N, 3) / h
    om, lam0, gY = ia._omega_S_and_lambda0_independent(mu.C, mu.G, mu.M, mu.Y, mu.r)
    idx = np.arange(n)
    qerr = 0.0
    Kp, Kr = {}, {}
    for sign, delta in (("minus", mu.G * h), ("plus", mu.M * h)):
        lp, lr = {}, {}
        for m in range(-(n - 1), n):
            v, e = ia.independent_lambda(m, delta, mu.Y, sign)
            lp[m] = v
            qerr = max(qerr, abs(e))
            lr[m] = real_line_lambda(m, delta, mu.Y, sign)
        Kp[sign] = np.vectorize(lambda i, j: lp[i - j])(idx[:, None], idx[None, :]) * h ** (-mu.Y)
        Kr[sign] = np.vectorize(lambda i, j: lr[i - j])(idx[:, None], idx[None, :]) * h ** (-mu.Y)
    A_conv = -(mu.r - om) * D_i - mu.C * gY * (Kp["plus"] + Kp["minus"]) + lam0 * M_i
    A_gal = -(mu.r - om) * D_i - mu.C * gY * (Kr["plus"] + Kr["minus"]) + lam0 * M_i
    t_ind = time.perf_counter() - t0

    fs = free_space_for_mesh(mi)
    I = fs.free_idx
    S_p = 0.5 * (A_p + A_p.T)
    eIp = np.linalg.eigvalsh(S_p[np.ix_(I, I)])
    ep = np.linalg.eigvalsh(S_p)
    rng = np.random.default_rng([20261008, N, int(q["query_id"].split("_")[-1])])
    V = rng.standard_normal((n, 5))
    PV = np.column_stack([ops.apply_A(mu, V[:, k]) for k in range(5)])
    rows = []
    for comp, A_i in (("convention_check", A_conv), ("galerkin_deviation", A_gal)):
        S_i = 0.5 * (A_i + A_i.T)
        ei, eIi = np.linalg.eigvalsh(S_i), np.linalg.eigvalsh(S_i[np.ix_(I, I)])
        acts = [float(np.linalg.norm(PV[:, k] - A_i @ V[:, k]) / np.linalg.norm(A_i @ V[:, k])) for k in range(5)]
        SIp, SIi = S_p[np.ix_(I, I)], S_i[np.ix_(I, I)]
        rows.append(dict(
            key=f"{q['query_id']}|{N}|{comp}", query_id=q["query_id"], N=N, n_dof=n, h=h, comparison=comp,
            rel_max_entry_err_A=float(np.max(np.abs(A_p - A_i)) / np.max(np.abs(A_i))),
            rel_max_entry_err_M=float(np.max(np.abs(M_p - M_i)) / np.max(np.abs(M_i))),
            rel_max_entry_err_D=float(np.max(np.abs(D_p - D_i)) / np.max(np.abs(D_i))),
            rel_action_err_max=max(acts), rel_action_err_median=float(np.median(acts)),
            rel_opnorm_err_A=float(np.linalg.norm(A_p - A_i, 2) / np.linalg.norm(A_i, 2)),
            rel_opnorm_err_S=float(np.linalg.norm(S_p - S_i, 2) / np.linalg.norm(S_i, 2)),
            rel_opnorm_err_S_I=float(np.linalg.norm(SIp - SIi, 2) / np.linalg.norm(SIi, 2)),
            lambda_min_S_prod=float(ep[0]), lambda_min_S_ind=float(ei[0]),
            lambda_max_S_prod=float(ep[-1]), lambda_max_S_ind=float(ei[-1]),
            lambda_min_S_I_prod=float(eIp[0]), lambda_min_S_I_ind=float(eIi[0]),
            lambda_max_S_I_prod=float(eIp[-1]), lambda_max_S_I_ind=float(eIi[-1]),
            rel_diff_lambda_min_S_I=float(abs(eIp[0] - eIi[0]) / abs(eIi[0])),
            free_gate_agrees=bool((eIp[0] > 0) == (eIi[0] > 0)),
            full_gate_agrees=bool((ep[0] > 0) == (ei[0] > 0)),
            max_quadrature_err_estimate_principal=qerr if comp == "convention_check" else "",
            t_production_s=t_prod, t_independent_s=t_ind))
    return rows


def real_line_lambda(m, d, Y, sign):
    """(1/2pi) int_R |hat N_3|^2 S e^{i tau m} d tau via QUADPACK (Hermitian fold to [0, inf))."""
    import warnings

    from scipy.integrate import quad

    def F(t):
        hat = 1.0 if abs(t) < 1e-12 else ((1 - np.exp(-1j * t)) / (1j * t)) ** 3
        S = (d - 1j * t) ** Y - d ** Y if sign == "plus" else (d + 1j * t) ** Y - d ** Y
        return abs(hat) ** 2 * S

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = quad(lambda t: (F(t) * np.exp(1j * t * m)).real, 0, 1, limit=200)[0]
        if m == 0:
            b = quad(lambda t: F(t).real, 1, np.inf, limit=500)[0]
        else:
            b = (quad(lambda t: F(t).real, 1, np.inf, weight="cos", wvar=m, limlst=200)[0]
                 - quad(lambda t: F(t).imag, 1, np.inf, weight="sin", wvar=m, limlst=200)[0])
    return (a + b) / np.pi


def main(argv=None):
    from common import run_manifest as rm
    from common.methods import METHOD_CONSTRAINED_FREE_CN_V1, NUMERICAL_SCOPE, ResultGroupWriter

    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    pops = rm.declare_populations()
    rows = pops["scale30"]["rows"]
    tasks = [(q, N) for N, k in POP_N.items() for q in rows[:k]]
    run_dir = args.resume or rm.create_run_dir(rm.new_run_id("m8_independent_assembly"))
    log = open(run_dir / "run.log", "a")
    log.write(f"[{time.strftime('%H:%M:%S')}] session_start git_head_at_start={rm.git_state()['head']} "
              f"population=scale30 first {POP_N} scope={NUMERICAL_SCOPE}\n")
    log.flush()
    w = ResultGroupWriter(run_dir / "assembly", METHOD_CONSTRAINED_FREE_CN_V1)
    done = w.existing_keys("assembly_vs_production.csv", "key")
    todo = [t for t in tasks if f"{t[0]['query_id']}|{t[1]}|galerkin_deviation" not in done]
    todo.sort(key=lambda t: -t[1])  # longest first
    with mpc.Pool(args.workers) as pool:
        for rows in pool.imap_unordered(task, todo):
            w.append_rows("assembly_vs_production.csv",
                          [dict(r, method_id=METHOD_CONSTRAINED_FREE_CN_V1) for r in rows])
            log.write(f"[{time.strftime('%H:%M:%S')}] {rows[0]['query_id']} N={rows[0]['N']} "
                      f"conv relop_A={rows[0]['rel_opnorm_err_A']:.3e} gal relop_A={rows[1]['rel_opnorm_err_A']:.3e} "
                      f"t_ind={rows[0]['t_independent_s']:.0f}s\n")
            log.flush()
    if not (run_dir / "manifest.json").exists():
        resolved = rm.resolve_config()
        m = rm.build_manifest(run_dir.name, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                              studies={"assembly_refinement": "independent assembly completed"},
                              seeds={"action_vectors": 20261008}, resolved_config=resolved, populations=pops,
                              population_used=[f"scale30 first {k} at N={N}" for N, k in POP_N.items()])
        m["comparator"] = "OperatorCache.apply_A dense (exact untruncated Toeplitz, the production operator)"
        rm.write_json(run_dir / "manifest.json", m)
    log.write(f"[{time.strftime('%H:%M:%S')}] done\n")


if __name__ == "__main__":
    main()
