#!/usr/bin/env python3
"""Phase 3 follow-up 1: validate the fast (online) mode of spectral_data and theta_bar against the fine mode, and split
the cold direct solve by component.  Writes results/spectral_fast.json.

  ratios   spectral288 at N = 1024: alpha_lb, beta, Theta (fast / fine) and timings; theta_bar (fast / fine) on the
           in-box queries of the set (cells CYm:3x3x2)
  etaX     change of eta^X on the validation30 POD-Galerkin candidates (r = 48, 56, 64, 80, basis 800) with the fast X_I
  split    cold direct solve at N = 1024 (10 bench120 queries, timing protocol): assembly (symbol on the L-grid with
           K = 100, Toeplitz FFTs), forcing, initial projection, circulant preconditioner, CN march; plus the assembly
           with the adaptive alias count K' (measured only: K = 100 is fixed by the production configuration)
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, BOX, load_queries_json  # noqa: E402
from cgmy_cert.spectral import spectral_data, fast_alias_count  # noqa: E402
from cgmy_cert.three_stage import theta_bar  # noqa: E402
from cgmy_cert.reduced_cert import Partition  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

N = 1024
RES = ROOT / "results"; DATA = ROOT / "data"
OUT = RES / "spectral_fast.json"


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1, default=float))


def in_box(q):
    return all(BOX[k][0] <= getattr(q, k) <= BOX[k][1] for k in ("C", "G", "M", "Y", "r"))


def _ratio_one(args):
    qd, cp = args
    q = Query(**qd)
    a = spectral_data(q, N)
    t0 = time.perf_counter()
    b = spectral_data(q, N, mode="fast")
    tf = time.perf_counter() - t0
    out = dict(query_id=q.query_id, alpha_fine=a.alpha_lb, alpha_fast=b.alpha_lb, beta_fine=a.beta, beta_fast=b.beta,
               Theta_fine=a.Theta, Theta_fast=b.Theta, K_fast=b.K, t_fast_s=tf)
    if in_box(q):
        c = cp[Partition("CYm:3x3x2").cell_of(q)]
        out.update(theta_bar_fine=theta_bar(q, N, *c, mode="fine"), theta_bar_fast=theta_bar(q, N, *c))
    return out


def do_ratios(res, nproc):
    pops = load_queries_json(ROOT / "tests" / "fixtures" / "queries.json")
    qs = pops["spectral288"]
    cp = {k: (v["rbar"], v["betabar"]) for k, v in json.loads((RES / "N1024.json").read_text())["cells"]["betabar"].items()}
    with Pool(nproc) as pool:
        rows = pool.map(_ratio_one, [(q.as_dict(), cp) for q in qs], chunksize=4)
    summ = {}
    for k in ("alpha", "beta", "Theta", "theta_bar"):
        rr = np.array([r[f"{k}_fast"] / r[f"{k}_fine"] for r in rows if f"{k}_fast" in r])
        summ[k] = dict(median=float(np.median(rr)), min=float(rr.min()), max=float(rr.max()), n=int(rr.size))
    summ["alpha"]["note"] = ("alpha_fast >= alpha_fine is expected and rigorous: the minimum of s_mu is at theta = 0 where "
                             "s_mu = r exactly and the sin^{2p}-weighted tail vanishes; the fine mode subtracts the unweighted "
                             "eps_K everywhere")
    summ["safe_side"] = dict(beta_all_le_1=bool(all(r["beta_fast"] <= r["beta_fine"] for r in rows)),
                             Theta_all_ge_1=bool(all(r["Theta_fast"] >= r["Theta_fine"] for r in rows)),
                             theta_bar_all_ge_1=bool(all(r["theta_bar_fast"] >= r["theta_bar_fine"] for r in rows if "theta_bar_fast" in r)),
                             alpha_le_r=bool(all(r["alpha_fast"] <= qd_r * (1 + 1e-12) for r, qd_r in
                                                 zip(rows, [q.r for q in qs]))))
    summ["K_fast_counts"] = {str(k): int(sum(r["K_fast"] == k for r in rows)) for k in sorted({r["K_fast"] for r in rows})}
    summ["t_fast_ms_median_pool"] = float(np.median([r["t_fast_s"] for r in rows]) * 1e3)
    res["ratios"] = dict(summary=summ, rows=rows, population="spectral288", N=N, provenance=provenance(nproc=nproc))
    print(json.dumps(summ, indent=1), flush=True)
    save(res)


def _etax_one(args):
    i, r = args
    from cgmy_cert.symbols import free_matrices
    from cgmy_cert.direct import free_forcing, initial_free_vector
    from cgmy_cert.certificate import certify
    from cgmy_cert.eim import load_eim
    from cgmy_cert.pod import AffineROM
    from cgmy_cert.reduced_cert import load_rc
    d = np.load(DATA / "references_N1024_NT240_validation30.npz")
    m = d["mu"][i]
    q = Query(C=m[0], G=m[1], M=m[2], Y=m[3], r=m[4], T=float(d["T"][i]))
    e = load_eim(N, build_if_missing=False)
    V = np.load(DATA / "pod_N1024_NT240_seed0_800_r128.npz")["V"][:, :r]
    rc = load_rc(DATA / f"redcert_N1024_r{r}_n800_CYm_3x3x2_Q36c_b128.npz")
    fs = free_matrices(q, N); forc = free_forcing(fs, q.T, 240); z0 = initial_free_vector(q, fs.mesh)
    Z = V @ AffineROM(V, e, Q=36).solve(q, rc.alpha0(), 240, with_eps=False).alpha
    a = certify(fs, forc, Z, z0, 0.0, mode="banded", spec=spectral_data(q, N)).etaX_aug
    b = certify(fs, forc, Z, z0, 0.0, mode="banded", spec=spectral_data(q, N, mode="fast")).etaX_aug
    return dict(i=i, r=r, etaX_fine=a, etaX_fast=b)


def do_etax(res, nproc):
    with Pool(nproc) as pool:
        rows = pool.map(_etax_one, [(i, r) for r in (48, 56, 64, 80) for i in range(30)], chunksize=2)
    summ = {}
    for r in (48, 56, 64, 80):
        rr = np.array([x["etaX_fast"] / x["etaX_fine"] for x in rows if x["r"] == r])
        summ[str(r)] = dict(median_ratio=float(np.median(rr)), min_ratio=float(rr.min()), max_ratio=float(rr.max()))
    res["etaX_change"] = dict(summary=summ, rows=rows, population="validation30, POD-Galerkin basis 800, Q = 36",
                              provenance=provenance(nproc=nproc))
    print(json.dumps(summ), flush=True)
    save(res)


def do_split(res, n_q=10):
    from cgmy_cert.timing import pin, check_single_thread, hardware, loadavg, protocol
    from cgmy_cert.symbols import free_matrices, symbols
    from cgmy_cert.direct import free_forcing, initial_free_vector, cn_march
    from cgmy_cert.riesz import circulant_preconditioner
    from cgmy_cert.params import Mesh
    check_single_thread(); pin()
    la0 = loadavg()
    pops = load_queries_json(ROOT / "tests" / "fixtures" / "queries.json")
    qs, warm = pops["bench120"][:n_q], pops["validation30"][0]
    ctx = {}
    for q in [warm] + qs:
        fs = free_matrices(q, N)
        ctx[q.query_id] = dict(fs=fs, forc=free_forcing(fs, q.T, 240), z0=initial_free_vector(q, fs.mesh),
                               Kp=fast_alias_count(q, N))
    meth = dict(
        assembly_symbol_K100=lambda q: symbols(q, N, K=100),
        assembly_total_K100=lambda q: free_matrices(q, N),
        assembly_total_Kadaptive=lambda q: free_matrices(q, N, K=ctx[q.query_id]["Kp"]),
        forcing=lambda q: free_forcing(ctx[q.query_id]["fs"], q.T, 240),
        initial_projection=lambda q: initial_free_vector(q, Mesh(N=N)),
        preconditioner=lambda q: circulant_preconditioner(sym_vals=ctx[q.query_id]["fs"].sym.m + 0.5 * (q.T / 240) * ctx[q.query_id]["fs"].sym.a,
                                                          n=ctx[q.query_id]["fs"].n, kind="symbol"),
        march=lambda q: cn_march(ctx[q.query_id]["fs"], ctx[q.query_id]["z0"], q.T, 240, solver="gmres", forcing=ctx[q.query_id]["forc"]),
        spectral_fine=lambda q: spectral_data(q, N),
        spectral_fast=lambda q: spectral_data(q, N, mode="fast"))
    t = protocol(meth, qs, warm, reps=5, seed=3)
    med = {k: float(np.median(v) * 1e3) for k, v in t.items()}
    res["split"] = dict(median_ms=med, K_adaptive_assembly=[ctx[q.query_id]["Kp"] for q in qs],
                        note="march includes the preconditioner construction (cn_march builds it); assembly_total = "
                             "symbol on the L-grid + Toeplitz FFTs; the adaptive-K assembly is measured, not adopted "
                             "(K = 100 is fixed in the production configuration)",
                        population=f"bench120[:{n_q}]", loadavg_before=la0, loadavg_after=loadavg(), hardware=hardware(),
                        provenance=provenance())
    print(json.dumps(med, indent=1), flush=True)
    save(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["ratios", "etax"])
    ap.add_argument("--nproc", type=int, default=14)
    a = ap.parse_args()
    res = load()
    res["meta"] = dict(description="fast (online) mode of spectral_data / theta_bar vs the fine mode; cold direct-solve split")
    for st in a.steps:
        dict(ratios=lambda: do_ratios(res, a.nproc), etax=lambda: do_etax(res, a.nproc), split=lambda: do_split(res))[st]()


if __name__ == "__main__":
    main()
