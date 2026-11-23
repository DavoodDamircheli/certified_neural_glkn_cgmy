#!/usr/bin/env python3
"""Phase 0.2: build tests/fixtures/{queries.json, realline_40digit.npz, eig24_40digit.json}.

realline_40digit.npz : exact Galerkin first row/column of the free matrix A_I (and of the two
    fractional blocks) at 40 digits (stored as float64 + decimal strings for a few lags) for
    the 33 (query, N) pairs of v8 Table 9 (Q_scale_0000..0009 at N = 64, 128, 256; 0000-0002 at 512)
    and the four audit queries of v8's coefficient-convention run at N = 64.
eig24_40digit.json : lambda_min(S_I) of the exact-assembly (K = 100, double) free matrix for the
    24 v8 Riesz-bracket queries at N = 64, computed with mpmath eigsy at 40 digits; plus the
    old (principal-period) values of v8 for the record.
queries.json : the v8 populations, byte-identical to the frozen CSVs, plus the 288-query
    spectral set (32 vertices + 256 Sobol, seed 0) used in E0.
"""
from __future__ import annotations

import json, sys, time, hashlib
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, Mesh, box_vertices, sobol_queries   # noqa: E402
from cgmy_cert.highprec import exact_toeplitz_entries_mp, eigvals_min_mp  # noqa: E402
from cgmy_cert.symbols import free_matrices, legacy_principal_period_free  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
FIX.mkdir(parents=True, exist_ok=True)
V8 = ROOT / "legacy_v8"
DPS = 40
STRING_LAGS = (0, 1, 2, 5, 10, 30, 65)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- queries.json
def build_queries():
    qs = V8 / "query_sets"
    pops = {}
    for name, fname in (("audit459", "Q_audit.csv"), ("bench120", "Q_bench120.csv"),
                        ("ref30", "Q_ref30.csv"), ("validation30", "Q_validation30.csv"),
                        ("scale30", "Q_scale.csv"), ("price_diag8", "Q_price_diag8.csv")):
        df = pd.read_csv(qs / fname)
        pops[name] = [dict(query_id=r.query_id, C=r.C, G=r.G, M=r.M, Y=r.Y, r=r.r, T=r.T, K=r.K,
                           source=getattr(r, "source", "")) for r in df.itertuples()]
    spec = box_vertices() + sobol_queries(256, seed=0, include_T=False, prefix="sobol")
    pops["spectral288"] = [dict(query_id=q.query_id, C=q.C, G=q.G, M=q.M, Y=q.Y, r=q.r, T=q.T, K=q.K,
                                source="vertex" if q.query_id.startswith("vertex") else "sobol_seed0") for q in spec]
    out = dict(
        description="v8 query populations (Table 2) copied byte-for-byte from the frozen CSVs, "
                    "plus the 288-query spectral set (32 vertices of the 5-d box + 256 scrambled Sobol, seed 0).",
        source_sha256={f: sha(qs / f) for f in ("Q_audit.csv", "Q_bench120.csv", "Q_ref30.csv",
                                               "Q_validation30.csv", "Q_scale.csv", "Q_price_diag8.csv")},
        populations=pops)
    (FIX / "queries.json").write_text(json.dumps(out, indent=1))
    print("queries.json:", {k: len(v) for k, v in pops.items()})


# ---------------------------------------------------------------- real-line fixture
def _one(args):
    qd, N, which = args
    q = Query(**{k: qd[k] for k in ("C", "G", "M", "Y", "r", "T", "K")}, query_id=qd["query_id"])
    n = Mesh(N).n_free
    lags = range(-(n + 2), n + 3)          # includes the p-1 extra lifting lags
    t0 = time.time()
    ent, L = exact_toeplitz_entries_mp(q, N, lags, dps=DPS, which=which)
    ms = np.array(sorted(ent))
    vals = np.array([complex(ent[m]) for m in ms])
    strs = {str(m): (mp.nstr(ent[m].real, DPS), mp.nstr(ent[m].imag, DPS)) for m in STRING_LAGS if m in ent}
    return dict(query_id=qd["query_id"], N=N, which=which, lags=ms, values=vals, strings=strs, L=L,
                seconds=time.time() - t0)


def build_realline(nproc=2):
    scale = pd.read_csv(V8 / "query_sets" / "Q_scale.csv")
    audit = pd.read_csv(V8 / "query_sets" / "Q_audit.csv")
    tasks = []
    for N in (64, 128, 256):
        for i in range(10):
            tasks.append((scale.iloc[i].to_dict(), N, "a"))
    for i in range(3):
        tasks.append((scale.iloc[i].to_dict(), 512, "a"))
    for qid in ("Q_audit_0000", "Q_audit_0001", "Q_audit_0401", "Q_audit_0139"):
        row = audit[audit.query_id == qid].iloc[0].to_dict()
        tasks.append((row, 64, "G")); tasks.append((row, 64, "M")); tasks.append((row, 64, "a"))
    # large first
    tasks.sort(key=lambda t: -t[1])
    print(f"real-line fixture: {len(tasks)} tasks on {nproc} processes")
    with Pool(nproc) as pool:
        res = pool.map(_one, tasks, chunksize=1)
    data = {}
    meta = []
    for r in res:
        key = f"{r['query_id']}|N{r['N']}|{r['which']}"
        data[key + "|lags"] = r["lags"]; data[key + "|values"] = r["values"]
        meta.append(dict(key=key, query_id=r["query_id"], N=r["N"], which=r["which"], L=r["L"],
                         seconds=r["seconds"], strings=r["strings"]))
        print(f"  {key}: L={r['L']} {r['seconds']:.1f}s")
    qlookup = {}
    for df in (scale, audit):
        for row in df.itertuples():
            qlookup[row.query_id] = dict(C=row.C, G=row.G, M=row.M, Y=row.Y, r=row.r, T=row.T, K=row.K)
    np.savez(FIX / "realline_40digit.npz", **data,
             meta=json.dumps(dict(dps=DPS, p=3, x_min=Mesh(64).x_min, x_max=Mesh(64).x_max,
                                  convention="t_m = (2pi)^-1 int a(theta) e^{i theta m}; (A_I)_{kk'} = t_{k-k'}; "
                                              "'G','M' blocks are sum_k |N^|^2 (Gh + i theta_k)^Y resp. conj(.) with Mh, "
                                              "without the factor C Gamma(-Y) h^-Y",
                                  queries=qlookup, entries=meta)))
    print("saved realline_40digit.npz")


# ---------------------------------------------------------------- 24 eigenvalues
def build_eig24():
    old = pd.read_csv(V8 / "fixtures_source" / "alpha_check_24_eigenvalues.csv")
    audit = pd.read_csv(V8 / "query_sets" / "Q_audit.csv").set_index("query_id")
    out = []
    for row in old.itertuples():
        a = audit.loc[row.query_id]
        q = Query(C=a.C, G=a.G, M=a.M, Y=a.Y, r=a.r, T=a.T, K=a.K, query_id=row.query_id)
        fs = free_matrices(q, 64, K=100)
        S = fs.S.dense()
        t0 = time.time()
        lmin_exact = eigvals_min_mp(S, DPS)
        leg = legacy_principal_period_free(q, 64)
        lmin_legacy = eigvals_min_mp(leg["S"], DPS)
        out.append(dict(query_id=row.query_id, N=64, K=100,
                        lambda_min_exact_assembly_40digit=lmin_exact,
                        lambda_min_principal_period_40digit=lmin_legacy,
                        v8_lambda_min_mp=float(row.lambda_min_mp), v8_alpha=float(row.alpha),
                        eps_K=fs.eps_K, seconds=time.time() - t0))
        print(f"  {row.query_id}: exact {lmin_exact[:12]}  legacy {lmin_legacy[:12]}  v8 {row.lambda_min_mp:.10f}")
    (FIX / "eig24_40digit.json").write_text(json.dumps(dict(
        description="lambda_min(S_I) at N=64 of the exact-assembly free matrix (K=100, double entries) and of the "
                    "v8 principal-period matrix, both by mpmath eigsy at 40 digits; v8 values from alpha_check.csv.",
        dps=DPS, rows=out), indent=1))


if __name__ == "__main__":
    what = sys.argv[1:] or ["queries", "eig24", "realline"]
    if "queries" in what:
        build_queries()
    if "eig24" in what:
        build_eig24()
    if "realline" in what:
        build_realline(nproc=2)
