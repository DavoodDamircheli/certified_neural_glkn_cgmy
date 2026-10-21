#!/usr/bin/env python3
"""Plan 2.2: exact-assembly reference trajectories, generated once.

    data/references_N{N}_NT{NT}_seed0_4000.npz         sobol_queries(4000, seed=0) (with T)
    data/references_N{N}_NT{NT}_seed0_800.npz          its first 800 (= sobol_queries(800, seed=0))
    data/references_N{N}_NT{NT}_validation30.npz       the validation30 population (tests/fixtures)
    data/references_N{N}_NT{NT}_{bench120,ref30}.npz   with --extra bench120 ref30 (follow-up 4)

Each file: Z (n_q, n_free, N_T+1) float64 free coefficient trajectories, mu (n_q, 5) = (C, G, M, Y, r),
T (n_q,), query_ids, gmres_max_iters (n_q,).  sha256 of every file goes to results/MANIFEST.json.
2.3, 2.4 and E1 read these files and never regenerate them.

Usage: python3 scripts/gen_references.py [--N 512] [--NT 240] [--nproc 20]
"""
from __future__ import annotations

import argparse, json, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, sobol_queries, load_queries_json  # noqa: E402
from cgmy_cert.direct import reference_trajectory  # noqa: E402
from cgmy_cert.eim import sha256  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

DATA = ROOT / "data"
MANIFEST = ROOT / "results" / "MANIFEST.json"


def ref_path(N: int, NT: int, tag: str) -> Path:
    return DATA / f"references_N{N}_NT{NT}_{tag}.npz"


def _one(args):
    qd, N, NT = args
    q = Query(**qd)
    _, _, tr = reference_trajectory(q, N, NT, solver="gmres")
    return tr.Z, max(tr.iters)


def generate(queries, N, NT, nproc):
    jobs = [(q.as_dict(), N, NT) for q in queries]
    with Pool(nproc) as pool:
        out = pool.map(_one, jobs, chunksize=4)
    Z = np.stack([o[0] for o in out]); it = np.array([o[1] for o in out])
    return Z, it


def write(path, queries, Z, it, N, NT, population):
    path.parent.mkdir(parents=True, exist_ok=True)
    mu = np.array([q.mu for q in queries]); T = np.array([q.T for q in queries])
    np.savez(path, Z=Z, mu=mu, T=T, query_ids=np.array([q.query_id for q in queries]), gmres_max_iters=it,
             N=N, N_T=NT, population=population)
    return dict(file=str(path.relative_to(ROOT)), sha256=sha256(path), bytes=path.stat().st_size,
                n_queries=len(queries), N=N, N_T=NT, population=population, dtype="float64",
                arrays="Z (n_q, n_free, N_T+1), mu (n_q, 5) = (C,G,M,Y,r), T, query_ids, gmres_max_iters",
                solver="reference_trajectory(q, N, N_T, solver='gmres'), K = 100, rtol 1e-13",
                max_gmres_iters_per_step=int(it.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--NT", type=int, default=240)
    ap.add_argument("--nproc", type=int, default=20)
    ap.add_argument("--extra", nargs="*", default=[], help="further fixture populations, e.g. bench120 ref30")
    a = ap.parse_args()
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q4000 = sobol_queries(4000, seed=0)
        q800 = sobol_queries(800, seed=0)
    assert all(x == y for x, y in zip(q800, q4000[:800])), "Sobol prefix property violated"
    fix = load_queries_json(ROOT / "tests" / "fixtures" / "queries.json")
    val = fix["validation30"]

    for tag, qs in [("seed0_4000", q4000), ("validation30", val)] + [(t, fix[t]) for t in a.extra]:
        path = ref_path(a.N, a.NT, tag)
        if path.exists():
            raise SystemExit(f"{path} exists; references are generated once (delete it deliberately to regenerate)")
        t0 = time.time()
        Z, it = generate(qs, a.N, a.NT, a.nproc)
        wall = time.time() - t0
        pop = "sobol_queries(4000, seed=0, include_T=True)" if tag == "seed0_4000" else f"{tag} (tests/fixtures/queries.json)"
        man["files"][path.name] = dict(write(path, qs, Z, it, a.N, a.NT, pop), wall_s=wall,
                                       provenance=provenance(nproc=a.nproc))
        print(f"{path.name}: {len(qs)} trajectories, {wall:.0f}s, max GMRES iters/step {it.max()}", flush=True)
        if tag == "seed0_4000":
            p800 = ref_path(a.N, a.NT, "seed0_800")
            pop800 = "sobol_queries(800, seed=0, include_T=True) = first 800 of the 4000 set"
            man["files"][p800.name] = dict(write(p800, q800, Z[:800], it[:800], a.N, a.NT, pop800),
                                           derived_from=path.name, provenance=provenance(nproc=a.nproc))
            print(f"{p800.name}: first 800", flush=True)
        del Z
        MANIFEST.write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
