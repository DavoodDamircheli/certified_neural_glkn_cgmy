"""M8 §8.2: spatial, temporal and domain refinement of the PRODUCTION discrete problem.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:experiments:. \
        taskset -c 11-22 python3 scripts/v5_m8_refinement.py [--resume <run_dir>] [--workers 12]

Discretisation error of the trusted solver (production assembly = the defining discrete
problem, see docs/v5_progress.md M8 decision), separated from surrogate/algebraic error:
no neural or POD output is involved. Common population for every study: the first 8 rows of
Q_ref30 (declared); spots 0.8K..1.2K (unchanged window); quantity: terminal put price.

  spatial   N in {64,128,256,512,1024}, domain log K +- 8 (h = 16/N), n_steps = 480 fixed.
            Temporal floor estimated at N=1024 by |V(n=480) - V(n=960)| (from the temporal study).
  temporal  N = 1024 fixed, n_steps in {15,30,60,120,240,480,960}, pure CN (no Rannacher).
            Spatial floor estimated by |V_N=512 - V_N=1024| at n_steps=480 (spatial study).
  domain    h = 0.125 fixed, half-width L in {6,8,10,12} (N = 2L/h = 96,128,160,192),
            n_steps = 240; exterior data = the trusted solver's prescribed Dirichlet put
            asymptotics (K e^{-r tau} sqrt(h) left, 0 right) and its zero boundary forcing.
  cos       converged COS price per (query, spot) -- independent continuum DIAGNOSTIC.

Raw output: refinement/prices.csv (one row per (query, study, level) with the 5 prices),
refinement/cos.csv. scripts/v5_m8_refinement_summary.py rebuilds every table.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mpc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
for sub in ("src", "experiments", "scripts"):
    sys.path.insert(0, str(REPO_ROOT / sub))

N_QUERIES = 8
FRACS = (0.8, 0.9, 1.0, 1.1, 1.2)
SPATIAL = [(N, 480, 8.0) for N in (64, 128, 256, 512, 1024)]
TEMPORAL = [(1024, nst, 8.0) for nst in (15, 30, 60, 120, 240, 480, 960)]
DOMAIN = [(int(round(2 * L / 0.125)), 240, L) for L in (6.0, 8.0, 10.0, 12.0)]


def solve_task(args):
    q, study, N, nst, L = args
    from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
    from certification.free_problem import raw_point_functional
    from pipeline.predict_certify_correct import build_mesh_info

    K = q["K"]
    mesh = Mesh(N=N, p=3, x_min=math.log(K) - L, x_max=math.log(K) + L)
    mi = build_mesh_info(mesh)
    mu = CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
    t0 = time.perf_counter()
    traj = solve_cgmy(mu, EuropeanPutPayoff(K), mesh, TimeGrid(n_steps=nst), mode="toeplitz", store_forcing=False)
    secs = time.perf_counter() - t0
    prices = [float(raw_point_functional(mi, math.log(f * K)) @ traj.coefficients[-1]) for f in FRACS]
    return dict(key=f"{q['query_id']}|{study}|{N}|{nst}|{L}", query_id=q["query_id"], study=study, N=N, h=mi.h,
                n_steps=nst, half_width=L, prices=json.dumps(prices), solve_seconds=secs,
                gmres_iterations=sum(traj.solver_metadata.iterations_per_step))


def cos_task(args):
    q, f = args
    from validation.cgmy_reference_pricing import CGMYMarketParams
    from validation.cos_pricer import converge_price

    mkt = CGMYMarketParams(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
    cv = converge_price(mkt, f * q["K"], q["K"])
    return dict(key=f"{q['query_id']}|{f}", query_id=q["query_id"], spot=f * q["K"], cos_price=cv["price"],
                N_used=cv["N_used"], L_used=cv["L_used"], converged_at=json.dumps(cv["converged_at"]),
                flagged_not_converged=cv["flagged_not_converged"])


def main(argv=None):
    from common import run_manifest as rm
    from common.methods import (METHOD_CONSTRAINED_FREE_CN_V1, METHOD_INDEPENDENT_COS_V1, NUMERICAL_SCOPE,
                                ResultGroupWriter)

    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    pops = rm.declare_populations()
    qs = pops["pricing_q_ref30"]["rows"][:N_QUERIES]
    run_dir = args.resume or rm.create_run_dir(rm.new_run_id("m8_refinement"))
    log = open(run_dir / "run.log", "a")
    log.write(f"[{time.strftime('%H:%M:%S')}] session_start git_head_at_start={rm.git_state()['head']} "
              f"population=pricing_q_ref30 first {N_QUERIES} scope={NUMERICAL_SCOPE}\n")
    log.flush()
    wd = ResultGroupWriter(run_dir / "refinement", METHOD_CONSTRAINED_FREE_CN_V1)
    wc = ResultGroupWriter(run_dir / "cos", METHOD_INDEPENDENT_COS_V1)
    ddone, cdone = wd.existing_keys("prices.csv", "key"), wc.existing_keys("cos.csv", "key")
    tasks = [(q, st, N, nst, L) for q in qs for st, levels in (("spatial", SPATIAL), ("temporal", TEMPORAL),
                                                               ("domain", DOMAIN)) for N, nst, L in levels]
    tasks = [t for t in tasks if f"{t[0]['query_id']}|{t[1]}|{t[2]}|{t[3]}|{t[4]}" not in ddone]
    tasks.sort(key=lambda t: -t[2] * t[3])
    ctasks = [(q, f) for q in qs for f in FRACS if f"{q['query_id']}|{f * q['K']}" not in cdone]
    with mpc.Pool(args.workers) as pool:
        for r in pool.imap_unordered(cos_task, ctasks):
            wc.append_rows("cos.csv", [dict(r, method_id=METHOD_INDEPENDENT_COS_V1)])
        for r in pool.imap_unordered(solve_task, tasks):
            wd.append_rows("prices.csv", [dict(r, method_id=METHOD_CONSTRAINED_FREE_CN_V1)])
            log.write(f"[{time.strftime('%H:%M:%S')}] {r['key']} {r['solve_seconds']:.1f}s\n")
            log.flush()
    if not (run_dir / "manifest.json").exists():
        resolved = rm.resolve_config()
        m = rm.build_manifest(run_dir.name, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                              studies={"assembly_refinement": "refinement completed"}, seeds={},
                              resolved_config=resolved, populations=pops,
                              population_used=[f"pricing_q_ref30 first {N_QUERIES}"])
        m.update(levels={"spatial": SPATIAL, "temporal": TEMPORAL, "domain": DOMAIN},
                 other_method_ids_in_run={"cos/*": METHOD_INDEPENDENT_COS_V1})
        rm.write_json(run_dir / "manifest.json", m)
    log.write(f"[{time.strftime('%H:%M:%S')}] done\n")


if __name__ == "__main__":
    main()
