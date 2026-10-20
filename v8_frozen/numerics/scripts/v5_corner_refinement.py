"""Corner projection/discretisation diagnostic (requested after G4; roadmap §8.2-style
study, DIAGNOSTIC ONLY -- not a certified quantity).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:experiments:. \
        python3 scripts/v5_corner_refinement.py [--resume <run_dir>]

Queries (declared): the two G4 'difficult' exact vertices Q_audit_0401, Q_audit_0397
(C=0.05, Y=1.1, T=0.1, r=0.01) and two typical contrasts Q_bench120_0000 (T=0.12),
Q_bench120_0001. Spots 0.8K..1.2K (unchanged window). Domain fixed (log K +- 8), so
h = 16/N; N in {64,128,256,512,1024}; n_steps = 240 at every N (time error held
small and fixed; the production n_steps=60 is added at N=64 for reference).
Recorded per (query, N, n_steps): trusted discrete price V_h at tau=0 (L2 payoff
projection) and tau=T, payoff and converged COS (validation.cos_pricer.converge_price)
at each spot; discrete-minus-COS discrepancy (independent continuum diagnostic,
includes discretisation AND COS error; COS convergence flags recorded); paired
refinement differences V_N - V_2N.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
for sub in ("src", "experiments", "scripts"):
    sys.path.insert(0, str(REPO_ROOT / sub))

from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy  # noqa: E402
from certification.free_problem import raw_point_functional  # noqa: E402
from common import run_manifest as rm  # noqa: E402
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1, METHOD_INDEPENDENT_COS_V1, NUMERICAL_SCOPE, ResultGroupWriter  # noqa: E402
from data.sampling import load_mesh_and_time_grid  # noqa: E402
from pipeline.predict_certify_correct import build_mesh_info  # noqa: E402
from validation.cgmy_reference_pricing import CGMYMarketParams  # noqa: E402
from validation.cos_pricer import converge_price  # noqa: E402

QUERIES = ("Q_audit_0401", "Q_audit_0397", "Q_bench120_0000", "Q_bench120_0001")
NS = (64, 128, 256, 512, 1024)
N_STEPS = 240
FRACS = (0.8, 0.9, 1.0, 1.1, 1.2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args(argv)
    pops = rm.declare_populations()
    allq = {r["query_id"]: r for p in ("audit459", "bench120") for r in pops[p]["rows"]}
    base, _ = load_mesh_and_time_grid(rm.BASE_CONFIG)
    run_dir = args.resume or rm.create_run_dir(rm.new_run_id("corner_refinement"))
    log = open(run_dir / "run.log", "a")

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"session_start run_dir={run_dir.name} git_head_at_start={rm.git_state()['head']} DIAGNOSTIC scope={NUMERICAL_SCOPE}")
    wd = ResultGroupWriter(run_dir / "discrete", METHOD_CONSTRAINED_FREE_CN_V1)
    wc = ResultGroupWriter(run_dir / "cos", METHOD_INDEPENDENT_COS_V1)
    ddone = wd.existing_keys("prices.csv", "key")
    cdone = wc.existing_keys("prices.csv", "key")
    for qid in QUERIES:
        q = allq[qid]
        mu = CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
        mkt = CGMYMarketParams(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
        for f in FRACS:
            key = f"{qid}|{f}"
            if key in cdone:
                continue
            S = f * q["K"]
            t0 = time.perf_counter()
            cv = converge_price(mkt, S, q["K"])
            wc.append_rows("prices.csv", [dict(key=key, method_id=METHOD_INDEPENDENT_COS_V1, query_id=qid, spot=S,
                                               cos_price=cv["price"], N_used=cv["N_used"], L_used=cv["L_used"],
                                               converged_at=json.dumps(cv["converged_at"]),
                                               flagged_not_converged=cv["flagged_not_converged"],
                                               payoff=max(q["K"] - S, 0.0), seconds=time.perf_counter() - t0)])
        for N, nst in [(n, N_STEPS) for n in NS] + [(64, 60)]:
            key = f"{qid}|{N}|{nst}"
            if key in ddone:
                continue
            mesh = Mesh(N=N, p=base.p, x_min=base.x_min, x_max=base.x_max)
            mi = build_mesh_info(mesh)
            t0 = time.perf_counter()
            traj = solve_cgmy(mu, EuropeanPutPayoff(q["K"]), mesh, TimeGrid(n_steps=nst), mode="toeplitz",
                              store_forcing=False)
            secs = time.perf_counter() - t0
            phis = [raw_point_functional(mi, math.log(f * q["K"])) for f in FRACS]
            wd.append_rows("prices.csv", [dict(
                key=key, method_id=METHOD_CONSTRAINED_FREE_CN_V1, query_id=qid, N=N, h=mi.h, n_steps=nst,
                price_tau0=json.dumps([float(p @ traj.coefficients[0]) for p in phis]),
                price_tauT=json.dumps([float(p @ traj.coefficients[-1]) for p in phis]),
                spots=json.dumps([f * q["K"] for f in FRACS]), solve_seconds=secs,
                solver="trusted solve_cgmy toeplitz (diagnostic, not certified here)")])
            say(f"{qid} N={N} n_steps={nst} ({secs:.1f}s)")
    if not (run_dir / "manifest.json").exists():
        resolved = rm.resolve_config()
        m = rm.build_manifest(run_dir.name, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                              studies={"assembly_refinement": "partial: corner refinement diagnostic only"},
                              seeds=resolved["workflow_config"]["seeds"], resolved_config=resolved, populations=pops,
                              population_used=list(QUERIES))
        m["other_method_ids_in_run"] = {"cos/*": METHOD_INDEPENDENT_COS_V1}
        m["role"] = "diagnostic: discrete-vs-continuum discrepancy at box corners; NOT a certified quantity"
        rm.write_json(run_dir / "manifest.json", m)
    say("done")


if __name__ == "__main__":
    main()
