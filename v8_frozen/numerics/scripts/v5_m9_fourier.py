"""M9 §9.1: same-query Fourier reruns (COS and Carr-Madan), DIAGNOSTIC continuum prices.

    OMP_NUM_THREADS=1 PYTHONPATH=src:experiments:. taskset -c 11-22 python3 scripts/v5_m9_fourier.py \
        [--resume <run_dir>] [--workers 12]

Populations, reported separately (never pooled):
  fourier_old27   the 27 coordinates of the historical cross-check (validation/reference_validation.csv,
                  incl. box_12), S0 = K exactly as before.
  pricing_q_ref30 all 30 Q_ref30 coordinates at spots 0.8K..1.2K.
Per (query, spot):
  cos_hist     converge_price (historical sweep N 32..1024, L 6..20): price, N/L used, flag
  cos_ext      COS at N=8192, L=30 (resolution and range increased)
  cm_hist      Carr-Madan, dps 40, cutoff cap 400 (historical): price, alpha, cutoff, cap hit
  cm_ext       Carr-Madan, dps 40, cutoff cap 51200, panelled quadrature (width 32): price, cutoff, cap hit
  cm_hp        (worst 3 queries of each population by |cos_ext - cm_ext|, plus box_12) Carr-Madan
               dps 60, cap 51200 -- higher-precision cross-check
  martingale   |E[e^{X_T}] e^{-rT} - 1| in double and mpmath (dps 50)
Agreement between successive settings and between methods is a convergence DIAGNOSTIC,
not an error enclosure.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mpc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
for sub in ("src", "experiments"):
    sys.path.insert(0, str(REPO_ROOT / sub))

FRACS = (0.8, 0.9, 1.0, 1.1, 1.2)


def task(args):
    pop, q, S0, hp = args
    from validation.cgmy_reference_pricing import CGMYMarketParams, martingale_check_mpmath, martingale_check_numpy
    from validation.cos_pricer import converge_price, cos_put_price, truncation_range
    from validation.independent_fourier_integration import carr_madan_put_price

    mkt = CGMYMarketParams(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
    K = q["K"]
    t0 = time.perf_counter()
    ch = converge_price(mkt, S0, K)
    ce = cos_put_price(mkt, S0, K, N=8192, trunc_range=truncation_range(mkt, L=30.0), L=30.0)
    mh = carr_madan_put_price(mkt, S0, K, dps=40, max_v=400.0)
    me = carr_madan_put_price(mkt, S0, K, dps=40, max_v=51200.0, panel_width=32.0)
    row = dict(key=f"{pop}|{q['query_id']}|{S0}", population=pop, query_id=q["query_id"], spot=S0, K=K,
               cos_hist_price=ch["price"], cos_hist_N=ch["N_used"], cos_hist_L=ch["L_used"],
               cos_hist_flag_not_converged=ch["flagged_not_converged"],
               cos_ext_price=ce["price"], cos_ext_N=ce["N"], cos_ext_L=ce["L"], cos_ext_range=f"[{ce['a']}, {ce['b']}]",
               cm_alpha=mh["alpha"], cm_hist_price=mh["price"], cm_hist_cutoff=mh["v_cutoff"],
               cm_hist_cap_hit=mh["cutoff_cap_hit"], cm_ext_price=me["price"], cm_ext_cutoff=me["v_cutoff"],
               cm_ext_cap_hit=me["cutoff_cap_hit"],
               rel_cos_hist_vs_cm_hist=abs(ch["price"] - mh["price"]) / max(abs(ch["price"]), 1.0),
               rel_cos_ext_vs_cm_ext=abs(ce["price"] - me["price"]) / max(abs(ce["price"]), 1.0),
               rel_cos_hist_vs_ext=abs(ch["price"] - ce["price"]) / max(abs(ce["price"]), 1.0),
               rel_cm_hist_vs_ext=abs(mh["price"] - me["price"]) / max(abs(me["price"]), 1.0),
               martingale_numpy=martingale_check_numpy(mkt), martingale_mpmath=martingale_check_mpmath(mkt, dps=50),
               cm_hp_price="", cm_hp_cutoff="", cm_hp_cap_hit="", rel_cm_hp_vs_cm_ext="", rel_cm_hp_vs_cos_ext="")
    if hp:
        mp60 = carr_madan_put_price(mkt, S0, K, dps=60, max_v=51200.0, panel_width=32.0)
        row.update(cm_hp_price=mp60["price"], cm_hp_cutoff=mp60["v_cutoff"], cm_hp_cap_hit=mp60["cutoff_cap_hit"],
                   rel_cm_hp_vs_cm_ext=abs(mp60["price"] - me["price"]) / max(abs(me["price"]), 1.0),
                   rel_cm_hp_vs_cos_ext=abs(mp60["price"] - ce["price"]) / max(abs(ce["price"]), 1.0))
    row["seconds"] = time.perf_counter() - t0
    return row


def main(argv=None):
    from common import run_manifest as rm
    from common.methods import METHOD_INDEPENDENT_COS_V1, NUMERICAL_SCOPE, ResultGroupWriter

    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    pops = rm.declare_populations()
    run_dir = args.resume or rm.create_run_dir(rm.new_run_id("m9_fourier"))
    log = open(run_dir / "run.log", "a")
    log.write(f"[{time.strftime('%H:%M:%S')}] session_start git_head_at_start={rm.git_state()['head']} DIAGNOSTIC "
              f"scope={NUMERICAL_SCOPE}\n")
    log.flush()
    w = ResultGroupWriter(run_dir / "fourier", METHOD_INDEPENDENT_COS_V1)
    done = w.existing_keys("fourier.csv", "key")
    tasks = [("fourier_old27", q, q["K"], q["query_id"] == "box_12") for q in pops["fourier_old27"]["rows"]]
    tasks += [("pricing_q_ref30", q, f * q["K"], False) for q in pops["pricing_q_ref30"]["rows"] for f in FRACS]
    tasks = [t for t in tasks if f"{t[0]}|{t[1]['query_id']}|{t[2]}" not in done]
    with mpc.Pool(args.workers) as pool:
        for r in pool.imap_unordered(task, tasks):
            w.append_rows("fourier.csv", [dict(r, method_id=METHOD_INDEPENDENT_COS_V1)])
            log.write(f"[{time.strftime('%H:%M:%S')}] {r['key']} {r['seconds']:.0f}s\n")
            log.flush()
    # higher-precision cross-check of the 3 worst (cos_ext vs cm_ext) per population
    import csv
    rows = list(csv.DictReader(open(run_dir / "fourier/fourier.csv")))
    hp_done = w.existing_keys("fourier_hp.csv", "key")
    extra = []
    for pop in ("fourier_old27", "pricing_q_ref30"):
        rs = sorted([r for r in rows if r["population"] == pop], key=lambda r: -float(r["rel_cos_ext_vs_cm_ext"]))[:3]
        allq = {q["query_id"]: q for q in pops[pop]["rows"]}
        extra += [(pop, allq[r["query_id"]], float(r["spot"]), True) for r in rs
                  if f"{pop}|{r['query_id']}|{float(r['spot'])}" not in hp_done and not (r["query_id"] == "box_12")]
    with mpc.Pool(args.workers) as pool:
        for r in pool.imap_unordered(task, extra):
            w.append_rows("fourier_hp.csv", [dict(r, method_id=METHOD_INDEPENDENT_COS_V1)])
    if not (run_dir / "manifest.json").exists():
        resolved = rm.resolve_config()
        m = rm.build_manifest(run_dir.name, method_id=METHOD_INDEPENDENT_COS_V1,
                              studies={"fourier_rerun": "completed"}, seeds={}, resolved_config=resolved,
                              populations=pops, population_used=["fourier_old27 (S0=K)", "pricing_q_ref30 (5 spots)"])
        m["role"] = "continuum-price DIAGNOSTIC (COS / Carr-Madan); not certified"
        rm.write_json(run_dir / "manifest.json", m)
    log.write(f"[{time.strftime('%H:%M:%S')}] done\n")


if __name__ == "__main__":
    main()
