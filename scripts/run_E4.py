#!/usr/bin/env python3
"""Phase 3, E4 (plan 3.5): multi-query demonstration at the production configuration.

10^4 Sobol queries (seed 4) in the box, eps = eps_N.  Methods: (i) direct, uncertified; (ii) certified POD-Galerkin
r = 80 (three-stage, correction fallback); (iii) the same with r = 64; (iv) certified space-time POD-NN with correction
fallback (the E2 pipeline, reusable setup).  One pass, method order shuffled per query (reproducibly), one performance
core, idle machine; checkpointed every 500 queries (results/E4_progress.json) so the run can resume.
Offline cost C0 from the recorded logs (reference generation, POD, EIM, cell beta-bar, tensors, training); total time
C0 + C1 Q per method, break-even vs (i) at Q* = C0 / (C1_direct - C1_method).  Writes results/E4.json, F8.
"""
from __future__ import annotations

import argparse, json, random, sys, time, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from cgmy_cert.params import sobol_queries, Query  # noqa: E402
from cgmy_cert.timing import pin, check_single_thread, hardware, loadavg  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

RES = ROOT / "results"; FIG = ROOT / "figures_v9"
OUT = RES / "E4.json"; PROG = RES / "E4_progress.json"
METHODS = ("direct", "podg80", "podg64", "podnn_st")


def do_run(n=10_000, chunk=500):
    from run_E2 import Pipeline, EPS_N
    check_single_thread(); pin()
    prog = json.loads(PROG.read_text()) if PROG.exists() else dict(rows=[], loadavg_start=loadavg(), started=time.time())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qs = sobol_queries(n, seed=4)
    pl = Pipeline()
    pl.direct(Query(0.5, 3.0, 5.0, 1.5, 0.05, T=0.5))                     # warm-up outside the population
    for m in METHODS[1:]:
        pl.certified(Query(0.5, 3.0, 5.0, 1.5, 0.05, T=0.5), m, EPS_N)
    rng = random.Random(4)
    done = len(prog["rows"])
    for _ in range(done):                                                 # keep the shuffle reproducible on resume
        rng.shuffle(list(METHODS))
    for i in range(done, n):
        q = qs[i]
        order = list(METHODS); rng.shuffle(order)
        row = dict(query_id=q.query_id)
        for m in order:
            t0 = time.perf_counter()
            r = pl.direct(q) if m == "direct" else pl.certified(q, m, EPS_N)
            row[m] = dict(time_s=time.perf_counter() - t0, status=r.get("status"), corrected=r.get("corrected", False))
        prog["rows"].append(row)
        if (i + 1) % chunk == 0 or i + 1 == n:
            prog["loadavg_last"] = loadavg()
            PROG.write_text(json.dumps(prog))
            print(f"{i + 1} queries; median times " + ", ".join(
                f"{m} {np.median([x[m]['time_s'] for x in prog['rows']]) * 1e3:.1f} ms" for m in METHODS), flush=True)
    prog["hardware"] = hardware()
    PROG.write_text(json.dumps(prog))


def offline_costs():
    """C0 components from the recorded logs (wall seconds, with the parallelism they ran at)."""
    man = json.loads((RES / "MANIFEST.json").read_text())["files"]
    n1 = json.loads((RES / "N1024.json").read_text())
    eim = json.loads((RES / "eim.json").read_text())
    cm = json.loads((RES / "cost_model.json").read_text())
    tl = json.loads((RES / "training_log.json").read_text())
    e2 = json.loads((RES / "E2.json").read_text())
    refs = man["references_N1024_NT240_seed0_4000.npz"]
    t_direct_1t = float(np.median([r["time_s"] for r in e2["bench"]["per_setup"]["reusable"]["rows"]["direct"]]))
    ref_cost = dict(recorded_wall_s=refs.get("wall_s"), recorded_parallelism=refs["provenance"].get("nproc", 22),
                    single_thread_estimate_800=800 * t_direct_1t, single_thread_estimate_4000=4000 * t_direct_1t,
                    note="single-thread estimate = n x median direct solve time (E2, reusable setup)")
    pod = {k: v["wall_s"] for k, v in n1["podext"].items()}
    eim_cost = dict(build_wall_s=eim["build"]["1024"]["wall_s"], nproc=eim["build"]["1024"].get("nproc"),
                    select_wall_s=eim.get("select", {}).get("1024", {}).get("wall_s"))
    sd = cm["median_times_full_order"]["1024"]["spectral_data"]
    n_pts = sum(len(v["corners"]) * v["nY"] for v in n1["cells"]["betabar"].values())
    cells = dict(spectral_data_evaluations=n_pts, single_thread_estimate_s=n_pts * sd,
                 note="corner-rule beta-bar: n_corners x 33 Y points per cell x spectral_data (cost model constant); not "
                      "separately logged")
    tens = {k: v["offline_s"] for k, v in n1["tensors_ext"].items() if "_r64_n800" in k or "_r80_n800" in k}
    tens.update({k: v["offline_s"] for k, v in n1["tensors"].items() if "_r48_n4000" in k})
    best = e2["bench"]["nn_candidate"]
    runs = {r["name"]: r["wall_s"] for r in tl["runs"]}
    training = dict(best_run=best, best_run_wall_s=runs.get(best), whole_sweep_wall_s_sum=float(sum(runs.values())),
                    note="wall time of each run with 3 threads, 8 runs in parallel on 24 threads (sweep_N1024.py); stated as "
                         "recorded, not single-thread")
    C0 = dict(podg=dict(references=ref_cost["single_thread_estimate_800"], pod=pod["800"], eim=eim_cost["build_wall_s"],
                        cells=cells["single_thread_estimate_s"], tensors=sum(v for k, v in tens.items() if "n800" in k)),
              podnn_st=dict(references=ref_cost["single_thread_estimate_4000"], pod=pod["4000"], eim=eim_cost["build_wall_s"],
                            cells=cells["single_thread_estimate_s"], tensors=sum(v for k, v in tens.items() if "n4000" in k),
                            training=training["best_run_wall_s"]))
    return dict(references=ref_cost, pod_wall_s=pod, eim=eim_cost, cells=cells, tensors_offline_s=tens, training=training,
                C0_s={k: dict(v, total=float(sum(x for x in v.values() if x))) for k, v in C0.items()})


def do_summary():
    prog = json.loads(PROG.read_text())
    rows = prog["rows"]
    n = len(rows)
    off = offline_costs()
    meth = {}
    for m in METHODS:
        t = np.array([r[m]["time_s"] for r in rows])
        s = dict(total_online_s=float(t.sum()), C1_mean_s=float(t.mean()), median_s=float(np.median(t)))
        if m != "direct":
            st = [r[m]["status"] for r in rows]
            s["status_fractions"] = {k: st.count(k) / n for k in sorted(set(st))}
            s["fraction_corrected"] = float(np.mean([r[m]["corrected"] for r in rows]))
        meth[m] = s
    c1d = meth["direct"]["C1_mean_s"]
    for m in METHODS[1:]:
        C0 = off["C0_s"]["podnn_st" if m == "podnn_st" else "podg"]["total"]
        meth[m]["C0_s"] = C0
        gain = c1d - meth[m]["C1_mean_s"]
        meth[m]["break_even_queries"] = float(C0 / gain) if gain > 0 else None
        meth[m]["model"] = f"T(Q) = {C0:.0f} s + {meth[m]['C1_mean_s'] * 1e3:.2f} ms * Q"
    meth["direct"]["model"] = f"T(Q) = {c1d * 1e3:.2f} ms * Q"
    res = dict(meta=dict(description="E4 (plan 3.5): 10^4-query demonstration at eps_N", n_queries=n, population="sobol_queries(10000, seed=4)",
                         eps=json.loads((RES / "PRODUCTION_CONFIG.json").read_text())["eps_N"], loadavg_start=prog["loadavg_start"],
                         loadavg_last=prog.get("loadavg_last"), hardware=prog.get("hardware"), provenance=provenance(),
                         protocol="one pass, method order shuffled per query, one performance core, reusable setup"),
               methods=meth, offline=off)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({m: {k: v for k, v in s.items() if k != "status_fractions"} for m, s in meth.items()}, indent=1), flush=True)
    # F8
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    labels = dict(direct="direct (uncertified)", podg80="certified POD-Galerkin r = 80", podg64="certified POD-Galerkin r = 64",
                  podnn_st="certified space-time POD-NN (+ correction)")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    Qx = np.arange(1, n + 1)
    for c, m in zip(PAL, METHODS):
        cum = np.cumsum([r[m]["time_s"] for r in rows]) + (meth[m].get("C0_s") or 0.0)
        ax.loglog(Qx, cum / 3600, color=c, lw=2, label=labels[m])
        be = meth[m].get("break_even_queries")
        if be and be <= n:
            ax.plot(be, (meth[m]["C0_s"] + meth[m]["C1_mean_s"] * be) / 3600, "o", color=c, ms=6)
    ax.set_xlabel("number of queries"); ax.set_ylabel("cumulative wall time incl. offline C0 (h)")
    ax.set_title("E4: amortisation at N = 1024 (one thread; dots: break-even with direct)", fontsize=9)
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.25, lw=0.5, which="both")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "F8_amortisation.png", dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["run", "summary"])
    a = ap.parse_args()
    if "run" in a.steps:
        do_run()
    if "summary" in a.steps:
        do_summary()


if __name__ == "__main__":
    main()
