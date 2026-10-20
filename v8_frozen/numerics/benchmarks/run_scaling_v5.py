"""Resolution scaling study, version scaling_v5.0 (roadmap §7.2, M7).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:experiments \
        python3 benchmarks/run_scaling_v5.py [--resume <run_dir>]

Frozen policy: configs/v5_scaling.yaml. Solver/certificate scaling only (no
neural/POD at N != 64: no transfer strategy; recorded in not_applicable.csv).
Reuses the gate_d_v5 harness's deterministic trajectory methods unchanged, in
cold mode, with a per-N context; every row carries N, n_dof, n_free, phase
timings, GMRES/PCG work, status, target status vs that N's own stored reference.
Memory: one extra untimed run per (N, query, method) under tracemalloc.
Resumable: completed (N, method, query, rep) keys are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
for sub in ("src", "experiments", "scripts", "benchmarks"):
    sys.path.insert(0, str(REPO_ROOT / sub))

import run_gate_d_benchmark_v5 as h  # noqa: E402
from cgmy.reference_solver import Mesh  # noqa: E402
from common import run_manifest as rm  # noqa: E402
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1, NUMERICAL_SCOPE, ResultGroupWriter  # noqa: E402
from data.sampling import load_mesh_and_time_grid  # noqa: E402
from pipeline import free_workflow as fw  # noqa: E402
from pipeline.legacy_guard import forbid_legacy_calls, summarize  # noqa: E402
from pipeline.predict_certify_correct import build_mesh_info  # noqa: E402

SCALING_CONFIG = REPO_ROOT / "configs" / "v5_scaling.yaml"


def ctx_for_N(N: int, bench_cfg: dict, wf_cfg: dict, max_n_dof: int) -> h.Ctx:
    base_mesh, tg = load_mesh_and_time_grid(rm.BASE_CONFIG)
    mesh = Mesh(N=N, p=base_mesh.p, x_min=base_mesh.x_min, x_max=base_mesh.x_max)
    mi = build_mesh_info(mesh)
    return h.Ctx(cfg=bench_cfg, wf_cfg=wf_cfg, policy=fw.FreeWorkflowPolicy.from_config(wf_cfg), mesh=mesh,
                 mesh_info=mi, n_steps=tg.n_steps, model=None, stats=None, pod={}, basis_ids={},
                 caches=h.Caches(mesh, mi), max_n_dof=max_n_dof)


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args(argv)
    if any(os.environ.get(v) != "1" for v in rm.THREAD_ENV_VARS):
        raise SystemExit("single-threaded settings required")
    scfg = yaml.safe_load(SCALING_CONFIG.read_text())
    bench_cfg = yaml.safe_load(h.BENCH_CONFIG.read_text())
    resolved = rm.resolve_config()
    wf_cfg = resolved["workflow_config"]
    pops = rm.declare_populations()
    queries = pops[scfg["population"]]["rows"]
    plan = [(N, queries) for N in scfg["resolutions"]]
    plan.append((scfg["extended"]["N"], [q for q in queries if q["query_id"] in scfg["extended"]["queries"]]))

    if args.resume:
        run_dir = args.resume
        session = sum("session_start" in l for l in (run_dir / "run.log").read_text().splitlines())
    else:
        run_dir = rm.create_run_dir(rm.new_run_id("scaling_v5"))
        session = 0
    log = open(run_dir / "run.log", "a")

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"session_start={session} pid={os.getpid()} affinity={sorted(os.sched_getaffinity(0))} "
        f"git_head_at_start={rm.git_state()['head']} scope={NUMERICAL_SCOPE}")
    w = ResultGroupWriter(run_dir / "scaling", METHOD_CONSTRAINED_FREE_CN_V1)
    na = [dict(key=f"{N}|{m}", method_id=METHOD_CONSTRAINED_FREE_CN_V1, N=N, solver_method=m,
               status="not_applicable_no_resolution_transfer_strategy")
          for N, _ in plan if N != 64 for m in scfg["not_applicable_at_N_ne_64"]]
    if not w.existing_keys("not_applicable.csv", "key"):
        w.append_rows("not_applicable.csv", na)
    done = w.existing_keys("raw_timings.csv", "key")
    mdone = w.existing_keys("memory.csv", "key")
    with forbid_legacy_calls() as guard_log:
        for N, qs in plan:
            ctx = ctx_for_N(N, bench_cfg, wf_cfg, scfg["max_n_dof_dense_guard"])
            ctx.caches.session = session
            refs = {}
            for q in qs:
                t0 = time.perf_counter()
                refs[q["query_id"]] = h.stored_reference(ctx, q)
                if f"{N}|{q['query_id']}" not in w.existing_keys("stored_reference.csv", "key"):
                    r = refs[q["query_id"]]
                    w.append_rows("stored_reference.csv", [dict(
                        key=f"{N}|{q['query_id']}", method_id=METHOD_CONSTRAINED_FREE_CN_V1, N=N, query_id=q["query_id"],
                        stored_reference_eta_aug=r["eta"], verification_cost_s=r["seconds"],
                        price_at_spots=json.dumps(r["prices"]), betas=json.dumps(r["betas"]))])
            say(f"N={N}: {len(qs)} stored references")
            wq = dict(bench_cfg["warmup_query"], query_id="WARMUP", source="warmup")
            wref = h.stored_reference(ctx, wq)
            for m in scfg["methods"]:
                h.run_trajectory_method(ctx, m, wq, "cold", -1, None, wref)
            for rep in range(scfg["repeats"]["timed"]):
                for qi, q in enumerate(qs):
                    order = list(scfg["methods"])
                    np.random.default_rng([bench_cfg["seed"], N, rep, qi]).shuffle(order)
                    for m in order:
                        key = f"{N}|{m}|{q['query_id']}|{rep}"
                        if key in done:
                            continue
                        tr, _, ta, _, tt, _ = h.run_trajectory_method(ctx, m, q, "cold", rep, None, refs[q["query_id"]])
                        extra = dict(N=N, n_dof=ctx.mesh_info.n_dof, n_free=ctx.mesh_info.n_dof - 2 * (ctx.mesh_info.p - 1))
                        w.append_rows("outcomes.csv", [dict(dict(key=key, **extra), **tr)])
                        w.append_rows("accuracy.csv", [dict(dict(key=key, **extra), **ta)])
                        w.append_rows("raw_timings.csv", [dict(dict(key=key, **extra), **tt)])
                say(f"N={N} repeat {rep} done")
            for q in qs:
                for m in scfg["methods"]:
                    key = f"{N}|{m}|{q['query_id']}|mem"
                    if key in mdone:
                        continue
                    tracemalloc.start()
                    tr = h.run_trajectory_method(ctx, m, q, "cold", -2, None, refs[q["query_id"]])[0]
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    w.append_rows("memory.csv", [dict(key=key, method_id=METHOD_CONSTRAINED_FREE_CN_V1, N=N,
                                                      query_id=q["query_id"], solver_method=m, peak_traced_bytes=peak,
                                                      final_status=tr["final_status"])])
            say(f"N={N} memory pass done")
    g = summarize(guard_log)
    rm.write_json(run_dir / f"legacy_guard_session{session}.json", {"blocked": g["blocked"], "calls": g["calls"]})
    if not (run_dir / "manifest.json").exists():
        m = rm.build_manifest(run_dir.name, method_id=METHOD_CONSTRAINED_FREE_CN_V1, studies={"scaling": "completed"},
                              seeds={"benchmark": bench_cfg["seed"], **wf_cfg["seeds"]}, resolved_config=resolved,
                              populations=pops, population_used=[f"N={N}: {len(qs)} queries" for N, qs in plan],
                              extra_inputs={"scaling_config": "configs/v5_scaling.yaml",
                                            "benchmark_config": "configs/v5_benchmark.yaml"})
        m.update(harness_version=scfg["harness_version"], scaling_config=scfg)
        rm.write_json(run_dir / "manifest.json", m)
        rm.write_json(run_dir / "resolved_config.json", resolved)
    say("done")
    return run_dir


if __name__ == "__main__":
    main()
