#!/usr/bin/env python3
"""Follow-up 4, item 3: the trimmed POD-NN sweep at the production resolution, end to end.

  phase A  800 / 4000 trajectories x lambda_R in {0, 1e-2, 1e-1, 1}, seed 0 (8 runs in parallel, 3 threads each)
  select   evaluate all checkpoints (scripts/eval_podnn.py); best (size, lambda_R) = min validation median
           full-order exact bound (exact_AI)
  phase B  best setting, seeds 1 and 2; space-time variant (r x r_t) at the best setting, seeds 0, 1, 2 (5 runs in
           parallel); the space-time runs are capped at one day of wall time
  final    evaluate everything again (results/E2_4_podnn_N1024.json, results/training_log.json)
State in results/sweep_N1024_state.json.  Usage: python3 scripts/sweep_N1024.py
"""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; LOGS = RES / "training_runs" / "logs"
STATE = RES / "sweep_N1024_state.json"
LAMS = (0.0, 0.01, 0.1, 1.0)
CAP_S = 24 * 3600


def log(state, msg):
    state.setdefault("events", []).append(dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), msg=msg))
    STATE.write_text(json.dumps(state, indent=1))
    print(msg, flush=True)


def launch(n, lam, seed, spacetime=False, threads=3):
    env = dict(os.environ, CGMY_N="1024", OMP_NUM_THREADS=str(threads))
    tag = f"n{n}_lam{lam:g}_s{seed}" + ("_st" if spacetime else "")
    cmd = [sys.executable, str(ROOT / "scripts" / "train_podnn.py"), "run", "--n", str(n), "--lam", f"{lam:g}",
           "--seed", str(seed), "--threads", str(threads)] + (["--spacetime"] if spacetime else [])
    f = open(LOGS / f"{tag}.log", "w")
    return tag, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)


def wait(procs, cap=None, state=None):
    t0 = time.time()
    while any(p.poll() is None for _, p in procs):
        if cap is not None and time.time() - t0 > cap:
            for tag, p in procs:
                if p.poll() is None:
                    p.terminate()
                    log(state, f"cap reached: terminated {tag}")
            break
        time.sleep(60)
    return {tag: p.returncode for tag, p in procs}


def evaluate():
    env = dict(os.environ, OMP_NUM_THREADS="1")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_podnn.py")], cwd=ROOT, env=env, check=True)
    return json.loads((RES / "E2_4_podnn_N1024.json").read_text())


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not state.get("phaseA_done"):
        log(state, "phase A: 8 runs")
        procs = [launch(n, lam, 0) for n in (800, 4000) for lam in LAMS]
        state["phaseA_returncodes"] = wait(procs)
        state["phaseA_done"] = True
        log(state, f"phase A done: {state['phaseA_returncodes']}")
    if "best" not in state:
        ev = evaluate()
        cands = {k: v["summary"] for k, v in ev["results"].items() if k.startswith("podnn")}
        best = min(cands, key=lambda k: cands[k]["exact_AI"]["median"])
        run = next(r for r in json.loads((RES / "training_log.json").read_text())["runs"] if r["name"] == best)
        state["best"] = dict(name=best, n_train=run["n_train"], lambda_R=run["lambda_R"],
                             median_exact_AI=cands[best]["exact_AI"]["median"],
                             all_medians_exact_AI={k: v["exact_AI"]["median"] for k, v in cands.items()})
        log(state, f"selected {state['best']}")
    if not state.get("phaseB_done"):
        b = state["best"]
        procs = [launch(b["n_train"], b["lambda_R"], s) for s in (1, 2)]
        st = [launch(b["n_train"], b["lambda_R"], s, spacetime=True) for s in (0, 1, 2)]
        log(state, "phase B: seeds 1, 2 and space-time seeds 0-2")
        rc_st = wait(st, cap=CAP_S, state=state)
        rc = wait(procs)
        state["phaseB_returncodes"] = dict(rc, **rc_st)
        state["phaseB_done"] = True
        log(state, f"phase B done: {state['phaseB_returncodes']}")
    evaluate()
    state["final_evaluation"] = "results/E2_4_podnn_N1024.json"
    log(state, "final evaluation written")


if __name__ == "__main__":
    main()
