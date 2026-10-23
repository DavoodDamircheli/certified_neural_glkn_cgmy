#!/usr/bin/env python3
"""Phase 4, item 1.3-1.4: clean-clone reproducibility check -> results/ARCHIVE_CHECK.json.

  1. git clone HEAD into a temporary directory; fresh venv; pip install -r requirements.txt (CPU torch index)
  2. bash build.sh; compare every regenerated table with the committed one (byte-identical) and every PNG with the
     committed PNG (identical pixel arrays via numpy)
  3. python3 scripts/run_E0.py --nproc 1 end to end, then CGMY_FULL_TESTS=0 python3 -m pytest tests -q; wall times
Usage: python3 scripts/archive_check.py [--workdir DIR] [--skip-e0]
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ARCHIVE_CHECK.json"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def run(cmd, cwd, env=None, log=None):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, shell=True, capture_output=True, text=True)
    if log is not None:
        Path(log).write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    return dict(cmd=cmd, returncode=p.returncode, wall_s=time.time() - t0, tail=(p.stdout + p.stderr)[-2000:])


def png_equal(a, b):
    import matplotlib.image as im
    x, y = im.imread(a), im.imread(b)
    return bool(x.shape == y.shape and np.array_equal(x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--skip-e0", action="store_true")
    a = ap.parse_args()
    work = Path(a.workdir or tempfile.mkdtemp(prefix="cgmy_clean_clone_"))
    clone = work / "repo"; venv = work / "venv"; logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    res = dict(workdir=str(work), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), steps={})
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    res["steps"]["clone"] = run(f"git clone --quiet {ROOT} {clone} && git -C {clone} checkout --quiet {head}", work)
    res["clone_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True).stdout.strip()
    res["steps"]["venv"] = run(f"python3 -m venv {venv}", work)
    py = venv / "bin" / "python3"
    res["steps"]["pip_install"] = run(f"{py} -m pip install --quiet --upgrade pip && {py} -m pip install --quiet -r requirements.txt "
                                      f"--extra-index-url {TORCH_INDEX}", clone, log=logs / "pip.log")
    res["installed"] = run(f"{py} -m pip freeze", clone)["tail"]
    env = dict(os.environ, PATH=f"{venv / 'bin'}:{os.environ['PATH']}", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    # committed outputs (before build.sh overwrites them)
    committed = {}
    for d in ("tables_v9", "figures_v9"):
        for p in sorted((clone / d).iterdir()):
            committed[f"{d}/{p.name}"] = p.read_bytes()
    snap = work / "committed"; snap.mkdir(exist_ok=True)
    for k, v in committed.items():
        (snap / k.replace("/", "__")).write_bytes(v)
    res["steps"]["build"] = run("bash build.sh", clone, env=env, log=logs / "build.log")
    cmp_ = {}
    for k in committed:
        new = clone / k
        if k.endswith(".tex"):
            cmp_[k] = dict(kind="table", identical=new.read_bytes() == committed[k])
        elif k.endswith(".png"):
            cmp_[k] = dict(kind="figure", pixel_identical=png_equal(str(new), str(snap / k.replace("/", "__"))))
    res["comparison"] = cmp_
    res["all_tables_identical"] = all(v["identical"] for v in cmp_.values() if v["kind"] == "table")
    res["all_figures_pixel_identical"] = all(v["pixel_identical"] for v in cmp_.values() if v["kind"] == "figure")
    res["n_tables"] = sum(v["kind"] == "table" for v in cmp_.values())
    res["n_figures"] = sum(v["kind"] == "figure" for v in cmp_.values())
    OUT.write_text(json.dumps(res, indent=1))
    print("build:", res["steps"]["build"]["returncode"], "tables identical:", res["all_tables_identical"],
          "figures pixel-identical:", res["all_figures_pixel_identical"], flush=True)
    if not a.skip_e0:
        res["steps"]["run_E0"] = run(f"{py} scripts/run_E0.py --nproc 1", clone, env=env, log=logs / "run_E0.log")
        print("run_E0:", res["steps"]["run_E0"]["returncode"], f"{res['steps']['run_E0']['wall_s'] / 60:.1f} min", flush=True)
        res["steps"]["pytest_quick"] = run(f"CGMY_FULL_TESTS=0 {py} -m pytest tests -q", clone, env=env, log=logs / "pytest.log")
        print("pytest:", res["steps"]["pytest_quick"]["returncode"], res["steps"]["pytest_quick"]["tail"][-200:], flush=True)
    sys.path.insert(0, str(ROOT))
    from cgmy_cert.provenance import cpu_model
    res["host"] = dict(cpu_model=cpu_model(), python=sys.version.split()[0])
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
