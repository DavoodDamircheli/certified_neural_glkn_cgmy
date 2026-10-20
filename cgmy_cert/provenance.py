"""Provenance block attached to every results JSON (brief, rule 4): hardware, threads, versions, git SHA."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance(**extra) -> dict:
    import numpy, scipy
    threads = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    return dict(platform=platform.platform(), cpu_model=cpu_model(), cpu_count=os.cpu_count(),
                affinity_cpus=affinity, thread_env=threads, python=platform.python_version(),
                numpy=numpy.__version__, scipy=scipy.__version__, git_sha=git_sha(),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
