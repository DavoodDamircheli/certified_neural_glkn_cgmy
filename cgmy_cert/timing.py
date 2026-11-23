"""Timing protocol for all wall-clock numbers that may enter the paper.

Idle machine (load averages recorded before/after), one thread (OMP/MKL/OPENBLAS = 1, torch 1 thread), pinned to one
performance core, one warm-up query outside the population, 5 repetitions per query in a reproducibly shuffled method
order, medians of per-query medians, geometric-mean ratios with 95% query-bootstrap intervals (10 000 resamples).
"""

from __future__ import annotations

import os
import platform
import random
import time
from pathlib import Path

import numpy as np

PERF_CORE = 3                 # a performance core of the Core Ultra 9 285K (cores 0-7 are P-cores; 0 handles interrupts)
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def loadavg() -> list[float]:
    return [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]


def pin(core: int = PERF_CORE) -> None:
    os.sched_setaffinity(0, {core})


def check_single_thread() -> None:
    bad = {k: os.environ.get(k) for k in THREAD_ENV if os.environ.get(k) != "1"}
    if bad:
        raise RuntimeError(f"timing requires one thread: set {bad} to 1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def hardware() -> dict:
    def rd(p):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return None
    import numpy, scipy
    try:
        import torch
        tv = torch.__version__
    except ImportError:
        tv = None
    from .provenance import cpu_model, git_sha
    core = sorted(os.sched_getaffinity(0))
    c = core[0]
    return dict(cpu_model=cpu_model(), pinned_cores=core,
                governor=rd(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor"),
                energy_performance_preference=rd(f"/sys/devices/system/cpu/cpu{c}/cpufreq/energy_performance_preference"),
                cpuinfo_max_freq_khz=rd(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq"),
                no_turbo=rd("/sys/devices/system/cpu/intel_pstate/no_turbo"),
                python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__, torch=tv,
                git_sha=git_sha(), threads={k: os.environ.get(k) for k in THREAD_ENV})


def time_call(fn, reps: int = 5) -> list[float]:
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def protocol(methods: dict, queries: list, warmup, reps: int = 5, seed: int = 0, inner: int = 1) -> dict:
    """methods: {name: f(query) -> None}; warmup: a query outside the population (run once per method).
    Returns {name: [per-query median seconds]} with the method order shuffled per query (reproducibly)."""
    for f in methods.values():
        f(warmup)
    rng = random.Random(seed)
    res = {k: [] for k in methods}
    for q in queries:
        order = list(methods)
        rng.shuffle(order)
        for k in order:
            f = methods[k]
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                for _ in range(inner):
                    f(q)
                ts.append((time.perf_counter() - t0) / inner)
            res[k].append(float(np.median(ts)))
    return res


def geo_ratio_ci(t: np.ndarray, t_ref: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Geometric-mean ratio t / t_ref over queries with a 95% query-bootstrap interval."""
    lr = np.log(np.asarray(t) / np.asarray(t_ref))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, lr.size, size=(n_boot, lr.size))
    boot = np.exp(lr[idx].mean(axis=1))
    return dict(geo_mean_ratio=float(np.exp(lr.mean())), ci95=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                n_queries=int(lr.size), n_boot=n_boot)
