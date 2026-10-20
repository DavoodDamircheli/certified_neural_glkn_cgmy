"""Run configuration and manifest generation for v5 results (roadmap §1, M0).

A v5 run lives in ``results/free_integrated_v5/<run_id>/`` and is identified by
``manifest.json``: method, source revision (git state AND a content hash of every
source/config file, since a commit id alone does not identify a dirty tree),
resolved configuration, query populations (by content hash), checkpoints and
bases (by content hash), software versions, hardware, thread settings, seeds and
the full discretization/solver policy (mesh, basis order/degree, boundary index
sets, time grid policy, scheme, solver tolerances, PCG policy).

Query populations are *declared* in ``configs/v5_query_populations.json`` with
their expected hashes; ``declare_populations`` recomputes the hashes and raises
on any mismatch instead of silently using a changed file. Query files are only
ever read here.

Studies that a run does not execute are recorded with status ``"not run"`` --
never represented by an empty success table.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.methods import KNOWN_METHODS, NUMERICAL_SCOPE, validate_method_id

REPO_ROOT = Path(__file__).resolve().parents[2]
V5_RESULTS_ROOT = REPO_ROOT / "results" / "free_integrated_v5"
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"
WORKFLOW_CONFIG = REPO_ROOT / "configs" / "v5_free_workflow.yaml"
POPULATIONS_DECLARATION = REPO_ROOT / "configs" / "v5_query_populations.json"

# Source/config files whose content defines the numerical method.
SOURCE_GLOBS = (
    "src/**/*.py", "vendor/**/*.py", "experiments/*.py", "scripts/*.py",
    "benchmarks/*.py", "financial/*.py", "validation/*.py", "audit/*.py",
    "configs/*.yaml", "configs/*.json", "tests/**/*.py", "pyproject.toml",
    "requirements.txt",
)
# Pre-existing evidence that v5 must never modify (G0, "old evidence unchanged").
EVIDENCE_ROOTS = ("results", "checkpoints", "data", "experiments/raw_results")
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

QUERY_COLUMNS = ("query_id", "C", "G", "M", "Y", "r", "T", "K", "source")

STUDIES = (
    "g0_dry_run", "g1_free_path_traces", "pod_free_baseline", "riesz_bracket_campaign",
    "audit459", "bench120_cold", "bench120_reusable", "scaling", "assembly_refinement",
    "fourier_rerun", "financial_q_ref30",
)


# ---------------------------------------------------------------- hashing

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    """Hash of a float array's exact bytes, dtype and shape (C order)."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def tree_hashes(root: Path, globs: tuple[str, ...], exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    files: dict[str, str] = {}
    for pattern in globs:
        for p in sorted(root.glob(pattern)):
            rel = p.relative_to(root).as_posix()
            if not p.is_file() or "__pycache__" in rel or any(rel.startswith(e) for e in exclude):
                continue
            files[rel] = sha256_file(p)
    combined = sha256_bytes("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode())
    return {"tree_sha256": combined, "n_files": len(files), "files": files}


def source_hashes(root: Path = REPO_ROOT) -> dict[str, Any]:
    return tree_hashes(root, SOURCE_GLOBS)


def evidence_snapshot(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Content hashes of all pre-existing evidence (excluding v5's own output tree)."""
    globs = tuple(f"{r}/**/*" for r in EVIDENCE_ROOTS)
    return tree_hashes(root, globs, exclude=("results/free_integrated_v5",))


def compare_evidence(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    b, a = before["files"], after["files"]
    return {
        "modified": sorted(k for k in b if k in a and a[k] != b[k]),
        "deleted": sorted(k for k in b if k not in a),
        "added": sorted(k for k in a if k not in b),
    }


# ---------------------------------------------------------------- environment

def git_state(root: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root).decode()

    status = run("status", "--porcelain").splitlines()
    diff = run("diff", "HEAD").encode()
    return {
        "head": run("rev-parse", "HEAD").strip(),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(status),
        "status_porcelain": status,
        "diff_vs_head_sha256": sha256_bytes(diff),
    }


def software_versions() -> dict[str, Any]:
    import h5py
    import scipy
    import torch

    blas: Any
    try:
        blas = np.show_config(mode="dicts").get("Build Dependencies", {})
    except TypeError:  # numpy < 1.25 has no mode= argument
        blas = "unavailable"
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "h5py": h5py.__version__,
        "pyyaml": yaml.__version__,
        "numpy_build_dependencies": blas,
    }


def hardware() -> dict[str, Any]:
    cpu_model = platform.processor() or None
    mem_total_kb = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem_total_kb = int(line.split()[1])
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "mem_total_kb": mem_total_kb,
        "hostname_sha256": sha256_bytes(platform.node().encode()),
    }


def thread_settings() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {v: os.environ.get(v) for v in THREAD_ENV_VARS}
    out["torch_num_threads"] = torch.get_num_threads()
    out["single_threaded"] = all(os.environ.get(v) == "1" for v in THREAD_ENV_VARS)
    return out


# ---------------------------------------------------------------- configuration

def load_workflow_config(path: Path = WORKFLOW_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def discretization_record(base_cfg: dict[str, Any]) -> dict[str, Any]:
    """Mesh, basis, boundary index sets, free partition and time-grid policy,
    derived from configs/base.yaml through the same functions production uses."""
    from certification.constrained import free_space_for_mesh
    from data.sampling import load_mesh_and_time_grid
    from pipeline.predict_certify_correct import build_mesh_info

    mesh, time_grid = load_mesh_and_time_grid(BASE_CONFIG)
    mi = build_mesh_info(mesh)
    fs = free_space_for_mesh(mi)
    return {
        "N": mi.N,
        "spline_order_p": mi.p,
        "spline_degree": mi.p - 1,
        "x_min": mi.x_min,
        "x_max": mi.x_max,
        "h": mi.h,
        "n_dof": mi.n_dof,
        "bc_left": list(fs.bc_left),
        "bc_right": list(fs.bc_right),
        "n_free": fs.n_free,
        "free_idx": fs.free_idx.tolist(),
        "time_grid_policy": {
            "n_steps": time_grid.n_steps,
            "grid": "uniform np.linspace(0, T, n_steps+1) per query; dt = T/n_steps",
            "tau0": 0.0,
        },
        "scheme": base_cfg["discretization"]["time_stepping"],
        "scheme_detail": "pure Crank-Nicolson from tau=0; no Rannacher/BE start-up",
    }


def resolve_config(workflow_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    base_cfg = yaml.safe_load(BASE_CONFIG.read_text())
    workflow_cfg = workflow_cfg if workflow_cfg is not None else load_workflow_config()
    validate_method_id(workflow_cfg["method_id"])
    return {
        "base_config": base_cfg,
        "base_config_sha256": sha256_file(BASE_CONFIG),
        "workflow_config": workflow_cfg,
        "discretization": discretization_record(base_cfg),
    }


# ---------------------------------------------------------------- populations

def queries_csv_text(rows: list[dict[str, Any]]) -> str:
    """Canonical CSV text for a query list (floats via repr, so it round-trips)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(QUERY_COLUMNS)
    for r in rows:
        w.writerow([r[c] if not isinstance(r[c], float) else repr(r[c]) for c in QUERY_COLUMNS])
    return buf.getvalue()


def generate_validation30() -> list[dict[str, Any]]:
    """The fixed 30-query validation population: the same Sobol draw
    (seed 999_999) ``scripts/generate_production_dataset.build_val_set`` uses,
    without running any solve."""
    from data.sampling import load_parameter_box, sample_sobol

    box = load_parameter_box(BASE_CONFIG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Sobol balance warning for n != 2^k, as in build_val_set
        samples = sample_sobol(box, 30, seed=999_999)
    return [
        {"query_id": f"V30_{i:04d}", "C": float(mu.C), "G": float(mu.G), "M": float(mu.M),
         "Y": float(mu.Y), "r": float(mu.r), "T": float(mu.T), "K": float(payoff.strike),
         "source": "sobol_seed999999"}
        for i, (mu, payoff) in enumerate(samples)
    ]


def read_query_file(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("source", "reference_validation_v3")  # historical results CSV without a source column
        for c in ("C", "G", "M", "Y", "r", "T", "K"):
            r[c] = float(r[c])
    return rows


class PopulationMismatchError(RuntimeError):
    pass


def declare_populations(declaration: Path = POPULATIONS_DECLARATION) -> dict[str, Any]:
    """Verify every declared population against its frozen hash and counts.

    Returns {name: {"rows": [...], "sha256": ..., ...}}. Raises
    PopulationMismatchError on any hash/count/source mismatch.
    """
    decl = json.loads(Path(declaration).read_text())
    out: dict[str, Any] = {}
    for name, spec in decl["populations"].items():
        if spec["kind"] == "file":
            path = REPO_ROOT / spec["path"]
            sha = sha256_file(path)
            rows = read_query_file(path)
        elif spec["kind"] == "generated_validation30":
            rows = generate_validation30()
            sha = sha256_bytes(queries_csv_text(rows).encode())
        else:
            raise PopulationMismatchError(f"{name}: unknown population kind {spec['kind']!r}")
        if sha != spec["sha256"]:
            raise PopulationMismatchError(
                f"population {name}: sha256 {sha} != declared {spec['sha256']} -- the query "
                f"set changed; refusing to run on an undeclared population."
            )
        if len(rows) != spec["n_queries"]:
            raise PopulationMismatchError(f"{name}: {len(rows)} rows != declared {spec['n_queries']}")
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        if counts != spec["source_counts"]:
            raise PopulationMismatchError(f"{name}: source counts {counts} != declared {spec['source_counts']}")
        out[name] = {**spec, "rows": rows}
    return out


def checkpoint_hashes(root: Path = REPO_ROOT) -> dict[str, str]:
    ck = root / "checkpoints"
    return {p.name: sha256_file(p) for p in sorted(ck.iterdir()) if p.is_file() and p.name != ".gitkeep"}


# ---------------------------------------------------------------- run directory

def new_run_id(label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    return f"{stamp}_{safe}"


def create_run_dir(run_id: str, root: Path = V5_RESULTS_ROOT) -> Path:
    run_dir = root / run_id
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; v5 never overwrites a run directory")
    run_dir.mkdir(parents=True)
    return run_dir


def build_manifest(
    run_id: str,
    *,
    method_id: str,
    studies: dict[str, str],
    seeds: dict[str, int],
    resolved_config: dict[str, Any],
    populations: dict[str, Any],
    population_used: list[str],
    extra_inputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dict. ``studies`` maps every name in STUDIES (others
    allowed) to a status; names missing from it are recorded as "not run"."""
    validate_method_id(method_id)
    status = {s: "not run" for s in STUDIES}
    status.update(studies)
    return {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method_id": method_id,
        "coordinate_space": KNOWN_METHODS[method_id]["coordinate_space"],
        "method_description": KNOWN_METHODS[method_id]["description"],
        "numerical_scope": NUMERICAL_SCOPE,
        "git": git_state(),
        "source_tree_sha256": source_hashes()["tree_sha256"],
        "resolved_config_sha256": sha256_bytes(
            json.dumps(resolved_config, sort_keys=True, default=str).encode()
        ),
        "base_config_sha256": resolved_config["base_config_sha256"],
        "discretization": {k: v for k, v in resolved_config["discretization"].items() if k != "free_idx"},
        "solver_policy": resolved_config["workflow_config"]["solver"],
        "certificate_policy": resolved_config["workflow_config"]["certificate"],
        "checkpoints_sha256": checkpoint_hashes(),
        "extra_inputs_sha256": {k: sha256_file(REPO_ROOT / v) for k, v in (extra_inputs or {}).items()},
        "query_populations": {
            name: {k: v for k, v in spec.items() if k != "rows"} for name, spec in populations.items()
        },
        "population_used": population_used,
        "seeds": seeds,
        "software": software_versions(),
        "hardware": hardware(),
        "threads": thread_settings(),
        "studies": status,
    }


def write_json(path: Path, obj: Any) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; v5 never overwrites output")
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")
