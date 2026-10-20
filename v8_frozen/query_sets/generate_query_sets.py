"""Generate the v3 predeclared query sets (experiments/config/v3_declared.yaml
Section 2: Q_audit, Q_ref30, Q_ref100, Q_bench120, Q_scale), once, with fixed
seeds, as CSVs with explicit IDs and parameter values.

Run: python experiments/generate_query_sets.py
Writes under experiments/raw_results/query_sets/.

This script only draws parameter tuples; it does not run the solver, the
certificate, or any model. Later gates consume these CSVs by query_id so
that "the same 120 queries" means literally the same rows everywhere.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "experiments" / "raw_results" / "query_sets"

with open(REPO_ROOT / "experiments" / "config" / "v3_declared.yaml") as f:
    DECL = yaml.safe_load(f)

BOX = DECL["main_parameter_box"]
PARAM_ORDER = ["C", "G", "M", "Y", "r", "T"]
K = BOX["K"]


def sobol_in_box(seed: int, n: int) -> list[dict]:
    """n Sobol points in the 6-dim main box, scrambled, fixed seed.
    scipy's qmc.Sobol requires n a power of 2 for `random_base2`; use
    `.random(n)` (scrambled, any n) for exact predeclared counts instead
    (slightly less equidistributed than a full base-2 sample, standard and
    documented tradeoff -- acceptable for a Sobol-sampled test/benchmark
    set, not for a convergence-rate claim)."""
    sampler = qmc.Sobol(d=6, scramble=True, seed=seed)
    unit = sampler.random(n)
    lo = np.array([BOX[p][0] for p in PARAM_ORDER])
    hi = np.array([BOX[p][1] for p in PARAM_ORDER])
    scaled = qmc.scale(unit, lo, hi)
    return [dict(zip(PARAM_ORDER, row)) for row in scaled]


def box_vertices() -> list[dict]:
    """All 2^6 = 64 exact vertices of the 6-dim box (CLAUDE.md's own
    "64 exact vertices" terminology, e.g. the N=64 SPD-sweep note)."""
    out = []
    for bits in range(2 ** len(PARAM_ORDER)):
        pt = {}
        for i, p in enumerate(PARAM_ORDER):
            lo, hi = BOX[p]
            pt[p] = hi if (bits >> i) & 1 else lo
        out.append(pt)
    return out


def load_known_failures() -> list[dict]:
    """All previously-known-failed queries (roadmap Q_audit requirement),
    from the two sources found during orientation. Deduplicated by rounded
    parameter tuple (two different historical runs could have drawn the
    same Sobol point)."""
    rows = []
    seen = set()

    spd_failures = REPO_ROOT / "results" / "archive_pre_v3" / "results" / "spd_failures.csv"
    if spd_failures.exists():
        with open(spd_failures) as f:
            for row in csv.DictReader(f):
                key = tuple(round(float(row[p]), 10) for p in PARAM_ORDER)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({p: float(row[p]) for p in PARAM_ORDER})

    corners = (
        REPO_ROOT / "results" / "archive_pre_v3" / "results"
        / "stage_a_difficult_corners" / "corners.csv"
    )
    if corners.exists():
        with open(corners) as f:
            for row in csv.DictReader(f):
                key = tuple(round(float(row[p]), 10) for p in PARAM_ORDER)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({p: float(row[p]) for p in PARAM_ORDER})

    return rows


def write_csv(path: Path, rows: list[dict], source_tags: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        fieldnames = ["query_id", *PARAM_ORDER, "K", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({
                "query_id": f"{path.stem}_{i:04d}",
                **{p: row[p] for p in PARAM_ORDER},
                "K": K,
                "source": source_tags[i] if source_tags else "sobol",
            })


def main():
    # Q_audit: known failures + 256 Sobol + 64 exact vertices.
    q_audit_decl = DECL["query_sets"]["Q_audit"]
    known = load_known_failures()
    sobol = sobol_in_box(seed=q_audit_decl["seed"], n=q_audit_decl["n_sobol"])
    vertices = box_vertices()
    rows = known + sobol + vertices
    tags = (
        ["known_failure"] * len(known)
        + ["sobol_main_box"] * len(sobol)
        + ["exact_vertex"] * len(vertices)
    )
    write_csv(OUT_DIR / "Q_audit.csv", rows, tags)
    print(f"Q_audit: {len(known)} known-failure + {len(sobol)} Sobol + "
          f"{len(vertices)} vertices = {len(rows)} total")

    for name in ("Q_ref30", "Q_ref100", "Q_bench120"):
        d = DECL["query_sets"][name]
        rows = sobol_in_box(seed=d["seed"], n=d["n"])
        write_csv(OUT_DIR / f"{name}.csv", rows)
        print(f"{name}: {len(rows)} Sobol queries (seed {d['seed']})")

    # Q_scale: a single common set, reused at every N in [32,64,128,256,512].
    d = DECL["query_sets"]["Q_scale"]
    n_scale = 30  # matches exp15_runtime_scaling_matrix's existing "30 common samples"
                  # convention (EFFICIENCY_NORM_KAPPA_FIX_REPORT.md Sec.7); not
                  # independently specified as a count in the roadmap itself.
    rows = sobol_in_box(seed=d["seed"], n=n_scale)
    write_csv(OUT_DIR / "Q_scale.csv", rows)
    print(f"Q_scale: {len(rows)} Sobol queries (seed {d['seed']}, shared across all N)")


if __name__ == "__main__":
    main()
