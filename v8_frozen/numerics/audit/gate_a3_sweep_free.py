"""v4 Phase 1, Acceptance Check 7: rerun the 459-query Q_audit sweep and the
64 exact-vertex subset in the FREE (constrained, n_free=62) coordinate
space, alongside the existing full (n_dof=66) result, for comparison.

This is the free-space counterpart of ``audit/gate_a3_sweep.py`` (unchanged,
still the "legacy_full" sweep) -- same query set, same production mesh,
same ``compute_stability_constants`` function (per
``certification.constrained``'s design, the free path is the SAME function
handed a ``FreeOperatorCache`` instead of a raw ``OperatorCache``, not a
second implementation).

Per the task file: "the accompanying v4 diagnostic already finds all 459
free matrices positive in ordinary floating point; that is a reproducible
diagnostic, not an enclosure or a theorem over the box." This script
reports exactly that -- a floating-point diagnostic -- and additionally a
legacy-vs-free eta_sharp comparison on every full-SPD query, using a FIXED
seeded random residual (not a real solved trajectory, for runtime reasons
-- 459 full CN solves is out of scope for tonight's smoke-level rerun) to
exercise Acceptance Check 4's inequality across the population rather than
just the single-mu unit test in tests/test_constrained_free.py. This is
explicitly a floating-point diagnostic of the inequality, not a new
numerical claim about real residuals.

Run: PYTHONPATH=src python3 audit/gate_a3_sweep_free.py
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, Mesh, MeshInfo  # noqa: E402
from certification.constants import compute_stability_constants  # noqa: E402
from certification.constrained import FreeOperatorCache, free_space_for_mesh  # noqa: E402
from certification.guaranteed_certificate import eta_sharp  # noqa: E402

N_PRODUCTION = 64
P_PRODUCTION = 3
X_MIN = -3.394829814011908
X_MAX = 12.605170185988092

Q_AUDIT_PATH = REPO_ROOT / "experiments" / "raw_results" / "query_sets" / "Q_audit.csv"
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = REPO_ROOT / "results" / "v4_constrained_free" / f"gate_a3_free_{RUN_STAMP}"
OUT_PATH = OUT_DIR / "gate_a3_free_results.csv"


def main() -> None:
    mesh = Mesh(N=N_PRODUCTION, p=P_PRODUCTION, x_min=X_MIN, x_max=X_MAX)
    ops = OperatorCache(mesh, X_MIN, X_MAX)
    mesh_info = MeshInfo(
        N=N_PRODUCTION, p=P_PRODUCTION, n_dof=ops.n_dof, h=ops.h,
        x_min=X_MIN, x_max=X_MAX, nodes=np.array([]),
    )
    fs = free_space_for_mesh(mesh_info)
    free_ops = FreeOperatorCache(ops, fs)
    print(f"n_dof={fs.n_dof}, n_free={fs.n_free}, bc_left={fs.bc_left}, bc_right={fs.bc_right}")

    rng = np.random.default_rng(459_001)
    fixed_residual_full = rng.standard_normal((5, ops.n_dof))
    fixed_residual_full[:, fs.bc_idx] = 0.0
    fixed_residual_free = fixed_residual_full[:, fs.free_idx]
    dt_dummy = 1e-2

    with open(Q_AUDIT_PATH) as f:
        queries = list(csv.DictReader(f))
    print(f"Sweeping {len(queries)} Q_audit queries at N={N_PRODUCTION} (free space, n_free={fs.n_free})...")

    out_rows = []
    t0 = time.time()
    for i, q in enumerate(queries):
        mu = CGMYParameters(
            C=float(q["C"]), G=float(q["G"]), M=float(q["M"]),
            Y=float(q["Y"]), r=float(q["r"]), T=float(q["T"]),
        )
        try:
            const_full = compute_stability_constants(mu, ops, coordinate_space="legacy_full")
            const_free = compute_stability_constants(mu, free_ops, coordinate_space="constrained_free")
            is_spd_full = const_full.is_spd
            is_spd_free = const_free.is_spd

            eta_full_cmp = float("nan")
            eta_free_cmp = float("nan")
            eta_free_leq_full = ""
            if is_spd_full:
                cert_full = eta_sharp(
                    fixed_residual_full, mu, ops, dt_dummy, coordinate_space="legacy_full"
                )
                cert_free = eta_sharp(
                    fixed_residual_free, mu, free_ops, dt_dummy, coordinate_space="constrained_free"
                )
                eta_full_cmp = cert_full.eta
                eta_free_cmp = cert_free.eta
                eta_free_leq_full = str(eta_free_cmp <= eta_full_cmp + 1e-9)

            error_note = ""
        except Exception as exc:  # pragma: no cover - diagnostic path
            is_spd_full = False
            is_spd_free = False
            const_full = None
            const_free = None
            eta_full_cmp = float("nan")
            eta_free_cmp = float("nan")
            eta_free_leq_full = ""
            error_note = str(exc)

        out_rows.append(dict(
            query_id=q["query_id"], source=q["source"],
            C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, T=mu.T,
            is_spd_full=is_spd_full,
            is_spd_free=is_spd_free,
            lambda_min_full=const_full.lambda_min_numerical if const_full else float("nan"),
            lambda_min_free=const_free.lambda_min_numerical if const_free else float("nan"),
            alpha_certified_full=const_full.alpha_certified if const_full else float("nan"),
            alpha_certified_free=const_free.alpha_certified if const_free else float("nan"),
            eta_sharp_legacy_full_fixed_residual=eta_full_cmp,
            eta_sharp_constrained_free_fixed_residual=eta_free_cmp,
            eta_free_leq_eta_full=eta_free_leq_full,
            error_note=error_note,
        ))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(queries)} done ({elapsed:.1f}s elapsed)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    from collections import Counter

    by_source_total = Counter(r["source"] for r in out_rows)
    by_source_full_fail = Counter(r["source"] for r in out_rows if not r["is_spd_full"])
    by_source_free_fail = Counter(r["source"] for r in out_rows if not r["is_spd_free"])

    print("\n=== Free-space (constrained_free, n_free=62) vs legacy (legacy_full, n_dof=66) SPD gate ===")
    for source in sorted(by_source_total):
        tot = by_source_total[source]
        print(
            f"  {source}: full {tot - by_source_full_fail[source]}/{tot} SPD "
            f"({by_source_full_fail[source]} uncertified)  |  "
            f"free {tot - by_source_free_fail[source]}/{tot} SPD "
            f"({by_source_free_fail[source]} uncertified)"
        )
    tot_all = len(out_rows)
    full_fail_all = sum(1 for r in out_rows if not r["is_spd_full"])
    free_fail_all = sum(1 for r in out_rows if not r["is_spd_free"])
    print(
        f"  TOTAL: full {tot_all - full_fail_all}/{tot_all} SPD ({full_fail_all} uncertified)  |  "
        f"free {tot_all - free_fail_all}/{tot_all} SPD ({free_fail_all} uncertified)"
    )

    # Exact-vertex subset (source == "exact_vertex", 64 queries).
    vertices = [r for r in out_rows if r["source"] == "exact_vertex"]
    v_full_fail = sum(1 for r in vertices if not r["is_spd_full"])
    v_free_fail = sum(1 for r in vertices if not r["is_spd_free"])
    print(
        f"\nExact vertices ({len(vertices)} queries): full "
        f"{len(vertices) - v_full_fail}/{len(vertices)} SPD ({v_full_fail} uncertified)  |  "
        f"free {len(vertices) - v_free_fail}/{len(vertices)} SPD ({v_free_fail} uncertified)"
    )

    full_spd_rows = [r for r in out_rows if r["is_spd_full"]]
    n_cmp = len(full_spd_rows)
    n_leq = sum(1 for r in full_spd_rows if r["eta_free_leq_eta_full"] == "True")
    min_lambda_full = min(r["lambda_min_full"] for r in out_rows)
    min_lambda_free = min(r["lambda_min_free"] for r in out_rows)
    print(
        f"\nLegacy-vs-free eta_sharp comparison on full-SPD queries (fixed seeded "
        f"residual, floating-point diagnostic only): eta_free <= eta_full + 1e-9 "
        f"held for {n_leq}/{n_cmp} full-SPD queries."
    )
    print(f"min(lambda_min_numerical) over all 459: full={min_lambda_full!r}, free={min_lambda_free!r}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
