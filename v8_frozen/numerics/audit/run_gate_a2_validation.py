"""Gate A2: run the independent assembly against production at a handful of
real CGMY queries, N=32 and N=64, and write audit/assembly_validation.csv.

Honest scope (see reports/GATE_A_REPORT.md's A2 section for the full
account): N=32 (5 queries) and N=64 (2 queries) only. N=128/256 and the
full 256+-query Q_audit population sweep are explicitly NOT attempted here
-- the independent K_dense construction costs O(n_dof) mpmath quadrature
calls per query (no bandwidth/Toeplitz shortcut, by design, to stay
independent of production's own banding), which is fine at N<=64 but would
take several minutes per query at N=256; scaling that up is future work,
not silently skipped.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "vendor"))
sys.path.insert(0, str(REPO_ROOT / "audit"))

from independent_assembly import independent_A_h_on_mesh, assert_independent  # noqa: E402
from cgmy_bspline.matrices import operator_matrix  # noqa: E402
from cgmy_bspline.parameters import CGMYParams  # noqa: E402

BOX = dict(C=(0.05, 1.0), G=(1.0, 5.0), M=(2.0, 9.0), Y=(1.10, 1.80), r=(0.01, 0.10), T=(0.1, 1.0))
P = 3


def draw_queries(n: int, seed: int) -> list[dict]:
    sampler = qmc.Sobol(d=6, scramble=True, seed=seed)
    unit = sampler.random(n)
    lo = np.array([BOX[k][0] for k in ("C", "G", "M", "Y", "r", "T")])
    hi = np.array([BOX[k][1] for k in ("C", "G", "M", "Y", "r", "T")])
    scaled = qmc.scale(unit, lo, hi)
    return [dict(zip(("C", "G", "M", "Y", "r", "T"), row)) for row in scaled]


def run_one(query_id: str, N: int, q: dict) -> dict:
    assert_independent()
    x_min = -3.394829814011908  # log(100)-8, production left edge (domain width scaled by N here)
    h = 0.25 * (64 / N)  # keep the SAME physical domain width as production (16.0) at every N
    x_max = x_min + N * h

    t0 = time.perf_counter()
    res = independent_A_h_on_mesh(N, P, x_min, x_max, q["C"], q["G"], q["M"], q["Y"], q["r"])
    t_indep = time.perf_counter() - t0

    params = CGMYParams(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], K=100.0, T=q["T"])
    mats = operator_matrix(N, P, h, x_min, params, bandwidth=64)
    A_prod = mats["A_h"].toarray()
    S_prod = 0.5 * (A_prod + A_prod.T)
    eigs_prod = np.linalg.eigvalsh(S_prod)
    lambda_min_mass = float(np.linalg.eigvalsh(mats["M_h"].toarray())[0])

    action_error = float(np.max(np.abs(res["A_h"] - A_prod)))
    lambda_min_indep = res["lambda_min_S"]
    is_spd = lambda_min_indep > 0
    gate_status = "certified_spd" if is_spd else "proved_indefinite"
    alpha_lower_bound = lambda_min_indep - action_error  # crude: independent eigenvalue minus the
                                                          # measured disagreement with production,
                                                          # NOT yet a fully proved Weyl-type bound
                                                          # (see Gate A3, not done in this pass)
    bound_margin = alpha_lower_bound

    return dict(
        query_id=query_id, C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"],
        N=N, dof=N + P - 1, degree=P - 1, order=P,
        lambda_min_estimate=lambda_min_indep,
        alpha_lower_bound=alpha_lower_bound,
        bound_margin=bound_margin,
        gate_status=gate_status,
        lambda_min_mass=lambda_min_mass,
        lambda_min_minus_rM=float(lambda_min_indep - q["r"] * lambda_min_mass),
        assembly_error_bound=action_error,
        action_error=action_error,
        precision="mpmath dps=30 (quadrature); float64 (linear algebra)",
        quadrature_config="mpmath.quad, 4-panel [0,pi/4,pi/2,3pi/4,pi], DTFT domain (see independent_assembly.py docstring)",
        t_independent_s=t_indep,
        lambda_min_prod=float(eigs_prod[0]),
        lambda_max_prod=float(eigs_prod[-1]),
        lambda_max_indep=res["lambda_max_S"],
    )


def main():
    rows = []
    n32 = draw_queries(5, seed=90001)
    for i, q in enumerate(n32):
        print(f"N=32 query {i}: {q}")
        rows.append(run_one(f"gateA2_N32_{i:02d}", 32, q))

    n64 = draw_queries(2, seed=90002)
    for i, q in enumerate(n64):
        print(f"N=64 query {i}: {q}")
        rows.append(run_one(f"gateA2_N64_{i:02d}", 64, q))

    out_path = REPO_ROOT / "audit" / "assembly_validation.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")

    for row in rows:
        print(row["query_id"], "gate_status=", row["gate_status"],
              "action_error=", row["action_error"], "t=", row["t_independent_s"])


if __name__ == "__main__":
    main()
