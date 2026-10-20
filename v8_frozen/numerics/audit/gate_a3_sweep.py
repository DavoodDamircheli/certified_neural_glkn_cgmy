"""Gate A3: alpha_lower = r*lambda_min(M)_lower - delta, and the population
SPD/coercivity gate success rate on the full Q_audit set, at the production
mesh (N=64), using the CURRENT, unmodified certification definition
(src/certification/constants.py's full-66-dim S_h(mu) check -- not the
interior-62 submatrix; see reports/BASIS_AND_BOUNDARY_AUDIT.md for why that
restriction is NOT implemented here).

Mass-matrix lower bound: M_h is MESH-ONLY (independent of mu -- confirmed
directly: OperatorCache.apply_M never touches mu), so lambda_min(M_h) is a
single number for the whole N=64 sweep, not one per query. Rather than a
fresh derivation, this reuses the project's own already-proven technique
for S_h's alpha_certified -- certification.constants.weyl_residual_enclosure
-- applied to M_h instead. That function is general symmetric-matrix
perturbation theory (Weyl's inequality via the eigenbasis residual
||V^T S V - D||_F), not CGMY-specific, so it applies to M_h verbatim and
gives a genuinely PROVED lower bound, not merely an estimate -- the same
rigor level this project already accepts for alpha_certified.

Assembly-error bound delta: taken from Gate A2's OWN measured
action_error (audit/assembly_validation.csv, max|A_h_indep - A_h_prod|
over unit test vectors, independent mpmath assembly vs. production), the
single most-conservative value across all 7 of A2's validated queries
(N=32 and N=64 combined). This is explicitly an EMPIRICAL bound between
two independent NUMERICAL assemblies (both potentially imperfect in
different, uncorrelated ways -- different quadrature, different
truncation), not a rigorously proved a priori PDE-approximation-theory
error bound. Stated as such in the report, not dressed up as more than it
is.

Run: python audit/gate_a3_sweep.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, Mesh  # noqa: E402
from certification.constants import (  # noqa: E402
    compute_stability_constants,
    dense_from_apply,
    weyl_residual_enclosure,
)

# Production mesh (configs/base.yaml): N=64, p=3, domain = log(100) +/- 8.
N_PRODUCTION = 64
P_PRODUCTION = 3
X_MIN = -3.394829814011908
X_MAX = 12.605170185988092

Q_AUDIT_PATH = REPO_ROOT / "experiments" / "raw_results" / "query_sets" / "Q_audit.csv"
A2_VALIDATION_PATH = REPO_ROOT / "audit" / "assembly_validation.csv"
OUT_PATH = REPO_ROOT / "audit" / "gate_a3_results.csv"


def load_delta() -> tuple[float, int]:
    """Empirical assembly-error bound: max action_error across every row of
    Gate A2's own validation CSV (N=32 and N=64 combined). Returns
    (delta, n_rows_used)."""
    with open(A2_VALIDATION_PATH) as f:
        rows = list(csv.DictReader(f))
    errs = [float(r["action_error"]) for r in rows]
    return max(errs), len(errs)


def main():
    mesh = Mesh(N=N_PRODUCTION, p=P_PRODUCTION, x_min=X_MIN, x_max=X_MAX)
    ops = OperatorCache(mesh, X_MIN, X_MAX)
    n = ops.n_dof

    # --- Mass-matrix rigorous lower bound (mesh-only, computed once) ---
    M = dense_from_apply(ops.apply_M, n)
    eigs_M, V_M = np.linalg.eigh(M)
    lambda_min_M_numerical = float(eigs_M[0])
    lambda_min_M_lower = weyl_residual_enclosure(M, eigs_M, V_M)
    weyl_gap_M = lambda_min_M_numerical - lambda_min_M_lower
    print(f"lambda_min(M_h) numerical estimate: {lambda_min_M_numerical!r}")
    print(f"lambda_min(M_h) PROVED lower bound (Weyl residual enclosure): {lambda_min_M_lower!r}")
    print(f"Weyl gap (proof cost): {weyl_gap_M:.3e}")

    # --- Assembly-error bound delta (empirical, from Gate A2) ---
    delta, n_a2_rows = load_delta()
    print(f"delta (empirical, max action_error over {n_a2_rows} Gate-A2 rows): {delta!r}")

    # --- Population sweep over Q_audit ---
    with open(Q_AUDIT_PATH) as f:
        queries = list(csv.DictReader(f))
    print(f"Sweeping {len(queries)} Q_audit queries at N={N_PRODUCTION}...")

    out_rows = []
    t0 = time.time()
    for i, q in enumerate(queries):
        C, G, M_param, Y, r, T = (
            float(q["C"]), float(q["G"]), float(q["M"]),
            float(q["Y"]), float(q["r"]), float(q["T"]),
        )
        mu = CGMYParameters(C=C, G=G, M=M_param, Y=Y, r=r, T=T)

        try:
            const = compute_stability_constants(mu, ops)
            lambda_min_numerical = const.lambda_min_numerical
            alpha_certified = const.alpha_certified
            is_spd = const.is_spd
            coercivity_strategy = const.coercivity_strategy

            alpha_lower = r * lambda_min_M_lower - delta

            # lambda_min(S_h - r*M_h): a DIFFERENT, stronger diagnostic than
            # the Loewner-ratio heuristic lambda_min(S)/(r*lambda_min(M)).
            S = dense_from_apply(lambda v: ops.apply_S(mu, v), n)
            SmrM = S - r * M
            eigs_SmrM, V_SmrM = np.linalg.eigh(SmrM)
            lambda_min_S_minus_rM = float(eigs_SmrM[0])
            # Same Weyl-residual uncertainty estimate used for alpha_certified
            # (general symmetric-perturbation theory, applies verbatim).
            lambda_min_S_minus_rM_lower = weyl_residual_enclosure(SmrM, eigs_SmrM, V_SmrM)
            S_minus_rM_uncertainty = lambda_min_S_minus_rM - lambda_min_S_minus_rM_lower

            gate_status = "certified_spd" if is_spd else "uncertified"
            error_note = ""
        except Exception as exc:  # pragma: no cover - diagnostic path
            lambda_min_numerical = float("nan")
            alpha_certified = float("nan")
            is_spd = False
            coercivity_strategy = "assembly_error"
            alpha_lower = float("nan")
            lambda_min_S_minus_rM = float("nan")
            S_minus_rM_uncertainty = float("nan")
            gate_status = "assembly_error"
            error_note = str(exc)

        out_rows.append(dict(
            query_id=q["query_id"], source=q["source"],
            C=C, G=G, M=M_param, Y=Y, r=r, T=T, N=N_PRODUCTION,
            lambda_min_numerical=lambda_min_numerical,
            alpha_certified=alpha_certified,
            is_spd=is_spd,
            coercivity_strategy=coercivity_strategy,
            alpha_lower=alpha_lower,
            lambda_min_S_minus_rM=lambda_min_S_minus_rM,
            lambda_min_S_minus_rM_uncertainty=S_minus_rM_uncertainty,
            gate_status=gate_status,
            error_note=error_note,
        ))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(queries)} done ({elapsed:.1f}s elapsed, "
                  f"{elapsed/(i+1)*len(queries):.1f}s est. total)")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # --- Summary, by source ---
    from collections import Counter
    by_source_total = Counter(r["source"] for r in out_rows)
    by_source_fail = Counter(r["source"] for r in out_rows if not r["is_spd"])
    print("\n=== Population gate-rate summary (current full-66-dim definition) ===")
    for source in sorted(by_source_total):
        tot = by_source_total[source]
        fail = by_source_fail[source]
        print(f"  {source}: {tot - fail}/{tot} certified_spd ({fail} uncertified)")
    tot_all = len(out_rows)
    fail_all = sum(1 for r in out_rows if not r["is_spd"])
    print(f"  TOTAL: {tot_all - fail_all}/{tot_all} certified_spd ({fail_all} uncertified)")
    n_assembly_error = sum(1 for r in out_rows if r["gate_status"] == "assembly_error")
    if n_assembly_error:
        print(f"  ({n_assembly_error} assembly_error / raised exception)")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
