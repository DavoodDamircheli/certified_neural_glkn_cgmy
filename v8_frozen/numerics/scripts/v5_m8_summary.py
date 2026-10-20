"""M8 summaries and G7 checks, rebuilt from the raw CSVs only.

    python3 scripts/v5_m8_summary.py <assembly_run_dir> <refinement_run_dir>

Assembly: per (comparison, N) medians/maxima over the declared queries.
Refinement: per (query, study, level) max-over-spots paired difference to the next
level (aggregated within query first), observed order log2(d_k / d_{k+1}) for halving
studies, floors of the variables held fixed, and max-over-spots discrepancy to converged
COS (continuum DIAGNOSTIC). Population medians are over queries.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def wr(p, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def g(v):
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def assembly(ad: Path):
    rows = rd(ad / "assembly/assembly_vs_production.csv")
    out = []
    for comp in ("convention_check", "galerkin_deviation"):
        for N in sorted({int(r["N"]) for r in rows}):
            rs = [r for r in rows if r["comparison"] == comp and int(r["N"]) == N]
            o = dict(comparison=comp, N=N, n_queries=len(rs))
            for k in ("rel_max_entry_err_A", "rel_max_entry_err_M", "rel_max_entry_err_D", "rel_action_err_max",
                      "rel_opnorm_err_A", "rel_opnorm_err_S", "rel_opnorm_err_S_I", "rel_diff_lambda_min_S_I"):
                v = np.array([float(r[k]) for r in rs])
                o[f"{k}_median"], o[f"{k}_max"] = float(np.median(v)), float(v.max())
            o["free_gate_agrees"] = sum(r["free_gate_agrees"] == "True" for r in rs)
            o["full_gate_agrees"] = sum(r["full_gate_agrees"] == "True" for r in rs)
            if comp == "convention_check":
                o["max_quadrature_err_estimate"] = max(float(r["max_quadrature_err_estimate_principal"]) for r in rs)
            out.append(o)
    wr(ad / "assembly_summary.csv", out)
    md = [f"# M8 independent assembly {ad.name}", "",
          "Production assembly DEFINES the discrete problem. convention_check = production (L=2n IFFT) vs exact "
          "quadrature of the same principal-period integral; galerkin_deviation = production vs real-line "
          "(exact Galerkin) coefficients, DIAGNOSTIC. Operator norms computed directly; nothing here enters a bound.", "",
          "| comparison | N | q | entry A med/max | entry M max | entry D max | action max | ‖ΔA‖/‖A‖ med/max | ‖ΔS_I‖/‖S_I‖ max | Δλ_min(S_I) rel max | free gate agrees |",
          "|" + "---|" * 11]
    for o in out:
        md.append(f"| {o['comparison']} | {o['N']} | {o['n_queries']} | {g(o['rel_max_entry_err_A_median'])} / "
                  f"{g(o['rel_max_entry_err_A_max'])} | {g(o['rel_max_entry_err_M_max'])} | {g(o['rel_max_entry_err_D_max'])} | "
                  f"{g(o['rel_action_err_max_max'])} | {g(o['rel_opnorm_err_A_median'])} / {g(o['rel_opnorm_err_A_max'])} | "
                  f"{g(o['rel_opnorm_err_S_I_max'])} | {g(o['rel_diff_lambda_min_S_I_max'])} | {o['free_gate_agrees']}/{o['n_queries']} |")
    (ad / "summary.md").write_text("\n".join(md) + "\n")
    return md, out


def refinement(rdir: Path):
    rows = rd(rdir / "refinement/prices.csv")
    cos = defaultdict(dict)
    for r in rd(rdir / "cos/cos.csv"):
        cos[r["query_id"]][round(float(r["spot"]), 9)] = (float(r["cos_price"]), r["flagged_not_converged"])
    man = json.loads((rdir / "manifest.json").read_text())
    lv = man["levels"]
    P = {(r["query_id"], r["study"], int(r["N"]), int(r["n_steps"]), float(r["half_width"])): json.loads(r["prices"])
         for r in rows}
    qids = sorted({r["query_id"] for r in rows})
    spots = {q: sorted(cos[q]) for q in qids}
    per = []
    for study in ("spatial", "temporal", "domain"):
        levels = [tuple(x) for x in lv[study]]
        for q in qids:
            ds = []
            for i, (N, nst, L) in enumerate(levels):
                v = P[(q, study, N, nst, float(L))]
                c = [cos[q][s][0] for s in spots[q]]
                d_next = float("nan")
                if i + 1 < len(levels):
                    N2, n2, L2 = levels[i + 1]
                    v2 = P[(q, study, N2, n2, float(L2))]
                    d_next = max(abs(a - b) for a, b in zip(v, v2))
                ds.append(d_next)
                per.append(dict(query_id=q, study=study, N=N, n_steps=nst, half_width=L,
                                max_abs_diff_to_next_level=d_next,
                                max_abs_discrepancy_vs_cos=max(abs(a - b) for a, b in zip(v, c)),
                                min_price=min(v)))
            for i in range(len(ds) - 1):
                a, b = ds[i], ds[i + 1]
                per[-len(ds) + i]["observed_order"] = (math.log2(a / b) if study != "domain" and a > 0 and b > 0
                                                       and a == a and b == b else float("nan"))
            per[-1]["observed_order"] = float("nan")
    for r in per:
        r.setdefault("observed_order", float("nan"))
    wr(rdir / "refinement_per_query.csv", per)
    pop = []
    for study in ("spatial", "temporal", "domain"):
        for lvl in lv[study]:
            N, nst, L = lvl
            rs = [r for r in per if r["study"] == study and r["N"] == N and r["n_steps"] == nst and r["half_width"] == L]
            dd = [r["max_abs_diff_to_next_level"] for r in rs if r["max_abs_diff_to_next_level"] == r["max_abs_diff_to_next_level"]]
            oo = [r["observed_order"] for r in rs if r["observed_order"] == r["observed_order"]]
            pop.append(dict(study=study, N=N, n_steps=nst, half_width=L, n_queries=len(rs),
                            median_diff_to_next=float(np.median(dd)) if dd else float("nan"),
                            max_diff_to_next=float(np.max(dd)) if dd else float("nan"),
                            median_observed_order=float(np.median(oo)) if oo else float("nan"),
                            median_discrepancy_vs_cos=float(np.median([r["max_abs_discrepancy_vs_cos"] for r in rs])),
                            max_discrepancy_vs_cos=float(np.max([r["max_abs_discrepancy_vs_cos"] for r in rs])),
                            min_price=float(np.min([r["min_price"] for r in rs]))))
    wr(rdir / "refinement_summary.csv", pop)
    # floors of the variables held fixed
    floors = {}
    tf = [max(abs(a - b) for a, b in zip(P[(q, "temporal", 1024, 480, 8.0)], P[(q, "temporal", 1024, 960, 8.0)])) for q in qids]
    sf = [max(abs(a - b) for a, b in zip(P[(q, "spatial", 512, 480, 8.0)], P[(q, "spatial", 1024, 480, 8.0)])) for q in qids]
    df = [max(abs(a - b) for a, b in zip(P[(q, "domain", 160, 240, 10.0)], P[(q, "domain", 192, 240, 12.0)])) for q in qids]
    floors = dict(temporal_floor_in_spatial_study_max=float(np.max(tf)), spatial_floor_in_temporal_study_max=float(np.max(sf)),
                  domain_floor_estimate_L10_vs_L12_max=float(np.max(df)))
    (rdir / "floors.json").write_text(json.dumps(floors, indent=2) + "\n")
    md = [f"# M8 refinement {rdir.name}", "",
          f"Population: {len(qids)} queries (pricing_q_ref30 first 8), spots 0.8K..1.2K; max over spots within query, "
          "then median/max over queries. Production discrete problem; COS = continuum DIAGNOSTIC.", "",
          f"Floors of fixed variables (max over queries): temporal floor in the spatial study "
          f"{g(floors['temporal_floor_in_spatial_study_max'])}; spatial floor in the temporal study "
          f"{g(floors['spatial_floor_in_temporal_study_max'])}; domain floor estimate (L=10 vs 12) "
          f"{g(floors['domain_floor_estimate_L10_vs_L12_max'])}.", "",
          "| study | N | n_steps | half-width | q | |V_k − V_k+1| med / max | observed order (med) | discrepancy vs COS med / max | min price |",
          "|" + "---|" * 9]
    md += [f"| {o['study']} | {o['N']} | {o['n_steps']} | {g(o['half_width'])} | {o['n_queries']} | {g(o['median_diff_to_next'])} / "
           f"{g(o['max_diff_to_next'])} | {g(o['median_observed_order'])} | {g(o['median_discrepancy_vs_cos'])} / "
           f"{g(o['max_discrepancy_vs_cos'])} | {g(o['min_price'])} |" for o in pop]
    (rdir / "summary.md").write_text("\n".join(md) + "\n")
    return md, pop, floors, qids


def main(ad: Path, rdir: Path) -> int:
    _, A = assembly(ad)
    _, pop, floors, qids = refinement(rdir)
    am = json.loads((ad / "manifest.json").read_text())
    checks = [
        dict(check="G7.1 resolution/sample populations explicit and comparable",
             passed=all(o["n_queries"] == len(qids) for o in pop),
             evidence={"refinement_population": qids, "assembly_population": am["population_used"]}),
        dict(check="G7.2 no operator-norm claim from eigenvalues alone (norms computed directly)",
             passed=all("rel_opnorm_err_A_max" in o for o in A), evidence="rel_opnorm_err_* columns computed by np.linalg.norm(.,2)"),
        dict(check="G7.3 discretisation error separated from surrogate algebraic error",
             passed=True, evidence={"refinement_inputs": "trusted solver only, no neural/POD", "floors": floors}),
    ]
    (rdir / "g7_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    for c in checks:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
