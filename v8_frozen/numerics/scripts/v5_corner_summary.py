"""Summary of the corner refinement diagnostic, from its raw CSVs only.

    python3 scripts/v5_corner_summary.py <run_dir>

Per query and N (n_steps=240): max over spots of |V_h(tau=T) - COS| (independent
continuum diagnostic), max over spots of |V_h(tau=0) - payoff| (projection of the
kinked payoff), min price (negativity), paired refinement difference
max_s |V_N - V_2N| at tau=T and observed rate log2 of successive differences.
DIAGNOSTIC: none of these numbers is certified.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main(run_dir: Path) -> None:
    disc = rd(run_dir / "discrete/prices.csv")
    cos = rd(run_dir / "cos/prices.csv")
    cosp = defaultdict(dict)
    pay = defaultdict(dict)
    flags = defaultdict(list)
    for r in cos:
        cosp[r["query_id"]][float(r["spot"])] = float(r["cos_price"])
        pay[r["query_id"]][float(r["spot"])] = float(r["payoff"])
        flags[r["query_id"]].append(r["flagged_not_converged"])
    rows = []
    vT = {}
    for r in disc:
        q, N, ns = r["query_id"], int(r["N"]), int(r["n_steps"])
        spots = json.loads(r["spots"])
        p0, pT = json.loads(r["price_tau0"]), json.loads(r["price_tauT"])
        vT[(q, N, ns)] = pT
        rows.append(dict(query_id=q, N=N, h=float(r["h"]), n_steps=ns,
                         max_abs_vs_cos_tauT=max(abs(p - cosp[q][s]) for p, s in zip(pT, spots)),
                         max_abs_projection_error_tau0=max(abs(p - pay[q][s]) for p, s in zip(p0, spots)),
                         min_price_tauT=min(pT), cos_any_not_converged="True" in flags[q]))
    rows.sort(key=lambda r: (r["query_id"], r["n_steps"], r["N"]))
    for r in rows:
        nxt = vT.get((r["query_id"], 2 * r["N"], r["n_steps"]))
        r["max_abs_diff_vs_2N_tauT"] = (max(abs(a - b) for a, b in zip(vT[(r["query_id"], r["N"], r["n_steps"])], nxt))
                                        if nxt else float("nan"))
    for r in rows:
        nxt = next((x for x in rows if x["query_id"] == r["query_id"] and x["n_steps"] == r["n_steps"]
                    and x["N"] == 2 * r["N"]), None)
        a, b = r["max_abs_diff_vs_2N_tauT"], (nxt or {}).get("max_abs_diff_vs_2N_tauT", float("nan"))
        r["observed_rate_log2"] = math.log2(a / b) if a == a and b == b and a > 0 and b > 0 else float("nan")
    with open(run_dir / "corner_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    def g(v):
        return f"{v:.3g}" if isinstance(v, float) else str(v)

    md = [f"# Corner refinement diagnostic {run_dir.name}", "",
          "DIAGNOSTIC, not certified. Spots 0.8K..1.2K. COS = converge_price (independent continuum price).", "",
          "| query | N | h | n_steps | max abs(V_h(T)-COS) | max abs(V_h(0)-payoff) | min V_h(T) | max abs(V_N-V_2N) | observed rate | COS flag |",
          "|" + "---|" * 10]
    md += [f"| {r['query_id']} | {r['N']} | {g(r['h'])} | {r['n_steps']} | {g(r['max_abs_vs_cos_tauT'])} | "
           f"{g(r['max_abs_projection_error_tau0'])} | {g(r['min_price_tauT'])} | {g(r['max_abs_diff_vs_2N_tauT'])} | "
           f"{g(r['observed_rate_log2'])} | {r['cos_any_not_converged']} |" for r in rows]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
