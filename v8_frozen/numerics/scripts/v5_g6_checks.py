"""G6 checks (roadmap §7) from the archived raw/summary CSVs of the M7 runs.

    python3 scripts/v5_g6_checks.py <bench_cold_dir> <bench_reuse_dir> <scaling_dir>

Writes g6_checks.json into the cold benchmark directory.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def rd(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main(cold: Path, reuse: Path, scal: Path) -> int:
    checks = []

    def check(n, ok, ev):
        checks.append(dict(check=n, passed=bool(ok), evidence=ev))

    for mode, d in (("cold", cold), ("reuse", reuse)):
        cfg = json.loads((d / "manifest.json").read_text())["benchmark_config"]
        n_m, n_q, n_r = len(cfg["methods"]["trajectory"]), 120, cfg["repeats"]["timed"]
        t = rd(d / f"trajectory_{mode}/outcomes.csv")
        p = rd(d / f"terminal_price_{mode}/outcomes.csv")
        c = rd(d / f"terminal_price_{mode}_cos/outcomes.csv")
        ref = rd(d / "reference/stored_reference.csv")
        check(f"G6.1 all declared attempts accounted ({mode})",
              len(t) == n_m * n_q * n_r and len(p) == len(t) and len(c) == n_q * n_r and len(ref) == n_q,
              dict(trajectory=len(t), price=len(p), cos=len(c), references=len(ref), expected=n_m * n_q * n_r))
        g4 = json.loads((d / "g4_checks.json").read_text())
        check(f"G6.1b raw CSVs reproduce every summary ({mode})", all(x["passed"] for x in g4),
              f"{sum(x['passed'] for x in g4)}/{len(g4)}")
        types = {r["workload"]: {x["target_type"] for x in (t if r["workload"] == "trajectory" else p + c)} for r in t[:1] + p[:1]}
        pr = rd(d / "paired_ratios.csv")
        check(f"G6.2 speed comparisons matched: one target per workload, ratios only on the common qualifying population ({mode})",
              all(len(v) == 1 for v in types.values()) and all(r["qualifying_rule"] for r in pr)
              and not any(r["workload"] == "trajectory" and r["baseline"] == "cos" for r in pr),
              dict(target_types={k: sorted(v) for k, v in types.items()},
                   setup_policy=json.loads((d / "manifest.json").read_text())["benchmark_config"]["repeats"]))
        nw = next(r for r in pr if r["workload"] == "trajectory" and r["method"] == "neural_free_workflow"
                  and r["baseline"] == "free_direct_lu")
        check(f"G6.3 neural workflow vs free dense LU reported as measured ({mode})", True,
              dict(geomean=nw["geomean_time_ratio"], ci=[nw["geomean_ci_low"], nw["geomean_ci_high"]],
                   statement=("SLOWER than a free dense-LU solve" if float(nw["geomean_ci_low"]) > 1
                              else "not slower")))
    sc = json.loads((scal / "scaling_checks.json").read_text())
    check("G6.1c scaling attempts accounted (rows, not_applicable, memory)", all(x["passed"] for x in sc),
          f"{sum(x['passed'] for x in sc)}/{len(sc)}")
    (cold / "g6_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    for x in checks:
        print(("PASS " if x["passed"] else "FAIL ") + x["check"], "|", json.dumps(x["evidence"])[:160])
    return 0 if all(x["passed"] for x in checks) else 1


if __name__ == "__main__":
    sys.exit(main(*(Path(a) for a in sys.argv[1:4])))
