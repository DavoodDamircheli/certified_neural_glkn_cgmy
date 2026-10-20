"""Summary of a scaling_v5 run, rebuilt from its raw CSVs only.

    PYTHONPATH=src:experiments python3 benchmarks/scaling_v5_summary.py <run_dir>

Per (N, method): attempted/expected rows, status counts, target-verified count,
median (over queries) of the median-over-repetitions total time and phases,
median GMRES/PCG work, median peak traced memory; empirical log-log growth
exponent of time vs n_dof over N=64..256 (all 30 queries) and, separately, the
N=512 subset ratio vs N=256 on the same 6 queries. Exponents are descriptive
fits on 3 points, not complexity claims.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def rd(p: Path) -> list[dict]:
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main(run_dir: Path) -> int:
    man = json.loads((run_dir / "manifest.json").read_text())
    scfg = man["scaling_config"]
    out = rd(run_dir / "scaling/outcomes.csv")
    tim = {r["key"]: r for r in rd(run_dir / "scaling/raw_timings.csv")}
    mem = rd(run_dir / "scaling/memory.csv")
    na = rd(run_dir / "scaling/not_applicable.csv")
    Ns = scfg["resolutions"] + [scfg["extended"]["N"]]
    n_q = {N: 30 for N in scfg["resolutions"]}
    n_q[scfg["extended"]["N"]] = len(scfg["extended"]["queries"])
    reps = scfg["repeats"]["timed"]
    checks = []
    rows = []
    qmed = defaultdict(dict)
    for N in Ns:
        for m in scfg["methods"]:
            rs = [r for r in out if int(r["N"]) == N and r["solver_method"] == m]
            exp = n_q[N] * reps
            checks.append(dict(check=f"rows N={N} {m}", passed=len(rs) == exp, evidence=[len(rs), exp]))
            byq = defaultdict(list)
            for r in rs:
                byq[r["query_id"]].append(r)
            phases = defaultdict(list)
            for q, qr in byq.items():
                qmed[(N, m)][q] = float(np.median([float(r["total_time_s"]) for r in qr]))
                ph = [json.loads(tim[r["key"]]["phases_json"]) for r in qr]
                for k in {k for p in ph for k in p}:
                    phases[k].append(float(np.median([p.get(k, 0.0) for p in ph])))
            st = Counter(r["final_status"] for r in rs)
            mm = [float(r["peak_traced_bytes"]) for r in mem if int(r["N"]) == N and r["solver_method"] == m]
            rows.append(dict(
                N=N, n_dof=int(rs[0]["n_dof"]) if rs else "", method=m, n_queries=len(byq), n_rows=len(rs),
                n_target_verified=sum(r["target_status"] == "verified" for r in rs),
                statuses=json.dumps(dict(st)),
                median_time_s=float(np.median(list(qmed[(N, m)].values()))) if byq else float("nan"),
                **{f"median_{k}_s": float(np.median(v)) for k, v in sorted(phases.items())
                   if k in ("setup_operator_cache", "setup_problem", "solve", "gate", "certify", "project")},
                median_gmres_iterations=float(np.median([float(r["gmres_iterations_total"]) for r in rs])) if rs else float("nan"),
                median_pcg_iterations=float(np.median([float(r["pcg_iterations_total"]) for r in rs])) if rs else float("nan"),
                median_peak_traced_MB=float(np.median(mm)) / 2**20 if mm else float("nan"),
                n_memory_rows=len(mm)))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("N", "n_dof", "method"), k))
    with open(run_dir / "scaling_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(rows)

    growth = []
    base = scfg["resolutions"]
    sub = scfg["extended"]["queries"]
    for m in scfg["methods"]:
        nd = np.array([next(int(r["n_dof"]) for r in out if int(r["N"]) == N) for N in base], dtype=float)
        t = np.array([np.median(list(qmed[(N, m)].values())) for N in base])
        slope = float(np.polyfit(np.log(nd), np.log(t), 1)[0])
        r512 = [qmed[(scfg["extended"]["N"], m)][q] / qmed[(256, m)][q] for q in sub
                if q in qmed[(scfg["extended"]["N"], m)] and q in qmed[(256, m)]]
        growth.append(dict(method=m, loglog_slope_time_vs_ndof_N64_256=slope,
                           median_ratio_N512_over_N256_subset=float(np.median(r512)) if r512 else float("nan"),
                           n_subset=len(r512)))
    with open(run_dir / "scaling_growth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(growth[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(growth)
    checks.append(dict(check="not_applicable rows for neural/POD at N != 64",
                       passed=len(na) == len(scfg["not_applicable_at_N_ne_64"]) * (len(Ns) - 1), evidence=len(na)))
    checks.append(dict(check="memory pass covers every (N, query, method)",
                       passed=len(mem) == sum(n_q[N] for N in Ns) * len(scfg["methods"]), evidence=len(mem)))
    (run_dir / "scaling_checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    def g(v):
        return f"{v:.3g}" if isinstance(v, float) else str(v)

    md = [f"# Scaling {run_dir.name}", "", "Cold mode, fixed n_steps=60, solver/certificate methods only "
          "(neural/POD: N=64 artifacts, no transfer strategy -> not applicable at N != 64). "
          "Times: median over queries of the median over repetitions. Floating-point evaluation of exact-arithmetic theory.", "",
          "| N | n_dof | method | queries | verified | median s | setup cache s | solve s | gate s | certify s | GMRES it | PCG it | peak MB |",
          "|" + "---|" * 13]
    for r in rows:
        md.append(f"| {r['N']} | {r['n_dof']} | {r['method']} | {r['n_queries']} | {r['n_target_verified']}/{r['n_rows']} | "
                  f"{g(r['median_time_s'])} | {g(r.get('median_setup_operator_cache_s', ''))} | {g(r.get('median_solve_s', ''))} | "
                  f"{g(r.get('median_gate_s', ''))} | {g(r.get('median_certify_s', ''))} | {g(r['median_gmres_iterations'])} | "
                  f"{g(r['median_pcg_iterations'])} | {g(r['median_peak_traced_MB'])} |")
    md += ["", "| method | log-log slope time vs n_dof (N=64..256) | median t(512)/t(256) on the 6-query subset |", "|---|---|---|"]
    md += [f"| {r['method']} | {g(r['loglog_slope_time_vs_ndof_N64_256'])} | {g(r['median_ratio_N512_over_N256_subset'])} (n={r['n_subset']}) |"
           for r in growth]
    md += ["", "Checks: " + ", ".join(f"{c['check']}: {'PASS' if c['passed'] else 'FAIL'}" for c in checks)]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
