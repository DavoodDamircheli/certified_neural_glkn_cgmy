"""Independent rebuild + G4 verification for a gate_d_v5 run (roadmap §5.4, G4).

    PYTHONPATH=src:experiments:benchmarks python3 scripts/v5_bench_rebuild.py <run_dir>

Re-implements (without importing benchmarks/gate_d_v5_summary.py's aggregation)
the query-level medians, population counts, paired ratios with bootstrap, and
amortisation from the RAW CSVs, asserts that every number in the harness's
summary CSVs is reproduced, asserts the expected row counts, and checks every
G4 item. Writes <run_dir>/g4_checks.json and exits non-zero on any failure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[dict] = []


def check(name: str, ok: bool, evidence) -> None:
    CHECKS.append({"check": name, "passed": bool(ok), "evidence": evidence})


def rd(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def close(a, b, rtol=1e-12) -> bool:
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b or abs(a - b) <= rtol * max(abs(a), abs(b))


def seed_for(seed: int, *parts: str) -> int:
    return (int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) ^ seed) & 0x7FFFFFFF


def main(run_dir: Path) -> int:
    man = json.loads((run_dir / "manifest.json").read_text())
    cfg = man["benchmark_config"]
    modes = sorted({p.name.split("_")[1] for p in run_dir.glob("trajectory_*")})
    methods = cfg["methods"]["trajectory"]
    reps = cfg["repeats"]["timed"]
    n_q = len(man["population_used"])

    out = {m: rd(run_dir / f"trajectory_{m}" / "outcomes.csv") for m in modes}
    pout = {m: rd(run_dir / f"terminal_price_{m}" / "outcomes.csv") + rd(run_dir / f"terminal_price_{m}_cos" / "outcomes.csv")
            for m in modes}
    forced = rd(run_dir / "forced_failures" / "outcomes.csv")
    timings = {m: rd(run_dir / f"trajectory_{m}" / "raw_timings.csv") for m in modes}

    # ---- counts
    for m in modes:
        check(f"row count trajectory_{m}", len(out[m]) == len(methods) * n_q * reps,
              {"rows": len(out[m]), "expected": len(methods) * n_q * reps})
        check(f"row count terminal_price_{m}", len(pout[m]) == (len(methods) + 1) * n_q * reps,
              {"rows": len(pout[m]), "expected": (len(methods) + 1) * n_q * reps})
        check(f"timings rows match outcomes trajectory_{m}", len(timings[m]) == len(out[m]),
              {"timings": len(timings[m]), "outcomes": len(out[m])})
        keys = {(r["solver_method"], r["query_id"], r["repetition"]) for r in out[m]}
        check(f"every (method, query, rep) attempted once, trajectory_{m}", len(keys) == len(out[m]), len(keys))

    # ---- independent query-level medians
    groups = defaultdict(list)
    for m in modes:
        for r in out[m] + pout[m]:
            groups[(r["workload"], r["cache_mode"], r["solver_method"], r["query_id"])].append(r)
    ql = {}
    for k, rs in groups.items():
        sts = {r["final_status"] for r in rs}
        ts = {r["target_status"] for r in rs}
        ql[k] = dict(t=statistics.median([float(r["total_time_s"]) for r in rs]), n=len(rs),
                     fs=sts.pop() if len(sts) == 1 else "INCONSISTENT_ACROSS_REPS",
                     ts=ts.pop() if len(ts) == 1 else "INCONSISTENT_ACROSS_REPS")
    h_ql = rd(run_dir / "query_level.csv")
    mism = [r for r in h_ql if not (close(r["median_total_time_s"], ql[(r["workload"], r["cache_mode"], r["solver_method"], r["query_id"])]["t"])
                                    and r["final_status"] == ql[(r["workload"], r["cache_mode"], r["solver_method"], r["query_id"])]["fs"]
                                    and r["target_status"] == ql[(r["workload"], r["cache_mode"], r["solver_method"], r["query_id"])]["ts"]
                                    and int(r["n_repetitions"]) == ql[(r["workload"], r["cache_mode"], r["solver_method"], r["query_id"])]["n"])]
    check("query_level.csv reproduced from raw outcomes", len(h_ql) == len(ql) and not mism,
          {"rows": len(h_ql), "rebuilt": len(ql), "mismatches": mism[:3]})
    check("statuses consistent across repetitions", all(v["fs"] != "INCONSISTENT_ACROSS_REPS" for v in ql.values()),
          sum(v["fs"] == "INCONSISTENT_ACROSS_REPS" for v in ql.values()))

    # ---- independent population counts
    pop = defaultdict(lambda: defaultdict(int))
    times = defaultdict(list)
    for (wl, mode, m, q), v in ql.items():
        pop[(wl, mode, m)]["n"] += 1
        pop[(wl, mode, m)]["s:" + v["fs"]] += 1
        pop[(wl, mode, m)]["t:" + v["ts"]] += 1
        times[(wl, mode, m)].append(v["t"])
    bad = []
    for r in rd(run_dir / "population_summary.csv"):
        k = (r["workload"], r["cache_mode"], r["solver_method"])
        p = pop[k]
        exp = {"n_queries_attempted": p["n"], **{f"n_{s}": p["s:" + s] for s in
               ("direct_accept", "corrected_accept", "qualified_fallback", "over_budget", "uncertifiable",
                "numerical_failure", "delivered_no_certificate")},
               **{f"n_target_{t}": p["t:" + t] for t in ("verified", "empirically_checked", "failed", "unavailable")}}
        for col, val in exp.items():
            if int(r[col]) != val:
                bad.append((k, col, r[col], val))
        if not close(r["median_query_time_s"], statistics.median(times[k])):
            bad.append((k, "median_query_time_s"))
    check("population_summary.csv counts and medians reproduced", not bad, bad[:5])

    # ---- independent paired ratios
    bad = []
    hr = rd(run_dir / "paired_ratios.csv")
    for r in hr:
        wl, mode, m, b = r["workload"], r["cache_mode"], r["method"], r["baseline"]
        ok = {"verified"} if wl == "trajectory" else {"verified", "empirically_checked"}
        qids = sorted({k[3] for k in ql if k[0] == wl and k[1] == mode})
        common = [q for q in qids if ql[(wl, mode, m, q)]["ts"] in ok and ql[(wl, mode, b, q)]["ts"] in ok]
        if int(r["n_common_qualifying"]) != len(common):
            bad.append((wl, mode, m, b, "n_common"))
            continue
        if not common:
            continue
        lr = np.log([ql[(wl, mode, m, q)]["t"] / ql[(wl, mode, b, q)]["t"] for q in common])
        rng = np.random.default_rng(seed_for(cfg["seed"], wl, mode, m, b))
        B, c = cfg["bootstrap"]["resamples"], cfg["bootstrap"]["confidence"]
        bs = [lr[rng.integers(0, len(lr), len(lr))].mean() for _ in range(B)]
        exp = (math.exp(lr.mean()), math.exp(np.quantile(bs, (1 - c) / 2)), math.exp(np.quantile(bs, (1 + c) / 2)),
               statistics.median(np.exp(lr)))
        got = (r["geomean_time_ratio"], r["geomean_ci_low"], r["geomean_ci_high"], r["median_time_ratio"])
        if not all(close(g, e, 1e-10) for g, e in zip(got, exp)):
            bad.append((wl, mode, m, b, got, exp))
    check("paired_ratios.csv reproduced (common population, bootstrap over queries)", not bad and hr, bad[:3])

    # ---- amortisation
    am = rd(run_dir / "amortised.csv")
    setup = rd(run_dir / "setup" / "one_time_setup.csv")
    c0: dict = {}
    for r in setup:  # first session's template time (a resumed session re-times, it does not replace)
        if r["item"].startswith("template:"):
            c0.setdefault(r["item"].split(":", 1)[1], float(r["seconds"]))
    bad = []
    for r in am:
        m = r["method"]
        c1 = float(np.mean([v["t"] for k, v in ql.items() if k[:3] == ("trajectory", "reuse", m)]))
        if not (close(r["online_cost_per_query_s"], c1) and close(r["one_time_setup_s"], c0[m])):
            bad.append(m)
        for Q in cfg["amortisation_population_sizes"]:
            if not close(r[f"amortised_per_query_s_Q{Q}"], c0[m] / Q + c1):
                bad.append((m, Q))
        for b in cfg["ratio_baselines"]["trajectory"]:
            if b == m:
                continue
            qids = sorted(k[3] for k in ql if k[:3] == ("trajectory", "reuse", m))
            vm = {q: ql[("trajectory", "reuse", m, q)]["ts"] == "verified" for q in qids}
            vb = {q: ql[("trajectory", "reuse", b, q)]["ts"] == "verified" for q in qids}
            common = [q for q in qids if vm[q] and vb[q]]
            got = r[f"break_even_Q_vs_{b}"]
            if not common or sum(vm.values()) < sum(vb.values()):
                if not got.startswith("none"):
                    bad.append((m, b, "break-even reported without verified coverage", got))
                continue
            d = np.array([ql[("trajectory", "reuse", b, q)]["t"] - ql[("trajectory", "reuse", m, q)]["t"] for q in common])
            rng = np.random.default_rng(seed_for(cfg["seed"], "amortised", m, b))
            lo = float(np.quantile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(cfg["bootstrap"]["resamples"])],
                                   (1 - cfg["bootstrap"]["confidence"]) / 2))
            if d.mean() > 0 and lo > 0:
                exp = 1 if c0[m] <= c0[b] else math.ceil((c0[m] - c0[b]) / d.mean())
                if str(exp) != got:
                    bad.append((m, b, got, exp))
            elif got != "none":
                bad.append((m, b, got, "none"))
    check("amortised.csv reproduced; break-even reported only with C1 advantage + CI", not bad and (am or "reuse" not in modes), bad)

    # ---- G4 item: repaired POD / free method loaded (source trace)
    lm = man["loaded_models"]
    pod_rows = [r for m in modes for r in out[m] if r["solver_method"].startswith("pod")]
    wf_cfg = yaml.safe_load((REPO_ROOT / "configs" / "v5_free_workflow.yaml").read_text())
    guard = json.loads((run_dir / "legacy_guard.json").read_text())
    audited = {k: v for k, v in guard["calls"].items() if "ops_class" in k or "FreeOperatorCache" in k or "OperatorCache [" in k}
    check("G4.1 repaired mass-weighted POD and free method loaded; no legacy call",
          lm["pod_basis_type"] == "mass_weighted_pod" and lm["legacy_pod_loaded"] is False and lm["pod_class"] == "FreePODBasis"
          and all(r["basis_id"].startswith("mass_weighted_pod:") and r["basis_id"].endswith(wf_cfg["pod"]["artifact_sha256"][:12]) for r in pod_rows)
          and all(r["method_id"] == "constrained_free_cn_v1" for m in modes for r in out[m])
          and guard["blocked"] == [] and all("FreeOperatorCache, dim=62" in k for k in guard["calls"] if "certification." in k),
          {"loaded_models": lm, "pod_rows": len(pod_rows), "guard_blocked": guard["blocked"], "guard_calls": guard["calls"]})

    # ---- G4 item: common target meanings within each workload
    tt = defaultdict(set)
    for m in modes:
        for r in out[m] + pout[m]:
            tt[r["workload"]].add((r["target_type"], r["tolerance"]))
    check("G4.2 one target type/tolerance per workload", all(len(v) == 1 for v in tt.values()),
          {k: sorted(v) for k, v in tt.items()})

    # ---- G4 item: no success when bound exceeds target
    allrows = [r for m in modes for r in out[m]] + forced
    viol = [r for r in allrows if (r["target_status"] == "verified" and not float(r["augmented_bound"]) <= float(r["tolerance"]))
            or (r["final_status"] in ("direct_accept", "corrected_accept", "qualified_fallback")
                and not float(r["augmented_bound"]) <= float(r["tolerance"]))
            or (r["output_delivered"] == "True" and r["final_status"] not in ("direct_accept", "corrected_accept", "qualified_fallback"))]
    pviol = [r for m in modes for r in pout[m] if r["target_status"] == "verified"
             and not float(r["price_bound_dollars"]) <= float(r["tolerance"])]
    rej = [dict(query=r["query_id"], mode=r["cache_mode"], rep=r["repetition"], bound=r["augmented_bound"],
                tol=r["tolerance"], status=r["final_status"], delivered=r["output_delivered"], target=r["target_status"])
           for r in allrows if r["solver_method"] == "neural_direct_riesz" and float(r["augmented_bound"]) > float(r["tolerance"])]
    check("G4.3 direct neural certification cannot succeed with bound > target",
          not viol and not pviol and rej and all(x["status"] == "over_budget" and x["delivered"] == "False" for x in rej),
          {"violations": len(viol) + len(pviol), "rejected_direct_cases": len(rej), "examples": rej[:3]})
    cos_claims = [r for m in modes for r in pout[m] if r["solver_method"] == "cos" and r["target_status"] == "verified"]
    check("G4.3b COS/reference never verified merely by returning output", not cos_claims, len(cos_claims))

    # ---- G4 item: forced failures
    fz = [dict(s=r["scenario"], q=r["query_id"], got=r["final_status"], exp=r["expected_status"],
               fallback_ran=r["fallback_verified_ran"], exc=r["exception"][:60]) for r in forced]
    if man["studies"].get("g4_smoke") == "completed":
        check("G4.4 forced failures reach explicit statuses; fallback verified to have run",
              fz and all(x["got"] == x["exp"] for x in fz)
              and all(x["fallback_ran"] == "True" for x in fz if x["got"] == "qualified_fallback"), fz)
    else:  # forced scenarios are a smoke-run (G4) feature; a full run must not contain any
        check("G4.4 (full run) no forced-failure rows mixed into a population run", not forced,
              {"forced_rows": len(forced), "note": "forced scenarios exercised in the G4 smoke run"})

    # ---- G4 item: cache isolation
    iso = {}
    for m in modes:
        rows = out[m]
        serials = [r["cache_serial"] for r in rows]
        iso[m] = dict(rows=len(rows), unique_serials=len(set(serials)),
                      toeplitz_nonempty_at_start=sum(r["toeplitz_entries_at_start"] != "0" for r in rows),
                      mass_factor_prebuilt=sorted({r["mass_I_factor_prebuilt"] for r in rows}))
    ok = all(v["unique_serials"] == v["rows"] and v["toeplitz_nonempty_at_start"] == 0 for v in iso.values())
    if "cold" in iso:
        ok &= iso["cold"]["mass_factor_prebuilt"] == ["False"]
    if "reuse" in iso:
        ok &= iso["reuse"]["mass_factor_prebuilt"] == ["True"] and {f"template:{m}" for m in methods} <= {r["item"] for r in setup}
    check("G4.5 cache isolation (cold: fresh caches, nothing prebuilt; reuse: only mesh-only objects shared)", ok, iso)
    setup_t = {}
    for m in modes:
        ph = [json.loads(r["phases_json"]) for r in timings[m]]
        setup_t[m] = dict(rows_with_phase=sum("setup_operator_cache" in x for x in ph), rows=len(ph),
                          median_setup_operator_cache_s=float(np.median([x.get("setup_operator_cache", float("nan")) for x in ph])))
    ok = all(v["rows_with_phase"] == v["rows"] for v in setup_t.values())
    if "cold" in setup_t and "reuse" in setup_t:
        ok &= setup_t["cold"]["median_setup_operator_cache_s"] > 10 * setup_t["reuse"]["median_setup_operator_cache_s"]
    check("G4.5b cold cache construction is charged inside every method timer", ok, setup_t)

    (run_dir / "g4_checks.json").write_text(json.dumps(CHECKS, indent=2, default=str) + "\n")
    failed = [c["check"] for c in CHECKS if not c["passed"]]
    for c in CHECKS:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
