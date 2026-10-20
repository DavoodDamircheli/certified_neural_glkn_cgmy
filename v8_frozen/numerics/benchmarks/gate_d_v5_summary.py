"""Aggregation for the gate_d_v5 harness (roadmap §5.4). Reads ONLY the raw
CSVs of a run directory and writes query_level.csv, population_summary.csv,
paired_ratios.csv, amortised.csv and summary.md.

Rules: medians over timed repetitions within each (workload, mode, method,
query) first; population statistics over queries; paired ratios only on the
common qualifying population of the two methods (with coverage reported);
bootstrap over queries (never repetitions/spots/steps). An independent
re-implementation (scripts/v5_bench_rebuild.py) recomputes all of this and
asserts agreement.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

STATUSES = ("direct_accept", "corrected_accept", "qualified_fallback", "over_budget", "uncertifiable",
            "numerical_failure", "delivered_no_certificate")
TARGET_STATUSES = ("verified", "empirically_checked", "failed", "unavailable")


def target_status(delivered: bool, verified: bool, empirically_within) -> str:
    if not delivered:
        return "unavailable"
    if verified:
        return "verified"
    if empirically_within is True:
        return "empirically_checked"
    return "failed"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def load_outcomes(run_dir: Path, modes: list[str]) -> list[dict]:
    rows = []
    for mode in modes:
        for g in (f"trajectory_{mode}", f"terminal_price_{mode}", f"terminal_price_{mode}_cos"):
            rows += _read(run_dir / g / "outcomes.csv")
    return rows


def bootstrap_seed(seed: int, *parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return (int(h[:8], 16) ^ seed) & 0x7FFFFFFF


def geomean_ci(ratios: np.ndarray, seed: int, B: int, conf: float) -> tuple[float, float, float]:
    logs = np.log(ratios)
    gm = float(np.exp(np.mean(logs)))
    rng = np.random.default_rng(seed)
    n = len(logs)
    boots = np.array([np.mean(logs[rng.integers(0, n, n)]) for _ in range(B)])
    lo, hi = np.quantile(boots, [(1 - conf) / 2, (1 + conf) / 2])
    return gm, float(np.exp(lo)), float(np.exp(hi))


def query_level(outcomes: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in outcomes:
        if r.get("scenario"):
            continue
        groups[(r["workload"], r["cache_mode"], r["solver_method"], r["query_id"])].append(r)
    out = []
    for (wl, mode, m, qid), rs in sorted(groups.items()):
        statuses = {r["final_status"] for r in rs}
        tstats = {r["target_status"] for r in rs}
        out.append(dict(workload=wl, cache_mode=mode, solver_method=m, query_id=qid, n_repetitions=len(rs),
                        median_total_time_s=float(np.median([float(r["total_time_s"]) for r in rs])),
                        final_status=statuses.pop() if len(statuses) == 1 else "INCONSISTENT_ACROSS_REPS",
                        target_status=tstats.pop() if len(tstats) == 1 else "INCONSISTENT_ACROSS_REPS",
                        output_delivered=rs[0]["output_delivered"],
                        augmented_bound=rs[0]["augmented_bound"]))
    return out


def population_summary(ql: list[dict], accuracy: dict) -> list[dict]:
    groups = defaultdict(list)
    for r in ql:
        groups[(r["workload"], r["cache_mode"], r["solver_method"])].append(r)
    out = []
    for (wl, mode, m), rs in sorted(groups.items()):
        row = dict(workload=wl, cache_mode=mode, solver_method=m, n_queries_attempted=len(rs))
        for s in STATUSES:
            row[f"n_{s}"] = sum(r["final_status"] == s for r in rs)
        row["n_inconsistent_across_reps"] = sum(r["final_status"].startswith("INCONSISTENT") for r in rs)
        for t in TARGET_STATUSES:
            row[f"n_target_{t}"] = sum(r["target_status"] == t for r in rs)
        row["n_output_delivered"] = sum(r["output_delivered"] == "True" for r in rs)
        row["median_query_time_s"] = float(np.median([r["median_total_time_s"] for r in rs]))
        acc = [accuracy[(wl, mode, m, r["query_id"])] for r in rs
               if r["output_delivered"] == "True" and (wl, mode, m, r["query_id"]) in accuracy]
        acc = [a for a in acc if not math.isnan(a)]
        row["delivered_error_vs_stored_reference_median"] = float(np.median(acc)) if acc else float("nan")
        row["delivered_error_vs_stored_reference_max"] = float(np.max(acc)) if acc else float("nan")
        out.append(row)
    return out


def load_accuracy(run_dir: Path, modes: list[str]) -> dict:
    acc = {}
    for mode in modes:
        for wl, g, col in (("trajectory", f"trajectory_{mode}", "max_mass_error_vs_stored_reference"),
                           ("terminal_price", f"terminal_price_{mode}", "max_abs_price_error_vs_stored_reference"),
                           ("terminal_price", f"terminal_price_{mode}_cos", "max_abs_price_error_vs_stored_reference")):
            for r in _read(run_dir / g / "accuracy.csv"):
                if r["scenario"] or r["repetition"] != "0":
                    continue  # deterministic outputs: repetition 0 is representative
                acc[(wl, mode, r["solver_method"], r["query_id"])] = float(r[col])
    return acc


def paired_ratios(ql: list[dict], cfg: dict) -> list[dict]:
    seed = cfg["seed"]
    B, conf = cfg["bootstrap"]["resamples"], cfg["bootstrap"]["confidence"]
    idx = {(r["workload"], r["cache_mode"], r["solver_method"], r["query_id"]): r for r in ql}
    out = []
    for wl, baselines in cfg["ratio_baselines"].items():
        rule = ("both target_status == verified" if wl == "trajectory"
                else "both empirically within budget (target_status in {verified, empirically_checked})")
        ok = ({"verified"} if wl == "trajectory" else {"verified", "empirically_checked"})
        modes = sorted({r["cache_mode"] for r in ql if r["workload"] == wl})
        for mode in modes:
            methods = sorted({r["solver_method"] for r in ql if r["workload"] == wl and r["cache_mode"] == mode})
            qids = sorted({r["query_id"] for r in ql if r["workload"] == wl and r["cache_mode"] == mode})
            for b in baselines:
                for m in methods:
                    if m == b:
                        continue
                    common = [q for q in qids if idx[(wl, mode, m, q)]["target_status"] in ok
                              and idx[(wl, mode, b, q)]["target_status"] in ok]
                    row = dict(workload=wl, cache_mode=mode, method=m, baseline=b, qualifying_rule=rule,
                               n_queries=len(qids), n_common_qualifying=len(common),
                               coverage_method=sum(idx[(wl, mode, m, q)]["target_status"] in ok for q in qids) / len(qids),
                               coverage_baseline=sum(idx[(wl, mode, b, q)]["target_status"] in ok for q in qids) / len(qids))
                    if common:
                        ratios = np.array([idx[(wl, mode, m, q)]["median_total_time_s"]
                                           / idx[(wl, mode, b, q)]["median_total_time_s"] for q in common])
                        gm, lo, hi = geomean_ci(ratios, bootstrap_seed(seed, wl, mode, m, b), B, conf)
                        row.update(median_time_ratio=float(np.median(ratios)), geomean_time_ratio=gm,
                                   geomean_ci_low=lo, geomean_ci_high=hi)
                    else:
                        row.update(median_time_ratio=float("nan"), geomean_time_ratio=float("nan"),
                                   geomean_ci_low=float("nan"), geomean_ci_high=float("nan"))
                    out.append(row)
    return out


def amortised(ql: list[dict], setup: list[dict], cfg: dict) -> list[dict]:
    """Reuse mode, trajectory workload: cost C0 + Q*C1 per method; C0 = the
    method's template build time (process-level model/basis load is common to
    all methods and reported separately); C1 = mean over queries of the
    query-median online time over ALL attempted queries. Break-even vs a baseline
    is evaluated only on the common population where BOTH outputs are target-
    verified, only if the method's verified coverage is not below the baseline's
    (a method that does not deliver cannot break even), and only if its C1 on that
    population is smaller with a paired bootstrap CI of the difference excluding 0."""
    seed, B, conf = cfg["seed"], cfg["bootstrap"]["resamples"], cfg["bootstrap"]["confidence"]
    rs = [r for r in ql if r["workload"] == "trajectory" and r["cache_mode"] == "reuse"]
    if not rs:
        return []
    c0: dict = {}
    for r in setup:  # first session's template time (a resumed session re-times, it does not replace)
        if r["item"].startswith("template:"):
            c0.setdefault(r["item"].split(":", 1)[1], float(r["seconds"]))
    by = defaultdict(dict)
    ver = defaultdict(dict)
    for r in rs:
        by[r["solver_method"]][r["query_id"]] = r["median_total_time_s"]
        ver[r["solver_method"]][r["query_id"]] = r["target_status"] == "verified"
    out = []
    for m in sorted(by):
        qids = sorted(by[m])
        c1 = float(np.mean([by[m][q] for q in qids]))
        row = dict(workload="trajectory", cache_mode="reuse", method=m, one_time_setup_s=c0.get(m, 0.0),
                   online_cost_per_query_s=c1)
        for Q in cfg["amortisation_population_sizes"]:
            row[f"amortised_per_query_s_Q{Q}"] = row["one_time_setup_s"] / Q + c1
        for b in cfg["ratio_baselines"]["trajectory"]:
            if b == m:
                row[f"break_even_Q_vs_{b}"] = "n/a"
                continue
            common = [q for q in qids if ver[m][q] and ver[b][q]]
            cov_m, cov_b = sum(ver[m].values()), sum(ver[b].values())
            if not common or cov_m < cov_b:
                row[f"break_even_Q_vs_{b}"] = f"none (verified {cov_m}/{len(qids)} vs {cov_b}/{len(qids)})"
                continue
            diff = np.array([by[b][q] - by[m][q] for q in common])
            rng = np.random.default_rng(bootstrap_seed(seed, "amortised", m, b))
            boots = np.array([np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(B)])
            lo = float(np.quantile(boots, (1 - conf) / 2))
            c0m, c0b = row["one_time_setup_s"], c0.get(b, 0.0)
            c1m_c = float(np.mean([by[m][q] for q in common]))
            c1b_c = float(np.mean([by[b][q] for q in common]))
            if c1m_c < c1b_c and lo > 0:
                row[f"break_even_Q_vs_{b}"] = 1 if c0m <= c0b else int(math.ceil((c0m - c0b) / (c1b_c - c1m_c)))
            else:
                row[f"break_even_Q_vs_{b}"] = "none"
        out.append(row)
    return out


def write_all(run_dir: Path, cfg: dict, modes: list[str]) -> dict:
    outcomes = load_outcomes(run_dir, modes)
    ql = query_level(outcomes)
    acc = load_accuracy(run_dir, modes)
    pop = population_summary(ql, acc)
    pr = paired_ratios(ql, cfg)
    setup = _read(run_dir / "setup" / "one_time_setup.csv")
    am = amortised(ql, setup, cfg)
    _write(run_dir / "query_level.csv", ql)
    _write(run_dir / "population_summary.csv", pop)
    _write(run_dir / "paired_ratios.csv", pr)
    _write(run_dir / "amortised.csv", am)

    def f(v):
        return f"{v:.3g}" if isinstance(v, float) else str(v)

    md = [f"# {run_dir.name}", "", "Two workloads, reported separately; never combined into one speed-up.", ""]
    for wl in ("trajectory", "terminal_price"):
        md += [f"## {wl}", "", "| mode | method | n | direct | corrected | fallback | over budget | uncertifiable | num. failure | no-cert delivered | verified | emp. checked | failed | unavailable | median s | delivered err med/max |",
               "|" + "---|" * 16]
        for r in pop:
            if r["workload"] != wl:
                continue
            md.append("| " + " | ".join(f(x) for x in (
                r["cache_mode"], r["solver_method"], r["n_queries_attempted"], r["n_direct_accept"],
                r["n_corrected_accept"], r["n_qualified_fallback"], r["n_over_budget"], r["n_uncertifiable"],
                r["n_numerical_failure"], r["n_delivered_no_certificate"], r["n_target_verified"],
                r["n_target_empirically_checked"], r["n_target_failed"], r["n_target_unavailable"],
                r["median_query_time_s"])) + f" | {f(r['delivered_error_vs_stored_reference_median'])} / {f(r['delivered_error_vs_stored_reference_max'])} |")
        md += ["", "| mode | method | baseline | common/total | coverage m / b | median ratio | geomean ratio [CI] |", "|---|---|---|---|---|---|---|"]
        for r in pr:
            if r["workload"] == wl:
                md.append(f"| {r['cache_mode']} | {r['method']} | {r['baseline']} | {r['n_common_qualifying']}/{r['n_queries']} | "
                          f"{f(r['coverage_method'])} / {f(r['coverage_baseline'])} | {f(r['median_time_ratio'])} | "
                          f"{f(r['geomean_time_ratio'])} [{f(r['geomean_ci_low'])}, {f(r['geomean_ci_high'])}] |")
        md.append("")
    if am:
        md += ["## reusable-setup amortisation (trajectory)", "", "| method | C0 s | C1 s/query | " + " | ".join(
            f"Q={q}" for q in cfg["amortisation_population_sizes"]) + " | " + " | ".join(
            f"break-even vs {b}" for b in cfg["ratio_baselines"]["trajectory"]) + " |",
            "|" + "---|" * (3 + len(cfg["amortisation_population_sizes"]) + len(cfg["ratio_baselines"]["trajectory"]))]
        for r in am:
            md.append("| " + " | ".join([r["method"], f(r["one_time_setup_s"]), f(r["online_cost_per_query_s"])]
                                        + [f(r[f"amortised_per_query_s_Q{q}"]) for q in cfg["amortisation_population_sizes"]]
                                        + [str(r[f"break_even_Q_vs_{b}"]) for b in cfg["ratio_baselines"]["trajectory"]]) + " |")
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    return dict(query_level=ql, population=pop, ratios=pr, amortised=am)
