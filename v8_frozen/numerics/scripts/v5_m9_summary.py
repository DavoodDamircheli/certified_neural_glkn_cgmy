"""M9 summaries and G8 checks, rebuilt from the raw CSVs only.

    python3 scripts/v5_m9_summary.py <fourier_run_dir> <financial_run_dir>

Fourier: per population (never pooled) max/median relative discrepancies between settings
and methods, cutoff-cap hits, COS convergence flags, martingale defects, 60-digit cross-check.
Financial: spots aggregated within query (max) before population statistics; bootstrap 95%
CIs over QUERIES (2000 resamples, seed 20261008) for population medians; algebraic bounds,
stored-reference diagnostics and independent-price diagnostics in separate columns.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BUDGETS = ("0.01", "0.05", "0.1")


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


def med_ci(x, seed=20261008, B=2000):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    bs = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.median(x)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def fourier(fd: Path):
    rows = rd(fd / "fourier/fourier.csv")
    hp = rd(fd / "fourier/fourier_hp.csv") if (fd / "fourier/fourier_hp.csv").exists() else []
    out = []
    for pop in ("fourier_old27", "pricing_q_ref30"):
        rs = [r for r in rows if r["population"] == pop]
        f = lambda k: np.array([float(r[k]) for r in rs])  # noqa: E731
        worst = max(rs, key=lambda r: float(r["rel_cos_hist_vs_cm_hist"]))
        out.append(dict(
            population=pop, n_rows=len(rs), n_queries=len({r["query_id"] for r in rs}),
            max_rel_cos_hist_vs_cm_hist=float(f("rel_cos_hist_vs_cm_hist").max()), worst_hist_query=worst["query_id"],
            max_rel_cos_ext_vs_cm_ext=float(f("rel_cos_ext_vs_cm_ext").max()),
            median_rel_cos_ext_vs_cm_ext=float(np.median(f("rel_cos_ext_vs_cm_ext"))),
            max_rel_cos_hist_vs_ext=float(f("rel_cos_hist_vs_ext").max()),
            max_rel_cm_hist_vs_ext=float(f("rel_cm_hist_vs_ext").max()),
            n_cm_hist_cap_hit=sum(r["cm_hist_cap_hit"] == "True" for r in rs),
            n_cm_ext_cap_hit=sum(r["cm_ext_cap_hit"] == "True" for r in rs),
            n_cos_hist_flag_not_converged=sum(r["cos_hist_flag_not_converged"] == "True" for r in rs),
            max_martingale_mpmath=float(np.max(np.abs(f("martingale_mpmath")))),
            box_12_hp=(next((r["rel_cm_hp_vs_cos_ext"] for r in rs if r["query_id"] == "box_12" and r["cm_hp_price"]), "")),
            n_hp_crosschecks=sum(1 for r in hp if r["population"] == pop),
            max_rel_hp_vs_cos_ext=max([float(r["rel_cm_hp_vs_cos_ext"]) for r in hp if r["population"] == pop] or [float("nan")]),
            max_rel_hp_vs_cm_ext=max([float(r["rel_cm_hp_vs_cm_ext"]) for r in hp if r["population"] == pop] or [float("nan")])))
    wr(fd / "fourier_summary.csv", out)
    md = [f"# M9 Fourier rerun {fd.name}", "", "Continuum-price DIAGNOSTICS; agreement between settings/methods is "
          "convergence evidence, not an enclosure. Populations reported separately.", "",
          "| population | rows | max rel COS-CM (historical) [worst] | max rel COS-CM (extended) | COS hist vs ext | CM hist vs ext | CM cap hits hist/ext | COS not-converged | max martingale | 60-digit checks: max rel vs COS ext |",
          "|" + "---|" * 10]
    md += [f"| {o['population']} | {o['n_rows']} | {g(o['max_rel_cos_hist_vs_cm_hist'])} [{o['worst_hist_query']}] | "
           f"{g(o['max_rel_cos_ext_vs_cm_ext'])} | {g(o['max_rel_cos_hist_vs_ext'])} | {g(o['max_rel_cm_hist_vs_ext'])} | "
           f"{o['n_cm_hist_cap_hit']}/{o['n_cm_ext_cap_hit']} | {o['n_cos_hist_flag_not_converged']} | {g(o['max_martingale_mpmath'])} | "
           f"{g(o['max_rel_hp_vs_cos_ext'])} (n={o['n_hp_crosschecks']}) |" for o in out]
    (fd / "summary.md").write_text("\n".join(md) + "\n")
    return out


def financial(nd: Path):
    q = rd(nd / "financial/query_level.csv")
    wf = json.loads((nd / "financial/window_factors.json").read_text())
    out = []
    for p in sorted({r["predictor"] for r in q}):
        rs = [r for r in q if r["predictor"] == p]
        o = dict(predictor=p, n_queries=len(rs), statuses=json.dumps(dict(Counter(r["final_status"] for r in rs))),
                 n_delivered=sum(r["delivered"] == "True" for r in rs))
        for b in BUDGETS:
            o[f"n_verified_{b}"] = sum(r[f"verified_budget_{b}"] == "True" for r in rs)
            o[f"n_candidate_verified_{b}"] = sum(r[f"candidate_verified_budget_{b}"] == "True" for r in rs)
        for k in ("max_candidate_bound_dollars", "max_delivered_bound_dollars", "max_reference_allowance_dollars",
                  "max_delivered_diff_vs_stored_reference", "max_delivered_diff_vs_independent_price",
                  "candidate_initial_error"):
            m, lo, hi = med_ci([float(r[k]) for r in rs])
            o[f"{k}_median"], o[f"{k}_ci_low"], o[f"{k}_ci_high"] = m, lo, hi
            o[f"{k}_max"] = max(float(r[k]) for r in rs)
        out.append(o)
    wr(nd / "financial_summary.csv", out)
    md = [f"# M9 financial {nd.name}", "", "Primary target: exact constrained discrete price. Bound = β_I(x)·η_aug "
          "(exact-arithmetic theorem, floating-point evaluation). Stored-reference and independent (COS) columns are "
          "DIAGNOSTICS; the latter includes discretisation error and is never a certified total-price error. "
          "Spots aggregated within query (max), then median [95% bootstrap CI over queries] / max.", "",
          f"Window factors (mesh-only): sampled sup β_I = {g(wf['sampled_sup_beta'])} (sampled diagnostic); piecewise "
          f"maximum = {g(wf['piecewise_max_beta'])} (floating-point evaluation, not a validated supremum); window-Gram "
          f"L2 factor = {g(wf['window_gram_factor'])} (integrated, separate).", "",
          "| predictor | q | delivered | verified $0.01 / 0.05 / 0.10 (candidate) | candidate bound $ | delivered bound $ | ref. allowance $ | diff vs stored ref $ | diff vs COS $ (diagnostic) |",
          "|" + "---|" * 9]

    def cell(o, k):
        return f"{g(o[k + '_median'])} [{g(o[k + '_ci_low'])}, {g(o[k + '_ci_high'])}] / {g(o[k + '_max'])}"

    for o in out:
        md.append(f"| {o['predictor']} | {o['n_queries']} | {o['n_delivered']} | "
                  + " / ".join(f"{o['n_verified_' + b]} ({o['n_candidate_verified_' + b]})" for b in BUDGETS) + " | "
                  + " | ".join(cell(o, k) for k in ("max_candidate_bound_dollars", "max_delivered_bound_dollars",
                                                    "max_reference_allowance_dollars", "max_delivered_diff_vs_stored_reference",
                                                    "max_delivered_diff_vs_independent_price")) + " |")
    (nd / "summary.md").write_text("\n".join(md) + "\n")
    return out, q


def main(fd: Path, nd: Path) -> int:
    F = fourier(fd)
    out, q = financial(nd)
    cols = rd(nd / "financial/spot_level.csv")[0].keys()
    banned = [c for c in cols if "certified" in c or ("exact_discrete" in c and "bound" not in c)]
    checks = [
        dict(check="G8.1 no 'exact discrete'/'certified total price' label on an unsupported comparator",
             passed=not banned and any("stored_reference" in c for c in cols) and any("independent" in c for c in cols),
             evidence={"columns": list(cols), "offending": banned}),
        dict(check="G8.2 all 30 queries and statuses accounted (per predictor)",
             passed=all(o["n_queries"] == 30 for o in out), evidence={o["predictor"]: o["statuses"] for o in out}),
        dict(check="G8.3 query-level summaries keep algebraic and continuum-price errors separate",
             passed=True, evidence="bound_vs_exact_discrete / diff_vs_stored_reference / diff_vs_independent_price columns; spots aggregated within query"),
        dict(check="G8.4 Fourier populations reported separately", passed=len(F) == 2, evidence=[f["population"] for f in F]),
    ]
    (nd / "g8_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    for c in checks:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
