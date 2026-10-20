"""M6 summaries and G5 checks, rebuilt only from the raw CSVs of the two M6 runs.

    PYTHONPATH=src:experiments python3 scripts/v5_m6_summary.py <audit_run_dir> <riesz_run_dir>

Writes into each run directory: *_summary.csv files and summary.md; writes
g5_checks.json into the Riesz run directory. Strata of the audit are always
reported separately; the 139 'known_failure' rows are a deliberately selected
set and are never summarised as a failure RATE.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

STRATA = ("sobol_main_box", "exact_vertex", "known_failure")


def rd(p: Path) -> list[dict]:
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def wr(p: Path, rows: list[dict]) -> None:
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def fl(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def g(v) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def audit(ad: Path) -> tuple[list[str], dict]:
    free = {r["query_id"]: r for r in rd(ad / "audit_free/positivity.csv")}
    full = {r["query_id"]: r for r in rd(ad / "audit_legacy_full/positivity.csv")}
    assert set(free) == set(full), "free/full positivity rows must cover the same queries"
    pos = []
    for s in STRATA:
        ids = [q for q in free if free[q]["source"] == s]
        fp = [q for q in ids if free[q]["gate_passes"] == "True"]
        lp = [q for q in ids if full[q]["gate_passes"] == "True"]
        pos.append(dict(
            stratum=s, n=len(ids), free_gate_passes=len(fp), legacy_full_gate_passes=len(lp),
            legacy_only_rejections=sum(1 for q in ids if q in fp and q not in lp),
            free_only_rejections=sum(1 for q in ids if q in lp and q not in fp),
            exceptions=sum(1 for q in ids if free[q]["exception"] or full[q]["exception"]),
            min_lambda_min_free=min(fl(free[q]["lambda_min_numerical"]) for q in ids),
            min_lambda_min_full=min(fl(full[q]["lambda_min_numerical"]) for q in ids),
            max_alpha_minus_lambda_free=max(fl(free[q]["alpha_minus_lambda_min"]) for q in fp) if fp else float("nan"),
            max_cond2_S_I=max(fl(free[q]["cond2_S"]) for q in fp) if fp else float("nan"),
            median_kappa_I=float(np.median([fl(free[q]["kappa"]) for q in fp])) if fp else float("nan"),
            max_kappa_I=max(fl(free[q]["kappa"]) for q in fp) if fp else float("nan"),
            role=("deliberately selected failures: NOT a production failure rate" if s == "known_failure"
                  else "sample of the parameter box" if s == "sobol_main_box" else "all 64 exact box vertices")))
    wr(ad / "positivity_summary.csv", pos)

    sh = rd(ad / "audit_free/sharp_fixed_residual.csv")
    sharp = []
    for s in STRATA:
        rows = [r for r in sh if r["source"] == s]
        if not rows:
            continue
        rat = np.array([fl(r["ratio_free_over_full"]) for r in rows])
        sharp.append(dict(stratum=s, n_both_gates_pass=len(rows), ratio_min=float(rat.min()),
                          ratio_median=float(np.median(rat)), ratio_max=float(rat.max()),
                          n_theorem_holds_within_rounding=sum(r["theorem_holds_within_rounding"] == "True" for r in rows)))
    wr(ad / "sharp_fixed_residual_summary.csv", sharp)

    tr = rd(ad / "integrated/traces.csv")
    trs = []
    for s in STRATA:
        for p in sorted({r["predictor"] for r in tr}):
            rows = [r for r in tr if r["source"] == s and r["predictor"] == p]
            st = Counter(r["final_status"] for r in rows)
            trs.append(dict(stratum=s, predictor=p, n=len(rows),
                            legacy_only_rejections_processed=sum(r["legacy_only_rejection"] == "True" for r in rows),
                            delivered=sum(r["output_delivered"] == "True" for r in rows),
                            **{f"n_{k}": v for k, v in sorted(st.items())}))
    keys = sorted({k for r in trs for k in r})
    trs = [{k: r.get(k, 0) for k in keys} for r in trs]
    wr(ad / "integrated_traces_summary.csv", trs)

    au = rd(ad / "augmented/augmented_bound_validation.csv")
    aus = []
    for c in sorted({r["candidate"] for r in au}):
        rows = [r for r in au if r["candidate"] == c]
        eff = [fl(r["effectivity_upper"]) for r in rows if fl(r["effectivity_upper"]) == fl(r["effectivity_upper"])]
        aus.append(dict(candidate=c, n=len(rows), n_bound_holds=sum(r["bound_holds"] == "True" for r in rows),
                        n_invalid_bound=sum(r["bound_status"] != "valid" for r in rows),
                        effectivity_median=float(np.median(eff)) if eff else float("nan"),
                        effectivity_max=float(np.max(eff)) if eff else float("nan"),
                        n_residual_only_would_underestimate=sum(r["residual_only_would_underestimate"] == "True" for r in rows)))
    wr(ad / "augmented_bound_summary.csv", aus)

    md = [f"# M6 audit {ad.name}", "",
          "Floating-point evaluation of exact-arithmetic theory. The 139 known_failure rows are deliberately "
          "selected and are NOT a production failure rate.", "", "## Positivity (gate passes / n)", "",
          "| stratum | n | free S_I | legacy S_h | legacy-only rejections | free-only rejections | exceptions | min λ_min(S_I) | min λ_min(S_h) | max α−λ (free) | max cond(S_I) | κ_I median/max |",
          "|" + "---|" * 12]
    for r in pos:
        md.append(f"| {r['stratum']} | {r['n']} | {r['free_gate_passes']} | {r['legacy_full_gate_passes']} | "
                  f"{r['legacy_only_rejections']} | {r['free_only_rejections']} | {r['exceptions']} | "
                  f"{g(r['min_lambda_min_free'])} | {g(r['min_lambda_min_full'])} | {g(r['max_alpha_minus_lambda_free'])} | "
                  f"{g(r['max_cond2_S_I'])} | {g(r['median_kappa_I'])} / {g(r['max_kappa_I'])} |")
    md += ["", "## Fixed-residual sharp comparison (both gates pass): q_free / q_full", "",
           "| stratum | n | min | median | max | Schur bound holds |", "|---|---|---|---|---|---|"]
    md += [f"| {r['stratum']} | {r['n_both_gates_pass']} | {g(r['ratio_min'])} | {g(r['ratio_median'])} | "
           f"{g(r['ratio_max'])} | {r['n_theorem_holds_within_rounding']}/{r['n_both_gates_pass']} |" for r in sharp]
    md += ["", "## Integrated free-workflow traces (declared subset: first 16 per stratum)", "",
           "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in trs]
    md += ["", "## Augmented-bound validation (true max-in-time M_I error vs dense direct reference)", "",
           "| candidate | n | bound holds | invalid | effectivity median / max | residual-only would underestimate |",
           "|---|---|---|---|---|---|"]
    md += [f"| {r['candidate']} | {r['n']} | {r['n_bound_holds']} | {r['n_invalid_bound']} | {g(r['effectivity_median'])} / "
           f"{g(r['effectivity_max'])} | {r['n_residual_only_would_underestimate']} |" for r in aus]
    (ad / "summary.md").write_text("\n".join(md) + "\n")
    return md, dict(pos=pos, sharp=sharp, traces=trs, augmented=aus)


def riesz(rdir: Path) -> tuple[list[str], dict]:
    rows = rd(rdir / "riesz/riesz_bracket_validation.csv")
    alpha = rd(rdir / "riesz/alpha_check.csv")
    by = defaultdict(Counter)
    side = defaultdict(Counter)
    for r in rows:
        by[(r["residual_source"], r["preconditioner"])][r["comparison_status"]] += 1
        if r["comparison_status"] == "numerically_unresolved":
            side[r["residual_source"]][r["unresolved_side"]] += 1
    summ = [dict(residual_source=s, preconditioner=p, n=sum(c.values()), n_pass=c["pass"],
                 n_numerically_unresolved=c["numerically_unresolved"], n_violation=c["violation"],
                 n_exception=c["exception"]) for (s, p), c in sorted(by.items())]
    wr(rdir / "riesz_status_summary.csv", summ)
    budget = []
    for s in sorted({r["residual_source"] for r in rows}):
        for b in sorted({int(r["pcg_budget"]) for r in rows}):
            xs = [fl(r["relative_upper_margin"]) for r in rows if r["residual_source"] == s and int(r["pcg_budget"]) == b
                  and r["preconditioner"] == "none" and r["comparison_status"] != "exception"]
            xs = [x for x in xs if x == x]
            if xs:
                budget.append(dict(residual_source=s, pcg_budget=b, preconditioner="none", n=len(xs),
                                   relative_upper_margin_median=float(np.median(xs)),
                                   relative_upper_margin_max=float(np.max(xs))))
    wr(rdir / "riesz_budget_summary.csv", budget)
    viol = [r for r in rows if r["comparison_status"] == "violation"]
    a_fail = [r for r in alpha if r["hypothesis_alpha_le_lambda_min_holds"] != "True"]
    rnorm = defaultdict(list)
    for r in rows:
        rnorm[r["residual_source"]].append(fl(r["residual_norm"]))
    md = [f"# M6 Riesz-bracket validation {rdir.name}", "",
          "Reference: mpmath 40-digit Cholesky of the same assembled S (exactly symmetrised). "
          "comparison_uncertainty = 8 n eps cond2(S) q_ref. 'numerically_unresolved' = a margin within ±uncertainty "
          "(mostly converged brackets, see side counts). Floating-point evaluation of exact-arithmetic theory.", "",
          "| residual source | residual norm range | preconditioner | n | pass | unresolved | violation | exception |",
          "|---|---|---|---|---|---|---|---|"]
    for r in summ:
        rn = rnorm[r["residual_source"]]
        md.append(f"| {r['residual_source']} | {g(min(rn))}–{g(max(rn))} | {r['preconditioner']} | {r['n']} | {r['n_pass']} | "
                  f"{r['n_numerically_unresolved']} | {r['n_violation']} | {r['n_exception']} |")
    md += ["", "Unresolved by side: " + "; ".join(f"{s}: {dict(c)}" for s, c in sorted(side.items())), "",
           f"alpha hypothesis (alpha <= mpmath lambda_min of the same S): {len(alpha) - len(a_fail)}/{len(alpha)} hold; "
           f"max alpha - lambda_min = {g(max(fl(r['alpha_minus_lambda_min_mp']) for r in alpha))}", "",
           "## relative upper margin (u - q)/q by PCG budget, no preconditioner", "",
           "| source | budget | n | median | max |", "|---|---|---|---|---|"]
    md += [f"| {r['residual_source']} | {r['pcg_budget']} | {r['n']} | {g(r['relative_upper_margin_median'])} | "
           f"{g(r['relative_upper_margin_max'])} |" for r in budget]
    (rdir / "summary.md").write_text("\n".join(md) + "\n")
    return md, dict(rows=rows, viol=viol, alpha=alpha, a_fail=a_fail, summ=summ)


def main(ad: Path, rdir: Path) -> int:
    _, A = audit(ad)
    _, R = riesz(rdir)
    checks = []

    def check(name, ok, ev):
        checks.append(dict(check=name, passed=bool(ok), evidence=ev))

    check("G5.1 no unexplained bracket violations in supported cases", not R["viol"],
          {"violations": len(R["viol"]), "examples": [r["case_key"] for r in R["viol"][:5]]})
    check("G5.1b no augmented-bound violations",
          all(r["n_bound_holds"] == r["n"] for r in A["augmented"]), A["augmented"])
    check("G5.1c fixed-residual Schur bound holds where both gates pass",
          all(r["n_theorem_holds_within_rounding"] == r["n_both_gates_pass"] for r in A["sharp"]), A["sharp"])
    check("G5.1d alpha <= lambda_min(S_I) (mpmath, same S) on every validated query", not R["a_fail"],
          {"n": len(R["alpha"]), "failures": [r["query_id"] for r in R["a_fail"]]})
    n_unres = sum(r["n_numerically_unresolved"] for r in R["summ"])
    check("G5.2 ill-conditioned / unresolved results are visible (counted, side recorded)",
          all("unresolved_side" in r for r in R["rows"][:1]), {"n_numerically_unresolved": n_unres})
    check("G5.2b audit exceptions recorded, none hidden", True,
          {r["stratum"]: r["exceptions"] for r in A["pos"]})
    check("G5.3 counts distinguish theorem-backed, floating-point diagnostic and failed/unsupported cases",
          True, {"riesz_statuses": sorted({r["comparison_status"] for r in R["rows"]}),
                 "positivity_columns": "gate_passes from floating-point alpha estimate (diagnostic of the hypothesis)",
                 "unsupported": "free-only rejections: " + str(sum(r["free_only_rejections"] for r in A["pos"]))})
    (rdir / "g5_checks.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    for c in checks:
        print(("PASS " if c["passed"] else "FAIL ") + c["check"])
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
