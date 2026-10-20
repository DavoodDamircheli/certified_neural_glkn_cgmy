"""v4 Phase 5, items 1 and 2 -- CGMY_V4_REMAINING_WORK.md item 5.

Reruns ``financial/beta_eval.py``'s campaign with the v4 label fix (item 1:
``uncertifiable_fallback`` -> ``not_spd_uncertifiable``, since
``compute_stability_constants``'s ``is_spd`` depends only on mu, so no
fallback could ever have changed this query's outcome -- see
``financial/beta_eval.py``'s own comment above the renamed constant),
writing to a NEW location
(``results/v4_constrained_free/financial_v4_<timestamp>/``) -- the
historical ``financial/budget_sweep.csv``/``window_gram_bound.csv`` are
UNCHANGED (ground rule 2; ``beta_eval.main`` now takes an ``out_dir``
parameter specifically so this rerun never touches them).

Then (item 2: "treat the financial file as 20 queries x five spots, not
100 independent queries") aggregates ``budget_sweep.csv``'s 100 rows
(20 query_ids x 5 moneyness spots each) to the QUERY level BEFORE taking
any summary statistic -- max-over-spots for each of the three error types
kept separate throughout:
  - learned-vs-exact-discrete      (final_err_vs_discrete_CERTIFIED)
  - stored-discrete-vs-independent (discretization_gap_vs_cos_NOT_CERTIFIED)
  - learned-vs-independent         (total_err_vs_cos_NOT_CERTIFIED)
rather than averaging/bootstrapping over all 100 rows as if they were
independent samples -- the 100 rows already carry all three as separate
columns (this was NOT a missing column, see the module docstring of
``financial/beta_eval.py``'s row-construction code); what was missing is
this query-level-first aggregation step.

Run: PYTHONPATH=src:experiments:. python3 scripts/run_financial_v4.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT))

from financial import beta_eval  # noqa: E402

RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = REPO_ROOT / "results" / "v4_constrained_free" / f"financial_v4_{RUN_STAMP}"

ERROR_TYPES = {
    "learned_vs_exact_discrete": "final_err_vs_discrete_CERTIFIED",
    "stored_discrete_vs_independent": "discretization_gap_vs_cos_NOT_CERTIFIED",
    "learned_vs_independent": "total_err_vs_cos_NOT_CERTIFIED",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    beta_eval.main(out_dir=OUT_DIR)

    with open(OUT_DIR / "budget_sweep.csv") as f:
        rows = list(csv.DictReader(f))

    by_query: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_query[r["query_id"]].append(r)

    query_level_rows = []
    for qid, qrows in by_query.items():
        assert len(qrows) == 5, f"{qid} has {len(qrows)} spot rows, expected 5"
        statuses = {r["status"] for r in qrows}
        row = dict(query_id=qid, n_spots=len(qrows), statuses="|".join(sorted(statuses)))
        for name, col in ERROR_TYPES.items():
            vals = np.array([float(r[col]) for r in qrows], dtype=float)
            row[f"{name}_max_over_spots"] = float(np.nanmax(vals))
            row[f"{name}_mean_over_spots"] = float(np.nanmean(vals))
        query_level_rows.append(row)

    with open(OUT_DIR / "financial_v4_query_level_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(query_level_rows[0].keys()))
        w.writeheader()
        w.writerows(query_level_rows)

    print(f"\n=== Query-level summary (20 queries x 5 spots each, aggregated BEFORE any overall statistic) ===")
    for name in ERROR_TYPES:
        max_vals = [r[f"{name}_max_over_spots"] for r in query_level_rows]
        print(
            f"{name:34s} median-of-per-query-max={np.median(max_vals):.3e}  "
            f"max-of-per-query-max={np.max(max_vals):.3e}  (n_queries={len(query_level_rows)})"
        )
    print(f"\nwrote {OUT_DIR / 'financial_v4_query_level_summary.csv'}")


if __name__ == "__main__":
    main()
