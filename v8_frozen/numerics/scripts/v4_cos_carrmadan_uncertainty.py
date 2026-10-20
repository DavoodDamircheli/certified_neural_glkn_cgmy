"""v4 Phase 5, item 3: per-query COS truncation and Carr-Madan
integration-uncertainty checks -- CGMY_V4_REMAINING_WORK.md item 5's last
bullet ("Validate COS truncation and Carr-Madan integration uncertainty
per query. One cross-check hit its integration cutoff, and the maximum
relative difference is 2.34e-8, not uniformly 1e-14").

Granularity is per QUERY (20 of them, the same Q_ref30 prefix
``financial/beta_eval.py`` uses), not per (query, moneyness spot) -- the
task file's own wording distinguishes this from the 20x5 structure used
elsewhere in the financial report; truncation/integration uncertainty is a
property of mu (and S0=K here, ATM) via the characteristic function, not of
which of the 5 spots is being priced.

For each query:
  - ``validation.cos_pricer.converge_price`` -- a doubling (N, L) sweep;
    ``flagged_not_converged`` is already implemented there (not re-derived
    here) and is simply read off and reported per query.
  - ``validation.independent_fourier_integration.carr_madan_put_price`` --
    its own doubling cutoff search (``_find_cutoff``, already implemented
    there) for the FFT/quadrature's effective integration range
    ``v_cutoff``; flagged here if ``v_cutoff`` hit the hard cap
    (``max_v=400.0`` in that function, i.e. the search never found decay
    below tolerance within the allowed range) -- a NEW check, since
    ``carr_madan_put_price`` itself does not currently flag this.
  - Cross-check: ``abs(cos_price - carr_madan_price) / max(|cos_price|, 1)``
    as an independent-method agreement diagnostic (two different Fourier
    pricing methods, same mu/S0/K) -- NOT a certified bound, a floating-
    point cross-check between two numerical methods.

Run: PYTHONPATH=src:experiments:. python3 scripts/v4_cos_carrmadan_uncertainty.py
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT))

from validation.cgmy_reference_pricing import CGMYMarketParams  # noqa: E402
from validation.cos_pricer import cos_put_price, converge_price  # noqa: E402
from validation.independent_fourier_integration import carr_madan_put_price, _find_cutoff  # noqa: E402

N_QUERIES = 20  # same prefix financial/beta_eval.py uses
CARR_MADAN_MAX_V = 400.0  # must match independent_fourier_integration._find_cutoff's default

RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = REPO_ROOT / "results" / "v4_constrained_free" / f"cos_carrmadan_uncertainty_{RUN_STAMP}"


def main() -> None:
    queries = []
    with open(REPO_ROOT / "experiments" / "raw_results" / "query_sets" / "Q_ref30.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= N_QUERIES:
                break
            queries.append(row)

    K = 100.0
    S0 = K  # ATM, matching the per-query (not per-spot) granularity above

    rows = []
    t_start = time.time()
    for q in queries:
        mu = CGMYMarketParams(
            C=float(q["C"]), G=float(q["G"]), M=float(q["M"]),
            Y=float(q["Y"]), r=float(q["r"]), T=float(q["T"]),
        )

        conv = converge_price(mu, S0, K)
        cm = carr_madan_put_price(mu, S0, K)
        cos_default = cos_put_price(mu, S0, K)

        cm_hit_cutoff = cm["v_cutoff"] >= CARR_MADAN_MAX_V
        abs_diff = abs(cos_default["price"] - cm["price"])
        rel_diff = abs_diff / max(abs(cos_default["price"]), 1.0)

        rows.append(dict(
            query_id=q["query_id"], C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, T=mu.T,
            cos_price_default_NL=cos_default["price"],
            cos_price_converged=conv["price"], cos_N_used=conv["N_used"], cos_L_used=conv["L_used"],
            cos_flagged_not_converged=conv["flagged_not_converged"],
            carr_madan_price=cm["price"], carr_madan_v_cutoff=cm["v_cutoff"],
            carr_madan_hit_cutoff=cm_hit_cutoff, carr_madan_alpha=cm["alpha"],
            abs_diff_cos_vs_carrmadan=abs_diff, rel_diff_cos_vs_carrmadan=rel_diff,
        ))
        print(
            f"{q['query_id']}: cos={cos_default['price']:.6f} carr_madan={cm['price']:.6f} "
            f"rel_diff={rel_diff:.2e} v_cutoff={cm['v_cutoff']:.1f} "
            f"{'HIT_CUTOFF' if cm_hit_cutoff else ''} "
            f"{'COS_NOT_CONVERGED' if conv['flagged_not_converged'] else ''}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "cos_carrmadan_uncertainty.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_cutoff = sum(r["carr_madan_hit_cutoff"] for r in rows)
    n_not_converged = sum(r["cos_flagged_not_converged"] for r in rows)
    max_rel_diff = max(r["rel_diff_cos_vs_carrmadan"] for r in rows)
    print(f"\n{len(rows)} queries checked in {time.time()-t_start:.1f}s.")
    print(f"Carr-Madan hit its integration cutoff: {n_cutoff}/{len(rows)} queries.")
    print(f"COS flagged not-converged (doubling sweep): {n_not_converged}/{len(rows)} queries.")
    print(f"Max COS-vs-Carr-Madan relative difference across all {len(rows)} queries: {max_rel_diff:.3e}")
    print(f"(NOT uniformly ~1e-14 -- this is the honest cross-check number, not forced to match any prior claim)")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
