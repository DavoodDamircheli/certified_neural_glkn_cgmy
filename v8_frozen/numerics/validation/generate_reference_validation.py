"""Gate C validation run: prices ~20 production-box queries + deliberately
extreme cases with both independent methods (COS, Carr-Madan), records
physical diagnostics, and writes validation/reference_validation.csv.

Does not call vendor/cgmy_bspline or src/cgmy. Domain/space/time
refinement against the Galerkin solver is a separate, larger piece of
Gate C (see reports/REFERENCE_ACCURACY_REPORT.md "NOT RUN") -- this script
only validates the independent pricer pair against each other and against
physical sanity diagnostics.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml

from validation.cgmy_reference_pricing import CGMYMarketParams, martingale_check_mpmath, martingale_check_numpy
from validation.cos_pricer import converge_price, cos_put_price, truncation_range
from validation.independent_fourier_integration import carr_madan_put_price


def _fast_price(mu, s0, k):
    """Fixed (N=256, L=12) setting -- already shown to be well past
    convergence for this family (tests/v3/test_cos_pricer.py). Used for
    the per-strike physical diagnostics below so the per-query cost isn't
    dominated by 5 redundant full convergence sweeps."""
    a, b = truncation_range(mu, L=12.0)
    return cos_put_price(mu, s0, k, N=256, trunc_range=(a, b), L=12.0)["price"]

with open(REPO_ROOT / "configs" / "base.yaml") as f:
    _cfg = yaml.safe_load(f)
BOX = _cfg["parameter_box"]
K_FIXED = _cfg["contract"]["strike"]

rng = np.random.default_rng(20261005)


def sample_box_query(i: int) -> CGMYMarketParams:
    C = rng.uniform(*BOX["C"])
    G = rng.uniform(*BOX["G"])
    M = rng.uniform(*BOX["M"])
    Y = rng.uniform(*BOX["Y"])
    r = rng.uniform(*BOX["r"])
    T = rng.uniform(*BOX["maturity"])
    return CGMYMarketParams(C=C, G=G, M=M, Y=Y, r=r, T=T)


N_BOX_QUERIES = 20
box_queries = [sample_box_query(i) for i in range(N_BOX_QUERIES)]

extreme_queries = {
    "extreme_short_maturity": CGMYMarketParams(C=0.3, G=3.0, M=5.0, Y=1.5, r=0.05, T=0.1),
    "extreme_low_C": CGMYMarketParams(C=0.05, G=3.0, M=5.0, Y=1.5, r=0.05, T=0.5),
    "extreme_smallest_G_M": CGMYMarketParams(C=0.3, G=1.0, M=2.0, Y=1.5, r=0.05, T=0.5),
    "extreme_high_Y": CGMYMarketParams(C=0.3, G=3.0, M=5.0, Y=1.79, r=0.05, T=0.5),
    "extreme_low_Y": CGMYMarketParams(C=0.3, G=3.0, M=5.0, Y=1.11, r=0.05, T=0.5),
    "extreme_long_maturity_low_r": CGMYMarketParams(C=0.3, G=3.0, M=5.0, Y=1.5, r=0.01, T=1.0),
    "extreme_high_r": CGMYMarketParams(C=0.3, G=3.0, M=5.0, Y=1.5, r=0.10, T=0.5),
}

S0 = K_FIXED
strikes_for_diagnostics = [0.8 * K_FIXED, 0.9 * K_FIXED, K_FIXED, 1.1 * K_FIXED, 1.2 * K_FIXED]

rows = []
all_queries = [(f"box_{i:02d}", mu) for i, mu in enumerate(box_queries)] + list(extreme_queries.items())

for qid, mu in all_queries:
    cos_out = converge_price(mu, S0, K_FIXED)
    cm_out = carr_madan_put_price(mu, S0, K_FIXED, dps=40)
    cos_price = cos_out["price"]
    cm_price = cm_out["price"]
    discrepancy = abs(cos_price - cm_price)
    rel_discrepancy = discrepancy / max(abs(cm_price), 1e-8)

    mart_numpy = martingale_check_numpy(mu)
    mart_mpmath = martingale_check_mpmath(mu, dps=50)

    prices_by_strike = [_fast_price(mu, S0, k) for k in strikes_for_diagnostics]
    diffs = np.diff(prices_by_strike)
    second_diffs = np.diff(prices_by_strike, n=2)
    upper_bound = K_FIXED * np.exp(-mu.r * mu.T)

    positivity_ok = all(p >= -1e-8 for p in prices_by_strike)
    bound_ok = all(p <= upper_bound + 1e-6 for p in prices_by_strike)
    monotonicity_ok = bool(np.all(diffs >= -1e-6))
    convexity_ok = bool(np.all(second_diffs >= -1e-5))

    rows.append(dict(
        query_id=qid, C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, T=mu.T, K=K_FIXED,
        cos_price=cos_price, independent_fourier_price=cm_price,
        discrepancy_between_methods=discrepancy, relative_discrepancy=rel_discrepancy,
        cos_truncation_terms_used=cos_out["N_used"], cos_range_used=f"[{cos_out['L_used']}]",
        cos_converged=not cos_out["flagged_not_converged"],
        carr_madan_alpha=cm_out["alpha"], carr_madan_v_cutoff=cm_out["v_cutoff"],
        martingale_check_numpy=mart_numpy, martingale_check_mpmath=mart_mpmath,
        positivity_ok=positivity_ok, bound_ok=bound_ok,
        monotonicity_ok=monotonicity_ok, convexity_ok=convexity_ok,
    ))
    print(f"{qid}: cos={cos_price:.6f} cm={cm_price:.6f} rel_diff={rel_discrepancy:.2e} "
          f"mart_np={mart_numpy:.2e} mono={monotonicity_ok} convex={convexity_ok}")

out_path = REPO_ROOT / "validation" / "reference_validation.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {len(rows)} rows -> {out_path}")
max_rel_discrepancy = max(r["relative_discrepancy"] for r in rows)
max_mart = max(abs(r["martingale_check_numpy"]) for r in rows)
n_all_physical_ok = sum(
    r["positivity_ok"] and r["bound_ok"] and r["monotonicity_ok"] and r["convexity_ok"] for r in rows
)
print(f"max relative discrepancy between methods: {max_rel_discrepancy:.3e}")
print(f"max |martingale check (numpy)|: {max_mart:.3e}")
print(f"queries passing all physical diagnostics: {n_all_physical_ok}/{len(rows)}")
