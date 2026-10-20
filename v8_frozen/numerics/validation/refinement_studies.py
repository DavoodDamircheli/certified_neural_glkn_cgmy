"""Gate C refinement studies: domain / space / time, separated, against the
independent COS reference pricer (validation/cos_pricer.py, already
validated: martingale check ~1e-14, cross-method agreement ~1e-14-1e-15).

Calls `src/cgmy.reference_solver.solve_cgmy` directly -- this is the
Galerkin solver UNDER TEST, not something to reimplement (Gate C is
independent of the certification/learning code, not of the solver itself).

Each of the three studies holds the other two axes fixed/fine so the
reported error is attributable to the one varying axis (roadmap: "never
confound domain width with mesh size").

Run: PYTHONPATH=src python3 validation/refinement_studies.py
Writes validation/{domain,space,time}_refinement.csv and
validation/galerkin_physical_diagnostics.csv.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy  # noqa: E402
from cgmy_bspline.projection import eval_solution  # noqa: E402

from validation.cgmy_reference_pricing import CGMYMarketParams  # noqa: E402
from validation.cos_pricer import cos_put_price, truncation_range  # noqa: E402

K = 100.0
MONEYNESS_GRID = [0.8, 0.9, 1.0, 1.1, 1.2]
PRODUCTION_H = 0.25  # configs/base.yaml: (log(100)+8 - (log(100)-8)) / 64

N_QUERIES = 6  # Q_ref30_0000..0005 -- first 6 rows of the shared Q_ref30 set


def load_queries() -> list[dict]:
    path = REPO_ROOT / "experiments" / "raw_results" / "query_sets" / "Q_ref30.csv"
    rows = []
    with open(path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= N_QUERIES:
                break
            rows.append({k: float(row[k]) for k in ("C", "G", "M", "Y", "r", "T")} | {"query_id": row["query_id"]})
    return rows


def cos_reference_prices(q: dict) -> dict[float, float]:
    """Independent COS price at each moneyness point (production N=256,
    L=12 -- already shown convergent for this family in the prior Gate C
    pass's tests/v3/test_cos_pricer.py)."""
    mu = CGMYMarketParams(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
    a, b = truncation_range(mu, L=12.0)
    out = {}
    for m in MONEYNESS_GRID:
        s0 = m * K
        out[m] = cos_put_price(mu, s0, K, N=256, trunc_range=(a, b))["price"]
    return out


def galerkin_prices_at_moneyness(
    q: dict, N: int, x_min: float, x_max: float, n_steps: int
) -> tuple[dict[float, float], float]:
    """Solve once, evaluate the terminal state at every moneyness point.
    Returns (prices_by_moneyness, wall_time_s)."""
    mu = CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
    payoff = EuropeanPutPayoff(strike=K)
    mesh = Mesh(N=N, p=3, x_min=x_min, x_max=x_max)
    tg = TimeGrid(n_steps=n_steps)
    t0 = time.perf_counter()
    traj = solve_cgmy(mu, payoff, mesh, tg, mode="toeplitz", store_forcing=False)
    wall = time.perf_counter() - t0
    c_final = traj.coefficients[-1]
    grid_info = {"N": N, "h": (x_max - x_min) / N, "x_min": x_min}
    out = {}
    for m in MONEYNESS_GRID:
        x0 = np.array([np.log(m * K)])
        out[m] = float(eval_solution(c_final, grid_info, 3, x0)[0])
    return out, wall


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_domain_refinement(queries: list[dict]) -> list[dict]:
    """Buffer in {6,8,10,12}; N chosen so h stays fixed at PRODUCTION_H
    across buffers (width = 2*buffer, N = width/h) -- domain width and mesh
    resolution are varied independently by construction, never confounded.
    Time grid fixed fine (N_T=240) so temporal error doesn't contaminate
    the domain-width read."""
    rows = []
    n_steps = 240
    log_K = np.log(K)
    for buffer in (6, 8, 10, 12):
        width = 2 * buffer
        N = round(width / PRODUCTION_H)
        assert abs(N * PRODUCTION_H - width) < 1e-9, "buffer does not divide evenly into PRODUCTION_H steps"
        x_min, x_max = log_K - buffer, log_K + buffer
        for q in queries:
            ref = cos_reference_prices(q)
            gal, wall = galerkin_prices_at_moneyness(q, N, x_min, x_max, n_steps)
            for m in MONEYNESS_GRID:
                abs_err = abs(gal[m] - ref[m])
                rel_err = abs_err / max(abs(ref[m]), 1e-8)
                rows.append(dict(
                    query_id=q["query_id"], buffer=buffer, N=N, h=PRODUCTION_H, n_steps=n_steps,
                    moneyness=m, galerkin_price=gal[m], cos_reference_price=ref[m],
                    abs_error=abs_err, rel_error=rel_err, wall_time_s=wall,
                ))
    return rows


def run_space_refinement(queries: list[dict]) -> list[dict]:
    """N in {32,64,128,256,512(reduced sample)} on the production domain
    (buffer=8), time error made small (N_T=240, 4x production's 60)."""
    rows = []
    buffer = 8
    log_K = np.log(K)
    x_min, x_max = log_K - buffer, log_K + buffer
    n_steps = 240
    full_Ns = (32, 64, 128, 256)
    stretch_Ns = (512,)
    for N in full_Ns:
        for q in queries:
            ref = cos_reference_prices(q)
            gal, wall = galerkin_prices_at_moneyness(q, N, x_min, x_max, n_steps)
            for m in MONEYNESS_GRID:
                abs_err = abs(gal[m] - ref[m])
                rel_err = abs_err / max(abs(ref[m]), 1e-8)
                rows.append(dict(
                    query_id=q["query_id"], N=N, buffer=buffer, n_steps=n_steps,
                    moneyness=m, galerkin_price=gal[m], cos_reference_price=ref[m],
                    abs_error=abs_err, rel_error=rel_err, wall_time_s=wall, sample="full",
                ))
    # Stretch: N=512, reduced to 2 queries to bound cost (reported honestly as reduced sample).
    for N in stretch_Ns:
        for q in queries[:2]:
            ref = cos_reference_prices(q)
            gal, wall = galerkin_prices_at_moneyness(q, N, x_min, x_max, n_steps)
            for m in MONEYNESS_GRID:
                abs_err = abs(gal[m] - ref[m])
                rel_err = abs_err / max(abs(ref[m]), 1e-8)
                rows.append(dict(
                    query_id=q["query_id"], N=N, buffer=buffer, n_steps=n_steps,
                    moneyness=m, galerkin_price=gal[m], cos_reference_price=ref[m],
                    abs_error=abs_err, rel_error=rel_err, wall_time_s=wall, sample="stretch_reduced_n",
                ))
    return rows


def run_time_refinement(queries: list[dict]) -> list[dict]:
    """N_T in {30,60,120,240,480(reduced sample)} on a fine spatial grid
    (N=256), buffer=8 fixed."""
    rows = []
    buffer = 8
    log_K = np.log(K)
    x_min, x_max = log_K - buffer, log_K + buffer
    N = 256
    full_NTs = (30, 60, 120, 240)
    stretch_NTs = (480,)
    for n_steps in full_NTs:
        for q in queries:
            ref = cos_reference_prices(q)
            gal, wall = galerkin_prices_at_moneyness(q, N, x_min, x_max, n_steps)
            for m in MONEYNESS_GRID:
                abs_err = abs(gal[m] - ref[m])
                rel_err = abs_err / max(abs(ref[m]), 1e-8)
                rows.append(dict(
                    query_id=q["query_id"], N_T=n_steps, N=N, buffer=buffer,
                    moneyness=m, galerkin_price=gal[m], cos_reference_price=ref[m],
                    abs_error=abs_err, rel_error=rel_err, wall_time_s=wall, sample="full",
                ))
    for n_steps in stretch_NTs:
        for q in queries[:2]:
            ref = cos_reference_prices(q)
            gal, wall = galerkin_prices_at_moneyness(q, N, x_min, x_max, n_steps)
            for m in MONEYNESS_GRID:
                abs_err = abs(gal[m] - ref[m])
                rel_err = abs_err / max(abs(ref[m]), 1e-8)
                rows.append(dict(
                    query_id=q["query_id"], N_T=n_steps, N=N, buffer=buffer,
                    moneyness=m, galerkin_price=gal[m], cos_reference_price=ref[m],
                    abs_error=abs_err, rel_error=rel_err, wall_time_s=wall, sample="stretch_reduced_n",
                ))
    return rows


def run_physical_diagnostics(queries: list[dict]) -> list[dict]:
    """Positivity, discounted-strike bound (V <= K*exp(-rT)), monotonicity
    (decreasing in S), convexity (discrete 2nd difference >= 0, up to
    solver tolerance) of the GALERKIN solver's own terminal-state output, on
    a finer moneyness scan than the common grid (production mesh/time)."""
    rows = []
    buffer, N, n_steps = 8, 64, 60  # production discretization
    log_K = np.log(K)
    x_min, x_max = log_K - buffer, log_K + buffer
    scan = np.linspace(0.5, 1.5, 21)
    for q in queries:
        mu = CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
        payoff = EuropeanPutPayoff(strike=K)
        mesh = Mesh(N=N, p=3, x_min=x_min, x_max=x_max)
        tg = TimeGrid(n_steps=n_steps)
        traj = solve_cgmy(mu, payoff, mesh, tg, mode="toeplitz", store_forcing=False)
        c_final = traj.coefficients[-1]
        grid_info = {"N": N, "h": (x_max - x_min) / N, "x_min": x_min}
        prices = []
        for m in scan:
            x0 = np.array([np.log(m * K)])
            prices.append(float(eval_solution(c_final, grid_info, 3, x0)[0]))
        prices = np.array(prices)
        discount_bound = K * np.exp(-q["r"] * q["T"])
        positivity_ok = bool(np.all(prices >= -1e-8))
        bound_ok = bool(np.all(prices <= discount_bound + 1e-6))
        monotonic_ok = bool(np.all(np.diff(prices) <= 1e-8))  # put price non-increasing in S
        second_diff = np.diff(prices, n=2)
        convex_ok = bool(np.all(second_diff >= -1e-6))
        rows.append(dict(
            query_id=q["query_id"], positivity_ok=positivity_ok, discount_bound_ok=bound_ok,
            monotonicity_ok=monotonic_ok, convexity_ok=convex_ok,
            min_price=float(prices.min()), max_price=float(prices.max()), discount_bound=discount_bound,
            min_second_diff=float(second_diff.min()),
        ))
    return rows


def main():
    queries = load_queries()
    print(f"Loaded {len(queries)} queries from Q_ref30.csv")

    domain_rows = run_domain_refinement(queries)
    write_csv(REPO_ROOT / "validation" / "domain_refinement.csv", domain_rows)
    print(f"domain_refinement.csv: {len(domain_rows)} rows")

    space_rows = run_space_refinement(queries)
    write_csv(REPO_ROOT / "validation" / "space_refinement.csv", space_rows)
    print(f"space_refinement.csv: {len(space_rows)} rows")

    time_rows = run_time_refinement(queries)
    write_csv(REPO_ROOT / "validation" / "time_refinement.csv", time_rows)
    print(f"time_refinement.csv: {len(time_rows)} rows")

    diag_rows = run_physical_diagnostics(queries)
    write_csv(REPO_ROOT / "validation" / "galerkin_physical_diagnostics.csv", diag_rows)
    print(f"galerkin_physical_diagnostics.csv: {len(diag_rows)} rows")


if __name__ == "__main__":
    main()
