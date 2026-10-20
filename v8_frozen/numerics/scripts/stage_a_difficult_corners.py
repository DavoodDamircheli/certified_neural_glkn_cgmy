"""Stage A: probe the candidate CGMY parameter box at its difficult corners.

Runs the wrapped trusted reference solver (``cgmy.reference_solver.solve_cgmy``)
at the min/max of each parameter in ``configs/base.yaml``'s parameter_box,
plus stress points near the CGMY singular limits Y -> 1+ and Y -> 2- (outside
the current box, since the box's own Y bounds are already tested as "min/max
Y"), plus a few multi-extreme combinations. For each point, reports:

  - spatial convergence:  |price(N) - price(2N)|, N_tau fixed
  - temporal convergence: |price(N_tau) - price(2*N_tau)|, N fixed
  - domain-truncation sensitivity: |price(domain_scale=10) - price(domain_scale=15)|
  - conditioning proxy: mean GMRES iterations/step (mode='toeplitz'), and
    cond(S_h) on a small (N=32) mesh at the same mu
  - stability: whether GMRES converged (no non-convergence warning) and
    whether the resulting price is a sane put price (in [0, K])
  - the resulting ATM (x = log K) option price

This script does not decide anything about the parameter box; it only
produces the evidence. See the printed recommendation section at the end,
which is deliberately conservative: it flags corners that look risky and
leaves the actual narrow/don't-narrow call to the user (per CLAUDE.md,
changing the frozen parameter box requires an explicit, committed decision).
"""

from __future__ import annotations

import csv
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from cgmy.reference_solver import (
    CGMYParameters,
    EuropeanPutPayoff,
    Mesh,
    TimeGrid,
    solve_cgmy,
)
from cgmy.operators import OperatorCache

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "stage_a_difficult_corners"

N_BASE = 64
N_TAU_BASE = 100
STRIKE = 100.0


@dataclass
class CornerResult:
    name: str
    C: float
    G: float
    M: float
    Y: float
    r: float
    T: float
    price_atm: float
    price_in_range: bool
    spatial_conv_abs: float
    temporal_conv_abs: float
    domain_trunc_abs: float
    mean_gmres_iters: float
    cond_S_h_small_mesh: float
    gmres_warned: bool
    wall_time_s: float
    error: str


def _load_box(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["parameter_box"], cfg["contract"]["strike"]


def _midpoint(bounds: list[float]) -> float:
    return 0.5 * (bounds[0] + bounds[1])


def _build_corners(box: dict) -> dict[str, CGMYParameters]:
    """Corner mu points: min/max of each param at baseline, Y-singularity
    stress points, and a few multi-extreme combinations."""
    baseline = dict(
        C=_midpoint(box["C"]),
        G=_midpoint(box["G"]),
        M=_midpoint(box["M"]),
        Y=_midpoint(box["Y"]),
        r=_midpoint(box["r"]),
        T=_midpoint(box["maturity"]),
    )

    corners: dict[str, CGMYParameters] = {"baseline": CGMYParameters(**baseline)}

    for key, box_key in [("C", "C"), ("G", "G"), ("M", "M"), ("Y", "Y"), ("r", "r"), ("T", "maturity")]:
        lo, hi = box[box_key]
        for tag, val in [("min", lo), ("max", hi)]:
            params = dict(baseline)
            params[key] = val
            corners[f"{key}_{tag}"] = CGMYParameters(**params)

    # CGMY singular-limit stress points, deliberately OUTSIDE the current
    # box's Y range [1.10, 1.80] -- these test how close to the true
    # (0,2)\{1} boundary the solver can safely go, informing whether the
    # box should stay conservative or could be widened.
    for tag, y in [("Y_to_1_plus", 1.01), ("Y_to_2_minus", 1.99)]:
        params = dict(baseline)
        params["Y"] = y
        corners[tag] = CGMYParameters(**params)

    # Multi-extreme combinations.
    corners["multi_all_min"] = CGMYParameters(
        C=box["C"][0], G=box["G"][0], M=box["M"][0], Y=box["Y"][0], r=box["r"][0], T=box["maturity"][0]
    )
    corners["multi_all_max"] = CGMYParameters(
        C=box["C"][1], G=box["G"][1], M=box["M"][1], Y=box["Y"][1], r=box["r"][1], T=box["maturity"][1]
    )
    # "Worst plausible": light tempering (small G, M -> heavier tails) with
    # high activity and high Y (closer to Brownian-like but still singular),
    # long maturity, high rate.
    corners["multi_worst_plausible"] = CGMYParameters(
        C=box["C"][1], G=box["G"][0], M=box["M"][0], Y=box["Y"][1], r=box["r"][1], T=box["maturity"][1]
    )

    return corners


def _price_at_strike(traj, strike: float) -> float:
    from cgmy_bspline.projection import eval_solution

    g = {
        "N": traj.mesh_information.N,
        "h": traj.mesh_information.h,
        "x_min": traj.mesh_information.x_min,
    }
    x0 = np.array([np.log(strike)])
    V = eval_solution(traj.coefficients[-1], g, traj.mesh_information.p, x0)
    return float(V[0])


def _cond_S_h_small_mesh(mu: CGMYParameters, x_min: float, x_max: float, N: int = 32, p: int = 3) -> float:
    cache = OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)
    n = cache.n_dof
    S = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        S[:, j] = cache.apply_S(mu, e)
    eigs = np.linalg.eigvalsh(S)
    lo, hi = eigs.min(), eigs.max()
    if lo <= 0:
        return float("inf")
    return float(hi / lo)


def _run_one(name: str, mu: CGMYParameters) -> CornerResult:
    t0 = time.time()
    payoff = EuropeanPutPayoff(strike=STRIKE)
    err = ""
    price_atm = float("nan")
    spatial_conv = float("nan")
    temporal_conv = float("nan")
    domain_trunc = float("nan")
    mean_iters = float("nan")
    cond_S = float("nan")
    gmres_warned = False

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            mesh_n = Mesh(N=N_BASE, p=3)
            tg_n = TimeGrid(n_steps=N_TAU_BASE)
            traj = solve_cgmy(mu, payoff, mesh_n, tg_n, mode="toeplitz")
            price_atm = _price_at_strike(traj, STRIKE)
            mean_iters = traj.solver_metadata.mean_gmres_iters or 0.0

            mesh_2n = Mesh(N=2 * N_BASE, p=3)
            traj_2n = solve_cgmy(mu, payoff, mesh_2n, tg_n, mode="toeplitz")
            price_2n = _price_at_strike(traj_2n, STRIKE)
            spatial_conv = abs(price_atm - price_2n)

            tg_2nt = TimeGrid(n_steps=2 * N_TAU_BASE)
            traj_2nt = solve_cgmy(mu, payoff, mesh_n, tg_2nt, mode="toeplitz")
            price_2nt = _price_at_strike(traj_2nt, STRIKE)
            temporal_conv = abs(price_atm - price_2nt)

            mesh_wide = Mesh(N=N_BASE, p=3, domain_scale=15.0)
            traj_wide = solve_cgmy(mu, payoff, mesh_wide, tg_n, mode="toeplitz")
            price_wide = _price_at_strike(traj_wide, STRIKE)
            domain_trunc = abs(price_atm - price_wide)

            cond_S = _cond_S_h_small_mesh(
                mu, traj.mesh_information.x_min, traj.mesh_information.x_max
            )

            gmres_warned = any("did not converge" in str(w.message) for w in caught)

    except Exception as exc:  # noqa: BLE001 -- report, don't crash the sweep
        err = f"{type(exc).__name__}: {exc}"

    price_in_range = (not np.isnan(price_atm)) and (0.0 <= price_atm <= STRIKE)

    return CornerResult(
        name=name,
        C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, T=mu.T,
        price_atm=price_atm,
        price_in_range=price_in_range,
        spatial_conv_abs=spatial_conv,
        temporal_conv_abs=temporal_conv,
        domain_trunc_abs=domain_trunc,
        mean_gmres_iters=mean_iters,
        cond_S_h_small_mesh=cond_S,
        gmres_warned=gmres_warned,
        wall_time_s=time.time() - t0,
        error=err,
    )


def main() -> None:
    box, strike = _load_box(CONFIG_PATH)
    global STRIKE
    STRIKE = strike

    corners = _build_corners(box)

    results: list[CornerResult] = []
    for name, mu in corners.items():
        print(f"running corner: {name:22s} mu=(C={mu.C:.3g}, G={mu.G:.3g}, "
              f"M={mu.M:.3g}, Y={mu.Y:.3g}, r={mu.r:.3g}, T={mu.T:.3g})")
        results.append(_run_one(name, mu))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(CONFIG_PATH, RESULTS_DIR / "config_snapshot.yaml")

    csv_path = RESULTS_DIR / "corners.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CornerResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)

    _print_report(results)
    print(f"\nFull results written to {csv_path}")


def _print_report(results: list[CornerResult]) -> None:
    header = (
        f"{'corner':22s} {'price':>8s} {'in[0,K]':>8s} {'spatial':>10s} "
        f"{'temporal':>10s} {'domain':>10s} {'gmres_it':>9s} {'cond(S)':>10s} "
        f"{'gmres_warn':>11s} {'error':>6s}"
    )
    print("\n" + header)
    print("-" * len(header))
    flagged: list[str] = []
    for r in results:
        print(
            f"{r.name:22s} {r.price_atm:8.3f} {str(r.price_in_range):>8s} "
            f"{r.spatial_conv_abs:10.2e} {r.temporal_conv_abs:10.2e} "
            f"{r.domain_trunc_abs:10.2e} {r.mean_gmres_iters:9.1f} "
            f"{r.cond_S_h_small_mesh:10.2e} {str(r.gmres_warned):>11s} "
            f"{('yes' if r.error else 'no'):>6s}"
        )
        risky = (
            r.error
            or not r.price_in_range
            or r.gmres_warned
            or r.domain_trunc_abs > 1e-2
            or r.spatial_conv_abs > 1e-1
        )
        if risky:
            flagged.append(r.name)

    print("\n" + "=" * len(header))
    print("Corners flagged for review:", flagged if flagged else "none")
    print(
        "\nThis script reports evidence only. It does not narrow or freeze the\n"
        "parameter box (per CLAUDE.md, that is an explicit, committed decision).\n"
        "Review the flagged corners above -- particularly domain-truncation\n"
        "sensitivity (the trusted solver's default domain is mu-dependent via\n"
        "domain_scale/G, domain_scale/M, not the fixed Omega assumed in\n"
        "configs/base.yaml) and the Y -> 1+/2- stress points -- before deciding."
    )


if __name__ == "__main__":
    main()
