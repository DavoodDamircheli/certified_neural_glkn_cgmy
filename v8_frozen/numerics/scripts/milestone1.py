"""Milestone 1: Galerkin trajectory -> residual -> eta_mass_indicator (heuristic) -> eta_sharp -> eta_guar.

The headline end-to-end demonstration of Stages A-C, run as a script (not
just pytest) so the numbers are visible in one place. Verifies:

  1. the exact reference trajectory's residual is ~0 (Gate 1a);
  2. a perturbed trajectory's residual is unambiguously nonzero (Gate 1b);
  3. the certificates bound a KNOWN perturbation's true energy-norm error
     (same caveat as tests/test_certificate_reference_solution.py: "true
     error" here means the S_h-energy norm of a perturbation this script
     itself constructs, not a manuscript-verified error representation);
  4. eta_sharp and eta_guar agree as PCG iterations increase (bracket
     collapse, Gate 2's essence);
  5. the Riesz bracket itself is valid at a representative step.

Prints a pass/fail summary. Exits nonzero if any check fails.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.constants import compute_stability_constants, dense_from_apply
from certification.guaranteed_certificate import eta_guar, eta_sharp
from certification.mass_indicator import eta_mass_indicator as compute_eta_mass_indicator
from certification.residuals import compute_all_residuals, residual_norm
from certification.riesz_solver import solve_riesz_pcg

MU = CGMYParameters(C=1.0, G=5.0, M=5.0, Y=0.5, r=0.05, T=0.5)
PAYOFF = EuropeanPutPayoff(strike=100.0)
N, N_TAU = 32, 20

checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str) -> None:
    checks.append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {detail}")


def main() -> int:
    print("=" * 78)
    print(f"Milestone 1 -- mu = (C={MU.C}, G={MU.G}, M={MU.M}, Y={MU.Y}, r={MU.r}, T={MU.T})")
    print(f"mesh: N={N}, p=3 (auto domain);  time_grid: n_steps={N_TAU}")
    print("=" * 78)

    mesh = Mesh(N=N, p=3)
    tg = TimeGrid(n_steps=N_TAU)
    traj = solve_cgmy(MU, PAYOFF, mesh, tg, mode="toeplitz")
    ops = OperatorCache(
        Mesh(N=traj.mesh_information.N, p=traj.mesh_information.p),
        traj.mesh_information.x_min, traj.mesh_information.x_max,
    )
    dt = traj.solver_metadata.dt
    print(f"\nrealized mesh: n_dof={ops.n_dof}, h={ops.h:.5f}, "
          f"x_min={ops.x_min:.4f}, x_max={ops.x_max:.4f}, dt={dt:.5f}")

    const = compute_stability_constants(MU, ops)
    check(
        "S_h(mu) is SPD at this mu",
        const.is_spd,
        f"lambda_min(S_h) numerical={const.lambda_min_numerical:.4e}, "
        f"alpha_certified={const.alpha_certified:.4e} ({const.coercivity_strategy}, "
        f"safety_ratio={const.safety_ratio:.6f}), "
        f"lambda_max(S_h)={const.continuity_constant:.4e}, cond(S_h)={const.condition_number_S_h:.3e} "
        f"(NOT the efficiency-theorem kappa(mu); see certification.constants.kappa_operator_norm)",
    )
    if not const.is_spd:
        print("\nCannot proceed: certification requires S_h(mu) SPD at this mu.")
        return 1

    # ── 1. exact trajectory residual ~ 0 ────────────────────────────────────
    exact_residuals = compute_all_residuals(traj, MU, ops)
    exact_norms = [residual_norm(exact_residuals[n], ops, traj.solver_metadata.bc_dofs)
                   for n in range(traj.solver_metadata.n_steps)]
    max_exact_norm = max(exact_norms)
    check(
        "1. exact trajectory residual ~ 0",
        max_exact_norm < 1e-6,
        f"max_n ||r^(n+1/2)|| = {max_exact_norm:.3e} (GMRES rtol={traj.solver_metadata.tolerance_per_step[0]:.0e})",
    )

    # ── 2. perturbed trajectory residual unambiguously nonzero ─────────────
    rng = np.random.default_rng(11)
    rel_eps = 1e-3
    coeffs_pert = traj.coefficients.copy()
    deltas = np.zeros_like(coeffs_pert)
    for n in range(1, traj.solver_metadata.n_steps):  # leave initial/final untouched
        direction = rng.standard_normal(ops.n_dof)
        direction /= np.linalg.norm(direction)
        deltas[n] = rel_eps * np.linalg.norm(coeffs_pert[n]) * direction
        coeffs_pert[n] = coeffs_pert[n] + deltas[n]
    traj_pert = dataclasses.replace(traj, coefficients=coeffs_pert)

    pert_residuals = compute_all_residuals(traj_pert, MU, ops)
    pert_norms = [residual_norm(pert_residuals[n], ops, traj.solver_metadata.bc_dofs)
                  for n in range(traj.solver_metadata.n_steps)]
    max_pert_norm = max(pert_norms)
    check(
        "2. perturbed trajectory residual unambiguously nonzero",
        max_pert_norm > 1e3 * max_exact_norm,
        f"max_n ||r|| = {max_pert_norm:.3e} vs exact {max_exact_norm:.3e} "
        f"(ratio {max_pert_norm / max_exact_norm:.1f}x)",
    )

    # ── 3. certificates bound the known perturbation's energy-norm error ───
    E_sq = dt * sum(float(deltas[n] @ ops.apply_S(MU, deltas[n])) for n in range(len(deltas)))
    E = float(np.sqrt(E_sq))

    m = compute_eta_mass_indicator(pert_residuals, ops, dt, const.alpha_certified)
    sharp = eta_sharp(pert_residuals, MU, ops, dt)
    guar = eta_guar(pert_residuals, MU, ops, dt, const.alpha_certified, max_pcg_iterations=200)

    print(f"\n  E (known perturbation energy norm) = {E:.6e}")
    print(f"  eta_mass_indicator = {m:.6e}  (effectivity {m / E:.2f}, heuristic -- NOT checked as a bound)")
    print(f"  eta_sharp = {sharp.eta:.6e}  (effectivity {sharp.eta / E:.2f})")
    print(f"  eta_guar  = {guar.eta:.6e}  (effectivity {guar.eta / E:.2f}, "
          f"max PCG iters/step = {guar.n_pcg_iterations.max()})")

    check("3a. E <= eta_sharp", E <= sharp.eta * (1 + 1e-6), f"E={E:.4e}, eta_sharp={sharp.eta:.4e}")
    check("3b. E <= eta_guar", E <= guar.eta * (1 + 1e-6), f"E={E:.4e}, eta_guar={guar.eta:.4e}")

    # ── 4. sharp and iterative certificates agree as PCG converges ─────────
    gaps = []
    for k in (1, 5, 10, 50, 300):
        g = eta_guar(pert_residuals, MU, ops, dt, const.alpha_certified, max_pcg_iterations=k)
        gaps.append((k, g.eta - sharp.eta))
    gap_str = ", ".join(f"k={k}:{gap:.2e}" for k, gap in gaps)
    check(
        "4. eta_guar -> eta_sharp as PCG iterations increase",
        gaps[-1][1] < 1e-4 and all(gaps[i][1] <= gaps[i - 1][1] + 1e-9 for i in range(1, len(gaps))),
        gap_str,
    )

    # ── 5. Riesz bracket verified at a representative step ─────────────────
    n_rep = traj.solver_metadata.n_steps // 2
    r_rep = pert_residuals[n_rep]
    S_dense = dense_from_apply(lambda v: ops.apply_S(MU, v), ops.n_dof)
    q_exact = float(r_rep @ np.linalg.solve(S_dense, r_rep))
    bracket_ok = True
    widths = []
    for k in (0, 1, 2, 5, 10, 200):
        b = solve_riesz_pcg(r_rep, MU, ops, const.alpha_certified, max_iter=k, preconditioner="none")
        widths.append(b.upper_bound - b.lower_bound)
        if not (b.lower_bound <= q_exact + 1e-6 <= b.upper_bound + 1e-6):
            bracket_ok = False
    monotone = all(widths[i] <= widths[i - 1] + 1e-9 for i in range(1, len(widths)))
    check(
        "5. Riesz bracket contains q_exact and shrinks monotonically",
        bracket_ok and monotone,
        f"q_exact={q_exact:.4e}, widths={['%.2e' % w for w in widths]}",
    )

    print("\n" + "=" * 78)
    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"MILESTONE 1: {n_pass}/{len(checks)} checks passed")
    print("=" * 78)
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
