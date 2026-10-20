"""Test 5 (per the task): take the exact Galerkin trajectory, perturb it by a
KNOWN amount, compute the true Galerkin-space error E and each of
eta_mass_indicator (heuristic, NOT a certificate -- see
certification.mass_indicator), eta_sharp, eta_guar, and check E <= eta_guar.
Report effectivities (E <= eta_mass_indicator is deliberately NOT asserted:
it has no proven bound property).

IMPORTANT caveat on what "E" means here: proving that a residual aggregated
over many time steps bounds the *trajectory* error in general requires a
discrete stability/Gronwall-type theorem connecting the single-step CN
operator to S_h(mu) -- that is exactly the kind of manuscript-specific
result I do not have verified. To avoid asserting something I cannot
justify, this test uses a scenario where E is unambiguous and directly
computable without needing that theorem: a SINGLE coefficient vector in the
trajectory is perturbed by a known delta, and

    E := ||delta||_{S_h} aggregated the same dt-weighted way the
         certificates are: E^2 = dt * delta^T S_h(mu) delta

(only one term is nonzero since only one time level is perturbed). This is
an internally consistent definition -- consistent with how eta_mass_indicator /
eta_sharp / eta_guar themselves aggregate over steps -- not a claim that it
matches the manuscript's own error norm verbatim. The E <= eta_guar
inequality is checked EMPIRICALLY over several random perturbations/
magnitudes and reported, not assumed to hold by construction: if it failed
here, that would be a critical finding to surface before any ML work, and
this test would need to fail loudly, not be adjusted to pass.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.constants import compute_stability_constants
from certification.guaranteed_certificate import eta_guar, eta_sharp
from certification.mass_indicator import eta_mass_indicator as compute_eta_mass_indicator
from certification.residuals import compute_all_residuals

MU = CGMYParameters(C=1.0, G=5.0, M=5.0, Y=0.5, r=0.05, T=0.5)
PAYOFF = EuropeanPutPayoff(strike=100.0)


def _trajectory_and_cache():
    mesh = Mesh(N=32, p=3)
    tg = TimeGrid(n_steps=20)
    traj = solve_cgmy(MU, PAYOFF, mesh, tg, mode="toeplitz")
    ops = OperatorCache(Mesh(N=traj.mesh_information.N, p=traj.mesh_information.p),
                         traj.mesh_information.x_min, traj.mesh_information.x_max)
    return traj, ops


@pytest.mark.parametrize("rel_magnitude", [1e-4, 1e-3, 1e-2])
def test_certificate_bounds_known_perturbation(rel_magnitude: float):
    traj, ops = _trajectory_and_cache()
    const = compute_stability_constants(MU, ops)
    assert const.is_spd

    n_perturb = traj.solver_metadata.n_steps // 2
    rng = np.random.default_rng(7)
    direction = rng.standard_normal(ops.n_dof)
    direction /= np.linalg.norm(direction)
    delta = rel_magnitude * np.linalg.norm(traj.coefficients[n_perturb]) * direction

    coeffs = traj.coefficients.copy()
    coeffs[n_perturb] = coeffs[n_perturb] + delta
    traj_pert = dataclasses.replace(traj, coefficients=coeffs)

    dt = traj.solver_metadata.dt
    E = float(np.sqrt(dt * (delta @ ops.apply_S(MU, delta))))

    residuals = compute_all_residuals(traj_pert, MU, ops)

    m = compute_eta_mass_indicator(residuals, ops, dt, const.alpha_certified)
    sharp = eta_sharp(residuals, MU, ops, dt)
    guar = eta_guar(residuals, MU, ops, dt, const.alpha_certified, max_pcg_iterations=200)

    eff_mass_indicator = m / E if E > 0 else float("nan")
    eff_sharp = sharp.eta / E if E > 0 else float("nan")
    eff_guar = guar.eta / E if E > 0 else float("nan")

    print(f"\nrel_magnitude={rel_magnitude:.0e}  E={E:.6e}")
    print(f"  eta_mass_indicator ={m:.6e}   effectivity={eff_mass_indicator:.3f}  (heuristic, not a bound)")
    print(f"  eta_sharp={sharp.eta:.6e}   effectivity={eff_sharp:.3f}")
    print(f"  eta_guar ={guar.eta:.6e}   effectivity={eff_guar:.3f}  "
          f"(max PCG iters/step used: {guar.n_pcg_iterations.max()})")

    assert E <= sharp.eta * (1 + 1e-6), (
        f"E={E:.6e} > eta_sharp={sharp.eta:.6e}: the sharp certificate did "
        f"not bound the known perturbation's energy norm -- this is a "
        f"critical finding, not a test bug, see module docstring"
    )
    assert E <= guar.eta * (1 + 1e-6), (
        f"E={E:.6e} > eta_guar={guar.eta:.6e}: CLAUDE.md Hard Invariant #5 "
        f"violated for this perturbation"
    )
    assert guar.eta >= sharp.eta * (1 - 1e-9), (
        "eta_guar (an upper-bound-based certificate) must be >= eta_sharp "
        "(the exact quadratic form), by construction of the PCG bracket"
    )


def test_eta_guar_converges_to_eta_sharp_as_pcg_iterations_increase():
    traj, ops = _trajectory_and_cache()
    const = compute_stability_constants(MU, ops)
    dt = traj.solver_metadata.dt

    n_perturb = traj.solver_metadata.n_steps // 2
    rng = np.random.default_rng(3)
    delta = 1e-3 * np.linalg.norm(traj.coefficients[n_perturb]) * rng.standard_normal(ops.n_dof)
    coeffs = traj.coefficients.copy()
    coeffs[n_perturb] = coeffs[n_perturb] + delta
    traj_pert = dataclasses.replace(traj, coefficients=coeffs)
    residuals = compute_all_residuals(traj_pert, MU, ops)

    sharp = eta_sharp(residuals, MU, ops, dt)

    prev_gap = float("inf")
    print(f"\neta_sharp = {sharp.eta:.8e}")
    for k in (1, 2, 5, 10, 50, 300):
        guar = eta_guar(residuals, MU, ops, dt, const.alpha_certified, max_pcg_iterations=k)
        gap = guar.eta - sharp.eta
        print(f"  max_pcg_iterations={k:4d}  eta_guar={guar.eta:.8e}  gap={gap:.3e}")
        assert gap >= -1e-6, f"eta_guar ({guar.eta:.8e}) fell below eta_sharp ({sharp.eta:.8e})"
        assert gap <= prev_gap + 1e-9, "gap to eta_sharp must shrink as PCG iterations increase"
        prev_gap = gap

    assert prev_gap < 1e-4, f"eta_guar did not converge close to eta_sharp: final gap {prev_gap:.3e}"
