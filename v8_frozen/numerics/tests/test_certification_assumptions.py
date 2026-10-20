r"""Task 7 (working prompt): explicit tests for the assumptions the
certificates rest on. Most of the eight assumptions the task lists were
already covered by tests added along the way in Tasks 1-6; this file adds
the two that had no home yet, and documents where the other six live so
this file can serve as the single index the task asks for.

| # | Assumption (working prompt's wording)                          | Covered by                                                  |
|---|------------------------------------------------------------------|---------------------------------------------------------------|
| 1 | Boundary: \|\|u_theta - g_D\|\|_{dOmega} <= 1e-12                | tests/test_dirichlet_bc.py::test_pipeline_E_boundary_is_machine_precision (Task 1) |
| 2 | Initial condition: \|\|u_theta(0) - u_0\|\| <= 1e-12             | tests/test_exact_initial_condition.py (all four models, Task 7 extended it beyond MLP-only) |
| 3 | Certificate upper bound: E_h <= eta_guar, random test parameters | THIS FILE: test_eta_guar_bounds_true_error_across_random_mu   |
| 4 | PCG safety: eta_guar^(k) remains an upper bound, several k       | tests/test_riesz_bracket.py (Gate 2)                          |
| 5 | Residual consistency: R_h(u_h) ~= 0 on the exact trajectory      | tests/test_reference_residual.py (Gate 1)                     |
| 6 | Correction: E(u_corr) <= E(u_theta)                              | THIS FILE: test_correction_never_increases_true_error          |
| 7 | SPD diagnostic: random mu sampling records violations, never skips | tests/test_spd_diagnostics.py (Task 4)                       |
| 8 | Checkpoint compat: mismatched N/Nt/domain raises                 | tests/test_checkpoint_provenance.py (Task 5)                  |

Rows 3 and 6 needed new tests: no existing test varied mu randomly while
checking E_h <= eta_guar (test_certificate_reference_solution.py checks it
at one FIXED mu, parametrized only over perturbation magnitude), and no
test explicitly compared a corrected trajectory's true error against the
raw (uncorrected) prediction's -- test_pipeline_predict_certify_correct.py
checks the corrected trajectory is close to the true one, not that
correction *improved on* the prediction it started from.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import torch

from cgmy.operators import OperatorCache
from cgmy.reference_solver import EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.constants import compute_stability_constants
from certification.guaranteed_certificate import eta_guar as compute_eta_guar
from certification.residuals import compute_all_residuals
from cgmy.dirichlet_bc import project_bc
from correction.krylov import compute_initial_coefficients, correct_trajectory
from data.sampling import load_parameter_box, sample_sobol
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import normalize_mu, stats_from_box

torch.set_default_dtype(torch.float64)

MESH = Mesh(N=32, p=3, x_min=float(np.log(100) - 6), x_max=float(np.log(100) + 6))
TIME_GRID_REQ = TimeGrid(n_steps=15)
PAYOFF = EuropeanPutPayoff(strike=100.0)


# ── Assumption 3: certificate upper bound, over RANDOM test parameters ──────


def _random_untrained_model(ops, seed: int) -> MLPCoeffNet:
    torch.manual_seed(seed)
    net = MLPCoeffNet(mu_dim=6, psi_dim=ops.n_dof, hidden_dims=(16, 16))
    for p in net.parameters():
        p.data = torch.randn_like(p) * 2.0  # deliberately poor/untrained -- see module docstring
    net.eval()
    return net


def test_eta_guar_bounds_true_error_across_random_mu():
    """E_h <= eta_guar (Task 7's "certificate upper bound" assumption),
    checked over SEVERAL RANDOM mu (not one fixed point) -- an untrained,
    deliberately poor MLP's prediction against the true reference
    trajectory, with E_h the same max-over-time-levels physical L2
    coefficient error exp02_effectivity.py uses (true_error_Ei), and
    eta_guar computed by certification's own machinery -- the same
    quantities the guaranteed certificate is actually meant to relate in
    production use, not a synthetic single-vector perturbation."""
    box = load_parameter_box("configs/base.yaml")
    stats = stats_from_box(box)
    ops = OperatorCache(MESH, MESH.x_min, MESH.x_max)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = sample_sobol(box, 8, seed=717171)

    n_checked = 0
    for i, (mu, payoff) in enumerate(samples):
        const = compute_stability_constants(mu, ops)
        if not const.is_spd:
            continue  # not every random mu is SPD (see notes/04_spd_study.md) -- skip, don't fabricate a bound

        traj = solve_cgmy(mu, PAYOFF, MESH, TIME_GRID_REQ, mode="toeplitz")
        net = _random_untrained_model(ops, seed=100 + i)
        mu_tilde = normalize_mu(mu, stats)
        psi = traj.initial_coefficients  # any fixed-length vector works as psi for this untrained net
        pred_raw = net.predict_trajectory(mu_tilde, psi, traj.initial_coefficients, traj.time_grid, mu.T)
        pred = project_bc(pred_raw, mu, PAYOFF, traj.mesh_information, traj.time_grid)

        diff = pred - traj.coefficients
        E_h = float(max(np.sqrt(max(d @ ops.apply_M(d), 0.0)) for d in diff))  # true_error_Ei, exp_common's definition

        traj_pred = dataclasses.replace(traj, coefficients=pred)
        residuals = compute_all_residuals(traj_pred, mu, ops)
        dt = traj.solver_metadata.dt
        guar = compute_eta_guar(
            residuals, mu, ops, dt, const.alpha_certified, max_pcg_iterations=200, preconditioner="mass",
        )

        assert E_h <= guar.eta * (1 + 1e-6), (
            f"mu={mu}: E_h={E_h:.6e} > eta_guar={guar.eta:.6e} -- CLAUDE.md Hard "
            f"Invariant #5 violated for this (untrained, deliberately poor) prediction"
        )
        n_checked += 1

    assert n_checked >= 3, (
        f"only {n_checked}/{len(samples)} sampled mu were SPD -- not enough to "
        f"meaningfully exercise this assumption; widen the sample or check the seed"
    )


# ── Assumption 6: correction never increases the true error ─────────────────


def test_correction_never_increases_true_error():
    """E(u_corr) <= E(u_theta) (Task 7's "correction" assumption): starting
    from a genuinely poor (untrained, deliberately-large-weight) neural
    prediction, correction.krylov.correct_trajectory's output must be at
    least as close to the true reference trajectory as the prediction it
    was warm-started from -- never worse. Uses the same E_h definition as
    the certificate-bound test above, for a consistent, meaningful unit."""
    box = load_parameter_box("configs/base.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, payoff = sample_sobol(box, 1, seed=90210)[0]

    ops = OperatorCache(MESH, MESH.x_min, MESH.x_max)
    const = compute_stability_constants(mu, ops)
    assert const.is_spd, "test setup: chosen mu must be SPD"

    traj = solve_cgmy(mu, PAYOFF, MESH, TIME_GRID_REQ, mode="toeplitz")
    stats = stats_from_box(box)
    net = _random_untrained_model(ops, seed=55)
    mu_tilde = normalize_mu(mu, stats)

    c0, _, _ = compute_initial_coefficients(mu, PAYOFF, traj.mesh_information, ops)
    pred_raw = net.predict_trajectory(mu_tilde, traj.initial_coefficients, c0, traj.time_grid, mu.T)
    pred = project_bc(pred_raw, mu, PAYOFF, traj.mesh_information, traj.time_grid)

    def true_error(coeffs: np.ndarray) -> float:
        diff = coeffs - traj.coefficients
        return float(max(np.sqrt(max(d @ ops.apply_M(d), 0.0)) for d in diff))

    E_prediction = true_error(pred)
    assert E_prediction > 1e-2, (
        "test setup: the untrained prediction should be genuinely poor, "
        "otherwise this test doesn't exercise the claim"
    )

    correction = correct_trajectory(
        mu, PAYOFF, traj.mesh_information, traj.time_grid, ops,
        seed_coefficients=pred, warm_start=True,
    )
    E_corrected = true_error(correction.coefficients)

    assert E_corrected <= E_prediction, (
        f"correction made the true error WORSE: E(u_corr)={E_corrected:.6e} > "
        f"E(u_theta)={E_prediction:.6e}"
    )
    # Correction should not just "not be worse" -- it should actually be
    # close to converged (GMRES rtol=1e-10), i.e. much better, not merely tied.
    assert E_corrected < 1e-6, f"corrected trajectory is not near-converged: E={E_corrected:.3e}"
