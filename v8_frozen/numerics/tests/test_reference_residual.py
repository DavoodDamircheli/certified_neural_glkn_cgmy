"""GATE 1 (CLAUDE.md): the discrete residual of the exact reference trajectory
must vanish at solver tolerance, and must grow unambiguously and
proportionally under perturbation. This is the mathematical precondition
for every later certification stage.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from cgmy.operators import OperatorCache
from cgmy.reference_solver import (
    CGMYParameters,
    EuropeanPutPayoff,
    GalerkinTrajectory,
    Mesh,
    MeshInfo,
    SolverMetadata,
    TimeGrid,
    solve_cgmy,
)
from certification.residuals import (
    compute_all_residuals,
    compute_be_residual,
    compute_cn_residual,
    residual_norm,
)

MU = CGMYParameters(C=1.0, G=5.0, M=5.0, Y=0.5, r=0.05, T=0.5)
PAYOFF = EuropeanPutPayoff(strike=100.0)


def _make_trajectory(N: int = 64, N_tau: int = 50, mode: str = "toeplitz") -> GalerkinTrajectory:
    mesh = Mesh(N=N, p=3)
    tg = TimeGrid(n_steps=N_tau)
    return solve_cgmy(MU, PAYOFF, mesh, tg, mode=mode)


def _cache_for(traj: GalerkinTrajectory) -> OperatorCache:
    mesh = Mesh(N=traj.mesh_information.N, p=traj.mesh_information.p)
    return OperatorCache(mesh, traj.mesh_information.x_min, traj.mesh_information.x_max)


# ── Gate 1a: exact trajectory residual at solver tolerance ───────────────────


def test_gate1a_exact_trajectory_residual_at_solver_tolerance():
    """Insert the exact stored coefficients; max_n ||r^{n+1/2}|| must be tiny.

    Uses mode='toeplitz' deliberately: OperatorCache.apply_A is the exact
    (untruncated) Toeplitz operator, identical to what solve_cn's
    mode='toeplitz' actually solved against (verified in Stage A). Using
    mode='banded' here would compare against a *different*, bandwidth-
    truncated operator and spuriously show a large residual -- that would
    be a real discrepancy, but it is the "wrong forcing/operator reused"
    failure mode, not a bug in the residual formula.
    """
    traj = _make_trajectory()
    assert set(traj.solver_metadata.scheme_per_step) == {"CN"}, (
        "this is the pure-CN coverage for Gate 1c; the trusted solver "
        "currently produces only CN steps (see Stage A findings)"
    )
    ops = _cache_for(traj)

    residuals = compute_all_residuals(traj, MU, ops)
    norms = np.array(
        [residual_norm(residuals[n], ops, traj.solver_metadata.bc_dofs)
         for n in range(traj.solver_metadata.n_steps)]
    )
    max_norm = float(norms.max())
    argmax_n = int(norms.argmax())

    print(f"\nGate 1a: max_n ||r^(n+1/2)|| (interior dofs) = {max_norm:.6e}  "
          f"(at step n={argmax_n} of {traj.solver_metadata.n_steps})")
    print(f"Gate 1a: mean_n ||r^(n+1/2)|| = {norms.mean():.6e}")
    print(f"Gate 1a: GMRES rtol used by the solver = "
          f"{traj.solver_metadata.tolerance_per_step[0]}, "
          f"mean GMRES iters/step = {traj.solver_metadata.mean_gmres_iters:.1f}")

    # GMRES converges the linear system at each step to a relative residual
    # of rtol=1e-10; this reported number is the actual max, not a tuned
    # threshold -- see printed value above.
    assert max_norm < 1e-6, f"max_n ||r|| = {max_norm:.3e}, expected well below 1e-6"


def test_gate1a_banded_mode_shows_truncation_not_solver_error():
    """Diagnostic companion to Gate 1a (not itself a gate requirement):
    evaluating the exact-operator residual against a mode='banded'
    trajectory (bandwidth=64, truncated) should show a *larger* residual
    than the toeplitz case, reflecting the banded truncation error -- not a
    bug, and a concrete illustration of the "wrong operator reused" failure
    mode in the Gate 1 checklist.
    """
    traj_exact = _make_trajectory(mode="toeplitz")
    traj_banded = _make_trajectory(mode="banded")
    ops = _cache_for(traj_exact)

    def max_interior_norm(traj):
        r = compute_all_residuals(traj, MU, ops)
        return max(
            residual_norm(r[n], ops, traj.solver_metadata.bc_dofs)
            for n in range(traj.solver_metadata.n_steps)
        )

    exact_norm = max_interior_norm(traj_exact)
    banded_norm = max_interior_norm(traj_banded)
    print(f"\nmode=toeplitz max ||r|| = {exact_norm:.3e}; "
          f"mode=banded max ||r|| = {banded_norm:.3e}")
    assert banded_norm > 100 * exact_norm


# ── Gate 1b: perturbation must grow the residual proportionally ──────────────


def _perturb_step(traj: GalerkinTrajectory, n: int, rel_eps: float, rng: np.random.Generator) -> GalerkinTrajectory:
    coeffs = traj.coefficients.copy()
    c = coeffs[n + 1]
    direction = rng.standard_normal(c.shape)
    direction /= np.linalg.norm(direction)
    coeffs[n + 1] = c + rel_eps * np.linalg.norm(c) * direction
    return dataclasses.replace(traj, coefficients=coeffs)


@pytest.mark.parametrize("rel_eps", [1e-6, 1e-4, 1e-2])
def test_gate1b_perturbation_grows_residual_unambiguously(rel_eps: float):
    traj = _make_trajectory()
    ops = _cache_for(traj)
    n = traj.solver_metadata.n_steps // 2

    baseline_norm = residual_norm(
        compute_cn_residual(traj, MU, ops, n), ops, traj.solver_metadata.bc_dofs
    )

    rng = np.random.default_rng(0)
    traj_pert = _perturb_step(traj, n, rel_eps, rng)
    perturbed_norm = residual_norm(
        compute_cn_residual(traj_pert, MU, ops, n), ops, traj.solver_metadata.bc_dofs
    )

    print(f"\nGate 1b: rel_eps={rel_eps:.0e}  baseline ||r||={baseline_norm:.3e}  "
          f"perturbed ||r||={perturbed_norm:.3e}")

    assert perturbed_norm > 1e3 * baseline_norm, (
        "perturbed residual must be unambiguously nonzero relative to the "
        "unperturbed (solver-tolerance-level) residual"
    )


def test_gate1b_perturbation_growth_is_proportional():
    """The residual is affine in c^{n+1}, so perturbing it by a scaled random
    direction should scale the residual by (approximately) the same factor
    -- checked directly here across three orders of magnitude of rel_eps.
    """
    traj = _make_trajectory()
    ops = _cache_for(traj)
    n = traj.solver_metadata.n_steps // 2

    norms = {}
    for rel_eps in (1e-6, 1e-4, 1e-2):
        rng = np.random.default_rng(0)  # same direction at each scale
        traj_pert = _perturb_step(traj, n, rel_eps, rng)
        norms[rel_eps] = residual_norm(
            compute_cn_residual(traj_pert, MU, ops, n), ops, traj.solver_metadata.bc_dofs
        )

    ratio_1 = norms[1e-4] / norms[1e-6]
    ratio_2 = norms[1e-2] / norms[1e-4]
    print(f"\nGate 1b proportionality: ||r||(1e-4)/||r||(1e-6) = {ratio_1:.1f} (expect ~100)")
    print(f"Gate 1b proportionality: ||r||(1e-2)/||r||(1e-4) = {ratio_2:.1f} (expect ~100)")

    assert 50.0 < ratio_1 < 200.0
    assert 50.0 < ratio_2 < 200.0


# ── Gate 1c: Rannacher-startup (BE) coverage ──────────────────────────────────


def test_gate1c_synthetic_be_step_residual_at_solver_tolerance():
    """The trusted solver has no Rannacher/BE startup (Stage A finding: it
    runs pure Crank-Nicolson from tau=0). There is therefore no real
    trajectory to exercise compute_be_residual against. This constructs a
    single synthetic backward-Euler step directly from OperatorCache's
    apply_M/apply_A (the same, single operator implementation used
    everywhere else in this repo, not a new one), solves it to GMRES
    tolerance, and checks compute_be_residual recovers ~0 on it. This tests
    the BE residual *formula*, not a claim that the trusted solver produces
    BE steps -- see CLAUDE.md's "Key objects" note that a matching BE
    residual "exists"; it exists as a formula here, ready for when/if
    Rannacher startup is added to the trusted solver.
    """
    N, p, x_min, x_max = 32, 3, -3.0, 3.0
    mesh = Mesh(N=N, p=p, x_min=x_min, x_max=x_max)
    ops = OperatorCache(mesh, x_min, x_max)
    n_dof = ops.n_dof
    dt = 0.01

    rng = np.random.default_rng(42)
    c0 = rng.standard_normal(n_dof)
    f1 = 0.1 * rng.standard_normal(n_dof)

    def be_matvec(v: np.ndarray) -> np.ndarray:
        return ops.apply_M(v) + dt * ops.apply_A(MU, v)

    be_op = spla.LinearOperator((n_dof, n_dof), matvec=be_matvec, dtype=float)
    rhs = ops.apply_M(c0) + dt * f1
    c1, info = spla.gmres(be_op, rhs, rtol=1e-12, restart=50, maxiter=1000)
    assert info == 0, f"synthetic BE step GMRES did not converge: info={info}"

    mesh_info = MeshInfo(
        N=N, p=p, n_dof=n_dof, h=(x_max - x_min) / N,
        x_min=x_min, x_max=x_max, nodes=np.linspace(x_min, x_max, N + 1),
    )
    metadata = SolverMetadata(
        mode="synthetic_be_for_gate1c",
        dt=dt, n_steps=1, scheme_per_step=["BE"],
        iterations_per_step=[0], tolerance_per_step=[1e-12],
        bc_dofs=([], []), cpu_time=0.0, mean_gmres_iters=None,
    )
    traj = GalerkinTrajectory(
        coefficients=np.stack([c0, c1]),
        forcing_vectors=np.stack([f1]),
        initial_coefficients=c0,
        time_grid=np.array([0.0, dt]),
        parameter_vector=MU,
        payoff=PAYOFF,
        mesh_information=mesh_info,
        solver_metadata=metadata,
    )

    r = compute_be_residual(traj, MU, ops, 0)
    norm = residual_norm(r, ops, ([], []))
    print(f"\nGate 1c (synthetic BE step): ||r|| = {norm:.6e}")
    assert norm < 1e-8

    all_r = compute_all_residuals(traj, MU, ops)
    assert np.allclose(all_r[0], r), "compute_all_residuals must dispatch to compute_be_residual for scheme='BE'"


# ── Missing-forcing must raise, never recompute or substitute ────────────────


def test_missing_forcing_raises_not_recomputes():
    traj = _make_trajectory(N=32, N_tau=10)
    traj_no_forcing = dataclasses.replace(traj, forcing_vectors=None)
    ops = _cache_for(traj)

    with pytest.raises(ValueError, match="forcing_vectors is None"):
        compute_cn_residual(traj_no_forcing, MU, ops, 0)
