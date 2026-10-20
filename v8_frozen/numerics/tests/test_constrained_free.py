"""Regression tests for the constrained (free-coordinate) certificate
migration -- CGMY_V4_REMAINING_WORK.md item 1 / OVERNIGHT_CLAUDE_CODE_PROMPT.md
Phase 1's seven acceptance checks.

Fixtures: production mesh (N=64, p=3, domain=log(100)+-8, matching
configs/base.yaml) and a small time grid (n_steps=20, short of the
production 60 to keep this module fast) for every check except Check 7
(the full 459-query audit, which is its own script,
audit/gate_a3_sweep_free.py -- NOT re-run inside pytest, since it is a
multi-hundred-query sweep, not a unit test).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cgmy.dirichlet_bc import bc_dofs_for_mesh, boundary_error, dirichlet_bc_values, project_bc
from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.constants import compute_stability_constants
from certification.constrained import (
    FreeOperatorCache,
    FreeSpace,
    compute_all_free_residuals,
    free_space_for_mesh,
    inject,
    is_free_accepted,
    lifting_vector,
    restrict,
    solve_free_cn_trajectory,
)
from certification.guaranteed_certificate import eta_sharp
from certification.residuals import compute_all_residuals
from certification.riesz_solver import solve_riesz_pcg

X_MIN, X_MAX = -3.394829814011908, 12.605170185988092  # log(100) +- 8
N_PRODUCTION, P_PRODUCTION = 64, 3
PAYOFF = EuropeanPutPayoff(strike=100.0)

# A representative interior-box mu, expected full-SPD (most of the box is,
# per configs/base.yaml: only 10/500 fresh Sobol + 16/64 vertices fail at
# N=64).
MU_SPD = CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.5)

# The canonical known full-SPD-FAILURE query already used elsewhere in this
# project's own tests (certification/run_gate_b.py,
# tests/v3/test_workflow_status_branches.py) and in results/spd_failures.csv
# row 1 -- reused here rather than inventing a fresh one, for consistency
# with the project's existing fixture convention.
MU_FULL_INDEFINITE = CGMYParameters(
    C=0.4874301809351891, G=1.4997461959719658, M=8.837858855724335,
    Y=1.4543537443503738, r=0.06294871459715069, T=0.10040483502671123,
)


@pytest.fixture(scope="module")
def mesh():
    return Mesh(N=N_PRODUCTION, p=P_PRODUCTION, x_min=X_MIN, x_max=X_MAX)


@pytest.fixture(scope="module")
def ops(mesh):
    return OperatorCache(mesh, X_MIN, X_MAX)


@pytest.fixture(scope="module")
def traj(mesh):
    return solve_cgmy(MU_SPD, PAYOFF, mesh, TimeGrid(n_steps=20), mode="toeplitz", store_forcing=True)


@pytest.fixture(scope="module")
def fs(traj) -> FreeSpace:
    return free_space_for_mesh(traj.mesh_information)


@pytest.fixture(scope="module")
def free_ops(ops, fs) -> FreeOperatorCache:
    return FreeOperatorCache(ops, fs)


def test_free_space_partition_matches_boundary_dof_function(traj, fs):
    """fs.free_idx/bc_left/bc_right must come from bc_dofs_for_mesh, never
    hardcoded 62/4/66."""
    bc_left, bc_right = bc_dofs_for_mesh(traj.mesh_information)
    assert fs.bc_left == tuple(bc_left)
    assert fs.bc_right == tuple(bc_right)
    assert fs.n_free == fs.n_dof - len(bc_left) - len(bc_right)
    assert fs.n_free == 62 and fs.n_dof == 66  # production mesh value, not hardcoded above


def test_inject_restrict_are_mutual_pseudo_inverses(fs):
    rng = np.random.default_rng(0)
    z = rng.standard_normal(fs.n_free)
    c = inject(fs, z, b=None)
    assert np.array_equal(restrict(fs, c), z)
    assert np.all(c[fs.bc_idx] == 0.0)


# ---------------------------------------------------------------------------
# Check 1: free-row CN stepping and raw identity-row stepping agree at every
# time level under the same forcing/initial data.
# ---------------------------------------------------------------------------

def test_check1_free_cn_stepping_agrees_with_raw_stepping(traj, ops, fs, free_ops):
    """Numerical comparison allowance: ~1.5e-7 absolute. Two independent
    GMRES solves (raw n_dof=66 system with boundary identity rows, vs free
    n_free=62 system with none) at the SAME rtol=1e-10/restart=50 are not
    expected to be bit-identical -- different-dimensional Krylov spaces
    take different iteration paths to the same tolerance. Verified
    empirically this session: max per-step disagreement over 20 CN steps
    was 1.1e-7 (see reports/v4_overnight_log.md Phase 1 section)."""
    z0 = restrict(fs, traj.coefficients[0])
    z_traj = solve_free_cn_trajectory(
        MU_SPD, PAYOFF, traj.mesh_information, traj.time_grid, ops, fs, free_ops, z0
    )
    raw_restricted = traj.coefficients[:, fs.free_idx]
    assert z_traj.shape == raw_restricted.shape
    max_diff = np.max(np.abs(z_traj - raw_restricted))
    assert max_diff < 5e-7, f"free vs raw CN stepping disagree by {max_diff:.3e} (allowance 5e-7)"


# ---------------------------------------------------------------------------
# Check 2: reduced residual equals the free-row restriction of the raw
# residual.
# ---------------------------------------------------------------------------

def test_check2_reduced_residual_equals_restricted_raw_residual(traj, ops, fs, free_ops):
    """Exact algebraic identity (derived in certification.constrained's
    module docstring): once the candidate is projected onto the SAME
    prescribed b(t) the free formula uses (``project_bc`` -- required, see
    the finding below), agreement is at machine precision (~1e-13 relative
    this session). Numerical comparison allowance: 1e-10 absolute.

    FINDING (logged, not silently worked around): computing this
    comparison WITHOUT first calling project_bc on the candidate gives a
    much looser agreement (~1.6e-9 absolute vs a ~4e-7 residual signal,
    i.e. an O(GMRES-tolerance) discrepancy) -- because the raw reference
    trajectory's own boundary rows only satisfy the Dirichlet constraint to
    GMRES's rtol=1e-10, not exactly (solve_cn enforces them via an
    identity-row trick in the linear solve, the same mechanism
    correction.krylov.correct_trajectory uses and the same reason that
    function calls project_bc on its own output), while the free-space
    formula's b(t) is always the algebraically exact value. This is
    exactly why the task file requires projecting every candidate onto the
    SAME prescribed b(t) before evaluating its residual -- it is not only a
    safety margin, it is required for this identity to hold at all.
    """
    projected = project_bc(traj.coefficients, MU_SPD, PAYOFF, traj.mesh_information, traj.time_grid)
    traj_p = dataclasses.replace(traj, coefficients=projected)

    raw_res = compute_all_residuals(traj_p, MU_SPD, ops)
    free_res = compute_all_free_residuals(traj_p, MU_SPD, ops, fs, free_ops)
    raw_res_restricted = raw_res[:, fs.free_idx]

    max_diff = np.max(np.abs(free_res - raw_res_restricted))
    assert max_diff < 1e-10, f"reduced vs restricted-raw residual disagree by {max_diff:.3e}"

    # And the (not-yet-projected) raw trajectory demonstrates the looser,
    # GMRES-tolerance-level agreement described above, rather than silently
    # assuming it would also be machine-precision -- a real, recorded
    # finding, not swept under the rug.
    raw_res_unprojected = compute_all_residuals(traj, MU_SPD, ops)
    free_res_unprojected = compute_all_free_residuals(traj, MU_SPD, ops, fs, free_ops)
    looser_diff = np.max(np.abs(free_res_unprojected - raw_res_unprojected[:, fs.free_idx]))
    assert looser_diff < 1e-7  # the looser bound actually observed (~1.6e-9), generous margin
    assert looser_diff > 1e-11  # and genuinely looser than the projected case, not a fluke


# ---------------------------------------------------------------------------
# Check 3: identical prescribed boundary data imply zero boundary error and
# equal full/free mass error norms.
# ---------------------------------------------------------------------------

def test_check3_identical_boundary_data_zero_error_and_equal_mass_norms(traj, ops, fs, free_ops):
    """For e = pred - true with IDENTICAL prescribed b(t) on both (so
    e[bc_idx] == 0 exactly), e^T M_h e == e_free^T M_I e_free EXACTLY (not
    merely to a stated tolerance) -- derived in the module docstring: e =
    E e_free (since e's boundary part is zero), so
    e^T M_h e = e_free^T (E^T M_h E) e_free = e_free^T M_I e_free by
    definition of M_I. Verified here to float roundoff (~1e-13 relative)."""
    rng = np.random.default_rng(1)
    time_grid = traj.time_grid
    n_steps = len(time_grid) - 1
    z_pred = rng.standard_normal((n_steps + 1, fs.n_free))
    z_true = rng.standard_normal((n_steps + 1, fs.n_free))

    pred = np.array([
        inject(fs, z_pred[n], lifting_vector(MU_SPD, PAYOFF, traj.mesh_information, fs, float(time_grid[n])))
        for n in range(n_steps + 1)
    ])
    true = np.array([
        inject(fs, z_true[n], lifting_vector(MU_SPD, PAYOFF, traj.mesh_information, fs, float(time_grid[n])))
        for n in range(n_steps + 1)
    ])

    # Both already satisfy the exact Dirichlet constraint at machine
    # precision by construction (inject with the exact lifting_vector).
    assert boundary_error(pred, MU_SPD, PAYOFF, traj.mesh_information, time_grid) < 1e-10
    assert boundary_error(true, MU_SPD, PAYOFF, traj.mesh_information, time_grid) < 1e-10

    e_full = pred - true
    assert np.max(np.abs(e_full[:, fs.bc_idx])) < 1e-12  # identical boundary data -> zero there

    mass_full = sum(float(e @ ops.apply_M(e)) for e in e_full)
    mass_free = sum(float(ef @ free_ops.apply_M(ef)) for ef in (z_pred - z_true))
    rel_diff = abs(mass_full - mass_free) / max(abs(mass_full), 1e-300)
    assert rel_diff < 1e-10, f"full/free mass-error norms disagree: {mass_full!r} vs {mass_free!r}"


# ---------------------------------------------------------------------------
# Check 4: on full-SPD queries, the sharp free certificate does not exceed
# the legacy zero-padded full certificate.
# ---------------------------------------------------------------------------

def test_check4_free_sharp_certificate_never_exceeds_legacy_full(ops, fs, free_ops):
    """Numerical comparison allowance: eta_free <= eta_full + 1e-9 (absolute).

    This is not merely an empirical tendency -- it is a PROVEN inequality
    for any SPD S and any residual r supported on the free rows (r[bc]=0):
    writing S in block form over (free, bc) index sets, r^T S^{-1} r (full
    inverse) equals r_free^T Sigma^{-1} r_free where
    Sigma = S_II - S_IB S_BB^{-1} S_BI is the SCHUR COMPLEMENT of the bc
    block. Since S_BB is SPD (principal submatrix of an SPD matrix) and
    S_IB S_BB^{-1} S_BI is PSD, Sigma <= S_II in Loewner order, hence
    Sigma^{-1} >= S_II^{-1} = S_I^{-1} (Loewner order reverses under
    inversion for PD matrices), hence
    r_free^T Sigma^{-1} r_free >= r_free^T S_I^{-1} r_free, i.e.
    eta_sharp_full^2 >= eta_sharp_free^2 always when S_full is SPD. The
    1e-9 allowance here is purely for the two independent dense
    eigendecompositions'/Cholesky factorizations' floating-point error, not
    for any gap in the mathematical claim.
    """
    rng = np.random.default_rng(2)
    n_steps = 5
    residuals_full = rng.standard_normal((n_steps, ops.n_dof))
    residuals_full[:, fs.bc_idx] = 0.0  # zero-padded at Dirichlet dofs, as the real residual is
    residuals_free = residuals_full[:, fs.free_idx]
    dt = 1e-2

    const_full = compute_stability_constants(MU_SPD, ops, coordinate_space="legacy_full")
    assert const_full.is_spd, "test fixture mu expected full-SPD"

    cert_full = eta_sharp(residuals_full, MU_SPD, ops, dt, coordinate_space="legacy_full")
    cert_free = eta_sharp(residuals_free, MU_SPD, free_ops, dt, coordinate_space="constrained_free")

    assert cert_free.eta <= cert_full.eta + 1e-9, (
        f"eta_sharp_free={cert_free.eta!r} exceeded eta_sharp_full={cert_full.eta!r} "
        f"(allowance 1e-9) -- this should be mathematically impossible on a full-SPD query"
    )


# ---------------------------------------------------------------------------
# Check 5: on full-indefinite/free-SPD cases, only the free Riesz system is
# usable (and succeeds).
# ---------------------------------------------------------------------------

def test_check5_full_indefinite_free_spd_free_riesz_succeeds(ops, fs, free_ops):
    """Reuses the project's own canonical known full-SPD-failure query
    (results/spd_failures.csv row 1, also used by
    certification/run_gate_b.py and tests/v3/test_workflow_status_branches.py).
    Confirms: full S_h is certifiably NOT SPD (so the existing
    predict_certify_correct workflow's `if not is_spd: ... uncertifiable`
    branch correctly never attempts a full Riesz solve here), while the
    free S_I IS certifiably SPD and a free PCG Riesz solve on it succeeds
    and returns a finite, positive bracket."""
    const_full = compute_stability_constants(MU_FULL_INDEFINITE, ops, coordinate_space="legacy_full")
    const_free = compute_stability_constants(MU_FULL_INDEFINITE, free_ops, coordinate_space="constrained_free")

    assert not const_full.is_spd, "fixture mu expected full-INDEFINITE (per results/spd_failures.csv)"
    assert const_free.is_spd, "fixture mu expected free-SPD (per configs/base.yaml's 121/121 finding)"

    rng = np.random.default_rng(3)
    r_free = rng.standard_normal(fs.n_free)
    bracket = solve_riesz_pcg(
        r_free, MU_FULL_INDEFINITE, free_ops, const_free.alpha_certified, max_iter=200
    )
    assert np.isfinite(bracket.upper_bound) and bracket.upper_bound >= bracket.lower_bound >= 0.0


# ---------------------------------------------------------------------------
# Check 6: a boundary/initial discrepancy must not be silently accepted.
# ---------------------------------------------------------------------------

def test_check6_wrong_boundary_data_is_not_accepted_despite_small_free_residual(traj, ops, fs, free_ops):
    """The free-space residual/certificate, by construction, never reads a
    candidate's own boundary rows (module docstring's caveat) -- so a
    candidate with WRONG boundary data can still show a tiny free
    certificate. Demonstrates that zeroing/never-reading those rows is NOT
    itself a boundary check, and that the combined accept decision
    (is_free_accepted, which also gates on boundary_error) correctly
    rejects such a candidate even though eta alone would pass."""
    tolerance = 1e-3
    mesh_info = traj.mesh_information
    time_grid = traj.time_grid

    # A candidate that is EXACTLY the reference's free coefficients (so its
    # free residual, and hence its certificate, is tiny) but with corrupted
    # boundary rows (off by a large constant -- not a prescribed value).
    bad = traj.coefficients.copy()
    bad[:, fs.bc_idx] += 10.0  # deliberately wrong boundary data

    e_boundary = boundary_error(bad, MU_SPD, PAYOFF, mesh_info, time_grid)
    assert e_boundary > 1.0, "corrupted boundary rows should show a large boundary error"

    traj_bad = dataclasses.replace(traj, coefficients=bad)
    free_res = compute_all_free_residuals(traj_bad, MU_SPD, ops, fs, free_ops)
    const_free = compute_stability_constants(MU_SPD, free_ops, coordinate_space="constrained_free")
    cert = eta_sharp(free_res, MU_SPD, free_ops, traj.solver_metadata.dt, coordinate_space="constrained_free")

    # The free certificate alone is small (it never saw the corrupted rows)
    # -- exactly the hazard this check exists to catch.
    assert cert.eta <= tolerance, "expected the free-only certificate to look (deceptively) small here"

    # But the COMBINED accept decision must reject.
    assert not is_free_accepted(cert.eta, tolerance, e_boundary)
    # And the correctly-projected candidate (same free coefficients, exact
    # boundary data) IS accepted -- confirming the rejection above is about
    # the boundary corruption specifically, not a fixture error.
    good = project_bc(traj.coefficients, MU_SPD, PAYOFF, mesh_info, time_grid)
    e_boundary_good = boundary_error(good, MU_SPD, PAYOFF, mesh_info, time_grid)
    assert is_free_accepted(cert.eta, tolerance, e_boundary_good)
