"""Gate A2 soundness tests: the independent assembly (audit/independent_assembly.py)
agrees with production at a stated tolerance, and the test is not vacuous
(a deliberately perturbed production matrix is caught)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vendor"))
sys.path.insert(0, str(REPO_ROOT / "audit"))

from independent_assembly import (  # noqa: E402
    assert_independent,
    independent_mass_matrix,
    independent_derivative_matrix,
    independent_A_h_on_mesh,
)
from cgmy_bspline.matrices import mass_matrix, derivative_matrix, operator_matrix  # noqa: E402
from cgmy_bspline.parameters import CGMYParams  # noqa: E402


def test_module_never_imports_audited_routines():
    assert_independent()


def test_mass_matrix_agrees_with_production_N32():
    N, p, h, x_min = 32, 3, 0.25, -3.394829814011908
    M_prod = mass_matrix(N, p, h, x_min).toarray()
    M_indep = independent_mass_matrix(N, p)
    assert np.max(np.abs(M_prod - M_indep)) < 1e-12, (
        "independent M_h should agree with production to ~machine precision "
        "(both integrate the exact same cardinal B-spline product exactly)"
    )


def test_derivative_matrix_agrees_with_production_N32():
    N, p, h, x_min = 32, 3, 0.25, -3.394829814011908
    D_prod = derivative_matrix(N, p, h, x_min).toarray()
    D_indep = independent_derivative_matrix(N, p) / h
    assert np.max(np.abs(D_prod - D_indep)) < 1e-12


def test_perturbed_mass_matrix_is_caught():
    """Sanity check that the comparison above is not vacuous: a
    deliberately corrupted production matrix must NOT pass."""
    N, p, h, x_min = 32, 3, 0.25, -3.394829814011908
    M_prod = mass_matrix(N, p, h, x_min).toarray()
    M_indep = independent_mass_matrix(N, p)
    M_perturbed = M_prod.copy()
    M_perturbed[3, 3] += 0.01  # a real, deliberate corruption well above any quadrature noise
    assert np.max(np.abs(M_perturbed - M_indep)) > 1e-6


def test_full_A_h_eigs_agree_at_N16():
    """Full independent A_h assembly (M_h + D_h + independently-quadratured
    K_G/K_M, combined via the shared linear formula) matches production's
    smallest/largest S_h(mu) eigenvalues to a tolerance consistent with the
    known, explicitly-quantified bandwidth=64 discretization gap in the
    fractional-kernel blocks (see reports/GATE_A_REPORT.md's A2 section for
    the per-lag convergence study that establishes this is a discretization
    effect, not a bug)."""
    N, p = 16, 3
    x_min = -3.394829814011908
    h = 0.25
    x_max = x_min + N * h
    C, G, M, Y, r = 0.3, 2.5, 5.0, 1.5, 0.05

    res = independent_A_h_on_mesh(N, p, x_min, x_max, C, G, M, Y, r)
    params = CGMYParams(C=C, G=G, M=M, Y=Y, r=r, K=100.0, T=1.0)
    mats = operator_matrix(N, p, h, x_min, params, bandwidth=64)
    A_prod = mats["A_h"].toarray()
    S_prod = 0.5 * (A_prod + A_prod.T)
    eigs_prod = np.linalg.eigvalsh(S_prod)

    assert abs(res["lambda_min_S"] - eigs_prod[0]) < 1e-5
    assert abs(res["lambda_max_S"] - eigs_prod[-1]) < 1e-5
    assert np.max(np.abs(res["A_h"] - A_prod)) < 1e-2  # bandwidth=64 discretization gap, not a bug


def test_omega_S_matches_production_closed_form():
    """omega_S (the martingale/drift correction) is independently
    re-derived from the Levy-Khintchine representation (not imported from
    cgmy_bspline.parameters) and must match production's closed form
    exactly (both state the identical, unambiguous textbook formula)."""
    from independent_assembly import _omega_S_and_lambda0_independent
    from cgmy_bspline.parameters import omega_stock

    C, G, M, Y, r = 0.3, 2.5, 5.0, 1.5, 0.05
    params = CGMYParams(C=C, G=G, M=M, Y=Y, r=r, K=100.0, T=1.0)
    om_prod = omega_stock(params)
    om_indep, _, _ = _omega_S_and_lambda0_independent(C, G, M, Y, r)
    assert abs(om_prod - om_indep) < 1e-10


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
