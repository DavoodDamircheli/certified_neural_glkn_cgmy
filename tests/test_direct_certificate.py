"""Plan 1.4 (certificate) and 1.5 (direct solver) done-when criteria."""
import numpy as np
import pytest

from conftest import FULL
from cgmy_cert.params import Query, Mesh, sobol_queries
from cgmy_cert.symbols import free_matrices
from cgmy_cert.spectral import spectral_data
from cgmy_cert.direct import (reference_trajectory, cn_march, free_forcing, initial_free_vector,
                              mass_norm, lifting, payoff_load_vector, raw_basis_eval)
from cgmy_cert.certificate import certify, cn_residuals

N_T = 60


def _pod_candidate(fs, Z, r):
    """Mass-weighted rank-r projection of a trajectory onto its own POD modes (a stand-in for a
    reduced candidate until Phase 2): keeps z^0 exact -> beta_0 = 0 unless perturbed."""
    from cgmy_cert.symbols import _dense_banded
    M = _dense_banded(fs.M_entries, fs.n)
    Lc = np.linalg.cholesky(M)
    U, s, Vt = np.linalg.svd(Lc.T @ Z, full_matrices=False)
    P = np.linalg.solve(Lc.T, U[:, :r])          # mass-orthonormal modes
    return P @ (P.T @ (M @ Z))


def test_payoff_projection_and_lifting():
    q = Query(0.5, 3.0, 5.0, 1.5, 0.05, T=0.5)
    msh = Mesh(64)
    b0 = payoff_load_vector(msh, q.K)
    # partition of unity: sum_k int phi_k psi = h^{-1/2} int psi  (quadrature check on an exact integral)
    x = np.linspace(msh.x_min, np.log(q.K), 200001)
    exact = np.trapezoid(np.maximum(q.K - np.exp(x), 0.0), x)
    assert abs(b0.sum() - exact / np.sqrt(msh.h)) < 1e-6 * exact
    bL = lifting(q, msh, np.array([0.0, 0.5]))
    assert abs(bL[0] - q.K * np.sqrt(msh.h)) < 1e-14 and abs(bL[1] - q.K * np.exp(-0.025) * np.sqrt(msh.h)) < 1e-12
    z0 = initial_free_vector(q, msh)
    # the projected payoff is close to K sqrt(h) at the far left and 0 at the far right
    assert abs(z0[0] - q.K * np.sqrt(msh.h)) < 0.05 * q.K * np.sqrt(msh.h)
    assert abs(z0[-1]) < 1e-6


@pytest.mark.parametrize("N", [64, 128, 256, 512, 1024])
def test_lu_vs_gmres_references(N):
    """Dense-LU and circulant-GMRES references agree to 1e-12 in the M_I norm; GMRES iterations <= 15."""
    qs = sobol_queries(4 if N <= 256 or FULL else 2, seed=5)
    for q in qs:
        fs, forc, trL = reference_trajectory(q, N, N_T, solver="lu")
        trG = cn_march(fs, trL.Z[:, 0], q.T, N_T, solver="gmres", forcing=forc)
        d = max(mass_norm(fs, trL.Z[:, k] - trG.Z[:, k]) for k in range(N_T + 1))
        scale = max(mass_norm(fs, trL.Z[:, k]) for k in range(N_T + 1))
        assert d <= 1e-12 * scale, (q.query_id, N, d / scale)
        assert max(trG.iters) <= 15, (q.query_id, N, trG.iters)
        # the reference residual is at roundoff level
        R = cn_residuals(fs, trL.Z, forc)
        assert np.max(np.abs(R)) < 1e-9 * scale


@pytest.mark.parametrize("N", [2048, 4096])
@pytest.mark.skipif(not FULL, reason="set CGMY_FULL_TESTS=1")
def test_gmres_iterations_large_N(N):
    for q in sobol_queries(3, seed=6):
        fs, forc, tr = reference_trajectory(q, N, N_T, solver="gmres")
        assert max(tr.iters) <= 15


def test_perturbed_initial_data_effectivity_one(audit48):
    """48 'exact dynamics with perturbed z^0' cases (v8 Table 5): the augmented bound holds with
    effectivity 1.00 +- 1e-6 in both modes, and the residual-only estimator would underestimate."""
    rng = np.random.default_rng(2024)
    cases = audit48 if FULL else audit48[:3] + audit48[16:19] + audit48[32:35]
    for q in cases:
        fs, forc, trL = reference_trajectory(q, 64, N_T, solver="lu")
        spec = spectral_data(q, 64)
        z0 = trL.Z[:, 0]
        dz = rng.standard_normal(fs.n); dz *= 1e-4 / mass_norm(fs, dz)
        trP = cn_march(fs, z0 + dz, q.T, N_T, solver="lu", forcing=forc)
        err = max(mass_norm(fs, trP.Z[:, k] - trL.Z[:, k]) for k in range(N_T + 1))
        for mode in ("banded", "exact"):
            c = certify(fs, forc, trP.Z, z0, 1e-3, mode=mode, spec=spec)
            assert c.accepted
            assert abs(c.bound / err - 1.0) < 1e-6, (q.query_id, mode, c.bound, err)
            res_only = c.etaX if mode == "banded" else c.etaS_upper
            assert res_only < 0.5 * err


def test_candidates_ordering_etaS_etaX_Theta(audit48):
    """On reduced candidates: eta^S <= eta^X <= sqrt(Theta) eta^S, both bounds hold, and the
    two-stage statuses are consistent."""
    cases = audit48 if FULL else audit48[::6]
    n_checked = 0
    for q in cases:
        fs, forc, trL = reference_trajectory(q, 64, N_T, solver="lu")
        spec = spectral_data(q, 64)
        z0 = trL.Z[:, 0]
        for r in (8, 16):
            Zc = _pod_candidate(fs, trL.Z, r)
            err = max(mass_norm(fs, Zc[:, k] - trL.Z[:, k]) for k in range(N_T + 1))
            cX = certify(fs, forc, Zc, z0, 1e-3, mode="banded", spec=spec)
            cS = certify(fs, forc, Zc, z0, 1e-3, mode="exact", spec=spec)
            assert cS.etaS_upper <= cX.etaX * (1 + 1e-10)
            assert cX.etaX <= np.sqrt(spec.Theta) * cS.etaS_upper * (1 + 1e-10)
            assert cX.bound >= err * (1 - 1e-10) and cS.bound >= err * (1 - 1e-10)
            assert cS.bracket_rel_width < 1e-15
            c2 = certify(fs, forc, Zc, z0, 1e-3, mode="two-stage", spec=spec)
            assert c2.status in ("direct_accept", "reject_cheap", "refined_accept", "refined_reject")
            if c2.status == "direct_accept":
                assert cX.bound <= 1e-3
            if c2.status == "reject_cheap":
                assert cS.bound > 1e-3
            n_checked += 1
    assert n_checked >= 4


def test_two_stage_eps_sweep_consistency():
    q = Query(0.3, 2.0, 7.0, 1.3, 0.02, T=0.7)
    fs, forc, trL = reference_trajectory(q, 64, N_T, solver="lu")
    spec = spectral_data(q, 64)
    Zc = _pod_candidate(fs, trL.Z, 12)
    err = max(mass_norm(fs, Zc[:, k] - trL.Z[:, k]) for k in range(N_T + 1))
    cS = certify(fs, forc, Zc, trL.Z[:, 0], 1.0, mode="exact", spec=spec)
    for eps in cS.bound * np.array([0.3, 0.9, 1.0001, 1.5, 3.0]):
        c = certify(fs, forc, Zc, trL.Z[:, 0], eps, mode="two-stage", spec=spec)
        assert c.accepted == (cS.bound <= eps) or (c.accepted and cS.bound <= eps)
        if c.accepted:
            assert err <= eps
