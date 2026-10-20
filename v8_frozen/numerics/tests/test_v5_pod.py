"""v5 M3: mass-weighted free POD, basis artifact/loader, reduced CN, and its
certification through the production free workflow (roadmap §3).

Rounding allowances: a backward-stable triangular solve / Cholesky in
dimension n perturbs V_r^T M_I V_r - I by O(n*eps*cond(M_I)) (Higham Thm 8.5
and 10.3); we use 8*n*eps*cond(M_I). Trajectory comparisons use the
residual-derived allowance of certification.free_problem.residual_rounding_eta
plus the CN energy estimate, as in tests/test_v5_free_path.py.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.constants import dense_from_apply
from certification.constrained import march_free_cn_direct
from certification.free_problem import (
    build_free_problem,
    free_eta_sharp,
    free_residuals,
    free_stability_constants,
    residual_rounding_eta,
)
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1
from learning import pod_free as pf
from pipeline import free_workflow as fw
from pipeline.predict_certify_correct import build_mesh_info

EPS = np.finfo(float).eps
X_MIN, X_MAX = -3.394829814011908, 12.605170185988092
PAYOFF = EuropeanPutPayoff(strike=100.0)
N_STEPS = 20
TRAIN_MUS = [
    CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.5),
    CGMYParameters(C=0.7, G=1.5, M=8.0, Y=1.7, r=0.02, T=0.9),
    CGMYParameters(C=0.1, G=4.5, M=3.0, Y=1.2, r=0.09, T=0.2),
    CGMYParameters(C=0.5, G=2.0, M=6.0, Y=1.5, r=0.06, T=0.6),
]
MU_TEST = CGMYParameters(C=0.4, G=2.5, M=5.5, Y=1.45, r=0.04, T=0.7)


@pytest.fixture(scope="module")
def mesh():
    return Mesh(N=64, p=3, x_min=X_MIN, x_max=X_MAX)


@pytest.fixture(scope="module")
def mesh_info(mesh):
    return build_mesh_info(mesh)


@pytest.fixture(scope="module")
def ops(mesh):
    return OperatorCache(mesh, X_MIN, X_MAX)


@pytest.fixture(scope="module")
def prob(mesh_info, ops):
    return build_free_problem(MU_TEST, PAYOFF, mesh_info, ops, np.linspace(0, MU_TEST.T, N_STEPS + 1),
                              method_id=METHOD_CONSTRAINED_FREE_CN_V1)


@pytest.fixture(scope="module")
def snapshots(mesh, prob):
    rows = []
    for mu in TRAIN_MUS:
        t = solve_cgmy(mu, PAYOFF, mesh, TimeGrid(n_steps=N_STEPS), mode="toeplitz")
        rows.append(t.coefficients[:, prob.fs.free_idx])
    return np.concatenate(rows)  # (84, 62)


@pytest.fixture(scope="module")
def M_I(prob):
    return dense_from_apply(prob.free_ops.apply_M, prob.fs.n_free)


def test_mass_orthonormal_and_triangular_construction(prob, snapshots, M_I):
    n = prob.fs.n_free
    for r in (8, 32, n):
        b = pf.compute_mass_weighted_pod_basis(snapshots, prob.free_ops, r)
        assert b.basis_type == pf.BASIS_MASS_WEIGHTED and b.modes.shape == (n, r)
        G = b.modes.T @ M_I @ b.modes
        assert np.max(np.abs(G - np.eye(r))) <= 8 * n * EPS * np.linalg.cond(M_I)
    L = pf.mass_cholesky_factor(prob.free_ops)
    assert np.max(np.abs(L @ L.T - M_I)) <= 4 * n * EPS * np.max(np.abs(M_I))
    assert np.allclose(L, np.tril(L), rtol=0, atol=0)


def test_eckart_young_in_mass_norm_and_beats_euclidean_diagnostic(prob, snapshots, M_I):
    """sum_j ||z_j - P_r z_j||_{M_I}^2 = sum_{k>r} sigma_k^2 for the mass-weighted
    basis (identity, up to rounding of the sums), and it is <= the same
    quantity for the Euclidean-SVD diagnostic subspace (optimality)."""
    full = pf.compute_mass_weighted_pod_basis(snapshots, prob.free_ops, prob.fs.n_free)
    tail_all = full.singular_values**2
    total = float(np.sum(tail_all))
    for r in (4, 8, 16):
        mw = pf.compute_mass_weighted_pod_basis(snapshots, prob.free_ops, r)
        eu = pf.compute_free_pod_basis(snapshots, prob.free_ops, r)
        assert eu.basis_type == pf.BASIS_EUCLIDEAN_DIAGNOSTIC

        def proj_err(b):
            Z = snapshots.T
            R = Z - b.modes @ (b.modes.T @ (M_I @ Z))
            return float(np.sum(R * (M_I @ R)))

        e_mw, e_eu = proj_err(mw), proj_err(eu)
        tail = float(np.sum(tail_all[r:]))
        assert abs(e_mw - tail) <= 8 * prob.fs.n_free * EPS * np.linalg.cond(M_I) * total
        assert e_mw <= e_eu * (1 + 8 * prob.fs.n_free * EPS * np.linalg.cond(M_I)) + 8 * prob.fs.n_free * EPS * total


def test_truncation_is_nested(prob, snapshots):
    b = pf.compute_mass_weighted_pod_basis(snapshots, prob.free_ops, 40)
    t = pf.truncate_basis(b, 12)
    assert np.array_equal(t.modes, b.modes[:, :12]) and t.basis_type == b.basis_type
    with pytest.raises(ValueError):
        pf.truncate_basis(b, 41)


def test_full_rank_reduced_cn_recovers_direct_solution(prob, snapshots):
    """Full rank: V square, reduced CN is a similarity transform of the free CN
    system. Allowance from residuals: ||z_pod - z_dir||_{M_I} <= eta_aug(pod) +
    eta_aug(dir), each eta_aug including beta_0 and the residual-rounding term."""
    b = pf.compute_mass_weighted_pod_basis(snapshots, prob.free_ops, prob.fs.n_free)
    alphas, z_pod = pf.reduced_cn_trajectory(b, prob)
    np.testing.assert_array_equal(alphas[0], b.modes.T @ prob.free_ops.apply_M(prob.z0_target))
    z_dir = march_free_cn_direct(prob.free_ops, prob.mu, prob.dt, prob.z0_target, prob.forcing.f_I).z
    lam = free_stability_constants(prob, coercivity_strategy="B_weyl_residual_enclosure").lambda_min_numerical

    def eta_aug(z):
        b0 = prob.mass_norm(z[0] - prob.z0_target)
        e = free_eta_sharp(prob, free_residuals(prob, z)).eta + residual_rounding_eta(prob, z, lam)
        return float(np.hypot(b0, e))

    err = max(prob.mass_norm(z_pod[n] - z_dir[n]) for n in range(N_STEPS + 1))
    assert err <= eta_aug(z_pod) + eta_aug(z_dir)


def test_finite_rank_through_free_workflow_includes_initial_error(prob, snapshots, mesh_info, ops):
    b = pf.compute_mass_weighted_pod_basis(snapshots, prob.free_ops, 8)
    pol = fw.FreeWorkflowPolicy(tolerance=1e-3, quick_pcg_iterations=10, max_pcg_iterations=200).validate()
    tr = fw.run_free_workflow(MU_TEST, PAYOFF, mesh_info, ops, np.linspace(0, MU_TEST.T, N_STEPS + 1),
                              pf.free_pod_predictor(b), pol)
    cc = tr.candidate_certificate
    assert tr.method_id == METHOD_CONSTRAINED_FREE_CN_V1
    assert cc.initial_error > 0.0  # finite-rank initial projection error is charged
    assert cc.augmented_certificate == pytest.approx(np.hypot(cc.initial_error, cc.residual_certificate), rel=4 * EPS)
    assert tr.final_status in fw.ACCEPTED_STATUSES


# ---------------------------------------------------------------- artifact / loader smoke tests

def _meta(prob):
    return {"method_id": METHOD_CONSTRAINED_FREE_CN_V1, "mass_matrix_identity": pf.mass_matrix_identity(prob.free_ops),
            "training": {"note": "unit-test snapshots"}, "discretization": {"N": 64}}


def test_artifact_roundtrip_and_strict_loader(prob, snapshots, tmp_path):
    b = pf.compute_mass_weighted_pod_basis(snapshots, prob.free_ops, prob.fs.n_free)
    path = tmp_path / "mw.npz"
    pf.save_pod_artifact(path, b, _meta(prob))
    with pytest.raises(FileExistsError):
        pf.save_pod_artifact(path, b, _meta(prob))
    got, meta = pf.load_pod_artifact(path, prob.free_ops, rank=16)
    assert meta["basis_type"] == pf.BASIS_MASS_WEIGHTED and meta["rank"] == prob.fs.n_free
    assert np.array_equal(got.modes, b.modes[:, :16])
    assert len(meta["singular_values"]) == prob.fs.n_free
    eu = pf.compute_free_pod_basis(snapshots, prob.free_ops, 16)
    pf.save_pod_artifact(tmp_path / "eu.npz", eu, _meta(prob))
    with pytest.raises(pf.LegacyPODCheckpointError):
        pf.load_pod_artifact(tmp_path / "eu.npz", prob.free_ops, rank=16)
    ops32 = OperatorCache(Mesh(N=32, p=3, x_min=X_MIN, x_max=X_MAX), X_MIN, X_MAX)
    from certification.constrained import FreeOperatorCache, free_space_for_mesh
    fo32 = FreeOperatorCache(ops32, free_space_for_mesh(build_mesh_info(Mesh(N=32, p=3, x_min=X_MIN, x_max=X_MAX))))
    with pytest.raises(pf.LegacyPODCheckpointError):
        pf.load_pod_artifact(path, fo32, rank=16)


def test_legacy_pod_checkpoint_cannot_be_loaded_by_v5_loader(prob):
    from common.run_manifest import REPO_ROOT

    with pytest.raises(pf.LegacyPODCheckpointError):
        pf.load_pod_artifact(REPO_ROOT / "checkpoints" / "pod_final.npz", prob.free_ops, rank=8)


def test_v4_harness_no_longer_constructs_legacy_pod():
    import sys

    from common.run_manifest import REPO_ROOT

    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    import run_gate_d_benchmark_v4 as h4
    import exp_common

    assert "certified_pod" not in h4.METHODS
    with pytest.raises(h4.LegacyPODRetiredError):
        h4.run_method("certified_pod", {"mu": None, "payoff": None}, None, None, None, {}, None, None)
    _, stats = exp_common.get_box_and_stats()
    models = exp_common.load_models(exp_common.get_ops(), stats, include_legacy_pod=False)
    assert "POD" not in models
    assert not any(type(m).__name__ == "PODGalerkinROM" for m in models.values())


def test_production_basis_artifact_metadata_if_built():
    """When the M3 run has produced the production artifact, its metadata must
    identify the mass-weighted basis, its training population and M_I."""
    from common.run_manifest import REPO_ROOT

    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "v5_free_workflow.yaml").read_text())
    path = cfg.get("pod", {}).get("artifact")
    if not path or not (REPO_ROOT / path).exists():
        pytest.skip("production POD artifact not present in this checkout (gitignored .npz)")
    data = np.load(REPO_ROOT / path)
    meta = json.loads(str(data["metadata_json"]))
    assert meta["basis_type"] == pf.BASIS_MASS_WEIGHTED and meta["method_id"] == METHOD_CONSTRAINED_FREE_CN_V1
    assert meta["training"]["n_trajectories"] == 800 and meta["training"]["centering"] == "uncentred"
    from common.run_manifest import sha256_file

    assert sha256_file(REPO_ROOT / path) == cfg["pod"]["artifact_sha256"]
