"""v5 M1: one free operator / one partition / one forcing for every stage
(roadmap §2.1, §2.5 and the "Required small tests" table of §2).

Rounding model used for tolerances (stated once, referenced below). For a
matrix-vector product computed in double precision, |fl(Ax) - Ax| <=
gamma_n |A| |x| componentwise with gamma_n = n*eps/(1-n*eps) (Higham, Thm 3.5);
the Toeplitz blocks of A_h are applied by FFT circulant embedding, whose error
is normwise, ||fl(Kx) - Kx||_inf <= c*eps*log2(2n)*||K||_inf*||x||_inf. Both are
dominated by GAMMA * ||A||_inf * ||x||_inf with GAMMA = 4*n_dof*eps (factor 4
covers the two FFTs, the pointwise product and the final sum). Every
tolerance below is GAMMA times the magnitudes actually involved, or a solver
residual turned into an error bound by the CN energy estimate -- never a bare
absolute number.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.linalg as sla

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid, solve_cgmy
from certification.constants import compute_stability_constants, dense_from_apply, kappa_operator_norm
from certification.constrained import inject, march_free_cn, march_free_cn_direct, restrict
from certification.free_problem import (
    FORCING_SOURCE_STORED,
    ForcingError,
    IncompatibleMetadataError,
    beta_I,
    build_free_problem,
    free_eta_res_upper,
    free_eta_sharp,
    free_kappa,
    free_point_functional,
    free_residuals,
    free_stability_constants,
    raw_point_functional,
)
from certification.residuals import compute_all_residuals
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1, METHOD_LEGACY_FULL
from pipeline.predict_certify_correct import build_mesh_info, build_skeleton_trajectory

EPS = np.finfo(float).eps
X_MIN, X_MAX = -3.394829814011908, 12.605170185988092
PAYOFF = EuropeanPutPayoff(strike=100.0)
N_STEPS = 20
MU_SPD = CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.5)
# Q_audit_0000: legacy full S_h indefinite, free S_I SPD (v4 Check 5).
MU_FULL_INDEFINITE = CGMYParameters(
    C=0.4874301809351891, G=1.4997461959719658, M=8.837858855724335,
    Y=1.4543537443503738, r=0.06294871459715069, T=0.10040483502671123,
)


@pytest.fixture(scope="module")
def mesh():
    return Mesh(N=64, p=3, x_min=X_MIN, x_max=X_MAX)


@pytest.fixture(scope="module")
def mesh_info(mesh):
    return build_mesh_info(mesh)


@pytest.fixture(scope="module")
def ops(mesh):
    return OperatorCache(mesh, X_MIN, X_MAX)


def _tg(mu):
    return np.linspace(0.0, mu.T, N_STEPS + 1)


@pytest.fixture(scope="module")
def traj(mesh):
    return solve_cgmy(MU_SPD, PAYOFF, mesh, TimeGrid(n_steps=N_STEPS), mode="toeplitz", store_forcing=True)


@pytest.fixture(scope="module")
def prob(mesh_info, ops):
    return build_free_problem(MU_SPD, PAYOFF, mesh_info, ops, _tg(MU_SPD), method_id=METHOD_CONSTRAINED_FREE_CN_V1)


@pytest.fixture(scope="module")
def dense(prob):
    n = prob.fs.n_free
    M = dense_from_apply(prob.free_ops.apply_M, n)
    A = dense_from_apply(lambda v: prob.free_ops.apply_A(prob.mu, v), n)
    S = 0.5 * (A + A.T)
    return M, A, S


def _gamma(n: int) -> float:
    return 4 * n * EPS


def _residual_rounding_eta(prob, z, M_abs_inf, A_abs_inf, lam_min_S):
    """Upper estimate of the contribution of rounding in the computed free
    residual to eta: sqrt(dt * sum_n ||delta r_n||_2^2 / lambda_min(S_I)), with
    ||delta r_n||_inf <= GAMMA*(||M||_inf ||dz||_inf/dt + ||A||_inf ||zbar||_inf
    + ||f_I||_inf) and ||.||_2 <= sqrt(n)||.||_inf."""
    n = prob.fs.n_free
    g = _gamma(prob.meta.n_dof)
    tot = 0.0
    for k in range(prob.n_steps):
        dz = np.max(np.abs(z[k + 1] - z[k])) / prob.dt
        zb = np.max(np.abs(0.5 * (z[k + 1] + z[k])))
        fk = np.max(np.abs(prob.forcing.f_I[k]))
        d_inf = g * (M_abs_inf * dz + A_abs_inf * zb + fk)
        tot += n * d_inf**2
    return float(np.sqrt(prob.dt * tot / lam_min_S))


# ---------------------------------------------------------------- forcing & target identity

def test_trusted_forcing_is_byte_identical_to_stored_solver_forcing(prob, traj, mesh_info, ops):
    assert np.array_equal(prob.forcing.f_bar_raw, traj.forcing_vectors)
    stored = build_free_problem(
        MU_SPD, PAYOFF, mesh_info, ops, traj.time_grid, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
        f_bar_raw=traj.forcing_vectors, forcing_source=FORCING_SOURCE_STORED,
        schemes=traj.solver_metadata.scheme_per_step, n_steps=traj.solver_metadata.n_steps,
    )
    assert stored.forcing.f_I_sha256 == prob.forcing.f_I_sha256
    assert stored.dt == traj.solver_metadata.dt  # trusted dt = T/n_steps, bit-for-bit


def test_forcing_arrays_are_frozen(prob):
    with pytest.raises(ValueError):
        prob.forcing.f_I[0, 0] = 1.0


def test_initial_target_agrees_with_trusted_c0(prob, traj, dense):
    """compute_initial_coefficients (splu) vs solve_cn's c0 (spsolve): two
    backward-stable solves of M_h c = b0, so ||dc|| / ||c|| <= 2*cond(M_h)*gamma_n."""
    M_raw = prob.ops.M_h.toarray()
    cond_M = np.linalg.cond(M_raw)
    rel = np.linalg.norm(prob.c0_target - traj.coefficients[0]) / np.linalg.norm(traj.coefficients[0])
    assert rel <= 2 * cond_M * _gamma(prob.meta.n_dof)
    assert np.array_equal(prob.z0_target, restrict(prob.fs, prob.c0_target))


# ---------------------------------------------------------------- boundary checks / rejection

def test_absent_or_incompatible_forcing_rejected(mesh_info, ops, prob):
    tg = _tg(MU_SPD)
    with pytest.raises(ForcingError):
        build_free_problem(MU_SPD, PAYOFF, mesh_info, ops, tg, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                           forcing_source=FORCING_SOURCE_STORED, f_bar_raw=None)
    with pytest.raises(ForcingError):
        build_free_problem(MU_SPD, PAYOFF, mesh_info, ops, tg, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                           forcing_source=FORCING_SOURCE_STORED, f_bar_raw=np.zeros((N_STEPS - 1, mesh_info.n_dof)))
    bad = np.zeros((N_STEPS, mesh_info.n_dof))
    bad[3, 7] = np.nan
    with pytest.raises(ForcingError):
        build_free_problem(MU_SPD, PAYOFF, mesh_info, ops, tg, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                           forcing_source=FORCING_SOURCE_STORED, f_bar_raw=bad)
    other = build_free_problem(dataclasses.replace(MU_SPD, r=0.06), PAYOFF, mesh_info, ops, tg,
                               method_id=METHOD_CONSTRAINED_FREE_CN_V1)
    z = np.tile(prob.z0_target, (N_STEPS + 1, 1))
    with pytest.raises(IncompatibleMetadataError):
        free_residuals(prob, z, forcing=other.forcing)


def test_metadata_mismatches_rejected(mesh_info, ops, prob):
    tg = _tg(MU_SPD)
    ops32 = OperatorCache(Mesh(N=32, p=3, x_min=X_MIN, x_max=X_MAX), X_MIN, X_MAX)
    with pytest.raises(IncompatibleMetadataError):
        build_free_problem(MU_SPD, PAYOFF, mesh_info, ops32, tg, method_id=METHOD_CONSTRAINED_FREE_CN_V1)
    with pytest.raises(IncompatibleMetadataError):
        build_free_problem(MU_SPD, PAYOFF, mesh_info, ops, tg, method_id=METHOD_LEGACY_FULL)
    with pytest.raises(IncompatibleMetadataError):
        free_residuals(prob, np.zeros((N_STEPS + 1, prob.meta.n_dof)))  # raw-space array
    with pytest.raises(IncompatibleMetadataError):
        free_residuals(prob, np.zeros((N_STEPS, prob.fs.n_free)))


# ---------------------------------------------------------------- required small tests

def test_free_raw_constrained_trajectory_agreement(prob, traj, dense):
    """Trusted raw GMRES trajectory (restricted) vs free dense-LU and free GMRES
    marches of the same constrained problem from the same z^0.

    Each computed trajectory z_k has free residual r_k in the exact constrained
    problem, so by the CN energy estimate ||z_k^n - z*^n||_{M_I} <=
    eta_sharp(r_k) (+ the residual's own rounding, see module docstring); the
    triangle inequality gives the allowance for every pair. No absolute
    tolerance is used."""
    M, A, S = dense
    lam_min = float(np.linalg.eigvalsh(S)[0])
    Minf, Ainf = np.max(np.sum(np.abs(M), 1)), np.max(np.sum(np.abs(A), 1))
    z_raw = traj.coefficients[:, prob.fs.free_idx]
    z0 = z_raw[0]
    direct = march_free_cn_direct(prob.free_ops, prob.mu, prob.dt, z0, prob.forcing.f_I)
    gm = march_free_cn(prob.free_ops, prob.mu, prob.dt, z0, prob.forcing.f_I)
    assert direct.all_converged and gm.all_converged and not gm.aborted

    def allowance(z):
        r = free_residuals(prob, z)
        return free_eta_sharp(prob, r).eta + _residual_rounding_eta(prob, z, Minf, Ainf, lam_min)

    a_raw, a_dir, a_gm = allowance(z_raw), allowance(direct.z), allowance(gm.z)
    d_raw_dir = max(prob.mass_norm(z_raw[n] - direct.z[n]) for n in range(N_STEPS + 1))
    d_gm_dir = max(prob.mass_norm(gm.z[n] - direct.z[n]) for n in range(N_STEPS + 1))
    assert d_raw_dir <= a_raw + a_dir, (d_raw_dir, a_raw, a_dir)
    assert d_gm_dir <= a_gm + a_dir, (d_gm_dir, a_gm, a_dir)
    assert d_raw_dir > 0.0  # two genuinely different computations


def test_free_raw_residual_identity_after_boundary_projection(prob, traj, mesh_info):
    """Same lifting + same forcing => legacy raw residual (bc rows zeroed),
    restricted, equals the free residual up to matvec rounding. Before
    projection the legacy residual reads the corrupted boundary rows and the
    identity fails by far more than rounding, which is why projection comes first."""
    rng = np.random.default_rng(1)
    cand = traj.coefficients + 1e-3 * rng.standard_normal(traj.coefficients.shape)
    z = cand[:, prob.fs.free_idx]
    projected = prob.lift(z)
    r_free = free_residuals(prob, z)

    def raw_restricted(coeffs):
        sk = build_skeleton_trajectory(MU_SPD, PAYOFF, mesh_info, traj.time_grid, coeffs)
        sk = dataclasses.replace(sk, forcing_vectors=np.array(prob.forcing.f_bar_raw),
                                 solver_metadata=dataclasses.replace(sk.solver_metadata, dt=prob.dt))
        return compute_all_residuals(sk, MU_SPD, prob.ops)[:, prob.fs.free_idx]

    M_raw = prob.ops.M_h.toarray()
    A_raw = dense_from_apply(lambda v: prob.ops.apply_A(MU_SPD, v), prob.meta.n_dof)
    Minf, Ainf = np.max(np.sum(np.abs(M_raw), 1)), np.max(np.sum(np.abs(A_raw), 1))
    g = _gamma(prob.meta.n_dof)
    diff_after = raw_restricted(projected) - r_free
    for n in range(N_STEPS):
        mag = (Minf * np.max(np.abs(projected[n + 1] - projected[n])) / prob.dt
               + Ainf * np.max(np.abs(projected[n + 1] + projected[n])) / 2
               + np.max(np.abs(prob.forcing.f_bar_raw[n])))
        # free route evaluates the same terms split into z- and b-parts: 2x the magnitude
        assert np.max(np.abs(diff_after[n])) <= 2 * g * mag
    diff_before = raw_restricted(cand) - r_free
    assert np.max(np.abs(diff_before)) > 1e6 * g * Ainf * np.max(np.abs(cand))


def test_mass_norm_identity(prob, traj, dense):
    """||z||_{M_I} == ||P z||_{M_h}, and for two candidates with the same
    lifting the free error norm equals the raw error norm of the injected
    homogeneous error; difference <= gamma_n |z|^T |M| |z| (summation order only)."""
    M, _, _ = dense
    rng = np.random.default_rng(2)
    z1 = traj.coefficients[:, prob.fs.free_idx]
    z2 = z1 + 1e-2 * rng.standard_normal(z1.shape)
    c1, c2 = prob.lift(z1), prob.lift(z2)
    absM = np.abs(prob.ops.M_h.toarray())
    for n in (0, 7, N_STEPS):
        e_free = z2[n] - z1[n]
        e_raw = c2[n] - c1[n]
        assert np.array_equal(e_raw[prob.fs.bc_idx], np.zeros(len(prob.fs.bc_idx)))
        q_free = prob.mass_norm(e_free) ** 2
        q_raw = float(e_raw @ prob.ops.apply_M(e_raw))
        bound = 2 * _gamma(prob.meta.n_dof) * float(np.abs(e_raw) @ absM @ np.abs(e_raw))
        assert abs(q_free - q_raw) <= bound
        assert abs(q_free - float(e_free @ M @ e_free)) <= bound


def test_principal_mass_inverse(prob, dense):
    """apply_M_inv (sparse LU of the principal submatrix M_I) == dense solve of
    M_I to 2*cond(M_I)*gamma_n relative; the restricted raw inverse P^T M_h^{-1} P
    is a different operator (differs by many orders more)."""
    M, _, _ = dense
    cond = np.linalg.cond(M)
    rng = np.random.default_rng(3)
    v = rng.standard_normal(prob.fs.n_free)
    x_sparse = prob.free_ops.apply_M_inv(v)
    x_dense = np.linalg.solve(M, v)
    tol = 2 * cond * _gamma(prob.fs.n_free)
    assert np.linalg.norm(x_sparse - x_dense) <= tol * np.linalg.norm(x_dense)
    assert np.linalg.norm(M @ x_sparse - v) <= _gamma(prob.fs.n_free) * np.linalg.norm(M, 2) * np.linalg.norm(x_sparse) * 4
    x_restricted_raw = restrict(prob.fs, prob.ops.apply_M_inv(inject(prob.fs, v)))
    assert np.linalg.norm(x_restricted_raw - x_dense) > 1e6 * tol * np.linalg.norm(x_dense)


# ---------------------------------------------------------------- routing through the free operator

def test_free_gate_on_legacy_failure(mesh_info, ops):
    p = build_free_problem(MU_FULL_INDEFINITE, PAYOFF, mesh_info, ops, _tg(MU_FULL_INDEFINITE),
                           method_id=METHOD_CONSTRAINED_FREE_CN_V1)
    c_free = free_stability_constants(p, coercivity_strategy="B_weyl_residual_enclosure")
    c_full = compute_stability_constants(MU_FULL_INDEFINITE, ops)
    assert c_free.is_spd and c_free.coordinate_space == "constrained_free"
    assert not c_full.is_spd
    assert c_free.alpha_lower_estimate <= c_free.lambda_min_numerical


def test_upper_bracket_dominates_sharp_in_free_space(prob, traj):
    rng = np.random.default_rng(4)
    z = traj.coefficients[:, prob.fs.free_idx] + 1e-3 * rng.standard_normal((N_STEPS + 1, prob.fs.n_free))
    z[0] = prob.z0_target
    r = free_residuals(prob, z)
    const = free_stability_constants(prob, coercivity_strategy="B_weyl_residual_enclosure")
    up = free_eta_res_upper(prob, r, const.alpha_lower_estimate, max_pcg_iterations=5, preconditioner="none")
    sh = free_eta_sharp(prob, r)
    assert up.coordinate_space == "constrained_free"
    assert up.eta >= sh.eta * (1 - _gamma(prob.fs.n_free))


# ---------------------------------------------------------------- kappa and point evaluation

def _kappa_symmetric_sqrt(A, S):
    w, V = np.linalg.eigh(S)
    S_mhalf = (V / np.sqrt(w)) @ V.T
    return float(np.linalg.norm(S_mhalf @ A @ S_mhalf, 2))


@pytest.mark.parametrize("which", ["production_free", "random_nonsymmetric"])
def test_kappa_cholesky_orientation_matches_independent_dense(prob, dense, which):
    """kappa = ||L^{-1} A L^{-T}||_2 (production) vs ||S^{-1/2} A S^{-1/2}||_2 via
    the symmetric square root (independent) vs sqrt(lambda_max(A^T S^{-1} A, S))
    (generalized eigenproblem). Forward-error allowance for the transformed
    operator: 8*n*eps*cond(S). The transposed (wrong) orientation L^{-T} A L^{-1}
    must NOT agree, otherwise the test could not detect an orientation error."""
    if which == "production_free":
        _, A, S = dense
        kappa_prod = free_kappa(prob)
    else:
        rng = np.random.default_rng(5)
        n = 12
        G = rng.standard_normal((n, n))
        S = G @ G.T + n * np.eye(n)
        K = rng.standard_normal((n, n))
        A = S + (K - K.T)  # sym(A) = S
        S = 0.5 * (A + A.T)

        class _Dense:
            n_dof = n

            def apply_S(self, mu, v):
                return S @ v

            def apply_A(self, mu, v):
                return A @ v

        kappa_prod = kappa_operator_norm(None, _Dense())
    n = S.shape[0]
    tol = 8 * n * EPS * np.linalg.cond(S)
    k_sqrt = _kappa_symmetric_sqrt(A, S)
    k_gen = float(np.sqrt(sla.eigh(A.T @ np.linalg.solve(S, A), S, eigvals_only=True)[-1]))
    assert abs(kappa_prod - k_sqrt) <= tol * k_sqrt
    assert abs(kappa_prod - k_gen) <= tol * k_sqrt
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    k_wrong = float(np.linalg.norm(Linv.T @ A @ Linv, 2))
    assert abs(k_wrong - k_sqrt) > 1e3 * tol * k_sqrt
    assert kappa_prod >= 1.0 - tol  # Lemma kappa: kappa >= 1


def test_point_functional_reproduces_trusted_evaluation(prob, traj):
    """phi(x)^T c == eval_solution(c, x): same dof ordering, no hidden scaling."""
    from cgmy_bspline.projection import eval_solution

    gi = {"N": prob.mesh_info.N, "h": prob.mesh_info.h, "x_min": prob.mesh_info.x_min}
    c = traj.coefficients[-1]
    for x in (np.log(80.0), np.log(100.0), np.log(120.0)):
        v_trusted = float(eval_solution(c, gi, prob.mesh_info.p, np.array([x]))[0])
        v_phi = float(raw_point_functional(prob.mesh_info, x) @ c)
        assert abs(v_phi - v_trusted) <= _gamma(prob.mesh_info.p) * float(np.abs(raw_point_functional(prob.mesh_info, x)) @ np.abs(c))


def test_beta_I_matches_dense_and_bounds_pointwise_error(prob, dense):
    """beta_I via the M_I factorization == dense sqrt(phi_I^T M_I^{-1} phi_I);
    and for any homogeneous free error e, |phi_I^T e| <= beta_I ||e||_{M_I}
    (Cauchy-Schwarz in the M_I inner product), checked on random e."""
    M, _, _ = dense
    rng = np.random.default_rng(6)
    for x in (np.log(90.0), np.log(100.0), np.log(110.0)):
        ph = free_point_functional(prob, x)
        b_dense = float(np.sqrt(ph @ np.linalg.solve(M, ph)))
        b = beta_I(prob, x)
        assert abs(b - b_dense) <= 2 * np.linalg.cond(M) * _gamma(prob.fs.n_free) * b_dense
        for _ in range(20):
            e = rng.standard_normal(prob.fs.n_free)
            assert abs(ph @ e) <= b * prob.mass_norm(e) * (1 + _gamma(prob.fs.n_free))
