"""Task 3 (working prompt): lambda_min_numerical vs. alpha_certified must be
split, never conflated, and alpha_certified must actually be a PROVEN lower
bound on lambda_min(S_h(mu)) -- not merely close to it.

certification.constants.weyl_residual_enclosure implements this via Weyl's
inequality applied to the off-diagonal residual of S in its own computed
eigenbasis (see that function's docstring and notes/03_coercivity.md for
the full derivation and empirical validation). These tests check the
underlying theorem on SYNTHETIC matrices with exactly known eigenvalues
first (so the check does not depend on trusting the same eigh call that
produced the bound), then check the strategy interface and its wiring
through compute_stability_constants on a real CGMY operator.
"""

from __future__ import annotations

import numpy as np
import pytest

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, Mesh
from certification.constants import compute_stability_constants, weyl_residual_enclosure

MU = CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.4)


def _random_orthogonal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q


def test_weyl_bound_is_tight_and_valid_on_a_synthetic_matrix_with_known_eigenvalues():
    """S = Q diag(d) Q^T has EXACTLY known eigenvalues d (by construction,
    independent of eigh) -- weyl_residual_enclosure, fed eigh's own
    recomputation of this same S, must be <= min(d) and very close to it
    for well-separated eigenvalues."""
    n = 40
    d = np.sort(np.concatenate([[0.05], np.random.default_rng(0).uniform(0.2, 5.0, n - 1)]))
    Q = _random_orthogonal(n, seed=1)
    S = Q @ np.diag(d) @ Q.T
    S = 0.5 * (S + S.T)  # symmetrize away any roundoff asymmetry

    eigs, V = np.linalg.eigh(S)
    alpha = weyl_residual_enclosure(S, eigs, V)

    assert alpha <= d[0] + 1e-12, f"bound {alpha} exceeds the TRUE lambda_min {d[0]}"
    assert alpha == pytest.approx(d[0], abs=1e-9), f"bound {alpha} not tight against true lambda_min {d[0]}"


def test_weyl_bound_stays_valid_with_a_deliberately_perturbed_eigenbasis():
    """The theorem (Weyl's inequality) holds for ANY orthogonal V and ANY
    diagonal reference, not just eigh's own converged output -- feed it a
    deliberately perturbed (still orthogonal) V and confirm the bound is
    still <= the TRUE lambda_min (just looser), not merely close to
    whatever eigh happened to return. This is a stronger test than the
    "trivial by construction" property alone."""
    n = 30
    d_true = np.sort(np.random.default_rng(2).uniform(0.5, 4.0, n))
    Q = _random_orthogonal(n, seed=3)
    S = Q @ np.diag(d_true) @ Q.T
    S = 0.5 * (S + S.T)

    # Perturb Q's columns and re-orthogonalize -- v_i^T S v_i is no longer
    # exactly lambda_i, so E_off is no longer tiny.
    rng = np.random.default_rng(4)
    Q_perturbed, _ = np.linalg.qr(Q + 0.05 * rng.standard_normal((n, n)))
    mu_perturbed = np.array([Q_perturbed[:, i] @ S @ Q_perturbed[:, i] for i in range(n)])
    order = np.argsort(mu_perturbed)
    mu_sorted = mu_perturbed[order]
    V_sorted = Q_perturbed[:, order]

    alpha = weyl_residual_enclosure(S, mu_sorted, V_sorted)
    assert alpha <= d_true.min() + 1e-9, (
        f"bound {alpha} exceeds the TRUE lambda_min {d_true.min()} even with a perturbed eigenbasis"
    )


def test_weyl_bound_correctly_flags_uncertified_when_fed_a_bad_eigenbasis():
    """A case with a small, genuinely positive true lambda_min, but fed a
    deliberately BAD (large-residual) trial eigenbasis instead of eigh's
    own converged one -- alpha_certified must go negative (uncertified),
    not silently report a positive bound it cannot actually prove. This is
    the intended failure mode: weyl_residual_enclosure trusts its inputs'
    quality, and a caller that feeds it a poorly-converged eigenbasis
    correctly gets "uncertified" back, not a false positive."""
    n = 20
    d_true = np.sort(np.concatenate([[0.02], np.random.default_rng(5).uniform(1.0, 5.0, n - 1)]))
    Q = _random_orthogonal(n, seed=6)
    S = Q @ np.diag(d_true) @ Q.T
    S = 0.5 * (S + S.T)

    # A LARGE perturbation to the eigenbasis (not a converged eigensolver's
    # output) -- big enough that the off-diagonal residual swamps the tiny
    # true lambda_min.
    rng = np.random.default_rng(7)
    Q_bad, _ = np.linalg.qr(Q + 1.0 * rng.standard_normal((n, n)))
    mu_bad = np.array([Q_bad[:, i] @ S @ Q_bad[:, i] for i in range(n)])
    order = np.argsort(mu_bad)
    alpha = weyl_residual_enclosure(S, mu_bad[order], Q_bad[:, order])

    assert alpha <= d_true.min() + 1e-9, "the bound must still be a valid (if useless) lower bound"
    assert alpha < 0, (
        f"expected the bad eigenbasis to defeat certification (alpha={alpha}); "
        f"if this now passes, the perturbation is no longer large enough to "
        f"exercise the uncertified path"
    )


@pytest.fixture(scope="module")
def ops():
    N, p = 32, 3
    x_min, x_max = -3.0, 3.0
    return OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)


def test_alpha_certified_never_exceeds_lambda_min_numerical(ops):
    const = compute_stability_constants(MU, ops)
    assert const.is_spd
    assert const.alpha_certified <= const.lambda_min_numerical
    assert 0.0 < const.safety_ratio <= 1.0


def test_default_strategy_is_essentially_lossless_on_a_real_operator(ops):
    """Empirical validation (see notes/03_coercivity.md): on this project's
    actual S_h(mu), the Weyl bound should match the raw numerical estimate
    to many significant digits, not just "be valid"."""
    const = compute_stability_constants(MU, ops)
    assert const.coercivity_strategy == "B_weyl_residual_enclosure"
    assert const.safety_ratio == pytest.approx(1.0, abs=1e-8)


def test_C_safety_margin_strategy_matches_its_formula(ops):
    const = compute_stability_constants(MU, ops, coercivity_strategy="C_safety_margin", coercivity_safety_delta=0.25)
    assert const.coercivity_strategy == "C_safety_margin"
    assert const.alpha_certified == pytest.approx(0.75 * const.lambda_min_numerical)
    assert const.safety_ratio == pytest.approx(0.75)


def test_A_analytical_strategy_is_not_implemented(ops):
    with pytest.raises(NotImplementedError):
        compute_stability_constants(MU, ops, coercivity_strategy="A_analytical")


def test_unknown_strategy_raises(ops):
    with pytest.raises(ValueError):
        compute_stability_constants(MU, ops, coercivity_strategy="bogus")


def test_is_spd_gates_on_alpha_certified_not_raw_estimate():
    """A mu/mesh combination that loses SPD (production mesh, a known
    SPD-losing corner per Stage D -- see configs/base.yaml): both
    lambda_min_numerical (reported, never NaN-suppressed, Task 3 working
    prompt) and alpha_certified/is_spd must reflect non-coercivity."""
    N, p = 64, 3
    x_min, x_max = float(np.log(100) - 8), float(np.log(100) + 8)
    ops_local = OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)
    bad_mu = CGMYParameters(C=1.0, G=1.0, M=9.0, Y=1.8, r=0.1, T=1.0)
    const = compute_stability_constants(bad_mu, ops_local)
    assert not const.is_spd
    assert const.coercivity_strategy == "uncertified"
    assert const.lambda_min_numerical == const.lambda_min_numerical  # not NaN
    assert const.lambda_min_numerical < 0
    assert const.alpha_certified != const.alpha_certified  # NaN
    assert const.safety_ratio != const.safety_ratio  # NaN
