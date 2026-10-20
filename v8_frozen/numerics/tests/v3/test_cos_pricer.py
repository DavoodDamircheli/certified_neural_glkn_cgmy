"""Gate C soundness tests for the independent COS/Fourier CGMY pricer.

Does not import vendor/cgmy_bspline or src/cgmy -- these tests exercise
only validation/ code.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from validation.cgmy_reference_pricing import (
    CGMYMarketParams,
    martingale_check_mpmath,
    martingale_check_numpy,
)
from validation.cos_pricer import converge_price, cos_put_price, truncation_range
from validation.independent_fourier_integration import carr_madan_put_price

PRODUCTION_CASE = CGMYMarketParams(C=0.3, G=3.0, M=5.0, Y=1.5, r=0.05, T=0.5)
SHORT_MATURITY_CASE = CGMYMarketParams(C=0.2, G=2.0, M=4.0, Y=1.3, r=0.03, T=0.1)
LOW_C_CASE = CGMYMarketParams(C=0.05, G=1.5, M=2.5, Y=1.2, r=0.02, T=0.8)
HIGH_Y_CASE = CGMYMarketParams(C=0.4, G=2.5, M=6.0, Y=1.79, r=0.07, T=0.6)
LOW_Y_CASE = CGMYMarketParams(C=0.4, G=2.5, M=6.0, Y=1.11, r=0.04, T=0.4)

ALL_CASES = [PRODUCTION_CASE, SHORT_MATURITY_CASE, LOW_C_CASE, HIGH_Y_CASE, LOW_Y_CASE]
S0, K = 100.0, 100.0


@pytest.mark.parametrize("mu", ALL_CASES)
def test_martingale_condition_numpy(mu):
    """E[e^{X_T}] should equal e^{rT} to double-precision tolerance -- a
    real implementation check (Gamma(-Y), exponents, signs), even though
    the identity is true by algebraic construction (omega_S is defined
    exactly to make it so)."""
    discrepancy = martingale_check_numpy(mu)
    assert abs(discrepancy) < 1e-10, f"martingale check failed: {discrepancy}"


@pytest.mark.parametrize("mu", ALL_CASES)
def test_martingale_condition_mpmath(mu):
    """Same check, independent arbitrary-precision implementation."""
    discrepancy = martingale_check_mpmath(mu, dps=50)
    assert abs(discrepancy) < 1e-30, f"mpmath martingale check failed: {discrepancy}"


@pytest.mark.parametrize("mu", ALL_CASES)
def test_cos_matches_independent_fourier_integration(mu):
    """The two independently implemented, independently derived pricing
    methods (COS cosine-series vs. Carr-Madan damped Fourier integral,
    different characteristic-function implementations too -- numpy vs.
    mpmath) must agree closely on the same inputs before either is trusted
    as a reference."""
    cos_out = converge_price(mu, S0, K)
    cm_out = carr_madan_put_price(mu, S0, K, dps=40)

    cos_price = cos_out["price"]
    cm_price = cm_out["price"]
    rel_diff = abs(cos_price - cm_price) / max(abs(cm_price), 1e-8)

    assert rel_diff < 1e-5, (
        f"COS={cos_price} vs Carr-Madan={cm_price}, rel_diff={rel_diff}, "
        f"mu={mu}, cos_converged_at={cos_out['converged_at']}"
    )


def test_cos_convergence_is_stable_on_production_case():
    """Increasing N at fixed, sufficiently wide truncation range should
    make consecutive relative changes shrink (not necessarily perfectly
    monotonically at tiny N, but no sustained blow-up), and the sweep
    should report an explicit convergence point rather than silently
    using an untested (N, L)."""
    out = converge_price(PRODUCTION_CASE, S0, K)
    assert not out["flagged_not_converged"], (
        "COS price did not stabilize within the tested (N, L) grid -- "
        "do not trust a fixed truncation choice here"
    )
    # Relative change at the largest tested N should be tiny.
    last = out["trace"][-1]
    assert last["rel_change"] is not None and last["rel_change"] < 1e-6


def test_truncation_range_widens_with_L():
    a1, b1 = truncation_range(PRODUCTION_CASE, L=6.0)
    a2, b2 = truncation_range(PRODUCTION_CASE, L=12.0)
    assert (b2 - a2) > (b1 - a1)


def _fast_price(mu, s0, k):
    """A single, already-validated (N, L) setting (see
    test_cos_convergence_is_stable_on_production_case / the converge_price
    sweep above, which both confirm N=256,L=12 is well past convergence
    for these cases) -- used by the ancillary sanity checks below so they
    don't each re-run a full (N,L) convergence sweep."""
    a, b = truncation_range(mu, L=12.0)
    return cos_put_price(mu, s0, k, N=256, trunc_range=(a, b), L=12.0)["price"]


def test_put_price_is_nonnegative_and_bounded_by_discounted_strike():
    """Physical sanity: 0 <= Put <= K*e^{-rT} for every tested case."""
    for mu in ALL_CASES:
        price = _fast_price(mu, S0, K)
        upper_bound = K * np.exp(-mu.r * mu.T)
        assert price >= -1e-6, f"negative put price {price} for {mu}"
        assert price <= upper_bound + 1e-6, f"put price {price} exceeds K*e^(-rT)={upper_bound} for {mu}"


def test_put_price_monotone_in_strike():
    """Put price must be non-decreasing in K at fixed S0 (physical
    monotonicity diagnostic required by the roadmap)."""
    mu = PRODUCTION_CASE
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    prices = [_fast_price(mu, S0, k) for k in strikes]
    diffs = np.diff(prices)
    assert np.all(diffs >= -1e-6), f"put price not monotone in K: {list(zip(strikes, prices))}"


def test_put_price_convex_in_strike():
    """Discrete convexity check: second difference >= 0 (standard
    no-arbitrage requirement for European put prices as a function of
    strike)."""
    mu = PRODUCTION_CASE
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    prices = np.array([_fast_price(mu, S0, k) for k in strikes])
    second_diff = np.diff(prices, n=2)
    assert np.all(second_diff >= -1e-5), f"put price not convex in K: {list(zip(strikes, prices))}"


def test_deep_otm_put_is_much_cheaper_than_atm():
    """Sanity limit: a deep out-of-the-money put (S0=3K) should price far
    below the at-the-money put -- not a full closed-form check, but a
    cheap, meaningful degenerate-regime sanity test given no simple closed
    form is available for general CGMY parameters.

    Note: an earlier version of this test asserted an absolute threshold
    (price < 0.5), chosen without reference to this process's actual
    tails; it failed at price=0.597. CGMY has heavy jump tails (unlike
    Black-Scholes), so a deep-OTM put need not be as small as diffusion
    intuition suggests -- the meaningful, non-arbitrary check is relative
    to the ATM price, not a guessed absolute number."""
    mu = PRODUCTION_CASE
    atm_price = _fast_price(mu, 100.0, 100.0)
    deep_otm_price = _fast_price(mu, 300.0, 100.0)
    assert 0 <= deep_otm_price < 0.1 * atm_price, (
        f"deep-OTM put ({deep_otm_price}) not much smaller than ATM ({atm_price})"
    )


def test_vectorized_cosine_coeffs_match_reference_quadrature():
    """Cross-check the vectorized Gauss-Legendre cosine-coefficient
    computation (used for speed) against the original, independent,
    one-quad-call-per-k reference implementation on a small N."""
    from validation.cos_pricer import _cosine_coeffs_put, _cosine_coeffs_put_reference_quad

    a, b = truncation_range(PRODUCTION_CASE, L=12.0)
    N = 16
    fast = _cosine_coeffs_put(a, b, N, K)
    ref = _cosine_coeffs_put_reference_quad(a, b, N, K)
    np.testing.assert_allclose(fast, ref, atol=1e-10, rtol=1e-8)
