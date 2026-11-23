"""Payoff families, each returning a B-spline coefficient vector p such that
psi_h(x) = sum_j p_j phi_j(x) is the L2 projection of the payoff onto the
same discrete space (mesh) the reference solver uses. p is the neural
branch-network input representing the contract.

Every coefficient vector here is built by calling
``cgmy_bspline.projection.assemble_payoff_rhs`` (the payoff-generic RHS
assembler generalized in Stage A) followed by
``OperatorCache.apply_M_inv`` (the cached M_h factorization) -- no new
quadrature or projection numerics.

SCOPE LIMIT, same one from Stage A: ``cgmy.reference_solver.solve_cgmy``
only accepts ``EuropeanPutPayoff``, because the trusted solver's Dirichlet
boundary conditions are derived for the put's specific far-field
asymptotics (constant K*exp(-r*tau) deep ITM, 0 deep OTM). The other
payoffs below have DIFFERENT far-field behavior -- a European call grows
*linearly* in S deep ITM, not to a constant, so reusing the put's boundary
condition for it would silently mis-price it. A call spread/put
spread/butterfly happen to be bounded at both ends (the linear terms
cancel between legs), but the actual limiting constants differ from the
put's and are not derived/validated here either.

Consequence: every function in this module can build a valid p vector
today (this is exactly the L2-projection step, payoff-agnostic and safe).
Only the "put" family can currently be pushed through solve_cgmy to get a
full solved trajectory. ``data.dataset``/``scripts.generate_debug_dataset``
reflect this: they generate full trajectories for put contracts, and
p-vectors only (no trajectory) for the others, logged as such rather than
silently skipped or silently mis-solved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cgmy import _external  # noqa: F401  (must precede any cgmy_bspline import)

from cgmy_bspline.boundary import put_payoff
from cgmy_bspline.projection import assemble_payoff_rhs

from cgmy.operators import OperatorCache


def call_payoff(x: np.ndarray, strike: float) -> np.ndarray:
    """European call payoff: max(exp(x) - K, 0), x = log-price."""
    return np.maximum(np.exp(np.asarray(x, dtype=float)) - strike, 0.0)


def _project(payoff_fn: Callable[[np.ndarray], np.ndarray], kink_x: float | None, ops: OperatorCache) -> np.ndarray:
    grid_info = {"N": ops.N, "h": ops.h, "x_min": ops.x_min}
    b = assemble_payoff_rhs(payoff_fn, grid_info, ops.p, kink_x=kink_x)
    return ops.apply_M_inv(b)


def european_call(strike: float, ops: OperatorCache) -> np.ndarray:
    """p for psi(x) = max(exp(x) - K, 0). NOT solvable via solve_cgmy yet (see module docstring)."""
    return _project(lambda x: call_payoff(x, strike), np.log(strike), ops)


def european_put(strike: float, ops: OperatorCache) -> np.ndarray:
    """p for psi(x) = max(K - exp(x), 0). The only family solvable via solve_cgmy today."""
    return _project(lambda x: put_payoff(x, strike), np.log(strike), ops)


def call_spread(k_low: float, k_high: float, ops: OperatorCache) -> np.ndarray:
    """Bull call spread: long call(k_low), short call(k_high). Requires k_low < k_high.

    p is exact by linearity of the L2 projection: p = p_call(k_low) - p_call(k_high).
    NOT solvable via solve_cgmy yet.
    """
    if not k_low < k_high:
        raise ValueError(f"call_spread requires k_low < k_high, got {k_low}, {k_high}")
    return european_call(k_low, ops) - european_call(k_high, ops)


def put_spread(k_low: float, k_high: float, ops: OperatorCache) -> np.ndarray:
    """Bear put spread: long put(k_high), short put(k_low). Requires k_low < k_high.

    p = p_put(k_high) - p_put(k_low). NOT solvable via solve_cgmy yet.
    """
    if not k_low < k_high:
        raise ValueError(f"put_spread requires k_low < k_high, got {k_low}, {k_high}")
    return european_put(k_high, ops) - european_put(k_low, ops)


def butterfly(k_low: float, k_mid: float, k_high: float, ops: OperatorCache) -> np.ndarray:
    """Long call(k_low) - 2*call(k_mid) + call(k_high). Requires k_low < k_mid < k_high.

    p = p_call(k_low) - 2*p_call(k_mid) + p_call(k_high). NOT solvable via solve_cgmy yet.
    """
    if not k_low < k_mid < k_high:
        raise ValueError(f"butterfly requires k_low < k_mid < k_high, got {k_low}, {k_mid}, {k_high}")
    return european_call(k_low, ops) - 2.0 * european_call(k_mid, ops) + european_call(k_high, ops)


def _softplus_call(x: np.ndarray, strike: float, smoothing: float) -> np.ndarray:
    """Smooth (kink-free) approximation of max(exp(x)-K, 0): smoothing*log(1+exp((exp(x)-K)/smoothing)).

    -> call_payoff(x, strike) as smoothing -> 0+. Used only to build smooth
    synthetic combinations below (no kink means no quadrature splitting is
    needed: kink_x=None).
    """
    z = (np.exp(np.asarray(x, dtype=float)) - strike) / smoothing
    # numerically stable softplus
    return smoothing * np.logaddexp(0.0, z)


@dataclass(frozen=True)
class SmoothCombinationLeg:
    """One leg of a smooth synthetic combination: sign * softplus_call(strike, smoothing)."""

    sign: float
    strike: float


def smooth_synthetic_combination(
    legs: list[SmoothCombinationLeg], ops: OperatorCache, smoothing: float = 0.05
) -> np.ndarray:
    """Smooth (mollified) analogue of a call-spread/butterfly-style combination.

    My interpretation of "smooth synthetic combinations": the same linear
    combinations of call payoffs as call_spread/butterfly/etc., but with
    each leg's kink replaced by a softplus mollifier, so the resulting
    payoff is smooth everywhere and kink_x=None is used throughout (no
    quadrature-splitting needed). Please confirm this matches what was
    intended; the exact meaning of "smooth synthetic combinations" is my
    best guess, not something the task defined precisely.
    """
    grid_info = {"N": ops.N, "h": ops.h, "x_min": ops.x_min}

    def combined(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(np.asarray(x, dtype=float))
        for leg in legs:
            out = out + leg.sign * _softplus_call(x, leg.strike, smoothing)
        return out

    b = assemble_payoff_rhs(combined, grid_info, ops.p, kink_x=None)
    return ops.apply_M_inv(b)


def available_payoffs() -> list[str]:
    """List the payoff family names known to this module."""
    return ["european_call", "european_put", "call_spread", "put_spread", "butterfly", "smooth_synthetic_combination"]


SOLVABLE_PAYOFFS = frozenset({"european_put"})
"""Family names ``cgmy.reference_solver.solve_cgmy`` can currently produce a full
trajectory for. All other families in ``available_payoffs()`` can only produce a
p-vector today -- see module docstring."""
