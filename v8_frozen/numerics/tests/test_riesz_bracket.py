"""GATE 2 (CLAUDE.md): the PCG Riesz bracket must satisfy l(v) <= q <= u(v)
at every checkpoint iteration count, with the bracket width shrinking
monotonically as PCG converges. q = r^T S_h(mu)^{-1} r is computed exactly
via a direct dense solve for comparison.

See certification/riesz_solver.py's module docstring for the bracket
derivation and the caveat that it is a standard, provably-valid
construction, not verified against the manuscript's own (possibly
different) bracket formula.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters, Mesh
from certification.constants import compute_stability_constants, dense_from_apply
from certification.riesz_solver import solve_riesz_pcg

MU = CGMYParameters(C=1.0, G=5.0, M=5.0, Y=0.5, r=0.05, T=0.5)
CHECKPOINTS = [0, 1, 2, 5, 10, 200]  # "near-converged" = generous cap
BRACKET_SLACK = 1e-6  # floating-point slack for the near-converged checkpoint


def _small_cache() -> OperatorCache:
    N, p, x_min, x_max = 32, 3, -3.0, 3.0
    return OperatorCache(Mesh(N=N, p=p, x_min=x_min, x_max=x_max), x_min, x_max)


def test_gate2_bracket_contains_exact_value_and_shrinks_monotonically():
    ops = _small_cache()
    const = compute_stability_constants(MU, ops)
    assert const.is_spd, "S_h(mu) must be SPD at the Gate 2 test point"

    S = dense_from_apply(lambda v: ops.apply_S(MU, v), ops.n_dof)
    rng = np.random.default_rng(0)
    r = rng.standard_normal(ops.n_dof)
    q_exact = float(r @ np.linalg.solve(S, r))

    rows = []
    widths = []
    for k in CHECKPOINTS:
        b = solve_riesz_pcg(r, MU, ops, const.alpha_certified, max_iter=k, preconditioner="mass")
        rows.append((k, b.lower_bound, b.upper_bound, b.n_pcg_iterations, b.relative_riesz_residual))
        widths.append(b.upper_bound - b.lower_bound)

        assert b.lower_bound <= q_exact + BRACKET_SLACK, (
            f"k={k}: lower_bound={b.lower_bound:.6f} > q_exact={q_exact:.6f}"
        )
        assert q_exact <= b.upper_bound + BRACKET_SLACK, (
            f"k={k}: q_exact={q_exact:.6f} > upper_bound={b.upper_bound:.6f}"
        )

    print(f"\nGate 2: q_exact = {q_exact:.6f}")
    print(f"{'k':>4} {'l(v)':>14} {'u(v)':>14} {'u-l':>14} {'n_iters':>8} {'rel_res':>10}")
    for (k, l, u, ni, rr), w in zip(rows, widths):
        print(f"{k:4d} {l:14.6f} {u:14.6f} {w:14.6e} {ni:8d} {rr:10.3e}")

    # Bracket width must decrease monotonically (non-increasing) across the
    # checkpoints, which are themselves increasing PCG iteration budgets.
    for i in range(1, len(widths)):
        assert widths[i] <= widths[i - 1] + 1e-9, (
            f"bracket width increased between checkpoint {CHECKPOINTS[i-1]} "
            f"({widths[i-1]:.3e}) and {CHECKPOINTS[i]} ({widths[i]:.3e})"
        )

    # Near-converged: bracket should have collapsed close to q_exact.
    assert widths[-1] < 1e-6, f"bracket did not collapse at k={CHECKPOINTS[-1]}: width={widths[-1]:.3e}"


@pytest.mark.parametrize("preconditioner", ["none", "mass", "jacobi", "banded"])
def test_gate2_bracket_holds_for_every_preconditioner(preconditioner: str):
    """The bracket's validity must not depend on which preconditioner PCG
    uses (only its convergence speed should)."""
    ops = _small_cache()
    const = compute_stability_constants(MU, ops)
    S = dense_from_apply(lambda v: ops.apply_S(MU, v), ops.n_dof)
    rng = np.random.default_rng(1)
    r = rng.standard_normal(ops.n_dof)
    q_exact = float(r @ np.linalg.solve(S, r))

    for k in CHECKPOINTS:
        b = solve_riesz_pcg(r, MU, ops, const.alpha_certified, max_iter=k, preconditioner=preconditioner)
        assert b.lower_bound <= q_exact + BRACKET_SLACK
        assert q_exact <= b.upper_bound + BRACKET_SLACK
