"""The sharp reference certificate and the practical guaranteed certificate.

eta_sharp: direct/reference certificate. Explicitly assembles and factors
the full S_h(mu) (practical mesh sizes only -- same guard as
``certification.constants``), then computes the residual's exact dual norm
via a direct solve. Role: ground truth for checking the PCG-based
eta_guar, not something to run online.

eta_guar: the practical guaranteed certificate. Uses
``certification.riesz_solver.solve_riesz_pcg``'s UPPER bound at every step
(CLAUDE.md Hard Invariant #5: "eta_guar uses the upper Riesz bracket bound
at every time step... zero certificate violations is a correctness
requirement, not a metric to optimize"), with a user-settable PCG iteration
cap giving a continuum from cheap screening (small max_iter) to
near-sharp (large max_iter, where eta_guar -> eta_sharp).

SCOPE (CGMY_V4_REMAINING_WORK.md item 2, added v4): both certificates here
are rigorous in EXACT arithmetic (the PCG bracket is an algebraic identity
plus one inequality, valid for any v -- see riesz_solver's own module
docstring; the Weyl enclosure behind alpha_certified is a rigorous theorem
given exact arithmetic -- see certification.constants.weyl_residual_enclosure's
scope caveat). Every number this module actually RETURNS is computed and
evaluated in ordinary double-precision floating point (dense
eigendecomposition, Cholesky/PCG solves, this function's own arithmetic) --
an empirical check of an exact-arithmetic guarantee, not a
validated/interval-arithmetic bound that accounts for floating-point
rounding at every step. Do not describe eta_sharp/eta_guar's returned
values as "machine-validated price bounds" in any report or manuscript
text -- see certification/CERTIFICATE_ARITHMETIC_SPEC.md and
reports/v4_validated_arithmetic_plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters
from certification.constants import dense_from_apply
from certification.riesz_solver import Preconditioner, solve_riesz_pcg


@dataclass(frozen=True)
class Certificate:
    """A certificate value together with its per-step ingredients.

    Attributes:
        eta: The certificate value (eta_sharp or eta_guar).
        per_step_quadratic_form: The n_steps values summed (dt-scaled) to
            form eta^2 -- r^T S_h^{-1} r exactly for eta_sharp, or the PCG
            upper bound u(v) at each step for eta_guar.
        n_pcg_iterations: Per-step PCG iteration counts (eta_guar only;
            None for eta_sharp, which uses a direct solve).
        coordinate_space: "legacy_full" or "constrained_free" -- see
            certification.constants.StabilityConstants's field of the same
            name; same caveat (a label set by the caller via `ops`, not
            inferred). CGMY_V4_REMAINING_WORK.md item 1.
    """

    eta: float
    per_step_quadratic_form: np.ndarray
    n_pcg_iterations: np.ndarray | None
    coordinate_space: str = "legacy_full"


def eta_sharp(
    residuals: np.ndarray,
    mu: CGMYParameters,
    ops: OperatorCache,
    dt: float,
    max_n_dof: int = 400,
    *,
    coordinate_space: str = "legacy_full",
) -> Certificate:
    """Direct/reference certificate: eta_sharp^2 = dt * sum_n r_n^T S_h(mu)^{-1} r_n.

    Explicitly assembles S_h(mu) as a dense (n_dof, n_dof) matrix and
    factors it once (Cholesky, since S_h(mu) must be SPD for this to be
    meaningful), then solves it exactly for every residual row.

    Generic in ``ops`` -- pass a ``certification.constrained.FreeOperatorCache``
    and ``residuals`` shaped (n_steps, n_free) to get the free-space
    eta_sharp_I with no code change; pass coordinate_space="constrained_free"
    to label the result accordingly.

    Raises:
        ValueError: If ops.n_dof > max_n_dof.
        np.linalg.LinAlgError: If S_h(mu) is not SPD (Cholesky fails) --
            this is a real finding (see Stage A's difficult-corners
            results), not something to work around silently.
    """
    n = ops.n_dof
    if n > max_n_dof:
        raise ValueError(
            f"n_dof={n} exceeds max_n_dof={max_n_dof}: dense factorization of "
            f"S_h(mu) is only practical at moderate mesh sizes."
        )

    S = dense_from_apply(lambda v: ops.apply_S(mu, v), n)
    try:
        cho = sla.cho_factor(S)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"S_h(mu) is not SPD (Cholesky factorization failed) for mu={mu}: "
            f"eta_sharp is not meaningful at this parameter point. This is a "
            f"finding, not a bug -- see Stage A's difficult-corners results "
            f"for corners where this is expected."
        ) from exc

    residuals = np.atleast_2d(residuals)
    quad_forms = np.array([r @ sla.cho_solve(cho, r) for r in residuals])
    eta = float(np.sqrt(dt * quad_forms.sum()))
    return Certificate(
        eta=eta, per_step_quadratic_form=quad_forms, n_pcg_iterations=None,
        coordinate_space=coordinate_space,
    )


def eta_guar(
    residuals: np.ndarray,
    mu: CGMYParameters,
    ops: OperatorCache,
    dt: float,
    coercivity_constant: float,
    *,
    max_pcg_iterations: int,
    preconditioner: Preconditioner = "none",  # Task 9, working prompt -- see notes/09_preconditioner.md
    coordinate_space: str = "legacy_full",
) -> Certificate:
    """Practical guaranteed certificate: eta_guar^2 = dt * sum_n u(v^{n+1/2}),
    using the PCG UPPER bound at every step (never the lower bound, never
    the exact value -- see module docstring).

    "Guaranteed" = by the exact-arithmetic bracket theorem, under the explicit
    HYPOTHESIS ``coercivity_constant <= lambda_min(S)``; the returned value is
    ordinary floating point (see module docstring SCOPE).

    As max_pcg_iterations increases, eta_guar converges toward eta_sharp
    (Gate 2's bracket contraction, summed over steps). It is NOT monotone in
    the iteration cap in general: ||s|| of a CG iterate need not decrease at
    every iteration (CLAUDE.md Gate 2; exp03 widths 563, 420, 631, ...).

    Generic in ``ops`` -- pass a ``certification.constrained.FreeOperatorCache``
    and ``residuals`` shaped (n_steps, n_free) to get the free-space
    eta_guar_I with no code change; pass coordinate_space="constrained_free"
    to label the result accordingly.
    """
    residuals = np.atleast_2d(residuals)
    upper_bounds = np.empty(len(residuals))
    n_iters = np.empty(len(residuals), dtype=int)
    for n, r in enumerate(residuals):
        bracket = solve_riesz_pcg(
            r, mu, ops, coercivity_constant,
            max_iter=max_pcg_iterations, preconditioner=preconditioner,
        )
        upper_bounds[n] = bracket.upper_bound
        n_iters[n] = bracket.n_pcg_iterations

    eta = float(np.sqrt(dt * upper_bounds.sum()))
    return Certificate(
        eta=eta, per_step_quadratic_form=upper_bounds, n_pcg_iterations=n_iters,
        coordinate_space=coordinate_space,
    )


def is_accepted(certificate: Certificate, tolerance: float) -> bool:
    """Decide whether a prediction's certificate is small enough to accept as-is."""
    return certificate.eta <= tolerance
