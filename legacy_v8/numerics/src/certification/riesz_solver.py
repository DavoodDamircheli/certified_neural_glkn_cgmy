"""PCG Riesz solve of S_h(mu) v = r, with a lower/upper bracket on
r^T S_h(mu)^{-1} r that holds in exact arithmetic for any v, under the
hypothesis coercivity_constant <= lambda_min(S) (v5 scope: evaluated in
ordinary floating point, not a validated enclosure).

IMPORTANT -- the bracket formula here is NOT copied from "the manuscript":
I did not have access to its exact bracket construction when writing this,
and fabricating a formula I couldn't verify would risk a "guaranteed" bound
that is not actually guaranteed -- precisely the failure mode CLAUDE.md's
Hard Invariant #5 exists to prevent. What is implemented instead is a
standard, textbook-provable bracket, derived below, that is mathematically
guaranteed valid for ANY symmetric positive-definite S and any vector v
(not just a converged PCG iterate). Please check it against the manuscript
and tell me if a different (e.g. Lanczos/Gauss-quadrature) construction
should replace it.

Derivation. Let S be SPD, r fixed, q := r^T S^{-1} r. For any v:

    q - (2 r^T v - v^T S v) = (v - S^{-1}r)^T S (v - S^{-1}r) >= 0

so l(v) := 2 r^T v - v^T S v is a valid lower bound for any v (this is the
standard variational/duality characterization q = max_v [2 r^T v - v^T S v],
attained at v = S^{-1} r). Writing s := r - S v (the PCG residual at v):

    q = l(v) + s^T S^{-1} s                          (exact identity)

and since S^{-1} has largest eigenvalue 1/lambda_min(S):

    s^T S^{-1} s <= ||s||^2 / lambda_min(S)

giving the upper bound u(v) := l(v) + ||s||^2 / lambda_min_bound, valid
whenever lambda_min_bound <= lambda_min(S) (i.e. coercivity_constant is a
genuine lower bound, not just an estimate). By construction l(v) <= q <=
u(v) for every v, and u(v) - l(v) = ||s||^2 / lambda_min_bound shrinks
with ||s|| (monotone in ||s||, though ||s|| itself need not decrease at
every CG iteration -- see CLAUDE.md Gate 2) -- which is what Gate 2 checks.

Task 3 (working prompt): this file's own derivation above already stated
the requirement precisely ("a genuine lower bound, not just an estimate")
-- what changed is that ``certification.constants`` now supplies a value
that is a lower bound in exact arithmetic (``StabilityConstants.
alpha_lower_estimate``, formerly ``alpha_certified``; Weyl enclosure, see
that module and notes/03_coercivity.md), instead of the caller passing a
raw ``numpy.linalg.eigh`` estimate. v5 scope: that value is computed in
floating point and is not a validated enclosure, so ``coercivity_constant
<= lambda_min(S)`` is a stated HYPOTHESIS of the upper bound, not a
machine-checked fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse.linalg as spla

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters

Preconditioner = Literal["none", "mass", "jacobi", "banded"]


@dataclass(frozen=True)
class RieszBracket:
    """Result of one PCG Riesz solve for a single residual vector.

    Attributes:
        lower_bound: l(v), a lower bound on r^T S_h(mu)^{-1} r (exact
            arithmetic identity + inequality; floating-point value).
        upper_bound: u(v), an upper bound in exact arithmetic under the
            HYPOTHESIS coercivity_constant <= lambda_min(S_h(mu)) (see
            module docstring); floating-point value.
        n_pcg_iterations: Number of PCG iterations actually run (may be
            less than max_iter if PCG's own tolerance was hit first).
        relative_riesz_residual: ||s|| / ||r||, where s = r - S_h(mu) v.
        v: The PCG iterate (Riesz-problem solution estimate).
    """

    lower_bound: float
    upper_bound: float
    n_pcg_iterations: int
    relative_riesz_residual: float
    v: np.ndarray


def _preconditioner_linop(
    preconditioner: Preconditioner, ops: OperatorCache, mu: CGMYParameters
) -> spla.LinearOperator | None:
    """Task 9 (working prompt): the four preconditioner candidates actually
    implemented (see notes/09_preconditioner.md for the benchmark and why
    incomplete Cholesky is NOT one of them -- S_h(mu) has no natural sparse
    structure to exploit without an ad hoc, separately-justified banding
    approximation, which 'banded' below already IS)."""
    if preconditioner == "none":
        return None
    if preconditioner == "mass":
        return spla.LinearOperator(
            (ops.n_dof, ops.n_dof), matvec=ops.apply_M_inv, dtype=float
        )
    if preconditioner == "jacobi":
        diag = ops.diag_S(mu)
        return spla.LinearOperator(
            (ops.n_dof, ops.n_dof), matvec=lambda v: v / diag, dtype=float
        )
    if preconditioner == "banded":
        lu = ops.banded_preconditioner_lu(mu)
        return spla.LinearOperator(
            (ops.n_dof, ops.n_dof), matvec=lu.solve, dtype=float
        )
    raise ValueError(
        f"unknown preconditioner {preconditioner!r}, expected 'none', 'mass', 'jacobi', or 'banded'"
    )


def solve_riesz_pcg(
    r: np.ndarray,
    mu: CGMYParameters,
    ops: OperatorCache,
    coercivity_constant: float,
    *,
    max_iter: int,
    preconditioner: Preconditioner = "none",  # Task 9, working prompt -- see notes/09_preconditioner.md
    pcg_tol: float = 1e-14,
) -> RieszBracket:
    """Run (P)CG on S_h(mu) v = r for at most max_iter iterations and bracket
    q = r^T S_h(mu)^{-1} r.

    Args:
        r: The residual vector (e.g. one row of compute_all_residuals'
            output). Zero at Dirichlet dofs by construction.
        mu: CGMY parameters (S_h(mu) is applied matrix-free via ops.apply_S).
        ops: OperatorCache built from the same mesh r was computed on.
        coercivity_constant: A genuine lower bound on lambda_min(S_h(mu)).
            Callers MUST pass ``StabilityConstants.alpha_lower_estimate``
            (formerly ``alpha_certified``; never ``.lambda_min_numerical``)
            -- see certification.constants and notes/03_coercivity.md. It is
            a lower bound in exact arithmetic evaluated in floating point;
            alpha <= lambda_min(S) is a hypothesis of the upper bound. The upper bound below is only valid if
            this is a real lower bound, not merely close to lambda_min.
        max_iter: Hard cap on PCG iterations (this is the "capped max_iter"
            the task asks for -- callers wanting a "near-converged" result
            should pass a generous value here, not rely on pcg_tol alone).
        preconditioner: 'none', 'mass' (OperatorCache.apply_M_inv),
            'jacobi' (diag(S_h(mu))^{-1}, Task 9 -- no dense assembly, see
            OperatorCache.diag_S), or 'banded' (Task 9's "operator-aware"
            option -- a truncated-bandwidth approximation of S_h(mu),
            factored once and cached per mu, see
            OperatorCache.banded_preconditioner_lu). This is the
            "preconditioner choice through config" -- read it from
            configs/base.yaml's certification.preconditioner field at the
            call site, this function just accepts it as a parameter.
        pcg_tol: scipy.sparse.linalg.cg's convergence tolerance; PCG may
            stop before max_iter if this is hit first.

    Raises:
        ValueError: If coercivity_constant <= 0 (no valid upper bound can
            be formed).
    """
    if coercivity_constant <= 0:
        raise ValueError(
            f"coercivity_constant must be > 0, got {coercivity_constant}: "
            f"cannot form a guaranteed upper bound without a genuine lower "
            f"bound on lambda_min(S_h(mu))."
        )

    n = ops.n_dof
    S_op = spla.LinearOperator((n, n), matvec=lambda v: ops.apply_S(mu, v), dtype=float)
    M_op = _preconditioner_linop(preconditioner, ops, mu)

    iterates: list[np.ndarray] = []

    def _cb(xk: np.ndarray) -> None:
        iterates.append(xk.copy())

    if max_iter <= 0:
        v = np.zeros(n)
    else:
        v, info = spla.cg(S_op, r, rtol=pcg_tol, atol=0.0, maxiter=max_iter, M=M_op, callback=_cb)
        v = iterates[-1] if iterates else np.zeros(n)

    s = r - ops.apply_S(mu, v)
    l_v = float(2.0 * r @ v - v @ ops.apply_S(mu, v))
    u_v = l_v + float(s @ s) / coercivity_constant
    r_norm = float(np.linalg.norm(r))
    rel_riesz_residual = float(np.linalg.norm(s)) / r_norm if r_norm > 0 else 0.0

    return RieszBracket(
        lower_bound=l_v,
        upper_bound=u_v,
        n_pcg_iterations=len(iterates),
        relative_riesz_residual=rel_riesz_residual,
        v=v,
    )
