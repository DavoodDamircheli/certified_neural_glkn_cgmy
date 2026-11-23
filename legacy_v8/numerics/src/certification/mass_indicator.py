"""Mass-matrix-based (L2-type) SCREENING INDICATOR for a discrete residual
trajectory -- NOT a certificate.

The cheap online screen: no factorization beyond the mass matrix's (already
cached in ``cgmy.operators.OperatorCache``), no PCG, no explicit S_h
assembly. It approximates the dual norm of the residual with the
mass-matrix (L2) norm instead of the true energy-space Riesz map, scaled by
1/coercivity_constant -- and that approximation has NO PROVEN BOUND
PROPERTY relative to the true error. It is not a lower bound, not an upper
bound, not "usually an upper bound": no inequality relating it to E_h or to
eta_sharp/eta_guar is established anywhere in this project.

Task 2 (working prompt): this quantity (``eta_mass_indicator``) may
therefore be used ONLY to decide whether it is worth attempting a cheap
guaranteed certificate at all (an early, cheap escalation-to-correction
decision -- see ``pipeline.predict_certify_correct``) -- never to accept a
prediction outright. Only ``certification.guaranteed_certificate.eta_guar``
(and, as ground truth, ``eta_sharp``) may set a ``certified``/``accepted``
flag. See CLAUDE.md's "Guaranteed means guaranteed" hard invariant and the
working prompt's standing rule #2 (prediction -> error indicator ->
guaranteed certificate; only the third may ever set ``certified=True``).
"""

from __future__ import annotations

import numpy as np

from cgmy.operators import OperatorCache


def mass_norm(residual: np.ndarray, mass_solver: OperatorCache) -> float:
    """sqrt(r^T M_h^{-1} r), using the cached M_h factorization.

    ``residual`` is the vector returned by
    ``certification.residuals.compute_cn_residual``/``compute_be_residual``
    (zero at Dirichlet dofs by construction).
    """
    Minv_r = mass_solver.apply_M_inv(residual)
    return float(np.sqrt(residual @ Minv_r))


def eta_mass_indicator(
    residuals: np.ndarray, mass_solver: OperatorCache, dt: float, coercivity_constant: float
) -> float:
    """eta_mass_indicator, the cheap online screening INDICATOR -- NOT a
    certificate (see module docstring). Must never gate a ``certified``/
    ``accepted`` decision on its own; only an early reject-to-correction.

        eta_mass_indicator^2 = (dt / coercivity_constant) * sum_n (r^{n+1/2})^T M_h^{-1} r^{n+1/2}

    Args:
        residuals: Array of shape (n_steps, n_dof), e.g. from
            ``certification.residuals.compute_all_residuals``.
        mass_solver: Supplies the cached M_h factorization via apply_M_inv
            (typically the same OperatorCache used to compute the residuals).
        dt: Time step size (traj.solver_metadata.dt).
        coercivity_constant: alpha, a lower bound on S_h(mu)'s smallest
            eigenvalue (see certification.constants). Must be > 0.

    Raises:
        ValueError: If coercivity_constant <= 0.
    """
    if coercivity_constant <= 0:
        raise ValueError(
            f"coercivity_constant must be > 0 to form the indicator, got "
            f"{coercivity_constant}. A non-positive value means S_h(mu) is not "
            f"known to be coercive at this mu -- eta_mass_indicator is not "
            f"meaningful here (see certification.constants.StabilityConstants.is_spd)."
        )

    total = 0.0
    for r in np.atleast_2d(residuals):
        Minv_r = mass_solver.apply_M_inv(r)
        total += float(r @ Minv_r)

    eta_mass_indicator_sq = (dt / coercivity_constant) * total
    return float(np.sqrt(eta_mass_indicator_sq))
