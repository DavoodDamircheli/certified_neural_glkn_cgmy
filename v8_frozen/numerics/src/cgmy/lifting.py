"""Boundary/terminal-condition lifting for the CGMY variational problem.

Homogenizes inhomogeneous boundary and terminal data so that the Galerkin
formulation can be posed over a homogeneous trial/test space, consistently
with the lifting convention used internally by the trusted reference
solver. Needed by ``certification.residuals`` so that residuals are
evaluated against the same lifted formulation the reference solver solves.
"""

from __future__ import annotations

import numpy as np

from cgmy.reference_solver import CGMYParameters, Mesh, Payoff


def terminal_condition(
    grid: np.ndarray, payoff: Payoff
) -> np.ndarray:
    """Evaluate the (unlifted) terminal payoff on the given grid."""
    raise NotImplementedError


def boundary_lift(
    grid: np.ndarray,
    mu: CGMYParameters,
    payoff: Payoff,
    mesh: Mesh,
) -> np.ndarray:
    """Construct a lifting function matching the far-field boundary behavior."""
    raise NotImplementedError


def homogenize(
    u_raw: np.ndarray,
    lift: np.ndarray,
) -> np.ndarray:
    """Subtract a lifting function from a raw solution to obtain the homogeneous part."""
    raise NotImplementedError
