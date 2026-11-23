"""B-spline basis utilities mirroring the reference solver's trial/test space.

Provides knot-vector construction and basis evaluation interfaces so that
other modules (certification, learning) can build vectors/matrices that
live in exactly the same discrete space as the trusted reference solver,
without duplicating or diverging from its internal spline implementation.
"""

from __future__ import annotations

import numpy as np

from cgmy.reference_solver import Mesh


def build_knot_vector(mesh: Mesh) -> np.ndarray:
    """Construct the (open, uniform-or-otherwise) knot vector for the given config."""
    raise NotImplementedError


def basis_values(
    knots: np.ndarray, degree: int, x: np.ndarray
) -> np.ndarray:
    """Evaluate all B-spline basis functions at the points ``x``.

    Returns:
        Array of shape (len(x), n_dof).
    """
    raise NotImplementedError


def basis_derivative_values(
    knots: np.ndarray, degree: int, x: np.ndarray, order: int = 1
) -> np.ndarray:
    """Evaluate derivatives of the B-spline basis functions at the points ``x``."""
    raise NotImplementedError


def n_dof(knots: np.ndarray, degree: int) -> int:
    """Number of degrees of freedom (basis functions) for a knot vector/degree pair."""
    raise NotImplementedError
