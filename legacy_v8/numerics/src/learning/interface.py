"""Shared evaluation interface every coefficient-prediction model implements.

MLPCoeffNet (Stage E), DeepONetCoeffNet, PODGalerkinROM, and ProjectedFNO
each implement ``predict_trajectory(mu_tilde, psi, c0, time_grid, T)``,
returning a ``(len(time_grid), n_dof)`` physical coefficient array whose
first row is ``c0`` EXACTLY (bit-identical, not "close"), in the same
B-spline dof ordering the reference solver uses -- so
``wrap_as_trajectory`` below can drop the result into
``certification.residuals.compute_all_residuals`` with no shape or
ordering fixups, exactly as it accepts a real ``solve_cgmy`` output.

Every model achieves the exact-IC property the SAME structural way:

    c_theta(tau) = c0 + [g(tau) - g(0)]

for whatever raw thing g the model's own architecture computes (a network
correction for MLP/DeepONet, a POD-Galerkin reconstruction for POD, a
Pi_h-projected grid function for FNO) -- g(0) cancels by construction, so
c_theta(0) = c0 regardless of untrained/random weights or basis choice.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np
import torch

from cgmy.reference_solver import GalerkinTrajectory


class CoefficientModel(Protocol):
    def predict_trajectory(
        self,
        mu_tilde: np.ndarray,
        psi: np.ndarray,
        c0: np.ndarray,
        time_grid: np.ndarray,
        T: float,
    ) -> np.ndarray:
        """Return (len(time_grid), n_dof) physical coefficients; row 0 == c0 exactly."""
        ...


def wrap_as_trajectory(reference: GalerkinTrajectory, predicted_coefficients: np.ndarray) -> GalerkinTrajectory:
    """Substitute predicted coefficients into a copy of a reference trajectory.

    Mesh/forcing/solver_metadata are kept unchanged, so
    certification.residuals functions accept the result exactly as they
    accept a real solve_cgmy output.

    Raises:
        ValueError: If the shape doesn't match the reference trajectory's
            coefficients array (n_steps+1, n_dof) -- callers must not need
            to fix this up themselves.
    """
    if predicted_coefficients.shape != reference.coefficients.shape:
        raise ValueError(
            f"predicted coefficients shape {predicted_coefficients.shape} != "
            f"reference trajectory shape {reference.coefficients.shape}"
        )
    return dataclasses.replace(reference, coefficients=predicted_coefficients)


def to_tensor(x: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert a numpy array to a torch tensor of the given dtype (helper
    shared by every torch-based model's predict_trajectory)."""
    return torch.tensor(np.asarray(x), dtype=dtype)
