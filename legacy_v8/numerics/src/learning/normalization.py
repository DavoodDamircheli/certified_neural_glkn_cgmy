"""Normalization of CGMY parameters for neural network inputs.

    mu_tilde_j = 2 * (f_j(mu_j) - lo_j) / (hi_j - lo_j) - 1

where f_j is the identity for most components, and log for the ones where
that is mathematically appropriate: C, G, M are positive scale/rate
parameters (C is the Levy density's overall activity/concentration; G, M
are exponential tail-decay rates) that enter the CGMY model multiplicatively
/ in exponentials, so a uniform step in log-space is the natural scale for
them -- the same reasoning used for e.g. learning rates or timescales
elsewhere in ML. Y is already a bounded shape exponent (no multiplicative
structure to exploit), and r, T span less than one decade each in the
frozen box, so they are kept linear. This choice of which dimensions get
the log transform is my modeling judgment, not something the task
specified numerically -- flagging it as such.

Certification always uses de-normalized physical coefficients: this module
only ever touches mu, never a B-spline coefficient vector. Nothing in
certification.* should import from here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cgmy.reference_solver import CGMYParameters
from data.sampling import ParameterBox

DIM_ORDER = ("C", "G", "M", "Y", "r", "T")
LOG_DIMS = frozenset({"C", "G", "M"})


@dataclass(frozen=True)
class NormalizationStats:
    """Per-dimension bounds in transformed (log, where applicable) space."""

    lo: np.ndarray
    hi: np.ndarray
    log_mask: np.ndarray


def stats_from_box(box: ParameterBox) -> NormalizationStats:
    """Build normalization stats from the frozen parameter box."""
    raw_lo = np.array([box.C[0], box.G[0], box.M[0], box.Y[0], box.r[0], box.maturity[0]])
    raw_hi = np.array([box.C[1], box.G[1], box.M[1], box.Y[1], box.r[1], box.maturity[1]])
    log_mask = np.array([d in LOG_DIMS for d in DIM_ORDER])
    lo = np.where(log_mask, np.log(raw_lo), raw_lo)
    hi = np.where(log_mask, np.log(raw_hi), raw_hi)
    return NormalizationStats(lo=lo, hi=hi, log_mask=log_mask)


def _mu_to_array(mu: CGMYParameters) -> np.ndarray:
    return np.array([mu.C, mu.G, mu.M, mu.Y, mu.r, mu.T])


def normalize_mu(mu: CGMYParameters, stats: NormalizationStats) -> np.ndarray:
    """mu -> mu_tilde in [-1, 1]^6 (component-wise; log-transformed first where applicable)."""
    raw = _mu_to_array(mu)
    transformed = np.where(stats.log_mask, np.log(raw), raw)
    return 2.0 * (transformed - stats.lo) / (stats.hi - stats.lo) - 1.0


def denormalize_mu(mu_tilde: np.ndarray, stats: NormalizationStats) -> CGMYParameters:
    """Exact inverse of normalize_mu."""
    transformed = (mu_tilde + 1.0) / 2.0 * (stats.hi - stats.lo) + stats.lo
    raw = np.where(stats.log_mask, np.exp(transformed), transformed)
    return CGMYParameters(C=raw[0], G=raw[1], M=raw[2], Y=raw[3], r=raw[4], T=raw[5])


def normalize_psi(psi: np.ndarray, strike: float) -> np.ndarray:
    """Rescale a payoff B-spline coefficient vector for use as a network
    INPUT feature (never fed into certification, which stays physical).

    Raw psi/c0 magnitudes scale with the strike and with sqrt(h) (h itself
    varying across mu since the reference solver's domain width depends on
    G, M) -- up to a couple hundred in this box, which saturates a tanh-MLP's
    first layer immediately (pre-activations far outside tanh's linear
    regime) and was the actual bug behind Stage E's first failed overfit
    attempt, not model capacity. Dividing by the strike (fixed and known)
    is a simple, dimensionally sensible fix; it does not fully remove the
    sqrt(h) spread across samples, which is smaller in comparison.
    """
    return psi / strike


def normalize_tau(tau: np.ndarray | float, T: float) -> np.ndarray | float:
    """tau in [0, T] -> tau_tilde in [-1, 1]. T-dependent since maturity varies per sample.

    This is ONLY for the value fed as a network *input feature*; the
    structural initial-condition embedding in learning.mlp_coeff must
    multiply by the RAW physical tau, not this normalized value, or
    c_theta(0) = c_h^0 would not hold exactly.
    """
    return 2.0 * tau / T - 1.0
