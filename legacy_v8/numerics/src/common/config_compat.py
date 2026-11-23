"""Backward-compatible config-key readers.

CGMY_V4_REMAINING_WORK.md item 1: ``configs/base.yaml``'s ``spline_degree``
key is renamed to ``spline_order`` (the vendored solver's own docstrings
consistently call this quantity "order", not "degree" -- order 3 = degree 2
= quadratic, see ``reports/BASIS_AND_BOUNDARY_AUDIT.md`` section 2; this is
a naming fix only, the value stays 3, NOT a switch to cubic splines). Old
checkpoints/configs/snapshots on disk (``results/*/config_snapshot.yaml``)
still say ``spline_degree`` and must keep loading -- this is the single
place that compatibility is handled, so a reader is never silently missing
it in one call site while having it in another.
"""

from __future__ import annotations

import warnings
from typing import Any


def read_spline_order(discretization: dict[str, Any]) -> int:
    """Read the B-spline order ``p`` from a ``configs/base.yaml``-shaped
    ``discretization`` dict: prefers the new ``spline_order`` key, falls
    back to the old ``spline_degree`` key with a ``DeprecationWarning``
    (never an error -- old on-disk config snapshots must keep loading).

    Raises:
        KeyError: If neither key is present.
    """
    if "spline_order" in discretization:
        return int(discretization["spline_order"])
    if "spline_degree" in discretization:
        warnings.warn(
            "discretization.spline_degree is deprecated (it names a B-spline "
            "ORDER, not a degree -- order 3 = degree 2 = quadratic; see "
            "reports/BASIS_AND_BOUNDARY_AUDIT.md). Use spline_order instead. "
            "Reading the old key for this call for backward compatibility.",
            DeprecationWarning,
            stacklevel=2,
        )
        return int(discretization["spline_degree"])
    raise KeyError(
        "discretization dict has neither 'spline_order' nor the deprecated "
        "'spline_degree' key"
    )
