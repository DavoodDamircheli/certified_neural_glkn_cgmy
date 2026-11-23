"""Task 4 (working prompt): SPD-failure diagnostic harness.

Every certificate in this project rests on S_h(mu) being SPD (Hard
Invariant, `certification.constants`/`riesz_solver`). Stage D's own prior
narrowing of the parameter box was validated against SPD failures, but the
working prompt's Task 4 says failures persist at some rate over the frozen
box -- this module exists to find out, honestly, whether that is still
true, and if so, to record every failure with enough detail to diagnose it,
never silently.

``check_and_log_spd`` is the single call site: compute the stability
constants for one (mesh, mu), and on failure (S_h(mu) not SPD, i.e.
``lambda_min_numerical <= 0`` -- alpha_certified (CSV column name kept for the
existing results/spd_failures.csv schema; v5: it holds
StabilityConstants.alpha_lower_estimate, a floating-point estimate) is NaN in
that case by construction, see certification.constants), append a full
diagnostic row to a CSV. A failure that is found but not logged is exactly
the kind of "quietly known, never written down" problem CLAUDE.md's
"When a numerical result is surprising, say so and investigate" rule exists
to prevent -- so this never returns without writing the row first when
there is one to write.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters
from certification.constants import StabilityConstants, compute_stability_constants, dense_from_apply

# Task 4 asks to bisect across "quadrature error" and "truncation of the
# nonlocal integral" as an axis independent of spatial resolution N. This
# project's actual assembly path (cgmy_bspline, unchanged, trusted) has no
# such independent knob:
#   - K_G_minus/K_M_plus (the nonlocal jump operators) are built from an
#     EXACT closed-form Fourier symbol evaluated on an IFFT grid of size
#     L=2*N (cgmy_bspline.connection_coefficients.tempered_coeffs_via_ifft)
#     -- not a truncated/approximate quadrature rule, and its resolution is
#     tied directly to N, not independently adjustable.
#   - M_h/D_h (mass/derivative matrices) use scipy.integrate.quad with
#     hardcoded epsabs/epsrel = 1e-12..1e-14 (cgmy_bspline.matrices) --
#     already accurate far beyond what any SPD failure at O(1) magnitude
#     could plausibly trace to, and not exposed as a parameter to this
#     repo or to OperatorCache.
# See notes/04_spd_study.md for the full account. This constant documents
# that fact in the CSV rather than silently omitting the column.
QUADRATURE_NOTE = (
    "no independent quadrature-order knob exists in this discretization: "
    "K_G/K_M use an exact closed-form Fourier symbol on an IFFT grid tied "
    "to N (not a quadrature rule); M_h/D_h use scipy.integrate.quad at "
    "fixed epsabs/epsrel=1e-12..1e-14 -- see notes/04_spd_study.md"
)

CSV_FIELDS = [
    "C", "G", "M", "Y", "r", "T",
    "lambda_min", "lambda_max", "cond",
    "N", "p", "dt", "domain_width", "x_min", "x_max",
    "quadrature_note",
    "symmetry_error_rel", "matrix_norm_fro", "min_diag_entry",
    "alpha_certified", "safety_ratio", "coercivity_strategy",
    "source",
]


def check_and_log_spd(
    mu: CGMYParameters,
    ops: OperatorCache,
    csv_path: str | Path,
    *,
    dt: float | None = None,
    source: str = "",
) -> StabilityConstants:
    """Compute stability constants for (mesh, mu); on SPD failure
    (lambda_min_numerical <= 0), append a full diagnostic row to csv_path.
    Returns the StabilityConstants either way -- callers get the same
    return value as a bare ``compute_stability_constants`` call, with
    logging as a side effect on failure only.
    """
    const = compute_stability_constants(mu, ops)
    if const.lambda_min_numerical <= 0:
        _log_failure(mu, ops, const, csv_path, dt=dt, source=source)
    return const


def _log_failure(
    mu: CGMYParameters,
    ops: OperatorCache,
    const: StabilityConstants,
    csv_path: str | Path,
    *,
    dt: float | None,
    source: str,
) -> None:
    S = dense_from_apply(lambda v: ops.apply_S(mu, v), ops.n_dof)
    sym_err_abs = float(np.max(np.abs(S - S.T)))
    s_norm = float(np.linalg.norm(S, "fro"))
    sym_err_rel = sym_err_abs / s_norm if s_norm > 0 else float("nan")
    min_diag = float(np.min(np.diag(S)))

    row = {
        "C": mu.C, "G": mu.G, "M": mu.M, "Y": mu.Y, "r": mu.r, "T": mu.T,
        "lambda_min": const.lambda_min_numerical,
        "lambda_max": const.continuity_constant,
        "cond": const.condition_number_S_h,
        "N": ops.N, "p": ops.p,
        "dt": dt if dt is not None else float("nan"),
        "domain_width": ops.x_max - ops.x_min,
        "x_min": ops.x_min, "x_max": ops.x_max,
        "quadrature_note": QUADRATURE_NOTE,
        "symmetry_error_rel": sym_err_rel,
        "matrix_norm_fro": s_norm,
        "min_diag_entry": min_diag,
        "alpha_certified": const.alpha_certified,
        "safety_ratio": const.safety_ratio,
        "coercivity_strategy": const.coercivity_strategy,
        "source": source,
    }

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def reproduce_failure(row: dict, domain_scale_override: tuple[float, float] | None = None) -> StabilityConstants:
    """Rebuild (mesh, mu) from one results/spd_failures.csv row and
    recompute stability constants standalone -- the "reproduce each
    failure from its saved mu" step (Task 4, working prompt).
    """
    from cgmy.reference_solver import Mesh

    mu = CGMYParameters(
        C=float(row["C"]), G=float(row["G"]), M=float(row["M"]),
        Y=float(row["Y"]), r=float(row["r"]), T=float(row["T"]),
    )
    x_min, x_max = float(row["x_min"]), float(row["x_max"])
    if domain_scale_override is not None:
        x_min, x_max = domain_scale_override
    N, p = int(float(row["N"])), int(float(row["p"]))
    mesh = Mesh(N=N, p=p, x_min=x_min, x_max=x_max)
    ops = OperatorCache(mesh, x_min, x_max)
    return compute_stability_constants(mu, ops)
