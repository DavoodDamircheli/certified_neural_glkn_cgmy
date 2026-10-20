"""Gate A1: identify the actual trial space from the code, not from the DOF
count quoted in the paper. Produces audit/basis_manifest.json and
audit/basis_supports.csv.

This module *imports* the production basis/mesh code under audit
(cgmy_bspline.bspline_basis, src/cgmy/dirichlet_bc.py) -- unlike Gate A2's
independent assembly, A1 is explicitly about *identifying* what the actual
code does, not re-deriving it independently. The energy-space/boundary
reasoning below is cross-checked against the general theory of cardinal
B-splines (closed-form support width, vanishing-at-support-edges), which is
independent of this project's code even though no project-external library
is called.

Run: python audit/basis_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "vendor"))

from cgmy_bspline.bspline_basis import active_indices, cardinal_bspline, phi  # noqa: E402
from common.config_compat import read_spline_order  # noqa: E402

# Production discretization (configs/base.yaml) -- read, not hardcoded blind:
# this script reads the same YAML the pipeline reads, rather than retyping
# the numbers.
import yaml  # noqa: E402

with open(REPO_ROOT / "configs" / "base.yaml") as f:
    _cfg = yaml.safe_load(f)

N = _cfg["discretization"]["n_elements"]
P = read_spline_order(_cfg["discretization"])  # v4: spline_order (was spline_degree); see finding below
X_MIN = _cfg["discretization"]["domain_left"]
X_MAX = _cfg["discretization"]["domain_right"]
H = (X_MAX - X_MIN) / N


def measure_support_width(p: int) -> float:
    """Numerically measure the support width of cardinal_bspline(., p) on a
    fine grid, independent of any docstring's claimed degree/order."""
    xs = np.linspace(-0.5, p + 0.5, 400_000)
    vals = cardinal_bspline(xs, p)
    nz = xs[vals > 1e-10]
    return float(nz.max() - nz.min())


def measure_value_at_support_edges(p: int) -> tuple[float, float]:
    """N_p(0) and N_p(p): should be exactly 0 for p>=2 (standard B-spline
    fact), confirming the zero-extension has no jump discontinuity there."""
    return float(cardinal_bspline(np.array([0.0]), p)[0]), \
        float(cardinal_bspline(np.array([float(p)]), p)[0])


def main():
    order = P  # see "degree vs order" finding in BASIS_AND_BOUNDARY_AUDIT.md
    degree = order - 1
    support_width = measure_support_width(order)
    val_left_edge, val_right_edge = measure_value_at_support_edges(order)

    ks = active_indices(N, order)
    n_dof = len(ks)
    assert n_dof == N + order - 1

    # Dirichlet dof indices: identical formula to
    # src/cgmy/dirichlet_bc.py:bc_dofs_for_mesh and
    # src/correction/krylov.py:bc_dofs_for_mesh (both mesh-only, (p-1) at
    # each end) -- reproduced here by the same closed-form, not imported,
    # so this script can run standalone; cross-checked equal below.
    bc_left = list(range(order - 1))
    bc_right = list(range(n_dof - order + 1, n_dof))

    try:
        from cgmy.dirichlet_bc import bc_dofs_for_mesh
        from cgmy.reference_solver import MeshInfo
        nodes = np.linspace(X_MIN, X_MAX, N + 1)
        mesh_info = MeshInfo(N=N, p=order, h=H, x_min=X_MIN, x_max=X_MAX, n_dof=n_dof, nodes=nodes)
        bc_left_prod, bc_right_prod = bc_dofs_for_mesh(mesh_info)
        cross_check_ok = (list(bc_left_prod) == bc_left) and (list(bc_right_prod) == bc_right)
    except Exception as exc:  # pragma: no cover - diagnostic only
        cross_check_ok = None
        cross_check_error = str(exc)
    else:
        cross_check_error = None

    rows = []
    for i, k in enumerate(ks):
        xi_left, xi_right = int(k), int(k) + order
        x_left = float(X_MIN + xi_left * H)
        x_right = float(X_MIN + xi_right * H)
        fully_interior = bool((k >= 0) and (k + order <= N))
        touches_left = bool(xi_left < 0)
        touches_right = bool(xi_right > N)
        is_dirichlet = (i in bc_left) or (i in bc_right)
        # Zero-extension energy-space membership: value at the extreme
        # points of this basis function's own support is 0 for order>=2
        # (measured above, universal, not k-dependent), so wherever the
        # support lies fully inside Omega the zero-extension is continuous
        # (no jump) hence in H^1. Where the support pokes outside Omega
        # (touches_left/touches_right True), the function is NOT part of
        # the homogeneous space at all -- it is a Dirichlet dof, prescribed
        # by the lifting, not a variational unknown; "energy-space
        # membership" does not apply to it in the same sense.
        if is_dirichlet:
            energy_space_role = "dirichlet_prescribed_not_variational"
        elif fully_interior:
            energy_space_role = "variational_H1_member"
        else:
            # Should not occur for p-1 boundary padding as derived above;
            # flag loudly if it ever does (would mean the bc_dofs formula
            # and the support geometry have silently diverged).
            energy_space_role = "UNEXPECTED_partial_support_non_dirichlet"
        rows.append(
            dict(
                i=i, k=int(k),
                xi_left=xi_left, xi_right=xi_right,
                x_left=x_left, x_right=x_right,
                fully_interior=fully_interior,
                touches_left_boundary=touches_left,
                touches_right_boundary=touches_right,
                is_dirichlet_dof=is_dirichlet,
                energy_space_role=energy_space_role,
            )
        )

    n_interior = int(sum(r["fully_interior"] for r in rows))
    n_dirichlet = int(sum(r["is_dirichlet_dof"] for r in rows))
    n_unexpected = int(sum(r["energy_space_role"].startswith("UNEXPECTED") for r in rows))

    manifest = {
        "source_config": "configs/base.yaml",
        "N_elements": N,
        "config_label_for_p": "spline_order (config key name; renamed from spline_degree, v4)",
        "code_convention_for_p": "B-spline ORDER (vendor/cgmy_bspline/solver_cn.py:72, "
                                  "projection.py: 'p: B-spline order')",
        "p_value": order,
        "implied_polynomial_degree": degree,
        "measured_support_width": support_width,
        "support_width_theory_check": (
            "An order-p cardinal B-spline has support width exactly p "
            "(a degree-d polynomial spline spans d+1 = order knot "
            "intervals). Measured width == p confirms p is ORDER, "
            "and degree = p-1."
        ),
        "measured_value_at_support_edges": {
            "N_p(0)": val_left_edge, "N_p(p)": val_right_edge,
            "expected_for_order_ge_2": 0.0,
        },
        "manuscript_label_vs_actual": {
            "manuscript_says": "cubic (p=3)",
            "actual_by_code_and_measurement": f"order {order} = degree {degree} "
                                               f"({'quadratic' if degree == 2 else 'degree ' + str(degree)})",
            "status": "DISCREPANCY" if degree != 3 else "consistent",
        },
        "n_dof_raw": n_dof,
        "n_dof_formula": "N + p - 1 (active_indices span), matches manuscript's stated 66 "
                          "for N=64, p=3 exactly",
        "n_dof_interior_variational": n_interior,
        "n_dof_dirichlet_prescribed": n_dirichlet,
        "n_dof_unexpected": n_unexpected,
        "roadmap_table_cross_check": {
            "roadmap_claims_quadratic_degree2_order3_interior": 62,
            "measured_interior_here": n_interior,
            "match": n_interior == 62,
        },
        "bc_dofs_cross_check_against_production_code": {
            "ok": cross_check_ok,
            "error": cross_check_error,
        },
        "domain": {"x_min": X_MIN, "x_max": X_MAX, "h": H},
    }

    out_dir = REPO_ROOT / "audit"
    with open(out_dir / "basis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    import csv
    with open(out_dir / "basis_supports.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
