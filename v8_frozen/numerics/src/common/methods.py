"""Numerical-method identifiers and a result-group writer that refuses to mix
methods (roadmap §1, v5 M0).

Every certificate, correction trace and result row produced by v5 code carries
a ``method_id``. Two identifiers exist:

- ``constrained_free_cn_v1``: the integrated free-coordinate path
  (``c = P z + b``, ``M_I = PᵀMP``, ``A_I = PᵀAP``, pure uniform Crank–Nicolson)
  -- ``pipeline.free_workflow``.
- ``legacy_full``: the pre-v5 raw n_dof path (``pipeline.predict_certify_correct``),
  kept selectable for regression comparisons only.

A result group (one directory) holds rows of exactly one method. The method is
pinned by a marker file the first time the group is opened; any later attempt to
open the group under another method, or to write a row whose ``method_id``
differs, raises ``MethodMixingError``. Existing files are appended to only when
their header matches; nothing is ever overwritten.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

METHOD_CONSTRAINED_FREE_CN_V1 = "constrained_free_cn_v1"
METHOD_LEGACY_FULL = "legacy_full"
METHOD_INDEPENDENT_COS_V1 = "independent_cos_v1"

KNOWN_METHODS: dict[str, dict[str, str]] = {
    METHOD_CONSTRAINED_FREE_CN_V1: {
        "coordinate_space": "constrained_free",
        "scheme": "CN_uniform",
        "description": (
            "Integrated free-coordinate predict/certify/correct path: c = P z + b, "
            "M_I = P^T M P, A_I = P^T A P, S_I = (A_I + A_I^T)/2, CN free forcing "
            "f_I = P^T(f_raw - M (b^{n+1}-b^n)/dt - A (b^{n+1}+b^n)/2); acceptance on "
            "eta_aug = sqrt(beta_0^2 + eta_res_upper^2)."
        ),
    },
    METHOD_LEGACY_FULL: {
        "coordinate_space": "legacy_full",
        "scheme": "CN_uniform",
        "description": (
            "Pre-v5 raw n_dof path (pipeline.predict_certify_correct): residual rows "
            "zeroed at Dirichlet dofs, full S_h SPD gate, no initial-error term. "
            "Regression comparisons only."
        ),
    },
    METHOD_INDEPENDENT_COS_V1: {
        "coordinate_space": "not_applicable",
        "scheme": "fourier_cos",
        "description": (
            "Independent Fourier-COS terminal-price method (validation.cos_pricer); no "
            "trajectory, no certificate. Terminal-price workload only; its own result group."
        ),
    },
}

# Stated verbatim in every v5 manifest and trace.
NUMERICAL_SCOPE = (
    "ordinary floating-point evaluation of exact-arithmetic theory: the reliability "
    "theorems hold in exact arithmetic under their stated hypotheses (including "
    "alpha <= lambda_min(S_I)); every reported number is computed in IEEE double "
    "precision without validated/outward rounding. No machine-validated enclosure and "
    "no total continuum-price error bound is claimed."
)

_MARKER = "method_id.json"


class MethodMixingError(ValueError):
    """Raised when rows of different numerical methods would share one result group."""


def validate_method_id(method_id: str) -> str:
    if method_id not in KNOWN_METHODS:
        raise ValueError(
            f"unknown method_id {method_id!r}; expected one of {sorted(KNOWN_METHODS)}"
        )
    return method_id


class ResultGroupWriter:
    """Append-only CSV writer for one result group of one method.

    Args:
        group_dir: Directory of the result group (created if absent).
        method_id: The single method this group may contain.
    """

    def __init__(self, group_dir: str | Path, method_id: str) -> None:
        self.method_id = validate_method_id(method_id)
        self.group_dir = Path(group_dir)
        self.group_dir.mkdir(parents=True, exist_ok=True)
        marker = self.group_dir / _MARKER
        if marker.exists():
            pinned = json.loads(marker.read_text())["method_id"]
            if pinned != method_id:
                raise MethodMixingError(
                    f"result group {self.group_dir} is pinned to method {pinned!r}; "
                    f"refusing to open it for {method_id!r}. Use a separate group."
                )
        else:
            marker.write_text(json.dumps({
                "method_id": method_id,
                "coordinate_space": KNOWN_METHODS[method_id]["coordinate_space"],
            }, indent=2) + "\n")

    def _check_row(self, row: dict[str, Any]) -> None:
        if "method_id" not in row:
            raise MethodMixingError("every result row must carry an explicit method_id")
        if row["method_id"] != self.method_id:
            raise MethodMixingError(
                f"row method_id {row['method_id']!r} != group method {self.method_id!r} "
                f"({self.group_dir})"
            )

    def existing_keys(self, filename: str, key: str) -> set[str]:
        """Values of column ``key`` already written to ``filename`` (for resumable runs)."""
        path = self.group_dir / filename
        if not path.exists():
            return set()
        with open(path, newline="") as f:
            return {r[key] for r in csv.DictReader(f)}

    def append_rows(self, filename: str, rows: Iterable[dict[str, Any]]) -> int:
        """Append rows to ``filename`` inside the group. All rows must share one
        header; an existing file must already have exactly that header and only
        this group's method in its ``method_id`` column."""
        rows = list(rows)
        if not rows:
            return 0
        for row in rows:
            self._check_row(row)
        header = list(rows[0].keys())
        for row in rows[1:]:
            if list(row.keys()) != header:
                raise ValueError(f"inconsistent row keys in {filename}: {list(row.keys())} vs {header}")
        path = self.group_dir / filename
        if path.exists():
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != header:
                    raise ValueError(
                        f"{path} has header {reader.fieldnames}, refusing to append rows with {header}"
                    )
                foreign = {r["method_id"] for r in reader} - {self.method_id}
                if foreign:
                    raise MethodMixingError(f"{path} already contains rows of methods {sorted(foreign)}")
            mode = "a"
        else:
            mode = "w"
        with open(path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if mode == "w":
                writer.writeheader()
            writer.writerows(rows)
        return len(rows)
