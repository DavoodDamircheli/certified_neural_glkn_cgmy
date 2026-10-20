"""Task 6 (working prompt): a clean clone of this repository alone must be
able to import cgmy_bspline, with no CGMY_BSPLINE_PATH set and no
pre-existing external checkout on the machine -- i.e. it must resolve to
the vendored copy at vendor/cgmy_bspline/, not a machine-specific path.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_cgmy_bspline_path_points_at_the_vendored_copy(monkeypatch):
    monkeypatch.delenv("CGMY_BSPLINE_PATH", raising=False)
    import cgmy._external

    importlib.reload(cgmy._external)
    assert cgmy._external.DEFAULT_CGMY_BSPLINE_PATH == str(REPO_ROOT / "vendor")


def test_vendored_cgmy_bspline_directory_has_every_needed_module():
    vendored = REPO_ROOT / "vendor" / "cgmy_bspline"
    required = {
        "__init__.py", "boundary.py", "bspline_basis.py", "connection_coefficients.py",
        "greeks.py", "grid.py", "matrices.py", "parameters.py", "projection.py",
        "solver_cn.py", "toeplitz.py",
    }
    present = {p.name for p in vendored.glob("*.py")}
    missing = required - present
    assert not missing, f"vendored cgmy_bspline is missing: {missing}"


def test_cgmy_bspline_actually_imports_from_the_vendored_directory():
    import cgmy_bspline

    resolved = Path(cgmy_bspline.__file__).resolve()
    assert resolved == (REPO_ROOT / "vendor" / "cgmy_bspline" / "__init__.py").resolve(), (
        f"cgmy_bspline resolved to {resolved}, not the vendored copy -- "
        f"CGMY_BSPLINE_PATH={os.environ.get('CGMY_BSPLINE_PATH')!r} may be overriding it"
    )
