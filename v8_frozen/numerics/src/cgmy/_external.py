"""Makes the trusted external CGMY B-spline solver importable as ``cgmy_bspline``.

Task 6 (working prompt): the reference solver lives in a separate,
independently-versioned repository (not a PyPI package). It is now
VENDORED into this repository at ``vendor/cgmy_bspline/`` (see that
directory's README.md for exactly what was vendored, why vendoring was
chosen over a git submodule or a ``pip install git+...`` dependency, and
its exact provenance) -- a clean clone of this repository alone is
sufficient to import ``cgmy_bspline``; no separate checkout is required by
default. This module puts the vendored copy's parent directory on
``sys.path`` before anything in this package imports it.

Set the ``CGMY_BSPLINE_PATH`` environment variable to point at a
DIFFERENT ``cgmy_bspline`` checkout instead (e.g. a local clone of the
upstream repo, for working against a newer or different version than the
vendored snapshot) -- it should be set to the directory CONTAINING a
``cgmy_bspline/`` package directory (i.e. the repository root you would
otherwise ``pip install -e``), matching how the vendored default is laid
out (``vendor/`` contains ``vendor/cgmy_bspline/``).

Every module that imports ``cgmy_bspline`` must import this module first
(``import cgmy._external  # noqa: F401``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CGMY_BSPLINE_PATH = str(_REPO_ROOT / "vendor")

_path = os.environ.get("CGMY_BSPLINE_PATH", DEFAULT_CGMY_BSPLINE_PATH)
if _path not in sys.path:
    sys.path.insert(0, _path)
