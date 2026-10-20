"""v5 M4 (roadmap §4.2): v5 code must not name floating-point diagnostics
"certified"; the pre-v5 alpha_certified name survives only as an alias."""

from __future__ import annotations

import re

import numpy as np

from common.run_manifest import REPO_ROOT

V5_FILES = [
    "src/pipeline/free_workflow.py", "src/pipeline/legacy_guard.py", "src/certification/free_problem.py",
    "src/common/methods.py", "src/common/run_manifest.py", "scripts/v5_make_run.py",
    "scripts/v5_g1_traces.py", "scripts/v5_pod_study.py", "scripts/v5_pod_summary.py",
    "scripts/v5_kappa_investigation.py",
]
# identifiers (containing "_") or quoted names containing "certified"; prose
# such as "the trajectory is certified" (a certificate was computed) is allowed.
_IDENT = re.compile(r"[A-Za-z]*_[A-Za-z_]*certified[A-Za-z_]*|[A-Za-z_]*certified_[A-Za-z_]*")
_QUOTED = re.compile(r"[\"'][A-Za-z_]*certified[A-Za-z_]*[\"']")


def test_v5_files_have_no_certified_identifiers():
    hits = []
    for rel in V5_FILES:
        text = (REPO_ROOT / rel).read_text()
        for m in list(_IDENT.finditer(text)) + list(_QUOTED.finditer(text)):
            tok = m.group(0).strip("\"'")
            if tok.endswith("not_certified") or tok.startswith("uncertified"):
                continue
            hits.append(f"{rel}: {tok}")
    assert hits == []


def test_alpha_certified_is_only_an_alias():
    from cgmy.operators import OperatorCache
    from cgmy.reference_solver import CGMYParameters, Mesh
    from certification.constants import StabilityConstants, compute_stability_constants

    assert "alpha_certified" not in StabilityConstants.__dataclass_fields__
    mesh = Mesh(N=16, p=3, x_min=-3.394829814011908, x_max=12.605170185988092)
    c = compute_stability_constants(CGMYParameters(C=0.3, G=3.0, M=5.0, Y=1.4, r=0.05, T=0.5),
                                    OperatorCache(mesh, mesh.x_min, mesh.x_max))
    assert np.isnan(c.alpha_lower_estimate) and np.isnan(c.alpha_certified) or c.alpha_certified == c.alpha_lower_estimate
