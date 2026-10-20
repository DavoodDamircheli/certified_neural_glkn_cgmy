"""v5 M0: method identifiers, result-group mixing guard, populations, manifest."""

from __future__ import annotations

import json

import numpy as np
import pytest

from common import run_manifest as rm
from common.methods import (
    METHOD_CONSTRAINED_FREE_CN_V1,
    METHOD_LEGACY_FULL,
    MethodMixingError,
    ResultGroupWriter,
    validate_method_id,
)


def _row(method: str, qid: str = "q0") -> dict:
    return {"query_id": qid, "method_id": method, "value": 1.0}


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        validate_method_id("free_v1")


def test_writer_refuses_foreign_row(tmp_path):
    w = ResultGroupWriter(tmp_path / "g", METHOD_CONSTRAINED_FREE_CN_V1)
    w.append_rows("outcomes.csv", [_row(METHOD_CONSTRAINED_FREE_CN_V1)])
    with pytest.raises(MethodMixingError):
        w.append_rows("outcomes.csv", [_row(METHOD_LEGACY_FULL, "q1")])
    with pytest.raises(MethodMixingError):
        w.append_rows("outcomes.csv", [{"query_id": "q2", "value": 2.0}])


def test_group_pinned_to_first_method(tmp_path):
    ResultGroupWriter(tmp_path / "g", METHOD_LEGACY_FULL)
    with pytest.raises(MethodMixingError):
        ResultGroupWriter(tmp_path / "g", METHOD_CONSTRAINED_FREE_CN_V1)


def test_writer_resume_keys_and_header_guard(tmp_path):
    w = ResultGroupWriter(tmp_path / "g", METHOD_CONSTRAINED_FREE_CN_V1)
    w.append_rows("o.csv", [_row(METHOD_CONSTRAINED_FREE_CN_V1, "a"), _row(METHOD_CONSTRAINED_FREE_CN_V1, "b")])
    assert w.existing_keys("o.csv", "query_id") == {"a", "b"}
    with pytest.raises(ValueError):
        w.append_rows("o.csv", [{"method_id": METHOD_CONSTRAINED_FREE_CN_V1, "other": 1}])


def test_declared_populations_match_frozen_hashes():
    pops = rm.declare_populations()
    assert set(pops) == {"audit459", "bench120", "pricing_q_ref30", "validation30", "scale30", "fourier_old27"}
    assert len(pops["audit459"]["rows"]) == 459
    assert len(pops["bench120"]["rows"]) == 120
    assert len(pops["validation30"]["rows"]) == 30


def test_population_hash_mismatch_is_rejected(tmp_path):
    decl = json.loads(rm.POPULATIONS_DECLARATION.read_text())
    decl["populations"]["bench120"]["sha256"] = "0" * 64
    p = tmp_path / "decl.json"
    p.write_text(json.dumps(decl))
    with pytest.raises(rm.PopulationMismatchError):
        rm.declare_populations(p)


def test_validation30_matches_build_val_set_draw():
    """Same Sobol call as scripts/generate_production_dataset.build_val_set."""
    import warnings

    from data.sampling import load_parameter_box, sample_sobol

    box = load_parameter_box(rm.BASE_CONFIG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sample_sobol(box, 30, seed=999_999)
    rows = rm.generate_validation30()
    for r, (mu, _) in zip(rows, ref):
        assert (r["C"], r["G"], r["M"], r["Y"], r["r"], r["T"]) == (mu.C, mu.G, mu.M, mu.Y, mu.r, mu.T)


def test_array_hash_sensitive_to_one_ulp():
    a = np.linspace(0.0, 1.0, 11)
    b = a.copy()
    b[3] = np.nextafter(b[3], 2.0)
    assert rm.sha256_array(a) != rm.sha256_array(b)
    assert rm.sha256_array(a) == rm.sha256_array(a.copy())


def test_manifest_records_required_fields():
    resolved = rm.resolve_config()
    pops = rm.declare_populations()
    m = rm.build_manifest(
        "test_run", method_id=METHOD_CONSTRAINED_FREE_CN_V1, studies={"g0_dry_run": "completed"},
        seeds={"global": 0}, resolved_config=resolved, populations=pops, population_used=[],
    )
    d = m["discretization"]
    assert (d["N"], d["spline_order_p"], d["n_dof"], d["n_free"]) == (64, 3, 66, 62)
    assert d["bc_left"] == [0, 1] and d["bc_right"] == [64, 65]
    assert m["studies"]["bench120_cold"] == "not run"
    assert m["studies"]["g0_dry_run"] == "completed"
    assert "floating-point" in m["numerical_scope"]
    for key in ("git", "source_tree_sha256", "checkpoints_sha256", "software", "hardware",
                "threads", "solver_policy", "certificate_policy", "seeds"):
        assert key in m
    assert set(m["checkpoints_sha256"]) >= {"mlp_final.pt", "deeponet_final.pt", "pod_final.npz"}


def test_run_dir_never_overwritten(tmp_path):
    rm.create_run_dir("x", root=tmp_path)
    with pytest.raises(FileExistsError):
        rm.create_run_dir("x", root=tmp_path)
