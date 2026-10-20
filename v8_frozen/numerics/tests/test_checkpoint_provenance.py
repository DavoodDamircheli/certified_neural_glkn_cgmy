"""Task 5 (working prompt): save_checkpoint/load_checkpoint must embed and
HARD-check (raise, never warn) discretization/parameter-box provenance --
config_snapshot alone is not enough (see common.checkpoint's module
docstring: every checkpoint in this repo already demonstrated why, its
config_snapshot text going stale relative to the mesh actually used).
"""

from __future__ import annotations

import torch

import pytest

from cgmy.reference_solver import Mesh, TimeGrid
from common.checkpoint import (
    CheckpointCompatibilityError,
    build_data_metadata,
    build_discretization_metadata,
    check_data_compatibility,
    check_discretization_compatibility,
    load_checkpoint,
    save_checkpoint,
)
from data.sampling import ParameterBox

MESH = Mesh(N=64, p=3, x_min=-3.4, x_max=12.6)
TIME_GRID = TimeGrid(n_steps=60)
BOX = ParameterBox(C=(0.05, 1.0), G=(1.0, 5.0), M=(2.0, 9.0), Y=(1.1, 1.8), r=(0.01, 0.1), maturity=(0.1, 1.0), strike=100.0)


class _TinyModel(torch.nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)


def test_build_discretization_metadata_matches_mesh_fields():
    d = build_discretization_metadata(MESH, TIME_GRID)
    assert d == {"N": 64, "p": 3, "domain_left": -3.4, "domain_right": 12.6, "n_time_steps": 60}


def test_build_data_metadata_matches_box_fields():
    d = build_data_metadata(BOX, 100.0)
    assert d["C"] == [0.05, 1.0] and d["M"] == [2.0, 9.0] and d["strike"] == 100.0


def test_check_discretization_compatibility_passes_on_exact_match():
    check_discretization_compatibility(build_discretization_metadata(MESH, TIME_GRID), MESH, TIME_GRID, source="t")


def test_check_discretization_compatibility_raises_on_missing_block():
    with pytest.raises(CheckpointCompatibilityError, match="no 'discretization'"):
        check_discretization_compatibility(None, MESH, TIME_GRID, source="t")


@pytest.mark.parametrize("field,bad_value", [("N", 256), ("p", 2), ("domain_left", -5.4), ("n_time_steps", 200)])
def test_check_discretization_compatibility_raises_on_each_mismatched_field(field, bad_value):
    actual = build_discretization_metadata(MESH, TIME_GRID)
    actual[field] = bad_value
    with pytest.raises(CheckpointCompatibilityError, match="discretization mismatch"):
        check_discretization_compatibility(actual, MESH, TIME_GRID, source="t")


def test_check_data_compatibility_passes_on_exact_match():
    check_data_compatibility(build_data_metadata(BOX, 100.0), BOX, 100.0, source="t")


def test_check_data_compatibility_raises_on_missing_block():
    with pytest.raises(CheckpointCompatibilityError, match="no 'data_box'"):
        check_data_compatibility(None, BOX, 100.0, source="t")


def test_check_data_compatibility_raises_on_strike_mismatch():
    actual = build_data_metadata(BOX, 100.0)
    actual["strike"] = 110.0
    with pytest.raises(CheckpointCompatibilityError, match="parameter-box/payoff mismatch"):
        check_data_compatibility(actual, BOX, 100.0, source="t")


def test_check_data_compatibility_raises_on_box_bound_mismatch():
    actual = build_data_metadata(BOX, 100.0)
    actual["M"] = [2.0, 20.0]  # the pre-Stage-D-narrowing bound
    with pytest.raises(CheckpointCompatibilityError, match="parameter-box/payoff mismatch"):
        check_data_compatibility(actual, BOX, 100.0, source="t")


def test_save_and_load_checkpoint_round_trip(tmp_path):
    model = _TinyModel()
    path = tmp_path / "tiny.pt"
    save_checkpoint(
        path, model, "_TinyModel", {"dim": 4}, config_snapshot="dummy", seed=0, dataset_path=None,
        discretization=build_discretization_metadata(MESH, TIME_GRID),
        data_box=build_data_metadata(BOX, 100.0),
    )
    loaded, meta = load_checkpoint(path, _TinyModel, mesh=MESH, time_grid=TIME_GRID, box=BOX, payoff_strike=100.0)
    assert isinstance(loaded, _TinyModel)
    assert meta["discretization"]["N"] == 64


def test_load_checkpoint_raises_on_mismatched_mesh(tmp_path):
    """The exact scenario Task 5 exists to prevent: loading a checkpoint
    trained at one discretization as if it matched a different one."""
    model = _TinyModel()
    path = tmp_path / "tiny.pt"
    save_checkpoint(
        path, model, "_TinyModel", {"dim": 4}, config_snapshot="dummy", seed=0, dataset_path=None,
        discretization=build_discretization_metadata(MESH, TIME_GRID),
        data_box=build_data_metadata(BOX, 100.0),
    )
    wrong_mesh = Mesh(N=256, p=3, x_min=-5.4, x_max=14.6)  # the old scaffold discretization
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint(path, _TinyModel, mesh=wrong_mesh, time_grid=TIME_GRID, box=BOX, payoff_strike=100.0)


def test_load_checkpoint_raises_on_a_pre_task5_checkpoint_missing_metadata(tmp_path):
    """A checkpoint saved before this fix (no discretization/data_box at
    all) must fail loudly, not be silently accepted."""
    path = tmp_path / "old_style.pt"
    torch.save({"model_state_dict": _TinyModel().state_dict(), "model_init_kwargs": {"dim": 4}}, path)
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint(path, _TinyModel, mesh=MESH, time_grid=TIME_GRID, box=BOX, payoff_strike=100.0)
