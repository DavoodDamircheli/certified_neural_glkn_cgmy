"""Checkpoint save/load with full reproducibility metadata (Stage H item 2).

Every checkpoint embeds: the config snapshot (configs/base.yaml's raw
text), the git SHA of HEAD (AND whether the working tree was dirty --
recording a SHA from a dirty tree without flagging it would silently claim
a reproducibility guarantee that does not hold), a hash of the training
dataset's manifest, and the RNG seed used for training.

Task 5 (working prompt): ``config_snapshot`` alone is NOT enough to verify
a checkpoint is compatible with how it is about to be used -- it is a
point-in-time copy of configs/base.yaml's raw text, and this project's own
history has an exact example of it going stale: every checkpoint currently
on disk (trained before commit 72d8bd3's discretization fix) embeds
``config_snapshot`` text saying ``n_elements: 256``/``domain +-10``, while
the weights themselves were actually trained at ``N=64``/``domain +-8``
(independently verifiable from ``model_init_kwargs['psi_dim']`` == 66 ==
64+3-1, and from ``ProjectedFNO``'s own separately-recorded ``mesh`` field).
The weights are fine; the config-snapshot-based provenance was
misleading. ``build_discretization_metadata``/``build_data_metadata``
below record the REALIZED mesh/time-grid/parameter-box actually used, from
the in-memory objects at save time, not by re-reading a config file that
may have since diverged from what was actually used -- and
``check_discretization_compatibility``/``check_data_compatibility`` are the
hard (raise, never warn) checks ``load_checkpoint`` runs before ever
constructing a model from a checkpoint's weights.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import torch

if TYPE_CHECKING:
    from cgmy.reference_solver import Mesh, TimeGrid
    from data.sampling import ParameterBox

REPO_ROOT = Path(__file__).resolve().parents[2]


class CheckpointCompatibilityError(Exception):
    """Raised (never a warning) when a checkpoint's actual training-time
    discretization, parameter box, or payoff convention does not match what
    the caller is about to use it with. A silently mismatched checkpoint
    would produce numerically meaningless -- or, worse, plausible-looking
    but wrong -- predictions (e.g. a mismatched psi_dim would either crash
    deep inside the network or, if it happened to match by coincidence,
    silently feed coefficients indexed against the wrong mesh)."""


def build_discretization_metadata(mesh: "Mesh", time_grid: "TimeGrid") -> dict[str, Any]:
    """The REALIZED discretization actually used for a training/generation
    run, read from the in-memory ``mesh``/``time_grid`` objects -- never
    re-derived from a config file, which may have since changed (see
    module docstring)."""
    return {
        "N": mesh.N, "p": mesh.p,
        "domain_left": mesh.x_min, "domain_right": mesh.x_max,
        "n_time_steps": time_grid.n_steps,
    }


def build_data_metadata(box: "ParameterBox", payoff_strike: float) -> dict[str, Any]:
    """The realized parameter box and payoff strike actually sampled from.
    Determines ``learning.normalization.NormalizationStats`` exactly
    (``stats_from_box`` is a pure function of the box), so checking this is
    equivalent to checking normalization compatibility -- no separate
    normalization-stats field is needed."""
    return {
        "C": list(box.C), "G": list(box.G), "M": list(box.M),
        "Y": list(box.Y), "r": list(box.r), "maturity": list(box.maturity),
        "strike": float(payoff_strike),
    }


def _mismatches(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    bad = {}
    for k, exp_v in expected.items():
        act_v = actual.get(k)
        if isinstance(exp_v, float) or isinstance(act_v, float):
            ok = act_v is not None and abs(float(act_v) - float(exp_v)) < 1e-9
        elif isinstance(exp_v, list):
            ok = act_v is not None and len(act_v) == len(exp_v) and all(
                abs(float(a) - float(e)) < 1e-9 for a, e in zip(act_v, exp_v)
            )
        else:
            ok = act_v == exp_v
        if not ok:
            bad[k] = (act_v, exp_v)
    return bad


def check_discretization_compatibility(
    actual: dict[str, Any] | None, mesh: "Mesh", time_grid: "TimeGrid", *, source: str
) -> None:
    """Raise CheckpointCompatibilityError if ``actual`` (a checkpoint's
    saved discretization block) doesn't match the caller's current
    mesh/time_grid on N, p (basis order), domain, or n_time_steps."""
    expected = build_discretization_metadata(mesh, time_grid)
    if actual is None:
        raise CheckpointCompatibilityError(
            f"{source}: no 'discretization' metadata recorded on this checkpoint "
            f"-- cannot verify N/Nt/domain/basis-order compatibility (a checkpoint "
            f"predating the Task 5 provenance fix?). Patch or regenerate its "
            f"metadata before loading it -- see notes/05_provenance.md."
        )
    bad = _mismatches(actual, expected)
    if bad:
        raise CheckpointCompatibilityError(
            f"{source}: discretization mismatch (checkpoint value, expected value): "
            f"{bad}. This checkpoint was NOT trained on the mesh/time-grid you are "
            f"about to use it with -- do not load it here."
        )


def check_data_compatibility(
    actual: dict[str, Any] | None, box: "ParameterBox", payoff_strike: float, *, source: str
) -> None:
    """Raise CheckpointCompatibilityError if ``actual`` (a checkpoint's
    saved parameter-box block) doesn't match the caller's current box and
    payoff strike -- equivalently, a normalization-scaling and
    payoff-convention check (see build_data_metadata)."""
    expected = build_data_metadata(box, payoff_strike)
    if actual is None:
        raise CheckpointCompatibilityError(
            f"{source}: no 'data_box' metadata recorded on this checkpoint -- "
            f"cannot verify parameter-scaling/payoff-convention compatibility "
            f"(a checkpoint predating the Task 5 provenance fix?). Patch or "
            f"regenerate its metadata before loading it -- see "
            f"notes/05_provenance.md."
        )
    bad = _mismatches(actual, expected)
    if bad:
        raise CheckpointCompatibilityError(
            f"{source}: parameter-box/payoff mismatch (checkpoint value, expected "
            f"value): {bad}. This checkpoint's normalization (learning.normalization."
            f"stats_from_box) and/or payoff convention does not match the caller's "
            f"-- do not load it here."
        )


def get_git_sha_and_dirty(repo_root: Path = REPO_ROOT) -> tuple[str, bool]:
    """Return (HEAD sha, whether the working tree has uncommitted changes).

    A dirty tree means the recorded SHA does NOT fully determine the code
    that produced the checkpoint -- callers should surface git_dirty
    prominently, not just log the SHA and imply full reproducibility.
    """
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root
    ).decode().strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root
    ).decode().strip()
    return sha, bool(status)


def dataset_manifest_hash(dataset_path: str | Path) -> str:
    """sha256 of the dataset file's top-level manifest attributes (sample
    counts, split sizes, seed, config snapshot) -- cheap to compute (does
    not hash the full array data) but changes if the dataset's identity
    (what was sampled, with what config) changes."""
    with h5py.File(dataset_path, "r") as f:
        attrs = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in f.attrs.items()}
    payload = json.dumps(attrs, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_class_name: str,
    model_init_kwargs: dict[str, Any],
    config_snapshot: str,
    seed: int,
    dataset_path: str | Path | None,
    *,
    discretization: dict[str, Any],
    data_box: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a checkpoint with full reproducibility metadata.

    Args:
        discretization: ``build_discretization_metadata(mesh, time_grid)``
            -- required (Task 5, working prompt): the REALIZED mesh/
            time-grid this checkpoint's weights were actually trained
            against, checked at load time, never left implicit in
            ``config_snapshot`` alone (see module docstring for why).
        data_box: ``build_data_metadata(box, payoff_strike)`` -- required,
            same reasoning, for the parameter box/payoff convention.

    Returns:
        The metadata dict that was embedded (for logging/reporting).
    """
    sha, dirty = get_git_sha_and_dirty()
    metadata = {
        "model_class_name": model_class_name,
        "model_init_kwargs": model_init_kwargs,
        "config_snapshot": config_snapshot,
        "discretization": discretization,
        "data_box": data_box,
        "git_sha": sha,
        "git_dirty": dirty,
        "dataset_manifest_hash": dataset_manifest_hash(dataset_path) if dataset_path else None,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "seed": seed,
    }
    if extra:
        metadata.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **metadata}, path)
    return metadata


def load_checkpoint(
    path: str | Path,
    model_ctor,
    *,
    mesh: "Mesh",
    time_grid: "TimeGrid",
    box: "ParameterBox",
    payoff_strike: float,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a checkpoint saved by save_checkpoint, with a HARD compatibility
    check (Task 5, working prompt) run BEFORE the model is ever
    constructed from the checkpoint's weights.

    Args:
        model_ctor: Callable(**model_init_kwargs) -> an unconstructed model
            of the right class (e.g. ``MLPCoeffNet``) -- the caller supplies
            this rather than the checkpoint embedding a class reference, to
            avoid unpickling arbitrary code.
        mesh, time_grid: The caller's CURRENT mesh/time-grid -- checked
            against the checkpoint's own ``discretization`` block.
        box, payoff_strike: The caller's CURRENT parameter box/strike --
            checked against the checkpoint's own ``data_box`` block.

    Returns:
        (model with loaded weights, metadata dict).

    Raises:
        CheckpointCompatibilityError: If the checkpoint's recorded
            discretization or parameter box/payoff does not match the
            caller's current ones, or if either is missing entirely
            (a pre-Task-5 checkpoint). Never a warning.
    """
    payload = torch.load(path, weights_only=False)
    state_dict = payload.pop("model_state_dict")
    check_discretization_compatibility(payload.get("discretization"), mesh, time_grid, source=str(path))
    check_data_compatibility(payload.get("data_box"), box, payoff_strike, source=str(path))
    model = model_ctor(**payload["model_init_kwargs"])
    model.load_state_dict(state_dict)
    return model, payload
