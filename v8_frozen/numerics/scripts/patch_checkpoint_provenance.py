"""Task 5 (working prompt): patch existing checkpoint/dataset provenance
metadata in place, without retraining.

Every checkpoint currently in checkpoints/ was trained before commit
72d8bd3 fixed configs/base.yaml's stale discretization scaffold values --
their embedded config_snapshot text still says n_elements=256/domain+-10,
even though the weights themselves were actually trained at N=64/domain
+-8 (independently verifiable: model_init_kwargs['psi_dim'] == 66 ==
64+3-1, and ProjectedFNO's own separately-recorded 'mesh' field already
said N=64). The weights are fine; only the recorded provenance was stale.

This script adds the explicit, checked 'discretization'/'data_box' blocks
(common.checkpoint.build_discretization_metadata/build_data_metadata) that
common.checkpoint.load_checkpoint / exp_common.load_models now hard-check
at load time, using the values ACTUALLY used to train each model (recovered
from the checkpoint's own dataset_manifest_hash-linked production dataset,
whose per-sample records are self-describing and correct -- see
notes/00_code_map.md/notes/05_provenance.md) -- never invented.

Run once: PYTHONPATH=src python3 scripts/patch_checkpoint_provenance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from common.checkpoint import build_data_metadata, build_discretization_metadata
from data.sampling import load_parameter_box

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DATASET_PATH = REPO_ROOT / "data" / "production_dataset.h5"


class _Mesh:
    def __init__(self, N, p, x_min, x_max):
        self.N, self.p, self.x_min, self.x_max = N, p, x_min, x_max


class _TimeGrid:
    def __init__(self, n_steps):
        self.n_steps = n_steps


def _realized_mesh_and_time_grid_from_dataset(dataset_path: Path) -> tuple[_Mesh, _TimeGrid]:
    """The REALIZED (N, p, domain, n_steps) actually used, read from sample
    0's own self-describing attrs -- not from the dataset's (possibly
    stale) top-level config_snapshot. See notes/00_code_map.md section 5."""
    with h5py.File(dataset_path, "r") as f:
        s0 = f["samples/000000"]
        mesh = _Mesh(N=int(s0.attrs["mesh_N"]), p=int(s0.attrs["mesh_p"]),
                     x_min=float(s0.attrs["mesh_x_min"]), x_max=float(s0.attrs["mesh_x_max"]))
        time_grid = _TimeGrid(n_steps=int(s0.attrs["n_steps"]))
        strike = float(s0.attrs["strike"])
    return mesh, time_grid, strike


def main() -> None:
    mesh, time_grid, strike = _realized_mesh_and_time_grid_from_dataset(DATASET_PATH)
    print(f"realized discretization (from data/production_dataset.h5 sample 0): "
          f"N={mesh.N} p={mesh.p} domain=[{mesh.x_min:.6f},{mesh.x_max:.6f}] n_steps={time_grid.n_steps}")

    box = load_parameter_box(CONFIG_PATH)
    discretization = build_discretization_metadata(mesh, time_grid)
    data_box = build_data_metadata(box, strike)
    print(f"discretization block to embed: {discretization}")
    print(f"data_box block to embed: {data_box}\n")

    for name in ("mlp_final.pt", "deeponet_final.pt", "projected_fno_final.pt"):
        path = CHECKPOINTS_DIR / name
        payload = torch.load(path, weights_only=False)
        if "discretization" in payload and "data_box" in payload:
            print(f"{name}: already has discretization/data_box, skipping")
            continue
        payload["discretization"] = discretization
        payload["data_box"] = data_box
        torch.save(payload, path)
        print(f"{name}: patched (discretization + data_box added)")

    pod_path = CHECKPOINTS_DIR / "pod_final.npz"
    pod_data = dict(np.load(pod_path))
    if "discretization_json" in pod_data and "data_box_json" in pod_data:
        print("pod_final.npz: already has discretization_json/data_box_json, skipping")
    else:
        pod_data["discretization_json"] = np.array(json.dumps(discretization))
        pod_data["data_box_json"] = np.array(json.dumps(data_box))
        np.savez(pod_path, **pod_data)
        print("pod_final.npz: patched (discretization_json + data_box_json added)")

    # Dataset snapshot: git_sha/git_dirty were never recorded at generation
    # time (write_manifest only started doing so as part of this same
    # task) -- record that honestly rather than guessing a commit. The
    # per-sample mesh_N/mesh_h/etc. attrs (already correct, self-describing
    # -- see notes/00_code_map.md) are untouched.
    with h5py.File(DATASET_PATH, "r+") as f:
        if "git_sha" in f.attrs:
            print(f"\n{DATASET_PATH.name}: already has git_sha, skipping")
        else:
            f.attrs["git_sha"] = "unknown (predates git_sha tracking in write_manifest, Task 5)"
            f.attrs["git_dirty"] = True  # unknown; True is the conservative (never-overclaim) default
            print(f"\n{DATASET_PATH.name}: patched (git_sha/git_dirty added, marked unknown honestly)")

    print("\ndone. Re-run pytest / an experiment script's load_models() call to verify.")


if __name__ == "__main__":
    main()
