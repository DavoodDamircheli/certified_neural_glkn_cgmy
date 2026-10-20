"""Stage H item 3 support: re-evaluate the four final-model checkpoints
(checkpoints/*.pt, *.npz) WITHOUT retraining, printing the same summary
table train_final_models.py produced plus each checkpoint's embedded
reproducibility metadata (git SHA + dirty flag, dataset manifest hash,
seed). scripts/reproduce.sh calls this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cgmy.operators import OperatorCache
from data.sampling import load_parameter_box
from learning.deeponet_coeff import DeepONetCoeffNet
from learning.fno import FNO1d
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import stats_from_box
from learning.pod import PODBasis, PODGalerkinROM
from learning.projected_fno import ProjectedFNO
from generate_production_dataset import FIXED_MESH, build_val_set
from train_final_models import evaluate_model

torch.set_default_dtype(torch.float64)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


def _print_meta(name: str, meta: dict) -> None:
    print(f"  git_sha={meta.get('git_sha')}  git_dirty={meta.get('git_dirty')}  "
          f"seed={meta.get('seed')}  n_train={meta.get('n_train')}  "
          f"dataset_manifest_hash={str(meta.get('dataset_manifest_hash'))[:12]}...")


def main():
    box = load_parameter_box(CONFIG_PATH)
    stats = stats_from_box(box)
    ops = OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)

    print("building the (fixed-seed, reproducible) validation set...")
    val_trajs = build_val_set(30)

    print("\n" + "=" * 70)
    print("RE-EVALUATING FINAL CHECKPOINTS (no retraining)")
    print("=" * 70)

    results = []

    mlp_path = CHECKPOINTS_DIR / "mlp_final.pt"
    if mlp_path.exists():
        payload = torch.load(mlp_path, weights_only=False)
        net = MLPCoeffNet(**payload["model_init_kwargs"])
        net.load_state_dict(payload["model_state_dict"])
        err = evaluate_model(net, val_trajs, ops, stats)
        print(f"\nMLPCoeffNet: recomputed val_L2={err:.4e} "
              f"(checkpoint recorded {payload.get('val_l2_error'):.4e})")
        _print_meta("MLPCoeffNet", payload)
        results.append(("MLPCoeffNet", err))
    else:
        print(f"\n[skip] {mlp_path} not found -- run scripts/train_final_models.py first")

    deeponet_path = CHECKPOINTS_DIR / "deeponet_final.pt"
    if deeponet_path.exists():
        payload = torch.load(deeponet_path, weights_only=False)
        net = DeepONetCoeffNet(**payload["model_init_kwargs"])
        net.load_state_dict(payload["model_state_dict"])
        err = evaluate_model(net, val_trajs, ops, stats)
        print(f"\nDeepONetCoeffNet: recomputed val_L2={err:.4e} "
              f"(checkpoint recorded {payload.get('val_l2_error'):.4e})")
        _print_meta("DeepONetCoeffNet", payload)
        results.append(("DeepONetCoeffNet", err))
    else:
        print(f"\n[skip] {deeponet_path} not found")

    pod_path = CHECKPOINTS_DIR / "pod_final.npz"
    if pod_path.exists():
        data = np.load(pod_path)
        basis = PODBasis(modes=data["modes"], singular_values=data["singular_values"], mean=data["mean"])
        rom = PODGalerkinROM(basis, ops, stats=stats)
        err = evaluate_model(rom, val_trajs, ops, stats)
        print(f"\nPODGalerkinROM: recomputed val_L2={err:.4e} "
              f"(checkpoint recorded {float(data['val_l2_error']):.4e})")
        print(f"  git_sha={str(data['git_sha'])}  git_dirty={bool(data['git_dirty'])}  "
              f"seed={int(data['seed'])}  n_train={int(data['n_train'])}  n_modes={int(data['n_modes'])}")
        results.append(("PODGalerkinROM", err))
    else:
        print(f"\n[skip] {pod_path} not found")

    fno_path = CHECKPOINTS_DIR / "projected_fno_final.pt"
    if fno_path.exists():
        payload = torch.load(fno_path, weights_only=False)
        fno = FNO1d(**payload["fno_init_kwargs"])
        fno.load_state_dict(payload["fno_state_dict"])
        m = payload["mesh"]
        from cgmy.reference_solver import Mesh
        mesh = Mesh(N=m["N"], p=m["p"], x_min=m["x_min"], x_max=m["x_max"])
        net = ProjectedFNO(fno, mesh, ops, mu_dim=payload["mu_dim"])
        err = evaluate_model(net, val_trajs, ops, stats)
        print(f"\nProjectedFNO: recomputed val_L2={err:.4e} "
              f"(checkpoint recorded {payload.get('val_l2_error'):.4e})")
        _print_meta("ProjectedFNO", payload)
        results.append(("ProjectedFNO", err))
    else:
        print(f"\n[skip] {fno_path} not found")

    print("\n" + "=" * 70)
    print("REPRODUCED SUMMARY")
    print("=" * 70)
    for name, err in results:
        print(f"{name:20s}  val_L2={err:.4e}")


if __name__ == "__main__":
    main()
