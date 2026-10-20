#!/usr/bin/env bash
# Regenerate every gate/milestone number from checkpoints + configs, WITHOUT
# retraining any model. Deterministic verification (Gates 1-2, Milestone 1)
# is re-run directly since it needs no trained model; the four final
# models are re-evaluated from their saved checkpoints
# (scripts/evaluate_checkpoints.py) rather than retrained.
#
# Requires: checkpoints/{mlp,deeponet,projected_fno}_final.pt and
# checkpoints/pod_final.npz already exist (run scripts/train_final_models.py
# once to produce them; that step is the one part of the pipeline this
# script deliberately does NOT redo, since it is the expensive one).
#
# Usage: bash scripts/reproduce.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=================================================================="
echo "1/4  Full test suite (Gates 1-2 and every other correctness check)"
echo "=================================================================="
PYTHONPATH=src python3 -m pytest -q

echo
echo "=================================================================="
echo "2/4  Milestone 1 (deterministic: reference solver -> residual ->"
echo "     eta_mass_indicator -> eta_sharp -> eta_guar; no trained model involved)"
echo "=================================================================="
PYTHONPATH=src python3 scripts/milestone1.py

echo
echo "=================================================================="
echo "3/4  Re-evaluating final model checkpoints (NO retraining)"
echo "=================================================================="
if [ ! -f checkpoints/mlp_final.pt ]; then
  echo "checkpoints/mlp_final.pt not found."
  echo "Run scripts/train_final_models.py --n-train <N> once first (the one"
  echo "step this script does not redo -- it is the expensive one)."
  exit 1
fi
PYTHONPATH=src python3 scripts/evaluate_checkpoints.py

echo
echo "=================================================================="
echo "4/4  Git provenance of this reproduction run"
echo "=================================================================="
echo "current HEAD: $(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: working tree has uncommitted changes -- this reproduction"
  echo "run's own code may not match any single commit SHA."
fi

echo
echo "Done. See checkpoints/*.pt, *.npz for each model's embedded"
echo "git SHA / dataset manifest hash / seed, and compare against the"
echo "val_L2 numbers just recomputed above."
