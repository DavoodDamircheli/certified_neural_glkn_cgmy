"""Gate G1 demonstration traces for constrained_free_cn_v1 (roadmap §2, M1-M2).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=src:experiments python3 scripts/v5_g1_traces.py [--test-log PATH]

Two queries, each run end-to-end through pipeline.free_workflow under
pipeline.legacy_guard.forbid_legacy_calls (any legacy entry point, raw full S_h
action, or raw M_h^{-1} outside the two labelled raw-projection scopes raises;
every generic certification call logs the operator it received):

  1. Q_bench120_0000 (declared bench population) with the DeepONet checkpoint.
  2. Q_audit_0000 (declared audit population, source=known_failure): legacy
     full-space gate fails, free gate passes -- also with DeepONet.

The legacy full-space gate value for each query is computed OUTSIDE the
guarded region, purely as a labelled diagnostic showing which queries the
legacy gate rejects. Outputs go to a new results/free_integrated_v5/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch  # noqa: E402

from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff  # noqa: E402
from certification.constants import compute_stability_constants  # noqa: E402
from common import io as common_io  # noqa: E402
from common import run_manifest as rm  # noqa: E402
from common.checkpoint import check_data_compatibility, check_discretization_compatibility  # noqa: E402
from common.methods import NUMERICAL_SCOPE, ResultGroupWriter  # noqa: E402
from data.sampling import load_mesh_and_time_grid, load_parameter_box  # noqa: E402
from learning.deeponet_coeff import DeepONetCoeffNet  # noqa: E402
from learning.normalization import stats_from_box  # noqa: E402
from pipeline import free_workflow as fw  # noqa: E402
from pipeline.legacy_guard import forbid_legacy_calls, summarize  # noqa: E402
from pipeline.predict_certify_correct import build_mesh_info  # noqa: E402

QUERIES = (("bench120", "Q_bench120_0000"), ("audit459", "Q_audit_0000"))
CHECKPOINT = REPO_ROOT / "checkpoints" / "deeponet_final.pt"
SPOT_FRACTIONS = (0.8, 0.9, 1.0, 1.1, 1.2)


def load_deeponet(mesh, time_grid, box, strike):
    torch.set_default_dtype(torch.float64)
    payload = torch.load(CHECKPOINT, weights_only=False)
    check_discretization_compatibility(payload.get("discretization"), mesh, time_grid, source=str(CHECKPOINT))
    check_data_compatibility(payload.get("data_box"), box, strike, source=str(CHECKPOINT))
    net = DeepONetCoeffNet(**payload["model_init_kwargs"])
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    return net


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-log", type=Path, default=None)
    args = ap.parse_args(argv)

    resolved = rm.resolve_config()
    wf_cfg = resolved["workflow_config"]
    policy = fw.FreeWorkflowPolicy.from_config(wf_cfg)
    pops = rm.declare_populations()
    mesh, tgrid = load_mesh_and_time_grid(rm.BASE_CONFIG)
    mesh_info = build_mesh_info(mesh)
    box = load_parameter_box(rm.BASE_CONFIG)
    stats = stats_from_box(box)
    np.random.seed(wf_cfg["seeds"]["global"])
    torch.manual_seed(wf_cfg["seeds"]["global"])
    model = load_deeponet(mesh, tgrid, box, box.strike)

    run_id = rm.new_run_id("g1_traces")
    run_dir = rm.create_run_dir(run_id)
    writer = ResultGroupWriter(run_dir / "outcomes", policy_method := wf_cfg["method_id"])
    log_lines = [f"run_id={run_id}", f"method_id={policy_method}", f"scope={NUMERICAL_SCOPE}"]
    arrays: dict[str, np.ndarray] = {}
    summary_rows = []

    for pop, qid in QUERIES:
        q = next(r for r in pops[pop]["rows"] if r["query_id"] == qid)
        mu = CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
        payoff = EuropeanPutPayoff(strike=q["K"])
        time_grid = np.linspace(0.0, mu.T, tgrid.n_steps + 1)
        ops = OperatorCache(mesh, mesh.x_min, mesh.x_max)  # fresh per query

        legacy = compute_stability_constants(mu, ops)  # diagnostic, outside the guard
        legacy_diag = {
            "computed_outside_guarded_workflow": True,
            "space": "legacy_full", "n": ops.n_dof,
            "lambda_min_numerical_S_h": legacy.lambda_min_numerical,
            "legacy_gate_passes": legacy.is_spd,
        }

        predictor = fw.neural_predictor(model, stats, time_grid)
        xs = tuple(math.log(f * q["K"]) for f in SPOT_FRACTIONS)
        with forbid_legacy_calls() as guard_log:
            trace = fw.run_free_workflow(
                mu, payoff, mesh_info, ops, time_grid, predictor, policy,
                query_id=qid, eval_points=xs,
            )
        guard = summarize(guard_log)
        if guard["blocked"]:
            raise SystemExit(f"G1 FAILED: legacy calls blocked during {qid}: {guard['blocked']}")

        tj = trace.to_json_dict()
        tj["population"] = pop
        tj["query_source"] = q["source"]
        tj["model"] = {"name": "DeepONet", "checkpoint": "checkpoints/deeponet_final.pt",
                       "sha256": rm.sha256_file(CHECKPOINT)}
        tj["legacy_full_gate_diagnostic"] = legacy_diag
        tj["legacy_call_guard"] = {"blocked": guard["blocked"], "n_events": guard["n_events"], "calls": guard["calls"]}
        tj["numerical_scope"] = NUMERICAL_SCOPE
        rm.write_json(run_dir / f"trace_{qid}.json", tj)
        if trace.candidate_coefficients is not None:
            arrays[f"{qid}/candidate_projected"] = trace.candidate_coefficients
        if trace.delivered_coefficients is not None:
            arrays[f"{qid}/delivered"] = trace.delivered_coefficients
        writer.append_rows("outcomes.csv", [fw.outcome_row(trace)])
        cc, fc = trace.candidate_certificate, trace.final_certificate
        summary_rows.append((qid, pop, q["source"], legacy.is_spd, trace.gate["is_spd_by_estimate"],
                             cc.initial_error if cc else float("nan"),
                             cc.residual_certificate if cc else float("nan"),
                             cc.augmented_certificate if cc else float("nan"),
                             trace.correction is not None,
                             trace.correction.all_converged if trace.correction else None,
                             fc.augmented_certificate if fc else float("nan"),
                             trace.final_status, trace.output_delivered))
        log_lines.append(f"{qid}: status={trace.final_status} delivered={trace.output_delivered} "
                         f"events={trace.events} guard_blocked={len(guard['blocked'])}")

    common_io.write_arrays(run_dir / "trace_arrays.h5", arrays, attrs={"method_id": policy_method})
    arrays_sha = rm.sha256_file(run_dir / "trace_arrays.h5")
    if args.test_log is not None:
        shutil.copyfile(args.test_log, run_dir / "test_log.txt")

    manifest = rm.build_manifest(
        run_id, method_id=policy_method, studies={"g1_free_path_traces": "completed"},
        seeds=wf_cfg["seeds"], resolved_config=resolved, populations=pops,
        population_used=[f"{p}:{q}" for p, q in QUERIES],
        extra_inputs={"deeponet_checkpoint": "checkpoints/deeponet_final.pt"},
    )
    manifest["trace_arrays_h5_sha256"] = arrays_sha
    rm.write_json(run_dir / "manifest.json", manifest)
    rm.write_json(run_dir / "resolved_config.json", resolved)
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n")
    hdr = ("| query | population | source | legacy full gate | free gate | beta_0 (cand) | eta_res (cand) "
           "| eta_aug (cand) | correction | GMRES all converged | eta_aug (final) | final status | delivered |")
    md = [f"# G1 traces {run_id}", "", f"Method `{policy_method}`; tolerance {policy.tolerance:g}.",
          f"Scope: {NUMERICAL_SCOPE}", "", hdr, "|" + "---|" * 13]
    for r in summary_rows:
        md.append("| " + " | ".join(f"{v:.3e}" if isinstance(v, float) else str(v) for v in r) + " |")
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(run_dir)
    print("\n".join(md))


if __name__ == "__main__":
    main()
