"""M6 integrated certificate campaign, part 1 (roadmap §6.1): the 459-query audit.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:experiments:scripts \
        python3 scripts/v5_m6_audit.py [--resume <run_dir>]

Stages (each resumable per query; result groups never mix methods):
  A  positivity audit on all 459 stored Q_audit coordinates
       audit_free/positivity.csv           (constrained_free_cn_v1: S_I, n=62)
       audit_legacy_full/positivity.csv    (legacy_full: S_h, n=66) -- comparator only
     source strata reported separately: sobol_main_box (256), exact_vertex (64),
     known_failure (139). The 139 are DELIBERATELY SELECTED failure cases: their
     legacy failure fraction is NOT a production failure rate.
  B  fixed-residual sharp comparison where BOTH gates pass: a seeded free residual
     r_I (per query) and its injection P r_I; q_free = r_I^T S_I^{-1} r_I in
     audit_free/sharp_fixed_residual.csv, q_full = (P r_I)^T S_h^{-1} (P r_I) in
     audit_legacy_full/sharp_fixed_residual.csv. Exact arithmetic: q_free <= q_full
     (S_I^{-1} <= (S_h^{-1})_II, Schur complement) whenever S_h is SPD.
  C  integrated free-workflow traces on a declared subset (first 16 rows of each
     stratum, in file order), DeepONet and mass-weighted POD r48; includes the
     legacy-only rejections (legacy gate fails, free gate passes).
       integrated/traces.csv
  D  augmented-bound validation on the same subset, separate from the Riesz study:
     candidates (i) exact free march from z_h^0 + delta (homogeneous error equation,
     ||delta||_{M_I} = 1e-4), (ii) raw DeepONet candidate, (iii) POD r16 candidate;
     true error vs the dense direct free solution, compared with eta_aug (upper,
     max PCG cap) and the direct solution's own allowance.
       augmented/augmented_bound_validation.csv
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "experiments", "scripts"):
    sys.path.insert(0, str(REPO_ROOT / sub))

import torch  # noqa: E402

from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff  # noqa: E402
from certification.constants import compute_stability_constants, dense_from_apply, kappa_operator_norm  # noqa: E402
from certification.constrained import (  # noqa: E402
    FreeOperatorCache, free_space_for_mesh, inject, march_free_cn_direct,
)
from certification.free_problem import (  # noqa: E402
    build_free_problem, free_eta_sharp, free_residuals, free_stability_constants, residual_rounding_eta,
)
from common import run_manifest as rm  # noqa: E402
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1, METHOD_LEGACY_FULL, NUMERICAL_SCOPE, ResultGroupWriter  # noqa: E402
from data.sampling import load_mesh_and_time_grid, load_parameter_box  # noqa: E402
from learning import pod_free as pf  # noqa: E402
from learning.normalization import stats_from_box  # noqa: E402
from pipeline import free_workflow as fw  # noqa: E402
from pipeline.predict_certify_correct import build_mesh_info  # noqa: E402

SUBSET_PER_STRATUM = 16
EPS = np.finfo(float).eps


def mu_of(q):
    return CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])


def positivity_row(q, ops, label, method_id):
    row = dict(query_id=q["query_id"], method_id=method_id, source=q["source"], space=label, n=ops.n_dof, exception="")
    try:
        c = compute_stability_constants(mu_of(q), ops, coordinate_space=label)
        row.update(lambda_min_numerical=c.lambda_min_numerical, alpha_lower_estimate=c.alpha_lower_estimate,
                   alpha_minus_lambda_min=(c.alpha_lower_estimate - c.lambda_min_numerical) if c.is_spd else float("nan"),
                   gate_passes=c.is_spd, coercivity_strategy=c.coercivity_strategy,
                   lambda_max_S=c.continuity_constant, cond2_S=(c.continuity_constant / c.lambda_min_numerical
                                                                if c.lambda_min_numerical > 0 else float("nan")),
                   lambda_min_M=c.lambda_min_M_h, lambda_max_M=c.lambda_max_M_h,
                   kappa=kappa_operator_norm(mu_of(q), ops) if c.is_spd else float("nan"))
    except Exception as exc:
        row.update(lambda_min_numerical=float("nan"), alpha_lower_estimate=float("nan"), alpha_minus_lambda_min=float("nan"),
                   gate_passes=False, coercivity_strategy="exception", lambda_max_S=float("nan"), cond2_S=float("nan"),
                   lambda_min_M=float("nan"), lambda_max_M=float("nan"), kappa=float("nan"),
                   exception=f"{type(exc).__name__}: {exc}")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    resolved = rm.resolve_config()
    wf = resolved["workflow_config"]
    policy = fw.FreeWorkflowPolicy.from_config(wf)
    pops = rm.declare_populations()
    audit = pops["audit459"]["rows"]
    mesh, tg = load_mesh_and_time_grid(rm.BASE_CONFIG)
    mi = build_mesh_info(mesh)
    ops = OperatorCache(mesh, mesh.x_min, mesh.x_max)
    fs = free_space_for_mesh(mi)
    fo = FreeOperatorCache(ops, fs)

    if args.resume:
        run_dir = args.resume
    else:
        run_dir = rm.create_run_dir(rm.new_run_id("m6_audit459"))
    log = open(run_dir / "run.log", "a")

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"session_start run_dir={run_dir.name} git_head_at_start={rm.git_state()['head']} scope={NUMERICAL_SCOPE}")
    wfree = ResultGroupWriter(run_dir / "audit_free", METHOD_CONSTRAINED_FREE_CN_V1)
    wfull = ResultGroupWriter(run_dir / "audit_legacy_full", METHOD_LEGACY_FULL)

    # ---- A: positivity
    done = wfree.existing_keys("positivity.csv", "query_id") & wfull.existing_keys("positivity.csv", "query_id")
    for i, q in enumerate(audit):
        if q["query_id"] in done:
            continue
        ops_q = OperatorCache(mesh, mesh.x_min, mesh.x_max)  # fresh mu-level caches per query
        wfull.append_rows("positivity.csv", [positivity_row(q, ops_q, "legacy_full", METHOD_LEGACY_FULL)])
        wfree.append_rows("positivity.csv", [positivity_row(q, FreeOperatorCache(ops_q, fs), "constrained_free",
                                                            METHOD_CONSTRAINED_FREE_CN_V1)])
        if i % 50 == 0:
            say(f"A positivity {i}/{len(audit)}")
    say("A done")

    # ---- B: fixed-residual sharp comparison (both gates pass)
    import csv as _csv

    def rows(path):
        with open(path, newline="") as f:
            return {r["query_id"]: r for r in _csv.DictReader(f)}

    pfree, pfull = rows(run_dir / "audit_free/positivity.csv"), rows(run_dir / "audit_legacy_full/positivity.csv")
    done = wfree.existing_keys("sharp_fixed_residual.csv", "query_id")
    for i, q in enumerate(audit):
        qid = q["query_id"]
        if qid in done or pfree[qid]["gate_passes"] != "True" or pfull[qid]["gate_passes"] != "True":
            continue
        rng = np.random.default_rng([20261007, i])
        r_I = rng.standard_normal(fs.n_free)
        ops_q = OperatorCache(mesh, mesh.x_min, mesh.x_max)
        mu = mu_of(q)
        S_I = dense_from_apply(lambda v: FreeOperatorCache(ops_q, fs).apply_S(mu, v), fs.n_free)
        S = dense_from_apply(lambda v: ops_q.apply_S(mu, v), mi.n_dof)
        S_I, S = 0.5 * (S_I + S_I.T), 0.5 * (S + S.T)
        q_free = float(r_I @ np.linalg.solve(S_I, r_I))
        r_full = inject(fs, r_I)
        q_full = float(r_full @ np.linalg.solve(S, r_full))
        # rounding allowance for comparing the two quadratic forms: each solve has
        # relative forward error <= ~n*eps*cond; allowance 4*n*eps*(cond_I*q_free + cond*q_full)
        allow = 4 * mi.n_dof * EPS * (np.linalg.cond(S_I) * q_free + np.linalg.cond(S) * q_full)
        wfree.append_rows("sharp_fixed_residual.csv", [dict(
            query_id=qid, method_id=METHOD_CONSTRAINED_FREE_CN_V1, source=q["source"], residual_seed=f"[20261007, {i}]",
            q_free=q_free, q_full_comparator=q_full, ratio_free_over_full=q_free / q_full,
            rounding_allowance=allow, theorem_holds_within_rounding=q_free <= q_full + allow)])
        wfull.append_rows("sharp_fixed_residual.csv", [dict(
            query_id=qid, method_id=METHOD_LEGACY_FULL, source=q["source"], residual_seed=f"[20261007, {i}]",
            q_full=q_full)])
    say("B done")

    # ---- subset for C and D
    subset = []
    for src in ("sobol_main_box", "exact_vertex", "known_failure"):
        subset += [q for q in audit if q["source"] == src][:SUBSET_PER_STRATUM]
    (run_dir / "declared_subset.txt").write_text(
        f"first {SUBSET_PER_STRATUM} rows of each Q_audit stratum in file order\n"
        + "\n".join(q["query_id"] for q in subset) + "\n")

    from v5_g1_traces import load_deeponet
    box = load_parameter_box(rm.BASE_CONFIG)
    model = load_deeponet(mesh, tg, box, box.strike)
    stats = stats_from_box(box)
    art = REPO_ROOT / wf["pod"]["artifact"]
    pod48, _ = pf.load_pod_artifact(art, fo, rank=48)
    pod16, _ = pf.load_pod_artifact(art, fo, rank=16)

    # ---- C: integrated traces
    wtr = ResultGroupWriter(run_dir / "integrated", METHOD_CONSTRAINED_FREE_CN_V1)
    done = {(r) for r in wtr.existing_keys("traces.csv", "trace_key")}
    for q in subset:
        mu = mu_of(q)
        tgq = np.linspace(0.0, mu.T, tg.n_steps + 1)
        for name, pred in (("DeepONet", fw.neural_predictor(model, stats, tgq)), ("pod_r48", pf.free_pod_predictor(pod48))):
            key = f"{q['query_id']}|{name}"
            if key in done:
                continue
            ops_q = OperatorCache(mesh, mesh.x_min, mesh.x_max)
            try:
                tr = fw.run_free_workflow(mu, EuropeanPutPayoff(q["K"]), mi, ops_q, tgq, pred, policy,
                                          query_id=q["query_id"], keep_coefficients=False)
                row = fw.outcome_row(tr)
                row.update(exception="")
            except Exception as exc:
                row = dict(query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1, final_status="exception",
                           exception=f"{type(exc).__name__}: {exc}")
            wtr.append_rows("traces.csv", [dict(
                trace_key=key, query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1, source=q["source"],
                predictor=name, legacy_full_gate_passes=pfull[q["query_id"]]["gate_passes"],
                free_gate_passes=pfree[q["query_id"]]["gate_passes"],
                legacy_only_rejection=(pfull[q["query_id"]]["gate_passes"] != "True"
                                       and pfree[q["query_id"]]["gate_passes"] == "True"),
                final_status=row.get("final_status"), output_delivered=row.get("output_delivered", False),
                candidate_initial_error=row.get("candidate_initial_error", ""),
                candidate_augmented_certificate=row.get("candidate_augmented_certificate", ""),
                correction_executed=row.get("correction_executed", ""),
                correction_all_converged=row.get("correction_all_converged", ""),
                fallback_executed=row.get("fallback_executed", ""),
                final_augmented_certificate=row.get("final_augmented_certificate", ""),
                returned_coefficients_sha256=row.get("returned_coefficients_sha256", ""),
                exception=row.get("exception", ""))])
    say("C done")

    # ---- D: augmented-bound validation
    waug = ResultGroupWriter(run_dir / "augmented", METHOD_CONSTRAINED_FREE_CN_V1)
    done = waug.existing_keys("augmented_bound_validation.csv", "case_key")
    for qi, q in enumerate(subset):
        mu = mu_of(q)
        tgq = np.linspace(0.0, mu.T, tg.n_steps + 1)
        ops_q = OperatorCache(mesh, mesh.x_min, mesh.x_max)
        prob = build_free_problem(mu, EuropeanPutPayoff(q["K"]), mi, ops_q, tgq, method_id=METHOD_CONSTRAINED_FREE_CN_V1)
        const = free_stability_constants(prob, coercivity_strategy=policy.coercivity_strategy)
        if not const.is_spd:
            continue
        z_dir = march_free_cn_direct(prob.free_ops, mu, prob.dt, prob.z0_target, prob.forcing.f_I).z
        lam = const.lambda_min_numerical
        dir_allow = float(np.hypot(0.0, free_eta_sharp(prob, free_residuals(prob, z_dir)).eta
                                   + residual_rounding_eta(prob, z_dir, lam)))
        rng = np.random.default_rng([20261007, 7, qi])
        delta = rng.standard_normal(fs.n_free)
        delta *= 1e-4 / prob.mass_norm(delta)
        cands = {
            "homogeneous_error_from_perturbed_ic": march_free_cn_direct(prob.free_ops, mu, prob.dt, prob.z0_target + delta,
                                                                        prob.forcing.f_I).z,
            "raw_deeponet": fw.neural_predictor(model, stats, tgq)(prob)[:, fs.free_idx],
            "pod_r16": pf.reduced_cn_trajectory(pod16, prob)[1],
        }
        for cname, z in cands.items():
            key = f"{q['query_id']}|{cname}"
            if key in done:
                continue
            c = prob.lift(z)
            rec = fw.certify_free_trajectory(prob, c, alpha=const.alpha_lower_estimate,
                                             pcg_iterations=policy.max_pcg_iterations,
                                             preconditioner=policy.preconditioner, tolerance=policy.tolerance)
            r = free_residuals(prob, z)
            sharp = free_eta_sharp(prob, r).eta
            true_err = max(prob.mass_norm(z[n] - z_dir[n]) for n in range(prob.n_steps + 1))
            waug.append_rows("augmented_bound_validation.csv", [dict(
                case_key=key, query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1, source=q["source"],
                candidate=cname, initial_error=rec.initial_error, residual_certificate_upper=rec.residual_certificate,
                augmented_certificate_upper=rec.augmented_certificate,
                augmented_sharp=float(np.hypot(rec.initial_error, sharp)), bound_status=rec.bound_status,
                true_max_mass_error_vs_direct=true_err, direct_reference_allowance=dir_allow,
                bound_holds=true_err <= rec.augmented_certificate + dir_allow,
                effectivity_upper=(rec.augmented_certificate / true_err) if true_err > 0 else float("nan"),
                residual_only_would_underestimate=rec.residual_certificate + dir_allow < true_err)])
        if qi % 8 == 0:
            say(f"D {qi}/{len(subset)}")
    say("D done")

    if not (run_dir / "manifest.json").exists():
        m = rm.build_manifest(run_dir.name, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                              studies={"audit459": "completed"}, seeds={"residual_seed_base": 20261007, **wf["seeds"]},
                              resolved_config=resolved, populations=pops,
                              population_used=["audit459 (all 459)", "declared subset: first 16 of each stratum"],
                              extra_inputs={"deeponet": "checkpoints/deeponet_final.pt", "pod_artifact": wf["pod"]["artifact"]})
        m["other_method_ids_in_run"] = {"audit_legacy_full/*": METHOD_LEGACY_FULL}
        rm.write_json(run_dir / "manifest.json", m)
        rm.write_json(run_dir / "resolved_config.json", resolved)
    say("done")


if __name__ == "__main__":
    main()
