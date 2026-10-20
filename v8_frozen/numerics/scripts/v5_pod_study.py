"""M3 POD study (roadmap §3): build the mass-weighted (production) and
Euclidean-SVD (diagnostic) free POD bases, run the rank sweep on the fixed
validation30 population, and the full-rank recovery check.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=src:experiments:scripts nohup python3 scripts/v5_pod_study.py > <log> 2>&1 &
    # resume an interrupted run:  ... scripts/v5_pod_study.py --resume <run_dir>

Outputs (results/free_integrated_v5/<run_id>/):
  basis/pod_basis_mass_weighted.npz (+ .metadata.json)       production basis, rank 62
  basis/pod_basis_euclidean_diagnostic.npz (+ .metadata.json) diagnostic basis
  sweep_mass_weighted/pod_rank_sweep.csv                       ranks 8,16,32,48,62 x validation30
  sweep_euclidean_diagnostic/pod_rank_sweep.csv                same, diagnostic basis
  recovery/full_rank_recovery.csv                              rank 62 vs dense direct free march
  references/reference_allowance.csv                           stored-reference certificates
  manifest.json, resolved_config.json, run.log, summary.md

Error definitions (all in the M_I norm on free coordinates, max over the time
levels n = 0..n_T of each query's own grid):
  projection_error   max_n ||z_ref^n - V V^T M_I z_ref^n||       (oracle best approximation)
  initial_error      ||V a^0 - z_h^0||, a^0 = V^T M_I z_h^0      (beta_0 of the ROM)
  evolution_error    max_n ||a^n - V^T M_I z_ref^n||_2            (= ||V a^n - P_V z_ref^n||_{M_I}, in-subspace)
  total_error        max_n ||V a^n - z_ref^n||
  E_rel              total_error / max_n ||z_ref^n||  (NaN + flag if the denominator is 0)
  terminal_price_error |phi(log K)^T (c_pod^{n_T} - c_ref^{n_T})|  (dollars, spot = K)
The stored reference z_ref is the trusted solve_cgmy trajectory projected onto the
problem lifting; its own augmented certificate eta_ref (reference allowance) bounds
its distance to the exact constrained discrete solution, so every error above is
"vs stored reference", accurate to +- eta_ref (and +- beta_I(log K)*eta_ref in price).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "experiments", "scripts"):
    sys.path.insert(0, str(REPO_ROOT / sub))

from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, TimeGrid, solve_cgmy  # noqa: E402
from certification.constrained import FreeOperatorCache, free_space_for_mesh, march_free_cn_direct  # noqa: E402
from certification.free_problem import (  # noqa: E402
    FORCING_SOURCE_STORED,
    beta_I,
    build_free_problem,
    free_eta_sharp,
    free_residuals,
    free_stability_constants,
    raw_point_functional,
    residual_rounding_eta,
)
from common import run_manifest as rm  # noqa: E402
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1, NUMERICAL_SCOPE, ResultGroupWriter  # noqa: E402
from data.sampling import load_mesh_and_time_grid  # noqa: E402
from learning import pod_free as pf  # noqa: E402
from pipeline import free_workflow as fw  # noqa: E402
from pipeline.predict_certify_correct import build_mesh_info  # noqa: E402

DATASET = REPO_ROOT / "data" / "production_dataset.h5"
N_TRAIN = 800  # same training population as checkpoints/pod_final.npz and the v4 POD study
RANKS = (8, 16, 32, 48, 62)
N_AUDIT_VERTEX = 8
N_AUDIT_FAILURE = 8


def log(msg: str, fh) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def mu_of(q) -> CGMYParameters:
    return CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])


def build_bases(run_dir: Path, mesh, mesh_info, free_ops, fh):
    from generate_production_dataset import load_all_train_samples

    bdir = run_dir / "basis"
    mw_path, eu_path = bdir / "pod_basis_mass_weighted.npz", bdir / "pod_basis_euclidean_diagnostic.npz"
    if mw_path.exists() and eu_path.exists():
        log("bases exist (resume) -- loading", fh)
        mw, _ = pf.load_pod_artifact(mw_path, free_ops, rank=free_ops.n_dof)
        eu, _ = pf.load_pod_artifact(eu_path, free_ops, rank=free_ops.n_dof,
                                     expected_basis_type=pf.BASIS_EUCLIDEAN_DIAGNOSTIC)
        return mw, eu
    bdir.mkdir(parents=True, exist_ok=True)
    log(f"loading {N_TRAIN} training trajectories from {DATASET}", fh)
    trajs = load_all_train_samples(DATASET, N_TRAIN)
    if len(trajs) != N_TRAIN:
        raise SystemExit(f"expected {N_TRAIN} training trajectories, got {len(trajs)}")
    for _, _, t in trajs:
        m = t.mesh_information
        if (m.N, m.p, m.n_dof, float(m.x_min), float(m.x_max)) != (
            mesh_info.N, mesh_info.p, mesh_info.n_dof, float(mesh_info.x_min), float(mesh_info.x_max)
        ):
            raise SystemExit(f"training trajectory mesh {m.N, m.x_min, m.x_max} != production mesh")
    fs = free_ops.fs
    Z = np.concatenate([t.coefficients[:, fs.free_idx] for _, _, t in trajs], axis=0)
    training = {
        "dataset_path": str(DATASET.relative_to(REPO_ROOT)), "dataset_sha256": rm.sha256_file(DATASET),
        "n_trajectories": len(trajs), "sample_selection": f"samples 0..{N_TRAIN - 1} (load_all_train_samples)",
        "n_snapshots": int(Z.shape[0]), "snapshot_matrix_sha256": rm.sha256_array(Z),
        "snapshot_weights": "uniform: every stored time level of every training trajectory weight 1",
        "snapshot_space": "free homogeneous coefficients z = P^T c (lifting is zero on free rows)",
        "centering": "uncentred",
    }
    disc = {k: v for k, v in rm.discretization_record(rm.resolve_config()["base_config"]).items() if k != "free_idx"}
    common = {"method_id": METHOD_CONSTRAINED_FREE_CN_V1, "mass_matrix_identity": pf.mass_matrix_identity(free_ops),
              "training": training, "discretization": disc, "git_head": rm.git_state()["head"]}
    n = free_ops.n_dof
    t0 = time.perf_counter()
    mw = pf.compute_mass_weighted_pod_basis(Z, free_ops, n)
    t_mw = time.perf_counter() - t0
    t0 = time.perf_counter()
    eu = pf.compute_free_pod_basis(Z, free_ops, n)
    t_eu = time.perf_counter() - t0
    pf.save_pod_artifact(mw_path, mw, dict(common, construction="M_I = L L^T; svd(L^T Z); V = L^{-T} U (triangular)",
                                          build_time_s=t_mw))
    pf.save_pod_artifact(eu_path, eu, dict(common, construction="svd(Z); V = U G^{-1/2}-type Cholesky re-orthonormalisation in M_I (DIAGNOSTIC)",
                                          build_time_s=t_eu))
    for p in (mw_path, eu_path):
        meta = json.loads(str(np.load(p)["metadata_json"]))
        meta["artifact_sha256"] = rm.sha256_file(p)
        rm.write_json(p.with_suffix(".metadata.json"), meta)
    log(f"bases built: {Z.shape[0]} snapshots, mass-weighted {t_mw:.2f}s, euclidean {t_eu:.2f}s", fh)
    return mw, eu


def reference(mu, payoff, mesh, mesh_info, ops, n_steps, policy, cache):
    """Trusted reference trajectory, its free problem (stored forcing) and its
    own augmented certificate (reference allowance)."""
    key = (mu, payoff.strike)
    if key in cache:
        return cache[key]
    traj = solve_cgmy(mu, payoff, mesh, TimeGrid(n_steps=n_steps), mode="toeplitz", store_forcing=True)
    prob = build_free_problem(mu, payoff, mesh_info, ops, traj.time_grid, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                              schemes=traj.solver_metadata.scheme_per_step, n_steps=n_steps,
                              f_bar_raw=traj.forcing_vectors, forcing_source=FORCING_SOURCE_STORED)
    const = free_stability_constants(prob, coercivity_strategy=policy.coercivity_strategy)
    z_ref = traj.coefficients[:, prob.fs.free_idx]
    c_ref = prob.lift(z_ref)
    cert = fw.certify_free_trajectory(prob, c_ref, alpha=const.alpha_lower_estimate, pcg_iterations=policy.max_pcg_iterations,
                                      preconditioner=policy.preconditioner, tolerance=policy.tolerance)
    out = dict(traj=traj, prob=prob, const=const, z_ref=z_ref, c_ref=c_ref, cert=cert)
    cache[key] = out
    return out


def sweep_rows(qid, q, basis_full, basis_label, ref, policy, mesh_info, ops):
    prob, z_ref, c_ref = ref["prob"], ref["z_ref"], ref["c_ref"]
    M = prob.free_ops.apply_M
    ref_norm = max(prob.mass_norm(z) for z in z_ref)
    x_k = math.log(q["K"])
    bK = beta_I(prob, x_k)
    phiK = raw_point_functional(prob.mesh_info, x_k)
    rows = []
    for r in RANKS:
        b = pf.truncate_basis(basis_full, r)
        V = b.modes
        proj = max(prob.mass_norm(z - V @ (V.T @ M(z))) for z in z_ref)
        alphas, z_pod = pf.reduced_cn_trajectory(b, prob)
        init = prob.mass_norm(z_pod[0] - prob.z0_target)
        evol = max(float(np.linalg.norm(alphas[n] - V.T @ M(z_ref[n]))) for n in range(prob.n_steps + 1))
        total = max(prob.mass_norm(z_pod[n] - z_ref[n]) for n in range(prob.n_steps + 1))
        e_rel, e_flag = (total / ref_norm, "") if ref_norm > 0 else (float("nan"), "zero_denominator")
        price_err = abs(float(phiK @ (prob.lift(z_pod)[-1] - c_ref[-1])))
        tr = fw.run_free_workflow(prob.mu, prob.payoff, mesh_info, ops, ref["traj"].time_grid,
                                  pf.free_pod_predictor(b), policy, query_id=qid,
                                  f_bar_raw=ref["traj"].forcing_vectors, forcing_source=FORCING_SOURCE_STORED,
                                  keep_coefficients=False)
        cc = tr.candidate_certificate
        corr = tr.correction
        pcg = sum(cc.pcg_iterations_per_step) if cc else 0
        for c in (tr.correction_certificate, tr.fallback_certificate):
            pcg += sum(c.pcg_iterations_per_step) if c else 0
        rows.append({
            "query_id": qid, "method_id": METHOD_CONSTRAINED_FREE_CN_V1, "basis_type": basis_label, "rank": r,
            "projection_error": proj, "initial_error": init, "evolution_error": evol, "total_error": total,
            "ref_max_mass_norm": ref_norm, "E_rel": e_rel, "E_rel_flag": e_flag,
            "terminal_price_error_vs_stored_reference": price_err,
            "reference_allowance_eta_aug": ref["cert"].augmented_certificate,
            "reference_allowance_price": bK * ref["cert"].augmented_certificate,
            "cand_initial_error": cc.initial_error, "cand_residual_certificate": cc.residual_certificate,
            "cand_augmented_certificate": cc.augmented_certificate, "tolerance": policy.tolerance,
            "final_status": tr.final_status, "output_delivered": tr.output_delivered,
            "correction_executed": corr is not None,
            "correction_gmres_iterations": sum(corr.iterations_per_step) if corr else 0,
            "correction_all_converged": corr.all_converged if corr else "",
            "fallback_executed": tr.fallback is not None,
            "pcg_iterations_total": pcg,
            "final_augmented_certificate": tr.final_certificate.augmented_certificate if tr.final_certificate else "",
            "t_setup_s": tr.timings_s.get("setup", 0.0), "t_predict_s": tr.timings_s.get("predict", 0.0),
            "t_gate_s": tr.timings_s.get("gate", 0.0), "t_certify_candidate_s": tr.timings_s.get("certify_candidate", 0.0),
            "t_correct_s": tr.timings_s.get("correct", 0.0), "t_recertify_s": tr.timings_s.get("recertify", 0.0),
            "t_fallback_s": tr.timings_s.get("fallback", 0.0) + tr.timings_s.get("recertify_fallback", 0.0),
            "t_total_s": sum(tr.timings_s.values()),
            "returned_coefficients_sha256": tr.delivered_trajectory_sha256 or "",
        })
    return rows


def recovery_row(qid, pop, source, mu, payoff, basis_full, mesh_info, ops, n_steps, policy):
    prob = build_free_problem(mu, payoff, mesh_info, ops, np.linspace(0.0, mu.T, n_steps + 1),
                              method_id=METHOD_CONSTRAINED_FREE_CN_V1)
    const = free_stability_constants(prob, coercivity_strategy=policy.coercivity_strategy)
    _, z_pod = pf.reduced_cn_trajectory(basis_full, prob)
    z_dir = march_free_cn_direct(prob.free_ops, mu, prob.dt, prob.z0_target, prob.forcing.f_I).z
    lam = const.lambda_min_numerical

    def parts(z):
        b0 = prob.mass_norm(z[0] - prob.z0_target)
        r = free_residuals(prob, z)
        es = free_eta_sharp(prob, r).eta
        rnd = residual_rounding_eta(prob, z, lam)
        return b0, es, rnd, float(np.max(np.abs(r))), float(np.hypot(b0, es + rnd))

    b0p, esp, rndp, rmaxp, augp = parts(z_pod)
    b0d, esd, rndd, rmaxd, augd = parts(z_dir)
    mass_err = max(prob.mass_norm(z_pod[n] - z_dir[n]) for n in range(n_steps + 1))
    coef_err = float(np.max(np.abs(z_pod - z_dir)))
    n = prob.fs.n_free
    from certification.constants import dense_from_apply

    Md = dense_from_apply(prob.free_ops.apply_M, n)
    Ad = dense_from_apply(lambda v: prob.free_ops.apply_A(mu, v), n)
    allowance = augp + augd
    return {
        "query_id": qid, "population": pop, "source": source, "method_id": METHOD_CONSTRAINED_FREE_CN_V1,
        "basis_type": basis_full.basis_type, "rank": basis_full.modes.shape[1],
        "free_gate_lambda_min_S_I": lam, "free_gate_passes": const.is_spd,
        "coef_max_abs_error": coef_err, "coef_rel_error": coef_err / float(np.max(np.abs(z_dir))),
        "mass_max_error": mass_err,
        "mass_rel_error": mass_err / max(prob.mass_norm(z) for z in z_dir),
        "pod_initial_error": b0p, "pod_eta_sharp": esp, "pod_residual_max_abs": rmaxp, "pod_rounding_eta": rndp,
        "direct_initial_error": b0d, "direct_eta_sharp": esd, "direct_residual_max_abs": rmaxd,
        "direct_rounding_eta": rndd, "allowance_from_residuals": allowance,
        "within_allowance": mass_err <= allowance,
        "cond_V": float(np.linalg.cond(basis_full.modes)), "cond_M_I": float(np.linalg.cond(Md)),
        "cond_F": float(np.linalg.cond(Md / prob.dt + 0.5 * Ad)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args(argv)
    resolved = rm.resolve_config()
    wf_cfg = resolved["workflow_config"]
    policy = fw.FreeWorkflowPolicy.from_config(wf_cfg)
    pops = rm.declare_populations()
    mesh, tgrid = load_mesh_and_time_grid(rm.BASE_CONFIG)
    mesh_info = build_mesh_info(mesh)
    ops = OperatorCache(mesh, mesh.x_min, mesh.x_max)
    free_ops = FreeOperatorCache(ops, free_space_for_mesh(mesh_info))
    np.random.seed(wf_cfg["seeds"]["global"])

    if args.resume is not None:
        run_dir = args.resume
        run_id = run_dir.name
    else:
        run_id = rm.new_run_id("m3_pod_study")
        run_dir = rm.create_run_dir(run_id)
    fh = open(run_dir / "run.log", "a")
    log(f"run_id={run_id} method={METHOD_CONSTRAINED_FREE_CN_V1} scope={NUMERICAL_SCOPE}", fh)
    mw, eu = build_bases(run_dir, mesh, mesh_info, free_ops, fh)

    groups = {
        pf.BASIS_MASS_WEIGHTED: (mw, ResultGroupWriter(run_dir / "sweep_mass_weighted", METHOD_CONSTRAINED_FREE_CN_V1)),
        pf.BASIS_EUCLIDEAN_DIAGNOSTIC: (eu, ResultGroupWriter(run_dir / "sweep_euclidean_diagnostic", METHOD_CONSTRAINED_FREE_CN_V1)),
    }
    refw = ResultGroupWriter(run_dir / "references", METHOD_CONSTRAINED_FREE_CN_V1)
    cache: dict = {}
    for q in pops["validation30"]["rows"]:
        qid = q["query_id"]
        payoff = EuropeanPutPayoff(strike=q["K"])
        ref = None
        for label, (basis, w) in groups.items():
            if qid in w.existing_keys("pod_rank_sweep.csv", "query_id"):
                continue
            ref = reference(mu_of(q), payoff, mesh, mesh_info, ops, tgrid.n_steps, policy, cache)
            w.append_rows("pod_rank_sweep.csv", sweep_rows(qid, q, basis, label, ref, policy, mesh_info, ops))
            log(f"sweep {label} {qid} done", fh)
        if ref is not None and qid not in refw.existing_keys("reference_allowance.csv", "query_id"):
            c = ref["cert"]
            refw.append_rows("reference_allowance.csv", [{
                "query_id": qid, "method_id": METHOD_CONSTRAINED_FREE_CN_V1,
                "reference": "solve_cgmy toeplitz GMRES rtol=1e-10, projected onto problem lifting",
                "initial_error": c.initial_error, "residual_certificate": c.residual_certificate,
                "augmented_certificate": c.augmented_certificate, "bound_status": c.bound_status,
                "boundary_discrepancy_raw": float(np.max(np.abs(ref["traj"].coefficients[:, ref["prob"].fs.bc_idx]
                                                                - ref["prob"].forcing.lifting[:, ref["prob"].fs.bc_idx]))),
            }])
        cache.clear()

    recw = ResultGroupWriter(run_dir / "recovery", METHOD_CONSTRAINED_FREE_CN_V1)
    audit = pops["audit459"]["rows"]
    selection = ([("validation30", q) for q in pops["validation30"]["rows"]]
                 + [("audit459", q) for q in [r for r in audit if r["source"] == "exact_vertex"][:N_AUDIT_VERTEX]]
                 + [("audit459", q) for q in [r for r in audit if r["source"] == "known_failure"][:N_AUDIT_FAILURE]])
    done = recw.existing_keys("full_rank_recovery.csv", "query_id")
    for pop, q in selection:
        if q["query_id"] in done:
            continue
        row = recovery_row(q["query_id"], pop, q["source"], mu_of(q), EuropeanPutPayoff(strike=q["K"]), mw,
                           mesh_info, ops, tgrid.n_steps, policy)
        recw.append_rows("full_rank_recovery.csv", [row])
        log(f"recovery {q['query_id']} mass_err={row['mass_max_error']:.3e} allowance={row['allowance_from_residuals']:.3e}", fh)

    if not (run_dir / "manifest.json").exists():
        manifest = rm.build_manifest(
            run_id, method_id=METHOD_CONSTRAINED_FREE_CN_V1, studies={"pod_free_baseline": "completed"},
            seeds=wf_cfg["seeds"], resolved_config=resolved, populations=pops,
            population_used=["validation30", f"audit459: first {N_AUDIT_VERTEX} exact_vertex + first "
                             f"{N_AUDIT_FAILURE} known_failure rows in file order"],
            extra_inputs={"training_dataset": str(DATASET.relative_to(REPO_ROOT))},
        )
        manifest["basis_artifacts_sha256"] = {
            p.name: rm.sha256_file(p) for p in sorted((run_dir / "basis").glob("*.npz"))
        }
        rm.write_json(run_dir / "manifest.json", manifest)
        rm.write_json(run_dir / "resolved_config.json", resolved)
    log("done", fh)


if __name__ == "__main__":
    main()
