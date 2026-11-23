"""M6 Riesz-bracket validation with nontrivial residuals (roadmap §6.2).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:experiments:scripts \
        python3 scripts/v5_m6_riesz.py [--resume <run_dir>]

Population: declared subset of Q_audit -- first 8 rows of each stratum in file order
(sobol_main_box, exact_vertex, known_failure) -- plus manufactured SPD cases.

Residual sources (free CN residuals r_I^{n+1/2}, production grid n_steps=60), at time
steps n in {0, 30, 59}:
  reference   trusted solve_cgmy trajectory, projected onto the lifting
  raw_neural  DeepONet candidate, projected
  pod_r16 / pod_r32  finite-rank mass-weighted POD reconstructions
  corrected   free GMRES march warm-started by the DeepONet candidate
  manufactured  S = Q diag(geomspace(1, cond)) Q^T (exactly symmetric), cond in
                {1e1,1e2,1e4,1e6,1e8}, n=62, three seeded residuals each
PCG budgets {0,1,2,5,10,20,50,200} x preconditioners {none, mass, jacobi, banded}
(manufactured: none, jacobi).

Independent reference q_ref = r^T S^{-1} r: mpmath (40 digits) Cholesky of the SAME
assembled double matrix S (dense columns of the production apply_S, symmetrised
exactly in mpmath: (S + S^T)/2), with our own mp triangular solves. It preserves
the rounded input matrix; it does not regenerate an ideal one. The PCG bracket
uses the matrix-free production apply_S, whose action differs from the dense
assembled matrix by rounding; comparison_uncertainty = 8*n*eps*cond2(S)*q_ref
covers that first-order operator perturbation (documented assumption).

Status per row: violation if (q_ref - q_lower) or (q_upper - q_ref) < -uncertainty;
numerically_unresolved if not a violation but a margin is within +-uncertainty;
pass otherwise. Also archived: the alpha estimator (Weyl, floating point) and its
check against the mpmath lambda_min of the same S (hypothesis alpha <= lambda_min).
Output: riesz/riesz_bracket_validation.csv (+ alpha_check.csv).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "experiments", "scripts"):
    sys.path.insert(0, str(REPO_ROOT / sub))

import torch  # noqa: E402

from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, TimeGrid, solve_cgmy  # noqa: E402
from certification.constants import dense_from_apply, weyl_residual_enclosure  # noqa: E402
from certification.constrained import FreeOperatorCache, free_space_for_mesh, march_free_cn  # noqa: E402
from certification.free_problem import FORCING_SOURCE_STORED, build_free_problem, free_residuals, free_stability_constants  # noqa: E402
from certification.riesz_solver import solve_riesz_pcg  # noqa: E402
from common import run_manifest as rm  # noqa: E402
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1, NUMERICAL_SCOPE, ResultGroupWriter  # noqa: E402
from data.sampling import load_mesh_and_time_grid, load_parameter_box  # noqa: E402
from learning import pod_free as pf  # noqa: E402
from learning.normalization import stats_from_box  # noqa: E402
from pipeline import free_workflow as fw  # noqa: E402
from pipeline.predict_certify_correct import build_mesh_info  # noqa: E402

mp.mp.dps = 40
EPS = np.finfo(float).eps
BUDGETS = (0, 1, 2, 5, 10, 20, 50, 200)
PRECONDS = ("none", "mass", "jacobi", "banded")
STEPS = (0, 30, 59)
PER_STRATUM = 8


class MpCholesky:
    """mpmath Cholesky of exactly-symmetrised double data, with O(n^2) solves."""

    def __init__(self, S: np.ndarray) -> None:
        n = S.shape[0]
        A = mp.matrix(n, n)
        for i in range(n):
            for j in range(n):
                A[i, j] = (mp.mpf(float(S[i, j])) + mp.mpf(float(S[j, i]))) / 2
        self.A, self.n = A, n
        self.L = mp.cholesky(A)
        w = mp.eigsy(A, eigvals_only=True)
        self.lam_min = min(w)

    def quad(self, r: np.ndarray) -> tuple[mp.mpf, mp.mpf]:
        n, L = self.n, self.L
        b = [mp.mpf(float(x)) for x in r]
        y = [mp.mpf(0)] * n
        for i in range(n):
            y[i] = (b[i] - mp.fsum(L[i, k] * y[k] for k in range(i))) / L[i, i]
        x = [mp.mpf(0)] * n
        for i in reversed(range(n)):
            x[i] = (y[i] - mp.fsum(L[k, i] * x[k] for k in range(i + 1, n))) / L[i, i]
        q = mp.fsum(b[i] * x[i] for i in range(n))
        res = mp.sqrt(mp.fsum((mp.fsum(self.A[i, k] * x[k] for k in range(n)) - b[i]) ** 2 for i in range(n)))
        bn = mp.sqrt(mp.fsum(v * v for v in b))
        return q, (res / bn if bn > 0 else mp.mpf(0))


class DenseOps:
    """Duck-typed ops for manufactured SPD cases (apply_S, diag_S, n_dof)."""

    def __init__(self, S):
        self.S, self.n_dof = S, S.shape[0]

    def apply_S(self, mu, v):
        return self.S @ np.asarray(v, dtype=float).ravel()

    def diag_S(self, mu):
        return np.diag(self.S).copy()


def classify(q_ref, lo, up, unc):
    lm, um = q_ref - lo, up - q_ref
    if lm < -unc or um < -unc:
        return lm, um, "violation"
    if lm <= unc or um <= unc:
        return lm, um, "numerically_unresolved"
    return lm, um, "pass"


def unresolved_side(lm, um, unc) -> str:
    sides = [s for s, m in (("lower", lm), ("upper", um)) if abs(m) <= unc]
    return "+".join(sides)


def bracket_rows(key_base, r, mu, ops, alpha, chol, cond, preconds, extra):
    q_mp, solve_res = chol.quad(r)
    q_ref = float(q_mp)
    unc = 8 * len(r) * EPS * cond * abs(q_ref)
    rows = []
    for pc in preconds:
        for b in BUDGETS:
            try:
                br = solve_riesz_pcg(r, mu, ops, alpha, max_iter=b, preconditioner=pc)
                lm, um, st = classify(q_ref, br.lower_bound, br.upper_bound, unc)
                rows.append(dict(extra, case_key=f"{key_base}|{pc}|{b}", pcg_iterations=br.n_pcg_iterations,
                                 pcg_budget=b, preconditioner=pc, q_reference=q_ref, q_lower=br.lower_bound,
                                 q_upper=br.upper_bound, lower_margin=lm, upper_margin=um,
                                 relative_upper_margin=um / q_ref if q_ref > 0 else float("nan"),
                                 relative_riesz_residual=br.relative_riesz_residual,
                                 reference_solve_residual=float(solve_res), estimated_condition=cond,
                                 comparison_uncertainty=unc, comparison_status=st,
                                 unresolved_side=unresolved_side(lm, um, unc), alpha=alpha, exception=""))
            except Exception as exc:
                rows.append(dict(extra, case_key=f"{key_base}|{pc}|{b}", pcg_iterations="", pcg_budget=b,
                                 preconditioner=pc, q_reference=q_ref, q_lower="", q_upper="", lower_margin="",
                                 upper_margin="", relative_upper_margin="", relative_riesz_residual="",
                                 reference_solve_residual=float(solve_res), estimated_condition=cond,
                                 comparison_uncertainty=unc, comparison_status="exception", unresolved_side="", alpha=alpha,
                                 exception=f"{type(exc).__name__}: {exc}"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    resolved = rm.resolve_config()
    wf = resolved["workflow_config"]
    policy = fw.FreeWorkflowPolicy.from_config(wf)
    pops = rm.declare_populations()
    audit = pops["audit459"]["rows"]
    subset = []
    for src in ("sobol_main_box", "exact_vertex", "known_failure"):
        subset += [q for q in audit if q["source"] == src][:PER_STRATUM]
    mesh, tg = load_mesh_and_time_grid(rm.BASE_CONFIG)
    mi = build_mesh_info(mesh)
    fs = free_space_for_mesh(mi)
    run_dir = args.resume or rm.create_run_dir(rm.new_run_id("m6_riesz_bracket"))
    log = open(run_dir / "run.log", "a")

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"session_start run_dir={run_dir.name} git_head_at_start={rm.git_state()['head']} scope={NUMERICAL_SCOPE}")
    (run_dir / "declared_subset.txt").write_text(
        f"first {PER_STRATUM} rows of each Q_audit stratum in file order\n" + "\n".join(q["query_id"] for q in subset) + "\n")
    w = ResultGroupWriter(run_dir / "riesz", METHOD_CONSTRAINED_FREE_CN_V1)
    done = w.existing_keys("riesz_bracket_validation.csv", "case_key")

    from v5_g1_traces import load_deeponet
    box = load_parameter_box(rm.BASE_CONFIG)
    model = load_deeponet(mesh, tg, box, box.strike)
    stats = stats_from_box(box)
    fo0 = FreeOperatorCache(OperatorCache(mesh, mesh.x_min, mesh.x_max), fs)
    art = REPO_ROOT / wf["pod"]["artifact"]
    pods = {r: pf.load_pod_artifact(art, fo0, rank=r)[0] for r in (16, 32)}

    for qi, q in enumerate(subset):
        if all(f"{q['query_id']}|{s}|{n}|banded|200" in done
               for s in ("reference", "raw_neural", "pod_r16", "pod_r32", "corrected") for n in STEPS):
            continue
        mu = CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
        payoff = EuropeanPutPayoff(q["K"])
        ops = OperatorCache(mesh, mesh.x_min, mesh.x_max)
        traj = solve_cgmy(mu, payoff, mesh, TimeGrid(n_steps=tg.n_steps), mode="toeplitz", store_forcing=True)
        prob = build_free_problem(mu, payoff, mi, ops, traj.time_grid, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                                  f_bar_raw=traj.forcing_vectors, forcing_source=FORCING_SOURCE_STORED,
                                  schemes=traj.solver_metadata.scheme_per_step)
        const = free_stability_constants(prob, coercivity_strategy=policy.coercivity_strategy)
        S = dense_from_apply(lambda v: prob.free_ops.apply_S(mu, v), fs.n_free)
        chol = MpCholesky(S)
        cond = float(np.linalg.cond(0.5 * (S + S.T)))
        alpha = const.alpha_lower_estimate
        w.append_rows("alpha_check.csv", [dict(
            case_key=f"{q['query_id']}|alpha", method_id=METHOD_CONSTRAINED_FREE_CN_V1, query_id=q["query_id"],
            source=q["source"], alpha_estimator=const.coercivity_strategy, alpha=alpha,
            lambda_min_double=const.lambda_min_numerical, lambda_min_mp=float(chol.lam_min),
            alpha_minus_lambda_min_mp=float(mp.mpf(alpha) - chol.lam_min),
            hypothesis_alpha_le_lambda_min_holds=bool(mp.mpf(alpha) <= chol.lam_min))]) \
            if f"{q['query_id']}|alpha" not in w.existing_keys("alpha_check.csv", "case_key") else None
        tgq = traj.time_grid
        cand = prob.lift(fw.neural_predictor(model, stats, tgq)(prob)[:, fs.free_idx])
        corr = march_free_cn(prob.free_ops, mu, prob.dt, prob.z0_target, prob.forcing.f_I,
                             warm_start=cand[:, fs.free_idx], gmres_rtol=policy.gmres_rtol,
                             gmres_restart=policy.gmres_restart)
        sources = {
            "reference": traj.coefficients[:, fs.free_idx],
            "raw_neural": cand[:, fs.free_idx],
            "pod_r16": pf.reduced_cn_trajectory(pods[16], prob)[1],
            "pod_r32": pf.reduced_cn_trajectory(pods[32], prob)[1],
            "corrected": corr.z,
        }
        for sname, z in sources.items():
            R = free_residuals(prob, z)
            for n in STEPS:
                kb = f"{q['query_id']}|{sname}|{n}"
                if f"{kb}|banded|200" in done:
                    continue
                r = R[n]
                extra = dict(query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1, source=q["source"],
                             residual_source=sname, time_step=n, residual_norm=float(np.linalg.norm(r)))
                w.append_rows("riesz_bracket_validation.csv",
                              bracket_rows(kb, r, mu, prob.free_ops, alpha, chol, cond, PRECONDS, extra))
        say(f"query {qi + 1}/{len(subset)} {q['query_id']} done")

    # manufactured SPD
    for cond_t in (1e1, 1e2, 1e4, 1e6, 1e8):
        for k in range(3):
            kb = f"manufactured_cond{cond_t:.0e}|r{k}|0"
            if f"{kb}|jacobi|200" in done:
                continue
            rng = np.random.default_rng([20261007, int(math.log10(cond_t)), k])
            Q, _ = np.linalg.qr(rng.standard_normal((fs.n_free, fs.n_free)))
            S = (Q * np.geomspace(1.0, cond_t, fs.n_free)) @ Q.T
            S = 0.5 * (S + S.T)
            w_, V = np.linalg.eigh(S)
            alpha = weyl_residual_enclosure(S, w_, V)
            chol = MpCholesky(S)
            r = rng.standard_normal(fs.n_free)
            extra = dict(query_id=f"manufactured_cond{cond_t:.0e}", method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                         source="manufactured_spd", residual_source="manufactured", time_step=k,
                         residual_norm=float(np.linalg.norm(r)))
            w.append_rows("riesz_bracket_validation.csv",
                          bracket_rows(kb, r, None, DenseOps(S), alpha, chol, float(np.linalg.cond(S)),
                                       ("none", "jacobi"), extra))
    say("manufactured done")
    if not (run_dir / "manifest.json").exists():
        m = rm.build_manifest(run_dir.name, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                              studies={"riesz_bracket_campaign": "completed"}, seeds={"base": 20261007, **wf["seeds"]},
                              resolved_config=resolved, populations=pops,
                              population_used=[f"audit459 subset: first {PER_STRATUM} per stratum", "manufactured SPD"],
                              extra_inputs={"deeponet": "checkpoints/deeponet_final.pt", "pod_artifact": wf["pod"]["artifact"]})
        m["reference_method"] = "mpmath dps=40 Cholesky of the same assembled double S (exactly symmetrised)"
        rm.write_json(run_dir / "manifest.json", m)
        rm.write_json(run_dir / "resolved_config.json", resolved)
    say("done")


if __name__ == "__main__":
    main()
