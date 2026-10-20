"""M9 §9.2-9.3: point/window factors in free coordinates and financial budgets on all 30 Q_ref30.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:experiments:scripts:benchmarks:. \
        python3 scripts/v5_m9_financial.py --fourier-run <m9_fourier_run_dir> [--resume <run_dir>]

Scope (decision 2026-10-07/08): the PRIMARY price target is the exact constrained discrete
price of the production problem. Three quantities are kept separate in every row:
  bound_vs_exact_discrete     beta_I(x) * eta_aug  -- exact-arithmetic theorem, evaluated in
                              floating point (requires the returned trajectory to carry the
                              target's boundary data; it does, by projection)
  diff_vs_stored_reference    |V(x) - V_ref(x)| with the stored reference's own allowance
                              beta_I(x) * eta_aug(ref) alongside -- empirical diagnostic
  diff_vs_independent_price   |V(x) - COS(x)| (converged COS from the M9 Fourier run) --
                              continuum DIAGNOSTIC, includes discretisation error; never a
                              certified total-price error.
Methods: DeepONet and mass-weighted POD r48 through pipeline.free_workflow (method
constrained_free_cn_v1); raw candidate bounds and delivered (final) bounds both recorded.
Budgets $0.01 / $0.05 / $0.10: verified iff max over spots of the delivered bound <= budget
(spots aggregated within query first). Relative errors use a $1 floor (1% of K) -- a
reporting CONVENTION, not a derived quantity.

Window factors (mesh-only, window [log 0.8K, log 1.2K]):
  sampled_sup_beta    max of beta_I on 4001 equispaced points -- "sampled diagnostic"
  piecewise_max_beta  per-cell exact maximisation of the piecewise-quartic beta_I(x)^2
                      (cell endpoints + real roots of the cubic derivative), evaluated in
                      floating point -- an improved diagnostic, NOT a validated supremum
  window_gram_factor  sqrt(lambda_max(M_I^{-1} G_W)), G_W = Gram of phi_I on the window:
                      ||e||_{L2(W)} <= factor * ||e||_{M_I} -- an integrated-error factor,
                      kept separate from the pointwise maximum.
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
import scipy.linalg as sla
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
for sub in ("src", "experiments", "scripts", "benchmarks"):
    sys.path.insert(0, str(REPO_ROOT / sub))

import run_gate_d_benchmark_v5 as h  # noqa: E402
from cgmy.reference_solver import EuropeanPutPayoff  # noqa: E402
from certification.constrained import FreeOperatorCache, free_space_for_mesh  # noqa: E402
from certification.free_problem import raw_point_functional  # noqa: E402
from cgmy.operators import OperatorCache  # noqa: E402
from common import run_manifest as rm  # noqa: E402
from common.methods import METHOD_CONSTRAINED_FREE_CN_V1, NUMERICAL_SCOPE, ResultGroupWriter  # noqa: E402
from learning import pod_free as pf  # noqa: E402
from pipeline import free_workflow as fw  # noqa: E402

FRACS = (0.8, 0.9, 1.0, 1.1, 1.2)
BUDGETS = (0.01, 0.05, 0.10)
FLOOR = 1.0


def window_factors(mi, K: float) -> dict:
    from cgmy.reference_solver import Mesh

    mesh = Mesh(N=mi.N, p=mi.p, x_min=mi.x_min, x_max=mi.x_max)
    fo = FreeOperatorCache(OperatorCache(mesh, mi.x_min, mi.x_max), free_space_for_mesh(mi))
    I = fo.fs.free_idx
    M_I = fo.ops.M_h[I, :][:, I].toarray()
    W = np.linalg.inv(M_I)  # small (62x62) dense inverse used only for polynomial coefficients
    a, b = math.log(0.8 * K), math.log(1.2 * K)

    def phi_I(x):
        return raw_point_functional(mi, x)[I]

    xs = np.linspace(a, b, 4001)
    beta = np.array([math.sqrt(float(phi_I(x) @ fo.apply_M_inv(phi_I(x)))) for x in xs])
    knots = [x for x in mi.nodes if a < x < b]
    edges = [a] + list(knots) + [b]
    best, argbest = -1.0, None
    for x0, x1 in zip(edges[:-1], edges[1:]):
        L = x1 - x0
        P0, Pm, P1 = phi_I(x0), phi_I(0.5 * (x0 + x1)), phi_I(x1)
        # quadratic in t in [0,1]: P(t) = A0 + A1 t + A2 t^2 (exact interpolation of quadratics)
        A0, A2 = P0, 2 * P0 - 4 * Pm + 2 * P1
        A1 = P1 - P0 - A2
        c = [A2 @ W @ A2, 2 * A1 @ W @ A2, A1 @ W @ A1 + 2 * A0 @ W @ A2, 2 * A0 @ W @ A1, A0 @ W @ A0]
        cand = [0.0, 1.0] + [float(t.real) for t in np.roots(np.polyder(c)) if abs(t.imag) < 1e-12 and 0 < t.real < 1]
        for t in cand:
            v = float(np.polyval(c, t))
            if v > best:
                best, argbest = v, x0 + t * L
    # window Gram: Gauss-Legendre 6 points per cell of the window
    gx, gw = np.polynomial.legendre.leggauss(6)
    G = np.zeros_like(M_I)
    for x0, x1 in zip(edges[:-1], edges[1:]):
        for t, wt in zip(gx, gw):
            x = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * t
            p = phi_I(x)
            G += 0.5 * (x1 - x0) * wt * np.outer(p, p)
    gram = float(np.sqrt(max(sla.eigh(G, M_I, eigvals_only=True)[-1], 0.0)))
    return dict(window=f"[log 0.8K, log 1.2K] = [{a}, {b}]", sampled_sup_beta=float(beta.max()),
                sampled_argmax_x=float(xs[beta.argmax()]), piecewise_max_beta=math.sqrt(best),
                piecewise_argmax_x=argbest, n_cells=len(edges) - 1, window_gram_factor=gram,
                piecewise_minus_sampled=math.sqrt(best) - float(beta.max()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fourier-run", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args(argv)
    args.fourier_run = args.fourier_run.resolve()  # manifest records it relative to the repo root
    cfg = yaml.safe_load(h.BENCH_CONFIG.read_text())
    resolved = rm.resolve_config()
    wf = resolved["workflow_config"]
    pops = rm.declare_populations()
    qs = pops["pricing_q_ref30"]["rows"]
    cos = {}
    with open(args.fourier_run / "fourier/fourier.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["population"] == "pricing_q_ref30":
                cos[(r["query_id"], round(float(r["spot"]), 9))] = float(r["cos_ext_price"])
    run_dir = args.resume or rm.create_run_dir(rm.new_run_id("m9_financial"))
    log = open(run_dir / "run.log", "a")
    log.write(f"[{time.strftime('%H:%M:%S')}] session_start git_head_at_start={rm.git_state()['head']} "
              f"fourier_run={args.fourier_run.name} scope={NUMERICAL_SCOPE}\n")
    log.flush()
    ctx = h.load_ctx(cfg, wf)
    w = ResultGroupWriter(run_dir / "financial", METHOD_CONSTRAINED_FREE_CN_V1)
    if not (run_dir / "financial/window_factors.json").exists():
        wfac = window_factors(ctx.mesh_info, 100.0)
        (run_dir / "financial/window_factors.json").write_text(json.dumps(wfac, indent=2) + "\n")
    done = w.existing_keys("query_level.csv", "key")
    preds = {
        "DeepONet": lambda q, tg: fw.neural_predictor(ctx.model, ctx.stats, tg),
        "pod_r48": lambda q, tg: pf.free_pod_predictor(ctx.pod[48]),
    }
    for q in qs:
        ref = None
        for name, mk in preds.items():
            key = f"{q['query_id']}|{name}"
            if key in done:
                continue
            ref = ref or h.stored_reference(ctx, q)
            tg = np.linspace(0.0, q["T"], ctx.n_steps + 1)
            xs = [math.log(f * q["K"]) for f in FRACS]
            ops = OperatorCache(ctx.mesh, ctx.mesh.x_min, ctx.mesh.x_max)
            tr = fw.run_free_workflow(h.mu_of(q), EuropeanPutPayoff(q["K"]), ctx.mesh_info, ops, tg, mk(q, tg),
                                      ctx.policy, query_id=q["query_id"], eval_points=tuple(xs))
            cc, fc = tr.candidate_certificate, tr.final_certificate
            phis = [raw_point_functional(ctx.mesh_info, x) for x in xs]
            spot_rows = []
            for i, (f, x) in enumerate(zip(FRACS, xs)):
                S = f * q["K"]
                bI = ref["betas"][i]
                cand_p = float(phis[i] @ tr.candidate_coefficients[-1]) if tr.candidate_coefficients is not None else float("nan")
                del_p = float(phis[i] @ tr.delivered_coefficients[-1]) if tr.output_delivered else float("nan")
                cp = cos[(q["query_id"], round(S, 9))]
                spot_rows.append(dict(
                    key=f"{key}|{f}", query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1, predictor=name,
                    spot=S, beta_I=bI, candidate_price=cand_p,
                    candidate_bound_vs_exact_discrete=bI * cc.augmented_certificate if cc else float("nan"),
                    delivered=tr.output_delivered, delivered_price=del_p,
                    delivered_bound_vs_exact_discrete=bI * fc.augmented_certificate if fc else float("nan"),
                    stored_reference_price=ref["prices"][i], stored_reference_allowance=bI * ref["eta"],
                    candidate_diff_vs_stored_reference=abs(cand_p - ref["prices"][i]),
                    delivered_diff_vs_stored_reference=abs(del_p - ref["prices"][i]),
                    independent_cos_price=cp, delivered_diff_vs_independent_price=abs(del_p - cp),
                    candidate_diff_vs_independent_price=abs(cand_p - cp),
                    delivered_rel_diff_vs_independent_price_floor1=abs(del_p - cp) / max(cp, FLOOR)))
            w.append_rows("spot_level.csv", spot_rows)
            qrow = dict(key=key, query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1, predictor=name,
                        final_status=tr.final_status, delivered=tr.output_delivered,
                        candidate_initial_error=cc.initial_error if cc else float("nan"),
                        candidate_residual_certificate=cc.residual_certificate if cc else float("nan"),
                        candidate_augmented_certificate=cc.augmented_certificate if cc else float("nan"),
                        final_augmented_certificate=fc.augmented_certificate if fc else float("nan"),
                        stored_reference_eta_aug=ref["eta"],
                        max_candidate_bound_dollars=max(r["candidate_bound_vs_exact_discrete"] for r in spot_rows),
                        max_delivered_bound_dollars=max(r["delivered_bound_vs_exact_discrete"] for r in spot_rows),
                        max_reference_allowance_dollars=max(r["stored_reference_allowance"] for r in spot_rows),
                        max_delivered_diff_vs_stored_reference=max(r["delivered_diff_vs_stored_reference"] for r in spot_rows),
                        max_delivered_diff_vs_independent_price=max(r["delivered_diff_vs_independent_price"] for r in spot_rows),
                        max_candidate_diff_vs_independent_price=max(r["candidate_diff_vs_independent_price"] for r in spot_rows),
                        **{f"verified_budget_{b}": bool(tr.output_delivered and max(r["delivered_bound_vs_exact_discrete"]
                                                                                    for r in spot_rows) <= b) for b in BUDGETS},
                        **{f"candidate_verified_budget_{b}": bool(cc is not None and max(r["candidate_bound_vs_exact_discrete"]
                                                                                         for r in spot_rows) <= b) for b in BUDGETS},
                        total_time_s=sum(tr.timings_s.values()))
            w.append_rows("query_level.csv", [qrow])
            log.write(f"[{time.strftime('%H:%M:%S')}] {key} {tr.final_status} bound=${qrow['max_delivered_bound_dollars']:.2e}\n")
            log.flush()
    if not (run_dir / "manifest.json").exists():
        m = rm.build_manifest(run_dir.name, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                              studies={"financial_q_ref30": "completed"}, seeds=wf["seeds"], resolved_config=resolved,
                              populations=pops, population_used=["pricing_q_ref30 (all 30) x 5 spots"],
                              extra_inputs={"fourier_run_cos": str((args.fourier_run / 'fourier/fourier.csv').relative_to(REPO_ROOT))})
        m.update(budgets_dollars=BUDGETS, relative_floor_dollars=FLOOR, primary_target="exact constrained discrete price")
        rm.write_json(run_dir / "manifest.json", m)
    log.write(f"[{time.strftime('%H:%M:%S')}] done\n")


if __name__ == "__main__":
    main()
