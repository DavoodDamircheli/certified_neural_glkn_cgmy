"""Repaired Gate-D benchmark harness, version gate_d_v5.0 (roadmap §5, M5).

Does NOT edit or reuse the v4 harness. Every trajectory method solves or
certifies the constrained free CN problem (method_id constrained_free_cn_v1);
COS is an independent terminal-price method with its own method id and result
group. Configuration is frozen in configs/v5_benchmark.yaml.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:experiments \
        python3 benchmarks/run_gate_d_benchmark_v5.py --smoke
    # --full (Q_bench120) refuses to run without --approved-full-campaign.

Workloads (never combined into one speed-up number):
  trajectory      target: eta_aug <= tolerance (algebraic, vs exact discrete CN solution)
  terminal_price  target: max over spots |V - V_h| <= budget dollars (vs exact discrete price)
Target status vocabulary: verified / empirically_checked / failed / unavailable
(definitions in the config). Delivery is never equated with target attainment.

Cache modes:
  cold   -- per (method, query, repetition): a fresh OperatorCache, built inside the
            method's timer; every lazy factor (Toeplitz blocks, M_h LU, M_I LU, S_I
            spectrum, beta_I point factors, POD reduced operator) is built inside the
            timer. Neural weights / POD basis FILE loading is a one-time process cost,
            recorded in setup/one_time_setup.csv and not charged per query (declared policy).
            Warmup: one run per method on the warmup query (in no population) with its
            own throwaway caches -- warms the interpreter/BLAS/torch, donates nothing.
  reuse  -- per method, a template of MESH-ONLY objects is built once (timed as one-time
            cost): OperatorCache mesh data (M_h, D_h, M_h LU), the free partition and the
            M_I factorization, point functionals phi(x_s) and beta_I(x_s) (depend on the
            mesh and spots only), model weights / POD basis. Each (query, repetition) gets a
            clone sharing only those, with EMPTY mu-level caches: Toeplitz blocks, A_I, S_I
            spectrum/alpha, kappa, POD reduced operator, forcing and lifting are rebuilt and
            charged online every time (they depend on mu and cannot be reused across
            different mu). Listed in setup/reusable_objects.json.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # validation/ lives at the repo root
for sub in ("src", "experiments", "scripts", "benchmarks"):
    sys.path.insert(0, str(REPO_ROOT / sub))

import torch  # noqa: E402

import certification.constrained as constrained_mod  # noqa: E402
from cgmy.operators import OperatorCache  # noqa: E402
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, TimeGrid, solve_cgmy  # noqa: E402
from certification.constrained import FreeOperatorCache, free_space_for_mesh, march_free_cn, march_free_cn_direct  # noqa: E402
from certification.free_problem import (  # noqa: E402
    FORCING_SOURCE_STORED,
    beta_I,
    build_free_problem,
    free_stability_constants,
    raw_point_functional,
)
from common import run_manifest as rm  # noqa: E402
from common.methods import (  # noqa: E402
    KNOWN_METHODS,
    METHOD_CONSTRAINED_FREE_CN_V1,
    METHOD_INDEPENDENT_COS_V1,
    NUMERICAL_SCOPE,
    ResultGroupWriter,
)
from data.sampling import load_mesh_and_time_grid, load_parameter_box  # noqa: E402
from learning import pod_free as pf  # noqa: E402
from learning.normalization import stats_from_box  # noqa: E402
from pipeline import free_workflow as fw  # noqa: E402
from pipeline.legacy_guard import forbid_legacy_calls, summarize  # noqa: E402
from pipeline.predict_certify_correct import build_mesh_info  # noqa: E402
from validation.cgmy_reference_pricing import CGMYMarketParams  # noqa: E402
from validation.cos_pricer import cos_put_price, truncation_range  # noqa: E402
import gate_d_v5_summary as summary  # noqa: E402

BENCH_CONFIG = REPO_ROOT / "configs" / "v5_benchmark.yaml"
DEEPONET = REPO_ROOT / "checkpoints" / "deeponet_final.pt"

# Benchmark status vocabulary (roadmap §5.4).
DIRECT_ACCEPT = "direct_accept"
CORRECTED_ACCEPT = "corrected_accept"
QUALIFIED_FALLBACK = "qualified_fallback"
OVER_BUDGET = "over_budget"
UNCERTIFIABLE = "uncertifiable"
NUMERICAL_FAILURE = "numerical_failure"
DELIVERED_NO_CERTIFICATE = "delivered_no_certificate"  # COS only; never a target claim
ACCEPT_STATUSES = (DIRECT_ACCEPT, CORRECTED_ACCEPT, QUALIFIED_FALLBACK)

_WORKFLOW_STATUS_MAP = {
    fw.STATUS_ACCEPTED_DIRECT: DIRECT_ACCEPT,
    fw.STATUS_ACCEPTED_CORRECTED: CORRECTED_ACCEPT,
    fw.STATUS_ACCEPTED_CORRECTED_NONCONVERGED: CORRECTED_ACCEPT,
    fw.STATUS_ACCEPTED_FALLBACK: QUALIFIED_FALLBACK,
    fw.STATUS_UNCERTIFIABLE: UNCERTIFIABLE,
    fw.STATUS_CORRECTION_FAILED: NUMERICAL_FAILURE,
    fw.STATUS_NEEDS_CORRECTION_NOT_RUN: OVER_BUDGET,
}


# ---------------------------------------------------------------- caches

class Caches:
    """Creates operator caches and records isolation evidence for every row."""

    def __init__(self, mesh, mesh_info, session: int = 0) -> None:
        self.mesh, self.mesh_info = mesh, mesh_info
        self.serial = 0
        self.session = session  # serials are "s<session>-<n>": unique across resumed sessions

    def _ops(self) -> OperatorCache:
        self.serial += 1
        ops = OperatorCache(self.mesh, self.mesh.x_min, self.mesh.x_max)
        ops._bench_serial = f"s{self.session}-{self.serial}"  # type: ignore[attr-defined]
        return ops

    def cold(self):
        ops = self._ops()
        return ops, None

    def template(self, spot_xs: list[float]) -> dict:
        ops = self._ops()
        fs = free_space_for_mesh(self.mesh_info)
        fo = FreeOperatorCache(ops, fs)
        fo._mass_lu()  # mesh-only factorization, part of the one-time cost
        phis = {x: raw_point_functional(self.mesh_info, x) for x in spot_xs}
        idx = fs.free_idx
        betas = {x: float(np.sqrt(phis[x][idx] @ fo.apply_M_inv(phis[x][idx]))) for x in spot_xs}
        return {"ops": ops, "fs": fs, "mass_lu": fo._M_I_lu, "phis": phis, "betas": betas}

    def from_template(self, tpl: dict):
        self.serial += 1
        ops = object.__new__(OperatorCache)
        ops.__dict__.update(tpl["ops"].__dict__)  # shares M_h, D_h, M_h LU (mesh-only)
        ops._toeplitz_cache = {}  # mu-level caches start EMPTY
        ops._banded_precond_cache = {}
        ops._bench_serial = f"s{self.session}-{self.serial}"  # type: ignore[attr-defined]
        fo = FreeOperatorCache(ops, tpl["fs"])
        fo._M_I_lu = tpl["mass_lu"]
        return ops, fo


def cache_evidence(ops, free_ops, mode: str) -> dict:
    return {
        "cache_mode": mode, "cache_serial": ops._bench_serial,
        "toeplitz_entries_at_start": len(ops._toeplitz_cache),
        "mass_I_factor_prebuilt": bool(free_ops is not None and free_ops._M_I_lu is not None),
    }


# ---------------------------------------------------------------- context

@dataclasses.dataclass
class Ctx:
    cfg: dict
    wf_cfg: dict
    policy: fw.FreeWorkflowPolicy
    mesh: Any
    mesh_info: Any
    n_steps: int
    model: Any
    stats: Any
    pod: dict  # rank -> FreePODBasis
    basis_ids: dict
    caches: Caches
    max_n_dof: int = 400  # dense eigen/LU size guard (raised deliberately by the resolution study)


def mu_of(q: dict) -> CGMYParameters:
    return CGMYParameters(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])


def spots_x(q: dict, cfg: dict) -> list[float]:
    return [math.log(f * q["K"]) for f in cfg["workloads"]["terminal_price"]["spot_fractions"]]


def time_grid(q: dict, n_steps: int) -> np.ndarray:
    return np.linspace(0.0, q["T"], n_steps + 1)


# ---------------------------------------------------------------- trajectory methods

class Phases:
    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0


def _base_result() -> dict:
    return dict(solver_converged="", certificate_computed=False, augmented_bound=float("nan"),
                initial_error=float("nan"), residual_certificate=float("nan"),
                correction_executed=False, fallback_executed=False, fallback_verified_ran="",
                output_delivered=False, final_status="", qualification="", coefficients=None,
                exception="", problem=None, pcg_iterations_total=0, gmres_iterations_total=0)


def _certify_deterministic(ph, ctx, problem, c, res, *, converged: bool):
    with ph("gate"):
        const = free_stability_constants(problem, coercivity_strategy=ctx.policy.coercivity_strategy,
                                         max_n_dof=ctx.max_n_dof)
    if not const.is_spd:
        res.update(final_status=UNCERTIFIABLE)
        return res
    with ph("certify"):
        rec = fw.certify_free_trajectory(problem, c, alpha=const.alpha_lower_estimate,
                                         pcg_iterations=ctx.policy.max_pcg_iterations,
                                         preconditioner=ctx.policy.preconditioner, tolerance=ctx.policy.tolerance)
    res["pcg_iterations_total"] = res.get("pcg_iterations_total", 0) + sum(rec.pcg_iterations_per_step)
    res.update(certificate_computed=rec.bound_status == "valid", augmented_bound=rec.augmented_certificate,
               initial_error=rec.initial_error, residual_certificate=rec.residual_certificate,
               solver_converged=converged)
    if rec.bound_status != "valid":
        res.update(final_status=NUMERICAL_FAILURE, qualification=rec.bound_status)
    elif rec.accepted:
        res.update(final_status=DIRECT_ACCEPT if converged else CORRECTED_ACCEPT,
                   qualification="" if converged else "certificate_qualified_nonconverged",
                   output_delivered=True, coefficients=c)
    else:
        res.update(final_status=OVER_BUDGET)
    return res


def _problem(ph, ctx, q, ops, free_ops, **kw):
    with ph("setup_problem"):
        return build_free_problem(mu_of(q), EuropeanPutPayoff(strike=q["K"]), ctx.mesh_info, ops,
                                  time_grid(q, ctx.n_steps), method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                                  free_ops=free_ops, **kw)


def m_reference_trusted(ctx, q, ops, free_ops, ph, policy=None, hooks=None):
    res = _base_result()
    with ph("solve"):
        traj = solve_cgmy(mu_of(q), EuropeanPutPayoff(strike=q["K"]), ctx.mesh, TimeGrid(n_steps=ctx.n_steps),
                          mode="toeplitz", store_forcing=True)
    problem = _problem(ph, ctx, q, ops, free_ops, f_bar_raw=traj.forcing_vectors,
                       forcing_source=FORCING_SOURCE_STORED, schemes=traj.solver_metadata.scheme_per_step)
    with ph("project"):
        c = problem.lift(traj.coefficients[:, problem.fs.free_idx])
    res["problem"] = problem
    # solve_cn only warns on nonconvergence; its per-step record is not exposed -> not claimed.
    return _certify_deterministic(ph, ctx, problem, c, res, converged=True) | {"qualification_note": "trusted solver convergence flags not exposed by solve_cn"}


def m_free_direct_lu(ctx, q, ops, free_ops, ph, policy=None, hooks=None):
    res = _base_result()
    problem = _problem(ph, ctx, q, ops, free_ops)
    res["problem"] = problem
    with ph("solve"):
        m = march_free_cn_direct(problem.free_ops, problem.mu, problem.dt, problem.z0_target, problem.forcing.f_I,
                                 max_n_free=ctx.max_n_dof)
    if m.aborted:
        res.update(final_status=NUMERICAL_FAILURE, exception="; ".join(s.exception or "" for s in m.steps))
        return res
    with ph("project"):
        c = problem.lift(m.z)
    return _certify_deterministic(ph, ctx, problem, c, res, converged=m.all_converged)


def m_free_cold_gmres(ctx, q, ops, free_ops, ph, policy=None, hooks=None):
    pol = policy or ctx.policy
    res = _base_result()
    problem = _problem(ph, ctx, q, ops, free_ops)
    res["problem"] = problem
    with ph("solve"):
        m = march_free_cn(problem.free_ops, problem.mu, problem.dt, problem.z0_target, problem.forcing.f_I,
                          gmres_rtol=pol.gmres_rtol, gmres_restart=pol.gmres_restart, gmres_maxiter=pol.gmres_maxiter)
    res["gmres_iterations_total"] = sum(s.iterations for s in m.steps)
    if m.aborted:
        res.update(final_status=NUMERICAL_FAILURE, exception="; ".join(s.exception or "" for s in m.steps))
        return res
    with ph("project"):
        c = problem.lift(m.z)
    return _certify_deterministic(ph, ctx, problem, c, res, converged=m.all_converged)


def _neural_candidate(ctx, q, problem, ph):
    with ph("predict"):
        raw = fw.neural_predictor(ctx.model, ctx.stats, time_grid(q, ctx.n_steps))(problem)
    with ph("project"):
        return problem.lift(raw[:, problem.fs.free_idx])


def m_neural_warm_gmres(ctx, q, ops, free_ops, ph, policy=None, hooks=None):
    pol = policy or ctx.policy
    res = _base_result()
    problem = _problem(ph, ctx, q, ops, free_ops)
    res["problem"] = problem
    cand = _neural_candidate(ctx, q, problem, ph)
    with ph("correct"):
        m = march_free_cn(problem.free_ops, problem.mu, problem.dt, problem.z0_target, problem.forcing.f_I,
                          warm_start=cand[:, problem.fs.free_idx], gmres_rtol=pol.gmres_rtol,
                          gmres_restart=pol.gmres_restart, gmres_maxiter=pol.gmres_maxiter)
    res["correction_executed"] = True
    res["gmres_iterations_total"] = sum(s.iterations for s in m.steps)
    if m.aborted:
        res.update(final_status=NUMERICAL_FAILURE, exception="; ".join(s.exception or "" for s in m.steps))
        return res
    with ph("project"):
        c = problem.lift(m.z)
    out = _certify_deterministic(ph, ctx, problem, c, res, converged=m.all_converged)
    if out["final_status"] == DIRECT_ACCEPT:  # corrected output, never a direct accept
        out["final_status"] = CORRECTED_ACCEPT
    return out


def m_neural_direct_riesz(ctx, q, ops, free_ops, ph, policy=None, hooks=None):
    """Candidate certification only (quick PCG cap); never corrected. A bound above
    the tolerance is OVER_BUDGET and the candidate is NOT delivered."""
    pol = policy or ctx.policy
    res = _base_result()
    problem = _problem(ph, ctx, q, ops, free_ops)
    res["problem"] = problem
    cand = _neural_candidate(ctx, q, problem, ph)
    with ph("gate"):
        const = free_stability_constants(problem, coercivity_strategy=pol.coercivity_strategy)
    if not const.is_spd:
        res.update(final_status=UNCERTIFIABLE)
        return res
    with ph("certify"):
        rec = fw.certify_free_trajectory(problem, cand, alpha=const.alpha_lower_estimate,
                                         pcg_iterations=pol.quick_pcg_iterations,
                                         preconditioner=pol.preconditioner, tolerance=pol.tolerance)
    res["pcg_iterations_total"] = sum(rec.pcg_iterations_per_step)
    res.update(certificate_computed=rec.bound_status == "valid", augmented_bound=rec.augmented_certificate,
               initial_error=rec.initial_error, residual_certificate=rec.residual_certificate)
    if rec.bound_status != "valid":
        res.update(final_status=NUMERICAL_FAILURE, qualification=rec.bound_status)
    elif rec.accepted:
        res.update(final_status=DIRECT_ACCEPT, output_delivered=True, coefficients=cand)
    else:
        res.update(final_status=OVER_BUDGET)
    return res


def _workflow(ctx, q, ops, free_ops, ph, predictor_factory, policy=None, hooks=None):
    pol = policy or ctx.policy
    res = _base_result()
    predictor = predictor_factory()
    if hooks and hooks.get("predictor_exception"):
        def predictor(problem):  # noqa: F811 -- forced-failure scenario
            raise RuntimeError("injected predictor failure (forced scenario)")
    with ph("workflow"):
        tr = fw.run_free_workflow(mu_of(q), EuropeanPutPayoff(strike=q["K"]), ctx.mesh_info, ops,
                                  time_grid(q, ctx.n_steps), predictor, pol, query_id=q["query_id"],
                                  free_ops=free_ops)
    for k, v in tr.timings_s.items():  # phase breakdown, already inside "workflow"
        ph.t[f"wf_{k}"] = v
    status = _WORKFLOW_STATUS_MAP.get(tr.final_status)
    last = tr.fallback_certificate or tr.correction_certificate or tr.candidate_certificate
    if tr.final_status == fw.STATUS_NOT_CERTIFIED:
        status = OVER_BUDGET if (last is not None and last.bound_status == "valid") else NUMERICAL_FAILURE
    fc = tr.final_certificate or last
    res.update(
        final_status=status, output_delivered=tr.output_delivered,
        certificate_computed=bool(fc is not None and fc.bound_status == "valid"),
        augmented_bound=fc.augmented_certificate if fc else float("nan"),
        initial_error=fc.initial_error if fc else float("nan"),
        residual_certificate=fc.residual_certificate if fc else float("nan"),
        correction_executed=tr.correction is not None, fallback_executed=tr.fallback is not None,
        solver_converged=(tr.correction.all_converged if tr.correction else ""),
        qualification=("certificate_qualified_nonconverged"
                       if tr.final_status == fw.STATUS_ACCEPTED_CORRECTED_NONCONVERGED else ""),
        coefficients=tr.delivered_coefficients, workflow_status=tr.final_status,
        pcg_iterations_total=sum(sum(c.pcg_iterations_per_step) for c in (tr.candidate_certificate,
                                 tr.correction_certificate, tr.fallback_certificate) if c is not None),
        gmres_iterations_total=sum(tr.correction.iterations_per_step) if tr.correction else 0,
        candidate_augmented_bound=(tr.candidate_certificate.augmented_certificate
                                   if tr.candidate_certificate else float("nan")),
    )
    if tr.fallback is not None:
        # verify the fallback actually ran: a complete march record and its own certificate
        ran = (not tr.fallback.aborted and len(tr.fallback.converged_per_step) == ctx.n_steps
               and tr.fallback_certificate is not None)
        res["fallback_verified_ran"] = ran
        if status == QUALIFIED_FALLBACK and not ran:
            res.update(final_status=NUMERICAL_FAILURE, output_delivered=False,
                       qualification="fallback reported but not verified to have run")
    res["problem"] = None
    return res


def m_neural_free_workflow(ctx, q, ops, free_ops, ph, policy=None, hooks=None):
    return _workflow(ctx, q, ops, free_ops, ph,
                     lambda: fw.neural_predictor(ctx.model, ctx.stats, time_grid(q, ctx.n_steps)), policy, hooks)


def _pod_method(rank):
    def m(ctx, q, ops, free_ops, ph, policy=None, hooks=None):
        return _workflow(ctx, q, ops, free_ops, ph, lambda: pf.free_pod_predictor(ctx.pod[rank]), policy, hooks)
    return m


TRAJECTORY_METHODS: dict[str, Callable] = {
    "reference_trusted": m_reference_trusted,
    "free_direct_lu": m_free_direct_lu,
    "free_cold_gmres": m_free_cold_gmres,
    "neural_warm_gmres": m_neural_warm_gmres,
    "neural_direct_riesz": m_neural_direct_riesz,
    "neural_free_workflow": m_neural_free_workflow,
    "pod32_free_workflow": _pod_method(32),
    "pod48_free_workflow": _pod_method(48),
}


def basis_id_for(ctx: Ctx, method: str) -> str:
    if method.startswith("pod"):
        return ctx.basis_ids[int(method[3:5])]
    if method.startswith("neural"):
        return ctx.basis_ids["deeponet"]
    return ""


# ---------------------------------------------------------------- stored reference (evaluation only)

def stored_reference(ctx: Ctx, q: dict) -> dict:
    """Trusted solve, projected, with its OWN augmented certificate (allowance).
    Timed separately as evaluation-only verification cost."""
    t0 = time.perf_counter()
    ops = OperatorCache(ctx.mesh, ctx.mesh.x_min, ctx.mesh.x_max)
    traj = solve_cgmy(mu_of(q), EuropeanPutPayoff(strike=q["K"]), ctx.mesh, TimeGrid(n_steps=ctx.n_steps),
                      mode="toeplitz", store_forcing=True)
    prob = build_free_problem(mu_of(q), EuropeanPutPayoff(strike=q["K"]), ctx.mesh_info, ops, traj.time_grid,
                              method_id=METHOD_CONSTRAINED_FREE_CN_V1, f_bar_raw=traj.forcing_vectors,
                              forcing_source=FORCING_SOURCE_STORED, schemes=traj.solver_metadata.scheme_per_step)
    const = free_stability_constants(prob, coercivity_strategy=ctx.policy.coercivity_strategy, max_n_dof=ctx.max_n_dof)
    z = traj.coefficients[:, prob.fs.free_idx]
    c = prob.lift(z)
    rec = fw.certify_free_trajectory(prob, c, alpha=const.alpha_lower_estimate,
                                     pcg_iterations=ctx.policy.max_pcg_iterations,
                                     preconditioner=ctx.policy.preconditioner, tolerance=ctx.policy.tolerance)
    xs = spots_x(q, ctx.cfg)
    betas = [beta_I(prob, x) for x in xs]
    prices = [float(raw_point_functional(ctx.mesh_info, x) @ c[-1]) for x in xs]
    if rec.bound_status != "valid":
        raise RuntimeError(f"stored reference for {q['query_id']} has invalid certificate: {rec.bound_status}")
    return dict(prob=prob, z=z, c=c, eta=rec.augmented_certificate, betas=betas, prices=prices, xs=xs,
                norm_max=max(prob.mass_norm(v) for v in z), seconds=time.perf_counter() - t0,
                sha=rec.trajectory_sha256)


# ---------------------------------------------------------------- one run -> rows

def run_trajectory_method(ctx, method, q, mode, rep, tpl, ref, *, policy=None, hooks=None, scenario=""):
    ph = Phases()
    t0 = time.perf_counter()  # cache creation is INSIDE the method's timer (cold: full mesh assembly)
    with ph("setup_operator_cache"):
        if mode == "cold":
            ops, free_ops = ctx.caches.cold()
        else:
            ops, free_ops = ctx.caches.from_template(tpl)
    ev = cache_evidence(ops, free_ops, mode)
    try:
        with contextlib.ExitStack() as stack:
            for patch in (hooks or {}).get("patches", []):
                stack.enter_context(patch())
            res = TRAJECTORY_METHODS[method](ctx, q, ops, free_ops, ph, policy=policy, hooks=hooks)
    except Exception as exc:  # every attempt is kept
        res = _base_result()
        res.update(final_status=NUMERICAL_FAILURE, exception=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc(limit=3))
    total = time.perf_counter() - t0
    ev["toeplitz_entries_at_end"] = len(ops._toeplitz_cache)

    # price evaluation (terminal-price workload only, timed separately)
    price = None
    pt0 = time.perf_counter()
    if res["output_delivered"]:
        c_last = res["coefficients"][-1]
        xs = spots_x(q, ctx.cfg)
        if mode == "cold":
            prob = res.get("problem")
            if prob is None:  # workflow methods: rebuild only the mesh-level pieces needed
                fo = FreeOperatorCache(ops, free_space_for_mesh(ctx.mesh_info))
                idx = fo.fs.free_idx
                phis = [raw_point_functional(ctx.mesh_info, x) for x in xs]
                betas = [float(np.sqrt(p[idx] @ fo.apply_M_inv(p[idx]))) for p in phis]
            else:
                phis = [raw_point_functional(ctx.mesh_info, x) for x in xs]
                betas = [beta_I(prob, x) for x in xs]
        else:
            phis = [tpl["phis"][x] for x in xs]
            betas = [tpl["betas"][x] for x in xs]
        price = dict(prices=[float(p @ c_last) for p in phis], betas=betas)
    t_price = time.perf_counter() - pt0

    common = dict(query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1,
                  coordinate_space=KNOWN_METHODS[METHOD_CONSTRAINED_FREE_CN_V1]["coordinate_space"],
                  solver_method=method, basis_id=basis_id_for(ctx, method), repetition=rep, scenario=scenario,
                  solver_converged=res["solver_converged"], certificate_computed=res["certificate_computed"],
                  augmented_bound=res["augmented_bound"], initial_error=res["initial_error"],
                  residual_certificate=res["residual_certificate"],
                  correction_executed=res["correction_executed"], fallback_executed=res["fallback_executed"],
                  fallback_verified_ran=res["fallback_verified_ran"], output_delivered=res["output_delivered"],
                  final_status=res["final_status"], qualification=res["qualification"],
                  workflow_status=res.get("workflow_status", ""),
                  candidate_augmented_bound=res.get("candidate_augmented_bound", float("nan")),
                  exception=res["exception"],
                  pcg_iterations_total=res.get("pcg_iterations_total", 0),
                  gmres_iterations_total=res.get("gmres_iterations_total", 0),
                  returned_coefficients_hash=(rm.sha256_array(res["coefficients"]) if res["output_delivered"] else ""),
                  **ev)

    # trajectory target status
    tol = ctx.cfg["workloads"]["trajectory"]["tolerance"]
    traj_err = float("nan")
    emp = ""
    if res["output_delivered"]:
        z = res["coefficients"][:, ref["prob"].fs.free_idx]
        traj_err = max(ref["prob"].mass_norm(z[n] - ref["z"][n]) for n in range(ctx.n_steps + 1))
        emp = bool(traj_err + ref["eta"] <= tol)
    verified = bool(res["output_delivered"] and res["certificate_computed"] and res["augmented_bound"] <= tol)
    tstat = summary.target_status(res["output_delivered"], verified, emp)
    traj_row = dict(common, workload="trajectory",
                    target_type=ctx.cfg["workloads"]["trajectory"]["target_type"], tolerance=tol,
                    target_verified=verified, empirically_within_target=emp, target_status=tstat,
                    total_time_s=total)
    traj_acc = dict(query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1, solver_method=method,
                    repetition=rep, scenario=scenario, cache_mode=mode,
                    max_mass_error_vs_stored_reference=traj_err,
                    E_rel_vs_stored_reference=(traj_err / ref["norm_max"] if ref["norm_max"] > 0 else float("nan")),
                    stored_reference_allowance_eta=ref["eta"])

    # terminal-price target status (spots aggregated within the query: max)
    budget = ctx.cfg["workloads"]["terminal_price"]["budget_dollars"]
    p_err = p_bound = float("nan")
    p_emp = ""
    if price is not None:
        errs = [abs(p - r) for p, r in zip(price["prices"], ref["prices"])]
        allow = [b * ref["eta"] for b in ref["betas"]]
        p_err = max(errs)
        p_bound = max(b * res["augmented_bound"] for b in price["betas"]) if res["certificate_computed"] else float("nan")
        p_emp = bool(max(e + a for e, a in zip(errs, allow)) <= budget)
    p_ver = bool(price is not None and res["certificate_computed"] and p_bound <= budget)
    price_row = dict(common, workload="terminal_price",
                     target_type=ctx.cfg["workloads"]["terminal_price"]["target_type"], tolerance=budget,
                     price_bound_dollars=p_bound, target_verified=p_ver, empirically_within_target=p_emp,
                     target_status=summary.target_status(price is not None, p_ver, p_emp),
                     total_time_s=total + t_price)
    price_acc = dict(query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1, solver_method=method,
                     repetition=rep, scenario=scenario, cache_mode=mode,
                     max_abs_price_error_vs_stored_reference=p_err,
                     stored_reference_allowance_dollars=max(b * ref["eta"] for b in ref["betas"]),
                     prices=json.dumps(price["prices"]) if price else "")
    phases = dict(ph.t)
    timing_traj = dict(query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1, solver_method=method,
                       repetition=rep, scenario=scenario, cache_mode=mode, workload="trajectory",
                       phases_json=json.dumps(phases, sort_keys=True), total_time_s=total)
    timing_price = dict(timing_traj, workload="terminal_price", price_eval_s=t_price, total_time_s=total + t_price)
    timing_traj["price_eval_s"] = 0.0
    return traj_row, price_row, traj_acc, price_acc, timing_traj, timing_price


def run_cos(ctx, q, rep, mode, ref):
    t0 = time.perf_counter()
    exc = ""
    prices = None
    try:
        mkt = CGMYMarketParams(C=q["C"], G=q["G"], M=q["M"], Y=q["Y"], r=q["r"], T=q["T"])
        cc = ctx.cfg["methods"]["cos"]
        ab = truncation_range(mkt, L=cc["L"])
        prices = [cos_put_price(mkt, S0=math.exp(x), K=q["K"], N=cc["N"], trunc_range=ab, L=cc["L"])["price"]
                  for x in spots_x(q, ctx.cfg)]
    except Exception as e:
        exc = f"{type(e).__name__}: {e}"
    total = time.perf_counter() - t0
    budget = ctx.cfg["workloads"]["terminal_price"]["budget_dollars"]
    delivered = prices is not None
    emp, err = "", float("nan")
    if delivered:
        errs = [abs(p - r) for p, r in zip(prices, ref["prices"])]
        err = max(errs)
        emp = bool(max(e + b * ref["eta"] for e, b in zip(errs, ref["betas"])) <= budget)
    row = dict(query_id=q["query_id"], method_id=METHOD_INDEPENDENT_COS_V1,
               coordinate_space=KNOWN_METHODS[METHOD_INDEPENDENT_COS_V1]["coordinate_space"],
               solver_method="cos", basis_id="", repetition=rep, scenario="", solver_converged="",
               certificate_computed=False, augmented_bound=float("nan"), initial_error=float("nan"),
               residual_certificate=float("nan"), correction_executed=False, fallback_executed=False,
               fallback_verified_ran="", output_delivered=delivered,
               final_status=DELIVERED_NO_CERTIFICATE if delivered else NUMERICAL_FAILURE,
               qualification="no certificate: target can only be empirically checked", workflow_status="",
               candidate_augmented_bound=float("nan"), exception=exc, pcg_iterations_total=0,
               gmres_iterations_total=0, returned_coefficients_hash="",
               cache_mode=mode, cache_serial="", toeplitz_entries_at_start="", mass_I_factor_prebuilt="",
               toeplitz_entries_at_end="", workload="terminal_price",
               target_type=ctx.cfg["workloads"]["terminal_price"]["target_type"], tolerance=budget,
               price_bound_dollars=float("nan"), target_verified=False, empirically_within_target=emp,
               target_status=summary.target_status(delivered, False, emp), total_time_s=total)
    acc = dict(query_id=q["query_id"], method_id=METHOD_INDEPENDENT_COS_V1, solver_method="cos", repetition=rep,
               scenario="", cache_mode=mode, max_abs_price_error_vs_stored_reference=err,
               stored_reference_allowance_dollars=max(b * ref["eta"] for b in ref["betas"]),
               prices=json.dumps(prices) if prices else "")
    timing = dict(query_id=q["query_id"], method_id=METHOD_INDEPENDENT_COS_V1, solver_method="cos", repetition=rep,
                  scenario="", cache_mode=mode, workload="terminal_price", phases_json=json.dumps({"cos": total}),
                  price_eval_s=0.0, total_time_s=total)
    return row, acc, timing


# ---------------------------------------------------------------- forced-failure scenarios

def forced_scenarios(ctx: Ctx) -> list[dict]:
    p = ctx.policy

    @contextlib.contextmanager
    def uncertifiable_gate():
        orig = fw.free_stability_constants

        def fake(problem, **kw):
            c = orig(problem, **kw)
            return dataclasses.replace(c, alpha_lower_estimate=float("nan"), is_spd=False)

        fw.free_stability_constants = fake
        try:
            yield
        finally:
            fw.free_stability_constants = orig

    @contextlib.contextmanager
    def correction_exception():
        orig = constrained_mod.free_cn_step

        def boom(*a, **k):
            raise RuntimeError("injected correction failure (forced scenario)")

        constrained_mod.free_cn_step = boom
        try:
            yield
        finally:
            constrained_mod.free_cn_step = orig

    nonconv = dataclasses.replace(p, gmres_maxiter=1, gmres_restart=2)
    return [
        dict(name="F1_gmres_nonconvergence_with_fallback", method="neural_free_workflow", policy=nonconv,
             expect=QUALIFIED_FALLBACK),
        dict(name="F2_gmres_nonconvergence_no_fallback", method="neural_free_workflow",
             policy=dataclasses.replace(nonconv, fallback="none"), expect=OVER_BUDGET),
        dict(name="F3_predictor_exception", method="neural_free_workflow", policy=p,
             hooks={"predictor_exception": True}, expect=NUMERICAL_FAILURE),
        dict(name="F4_forced_uncertifiable_gate", method="neural_free_workflow", policy=p,
             hooks={"patches": [uncertifiable_gate]}, expect=UNCERTIFIABLE),
        dict(name="F5_correction_exception_with_fallback", method="neural_free_workflow", policy=p,
             hooks={"patches": [correction_exception]}, expect=QUALIFIED_FALLBACK),
        dict(name="F6_correction_exception_no_fallback", method="neural_free_workflow",
             policy=dataclasses.replace(p, fallback="none"), hooks={"patches": [correction_exception]},
             expect=NUMERICAL_FAILURE),
        dict(name="F7_direct_riesz_bound_above_target", method="neural_direct_riesz", policy=p, expect=OVER_BUDGET),
        dict(name="F8_gmres_nonconvergence_cold_solver", method="free_cold_gmres", policy=nonconv,
             expect=OVER_BUDGET),
    ]


# ---------------------------------------------------------------- main

def load_ctx(cfg, wf_cfg) -> Ctx:
    mesh, tg = load_mesh_and_time_grid(rm.BASE_CONFIG)
    mesh_info = build_mesh_info(mesh)
    box = load_parameter_box(rm.BASE_CONFIG)
    from v5_g1_traces import load_deeponet

    model = load_deeponet(mesh, tg, box, box.strike)
    fo = FreeOperatorCache(OperatorCache(mesh, mesh.x_min, mesh.x_max), free_space_for_mesh(mesh_info))
    art = REPO_ROOT / wf_cfg["pod"]["artifact"]
    if rm.sha256_file(art) != wf_cfg["pod"]["artifact_sha256"]:
        raise SystemExit("POD artifact hash mismatch")
    pod, ids = {}, {"deeponet": f"deeponet_final.pt:{rm.sha256_file(DEEPONET)[:12]}"}
    for r in cfg["methods"]["pod_ranks"]:
        b, meta = pf.load_pod_artifact(art, fo, rank=r, expected_basis_type=pf.BASIS_MASS_WEIGHTED)
        pod[r] = b
        ids[r] = f"{meta['basis_type']}:r{r}:{wf_cfg['pod']['artifact_sha256'][:12]}"
    return Ctx(cfg=cfg, wf_cfg=wf_cfg, policy=fw.FreeWorkflowPolicy.from_config(wf_cfg), mesh=mesh,
               mesh_info=mesh_info, n_steps=tg.n_steps, model=model, stats=stats_from_box(box), pod=pod,
               basis_ids=ids, caches=Caches(mesh, mesh_info))


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--approved-full-campaign", action="store_true")
    ap.add_argument("--modes", default="cold,reuse")
    ap.add_argument("--resume", type=Path, default=None,
                    help="continue an interrupted run in this directory (skips completed keys)")
    args = ap.parse_args(argv)
    if args.full and not args.approved_full_campaign:
        raise SystemExit("--full (Q_bench120) needs explicit user approval: pass --approved-full-campaign")
    if any(os.environ.get(v) != "1" for v in rm.THREAD_ENV_VARS):
        raise SystemExit(f"set {rm.THREAD_ENV_VARS}=1 for timings")
    torch.set_num_threads(1)

    cfg = yaml.safe_load(BENCH_CONFIG.read_text())
    resolved = rm.resolve_config()
    wf_cfg = resolved["workflow_config"]
    pops = rm.declare_populations()
    if args.smoke:
        ids = [i for grp in cfg["populations"]["smoke"].values() for i in grp]
        allq = {r["query_id"]: r for p in ("bench120", "audit459") for r in pops[p]["rows"]}
        queries = [allq[i] for i in ids]
        group_of = {i: g for g, lst in cfg["populations"]["smoke"].items() for i in lst}
        label = "gate_d_v5_smoke"
    else:
        queries = pops["bench120"]["rows"]
        group_of = {q["query_id"]: "bench120" for q in queries}
        label = "gate_d_v5_full"
    modes = args.modes.split(",")

    if args.resume is not None:
        run_dir = args.resume
        run_id = run_dir.name
        if not (run_dir / "run.log").exists():
            raise SystemExit(f"--resume: {run_dir} is not a harness run directory")
        session = sum(1 for line in (run_dir / "run.log").read_text().splitlines() if "session_start" in line)
    else:
        # the mode is part of the run id: separate cold/reuse runs started in the
        # same second must not collide (they did on first launch; nothing was lost)
        run_id = rm.new_run_id(f"{label}_{'+'.join(modes)}")
        run_dir = rm.create_run_dir(run_id)
        session = 0
    log = open(run_dir / "run.log", "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"session_start={session} run_id={run_id} harness={cfg['harness_version']} modes={modes} "
        f"pid={os.getpid()} affinity={sorted(os.sched_getaffinity(0))} git_head_at_start={rm.git_state()['head']} "
        f"git_dirty_at_start={rm.git_state()['dirty']} scope={NUMERICAL_SCOPE}")
    t0 = time.perf_counter()
    ctx = load_ctx(cfg, wf_cfg)
    ctx.caches.session = session
    load_s = time.perf_counter() - t0
    methods = cfg["methods"]["trajectory"]
    qindex = {q["query_id"]: i for i, q in enumerate(queries)}
    mode_index = {"cold": 0, "reuse": 1}

    def method_order(mode: str, rep: int, qid: str) -> list[str]:
        # deterministic per (mode, repetition, query): a resumed run reproduces it
        order = list(methods) + ["cos"]
        np.random.default_rng([cfg["seed"], mode_index[mode], rep, qindex[qid]]).shuffle(order)
        return order

    def done_keys(mode: str) -> set:
        """Completed (method, query, rep): the LAST file written for a key is the
        price raw_timings row (cos: its own raw_timings). A key present in an
        earlier file but not there is a partial write -> refuse to resume."""
        def keys(path):
            p = run_dir / path
            if not p.exists():
                return set()
            with open(p, newline="") as f:
                return {(r["solver_method"], r["query_id"], r["repetition"]) for r in csv.DictReader(f)}
        full = keys(f"terminal_price_{mode}/raw_timings.csv") | keys(f"terminal_price_{mode}_cos/raw_timings.csv")
        started = keys(f"trajectory_{mode}/outcomes.csv") | keys(f"terminal_price_{mode}_cos/outcomes.csv")
        if started - full:
            raise SystemExit(f"partial rows for {sorted(started - full)[:3]} in {run_dir}; inspect before resuming")
        return full
    W = {}

    def writer(name, method_id=METHOD_CONSTRAINED_FREE_CN_V1):
        key = (name, method_id)
        if key not in W:
            W[key] = ResultGroupWriter(run_dir / name, method_id)
        return W[key]

    with forbid_legacy_calls() as guard_log:
        # stored references: evaluation only, timed separately
        refs = {}
        have_ref = writer("reference").existing_keys("stored_reference.csv", "query_id")
        for q in queries:
            refs[q["query_id"]] = stored_reference(ctx, q)
            if q["query_id"] in have_ref:
                continue  # resumed: row already archived (recomputed reference is identical, deterministic)
            writer("reference").append_rows("stored_reference.csv", [dict(
                query_id=q["query_id"], method_id=METHOD_CONSTRAINED_FREE_CN_V1, query_group=group_of[q["query_id"]],
                stored_reference="trusted solve_cgmy (toeplitz GMRES rtol 1e-10) projected onto the lifting",
                stored_reference_eta_aug=refs[q["query_id"]]["eta"],
                stored_reference_dollar_allowance_max=max(b * refs[q["query_id"]]["eta"] for b in refs[q["query_id"]]["betas"]),
                stored_reference_trajectory_sha256=refs[q["query_id"]]["sha"],
                verification_cost_s=refs[q["query_id"]]["seconds"],
                cost_role="evaluation-only (not part of any online method)")])
        say(f"stored references: {len(refs)}")

        wq = dict(cfg["warmup_query"], query_id="WARMUP", source="warmup")
        wref = stored_reference(ctx, wq)
        writer("setup").append_rows("one_time_setup.csv", [dict(
            method_id=METHOD_CONSTRAINED_FREE_CN_V1, item="neural model + POD basis file load (process-level)",
            cache_mode="both", seconds=load_s, charged_per_query=False, session=session)])
        for mode in modes:
            done = done_keys(mode)
            # warmup: own throwaway caches, warmup query only
            for m in methods:
                run_trajectory_method(ctx, m, wq, "cold", -1, None, wref)
            run_cos(ctx, wq, -1, "cold", wref)
            say(f"[{mode}] warmup done")
            tpls = {}
            if mode == "reuse":
                for m in methods:
                    t = time.perf_counter()
                    tpls[m] = ctx.caches.template(sorted({x for q in queries for x in spots_x(q, cfg)}))
                    writer("setup").append_rows("one_time_setup.csv", [dict(
                        method_id=METHOD_CONSTRAINED_FREE_CN_V1, item=f"template:{m}", cache_mode="reuse",
                        seconds=time.perf_counter() - t, charged_per_query=False, session=session)])
            for rep in range(cfg["repeats"]["timed"]):
                for q in queries:
                    for m in method_order(mode, rep, q["query_id"]):
                        if (m, q["query_id"], str(rep)) in done:
                            continue
                        ref = refs[q["query_id"]]
                        if m == "cos":
                            row, acc, tim = run_cos(ctx, q, rep, mode, ref)
                            writer(f"terminal_price_{mode}_cos", METHOD_INDEPENDENT_COS_V1).append_rows("outcomes.csv", [row])
                            writer(f"terminal_price_{mode}_cos", METHOD_INDEPENDENT_COS_V1).append_rows("accuracy.csv", [acc])
                            writer(f"terminal_price_{mode}_cos", METHOD_INDEPENDENT_COS_V1).append_rows("raw_timings.csv", [tim])
                            continue
                        tr, pr, ta, pa, tt, tp = run_trajectory_method(ctx, m, q, mode, rep, tpls.get(m), ref)
                        writer(f"trajectory_{mode}").append_rows("outcomes.csv", [tr])
                        writer(f"trajectory_{mode}").append_rows("accuracy.csv", [ta])
                        writer(f"trajectory_{mode}").append_rows("raw_timings.csv", [tt])
                        writer(f"terminal_price_{mode}").append_rows("outcomes.csv", [pr])
                        writer(f"terminal_price_{mode}").append_rows("accuracy.csv", [pa])
                        writer(f"terminal_price_{mode}").append_rows("raw_timings.csv", [tp])
                say(f"[{mode}] repeat {rep} done")

        if args.smoke:
            forced_q = [q for q in queries if q["query_id"] in ("Q_bench120_0000", "Q_audit_0000")]
            for sc in forced_scenarios(ctx):
                for q in forced_q:
                    tr, _, ta, _, tt, _ = run_trajectory_method(
                        ctx, sc["method"], q, "cold", 0, None, refs[q["query_id"]], policy=sc["policy"],
                        hooks=sc.get("hooks"), scenario=sc["name"])
                    tr["expected_status"] = sc["expect"]
                    writer("forced_failures").append_rows("outcomes.csv", [tr])
                    writer("forced_failures").append_rows("raw_timings.csv", [tt])
            say("forced-failure scenarios done")
    guard = summarize(guard_log)
    rm.write_json(run_dir / "legacy_guard.json", {"blocked": guard["blocked"], "calls": guard["calls"]})

    reusable = {
        "reused_across_queries_in_reuse_mode": [
            "OperatorCache mesh data: M_h, D_h, M_h sparse LU (mesh only)",
            "free partition and the M_I sparse LU (mesh only)",
            "point functionals phi(x_s) and beta_I(x_s) for the declared spots (mesh + spots only)",
            "neural model weights; POD basis V (process-level, loaded once in both modes)"],
        "never_reused_across_queries_or_repetitions": [
            "Toeplitz blocks K_G, K_M (depend on G, M, Y)", "A_I actions, S_I spectrum, alpha, kappa (mu)",
            "forcing f_I and lifting b(t) (mu, T)", "POD reduced operator A_r and forcing f_r (mu, T)",
            "initial target c_h^0 and network input psi (K; reusable in principle, not reused here)",
            "trusted solver internals (solve_cgmy builds its own matrices every call)",
            "COS truncation range / characteristic function (mu)"],
    }
    rm.write_json(run_dir / "setup" / "reusable_objects.json", reusable)
    manifest = rm.build_manifest(
        run_id, method_id=METHOD_CONSTRAINED_FREE_CN_V1,
        studies={"bench120_cold": "not run", "bench120_reusable": "not run",
                 "g4_smoke": "completed" if args.smoke else "not run"},
        seeds={"benchmark": cfg["seed"], **wf_cfg["seeds"]}, resolved_config=resolved, populations=pops,
        population_used=[f"{q['query_id']} ({group_of[q['query_id']]})" for q in queries],
        extra_inputs={"benchmark_config": "configs/v5_benchmark.yaml", "deeponet": "checkpoints/deeponet_final.pt",
                      "pod_artifact": wf_cfg["pod"]["artifact"]},
    )
    manifest.update(harness_version=cfg["harness_version"], benchmark_config=cfg,
                    other_method_ids_in_run={"terminal_price_*_cos": METHOD_INDEPENDENT_COS_V1},
                    loaded_models={"neural": ctx.basis_ids["deeponet"], "neural_class": type(ctx.model).__name__,
                                   "pod": {r: ctx.basis_ids[r] for r in ctx.pod},
                                   "pod_class": type(next(iter(ctx.pod.values()))).__name__,
                                   "pod_basis_type": next(iter(ctx.pod.values())).basis_type,
                                   "legacy_pod_loaded": False})
    rm.write_json(run_dir / "manifest.json", manifest)
    rm.write_json(run_dir / "resolved_config.json", resolved)
    summary.write_all(run_dir, cfg, modes)
    say("summaries written")
    return run_dir


if __name__ == "__main__":
    main()
