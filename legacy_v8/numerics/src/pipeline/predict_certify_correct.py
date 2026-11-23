"""Stage J -- the online predict-certify-correct workflow. The paper's
headline claim: a neural coefficient prediction, certified cheaply first
and only escalated to a correction when a cheap certificate fails, should
usually be much faster than a full reference Galerkin solve while still
carrying an error bound guaranteed by an exact-arithmetic theorem (evaluated
in ordinary floating point; alpha <= lambda_min(S_h) is a hypothesis).

v5: this is the ``legacy_full`` method (raw n_dof residual with zeroed
Dirichlet rows, full-S_h SPD gate, no initial-error term). The v5 production
path is ``pipeline.free_workflow`` (constrained_free_cn_v1). Field name
``alpha_certified`` below is kept for compatibility with stored traces; it
holds StabilityConstants.alpha_lower_estimate, a floating-point estimate.

Six-step decision workflow, per query (mu, payoff):

  1. predict:   c_theta = model.predict_trajectory(...), seeded by a CHEAP
                c0 (a single mass-solve, not a full time-marching solve --
                see correction.krylov.compute_initial_coefficients), then
                projected onto the exact Dirichlet BC at every time level
                via cgmy.dirichlet_bc.project_bc (Task 1, working prompt):
                the exact-IC structural trick only fixes tau=0, and step 2's
                residual is deliberately zero at Dirichlet dofs, so nothing
                downstream would otherwise ever check them. This is the one
                prediction every later step sees -- E_boundary_raw/
                E_boundary on DecisionTrace record the violation before/
                after (the latter must be ~0, at machine precision).
  2. residual:  r_theta^{n+1/2} at every step, via the SAME certification
                machinery (certification.residuals) used everywhere else in
                this project -- built on a "skeleton" GalerkinTrajectory
                that carries the predicted coefficients (see
                build_skeleton_trajectory below); no real solve is needed
                to build it.
  3. eta_mass_indicator: cheap, NON-guaranteed screening indicator (Task 2,
                working prompt -- see certification.mass_indicator's module
                docstring: no proven bound property). May ONLY reject
                early, sending a query straight to correction when it is
                clearly hopeless (eta_mass_indicator >
                mass_indicator_reject_threshold, an absolute value, off by
                default -- see the parameter docstring) -- a pure
                efficiency decision that skips step 4's PCG cost, never an
                accept/reject verdict on its own.
  4. eta_guar (quick): otherwise, a SMALL capped-iteration PCG guaranteed
                certificate -- the ONLY thing that may set accepted=True
                before correction. eta_guar <= tolerance -> certified accept.
  5. correct:   else (either eta_guar failed, or step 3 skipped straight
                here), run correction.krylov.correct_trajectory,
                WARM-STARTED by the neural prediction (never discarded --
                the whole point of this workflow) via GMRES at every CN
                step.
  6. recertify: residual on the corrected trajectory, then the guaranteed
                certificate ONLY (eta_mass_indicator_corrected is recorded
                as a diagnostic, never used to accept -- correction has
                already paid the expensive step, so a confident final
                verdict is always worth it). Accept once tolerance is met;
                if this somehow still fails (should not happen once GMRES
                has converged to rtol=1e-10), report that honestly rather
                than looping or fabricating acceptance.

Every phase is separately timed (see ``DecisionTrace``) so
``experiments/exp08_runtime.py`` can report the complete workflow's cost,
never inference alone, and compare it honestly against a full reference
solve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cgmy import _external  # noqa: F401  (must precede any cgmy_bspline import)

from cgmy_bspline import boundary as cgmy_boundary
from cgmy_bspline import grid as cgmy_grid
from cgmy_bspline.parameters import CGMYParams

from cgmy.operators import OperatorCache
from cgmy.reference_solver import (
    CGMYParameters,
    EuropeanPutPayoff,
    GalerkinTrajectory,
    Mesh,
    MeshInfo,
    SolverMetadata,
)
from certification.constants import StabilityConstants, compute_stability_constants
from certification.guaranteed_certificate import eta_guar as compute_eta_guar
from certification.guaranteed_certificate import is_accepted
from certification.mass_indicator import eta_mass_indicator as compute_eta_mass_indicator
from certification.residuals import compute_all_residuals
from certification.riesz_solver import Preconditioner
from cgmy.dirichlet_bc import boundary_error, project_bc
from correction.krylov import bc_dofs_for_mesh, compute_initial_coefficients, correct_trajectory
from data import payoff_family
from learning.normalization import NormalizationStats, normalize_mu


def build_mesh_info(mesh: Mesh) -> MeshInfo:
    """Realized mesh geometry for a manually-specified (mu-independent)
    domain, WITHOUT running any solve -- cgmy_bspline.grid.make_grid_manual
    is a cheap O(N) grid-construction call, the same one solve_cgmy itself
    uses internally when mesh.x_min/x_max are set.
    """
    if mesh.x_min is None or mesh.x_max is None:
        raise ValueError(
            "predict_certify_correct requires a manually-specified mesh "
            "domain (mesh.x_min/x_max set): the fixed-mesh convention used "
            "for training/certification throughout this project. An "
            "auto-selected (mu-dependent) domain would break "
            "cross-call/model comparability."
        )
    grid_dict = cgmy_grid.make_grid_manual(mesh.x_min, mesh.x_max, mesh.N)
    n_dof = grid_dict["N"] + mesh.p - 1
    return MeshInfo(
        N=grid_dict["N"], p=mesh.p, n_dof=n_dof, h=grid_dict["h"],
        x_min=grid_dict["x_min"], x_max=grid_dict["x_max"], nodes=grid_dict["nodes"],
    )


def build_skeleton_trajectory(
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh_info: MeshInfo,
    time_grid: np.ndarray,
    coefficients: np.ndarray,
) -> GalerkinTrajectory:
    """A GalerkinTrajectory carrying ``coefficients`` (a prediction or a
    correction, not a real solve's output), with every OTHER field a
    trajectory needs for certification computed cheaply -- no time-marching
    solve is run to build it.

    forcing_vectors is built by calling the trusted
    cgmy_bspline.boundary.boundary_forcing_vector (currently always zero,
    confirmed by direct inspection -- see docs/issues_report.md and
    CLAUDE.md Hard Invariant #2) rather than hardcoding zeros, so this stays
    correct the day that function stops being a zero-forcing approximation.
    """
    n_steps = len(time_grid) - 1
    dt = float(time_grid[1] - time_grid[0])
    bc_left, bc_right = bc_dofs_for_mesh(mesh_info)
    params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=payoff.strike, T=mu.T)

    forcing = np.empty((n_steps, mesh_info.n_dof))
    tau = 0.0
    for n in range(n_steps):
        tau_new = float(time_grid[n + 1])
        f_old = cgmy_boundary.boundary_forcing_vector(
            mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, tau, None
        )
        f_new = cgmy_boundary.boundary_forcing_vector(
            mesh_info.N, mesh_info.p, mesh_info.h, mesh_info.x_min, params, tau_new, None
        )
        forcing[n] = 0.5 * (f_old + f_new)
        tau = tau_new

    metadata = SolverMetadata(
        mode="toeplitz",
        dt=dt,
        n_steps=n_steps,
        scheme_per_step=["CN"] * n_steps,
        iterations_per_step=[0] * n_steps,
        tolerance_per_step=[None] * n_steps,
        bc_dofs=(bc_left, bc_right),
        cpu_time=0.0,
        mean_gmres_iters=None,
    )
    return GalerkinTrajectory(
        coefficients=coefficients,
        forcing_vectors=forcing,
        initial_coefficients=coefficients[0],
        time_grid=time_grid,
        parameter_vector=mu,
        payoff=payoff,
        mesh_information=mesh_info,
        solver_metadata=metadata,
    )


Branch = str
# Task 2 (working prompt): BRANCH_ACCEPTED_MASS and
# BRANCH_ACCEPTED_AFTER_CORRECTION_MASS are deliberately REMOVED, not
# renamed -- eta_mass_indicator has no proven bound property and may never
# be the sole basis for an accepted=True decision (see
# certification.mass_indicator's module docstring). Only a *_GUAR branch
# may set accepted=True.
BRANCH_ACCEPTED_GUAR_QUICK = "accepted_guar_quick"
BRANCH_ACCEPTED_AFTER_CORRECTION_GUAR = "accepted_after_correction_guar"
BRANCH_CORRECTED_BUT_NOT_CERTIFIED = "corrected_but_not_certified"
BRANCH_NOT_SPD_UNCERTIFIABLE = "not_spd_uncertifiable"
BRANCH_NEEDS_CORRECTION_NOT_RUN = "needs_correction_not_run"


@dataclass
class DecisionTrace:
    """Full per-query decision trace: which branch was taken, iteration
    counts, and a timing breakdown of every phase actually executed.
    Un-executed phases keep their timing at 0.0 and their certificate value
    at NaN -- never a fabricated number standing in for "not computed".
    """

    mu: CGMYParameters
    branch: Branch
    accepted: bool
    is_spd: bool  # alpha_certified > 0 (a floating-point lower-bound ESTIMATE; see certification.constants)

    # Task 3 (working prompt): split, never conflated -- certification-critical
    # code below is passed alpha_certified exclusively, never lambda_min_numerical.
    lambda_min_numerical: float
    alpha_certified: float
    safety_ratio: float
    coercivity_strategy: str

    eta_mass_indicator: float  # NOT a certificate -- see certification.mass_indicator
    mass_indicator_early_reject: bool  # True if step 4 (eta_guar quick) was skipped because
                                        # eta_mass_indicator was clearly too large (efficiency
                                        # only -- never causes an accept)
    eta_guar_quick: float
    eta_guar_quick_n_pcg_iterations: np.ndarray | None
    eta_final: float  # whichever certificate the final accept/reject decision used --
                       # ALWAYS an eta_guar value (or NaN), never eta_mass_indicator
    tolerance: float

    E_boundary_raw: float  # max |c_theta - c_BC| at Dirichlet dofs BEFORE projection
    E_boundary: float  # same, on the trajectory actually consumed downstream; must be ~0

    correction_ran: bool
    correction_n_iterations_per_step: list[int] | None
    correction_converged: bool | None
    eta_mass_indicator_corrected: float  # diagnostic only -- see step 6 in the module docstring
    eta_guar_corrected: float

    t_predict: float = 0.0
    t_residual: float = 0.0
    t_mass_indicator: float = 0.0
    t_riesz_certificate: float = 0.0
    t_correction: float = 0.0
    t_recertify: float = 0.0
    t_total: float = 0.0

    predicted_coefficients: np.ndarray | None = field(default=None, repr=False)
    final_coefficients: np.ndarray | None = field(default=None, repr=False)


def predict_certify_correct(
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh: Mesh,
    time_grid: np.ndarray,
    model: Any,
    stats: NormalizationStats,
    ops: OperatorCache,
    *,
    tolerance: float,
    quick_pcg_iterations: int,
    max_pcg_iterations: int,
    preconditioner: Preconditioner = "none",  # Task 9, working prompt -- see notes/09_preconditioner.md
    correction_gmres_rtol: float = 1e-10,
    correction_gmres_restart: int = 50,
    keep_coefficients: bool = False,
    run_correction: bool = True,
    mass_indicator_reject_threshold: float = float("inf"),
) -> DecisionTrace:
    """Run the full six-step online workflow for one query (mu, payoff).

    Args:
        mesh: Manually-specified-domain mesh request (mesh.x_min/x_max set)
            -- matches ``ops`` and the mesh the model was trained on.
        time_grid: (n_steps+1,) uniform time nodes, 0..mu.T (matching
            ``TimeGrid(n_steps=...)``'s implied grid).
        model: A ``learning.interface.CoefficientModel``.
        tolerance: Acceptance tolerance for eta_guar (quick) and the
            post-correction recertification's eta_guar. eta_mass_indicator
            never accepts against this (or any) tolerance (Task 2, working
            prompt) -- see ``mass_indicator_reject_threshold`` for its only
            role.
        quick_pcg_iterations: Small PCG iteration cap for step 4's cheap
            escalation (a "small number of PCG Riesz iterations" before
            resorting to correction).
        max_pcg_iterations: Larger PCG cap used only for the post-correction
            recertification (step 6), where a confident final certificate
            is worth the extra iterations since correction is already the
            expensive branch.
        keep_coefficients: If True, the predicted and final coefficient
            trajectories are kept on the returned trace (off by default so
            large batch runs, e.g. exp08/exp09, don't retain every array).
        run_correction: If False, a query that fails the quick guaranteed
            certificate (or is early-rejected by eta_mass_indicator) returns
            immediately with branch=BRANCH_NEEDS_CORRECTION_NOT_RUN instead
            of paying for the (expensive) correction step -- a cheap
            diagnostic mode for calibrating a tolerance or surveying how
            many queries WOULD need correction, without actually running
            it. Never used to skip correction in a real accept/reject
            decision.
        mass_indicator_reject_threshold: Step 3's early-reject cutoff: if
            eta_mass_indicator exceeds this ABSOLUTE value, step 4 (eta_guar
            quick) is skipped and the query goes straight toward correction
            -- a PURE efficiency knob (Task 2, working prompt).
            eta_mass_indicator has no proven relationship to eta_guar, so
            this can only route MORE queries to correction than strictly
            necessary, never fewer, and can never cause an incorrect
            accept. Deliberately NOT tied to ``tolerance``: several callers
            in this project pass extreme tolerance values for calibration/
            diagnostic purposes (e.g. exp08_runtime's tolerance=1e-12
            probing pass), and a threshold derived from tolerance would
            silently defeat those. Defaults to float("inf") (disabled --
            always compute eta_guar quick); pass a concrete value to opt
            into the efficiency gain.

    Returns:
        A DecisionTrace with the branch taken, iteration counts, and a
        phase-by-phase timing breakdown.
    """
    mesh_info = build_mesh_info(mesh)
    if ops.N != mesh_info.N or ops.p != mesh_info.p or abs(ops.x_min - mesh_info.x_min) > 1e-12:
        raise ValueError("ops (OperatorCache) must be built from the same mesh as `mesh`")

    t0 = time.perf_counter()
    c0, _, _ = compute_initial_coefficients(mu, payoff, mesh_info, ops)
    mu_tilde = normalize_mu(mu, stats)
    psi = payoff_family.european_put(payoff.strike, ops)
    pred_coeffs_raw = model.predict_trajectory(mu_tilde, psi, c0, time_grid, mu.T)
    # Task 1 (working prompt): the exact IC is enforced structurally by every
    # model, but the time-dependent Dirichlet BC at tau>0 is not -- and
    # certification.residuals zeroes exactly those rows, so nothing else
    # would ever catch a violation. Project before this prediction is
    # consumed by anything downstream (residual eval, certification,
    # correction warm-start) -- the one code path everything else sees.
    E_boundary_raw = boundary_error(pred_coeffs_raw, mu, payoff, mesh_info, time_grid)
    pred_coeffs = project_bc(pred_coeffs_raw, mu, payoff, mesh_info, time_grid)
    E_boundary = boundary_error(pred_coeffs, mu, payoff, mesh_info, time_grid)
    t_predict = time.perf_counter() - t0

    t0 = time.perf_counter()
    traj_pred = build_skeleton_trajectory(mu, payoff, mesh_info, time_grid, pred_coeffs)
    residuals = compute_all_residuals(traj_pred, mu, ops)
    t_residual = time.perf_counter() - t0

    t0 = time.perf_counter()
    const: StabilityConstants = compute_stability_constants(mu, ops)
    is_spd = const.is_spd
    dt = traj_pred.solver_metadata.dt
    eta_mass_indicator_value = float("nan")
    if is_spd:
        eta_mass_indicator_value = compute_eta_mass_indicator(residuals, ops, dt, const.alpha_certified)
    t_mass_indicator = time.perf_counter() - t0

    trace_kwargs: dict[str, Any] = dict(
        mu=mu, is_spd=is_spd,
        lambda_min_numerical=const.lambda_min_numerical, alpha_certified=const.alpha_certified,
        safety_ratio=const.safety_ratio, coercivity_strategy=const.coercivity_strategy,
        eta_mass_indicator=eta_mass_indicator_value, mass_indicator_early_reject=False,
        tolerance=tolerance,
        E_boundary_raw=E_boundary_raw, E_boundary=E_boundary,
        eta_guar_quick=float("nan"), eta_guar_quick_n_pcg_iterations=None,
        correction_ran=False, correction_n_iterations_per_step=None,
        correction_converged=None, eta_mass_indicator_corrected=float("nan"),
        eta_guar_corrected=float("nan"),
        t_predict=t_predict, t_residual=t_residual, t_mass_indicator=t_mass_indicator,
    )

    if not is_spd:
        t_total = t_predict + t_residual + t_mass_indicator
        return DecisionTrace(
            branch=BRANCH_NOT_SPD_UNCERTIFIABLE, accepted=False, eta_final=float("nan"),
            t_total=t_total,
            predicted_coefficients=pred_coeffs if keep_coefficients else None,
            final_coefficients=pred_coeffs if keep_coefficients else None,
            **trace_kwargs,
        )

    # ── Step 3: eta_mass_indicator -- may only reject early, NEVER accept ──
    # (Task 2, working prompt): eta_mass_indicator has no proven bound
    # property (certification.mass_indicator), so the only decision it may
    # make is whether step 4's PCG cost is worth paying at all -- a pure
    # efficiency choice that can route MORE queries to correction than
    # strictly necessary, never cause an incorrect accept.
    skip_quick_guar = eta_mass_indicator_value > mass_indicator_reject_threshold
    trace_kwargs["mass_indicator_early_reject"] = skip_quick_guar

    t_riesz_certificate = 0.0
    guar_quick = None
    if not skip_quick_guar:
        # ── Step 4: quick guaranteed certificate -- the ONLY way to accept ──
        t0 = time.perf_counter()
        guar_quick = compute_eta_guar(
            residuals, mu, ops, dt, const.alpha_certified,
            max_pcg_iterations=quick_pcg_iterations, preconditioner=preconditioner,
        )
        t_riesz_certificate = time.perf_counter() - t0
        trace_kwargs["eta_guar_quick"] = guar_quick.eta
        trace_kwargs["eta_guar_quick_n_pcg_iterations"] = guar_quick.n_pcg_iterations
        trace_kwargs["t_riesz_certificate"] = t_riesz_certificate

        assert guar_quick is not None and not np.isnan(guar_quick.eta), (
            "certified acceptance requires eta_guar to have actually been computed"
        )
        if is_accepted(guar_quick, tolerance):
            t_total = t_predict + t_residual + t_mass_indicator + t_riesz_certificate
            return DecisionTrace(
                branch=BRANCH_ACCEPTED_GUAR_QUICK, accepted=True, eta_final=guar_quick.eta,
                t_total=t_total,
                predicted_coefficients=pred_coeffs if keep_coefficients else None,
                final_coefficients=pred_coeffs if keep_coefficients else None,
                **trace_kwargs,
            )

    if not run_correction:
        t_total = t_predict + t_residual + t_mass_indicator + t_riesz_certificate
        eta_final = guar_quick.eta if guar_quick is not None else float("nan")
        return DecisionTrace(
            branch=BRANCH_NEEDS_CORRECTION_NOT_RUN, accepted=False, eta_final=eta_final,
            t_total=t_total,
            predicted_coefficients=pred_coeffs if keep_coefficients else None,
            final_coefficients=pred_coeffs if keep_coefficients else None,
            **trace_kwargs,
        )

    # ── Step 5: correction, warm-started by the (rejected) prediction ──
    t0 = time.perf_counter()
    correction = correct_trajectory(
        mu, payoff, mesh_info, time_grid, ops, seed_coefficients=pred_coeffs,
        warm_start=True, gmres_rtol=correction_gmres_rtol, gmres_restart=correction_gmres_restart,
    )
    t_correction = time.perf_counter() - t0
    trace_kwargs["correction_ran"] = True
    trace_kwargs["correction_n_iterations_per_step"] = correction.n_iterations_per_step
    trace_kwargs["correction_converged"] = all(correction.converged_per_step)
    trace_kwargs["t_correction"] = t_correction

    # ── Step 6: recertify the corrected trajectory -- eta_guar decides,
    # eta_mass_indicator_corrected is recorded as a diagnostic ONLY (Task 2,
    # working prompt): correction has already paid the expensive step, so a
    # confident final verdict from the guaranteed certificate is always
    # worth it -- never accept post-correction on the indicator alone. ──
    t0 = time.perf_counter()
    traj_corrected = build_skeleton_trajectory(mu, payoff, mesh_info, time_grid, correction.coefficients)
    residuals_corrected = compute_all_residuals(traj_corrected, mu, ops)
    eta_mass_indicator_corrected = compute_eta_mass_indicator(residuals_corrected, ops, dt, const.alpha_certified)
    trace_kwargs["eta_mass_indicator_corrected"] = eta_mass_indicator_corrected

    guar_final = compute_eta_guar(
        residuals_corrected, mu, ops, dt, const.alpha_certified,
        max_pcg_iterations=max_pcg_iterations, preconditioner=preconditioner,
    )
    assert not np.isnan(guar_final.eta), (
        "certified acceptance requires eta_guar to have actually been computed"
    )
    trace_kwargs["eta_guar_corrected"] = guar_final.eta
    t_recertify = time.perf_counter() - t0
    trace_kwargs["t_recertify"] = t_recertify
    t_total = t_predict + t_residual + t_mass_indicator + t_riesz_certificate + t_correction + t_recertify

    accepted = is_accepted(guar_final, tolerance)
    branch = BRANCH_ACCEPTED_AFTER_CORRECTION_GUAR if accepted else BRANCH_CORRECTED_BUT_NOT_CERTIFIED
    return DecisionTrace(
        branch=branch, accepted=accepted, eta_final=guar_final.eta,
        t_total=t_total,
        predicted_coefficients=pred_coeffs if keep_coefficients else None,
        final_coefficients=correction.coefficients if keep_coefficients else None,
        **trace_kwargs,
    )
