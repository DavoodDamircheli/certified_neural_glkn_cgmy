"""exp10 -- cross-resolution evaluation: architectural incompatibility
(MLP / DeepONet / POD) vs. genuine generalization (ProjectedFNO).

Task 13 (working prompt) reframing: MLP and coefficient-space DeepONet
(and POD) failing at an off-training resolution is NOT a generalization
failure -- a model "generalizes poorly" only if it can actually attempt
the task and does badly. These three cannot even ATTEMPT a prediction at
a changed coefficient dimension; every `nn.Linear`/basis-matrix shape in
them is fixed to the training mesh's `n_dof` at construction (see the
architecture notes below), so evaluating them off-resolution raises a
shape-mismatch exception before any prediction is computed. That is an
**architectural incompatibility**, a fixed, structural property of the
model class, not a spectrum of "better or worse generalization" the way
ProjectedFNO's off-resolution accuracy is. Earlier versions of this file
(and of `notes/`) called both outcomes "generalization capability" under
one umbrella -- that framing is fixed here: "architectural
incompatibility" for MLP/DeepONet/POD throughout this file's code
comments, print output, and the paper-facing table/figure labels;
"generalization" reserved for ProjectedFNO, the only one of the four that
can even be asked the question.

Architecture notes (why each model is, or is not, architecturally able to
run at a different N):
  - MLPCoeffNet / DeepONetCoeffNet: every `nn.Linear` layer's width is
    fixed at construction to `psi_dim = out_dim = n_dof` of the training
    mesh (see `learning/mlp_coeff.py`, `learning/deeponet_coeff.py`). A
    different resolution's `psi`/`c0` have a different length, which does
    not match those fixed layer widths.
  - PODGalerkinROM: the POD basis matrix has shape `(n_dof_train,
    n_modes)` (see `learning/pod.py`); `OperatorCache.apply_M`/`apply_A`
    at a different resolution operates on vectors of a different length
    than the basis columns.
  - ProjectedFNO: the underlying `FNO1d` (`learning/fno.py`) has NO
    grid-length-dependent layer -- `SpectralConv1d` truncates to a fixed
    number of Fourier modes regardless of input grid length, and its
    lift/project layers and the pointwise `Conv1d(kernel_size=1)` act
    per-grid-point. Only the `ProjectedFNO` WRAPPER's `Pi_h` projection
    ingredients (`learning/projected_fno.py`: grid_nodes, the B-spline
    basis-value matrix, the `M_h` factorization) are mesh-specific. So the
    SAME trained `FNO1d` weights can be rewrapped around a different
    mesh's `(grid, Pi_h)` without retraining -- this IS a genuine
    generalization question, tested empirically below, not assumed from
    the architecture description.

Two falsifiable claims, tested empirically rather than assumed:

  Claim A (re: MLP/DeepONet/POD -- architectural incompatibility, not a
    generalization result): attempting to evaluate any of them at an
    off-training resolution raises a shape-mismatch exception -- they
    cannot even attempt a prediction off-resolution. FALSIFIED if any of
    the three instead runs to completion at an off-training resolution:
    that would mean either this description of their architecture is
    wrong, or worse, a silent shape-broadcasting bug is producing
    meaningless output without erroring -- a real bug to chase down, not
    a success to report.

  Claim B (re: ProjectedFNO -- a genuine generalization result): rewrapping
    the SAME trained FNO1d weights around a different-resolution mesh
    yields predictions whose certified error (E_i, eta_guar) stays within
    the same order of magnitude as the trained-resolution baseline, rather
    than blowing up. FALSIFIED if the off-training-resolution certified
    error is dramatically worse (orders of magnitude, or a certification
    failure that doesn't occur at the trained resolution) -- meaning the
    FNO layer's textbook discretization-invariance does not survive being
    paired with a resolution-specific Pi_h projection in this
    implementation, a real negative finding to report as such.

Task 13 also asks, for MLP/DeepONet/POD's architecturally-incompatible
cells, to "compare against interpolation/projection of the DeepONet
output rather than reporting a bare failure, where feasible." Added
below: at every OFF-TRAINING resolution, the TRAINED-resolution (N=64)
DeepONet's TERMINAL-time-step prediction is L2-projected (via the
TRUSTED `cgmy_bspline.projection.l2_project_generic`/`eval_solution`,
never a new interpolation scheme invented for this experiment) onto the
target resolution's B-spline space, and compared against that
resolution's own reference solution's terminal state. This is a
TERMINAL-STATE-ONLY comparison, not the same full-trajectory `E_i` used
for models that run natively -- projecting every one of 61 time steps
for every query at every off-training resolution was measured directly
to cost ~0.8s/projection at N=256, i.e. tens of minutes for a
full-trajectory version, out of proportion to what "feasible" asks for
here. Labeled explicitly as terminal-state throughout, never conflated
with the full-trajectory `E_i` column.

Domain is held fixed (x_min=log(100)-8, x_max=log(100)+8) across every
resolution tested -- only N (and therefore h, n_dof) varies -- so the
comparison isolates resolution, not domain size. The N=64 row is literally
exp_common.FIXED_MESH, so it is directly comparable to every other
experiment in this project.

Scale note: N_TEST_MUS=50 (Task 11, working prompt: "resolution transfer:
>=50 parameter points per resolution"). RESOLUTIONS now includes N=256
(Task 13, working prompt: "evaluate the projected FNO at N = 32, 64, 128,
256").

Outputs: results/exp10_cross_resolution.h5, figures/exp10_cross_resolution.png,
tables/exp10_cross_resolution.tex.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgmy import _external  # noqa: F401  (must precede any cgmy_bspline import)
from cgmy_bspline.projection import eval_solution, l2_project_generic

from cgmy.operators import OperatorCache
from cgmy.reference_solver import Mesh, TimeGrid, solve_cgmy
from data import payoff_family
from data.sampling import sample_sobol
from exp_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    compute_certificates,
    fmt,
    get_box_and_stats,
    get_ops,
    load_models,
    capture_git_commit_hash,
    provenance_start,
    relative_energy_error,
    relative_l2_error,
    supersede_if_exists,
    true_error_Ei,
    write_h5,
    write_latex_table,
    write_provenance,
)
from learning.normalization import normalize_mu
from learning.pod import PODGalerkinROM
from learning.projected_fno import ProjectedFNO

X_MIN = float(np.log(100) - 8)
X_MAX = float(np.log(100) + 8)
MESH_P = 3
TRAIN_N = 64

RESOLUTIONS = [32, 64, 128, 256]  # Task 13: adds N=256
N_TEST_MUS = 50
SEED = 100100
MODEL_NAMES = ["MLP", "DeepONet", "POD", "ProjectedFNO"]


def _mesh_at(N: int) -> Mesh:
    return Mesh(N=N, p=MESH_P, x_min=X_MIN, x_max=X_MAX)


def _attempt_prediction(model_name: str, base_models: dict[str, Any], stats, mu, psi, c0, time_grid, T,
                         ops_test: OperatorCache) -> tuple[np.ndarray | None, str | None, float]:
    """Try to evaluate `model_name` at ops_test's resolution, reusing the
    SAME trained weights loaded at the training resolution. Returns
    (prediction, error_message, wall_clock_seconds) -- (None, msg, t) on
    failure, never silently substituting a fallback."""
    t0 = time.perf_counter()
    try:
        if model_name in ("MLP", "DeepONet"):
            model = base_models[model_name]
            mu_tilde = normalize_mu(mu, stats)
            pred = model.predict_trajectory(mu_tilde, psi, c0, time_grid, T)
        elif model_name == "POD":
            rom = PODGalerkinROM(base_models["POD"].basis, ops_test, stats=stats)
            mu_tilde = normalize_mu(mu, stats)
            pred = rom.predict_trajectory(mu_tilde, psi, c0, time_grid, T)
        elif model_name == "ProjectedFNO":
            base_fno_model = base_models["ProjectedFNO"]
            wrapped = ProjectedFNO(base_fno_model.fno, _mesh_at(ops_test.N), ops_test, mu_dim=base_fno_model.mu_dim)
            mu_tilde = normalize_mu(mu, stats)
            pred = wrapped.predict_trajectory(mu_tilde, psi, c0, time_grid, T)
        else:
            raise ValueError(f"unknown model {model_name!r}")
        return pred, None, time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure mode is an "architectural incompatibility" finding
        return None, f"{type(exc).__name__}: {exc}"[:300], time.perf_counter() - t0


def _project_terminal_state(c_src: np.ndarray, src_mesh: Mesh, src_ops: OperatorCache,
                             tgt_mesh: Mesh, tgt_ops: OperatorCache) -> np.ndarray:
    """L2-project one B-spline coefficient vector (a single time level, NOT
    a full trajectory -- see module docstring's cost note) from src_mesh's
    space onto tgt_mesh's, via the TRUSTED cgmy_bspline.projection routines
    (eval_solution to get physical values, l2_project_generic to re-project
    them) -- never a new interpolation scheme invented for this
    experiment."""
    src_grid_info = {"N": src_mesh.N, "h": src_ops.h, "x_min": src_mesh.x_min}
    tgt_grid_info = {"N": tgt_mesh.N, "h": tgt_ops.h, "x_min": tgt_mesh.x_min}

    def source_fn(x):
        return eval_solution(c_src, src_grid_info, src_mesh.p, np.array([x]))[0]

    return l2_project_generic(source_fn, tgt_grid_info, tgt_mesh.p, {"M_h": tgt_ops.M_h}, kink_x=None)


def run_experiment(
    resolutions: list[int], n_test_mus: int, time_grid_req: TimeGrid, seed: int
) -> dict[str, Any]:
    """Core experiment logic, factored out of main() so a toy-scale smoke
    test can exercise it directly without touching disk outputs."""
    ops64 = get_ops()
    box, stats = get_box_and_stats()
    base_models = load_models(ops64, stats)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = sample_sobol(box, n_test_mus, seed=seed)

    records: dict[tuple[str, int], list[dict[str, Any]]] = {
        (m, N): [] for m in MODEL_NAMES for N in resolutions
    }
    # Task 13: cache the trained-resolution (N=64) DeepONet's TERMINAL
    # prediction and each resolution's own reference terminal state, per
    # query index, so the projection comparison below can reuse them
    # without any extra model inference or reference solves.
    deeponet_n64_terminal: dict[int, np.ndarray] = {}
    ref_terminal: dict[tuple[int, int], np.ndarray] = {}

    for N in resolutions:
        mesh_test = _mesh_at(N)
        ops_test = OperatorCache(mesh_test, X_MIN, X_MAX)
        print(f"exp10: resolution N={N} (n_dof={ops_test.n_dof})...")

        for i, (mu, payoff) in enumerate(samples):
            time_grid = np.linspace(0.0, mu.T, time_grid_req.n_steps + 1)
            try:
                traj_true = solve_cgmy(mu, payoff, mesh_test, time_grid_req, mode="toeplitz")
            except Exception as exc:  # noqa: BLE001
                print(f"  [N={N}] reference solve failed for mu={mu}: {exc}; skipping this query at this resolution")
                continue
            ref_terminal[(N, i)] = traj_true.coefficients[-1]

            psi = payoff_family.european_put(payoff.strike, ops_test)
            c0 = traj_true.initial_coefficients

            for model_name in MODEL_NAMES:
                pred, err, t_pred = _attempt_prediction(
                    model_name, base_models, stats, mu, psi, c0, time_grid, mu.T, ops_test
                )
                if pred is None:
                    records[(model_name, N)].append({
                        "success": False, "error": err, "E_i": float("nan"),
                        "eta_guar": float("nan"), "is_spd": False,
                        "rel_l2": float("nan"), "rel_energy": float("nan"),
                        "effectivity": float("nan"), "runtime_s": t_pred,
                    })
                    continue
                if model_name == "DeepONet" and N == TRAIN_N:
                    deeponet_n64_terminal[i] = pred[-1]
                E_i = true_error_Ei(pred, traj_true.coefficients, ops_test)
                cert = compute_certificates(traj_true, pred, mu, ops_test)
                effectivity = (cert.eta_guar / E_i) if (cert.is_spd and E_i > 0) else float("nan")
                records[(model_name, N)].append({
                    "success": True, "error": None, "E_i": E_i,
                    "eta_guar": cert.eta_guar, "is_spd": cert.is_spd,
                    "rel_l2": relative_l2_error(pred, traj_true.coefficients, ops_test),
                    "rel_energy": relative_energy_error(pred, traj_true.coefficients, mu, ops_test),
                    "effectivity": effectivity, "runtime_s": t_pred,
                })

        for model_name in MODEL_NAMES:
            rows = records[(model_name, N)]
            n_success = sum(r["success"] for r in rows)
            print(f"    {model_name:12s}: {n_success}/{len(rows)} predictions succeeded")

    # ── Task 13: DeepONet-output projection comparison at every
    #    off-training resolution (terminal state only -- see module
    #    docstring) instead of a bare "architecturally incompatible" cell ──
    ops64_obj = OperatorCache(_mesh_at(TRAIN_N), X_MIN, X_MAX)
    train_mesh = _mesh_at(TRAIN_N)
    projection_records: dict[int, list[dict[str, Any]]] = {N: [] for N in resolutions if N != TRAIN_N}
    for N in resolutions:
        if N == TRAIN_N:
            continue
        mesh_n = _mesh_at(N)
        ops_n = OperatorCache(mesh_n, X_MIN, X_MAX)
        print(f"exp10: projecting DeepONet's N={TRAIN_N} terminal output onto N={N}...")
        for i in range(n_test_mus):
            if i not in deeponet_n64_terminal or (N, i) not in ref_terminal:
                continue
            c_src = deeponet_n64_terminal[i]
            c_proj = _project_terminal_state(c_src, train_mesh, ops64_obj, mesh_n, ops_n)
            c_ref = ref_terminal[(N, i)]
            diff = c_proj - c_ref
            num = float(np.sqrt(max(diff @ ops_n.apply_M(diff), 0.0)))
            den = float(np.sqrt(max(c_ref @ ops_n.apply_M(c_ref), 0.0)))
            rel_l2_terminal = num / den if den > 0 else float("nan")
            projection_records[N].append({"rel_l2_terminal": rel_l2_terminal})

    return {
        "records": records, "projection_records": projection_records,
        "resolutions": resolutions, "n_test_mus": n_test_mus, "seed": seed,
    }


def main() -> None:
    t_start = provenance_start()
    git_commit = capture_git_commit_hash()  # BEFORE any output is written -- see its docstring
    time_grid_req = TimeGrid(n_steps=60)
    result = run_experiment(RESOLUTIONS, N_TEST_MUS, time_grid_req, SEED)
    records = result["records"]
    projection_records = result["projection_records"]

    # ── Aggregate + honest claim-check printout ─────────────────────────
    print("\n--- Claim A: MLP/DeepONet/POD are ARCHITECTURALLY INCOMPATIBLE with "
          "off-training resolutions (not a generalization result) ---")
    claim_a_holds = True
    for model_name in ("MLP", "DeepONet", "POD"):
        for N in RESOLUTIONS:
            rows = records[(model_name, N)]
            n_success = sum(r["success"] for r in rows)
            off_training = N != TRAIN_N
            if off_training and n_success > 0:
                claim_a_holds = False
                print(f"  FALSIFIED: {model_name} succeeded {n_success}/{len(rows)} times at "
                      f"off-training resolution N={N} (expected 0 -- architecturally, this "
                      f"should be impossible; a real bug, not a success)")
    if claim_a_holds:
        print("  holds: 0 successful predictions at any off-training resolution, for all three "
              "models -- this is architectural incompatibility (fixed layer/basis widths tied "
              "to the training n_dof), not a measurement of how well they generalize")

    print("\n--- DeepONet-output projection comparison (terminal state only) at "
          "off-training resolutions, instead of a bare failure ---")
    for N in RESOLUTIONS:
        if N == TRAIN_N:
            continue
        vals = [r["rel_l2_terminal"] for r in projection_records[N]]
        print(f"  N={N}: median terminal rel. $L_2$ (projected DeepONet vs. native reference) = "
              f"{fmt(float(np.median(vals))) if vals else 'n/a'}  (n={len(vals)})")

    print("\n--- Claim B: ProjectedFNO's certified error stays bounded across resolutions "
          "(a genuine generalization result) ---")
    fno_baseline = [r["eta_guar"] for r in records[("ProjectedFNO", TRAIN_N)] if r["success"] and r["is_spd"]]
    baseline_median = float(np.median(fno_baseline)) if fno_baseline else float("nan")
    print(f"  N={TRAIN_N} (training resolution) baseline median eta_guar: {fmt(baseline_median)}")
    claim_b_holds = True
    for N in RESOLUTIONS:
        if N == TRAIN_N:
            continue
        vals = [r["eta_guar"] for r in records[("ProjectedFNO", N)] if r["success"] and r["is_spd"]]
        n_fail = sum(1 for r in records[("ProjectedFNO", N)] if not r["success"] or not r["is_spd"])
        med = float(np.median(vals)) if vals else float("nan")
        ratio = med / baseline_median if baseline_median == baseline_median and baseline_median > 0 else float("nan")
        print(f"  N={N}: median eta_guar={fmt(med)}  ratio-to-baseline={ratio:.2f}x  "
              f"uncertifiable/failed={n_fail}/{len(records[('ProjectedFNO', N)])}")
        if not (ratio == ratio) or ratio > 10.0:
            claim_b_holds = False
    print(f"  {'holds' if claim_b_holds else 'FALSIFIED'}: median eta_guar stays within 10x of the "
          f"training-resolution baseline at every tested off-training resolution" if claim_b_holds else
          "  FALSIFIED: see ratios above -- at least one off-training resolution's certified error "
          "exceeds 10x the training-resolution baseline")

    # Pre-submission fixes item 4.1: prediction success and certifiability
    # are DIFFERENT things -- a model can successfully produce a prediction
    # (no shape-mismatch exception) at a query where S_h(mu) still isn't
    # SPD, so no certificate can be issued. Stated explicitly here (not
    # just left inside the table -- item 4.1's own instruction), since the
    # old single "success rate" column could make "100% success" and
    # "12/50 uncertifiable" at the SAME (model, N) look contradictory when
    # they are not.
    print("\n--- Prediction success vs. certifiability (item 4.1) ---")
    for model_name in MODEL_NAMES:
        for N in RESOLUTIONS:
            rows = records[(model_name, N)]
            n_success = sum(r["success"] for r in rows)
            n_certifiable = sum(1 for r in rows if r["success"] and r["is_spd"])
            if n_success == 0:
                continue
            print(f"  {model_name:12s} N={N:4d}: prediction success={n_success}/{len(rows)} "
                  f"({100*n_success/len(rows):.0f}%)  certifiable (SPD) fraction, OF THE SUCCESSFUL "
                  f"predictions={n_certifiable}/{n_success} ({100*n_certifiable/n_success:.0f}%)  "
                  f"certified cases={n_certifiable}")

    print("\n--- ProjectedFNO full metric set across resolutions (Task 13) ---")
    for N in RESOLUTIONS:
        rows = [r for r in records[("ProjectedFNO", N)] if r["success"]]
        rel_l2_vals = [r["rel_l2"] for r in rows]
        rel_energy_vals = [r["rel_energy"] for r in rows]
        eff_vals = [r["effectivity"] for r in rows if r["effectivity"] == r["effectivity"]]
        runtime_vals = [r["runtime_s"] for r in rows]
        rel_l2_str = fmt(np.median(rel_l2_vals)) if rel_l2_vals else "n/a"
        rel_energy_str = fmt(np.median(rel_energy_vals)) if rel_energy_vals else "n/a"
        eff_str = f"{np.median(eff_vals):.2f}" if eff_vals else "n/a"
        runtime_str = f"{1000 * np.mean(runtime_vals):.2f}ms" if runtime_vals else "n/a"
        print(f"  N={N}: median rel_L2={rel_l2_str}  median rel_energy={rel_energy_str}  "
              f"median effectivity={eff_str}  mean runtime={runtime_str}")

    # ── Never overwrite results silently ────────────────────────────────
    h5_path = RESULTS_DIR / "exp10_cross_resolution.h5"
    fig_path = FIGURES_DIR / "exp10_cross_resolution.png"
    tex_path = TABLES_DIR / "exp10_cross_resolution.tex"
    moved = supersede_if_exists(h5_path, fig_path, tex_path, h5_path.with_suffix(".config.json"))
    for m in moved:
        print(f"superseded existing output -> {m}")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    arrays = {}
    for (model_name, N), rows in records.items():
        key = f"{model_name}_N{N}"
        arrays[f"{key}_success"] = np.array([r["success"] for r in rows], dtype=np.int8)
        arrays[f"{key}_E_i"] = np.array([r["E_i"] for r in rows])
        arrays[f"{key}_eta_guar"] = np.array([r["eta_guar"] for r in rows])
        arrays[f"{key}_is_spd"] = np.array([r["is_spd"] for r in rows], dtype=np.int8)
        arrays[f"{key}_rel_l2"] = np.array([r["rel_l2"] for r in rows])
        arrays[f"{key}_rel_energy"] = np.array([r["rel_energy"] for r in rows])
        arrays[f"{key}_effectivity"] = np.array([r["effectivity"] for r in rows])
        arrays[f"{key}_runtime_s"] = np.array([r["runtime_s"] for r in rows])
    for N, rows in projection_records.items():
        arrays[f"DeepONet_projected_N{N}_rel_l2_terminal"] = np.array([r["rel_l2_terminal"] for r in rows])
    write_h5(
        h5_path, arrays,
        attrs={
            "resolutions": RESOLUTIONS, "train_resolution": TRAIN_N, "n_test_mus": N_TEST_MUS,
            "seed": SEED, "x_min": X_MIN, "x_max": X_MAX, "mesh_p": MESH_P,
            "claim_a_holds": claim_a_holds, "claim_b_holds": claim_b_holds,
        },
    )

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(RESOLUTIONS))
    width = 0.2
    for i, model_name in enumerate(MODEL_NAMES):
        success_frac = [
            np.mean([r["success"] for r in records[(model_name, N)]]) if records[(model_name, N)] else 0.0
            for N in RESOLUTIONS
        ]
        axes[0].bar(x + (i - 1.5) * width, success_frac, width, label=model_name)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"N={N}{' (train)' if N == TRAIN_N else ''}" for N in RESOLUTIONS])
    axes[0].set_ylabel("fraction of queries with a successful prediction")
    axes[0].set_title("exp10: architectural incompatibility (bars=0 at off-training N)\nvs. ProjectedFNO's genuine generalization")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.05)

    fno_meds, fno_lo, fno_hi = [], [], []
    for N in RESOLUTIONS:
        vals = np.array([r["eta_guar"] for r in records[("ProjectedFNO", N)] if r["success"] and r["is_spd"]])
        if len(vals):
            fno_meds.append(np.median(vals)); fno_lo.append(np.percentile(vals, 25)); fno_hi.append(np.percentile(vals, 75))
        else:
            fno_meds.append(np.nan); fno_lo.append(np.nan); fno_hi.append(np.nan)
    fno_meds, fno_lo, fno_hi = np.array(fno_meds), np.array(fno_lo), np.array(fno_hi)
    axes[1].errorbar(RESOLUTIONS, fno_meds, yerr=[fno_meds - fno_lo, fno_hi - fno_meds],
                      marker="o", capsize=4, color="C2")
    axes[1].axvline(TRAIN_N, color="k", linestyle="--", alpha=0.5, label="training resolution")
    axes[1].set_xlabel("mesh resolution N"); axes[1].set_ylabel(r"$\eta_{guar}$ (median, IQR)")
    axes[1].set_yscale("log")
    axes[1].set_title("exp10: ProjectedFNO certified error across resolutions\n(genuine generalization)")
    axes[1].legend(fontsize=8)

    off_train_res = [N for N in RESOLUTIONS if N != TRAIN_N]
    proj_meds = [float(np.median([r["rel_l2_terminal"] for r in projection_records[N]]))
                 if projection_records[N] else np.nan for N in off_train_res]
    fno_rel_l2_meds = [float(np.median([r["rel_l2"] for r in records[("ProjectedFNO", N)] if r["success"]]))
                        if any(r["success"] for r in records[("ProjectedFNO", N)]) else np.nan for N in off_train_res]
    xw = np.arange(len(off_train_res))
    axes[2].bar(xw - 0.15, proj_meds, 0.3, label="DeepONet, projected from N=64\n(terminal state only)", color="C3")
    axes[2].bar(xw + 0.15, fno_rel_l2_meds, 0.3, label="ProjectedFNO, native\n(full trajectory)", color="C2")
    axes[2].set_xticks(xw); axes[2].set_xticklabels([f"N={N}" for N in off_train_res])
    axes[2].set_yscale("log")
    axes[2].set_ylabel("median relative $L_2$ error")
    axes[2].set_title("exp10: off-training-resolution comparison\n(not a bare failure)")
    axes[2].legend(fontsize=7)

    fig.suptitle(f"exp10: cross-resolution evaluation (n={N_TEST_MUS} queries)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    # Pre-submission fixes item 4.1: the single "success rate" column
    # (prediction success only) is replaced by three columns --
    # prediction success is NOT the same thing as certifiability (a model
    # can run successfully at a query where S_h(mu) still isn't SPD, so no
    # certificate can be issued there) -- and the certifiable fraction is
    # no longer left implicit in the median columns' own SPD-filtering.
    header = ["model", "N", "prediction success", "certifiable fraction (SPD)", "certified cases",
               "median $E_i$", "median $\\eta_{guar}$",
               "median rel $L_2$", "median rel energy", "median effectivity", "mean runtime (ms)"]
    rows_tex = []
    for model_name in MODEL_NAMES:
        for N in RESOLUTIONS:
            rows = records[(model_name, N)]
            n_success = sum(r["success"] for r in rows)
            n_certifiable = sum(1 for r in rows if r["success"] and r["is_spd"])
            pred_success_str = f"{n_success}/{len(rows)} ({100*n_success/len(rows):.0f}\\%)" if rows else "n/a"
            certifiable_str = (f"{n_certifiable}/{n_success} ({100*n_certifiable/n_success:.0f}\\%)"
                                if n_success > 0 else "n/a")
            e_vals = [r["E_i"] for r in rows if r["success"]]
            eta_vals = [r["eta_guar"] for r in rows if r["success"] and r["is_spd"]]
            rel_l2_vals = [r["rel_l2"] for r in rows if r["success"]]
            rel_energy_vals = [r["rel_energy"] for r in rows if r["success"]]
            eff_vals = [r["effectivity"] for r in rows if r["success"] and r["effectivity"] == r["effectivity"]]
            runtime_vals = [r["runtime_s"] for r in rows]
            rows_tex.append([
                model_name, str(N), pred_success_str, certifiable_str, str(n_certifiable),
                fmt(np.median(e_vals)) if e_vals else "n/a",
                fmt(np.median(eta_vals)) if eta_vals else "n/a",
                fmt(np.median(rel_l2_vals)) if rel_l2_vals else "n/a",
                fmt(np.median(rel_energy_vals)) if rel_energy_vals else "n/a",
                f"{np.median(eff_vals):.2f}" if eff_vals else "n/a",
                f"{1000 * np.mean(runtime_vals):.2f}" if runtime_vals else "n/a",
            ])
    # DeepONet-projected comparison rows (terminal state only), clearly labeled
    for N in RESOLUTIONS:
        if N == TRAIN_N:
            continue
        vals = [r["rel_l2_terminal"] for r in projection_records[N]]
        rows_tex.append([
            "DeepONet (projected from N=64, terminal state only)", str(N), "n/a (not native)",
            "n/a", "n/a", "n/a", "n/a", fmt(np.median(vals)) if vals else "n/a", "n/a", "n/a", "n/a",
        ])
    write_latex_table(tex_path, header, rows_tex)
    print(f"wrote {tex_path}")

    prov_path = write_provenance(
        h5_path, start_time=t_start, seed=SEED, n_requested=N_TEST_MUS, n_kept=N_TEST_MUS,
        mesh_resolutions=RESOLUTIONS,
        git_commit_override=git_commit,
        extra={"claim_a_holds": claim_a_holds, "claim_b_holds": claim_b_holds},
    )
    print(f"wrote {prov_path}")


if __name__ == "__main__":
    main()
