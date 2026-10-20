"""Shared utilities for the Stage I paper experiments (exp01-exp07).

Every experiment script: loads the four Stage H checkpoints on one common
fixed mesh (the same convention established in Stage F/G/H -- POD and any
cross-model comparison need a shared discrete space), evaluates them
through the SAME certification machinery used throughout this project
(certification.residuals / mass_indicator / guaranteed_certificate --
never a reimplementation), and writes results as one HDF5 file in
results/, one figure in figures/, and one LaTeX table in tables/.

Named "exp_common.py" (not "common.py") to avoid shadowing src/common/.

Scale note: sample counts in each experiment script are reduced from
"paper scale" for interactive-session wall-clock reasons, the same
scope reduction already flagged for Stage G's ablation and Stage H's
production dataset -- each script exposes its sample count as a
module-level constant so it can trivially be rerun larger.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from cgmy.dirichlet_bc import boundary_error, project_bc
from cgmy.operators import OperatorCache
from cgmy.reference_solver import (
    CGMYParameters,
    EuropeanPutPayoff,
    GalerkinTrajectory,
    Mesh,
    TimeGrid,
    solve_cgmy,
)
from cgmy_bspline.projection import eval_solution
from certification.constants import compute_stability_constants
from certification.guaranteed_certificate import eta_guar as compute_eta_guar
from certification.guaranteed_certificate import eta_sharp as compute_eta_sharp
from certification.mass_indicator import eta_mass_indicator as compute_eta_mass_indicator
from certification.residuals import _interior_mask, compute_all_residuals, residual_norm
from common import io as common_io
from common.checkpoint import (
    CheckpointCompatibilityError,
    check_data_compatibility,
    check_discretization_compatibility,
)
from data import payoff_family
from data.sampling import ParameterBox, load_mesh_and_time_grid, load_parameter_box, sample_sobol
from learning.deeponet_coeff import DeepONetCoeffNet
from learning.fno import FNO1d
from learning.interface import wrap_as_trajectory
from learning.mlp_coeff import MLPCoeffNet
from learning.normalization import normalize_mu, stats_from_box
from learning.pod import PODBasis, PODGalerkinROM
from learning.projected_fno import ProjectedFNO
from pipeline.predict_certify_correct import build_mesh_info

torch.set_default_dtype(torch.float64)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
TABLES_DIR = REPO_ROOT / "tables"

# Same fixed mesh used for every Stage F/G/H model (POD-compatibility
# requirement -- see learning/pod.py's module docstring), read from
# configs/base.yaml's discretization block (the single source of truth --
# see data.sampling.load_mesh_and_time_grid's docstring).
FIXED_MESH, TIME_GRID = load_mesh_and_time_grid(CONFIG_PATH)
FIXED_MESH_INFO = build_mesh_info(FIXED_MESH)
MODE = "toeplitz"
STRIKE = 100.0
PAYOFF = EuropeanPutPayoff(strike=STRIKE)

# The model Stage H's validation numbers favor (lowest val_L2: 0.430 vs
# MLP 0.796, ProjectedFNO 1.001-1.265, POD 9.458) -- reported as "the final
# model" where the experiment scripts ask for it, not a fifth distinct model.
FINAL_MODEL_NAME = "DeepONet"

CERT_CONFIG = yaml.safe_load(CONFIG_PATH.read_text())["certification"]
MAX_PCG_ITERATIONS = CERT_CONFIG["max_pcg_iterations"]
PRECONDITIONER = CERT_CONFIG["preconditioner"]


def get_ops() -> OperatorCache:
    return OperatorCache(FIXED_MESH, FIXED_MESH.x_min, FIXED_MESH.x_max)


def get_box_and_stats():
    box = load_parameter_box(CONFIG_PATH)
    stats = stats_from_box(box)
    return box, stats


def load_models(ops: OperatorCache, stats, *, include_legacy_pod: bool = True) -> dict[str, Any]:
    """Load all four Stage H checkpoints. Returns {name: model} where each
    model implements predict_trajectory(mu_tilde, psi, c0, time_grid, T)
    (learning.interface.CoefficientModel).

    Task 5 (working prompt): every checkpoint's discretization/data-box
    provenance is HARD-checked (raise, never warn) against this project's
    current FIXED_MESH/TIME_GRID/parameter box before its weights are ever
    used -- see common.checkpoint.check_discretization_compatibility/
    check_data_compatibility and their module docstring for why
    config_snapshot alone was not enough (it already went stale once).
    """
    box = load_parameter_box(CONFIG_PATH)
    models: dict[str, Any] = {}

    payload = torch.load(CHECKPOINTS_DIR / "mlp_final.pt", weights_only=False)
    check_discretization_compatibility(payload.get("discretization"), FIXED_MESH, TIME_GRID, source="checkpoints/mlp_final.pt")
    check_data_compatibility(payload.get("data_box"), box, STRIKE, source="checkpoints/mlp_final.pt")
    mlp = MLPCoeffNet(**payload["model_init_kwargs"])
    mlp.load_state_dict(payload["model_state_dict"])
    mlp.eval()
    models["MLP"] = mlp

    payload = torch.load(CHECKPOINTS_DIR / "deeponet_final.pt", weights_only=False)
    check_discretization_compatibility(payload.get("discretization"), FIXED_MESH, TIME_GRID, source="checkpoints/deeponet_final.pt")
    check_data_compatibility(payload.get("data_box"), box, STRIKE, source="checkpoints/deeponet_final.pt")
    deeponet = DeepONetCoeffNet(**payload["model_init_kwargs"])
    deeponet.load_state_dict(payload["model_state_dict"])
    deeponet.eval()
    models["DeepONet"] = deeponet

    # v5 (roadmap §3.2): the legacy POD (raw modes, no forcing in the reduced
    # equation) is only loaded for the historical exp01-exp15 scripts. v5 code
    # passes include_legacy_pod=False and uses learning.pod_free.load_pod_artifact.
    if include_legacy_pod:
        data = np.load(CHECKPOINTS_DIR / "pod_final.npz")
        _check_pod_compatibility(data, source="checkpoints/pod_final.npz", box=box)
        basis = PODBasis(modes=data["modes"], singular_values=data["singular_values"], mean=data["mean"])
        models["POD"] = PODGalerkinROM(basis, ops, stats=stats)

    payload = torch.load(CHECKPOINTS_DIR / "projected_fno_final.pt", weights_only=False)
    check_discretization_compatibility(payload.get("discretization"), FIXED_MESH, TIME_GRID, source="checkpoints/projected_fno_final.pt")
    check_data_compatibility(payload.get("data_box"), box, STRIKE, source="checkpoints/projected_fno_final.pt")
    fno = FNO1d(**payload["fno_init_kwargs"])
    fno.load_state_dict(payload["fno_state_dict"])
    fno.eval()
    m = payload["mesh"]
    fno_mesh = Mesh(N=m["N"], p=m["p"], x_min=m["x_min"], x_max=m["x_max"])
    models["ProjectedFNO"] = ProjectedFNO(fno, fno_mesh, ops, mu_dim=payload["mu_dim"])

    return models


def _check_pod_compatibility(npz_data, *, source: str, box) -> None:
    """POD's checkpoint is an .npz (not torch.save), so its discretization/
    data_box blocks are JSON-encoded strings (discretization_json/
    data_box_json) rather than nested dict values -- see
    scripts/train_final_models.py's POD save site."""
    if "discretization_json" not in npz_data or "data_box_json" not in npz_data:
        raise CheckpointCompatibilityError(
            f"{source}: no discretization_json/data_box_json recorded on this "
            f"checkpoint -- cannot verify compatibility (a checkpoint predating "
            f"the Task 5 provenance fix?). Patch or regenerate it -- see "
            f"notes/05_provenance.md."
        )
    discretization = json.loads(str(npz_data["discretization_json"]))
    data_box = json.loads(str(npz_data["data_box_json"]))
    check_discretization_compatibility(discretization, FIXED_MESH, TIME_GRID, source=source)
    check_data_compatibility(data_box, box, STRIKE, source=source)


def solve_reference(mu: CGMYParameters) -> GalerkinTrajectory:
    return solve_cgmy(mu, PAYOFF, FIXED_MESH, TIME_GRID, mode=MODE)


def predict(model, mu: CGMYParameters, ops: OperatorCache, stats, c0: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    """Uniform prediction call across all four models.

    Projects onto the exact Dirichlet BC at every time level before
    returning (Task 1, working prompt): the exact-IC structural trick only
    fixes tau=0, and certification.residuals zeroes the residual rows at
    Dirichlet dofs, so nothing downstream would otherwise ever check them.
    This is the one prediction path every exp01-07 script consumes.
    """
    mu_tilde = normalize_mu(mu, stats)
    psi = payoff_family.european_put(PAYOFF.strike, ops)
    raw = model.predict_trajectory(mu_tilde, psi, c0, time_grid, mu.T)
    return project_bc(raw, mu, PAYOFF, FIXED_MESH_INFO, time_grid)


def l2_norm(v: np.ndarray, ops: OperatorCache) -> float:
    """Physical L2(Omega) norm: ||u_h||_{L2}^2 = c^T M_h c."""
    return float(np.sqrt(max(v @ ops.apply_M(v), 0.0)))


def energy_norm(v: np.ndarray, mu: CGMYParameters, ops: OperatorCache) -> float:
    """Energy norm induced by S_h(mu): ||u_h||_S^2 = c^T S_h(mu) c."""
    return float(np.sqrt(max(v @ ops.apply_S(mu, v), 0.0)))


def true_error_Ei(pred: np.ndarray, true: np.ndarray, ops: OperatorCache) -> float:
    """E_i = max_m ||U_h^m - U_theta,h^m||_{L2} (exp02's definition): the
    max-over-time-levels physical L2 norm of the coefficient error."""
    diff = pred - true
    return float(max(l2_norm(d, ops) for d in diff))


def relative_l2_error(pred: np.ndarray, true: np.ndarray, ops: OperatorCache) -> float:
    """max_m ||e^m||_{L2} / max_m ||U_h^m||_{L2}."""
    num = true_error_Ei(pred, true, ops)
    den = max(l2_norm(t, ops) for t in true)
    return num / den if den > 0 else float("nan")


def relative_energy_error(pred: np.ndarray, true: np.ndarray, mu: CGMYParameters, ops: OperatorCache) -> float:
    diff = pred - true
    num = max(energy_norm(d, mu, ops) for d in diff)
    den = max(energy_norm(t, mu, ops) for t in true)
    return num / den if den > 0 else float("nan")


def option_price(coeffs_final: np.ndarray, ops: OperatorCache) -> float:
    grid_info = {"N": FIXED_MESH.N, "h": ops.h, "x_min": FIXED_MESH.x_min}
    x0 = np.array([np.log(STRIKE)])
    return float(eval_solution(coeffs_final, grid_info, FIXED_MESH.p, x0)[0])


@dataclasses.dataclass
class CertificateResult:
    is_spd: bool
    lambda_min_numerical: float = float("nan")
    alpha_certified: float = float("nan")
    safety_ratio: float = float("nan")
    coercivity_strategy: str = ""
    condition_number: float = float("nan")
    max_residual_norm: float = float("nan")
    eta_mass_indicator: float = float("nan")  # NOT a certificate -- see certification.mass_indicator
    eta_sharp: float = float("nan")
    eta_guar: float = float("nan")
    n_pcg_iterations_mean: float = float("nan")
    E_boundary_raw: float = float("nan")  # max Dirichlet-dof violation BEFORE projection
    E_boundary: float = float("nan")  # same, on the trajectory certificates were computed from


def _project_and_residuals(
    traj_true: GalerkinTrajectory,
    pred_coeffs: np.ndarray,
    mu: CGMYParameters,
    ops: OperatorCache,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Project pred_coeffs onto the exact Dirichlet BC and compute its CN
    residuals -- the one code path every certificate computation AND the
    Step-4 identity-discrepancy check (working prompt) funnels through, so
    there is exactly one projection/residual computation, never two
    independently hand-synced copies that could silently drift apart.

    Returns (projected pred_coeffs, residuals array (n_steps, n_dof),
    E_boundary_raw, E_boundary) -- see compute_certificates for what the two
    boundary-error values mean.
    """
    if pred_coeffs.shape != traj_true.coefficients.shape:
        raise ValueError(
            f"predicted coefficients shape {pred_coeffs.shape} != reference shape "
            f"{traj_true.coefficients.shape}"
        )
    # Task 1 (working prompt): certification.residuals zeroes the residual
    # rows at Dirichlet dofs, so a certificate computed on an unprojected
    # prediction is blind to exactly the entries the network is free to get
    # wrong. Project here -- the one place every exp01/exp02/exp04/exp05/
    # exp10 certificate computation funnels through -- before touching the
    # residual machinery at all.
    E_boundary_raw = boundary_error(pred_coeffs, mu, traj_true.payoff, traj_true.mesh_information, traj_true.time_grid)
    pred_coeffs = project_bc(pred_coeffs, mu, traj_true.payoff, traj_true.mesh_information, traj_true.time_grid)
    E_boundary = boundary_error(pred_coeffs, mu, traj_true.payoff, traj_true.mesh_information, traj_true.time_grid)
    traj_pred = wrap_as_trajectory(traj_true, pred_coeffs)
    residuals = compute_all_residuals(traj_pred, mu, ops)
    return pred_coeffs, residuals, E_boundary_raw, E_boundary


def compute_certificates(
    traj_true: GalerkinTrajectory,
    pred_coeffs: np.ndarray,
    mu: CGMYParameters,
    ops: OperatorCache,
    max_pcg_iterations: int = MAX_PCG_ITERATIONS,
    preconditioner: str = PRECONDITIONER,
) -> CertificateResult:
    """eta_mass_indicator (heuristic, NOT a certificate), eta_sharp, eta_guar
    for a candidate PREDICTED trajectory, via the same certification.*
    machinery used everywhere else in this project -- reported side by side
    purely for comparison (see exp02/exp03), never to gate an accept/reject
    decision here. Returns is_spd=False (all else NaN) if S_h(mu) is not
    SPD -- never fabricates a certificate value in that case."""
    const = compute_stability_constants(mu, ops)
    if not const.is_spd:
        return CertificateResult(is_spd=False)

    pred_coeffs, residuals, E_boundary_raw, E_boundary = _project_and_residuals(traj_true, pred_coeffs, mu, ops)
    dt = traj_true.solver_metadata.dt
    bc_dofs = traj_true.solver_metadata.bc_dofs

    max_r = max(residual_norm(residuals[n], ops, bc_dofs) for n in range(residuals.shape[0]))
    em = compute_eta_mass_indicator(residuals, ops, dt, const.alpha_certified)
    es = compute_eta_sharp(residuals, mu, ops, dt)
    eg = compute_eta_guar(
        residuals, mu, ops, dt, const.alpha_certified,
        max_pcg_iterations=max_pcg_iterations, preconditioner=preconditioner,
    )
    return CertificateResult(
        is_spd=True,
        lambda_min_numerical=const.lambda_min_numerical,
        alpha_certified=const.alpha_certified,
        safety_ratio=const.safety_ratio,
        coercivity_strategy=const.coercivity_strategy,
        condition_number=const.condition_number_S_h,
        max_residual_norm=max_r,
        eta_mass_indicator=em,
        eta_sharp=es.eta,
        eta_guar=eg.eta,
        E_boundary_raw=E_boundary_raw,
        E_boundary=E_boundary,
        n_pcg_iterations_mean=float(np.mean(eg.n_pcg_iterations)),
    )


def identity_discrepancy(
    traj_true: GalerkinTrajectory,
    pred_coeffs: np.ndarray,
    mu: CGMYParameters,
    ops: OperatorCache,
) -> dict[str, Any]:
    """Step 4 (working prompt, Priority 2): verify the discrete error
    equation (main.tex eq:discrete-error-equation) holds at machine
    precision on the ACTUAL code path the certificate/triple-norm
    computations use -- i.e. that ``certification.residuals``' r_theta and
    ``certification.constants.triple_norm_squared``'s M_h*de/dt + A_h*ebar
    terms are literally consistent (same A_h, M_h, midpoint convention,
    forcing, boundary realization, sign convention), not merely each
    "looking right" in isolation.

    For every step n:

        d^{n+1/2} = r_theta^{n+1/2} + M_h*(e^{n+1}-e^n)/dt + A_h(mu)*ebar^{n+1/2}
        normalized = ||d||_2 / (||r_theta||_2 + ||M_h*de/dt||_2 + ||A_h*ebar||_2)

    where e = (projected) pred_coeffs - traj_true.coefficients. Both r_theta
    and the M_h*de/dt / A_h*ebar terms are restricted to interior
    (non-Dirichlet) dofs before taking norms -- the same restriction
    ``residual_norm``/Gate 1 use -- since r_theta is zeroed at boundary dofs
    by construction (certification.residuals module docstring) while a raw
    M_h*de/dt + A_h*ebar computation is not, so comparing the unmasked
    quantities would manufacture a spurious discrepancy at boundary dofs
    that has nothing to do with the identity being checked.

    Returns a dict with the per-step normalized-discrepancy array and its
    max/median/p99 for this one sample. If this is not near machine
    precision, STOP and diagnose (see module docstring / working prompt
    Step 4) rather than patch the efficiency bound.
    """
    pred_proj, residuals, _, _ = _project_and_residuals(traj_true, pred_coeffs, mu, ops)
    # main.tex eq:discrete-error-equation defines e^n := c_h^n - c_theta^n
    # (reference MINUS predicted, not predicted minus reference) -- the sign
    # only matters here because this is a LINEAR identity check; triple_norm_squared
    # is quadratic in e and so is unaffected by this convention either way.
    e = traj_true.coefficients - pred_proj
    dt = traj_true.solver_metadata.dt
    mask = _interior_mask(ops.n_dof, traj_true.solver_metadata.bc_dofs)

    n_steps = residuals.shape[0]
    normalized = np.empty(n_steps)
    for n in range(n_steps):
        r = residuals[n][mask]
        Mde = ops.apply_M((e[n + 1] - e[n]) / dt)[mask]
        ebar = 0.5 * (e[n + 1] + e[n])
        Aebar = ops.apply_A(mu, ebar)[mask]
        d = r + Mde + Aebar
        denom = float(np.linalg.norm(r) + np.linalg.norm(Mde) + np.linalg.norm(Aebar))
        normalized[n] = float(np.linalg.norm(d)) / denom if denom > 0 else 0.0

    return {
        "per_step": normalized,
        "max": float(normalized.max()),
        "median": float(np.median(normalized)),
        "p99": float(np.percentile(normalized, 99)),
    }


def write_h5(path: str | Path, arrays: dict[str, np.ndarray], attrs: dict[str, Any] | None = None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common_io.write_arrays(path, arrays, attrs=attrs, mode="w")


def write_latex_table(
    path: str | Path, header: list[str], rows: list[list[str]], col_align: str | None = None
) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    align = col_align or ("l" + "r" * (len(header) - 1))
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


def fmt(x: float, sig: int = 4) -> str:
    if x != x:  # NaN
        return "NaN"
    return f"{x:.{sig}e}"


# ── Pre-submission-fixes provenance/superseding helpers ─────────────────
# (Ground Rules 2-3, PRE_SUBMISSION_FIXES_CGMY_NEURAL_OPERATOR.md): never
# overwrite a result silently, and record enough about how every rerun was
# produced that "which run made this number" is never a guess.

def supersede_if_exists(*paths: str | Path) -> list[Path]:
    """Move each path that currently exists into
    results/_superseded/<YYYY-MM-DD>/<same-relative-name>, preserving the
    old file rather than deleting or silently overwriting it. New output
    is then written back to the ORIGINAL path by the caller, exactly
    where the LaTeX build already expects it. Returns the list of moved
    destinations (for the printed rerun log); paths that didn't exist are
    skipped silently (nothing to supersede)."""
    import datetime
    import shutil

    date_dir = REPO_ROOT / "results" / "_superseded" / datetime.date.today().isoformat()
    moved = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            rel = p.relative_to(REPO_ROOT)
        except ValueError:
            rel = Path(p.name)
        dest = date_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
        moved.append(dest)
    return moved


def provenance_start() -> float:
    """Call at the top of an experiment's main(); pass the return value to
    write_provenance's start_time."""
    import time
    return time.time()


def capture_git_commit_hash() -> str:
    """Call at the TOP of an experiment's main() (right after
    provenance_start(), before anything is written to disk) and pass the
    result to write_provenance's git_commit_override.

    Fix (fix/efficiency-norm-kappa, Step 8): _git_commit_hash()'s
    `git status --porcelain` check is inherently self-referential if it is
    called AFTER an experiment has already written its own output to
    TRACKED paths (results/*.csv, figures/*.png, tables/*.tex,
    *.config.json are all tracked by convention) -- the run's own
    not-yet-committed output makes the tree dirty regardless of whether the
    CODE that produced it was clean, so a sidecar written the old way could
    never report a clean commit even immediately after a fully clean
    `git commit`. Capturing the hash before any output exists reflects the
    actually-meaningful question ("was the code that produced this run
    committed and the tree clean at the time it started"), not "is the
    tree clean after this very run already overwrote its own tracked
    outputs."
    """
    return _git_commit_hash()


def _git_commit_hash() -> str:
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _hardware_info() -> dict[str, Any]:
    import os
    import re

    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    thread_env = {
        k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    blas_name = "unknown"
    try:
        cfg = np.show_config(mode="dicts")
        blas_name = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
    except Exception:  # noqa: BLE001
        pass
    return {
        "cpu_model": cpu_model, "cpu_count_logical": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "thread_env_vars": thread_env, "blas_backend": blas_name,
    }


def _library_versions() -> dict[str, str]:
    import scipy
    return {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}


def write_provenance(
    results_path: str | Path,
    *,
    start_time: float,
    seed: int | dict[str, int],
    n_requested: int | dict[str, int],
    n_kept: int | dict[str, int],
    n_rejected: int | dict[str, int] = 0,
    mesh_resolutions: list[int] | int | None = None,
    extra: dict[str, Any] | None = None,
    git_commit_override: str | None = None,
) -> Path:
    """Write <results_path stem>.config.json next to a results file,
    recording: git commit, preconditioner, solver tolerances, mesh
    resolution(s), seeds, the frozen parameter-sampling box, sample
    counts (requested/kept/rejected), hardware, library versions, and
    wall-clock start/end. `extra` merges in any experiment-specific
    fields (e.g. per-resolution sample counts) without needing a new
    helper signature per script.

    Pass `git_commit_override` (from ``capture_git_commit_hash()``, called
    at the top of main() before any output is written) to avoid the
    self-referential "-dirty" a same-run re-check would produce -- see
    ``capture_git_commit_hash``'s docstring. Falls back to a fresh
    ``_git_commit_hash()`` call if not given (unchanged behavior for
    callers that haven't adopted the pattern yet)."""
    import time

    results_path = Path(results_path)
    box = load_parameter_box(CONFIG_PATH)
    payload = {
        "git_commit": git_commit_override if git_commit_override is not None else _git_commit_hash(),
        "preconditioner": PRECONDITIONER,
        "solver_tolerances": {
            "pcg_rtol": 1e-14,  # certification.riesz_solver.solve_riesz_pcg's own default
            "pcg_atol": 0.0,
            "max_pcg_iterations": MAX_PCG_ITERATIONS,
            "eigensolver": "numpy.linalg.eigh (dense, exact -- no tolerance parameter)",
            "correction_gmres_rtol": 1e-10, "correction_gmres_restart": 50,
        },
        "mesh_resolutions": mesh_resolutions if mesh_resolutions is not None else FIXED_MESH.N,
        "seed": seed,
        "parameter_sampling_ranges": {
            "C": box.C, "G": box.G, "M": box.M, "Y": box.Y, "r": box.r,
            "maturity": box.maturity, "strike": box.strike,
        },
        "n_samples_requested": n_requested, "n_samples_kept": n_kept, "n_samples_rejected": n_rejected,
        "hardware": _hardware_info(),
        "library_versions": _library_versions(),
        "wall_clock_start": start_time, "wall_clock_end": time.time(),
        "wall_clock_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_time)),
        "wall_clock_end_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    out_path = results_path.with_suffix(".config.json")
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path
