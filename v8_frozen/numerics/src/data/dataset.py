"""Reference-solution dataset generation and storage.

One HDF5 file per dataset, per-sample groups (``samples/000000``,
``samples/000001``, ...), a top-level manifest recording split membership,
config snapshot, and generation statistics, and a separate ``failures/``
tree so a failed solve is recorded, never silently dropped (see
``scripts/generate_debug_dataset.py``, which drives this in parallel).

Schema per successful sample group (``samples/{i:06d}``):
  Datasets: mu (6,) [C,G,M,Y,r,T], payoff_p (n_dof,), coefficients
    (n_steps+1, n_dof), c0 (n_dof,), time_grid (n_steps+1,),
    forcing_vectors (n_steps, n_dof) or absent if not stored,
    scheme_per_step (n_steps,) str, iterations_per_step (n_steps,) int,
    tolerance_per_step (n_steps,) float (NaN where the solver used a direct
    solve), bc_left, bc_right (int arrays), mesh_nodes (N+1,).
  Attrs: split, strike, mode, dt, n_steps, cpu_time, mean_gmres_iters,
    mesh_N, mesh_p, mesh_h, mesh_x_min, mesh_x_max, mesh_n_dof,
    price_atm, delta_atm, gamma_atm (if computed),
    lambda_min_numerical, alpha_certified, safety_ratio, coercivity_strategy
    (Task 3, working prompt -- split, never a single conflated
    "coercivity_constant"; see certification.constants), continuity_constant,
    condition_number_S_h, is_spd (if computed).

Schema per failure group (``failures/{i:06d}``): attrs split, C, G, M, Y, r,
T, strike, error (full traceback string).
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cgmy.operators import OperatorCache
from cgmy.reference_solver import (
    CGMYParameters,
    EuropeanPutPayoff,
    GalerkinTrajectory,
    Mesh,
    MeshInfo,
    SolverMetadata,
    TimeGrid,
    solve_cgmy,
)
from certification.constants import compute_stability_constants
from common import io as common_io
from data import payoff_family
from data.sampling import ParameterBox


@dataclass
class SampleResult:
    """Outcome of one solve_cgmy attempt, ready to be written to HDF5.

    ``ok=False`` samples carry only ``mu``/``payoff``/``error`` -- callers
    must write these to the failures/ tree, never drop them.
    """

    ok: bool
    mu: CGMYParameters
    payoff: EuropeanPutPayoff
    error: str | None = None
    traj: object | None = None  # GalerkinTrajectory, kept generic to avoid a heavy import here
    payoff_p: np.ndarray | None = None
    price_atm: float | None = None
    delta_atm: float | None = None
    gamma_atm: float | None = None
    lambda_min_numerical: float | None = None
    alpha_certified: float | None = None  # on-disk key kept (pre-v5 schema); value = StabilityConstants.alpha_lower_estimate, a floating-point estimate (v5 naming audit)
    safety_ratio: float | None = None
    coercivity_strategy: str | None = None
    continuity_constant: float | None = None
    condition_number_S_h: float | None = None
    is_spd: bool | None = None


def generate_one_sample(
    mu: CGMYParameters,
    payoff: EuropeanPutPayoff,
    mesh: Mesh,
    time_grid: TimeGrid,
    *,
    mode: str = "toeplitz",
    compute_extras: bool = True,
) -> SampleResult:
    """Solve one (mu, payoff) point via solve_cgmy and package everything the
    HDF5 schema needs. Never raises: failures are captured in the returned
    SampleResult (ok=False, error=<full traceback text>).
    """
    try:
        traj = solve_cgmy(mu, payoff, mesh, time_grid, mode=mode)
    except Exception:
        return SampleResult(ok=False, mu=mu, payoff=payoff, error=traceback.format_exc())

    ops = OperatorCache(
        Mesh(N=traj.mesh_information.N, p=traj.mesh_information.p),
        traj.mesh_information.x_min, traj.mesh_information.x_max,
    )
    p_vec = payoff_family.european_put(payoff.strike, ops)

    result = SampleResult(ok=True, mu=mu, payoff=payoff, traj=traj, payoff_p=p_vec)

    if compute_extras:
        from cgmy_bspline.greeks import compute_greeks
        from cgmy_bspline.projection import eval_solution

        grid_info = {"N": traj.mesh_information.N, "h": traj.mesh_information.h,
                     "x_min": traj.mesh_information.x_min}
        x0 = np.array([np.log(payoff.strike)])
        V0 = eval_solution(traj.coefficients[-1], grid_info, traj.mesh_information.p, x0)
        result.price_atm = float(V0[0])
        delta, gamma = compute_greeks(traj.coefficients[-1], grid_info, traj.mesh_information.p, x0)
        result.delta_atm, result.gamma_atm = float(delta[0]), float(gamma[0])

        try:
            const = compute_stability_constants(mu, ops)
            result.lambda_min_numerical = const.lambda_min_numerical
            result.alpha_certified = const.alpha_certified
            result.safety_ratio = const.safety_ratio
            result.coercivity_strategy = const.coercivity_strategy
            result.continuity_constant = const.continuity_constant
            result.condition_number_S_h = const.condition_number_S_h
            result.is_spd = const.is_spd
        except ValueError:
            pass  # mesh too large for dense constants -- leave as None, not a failure

    return result


def write_sample(path: str | Path, index: int, split: str, result: SampleResult) -> None:
    """Append one sample (success or failure) to the dataset HDF5 file."""
    mu = result.mu
    if not result.ok:
        common_io.append_record(
            path, f"failures/{index:06d}",
            arrays={},
            attrs={
                "split": split, "C": mu.C, "G": mu.G, "M": mu.M, "Y": mu.Y,
                "r": mu.r, "T": mu.T, "strike": result.payoff.strike,
                "error": result.error,
            },
        )
        return

    traj = result.traj
    md = traj.solver_metadata
    tol_per_step = np.array([t if t is not None else np.nan for t in md.tolerance_per_step])
    arrays = {
        "mu": np.array([mu.C, mu.G, mu.M, mu.Y, mu.r, mu.T]),
        "payoff_p": result.payoff_p,
        "c0": traj.initial_coefficients,
        "coefficients": traj.coefficients,
        "time_grid": traj.time_grid,
        "scheme_per_step": np.array(md.scheme_per_step),
        "iterations_per_step": np.array(md.iterations_per_step),
        "tolerance_per_step": tol_per_step,
        "bc_left": np.array(md.bc_dofs[0]),
        "bc_right": np.array(md.bc_dofs[1]),
        "mesh_nodes": traj.mesh_information.nodes,
    }
    if traj.forcing_vectors is not None:
        arrays["forcing_vectors"] = traj.forcing_vectors

    attrs = {
        "split": split,
        "strike": result.payoff.strike,
        "mode": md.mode,
        "dt": md.dt,
        "n_steps": md.n_steps,
        "cpu_time": md.cpu_time,
        "mean_gmres_iters": md.mean_gmres_iters if md.mean_gmres_iters is not None else np.nan,
        "mesh_N": traj.mesh_information.N,
        "mesh_p": traj.mesh_information.p,
        "mesh_h": traj.mesh_information.h,
        "mesh_x_min": traj.mesh_information.x_min,
        "mesh_x_max": traj.mesh_information.x_max,
        "mesh_n_dof": traj.mesh_information.n_dof,
    }
    for field in ("price_atm", "delta_atm", "gamma_atm", "lambda_min_numerical",
                  "alpha_certified", "safety_ratio", "continuity_constant", "condition_number_S_h"):
        value = getattr(result, field)
        attrs[field] = value if value is not None else np.nan
    if result.coercivity_strategy is not None:
        attrs["coercivity_strategy"] = result.coercivity_strategy
    if result.is_spd is not None:
        attrs["is_spd"] = result.is_spd

    common_io.append_record(path, f"samples/{index:06d}", arrays=arrays, attrs=attrs)


def read_trajectory(path: str | Path, index: int) -> GalerkinTrajectory:
    """Reconstruct a GalerkinTrajectory from a successful sample group.

    Round-trips exactly what write_sample stored -- used by
    tests/test_dataset_roundtrip.py to confirm storage/ordering did not
    silently corrupt anything (checked by recomputing the Stage B residual
    on the reconstructed object, not just comparing arrays for equality).

    Raises:
        KeyError: If ``samples/{index:06d}`` does not exist (e.g. that
            index was a logged failure, not a success -- check
            failures/{index:06d} instead).
    """
    arrays, attrs = common_io.read_group(path, f"samples/{index:06d}")

    mu = CGMYParameters(*arrays["mu"])
    payoff = EuropeanPutPayoff(strike=float(attrs["strike"]))

    mesh_info = MeshInfo(
        N=int(attrs["mesh_N"]), p=int(attrs["mesh_p"]), n_dof=int(attrs["mesh_n_dof"]),
        h=float(attrs["mesh_h"]), x_min=float(attrs["mesh_x_min"]), x_max=float(attrs["mesh_x_max"]),
        nodes=arrays["mesh_nodes"],
    )
    tol_per_step = [None if np.isnan(t) else float(t) for t in arrays["tolerance_per_step"]]
    metadata = SolverMetadata(
        mode=str(attrs["mode"]), dt=float(attrs["dt"]), n_steps=int(attrs["n_steps"]),
        scheme_per_step=[s.decode() if isinstance(s, bytes) else str(s) for s in arrays["scheme_per_step"]],
        iterations_per_step=[int(i) for i in arrays["iterations_per_step"]],
        tolerance_per_step=tol_per_step,
        bc_dofs=(list(arrays["bc_left"]), list(arrays["bc_right"])),
        cpu_time=float(attrs["cpu_time"]),
        mean_gmres_iters=None if np.isnan(attrs["mean_gmres_iters"]) else float(attrs["mean_gmres_iters"]),
    )

    return GalerkinTrajectory(
        coefficients=arrays["coefficients"],
        forcing_vectors=arrays.get("forcing_vectors"),
        initial_coefficients=arrays["c0"],
        time_grid=arrays["time_grid"],
        parameter_vector=mu,
        payoff=payoff,
        mesh_information=mesh_info,
        solver_metadata=metadata,
    )


def write_manifest(
    path: str | Path,
    split_sizes: dict[str, int],
    n_failures: int,
    config_snapshot: str,
    seed: int,
) -> None:
    """Write top-level manifest attributes (call once, after all samples are written).

    Task 5 (working prompt): records git_sha/git_dirty at write time, the
    same way common.checkpoint.save_checkpoint already does for model
    checkpoints -- previously missing here (per-sample mesh_N/mesh_h/etc.
    attrs are already self-describing and correct, see write_sample; only
    the git-commit provenance was absent at the manifest level).
    """
    from common.checkpoint import get_git_sha_and_dirty

    try:
        git_sha, git_dirty = get_git_sha_and_dirty()
    except Exception:
        git_sha, git_dirty = "unknown", True

    common_io.write_arrays(
        path, arrays={}, mode="a",
        attrs={
            "n_samples": sum(split_sizes.values()),
            "n_failures": n_failures,
            "config_snapshot": config_snapshot,
            "seed": seed,
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            **{f"split_size_{name}": n for name, n in split_sizes.items()},
        },
    )


def generate_dataset(
    box: ParameterBox,
    mesh: Mesh,
    time_grid: TimeGrid,
    n_samples: int,
    seed: int,
    output_path: str | Path,
) -> None:
    """Serial reference implementation: sample the box, solve each point via
    solve_cgmy, and write results to HDF5. See
    ``scripts/generate_debug_dataset.py`` for the parallel, named-split
    driver actually used to build datasets; this function is the simple,
    single-process building block it calls per sample.
    """
    from data.sampling import sample_sobol

    samples = sample_sobol(box, n_samples, seed=seed)
    for i, (mu, payoff) in enumerate(samples):
        result = generate_one_sample(mu, payoff, mesh, time_grid)
        write_sample(output_path, i, split="unsplit", result=result)


class CGMYCoefficientDataset:
    """Map-style dataset of (parameter+payoff feature vector, spline
    coefficient trajectory) pairs, reading successful samples from one split
    of an HDF5 file written by ``generate_dataset``/
    ``scripts.generate_debug_dataset``.

    Implements the ``__len__``/``__getitem__`` map-style protocol so it can
    be wrapped directly in ``torch.utils.data.DataLoader``; it does not
    subclass ``torch.utils.data.Dataset`` or import torch at module level,
    so ``data.dataset`` stays importable (and its HDF5 I/O testable) in
    environments without torch installed. ``torch`` is imported lazily in
    ``__getitem__``, where it is actually needed.

    Args:
        hdf5_path: Path to an HDF5 file written by this module.
        split: Which named split to expose (e.g. 'train_interior'); None
            exposes every successful sample regardless of split.
    """

    def __init__(self, hdf5_path: str | Path, split: str | None = None) -> None:
        self.hdf5_path = Path(hdf5_path)
        all_ids = common_io.list_groups(self.hdf5_path, parent="samples")
        if split is None:
            self._ids = all_ids
        else:
            self._ids = [
                sid for sid in all_ids
                if common_io.read_attrs(self.hdf5_path, group=f"samples/{sid}").get("split") == split
            ]

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int):
        import torch

        arrays, attrs = common_io.read_group(self.hdf5_path, f"samples/{self._ids[index]}")
        features = np.concatenate([arrays["mu"], arrays["payoff_p"]])
        target = arrays["coefficients"]
        return torch.from_numpy(features).double(), torch.from_numpy(target).double()
