"""Thin wrapper around the existing, trusted B-spline Galerkin CGMY solver.

Stage 1 of the six-stage dependency chain:

    Galerkin solver -> discrete residual -> certificate -> neural model
    -> predict-certify-correct -> experiments

This module does not implement any discretization, quadrature, time-stepping,
or nonlocal-operator numerics. It adapts the pre-existing, independently
validated solver at ``cgmy_bspline`` (see ``cgmy._external``) to the
dataclasses and function signature the rest of this package (``operators``,
``certification``, ``learning``, ``data``) expects.

Scope note (Stage A): only the European put payoff is supported. The wrapped
solver's Dirichlet boundary conditions encode the put's own analytic
far-field asymptotics (V -> K*exp(-r*tau) deep in the money as x -> x_min,
V -> 0 deep out of the money as x -> x_max); they are not valid for any other
payoff. Supporting a different payoff requires deriving its own boundary
conditions first, which is new numerics and out of scope here.

The time-stepping scheme is pure Crank-Nicolson from tau=0 -- the wrapped
solver has no Rannacher/backward-Euler startup, despite CLAUDE.md's "Key
objects" section describing a BE/CN mix. ``SolverMetadata.scheme_per_step``
reports this honestly (every entry is ``"CN"``); it does not fabricate a
startup scheme that does not exist in the trusted solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cgmy import _external  # noqa: F401  (must precede any cgmy_bspline import)

from cgmy_bspline import grid as cgmy_grid
from cgmy_bspline.parameters import CGMYParams
from cgmy_bspline.solver_cn import solve_cn

_SUPPORTED_MODES = ("banded", "toeplitz", "toeplitz_precond")


@dataclass(frozen=True)
class CGMYParameters:
    """mu = (C, G, M, Y, r, T): the CGMY Levy triplet, rate, and maturity.

    Attributes:
        C: Overall activity/scale of the Lévy density. C > 0.
        G: Right-tail (downward jump) decay rate. G > 0.
        M: Left-tail (upward jump) decay rate. M > 1 (risk-neutral moment
           condition; enforced by the wrapped solver's ``CGMYParams``).
        Y: Fine-structure / singularity index. Y in (0, 2), Y != 1.
        r: Risk-free rate.
        T: Maturity.
    """

    C: float
    G: float
    M: float
    Y: float
    r: float
    T: float


@dataclass(frozen=True)
class EuropeanPutPayoff:
    """The only payoff currently supported by the wrapped reference solver.

    Attributes:
        strike: Strike price K.
    """

    strike: float


# Alias used in type hints elsewhere in this package. Extend with a Union
# once a second payoff has validated boundary conditions (see module
# docstring); until then, EuropeanPutPayoff is the only member.
Payoff = EuropeanPutPayoff


@dataclass(frozen=True)
class Mesh:
    """Spatial discretization request for ``solve_cgmy``.

    Attributes:
        N: Number of spatial intervals (n_dof = N + p - 1).
        p: B-spline order/degree (the wrapped solver expects p=3 throughout;
           other values are untested there).
        x_min: Manual left domain endpoint (log-price). Leave None, together
           with x_max, to auto-select the domain via
           ``cgmy_bspline.grid.make_grid`` (domain_scale/G, domain_scale/M).
        x_max: Manual right domain endpoint (log-price).
        domain_scale: Passed to ``make_grid`` when x_min/x_max are None.
    """

    N: int
    p: int = 3
    x_min: float | None = None
    x_max: float | None = None
    domain_scale: float = 10.0


@dataclass(frozen=True)
class TimeGrid:
    """Uniform time-stepping request: dt = mu.T / n_steps.

    The wrapped solver only supports a uniform time grid; nonuniform
    stepping (e.g. Rannacher-graded near tau=0) would be new numerics and is
    out of scope for Stage A.
    """

    n_steps: int


@dataclass(frozen=True)
class MeshInfo:
    """Realized mesh geometry, as actually built for a given solve."""

    N: int
    p: int
    n_dof: int
    h: float
    x_min: float
    x_max: float
    nodes: np.ndarray


@dataclass(frozen=True)
class SolverMetadata:
    """Per-solve and per-step bookkeeping from the trusted solver.

    Attributes:
        mode: Which of the trusted solver's three modes was used
              ('banded' | 'toeplitz' | 'toeplitz_precond').
        dt: Time step size, mu.T / n_steps.
        n_steps: Number of time steps taken (== TimeGrid.n_steps).
        scheme_per_step: Time-integration scheme used at each step. Always
              all-"CN" for the current trusted solver (no Rannacher/BE
              startup exists there yet -- see module docstring).
        iterations_per_step: Linear-solver iteration count at each step (1
              for the direct 'banded' LU solve; GMRES iteration count for
              the Toeplitz modes).
        tolerance_per_step: Linear-solver tolerance at each step (None for
              the direct solve; GMRES rtol, currently 1e-10, for the
              Toeplitz modes).
        bc_dofs: (left_dof_indices, right_dof_indices) where Dirichlet
              boundary values were enforced.
        cpu_time: Wall-clock time for the whole solve, in seconds.
        mean_gmres_iters: Average GMRES iterations per step (None for
              mode='banded').
    """

    mode: str
    dt: float
    n_steps: int
    scheme_per_step: list[str]
    iterations_per_step: list[int]
    tolerance_per_step: list[float | None]
    bc_dofs: tuple[list[int], list[int]]
    cpu_time: float
    mean_gmres_iters: float | None


@dataclass(frozen=True)
class GalerkinTrajectory:
    """Full output of a single reference-solver run.

    Attributes:
        coefficients: B-spline coefficients at every time level,
              shape (n_steps + 1, n_dof); coefficients[0] is the (boundary-
              adjusted) initial condition, coefficients[-1] the final state.
        forcing_vectors: CN-averaged forcing (f(tau_n) + f(tau_{n+1})) / 2
              applied at each step, shape (n_steps, n_dof), or None if this
              solve was run with store_forcing=False. Certification code
              (``certification.residuals``) must raise rather than recompute
              or substitute zeros when this is None -- never treat "not
              stored" as "zero forcing". When populated, it is currently
              all-zero anyway: the trusted solver's
              ``boundary_forcing_vector`` is an explicit "zero-forcing
              approximation" (see ``cgmy_bspline.boundary``), not a
              limitation introduced here.
        initial_coefficients: Alias for coefficients[0], kept explicit since
              certification/correction code will want it without indexing.
        time_grid: Time nodes, shape (n_steps + 1,), from 0 to mu.T.
        parameter_vector: The CGMY parameters (mu) used for this solve.
        payoff: The payoff used for this solve.
        mesh_information: The realized mesh geometry.
        solver_metadata: Per-solve and per-step solver bookkeeping.
    """

    coefficients: np.ndarray
    forcing_vectors: np.ndarray | None
    initial_coefficients: np.ndarray
    time_grid: np.ndarray
    parameter_vector: CGMYParameters
    payoff: Payoff
    mesh_information: MeshInfo
    solver_metadata: SolverMetadata


def solve_cgmy(
    mu: CGMYParameters,
    payoff: Payoff,
    mesh: Mesh,
    time_grid: TimeGrid,
    *,
    mode: str = "toeplitz",
    bandwidth: int = 64,
    store_forcing: bool = True,
) -> GalerkinTrajectory:
    """Solve the CGMY pricing PIDE via the trusted external Galerkin solver.

    This function performs no discretization or time-stepping itself; it
    builds the trusted solver's own parameter/grid objects, calls
    ``cgmy_bspline.solver_cn.solve_cn`` with ``record_history=True``, and
    repackages the result into a ``GalerkinTrajectory``.

    Args:
        mu: CGMY Lévy triplet, rate, and maturity.
        payoff: Contract payoff. Only ``EuropeanPutPayoff`` is supported
            (see module docstring for why).
        mesh: Spatial discretization request.
        time_grid: Uniform time-stepping request.
        mode: Which solver mode to use -- 'banded' (fast, bandwidth-truncated
            approximation), 'toeplitz' (exact, matrix-free, GMRES), or
            'toeplitz_precond' (exact, GMRES with a banded preconditioner).
            Defaults to 'toeplitz': the exact nonlocal operator, matching the
            "preserve the existing Toeplitz/FFT acceleration ... single
            implementation used everywhere downstream" requirement.
        bandwidth: Truncation half-bandwidth for mode='banded', and the
            preconditioner bandwidth for mode='toeplitz_precond'. Unused for
            mode='toeplitz'.
        store_forcing: If False, the trajectory's forcing_vectors is None
            instead of populated from the solver (cheap escape hatch for
            callers that don't need it; the solve itself is unaffected).
            Never zeroed -- None must mean "not computed", distinct from a
            genuinely zero forcing vector, so residual code can raise
            instead of silently treating "not stored" as "zero".

    Returns:
        The full solution trajectory, mesh, and solver metadata.

    Raises:
        NotImplementedError: If ``payoff`` is not an ``EuropeanPutPayoff``.
        ValueError: If ``mode`` is not one of the trusted solver's three
            modes, or if exactly one of ``mesh.x_min``/``mesh.x_max`` is set.
    """
    if not isinstance(payoff, EuropeanPutPayoff):
        raise NotImplementedError(
            "solve_cgmy only supports EuropeanPutPayoff: the wrapped solver's "
            "Dirichlet boundary conditions are derived specifically for the "
            "European put's far-field asymptotics. Supporting another payoff "
            "requires deriving its boundary conditions first (new numerics, "
            "out of scope for Stage A)."
        )
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"mode must be one of {_SUPPORTED_MODES}, got {mode!r}")
    if (mesh.x_min is None) != (mesh.x_max is None):
        raise ValueError("Mesh.x_min and Mesh.x_max must both be set, or both left None")

    params = CGMYParams(C=mu.C, G=mu.G, M=mu.M, Y=mu.Y, r=mu.r, K=payoff.strike, T=mu.T)

    if mesh.x_min is not None:
        grid_dict = cgmy_grid.make_grid_manual(mesh.x_min, mesh.x_max, mesh.N)
    else:
        grid_dict = cgmy_grid.make_grid(params, mesh.N, domain_scale=mesh.domain_scale)

    result = solve_cn(
        params,
        N=grid_dict["N"],
        p=mesh.p,
        N_tau=time_grid.n_steps,
        bandwidth=bandwidth,
        mode=mode,
        grid=grid_dict,
        record_history=True,
    )

    n_dof = grid_dict["N"] + mesh.p - 1

    mesh_info = MeshInfo(
        N=grid_dict["N"],
        p=mesh.p,
        n_dof=n_dof,
        h=grid_dict["h"],
        x_min=grid_dict["x_min"],
        x_max=grid_dict["x_max"],
        nodes=grid_dict["nodes"],
    )

    step_metadata = result["step_metadata"]
    metadata = SolverMetadata(
        mode=mode,
        dt=result["dt"],
        n_steps=time_grid.n_steps,
        scheme_per_step=[m["scheme"] for m in step_metadata],
        iterations_per_step=[m["iterations"] for m in step_metadata],
        tolerance_per_step=[m["tol"] for m in step_metadata],
        bc_dofs=result["bc_dofs"],
        cpu_time=result["cpu_time"],
        mean_gmres_iters=result["gmres_iters"],
    )

    forcing = result["forcing_history"] if store_forcing else None

    return GalerkinTrajectory(
        coefficients=result["coeffs_history"],
        forcing_vectors=forcing,
        initial_coefficients=result["c_init"],
        time_grid=np.linspace(0.0, mu.T, time_grid.n_steps + 1),
        parameter_vector=mu,
        payoff=payoff,
        mesh_information=mesh_info,
        solver_metadata=metadata,
    )
