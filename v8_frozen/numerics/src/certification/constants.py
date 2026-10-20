"""Stability constants underlying the guaranteed certificate.

("Guaranteed" throughout this package means: guaranteed by an
exact-arithmetic theorem under its stated hypotheses; every number is
evaluated in ordinary floating point -- see common.methods.NUMERICAL_SCOPE.)

Computes, for a fixed mesh and mu, the spectral quantities that convert a
PCG-based Riesz residual into a bound (exact-arithmetic theorem): the smallest/largest
eigenvalues of the (explicitly assembled, full n_dof) symmetrized operator
S_h(mu), the mass matrix's extreme eigenvalues, the resulting condition
numbers, and the standard PCG convergence-rate bound. All assembly reuses
``cgmy.operators.OperatorCache`` (the single operator implementation); this
module only adds eigenvalue computations on top of it.

IMPORTANT -- scope of what is and is not implemented here:

The manuscript this project is based on names several additional constants
(c_1, gamma_inf(h), and possibly others) that this module does NOT compute,
because their precise definitions were not available when this file was
written. Fabricating a formula for a symbol I have not seen defined would
risk silently mislabeling a numerical quantity as a "certified constant" --
exactly the failure mode CLAUDE.md's hard invariants exist to prevent. Where
a manuscript symbol has an unambiguous, standard meaning inferable from its
name (coercivity/continuity constants, Lambda_1(h)/Lambda_2(h) as the
h-dependent extreme eigenvalues, kappa(mu) as their ratio,
lambda_min(M_h)), it is computed and the mapping is stated explicitly in
the docstring below. Where it is not inferable (c_1, gamma_inf(h)), the
field is left ``None`` with a TODO. See the Stage C report for the full
list.

Task 3 (working prompt) -- lambda_min_numerical vs. alpha_certified:

``np.linalg.eigh``'s reported smallest eigenvalue of the explicitly
assembled S_h(mu) is an extremely accurate NUMERICAL ESTIMATE of
lambda_min(S_h(mu)) -- LAPACK's dense symmetric eigensolvers are backward
stable -- but "accurate estimate" is not the same thing as "certified lower
bound", and ``certification.riesz_solver``'s upper-bound formula is only
valid given a GENUINE lower bound (its own docstring says so explicitly).
This module now reports both, never conflated: ``lambda_min_numerical``
(the raw estimate, for diagnostics) and ``alpha_lower_estimate`` (formerly
``alpha_certified``; the old name remains as a read-only alias): a value that
is a lower bound on lambda_min(S) IN EXACT ARITHMETIC under the named
strategy's hypotheses (Weyl's inequality with exactly orthogonal
eigenvectors), evaluated in ordinary floating point -- see
``coercivity_strategy`` and notes/03_coercivity.md. v5 scope (roadmap §4.2):
it is a floating-point estimate, not a validated enclosure;
``alpha <= lambda_min(S)`` remains an explicit HYPOTHESIS of the
upper-bracket theorem in ``riesz_solver``. Every certificate consumer
(``mass_indicator``, ``guaranteed_certificate``, ``riesz_solver``) must be
passed this value, never ``lambda_min_numerical``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from cgmy.operators import OperatorCache
from cgmy.reference_solver import CGMYParameters


@dataclass(frozen=True)
class StabilityConstants:
    """Numerically estimated stability constants for one (mesh, mu) pair.

    All eigenvalue quantities are computed by explicit dense assembly of
    the full n_dof operator (via repeated OperatorCache.apply_S/apply_M on
    unit vectors) followed by ``numpy.linalg.eigh`` -- practical only at
    moderate n_dof (see ``compute_stability_constants``'s ``max_n_dof``
    guard). This mirrors ``guaranteed_certificate.eta_sharp``'s own
    "explicitly assemble and factor, practical sizes only" approach, so the
    coercivity constant reported here is consistent with what eta_sharp
    itself relies on.

    Attributes:
        lambda_min_numerical: The raw numerical estimate of
            lambda_min(S_h(mu)) over ALL n_dof dofs (not restricted to
            interior dofs -- see certification.residuals module docstring
            for why the full operator, not an interior submatrix, is used
            throughout this certification stack) -- NOT itself a certified
            bound (Task 3, working prompt). Never NaN-gated on is_spd: even
            a negative value is diagnostically meaningful. Also reported as
            Lambda_1(h) below.
        alpha_lower_estimate: Floating-point evaluation of a quantity that is
            <= lambda_min(S_h(mu)) in exact arithmetic under the strategy
            named in ``coercivity_strategy`` (v5: renamed from
            ``alpha_certified``, which survives as a read-only alias; it was
            never a validated enclosure). NaN if lambda_min_numerical <= 0.
            The ONLY coercivity-related quantity certificate code
            (mass_indicator, guaranteed_certificate, riesz_solver) may
            consume; alpha <= lambda_min is a hypothesis of their bounds.
        safety_ratio: alpha_lower_estimate / lambda_min_numerical (in (0, 1],
            NaN if either is NaN/non-positive) -- how much margin the
            chosen strategy gave up relative to the raw estimate; report
            this alongside any certificate (Task 3, working prompt).
        coercivity_strategy: Which named strategy actually produced
            alpha_lower_estimate for this (mesh, mu): "B_weyl_residual_enclosure"
            (default; a lower bound in exact arithmetic, evaluated in
            floating point -- see notes/03_coercivity.md),
            "C_safety_margin" (non-rigorous fallback, must be labeled as
            such wherever reported), or "uncertified" (lambda_min_numerical
            <= 0, no bound attempted).
        continuity_constant: gamma = lambda_max(S_h(mu)). Also reported as
            Lambda_2_h below.
        Lambda_1_h: Alias for lambda_min_numerical, naming its
            h-dependence explicitly (my inferred reading of the
            manuscript's Lambda_1(h) symbol -- please confirm; if the
            manuscript's Lambda_1(h) instead means the CERTIFIED constant,
            that is alpha_certified, not this field).
        Lambda_2_h: Alias for continuity_constant (Lambda_2(h), same caveat).
        condition_number_S_h: cond(S_h(mu)) = Lambda_2_h / Lambda_1_h, the
            ordinary matrix condition number of S_h(mu) (a diagnostic
            computed from the numerical estimate, not a certified bound).
            NOTE (fix/efficiency-norm-kappa): this is NOT the manuscript's
            kappa(mu) from Lemma~\ref{lem:kappa} -- that quantity is
            ``kappa_operator_norm(mu, ops)`` / ``kappa_operator_norm_matrix_free``
            below, kappa(mu) := ||S_h(mu)^{-1/2} A_h(mu) S_h(mu)^{-1/2}||_2.
            The two can differ by orders of magnitude (e.g. A_h(mu)=S_h(mu)
            gives kappa(mu)=1 exactly while cond(S_h(mu)) can be huge) -- do
            not use this field as kappa in any efficiency-bound computation.
        lambda_min_M_h: Smallest eigenvalue of the (unconstrained) mass
            matrix M_h.
        lambda_max_M_h: Largest eigenvalue of M_h.
        mass_preconditioned_condition_number: kappa(M_h^{-1} S_h(mu)) --
            my inferred reading of "preconditioner constants": the
            condition number of the mass-preconditioned system PCG
            actually solves in ``certification.riesz_solver`` with
            preconditioner='mass'. Please confirm this is what the
            manuscript means by that phrase.
        pcg_convergence_rate: Standard CG linear convergence-rate bound
            rho = (sqrt(kappa)-1)/(sqrt(kappa)+1) for the mass-preconditioned
            system -- my inferred reading of "Riesz convergence quantities".
        c_1: NOT COMPUTED. Symbol meaning unknown -- see module docstring.
        gamma_inf_h: NOT COMPUTED. Symbol meaning unknown (possibly an
            inf-sup constant, or a domain-truncation/far-field decay
            constant) -- see module docstring.
        mesh_N, mesh_p, x_min, x_max: The mesh this was computed for.
        mu: The CGMY parameters this was computed for.
        is_spd: Whether alpha_lower_estimate > 0, i.e. S_h(mu) passes the SPD
            gate as judged by the floating-point lower-bound estimate (Task 3,
            working prompt: this gates on that estimate, not on the raw
            lambda_min_numerical -- stricter than before, though
            empirically indistinguishable in practice, see
            notes/03_coercivity.md). If False, alpha_certified/kappa/
            pcg_convergence_rate are NaN and the guaranteed certificate
            cannot be trusted at this mu -- see Stage A's difficult-corners
            findings for known SPD-losing corners.
        coordinate_space: "legacy_full" (the raw n_dof operator, including
            the 4 Dirichlet-prescribed dofs that are not actually part of
            the variational problem -- see reports/BASIS_AND_BOUNDARY_AUDIT.md)
            or "constrained_free" (the n_free interior/variational operator
            S_I = E^T S_raw E, see certification.constrained). Set by the
            caller (compute_stability_constants's `ops` argument determines
            which space the eigendecomposition actually ran in; this field
            only labels the result, it does not change any computation) --
            CGMY_V4_REMAINING_WORK.md item 1.
    """

    lambda_min_numerical: float
    alpha_lower_estimate: float
    safety_ratio: float
    coercivity_strategy: str
    continuity_constant: float
    Lambda_1_h: float
    Lambda_2_h: float
    condition_number_S_h: float
    lambda_min_M_h: float
    lambda_max_M_h: float
    mass_preconditioned_condition_number: float
    pcg_convergence_rate: float
    c_1: float | None
    gamma_inf_h: float | None
    mesh_N: int
    mesh_p: int
    x_min: float
    x_max: float
    mu: CGMYParameters
    is_spd: bool
    coordinate_space: str = "legacy_full"

    @property
    def alpha_certified(self) -> float:
        """DEPRECATED alias of ``alpha_lower_estimate`` (pre-v5 name), kept so
        historical scripts and stored-result readers run unchanged. The value
        is a floating-point estimate, not a machine-validated bound."""
        return self.alpha_lower_estimate


def dense_from_apply(apply_fn, n: int) -> np.ndarray:
    """Materialize an n x n dense matrix from a matrix-free apply function.

    Shared by ``compute_stability_constants`` and
    ``guaranteed_certificate.eta_sharp`` -- both need an explicit dense
    S_h(mu) "at practical sizes only"; this is the one place that builds it.
    """
    M = np.zeros((n, n))
    e = np.zeros(n)
    for j in range(n):
        e[j] = 1.0
        M[:, j] = apply_fn(e)
        e[j] = 0.0
    return M


def weyl_residual_enclosure(S: np.ndarray, eigs: np.ndarray, V: np.ndarray) -> float:
    """Option B (Task 3, working prompt): a lower bound on lambda_min(S) from
    Weyl's inequality -- exact GIVEN exactly orthogonal eigenvectors and exact
    arithmetic (see the scope caveat below); the returned number is a
    floating-point evaluation of it.

    Derivation. Let V (n x n, columns v_1..v_n) and eigs (mu_1<=...<=mu_n)
    be ``numpy.linalg.eigh(S)``'s own output for symmetric S: V is
    orthogonal to machine precision and mu_i = v_i^T S v_i exactly (since
    S v_i = mu_i v_i and ||v_i||=1). Let D = diag(mu_1,...,mu_n) and
    E := V^T S V - D (symmetric, since S and D are; zero diagonal by
    construction). Because V is orthogonal, S and D+E = V^T S V share the
    same eigenvalues (orthogonal similarity). Weyl's inequality for
    symmetric matrices (unconditional -- no gap or genericity assumption)
    gives, for every i, |lambda_i(D+E) - lambda_i(D)| <= ||E||_2, i.e.

        |lambda_i(S) - mu_i| <= ||E||_2   for every i,

    since lambda_i(D) = mu_i exactly (D is diagonal with the mu_i already
    sorted ascending, by construction of eigh's output). In particular for
    i=1 (the smallest):

        lambda_min(S) >= mu_1 - ||E||_2 >= mu_1 - ||E||_F

    (the last step uses the standard, always-true ||.||_2 <= ||.||_F, to
    avoid a second eigendecomposition just to get the spectral norm of E --
    ||E||_F is a single sqrt-sum-of-squares over E's off-diagonal entries).

    Empirically (see notes/03_coercivity.md): on this project's actual
    S_h(mu) matrices (production mesh, Sobol samples across the frozen
    parameter box), ||E||_F is O(1e-13) to O(1e-14) -- i.e. this bound
    matches the raw numerical estimate to ~13 significant digits. No
    CGMY-specific mathematics is used anywhere in this derivation -- it is
    general symmetric-matrix perturbation theory, applicable to any S this
    project might ever assemble.

    SCOPE CAVEAT (CGMY_V4_REMAINING_WORK.md item 2, added v4 -- do not
    delete without re-reading that item): the derivation above is exact
    GIVEN that ``V`` is an exactly orthogonal matrix. ``numpy.linalg.eigh``
    does not return an exactly orthogonal ``V`` -- only one that is
    orthogonal to floating-point rounding error (machine epsilon, backward
    stability of LAPACK's symmetric eigensolver). This function does not
    bound or account for that nonorthogonality; it evaluates ``E`` and
    ``||E||_F`` in ordinary double-precision floating point, using the
    computed (not exactly orthogonal) ``V`` as if it were exact. The
    resulting ``alpha_certified`` is therefore an ALGEBRAICALLY exact lower
    bound only under the idealization "the eigendecomposition is exact" --
    it is a rigorous theorem given exact arithmetic, evaluated empirically
    (and, on this project's own matrices, checked to be extremely tight) in
    double precision, NOT a validated/interval-arithmetic enclosure that
    accounts for ``eigh``'s own floating-point error, nonorthogonality of
    ``V``, or this function's own Frobenius-norm evaluation. Do not describe
    ``alpha_certified`` or any certificate built from it as "machine-
    validated" -- see certification/CERTIFICATE_ARITHMETIC_SPEC.md (Option 1
    scope) and reports/v4_validated_arithmetic_plan.md for what an actually
    validated (Option 2) route would additionally require.
    """
    D = np.diag(eigs)
    E = V.T @ S @ V - D
    e_frob = float(np.linalg.norm(E, ord="fro"))
    return float(eigs[0]) - e_frob


def compute_stability_constants(
    mu: CGMYParameters,
    ops: OperatorCache,
    max_n_dof: int = 400,
    *,
    coercivity_strategy: str = "B_weyl_residual_enclosure",
    coercivity_safety_delta: float = 0.1,
    coordinate_space: str = "legacy_full",
) -> StabilityConstants:
    """Compute stability constants for one (mesh, mu) pair via dense assembly.

    Args:
        ops: Anything exposing the OperatorCache interface
            (apply_S/apply_M/n_dof) -- a raw ``cgmy.operators.OperatorCache``
            (full n_dof, "legacy_full") or a
            ``certification.constrained.FreeOperatorCache`` (n_free,
            "constrained_free"). This function's own math is identical
            either way; only `coordinate_space` (a label, below) and which
            object happened to be passed in distinguish the two calls.
        coordinate_space: Label stored on the returned ``StabilityConstants``
            ("legacy_full" or "constrained_free") -- purely descriptive,
            does not affect the computation. Callers passing a
            ``FreeOperatorCache`` must pass "constrained_free" here.
        coercivity_strategy: How alpha_certified is obtained -- see
            configs/base.yaml's certification block and
            notes/03_coercivity.md. "B_weyl_residual_enclosure" (default,
            rigorous), "C_safety_margin" (non-rigorous fallback, explicitly
            labeled wherever it flows downstream), or "A_analytical" (not
            implemented -- raises).
        coercivity_safety_delta: Only consumed by "C_safety_margin":
            alpha_certified = (1 - coercivity_safety_delta) * lambda_min_numerical.

    Raises:
        ValueError: If ops.n_dof > max_n_dof (dense eigendecomposition is
            not "practical sizes" beyond this -- raise rather than silently
            run an expensive/slow computation), or if coercivity_strategy
            is not a recognized name.
        NotImplementedError: If coercivity_strategy == "A_analytical".
    """
    n = ops.n_dof
    if n > max_n_dof:
        raise ValueError(
            f"n_dof={n} exceeds max_n_dof={max_n_dof}: dense eigendecomposition "
            f"is only practical at moderate mesh sizes. Pass a smaller mesh, or "
            f"raise max_n_dof if you have verified the cost is acceptable."
        )

    S = dense_from_apply(lambda v: ops.apply_S(mu, v), n)
    S_sym_err = np.max(np.abs(S - S.T))
    assert S_sym_err < 1e-8, f"apply_S is not producing a symmetric matrix: max asymmetry {S_sym_err:.3e}"

    # eigh (not eigvalsh): weyl_residual_enclosure needs the full eigenbasis
    # V, not just the eigenvalues.
    eigs_S, V_S = np.linalg.eigh(S)
    lambda_min_S, lambda_max_S = float(eigs_S[0]), float(eigs_S[-1])

    if lambda_min_S <= 0:
        alpha_certified = float("nan")
        coercivity_strategy_used = "uncertified"
    elif coercivity_strategy == "B_weyl_residual_enclosure":
        alpha_certified = weyl_residual_enclosure(S, eigs_S, V_S)
        coercivity_strategy_used = "B_weyl_residual_enclosure"
    elif coercivity_strategy == "C_safety_margin":
        alpha_certified = (1.0 - coercivity_safety_delta) * lambda_min_S
        coercivity_strategy_used = "C_safety_margin"
    elif coercivity_strategy == "A_analytical":
        raise NotImplementedError(
            "coercivity_strategy='A_analytical' is not implemented: it would "
            "require a from-scratch coercivity derivation for the CGMY "
            "bilinear form, which this project does not have verified "
            "access to (no manuscript reference available) -- see "
            "notes/03_coercivity.md for the full assessment and why this "
            "was not attempted. Use 'B_weyl_residual_enclosure' (default, "
            "rigorous) or 'C_safety_margin' (fallback, non-rigorous)."
        )
    else:
        raise ValueError(
            f"unknown coercivity_strategy {coercivity_strategy!r}; expected "
            f"'B_weyl_residual_enclosure', 'C_safety_margin', or 'A_analytical'"
        )

    if alpha_certified == alpha_certified and alpha_certified <= 0:
        # The strategy ran but could not certify positivity (only possible
        # for C_safety_margin when lambda_min_S is already tiny+positive,
        # or in principle a future strategy) -- report honestly as
        # uncertified rather than a non-positive "certified" bound.
        alpha_certified = float("nan")
        coercivity_strategy_used = "uncertified"

    is_spd = alpha_certified == alpha_certified and alpha_certified > 0  # certified SPD, not just numerically so
    safety_ratio = (
        alpha_certified / lambda_min_S
        if is_spd and lambda_min_S > 0
        else float("nan")
    )

    M = dense_from_apply(ops.apply_M, n)
    eigs_M = np.linalg.eigvalsh(M)
    lambda_min_M, lambda_max_M = float(eigs_M[0]), float(eigs_M[-1])

    if is_spd:
        kappa = lambda_max_S / lambda_min_S
        # Generalized eigenvalue problem S x = lambda M x -- these are the
        # true eigenvalues of the mass-preconditioned operator M^{-1} S.
        # M^{-1} S itself is not symmetric even though M and S both are, so
        # eigh(S, M) (not eigvalsh(M^{-1} S) or any symmetrization of it) is
        # the correct way to get its (real, since M, S SPD) eigenvalues.
        eigs_precond = sla.eigh(S, M, eigvals_only=True)
        kappa_precond = (
            float(eigs_precond[-1] / eigs_precond[0]) if eigs_precond[0] > 0 else float("nan")
        )
        sqrt_kappa = np.sqrt(kappa_precond) if kappa_precond == kappa_precond and kappa_precond > 0 else float("nan")
        rho = (sqrt_kappa - 1.0) / (sqrt_kappa + 1.0) if sqrt_kappa == sqrt_kappa else float("nan")
    else:
        kappa = float("nan")
        kappa_precond = float("nan")
        rho = float("nan")

    return StabilityConstants(
        lambda_min_numerical=lambda_min_S,
        alpha_lower_estimate=alpha_certified,
        safety_ratio=safety_ratio,
        coercivity_strategy=coercivity_strategy_used,
        continuity_constant=lambda_max_S,
        Lambda_1_h=lambda_min_S,
        Lambda_2_h=lambda_max_S,
        condition_number_S_h=kappa,
        lambda_min_M_h=lambda_min_M,
        lambda_max_M_h=lambda_max_M,
        mass_preconditioned_condition_number=kappa_precond,
        pcg_convergence_rate=rho,
        c_1=None,
        gamma_inf_h=None,
        mesh_N=ops.N,
        mesh_p=ops.p,
        x_min=ops.x_min,
        x_max=ops.x_max,
        mu=mu,
        is_spd=is_spd,
        coordinate_space=coordinate_space,
    )


def write_constants_table_tex(constants: list[StabilityConstants], path: str | Path) -> None:
    """Emit a LaTeX table of stability constants for one or more (mesh, mu) points."""
    lines = [
        r"\begin{tabular}{lrrrrrrrl}",
        r"\toprule",
        r"$(C,G,M,Y,r,T)$ & $N$ & $\Lambda_1(h)$ & $\alpha_{\mathrm{cert}}$ & safety & $\Lambda_2(h)$ & "
        r"$\kappa(\mu)$ & $\lambda_{\min}(M_h)$ & SPD (strategy) \\",
        r"\midrule",
    ]
    for c in constants:
        mu = c.mu
        mu_str = f"({mu.C:.3g},{mu.G:.3g},{mu.M:.3g},{mu.Y:.3g},{mu.r:.3g},{mu.T:.3g})"
        alpha_str = f"{c.alpha_certified:.3e}" if c.is_spd else "n/a"
        ratio_str = f"{c.safety_ratio:.6f}" if c.is_spd else "n/a"
        spd_str = f"{'yes' if c.is_spd else 'NO'} ({c.coercivity_strategy})"
        lines.append(
            f"{mu_str} & {c.mesh_N} & {c.Lambda_1_h:.3e} & {alpha_str} & {ratio_str} & {c.Lambda_2_h:.3e} & "
            f"{c.condition_number_S_h:.3e} & {c.lambda_min_M_h:.3e} & "
            f"{spd_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def triple_norm_squared(
    pred: np.ndarray,
    true: np.ndarray,
    mu: CGMYParameters,
    ops: OperatorCache,
    dt: float,
    *,
    max_n_dof: int = 400,
) -> float:
    r"""The manuscript's exact efficiency-theorem norm (main.tex, eq:triple-norm):

        |||e|||^2 = dt * sum_{n=0}^{N_T-1} [
            || M_h (e^{n+1}-e^n)/dt ||_{S_h(mu)^{-1}}^2
            + || (e^{n+1}+e^n)/2 ||_{S_h(mu)}^2
        ]

    where e = pred - true (coefficient trajectories, shape (N_T+1, n_dof)).
    Both terms are required -- the time-derivative term (in the S_h^{-1}
    dual norm) is the discrete analogue of ||d_tau e||_{L2(0,T;V')}, and the
    midpoint term (in the S_h energy norm) is the discrete analogue of
    ||e||_{L2(0,T;V)}; omitting either one is not the norm the efficiency
    theorem (thm:efficiency) is stated in. See
    CGMY_presubmission_review_and_efficiency_fix_notes.md for the bug this
    replaces (a mass/L2-only, single-term quantity).

    S_h(mu) is assembled densely ONCE and Cholesky-factored ONCE (never
    formed as S_h^{-1} explicitly); the same factorization is reused for
    every time step's S_h^{-1} term via ``scipy.linalg.cho_solve``, mirroring
    ``guaranteed_certificate.eta_sharp``'s own dense-assemble-and-factor-once
    pattern.

    Raises:
        ValueError: If ops.n_dof > max_n_dof.
        np.linalg.LinAlgError: If S_h(mu) is not SPD (Cholesky fails).
    """
    n = ops.n_dof
    if n > max_n_dof:
        raise ValueError(
            f"n_dof={n} exceeds max_n_dof={max_n_dof}: dense factorization of "
            f"S_h(mu) is only practical at moderate mesh sizes."
        )
    S = dense_from_apply(lambda v: ops.apply_S(mu, v), n)
    try:
        cho = sla.cho_factor(S)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"S_h(mu) is not SPD (Cholesky factorization failed) for mu={mu}: "
            f"the triple norm's S_h^{{-1}} term is not meaningful at this "
            f"parameter point."
        ) from exc

    diff = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    total = 0.0
    for k in range(len(diff) - 1):
        e_n = diff[k]
        e_np1 = diff[k + 1]
        Mde = ops.apply_M((e_np1 - e_n) / dt)
        time_term = float(Mde @ sla.cho_solve(cho, Mde))
        ebar = 0.5 * (e_np1 + e_n)
        space_term = float(ebar @ ops.apply_S(mu, ebar))
        total += dt * (time_term + space_term)
    return total


def mass_l2_error_squared_legacy(pred: np.ndarray, true: np.ndarray, ops: OperatorCache, dt: float) -> float:
    """The OLD (pre-Step-1-fix) mass/L2 quantity dt*sum_n ||e^n||_{M_h}^2 --
    kept ONLY for side-by-side comparison against ``triple_norm_squared`` in
    diagnostics/reports. This is NOT the manuscript's |||e|||^2 (it has no
    time-derivative term and is not the norm thm:efficiency is stated in) --
    never use it in the actual bound check."""
    diff = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    return float(dt * sum(float(d @ ops.apply_M(d)) for d in diff))


def kappa_operator_norm(mu: CGMYParameters, ops: OperatorCache, *, max_n_dof: int = 400) -> float:
    r"""The manuscript's exact kappa (main.tex, eq:kappa-def, Lemma~\ref{lem:kappa}):

        kappa(mu) = || S_h(mu)^{-1/2} A_h(mu) S_h(mu)^{-1/2} ||_2

    Generic in ``ops`` (anything exposing apply_S/apply_A/n_dof): passing a
    raw ``cgmy.operators.OperatorCache`` computes the legacy full-space
    kappa(mu); passing a ``certification.constrained.FreeOperatorCache``
    computes kappa_I(mu) = ||S_I(mu)^{-1/2} A_I(mu) S_I(mu)^{-1/2}||_2
    (CGMY_V4_REMAINING_WORK.md item 1) with no code change -- the two
    differ only in which operator was passed in.

    Computed stably via a single Cholesky factorization S_h(mu) = L L^T and
    two triangular solves (never forming S_h^{-1} or A_h^T S_h^{-1} A_h
    explicitly, which squares the conditioning):

        X = L^{-1} A_h         (forward solve L X = A_h)
        B^T = L^{-1} X^T       (forward solve, so B = X L^{-T} = L^{-1} A_h L^{-T})
        kappa = largest singular value of B

    (B need not be symmetric since A_h need not be; for a Cholesky factor
    L, using L in place of the symmetric square root S_h^{1/2} gives the
    same operator 2-norm -- L = S_h^{1/2} U for an orthogonal U depending
    only on S_h, and pre/post-multiplying by an orthogonal matrix never
    changes singular values.)

    A_h=S_h gives kappa=1 EXACTLY in exact arithmetic (B is then the
    identity), however large cond(S_h) is; this is the case the old
    cond(S_h)-as-kappa convention got wrong. In floating point the computed
    B carries a forward error of first order 3*n*gamma_{n+1}*cond(S_h)
    (Cholesky + two triangular solves, each backward stable), so the
    computed kappa deviates from 1 by O(eps*cond(S_h)) -- e.g. 1.6e-9 at
    cond 1e8, n=12 (v5 M4, scripts/v5_kappa_investigation.py). Note also
    that ``sla.cholesky`` reads one triangle of the assembled S, while A is
    used in full: if the assembled S is not bitwise symmetric, the computed
    quantity is kappa of (A, lower-triangle S), which differs from kappa of
    (A, sym(A)) by O(||S - S^T|| / lambda_min(S)). See
    tests/test_efficiency_norm_kappa.py for the derived tolerances.

    Raises:
        ValueError: If ops.n_dof > max_n_dof.
        np.linalg.LinAlgError: If S_h(mu) is not SPD (Cholesky fails).
    """
    n = ops.n_dof
    if n > max_n_dof:
        raise ValueError(
            f"n_dof={n} exceeds max_n_dof={max_n_dof}: dense assembly/Cholesky of "
            f"S_h(mu)/A_h(mu) is only practical at moderate mesh sizes. Use "
            f"kappa_operator_norm_matrix_free for larger n_dof."
        )
    S = dense_from_apply(lambda v: ops.apply_S(mu, v), n)
    A = dense_from_apply(lambda v: ops.apply_A(mu, v), n)
    try:
        L = sla.cholesky(S, lower=True)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"S_h(mu) is not SPD (Cholesky factorization failed) for mu={mu}: "
            f"kappa(mu) is not meaningful at this parameter point."
        ) from exc
    X = sla.solve_triangular(L, A, lower=True)
    B = sla.solve_triangular(L, X.T, lower=True).T
    sv = np.linalg.svd(B, compute_uv=False)
    return float(sv[0])


def kappa_operator_norm_matrix_free(
    mu: CGMYParameters,
    ops: OperatorCache,
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
    seed: int = 0,
) -> float:
    r"""Matrix-free kappa(mu) for n_dof beyond ``kappa_operator_norm``'s dense
    reach (Step 2, working prompt): the top eigenvalue of the generalized
    eigenproblem

        A_h(mu)^T S_h(mu)^{-1} A_h(mu) v = kappa(mu)^2 S_h(mu) v

    via LOBPCG, with S_h(mu)^{-1} itself applied by an inner (matrix-free)
    CG solve against ``ops.apply_S`` -- no dense n_dof x n_dof matrix is ever
    formed. Intended for large n_dof; validated at SMALL n_dof against
    ``kappa_operator_norm`` as a cross-check (tests/test_efficiency_norm_kappa.py),
    per the working prompt's explicit instruction that the dense generalized
    eigenproblem is "acceptable ONLY as a small-N cross-check", not the
    large-N production path.

    `tol` is both LOBPCG's convergence tolerance and the inner CG's rtol --
    report this value alongside any kappa computed this way.

    Raises:
        RuntimeError: If the inner CG solve for S_h(mu)^{-1} fails to converge.
    """
    from scipy.sparse.linalg import LinearOperator, cg, lobpcg

    n = ops.n_dof

    def s_matvec(v: np.ndarray) -> np.ndarray:
        # Toeplitz matvec (cgmy_bspline.toeplitz.ToeplitzOperator.matvec)
        # requires a strictly 1-D input; LOBPCG's block-vector matmat calls
        # this per-column as shape (n, 1), so ravel defensively.
        return ops.apply_S(mu, np.asarray(v, dtype=float).ravel())

    S_op = LinearOperator((n, n), matvec=s_matvec, dtype=float)

    def s_inv_matvec(v: np.ndarray) -> np.ndarray:
        x, info = cg(S_op, np.asarray(v, dtype=float).ravel(), rtol=tol, atol=0.0, maxiter=max_iter)
        if info != 0:
            raise RuntimeError(
                f"inner CG solve for S_h(mu)^-1 did not converge (info={info}) "
                f"while computing kappa_operator_norm_matrix_free"
            )
        return x

    def c_matvec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).ravel()
        return ops.apply_AT(mu, s_inv_matvec(ops.apply_A(mu, v)))

    C_op = LinearOperator((n, n), matvec=c_matvec, dtype=float)

    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, 1))
    eigvals, _ = lobpcg(C_op, X0, B=S_op, largest=True, tol=tol, maxiter=max_iter)
    kappa_sq = float(eigvals[0])
    return float(np.sqrt(max(kappa_sq, 0.0)))


def load_constants(path: str) -> StabilityConstants:
    """Load previously computed stability constants from disk (e.g. YAML/JSON)."""
    raise NotImplementedError


def save_constants(constants: StabilityConstants, path: str) -> None:
    """Persist stability constants to disk."""
    raise NotImplementedError
