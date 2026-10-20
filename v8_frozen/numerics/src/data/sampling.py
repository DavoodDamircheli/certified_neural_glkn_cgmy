"""Sampling strategies over the frozen CGMY/contract parameter box.

Uses scipy's quasi-Monte Carlo module (Sobol preferred, Latin hypercube as a
fallback) -- no tensor-product grids, so cost scales linearly with sample
count regardless of the 6 free dimensions (C, G, M, Y, r, maturity; strike
is fixed, not sampled).

Also builds the five named, permanently-stored splits requested for Stage D:
train_interior, interpolation_test, edge_test, stress_test, ood_test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff, Mesh, TimeGrid
from common.config_compat import read_spline_order

_DIM_ORDER = ("C", "G", "M", "Y", "r", "maturity")
Sample = tuple[CGMYParameters, EuropeanPutPayoff]


@dataclass(frozen=True)
class ParameterBox:
    """Box constraints for CGMY and contract parameters.

    Each of ``C``, ``G``, ``M``, ``Y``, ``r``, ``maturity`` is a (low, high)
    bound sampled from; ``strike`` is fixed, not sampled. Loaded from
    ``configs/base.yaml``.
    """

    C: tuple[float, float]
    G: tuple[float, float]
    M: tuple[float, float]
    Y: tuple[float, float]
    r: tuple[float, float]
    maturity: tuple[float, float]
    strike: float


def load_parameter_box(path: str | Path) -> ParameterBox:
    """Load the frozen parameter box from configs/base.yaml."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    box = cfg["parameter_box"]
    strike = float(cfg["contract"]["strike"])
    return ParameterBox(
        C=tuple(box["C"]), G=tuple(box["G"]), M=tuple(box["M"]),
        Y=tuple(box["Y"]), r=tuple(box["r"]), maturity=tuple(box["maturity"]),
        strike=strike,
    )


def load_mesh_and_time_grid(path: str | Path) -> tuple[Mesh, TimeGrid]:
    """Load the frozen production mesh and time grid from
    configs/base.yaml's ``discretization`` block -- the single source of
    truth CLAUDE.md Hard Invariant #6 requires.

    Before this function existed, ``experiments/exp_common.py``,
    ``scripts/generate_production_dataset.py``, ``scripts/lossablation.py``,
    and ``scripts/stage_f_baselines_report.py`` each independently
    hardcoded ``Mesh(N=64, p=3, x_min=..., x_max=...)`` and
    ``TimeGrid(n_steps=60)`` as Python literals -- duplicated four times,
    and never read from this config (see zips/hard_invariant_audit.md,
    finding #5). All four now call this function instead.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    d = cfg["discretization"]
    mesh = Mesh(
        N=int(d["n_elements"]), p=read_spline_order(d),
        x_min=float(d["domain_left"]), x_max=float(d["domain_right"]),
    )
    time_grid = TimeGrid(n_steps=int(d["n_time_steps"]))
    return mesh, time_grid


def _bounds(box: ParameterBox) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([getattr(box, d)[0] for d in _DIM_ORDER], dtype=float)
    hi = np.array([getattr(box, d)[1] for d in _DIM_ORDER], dtype=float)
    return lo, hi


def parameter_distance_from_box(mu: CGMYParameters, box: ParameterBox) -> float:
    """Normalized L2 distance of `mu` from `box`: 0 if `mu` is inside the
    box in every dimension; otherwise the L2 norm, across dimensions, of
    each dimension's excess beyond the box edge, normalized by that
    dimension's own box width (so no single wide-range dimension, e.g. M
    in [2,9], dominates the metric just from having larger raw units than
    e.g. Y in [1.10,1.80]). Used (pre-submission fixes, item 1.3) to check
    that a resampled OOD tier's distance from the training box stays
    comparable to what it replaced, keeping the interior -> near -> edge
    -> beyond hierarchy monotone in distance, not just in name."""
    lo, hi = _bounds(box)
    vals = np.array([mu.C, mu.G, mu.M, mu.Y, mu.r, mu.T])
    excess = np.maximum(0.0, np.maximum(lo - vals, vals - hi)) / (hi - lo)
    return float(np.linalg.norm(excess))


def _shrink(lo: np.ndarray, hi: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    margin = frac * (hi - lo)
    return lo + margin, hi - margin


def _rows_to_samples(rows: np.ndarray, strike: float) -> list[Sample]:
    payoff = EuropeanPutPayoff(strike=strike)
    return [
        (CGMYParameters(C=row[0], G=row[1], M=row[2], Y=row[3], r=row[4], T=row[5]), payoff)
        for row in rows
    ]


def sample_sobol(
    box: ParameterBox, n_samples: int, seed: int, shrink_frac: float = 0.0
) -> list[Sample]:
    """Draw a scrambled Sobol sequence over the (optionally shrunk) box."""
    lo, hi = _bounds(box)
    if shrink_frac:
        lo, hi = _shrink(lo, hi, shrink_frac)
    sampler = qmc.Sobol(d=len(_DIM_ORDER), scramble=True, seed=seed)
    unit = sampler.random(n_samples)
    return _rows_to_samples(qmc.scale(unit, lo, hi), box.strike)


def sample_latin_hypercube(
    box: ParameterBox, n_samples: int, seed: int, shrink_frac: float = 0.0
) -> list[Sample]:
    """Draw a Latin-hypercube sample over the (optionally shrunk) box."""
    lo, hi = _bounds(box)
    if shrink_frac:
        lo, hi = _shrink(lo, hi, shrink_frac)
    sampler = qmc.LatinHypercube(d=len(_DIM_ORDER), seed=seed)
    unit = sampler.random(n_samples)
    return _rows_to_samples(qmc.scale(unit, lo, hi), box.strike)


def sample_grid(box: ParameterBox, n_per_dim: int) -> list[Sample]:
    """Tensor-product grid -- NOT used for training data (explicitly excluded by
    the task: "no high-dimensional tensor grids"). Kept only for small
    diagnostic sweeps (n_per_dim small, e.g. Stage A's corner script), where
    the combinatorial blow-up is intentional and bounded by the caller.
    """
    lo, hi = _bounds(box)
    axes = [np.linspace(lo[i], hi[i], n_per_dim) for i in range(len(_DIM_ORDER))]
    mesh = np.meshgrid(*axes, indexing="ij")
    rows = np.stack([m.ravel() for m in mesh], axis=1)
    return _rows_to_samples(rows, box.strike)


def _edge_samples(box: ParameterBox, n_samples: int, seed: int) -> list[Sample]:
    """Each sample has exactly one randomly chosen dimension snapped to its
    box min/max; the rest are drawn uniformly within the full box."""
    lo, hi = _bounds(box)
    rng = np.random.default_rng(seed)
    rows = rng.uniform(lo, hi, size=(n_samples, len(_DIM_ORDER)))
    edge_dims = rng.integers(0, len(_DIM_ORDER), size=n_samples)
    edge_side = rng.integers(0, 2, size=n_samples)
    for i in range(n_samples):
        rows[i, edge_dims[i]] = lo[edge_dims[i]] if edge_side[i] == 0 else hi[edge_dims[i]]
    return _rows_to_samples(rows, box.strike)


def _stress_samples(box: ParameterBox, n_samples: int, seed: int) -> list[Sample]:
    """Biased toward the corners Stage A/C found numerically stressed: high Y
    (poor conditioning, kappa(S_h) ~ 6000 near Y=1.8), low C/G, high M."""
    lo, hi = _bounds(box)
    rng = np.random.default_rng(seed)
    # Sample each dimension from the outer quarter of its range, biased
    # toward the stressed side found in Stage A/C: high Y, low C, low G.
    def biased(dim_lo, dim_hi, low_side: bool, frac=0.25):
        span = dim_hi - dim_lo
        return (dim_lo, dim_lo + frac * span) if low_side else (dim_hi - frac * span, dim_hi)

    ranges = [
        biased(lo[0], hi[0], low_side=True),   # C: low
        biased(lo[1], hi[1], low_side=True),   # G: low
        biased(lo[2], hi[2], low_side=False),  # M: high
        biased(lo[3], hi[3], low_side=False),  # Y: high (kappa(S_h) blow-up)
        (lo[4], hi[4]),                        # r: full range
        (lo[5], hi[5]),                        # maturity: full range
    ]
    rows = np.array([[rng.uniform(a, b) for a, b in ranges] for _ in range(n_samples)])
    return _rows_to_samples(rows, box.strike)


def _ood_samples(
    box: ParameterBox, n_samples: int, seed: int, widen_frac: float = 0.2, return_stats: bool = False
) -> list[Sample] | tuple[list[Sample], dict]:
    """Sobol over a widened box, keeping only points strictly outside the
    original box in at least one dimension AND satisfying every CGMY
    admissibility constraint -- a genuine out-of-distribution test set,
    not merely "outside the training box."

    CHANGED (pre-submission fixes, item 1.3): C>0 and G>0 used to be left
    to `CGMYParams`'s own downstream ValueError (a "clean, catchable
    failure" the caller was expected to skip) -- but that meant candidate
    points with C<0 or G<0 (both outside the CGMY model's OWN admissible
    domain, not merely outside the training distribution) were routinely
    DRAWN and only discovered inadmissible at solve time, silently
    shrinking the effective "beyond" tier size (exp05_ood.py's own last
    run: 7/50 such candidates were drawn and discarded there). C and G
    are now floored here directly, the same way M/Y/r/maturity already
    are, so every returned sample is admissible by construction --
    nothing downstream needs to catch a ValueError for this anymore.

    Args:
        return_stats: If True, return (samples, stats) instead of just
            samples, where stats records n_drawn (total raw QMC
            candidates generated, across every batch), n_rejected_outside
            (candidates inside the ORIGINAL box, i.e. not OOD at all),
            n_rejected_inadmissible (candidates outside the box but
            violating admissibility -- should be near-zero now, since
            floors are applied before drawing, not after), and n_kept.
    """
    lo, hi = _bounds(box)
    wide_lo, wide_hi = lo - widen_frac * (hi - lo), hi + widen_frac * (hi - lo)
    # Respect the model's hard constraints regardless of the widened box --
    # EVERY dimension with a hard admissibility floor/ceiling, not just the
    # two (r, maturity) that would silently corrupt a solve instead of
    # raising. C>0, G>0, M>1 are all genuine CGMY-model admissibility
    # requirements (see cgmy_bspline.parameters.CGMYParams), not just
    # convenient numerical guards.
    wide_lo[0] = max(wide_lo[0], 0.01)    # C > 0
    wide_lo[1] = max(wide_lo[1], 0.01)    # G > 0
    wide_lo[2] = max(wide_lo[2], 1.001)   # M > 1
    wide_lo[3] = max(wide_lo[3], 0.01)    # Y > 0
    wide_hi[3] = min(wide_hi[3], 1.99)    # Y < 2
    wide_lo[4] = max(wide_lo[4], 0.0)     # r >= 0
    wide_lo[5] = max(wide_lo[5], 0.01)    # maturity > 0

    sampler = qmc.Sobol(d=len(_DIM_ORDER), scramble=True, seed=seed)
    accepted: list[np.ndarray] = []
    n_drawn = 0
    n_rejected_outside = 0  # candidates that landed INSIDE the original box (not OOD)
    batch = max(64, 2 * n_samples)
    while len(accepted) < n_samples:
        unit = sampler.random(batch)
        rows = qmc.scale(unit, wide_lo, wide_hi)
        n_drawn += len(rows)
        outside = np.any((rows < lo) | (rows > hi), axis=1)
        n_rejected_outside += int(np.sum(~outside))
        accepted.extend(rows[outside])
        batch *= 2
        if batch > 1 << 16:
            break
    rows = np.array(accepted[:n_samples])

    # Admissibility assertion (item 1.3's explicit ask): every returned row
    # must satisfy every hard CGMY constraint, by construction -- fail
    # loudly here rather than let an inadmissible sample slip through to a
    # downstream solve-time ValueError again.
    assert np.all(rows[:, 0] > 0), "OOD sampler produced C <= 0 -- admissibility floor is broken"
    assert np.all(rows[:, 1] > 0), "OOD sampler produced G <= 0 -- admissibility floor is broken"
    assert np.all(rows[:, 2] > 1), "OOD sampler produced M <= 1 -- admissibility floor is broken"
    assert np.all((rows[:, 3] > 0) & (rows[:, 3] < 2)), "OOD sampler produced Y outside (0,2) -- admissibility floor is broken"
    assert np.all(rows[:, 4] >= 0), "OOD sampler produced r < 0 -- admissibility floor is broken"
    assert np.all(rows[:, 5] > 0), "OOD sampler produced maturity <= 0 -- admissibility floor is broken"

    samples = _rows_to_samples(rows, box.strike)
    if not return_stats:
        return samples
    stats = {
        "n_drawn": n_drawn, "n_rejected_outside": n_rejected_outside,
        "n_rejected_inadmissible": 0,  # floored before drawing now -- see docstring
        "n_kept": len(samples),
    }
    return samples, stats


def generate_named_splits(
    box: ParameterBox,
    seed: int,
    counts: dict[str, int] | None = None,
) -> dict[str, list[Sample]]:
    """Build the five named splits, each deterministic given (box, seed, counts).

    Args:
        counts: Number of samples per split; defaults to
            {train_interior: 100, interpolation_test: 30, edge_test: 30,
             stress_test: 20, ood_test: 20}.
    """
    counts = counts or {
        "train_interior": 100,
        "interpolation_test": 30,
        "edge_test": 30,
        "stress_test": 20,
        "ood_test": 20,
    }
    return {
        "train_interior": sample_sobol(box, counts["train_interior"], seed=seed, shrink_frac=0.15),
        "interpolation_test": sample_sobol(box, counts["interpolation_test"], seed=seed + 1, shrink_frac=0.15),
        "edge_test": _edge_samples(box, counts["edge_test"], seed=seed + 2),
        "stress_test": _stress_samples(box, counts["stress_test"], seed=seed + 3),
        "ood_test": _ood_samples(box, counts["ood_test"], seed=seed + 4),
    }


def _sample_to_dict(sample: Sample) -> dict:
    mu, payoff = sample
    return {"C": mu.C, "G": mu.G, "M": mu.M, "Y": mu.Y, "r": mu.r, "T": mu.T, "strike": payoff.strike}


def _dict_to_sample(d: dict) -> Sample:
    return (
        CGMYParameters(C=d["C"], G=d["G"], M=d["M"], Y=d["Y"], r=d["r"], T=d["T"]),
        EuropeanPutPayoff(strike=d["strike"]),
    )


def save_splits(splits: dict[str, list[Sample]], path: str | Path) -> None:
    """Permanently record split membership as JSON, so a dataset's exact
    sampled points survive independent of the code/seed that produced them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: [_sample_to_dict(s) for s in samples] for name, samples in splits.items()}
    path.write_text(json.dumps(payload, indent=2))


def load_splits(path: str | Path) -> dict[str, list[Sample]]:
    """Load previously saved split membership."""
    payload = json.loads(Path(path).read_text())
    return {name: [_dict_to_sample(d) for d in samples] for name, samples in payload.items()}
