"""Task 14 (working prompt): sampling with a VARYING strike, so a trained
model's branch input `psi` (the discretized payoff) actually differs
across training queries -- see `notes/14_operator_not_parameter_map.md`
for the full investigation this module implements.

Deliberately separate from `data.sampling`: `ParameterBox.strike` there is
a single frozen scalar (CLAUDE.md's `contract.strike: 100.0 # K -- fixed,
not sampled`), used by every other experiment/checkpoint in this project,
and this module does not touch it. Varying the strike is scoped to this
one ablation via its own box/sampler, not a project-wide change.

K range and domain adequacy: `[70, 150]` (see
`configs/base.yaml`'s new `operator_generalization` block, the single
source of truth for this range -- CLAUDE.md Hard Invariant #6 spirit,
applied to this ablation's own frozen setting). Validated empirically,
not assumed: solving one representative `mu` at K=70/100/150 on the
project's existing FIXED mesh (`x_min=log(100)-8, x_max=log(100)+8`)
gives `max_residual` ~2-5e-6 at every K (same order as the K=100
baseline), and `S_h(mu)`'s SPD-ness is UNAFFECTED by K entirely -- `A_h`/
`M_h`/`S_h` are built from `mu` and the mesh alone; K only enters the
Dirichlet boundary values and the terminal payoff projection (both
already fully K-general, confirmed by direct code inspection -- see the
investigation note), never the operator itself. So there is no
SPD/coercivity risk from varying K, only a domain-truncation-adequacy
question, and that was checked directly to be fine over this range (the
payoff kink at `log(K)` stays >=7.5 log-units from either domain edge for
every K in `[70, 150]`, essentially unchanged from K=100's own ~8-unit
margin).

`STRIKE_HOLDOUT = (95.0, 105.0)`: a gap band excluded from every TRAINING
sample, reserved for the "unseen strike" generalization tier -- and
including K=100 itself, so the SAME query the old (K=100-only) checkpoint
was exclusively trained on doubles as a fair common reference point in
the operator-generalization comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff
from data.sampling import ParameterBox, Sample, _DIM_ORDER, _bounds

# Fallback defaults, kept in sync with configs/base.yaml's
# `operator_generalization` block (the authoritative source -- see
# `load_operator_range`, CLAUDE.md Hard Invariant #6 spirit).
STRIKE_MIN = 70.0
STRIKE_MAX = 150.0
STRIKE_HOLDOUT = (95.0, 105.0)

_OP_DIM_ORDER = _DIM_ORDER + ("strike",)


def load_operator_range(path: str | Path) -> tuple[float, float, tuple[float, float]]:
    """Read (strike_min, strike_max, strike_holdout) from configs/base.yaml's
    `operator_generalization` block -- the single source of truth for this
    ablation's own frozen range (CLAUDE.md Hard Invariant #6)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    og = cfg["operator_generalization"]
    return float(og["strike_min"]), float(og["strike_max"]), tuple(float(v) for v in og["strike_holdout"])


@dataclass(frozen=True)
class OperatorParameterBox:
    """Same 6 CGMY/contract dimensions as `data.sampling.ParameterBox`,
    plus `strike` as a SEVENTH sampled dimension over
    `[strike_min, strike_max]`, instead of `ParameterBox`'s single frozen
    scalar."""

    C: tuple[float, float]
    G: tuple[float, float]
    M: tuple[float, float]
    Y: tuple[float, float]
    r: tuple[float, float]
    maturity: tuple[float, float]
    strike_min: float
    strike_max: float


def operator_box_from(
    box: ParameterBox, strike_min: float = STRIKE_MIN, strike_max: float = STRIKE_MAX
) -> OperatorParameterBox:
    """Build an OperatorParameterBox reusing the frozen box's 6 CGMY/
    contract ranges (C, G, M, Y, r, maturity) unchanged -- only the strike
    dimension differs from `box`. Pass `strike_min`/`strike_max` from
    `load_operator_range` (configs/base.yaml) rather than relying on the
    module defaults where the config is available."""
    return OperatorParameterBox(
        C=box.C, G=box.G, M=box.M, Y=box.Y, r=box.r, maturity=box.maturity,
        strike_min=strike_min, strike_max=strike_max,
    )


def _op_bounds(box: OperatorParameterBox) -> tuple[np.ndarray, np.ndarray]:
    lo6, hi6 = _bounds(box)  # duck-typed: box has the same 6 attrs ParameterBox does
    lo = np.append(lo6, box.strike_min)
    hi = np.append(hi6, box.strike_max)
    return lo, hi


def _rows_to_operator_samples(rows: np.ndarray) -> list[Sample]:
    out = []
    for row in rows:
        mu = CGMYParameters(C=row[0], G=row[1], M=row[2], Y=row[3], r=row[4], T=row[5])
        payoff = EuropeanPutPayoff(strike=float(row[6]))
        out.append((mu, payoff))
    return out


def sample_operator_sobol(
    box: OperatorParameterBox, n_samples: int, seed: int, exclude_strike_range: tuple[float, float] | None = None
) -> list[Sample]:
    """Joint Sobol draw over all 7 dimensions (C, G, M, Y, r, maturity,
    strike). If `exclude_strike_range` is given, points whose strike falls
    inside that (inclusive) range are rejected and redrawn -- used to keep
    `STRIKE_HOLDOUT` entirely out of the training distribution."""
    lo, hi = _op_bounds(box)
    sampler = qmc.Sobol(d=len(_OP_DIM_ORDER), scramble=True, seed=seed)
    accepted: list[np.ndarray] = []
    batch = max(64, 2 * n_samples)
    while len(accepted) < n_samples:
        unit = sampler.random(batch)
        rows = qmc.scale(unit, lo, hi)
        if exclude_strike_range is not None:
            lo_ex, hi_ex = exclude_strike_range
            keep = (rows[:, 6] < lo_ex) | (rows[:, 6] > hi_ex)
            rows = rows[keep]
        accepted.extend(rows)
        batch *= 2
        if batch > 1 << 16:
            break
    rows = np.array(accepted[:n_samples])
    return _rows_to_operator_samples(rows)


def sample_operator_gap(box: OperatorParameterBox, n_samples: int, seed: int,
                         strike_range: tuple[float, float] = STRIKE_HOLDOUT) -> list[Sample]:
    """(C, G, M, Y, r, maturity) drawn from the full box as usual; strike
    drawn ONLY from `strike_range` (the held-out gap) -- the "unseen
    strike" generalization tier."""
    lo6, hi6 = _bounds(box)
    sampler = qmc.Sobol(d=len(_DIM_ORDER) + 1, scramble=True, seed=seed)
    unit = sampler.random(n_samples)
    lo = np.append(lo6, strike_range[0])
    hi = np.append(hi6, strike_range[1])
    rows = qmc.scale(unit, lo, hi)
    return _rows_to_operator_samples(rows)


def sample_operator_combo(box: OperatorParameterBox, n_samples: int, seed: int,
                           frac: float = 0.15) -> list[Sample]:
    """Deliberately CORRELATED, edge-biased draw: each sample has (mu, K)
    jointly pushed toward a combination style a joint-Sobol-over-the-full-
    box training set under-represents -- one randomly chosen mu-dimension
    snapped near its box edge, PAIRED with strike snapped near the strike
    range's own edge (same side, i.e. low-mu-edge with low-strike-edge or
    high-mu-edge with high-strike-edge) -- the "unseen combination" tier.
    Same style as `data.sampling._edge_samples`, extended to correlate the
    edge choice with strike rather than leaving strike independent."""
    lo6, hi6 = _bounds(box)
    rng = np.random.default_rng(seed)
    rows = rng.uniform(lo6, hi6, size=(n_samples, len(_DIM_ORDER)))
    edge_dims = rng.integers(0, len(_DIM_ORDER), size=n_samples)
    edge_side = rng.integers(0, 2, size=n_samples)
    strikes = np.empty(n_samples)
    for i in range(n_samples):
        d = edge_dims[i]
        if edge_side[i] == 0:
            rows[i, d] = lo6[d] + frac * (hi6[d] - lo6[d])
            strikes[i] = box.strike_min + frac * (box.strike_max - box.strike_min)
        else:
            rows[i, d] = hi6[d] - frac * (hi6[d] - lo6[d])
            strikes[i] = box.strike_max - frac * (box.strike_max - box.strike_min)
    rows = np.concatenate([rows, strikes[:, None]], axis=1)
    return _rows_to_operator_samples(rows)
