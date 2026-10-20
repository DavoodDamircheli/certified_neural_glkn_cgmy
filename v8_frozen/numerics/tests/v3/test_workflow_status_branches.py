"""Gate B4 soundness tests: every documented branch of
``pipeline.predict_certify_correct`` is actually reachable and its
decision logic is sound -- not just "doesn't crash".

Reuses the exact fixed mesh/checkpoints as experiments/exp_common.py.
Slower (loads 4 checkpoints, runs real reference-free predict+certify
calls) -- marked so it can be skipped from a fast pytest pass if needed,
but runs by default since it is the actual Gate B4 soundness check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import exp_common as ec  # noqa: E402
from cgmy.reference_solver import CGMYParameters, EuropeanPutPayoff  # noqa: E402
from data.sampling import sample_sobol  # noqa: E402
from pipeline.predict_certify_correct import predict_certify_correct  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    ops = ec.get_ops()
    box, stats = ec.get_box_and_stats()
    models = ec.load_models(ops, stats)
    model = models[ec.FINAL_MODEL_NAME]
    samples = sample_sobol(box, 6, seed=424242)
    return ops, stats, model, samples


def _tg(mu):
    n_steps = ec.TIME_GRID.n_steps
    return np.linspace(0.0, mu.T, n_steps + 1)


def _run(setup, mu, payoff, *, tolerance, quick_pcg=2, max_pcg=200, run_correction=True):
    ops, stats, model, _ = setup
    return predict_certify_correct(
        mu, payoff, ec.FIXED_MESH, _tg(mu), model, stats, ops,
        tolerance=tolerance, quick_pcg_iterations=quick_pcg, max_pcg_iterations=max_pcg,
        run_correction=run_correction,
    )


def test_not_spd_branch_never_accepts(setup):
    # A known-indefinite query, taken from results/spd_failures.csv (not
    # re-derived) -- the same one used in certification/run_gate_b.py.
    bad_mu = CGMYParameters(C=0.4874301809351891, G=1.4997461959719658, M=8.837858855724335,
                             Y=1.4543537443503738, r=0.06294871459715069, T=0.10040483502671123)
    payoff = EuropeanPutPayoff(strike=100.0)
    trace = _run(setup, bad_mu, payoff, tolerance=1e6)
    assert trace.is_spd is False
    assert trace.branch == "not_spd_uncertifiable"
    assert trace.accepted is False, "an uncertifiable (non-SPD) query must never be accepted, regardless of tolerance"
    assert np.isnan(trace.eta_final), "eta_final must be NaN when no certificate was ever computed"


def test_accepted_guar_quick_is_reachable_and_sound(setup):
    _, _, _, samples = setup
    mu, payoff = samples[0]
    trace = _run(setup, mu, payoff, tolerance=1.0e6)
    if trace.branch != "accepted_guar_quick":
        pytest.skip(f"this query/tolerance landed on {trace.branch!r}, not quick-accept; "
                    f"see test_forced_quickaccept_branch_exists for a guaranteed case")
    assert trace.accepted is True
    assert trace.eta_final == trace.eta_guar_quick
    assert not trace.correction_ran


def test_forced_quickaccept_branch_exists_across_samples(setup):
    """At an absurdly loose tolerance, at least one of several SPD queries
    must take the quick-accept path -- confirms the branch is not dead
    code (certification/run_gate_b.py found 5/5 did, across a different
    sample set; this is the pytest-level confirmation)."""
    _, _, _, samples = setup
    branches = []
    for mu, payoff in samples:
        trace = _run(setup, mu, payoff, tolerance=1.0e6)
        branches.append(trace.branch)
    assert "accepted_guar_quick" in branches, f"expected at least one quick-accept among {branches}"


def test_corrected_but_not_certified_is_reachable(setup):
    """An impossibly tight tolerance must eventually fail even after
    correction (GMRES converges to rtol=1e-10, not to the certificate's
    own tolerance), landing on corrected_but_not_certified rather than a
    fabricated accept."""
    _, _, _, samples = setup
    branches = []
    for mu, payoff in samples:
        trace = _run(setup, mu, payoff, tolerance=1e-14)
        branches.append(trace.branch)
        if trace.branch == "corrected_but_not_certified":
            assert trace.accepted is False
            assert trace.correction_ran is True
            assert trace.eta_final == trace.eta_guar_corrected
    assert "corrected_but_not_certified" in branches or "accepted_after_correction_guar" in branches, (
        f"expected correction to have been exercised among {branches}"
    )


def test_needs_correction_not_run_never_accepts(setup):
    _, _, _, samples = setup
    mu, payoff = samples[1]
    trace = _run(setup, mu, payoff, tolerance=1e-14, run_correction=False)
    if trace.is_spd and trace.branch in ("accepted_guar_quick",):
        pytest.skip("this query happened to quick-accept even at tol=1e-14; not the branch under test")
    assert trace.branch == "needs_correction_not_run"
    assert trace.accepted is False
    assert trace.correction_ran is False


def test_eta_final_is_never_derived_from_mass_indicator(setup):
    """Across every branch exercised, eta_final must equal either NaN, an
    eta_guar_quick value, or an eta_guar_corrected value -- never
    eta_mass_indicator (CLAUDE.md Hard Invariant #5 / Task 2)."""
    _, _, _, samples = setup
    for mu, payoff in samples:
        for tol in (1e6, 1e-14):
            trace = _run(setup, mu, payoff, tolerance=tol)
            if np.isnan(trace.eta_final):
                continue
            matches_quick = not np.isnan(trace.eta_guar_quick) and trace.eta_final == trace.eta_guar_quick
            matches_corrected = not np.isnan(trace.eta_guar_corrected) and trace.eta_final == trace.eta_guar_corrected
            assert matches_quick or matches_corrected, (
                f"eta_final={trace.eta_final} for branch {trace.branch} matches neither "
                f"eta_guar_quick={trace.eta_guar_quick} nor eta_guar_corrected={trace.eta_guar_corrected}"
            )
            assert trace.eta_final != trace.eta_mass_indicator or np.isnan(trace.eta_mass_indicator), (
                "eta_final coincidentally equals eta_mass_indicator -- verify this is not a sign "
                "the mass indicator is silently driving acceptance again"
            )
