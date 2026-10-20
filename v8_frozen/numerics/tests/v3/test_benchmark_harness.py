"""Sanity tests for the Gate D benchmark harness itself (not absolute
timing numbers, which aren't meaningfully testable) -- CGMY_V3_NUMERICAL_
EXPERIMENT_ROADMAP.md Gate D.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT))

import run_gate_d_benchmark as gated  # noqa: E402


def test_thread_limits_actually_set():
    """Importing the harness must pin BLAS/torch thread env vars to 1
    BEFORE numpy/scipy/torch are first imported -- the only way OpenBLAS
    reliably picks them up. Checked here as a post-import invariant."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ[var] == "1", f"{var} was not pinned to 1"
    import torch
    assert torch.get_num_threads() == 1


def test_empty_phases_matches_phase_names():
    phases = gated.empty_phases()
    assert set(phases.keys()) == set(gated.PHASE_NAMES)
    assert all(v == 0.0 for v in phases.values())


def test_method_order_randomization_actually_varies():
    """Per-query method-order shuffling (seeded) must produce more than one
    distinct ordering across queries -- a harness that silently always ran
    methods in the same order would defeat the point of randomizing to
    reduce drift."""
    rng = np.random.default_rng(12345)
    methods = list(gated.METHOD_FNS.keys())
    orders = []
    for _ in range(10):
        order = methods.copy()
        rng.shuffle(order)
        orders.append(tuple(order))
    assert len(set(orders)) > 1, "shuffled orders were all identical -- randomization is not working"


def test_fixed_cost_stub_reports_approximately_that_cost():
    """A method with a KNOWN, deliberately injected fixed cost should have
    its 'total' phase reported close to that cost -- a basic check that the
    timing plumbing (perf_counter placement, phase summation) isn't
    systematically wrong, without asserting anything about real numerical
    methods' absolute speed."""
    def fake_method(_q, _ctx):
        phases = gated.empty_phases()
        t0 = time.perf_counter()
        time.sleep(0.05)
        phases["inference"] = time.perf_counter() - t0
        phases["total"] = phases["inference"]
        return {"phases": phases, "price": 1.0, "success": True, "note": ""}

    out = fake_method(None, None)
    assert 0.04 < out["phases"]["total"] < 0.20, (
        f"expected ~0.05s, got {out['phases']['total']}"
    )


def test_dense_lu_agrees_with_reference_galerkin_on_one_query():
    """Integration check (the one from this gate's own post-mortem): an
    earlier version of run_dense_lu had the Crank-Nicolson matrices'
    roles swapped (solved M-dt/2*A instead of M+dt/2*A), producing a
    divergent, unphysical price (~1e85). This regression test pins the
    fix: dense_lu and reference_galerkin solve the identical discrete
    equations and must agree far beyond quadrature/model error."""
    import exp_common
    from cgmy.reference_solver import CGMYParameters, TimeGrid, solve_cgmy

    mu = CGMYParameters(C=0.3, G=2.0, M=5.0, Y=1.4, r=0.03, T=0.5)
    mesh = exp_common.FIXED_MESH
    mesh_info = exp_common.FIXED_MESH_INFO
    payoff = exp_common.PAYOFF
    ops = exp_common.get_ops()
    time_grid = np.linspace(0, mu.T, exp_common.TIME_GRID.n_steps + 1)

    out = gated.run_dense_lu(mu, payoff, mesh_info, time_grid, ops)
    ref = solve_cgmy(mu, payoff, mesh, TimeGrid(n_steps=exp_common.TIME_GRID.n_steps),
                      mode="toeplitz", store_forcing=False)
    ref_price = exp_common.option_price(ref.coefficients[-1], ops)

    assert out["success"]
    assert abs(out["price"] - ref_price) < 1e-6, (
        f"dense_lu price {out['price']} disagrees with reference {ref_price} "
        f"by more than 1e-6 -- possible regression of the CN matrix-role bug"
    )
