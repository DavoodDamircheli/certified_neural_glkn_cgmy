"""v5 M9: memory-safe extensions of the independent Fourier pricers must not change
historical values (validation/cos_pricer.py, validation/independent_fourier_integration.py)."""
from __future__ import annotations

import sys

import numpy as np

from common.run_manifest import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
import validation.cos_pricer as cp  # noqa: E402
from validation.cgmy_reference_pricing import CGMYMarketParams  # noqa: E402
from validation.independent_fourier_integration import carr_madan_put_price  # noqa: E402

EPS = np.finfo(float).eps


def test_cos_blocked_path_agrees_with_dense_to_rounding(monkeypatch):
    """Forced blocked path vs the historical dense path at the same N: same dot
    products, different BLAS reduction order -> agreement to n_nodes*eps relative."""
    dense = cp._cosine_coeffs_put(-8.8, 8.9, 257, 100.0)
    monkeypatch.setattr(cp, "_COS_DENSE_LIMIT", 0)
    blocked = cp._cosine_coeffs_put(-8.8, 8.9, 257, 100.0)
    n_nodes = 8 * max(200, 4 * 257)
    assert np.max(np.abs(dense - blocked)) <= n_nodes * EPS * np.max(np.abs(dense))


def test_cos_historical_sizes_use_dense_path():
    assert 1024 * 8 * max(200, 4 * 1024) <= cp._COS_DENSE_LIMIT


def test_carr_madan_panelled_matches_historical_quadrature():
    mu = CGMYMarketParams(C=0.3723044017493263, G=2.611772942503739, M=3.3090385440108108,
                          Y=1.4137219209190033, r=0.06651469757443157, T=0.7851279704841273)  # box_00
    h = carr_madan_put_price(mu, 100.0, 100.0, dps=40)
    p = carr_madan_put_price(mu, 100.0, 100.0, dps=40, panel_width=8.0)
    assert h["v_cutoff"] == p["v_cutoff"] and not h["cutoff_cap_hit"]
    assert abs(h["price"] - p["price"]) <= 4 * EPS * abs(h["price"])
