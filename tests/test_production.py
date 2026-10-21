"""Follow-up 4: the production configuration and the N = 1024 pipeline's done-when checks (results/N1024.json)."""
import json

import pytest

from conftest import ROOT

PC = ROOT / "results" / "PRODUCTION_CONFIG.json"
N1 = ROOT / "results" / "N1024.json"
pytestmark = pytest.mark.skipif(not (PC.exists() and N1.exists()), reason="run scripts/run_E3_1.py and scripts/run_N1024.py")


def test_production_config_frozen():
    pc = json.loads(PC.read_text())
    assert (pc["N"], pc["N_T"]) == (1024, 240)
    assert abs(pc["eps_N"] - 0.01 / pc["max_beta_I"]) < 1e-15 and abs(pc["eps_N"] - 1.1846e-3) < 1e-6
    assert pc["max_price_err"] <= 0.01


def test_n1024_checks():
    d = json.loads(N1.read_text())
    c = d["checks"]
    assert c["full_basis_vs_direct"]["max_rel_err"] < 1e-10
    assert c["online_vs_full_perturbed"]["max_uncompressed_eta_rel_diff"] < 1e-10
    assert c["online_vs_full_pod_galerkin"]["max_uncompressed_eta_rel_diff"] < 3e-10          # eps*kappa floor
    for k in ("online_vs_full_perturbed", "online_vs_full_pod_galerkin"):
        assert c[k]["all_brackets_ok_to_floor"]
    for Q, t in c["timing_ms"]["per_Q"].items():
        assert t["online_certificate_total"] < 5.0                                                # idle, one thread
    assert d["cells"]["validity"]["n_below_1"] == 0
    assert d["cells"]["effectivity"]["median_over_etaX"] <= 2.0


def test_n1024_pod_galerkin_no_false_accepts():
    s = json.loads(N1.read_text())["pod_galerkin_table"]["summary"]
    for r, v in s.items():
        for b in ("banded_AI", "exact_AI", "reduced", "reduced_tightened"):
            assert all(x == 0 for x in v[b]["false_accepts"].values()), (r, b)
