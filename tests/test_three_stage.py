"""Follow-up 5, item 2: the three-stage rule (reduced -> banded -> exact) at N = 1024."""
import numpy as np
import pytest

from conftest import ROOT

RC = ROOT / "data" / "redcert_N1024_r48_n800_CYm_3x3x2_Q36c.npz"
pytestmark = pytest.mark.skipif(not RC.exists(), reason="run scripts/run_N1024.py first")


def test_three_stage_rule_and_lower_bound():
    from cgmy_cert.three_stage import stage_values, certify_three_stage
    from cgmy_cert.eim import load_eim
    from cgmy_cert.pod import AffineROM
    from cgmy_cert.reduced_cert import load_rc, Partition
    from cgmy_cert.symbols import free_matrices
    from cgmy_cert.direct import free_forcing, initial_free_vector
    from cgmy_cert.params import Query
    d = np.load(ROOT / "data" / "references_N1024_NT240_validation30.npz")
    e = load_eim(1024, build_if_missing=False)
    V = np.load(ROOT / "data" / "pod_N1024_NT240_seed0_800.npz")["V"][:, :48]
    rc = load_rc(RC)
    for i in (0, 3):
        m = d["mu"][i]
        q = Query(C=m[0], G=m[1], M=m[2], Y=m[3], r=m[4], T=float(d["T"][i]))
        fs = free_matrices(q, 1024); forc = free_forcing(fs, q.T, 240); z0 = initial_free_vector(q, fs.mesh)
        info = e.eps_Q_graded(q, 36); cid = Partition(rc.partition).cell_of(q)
        a = AffineROM(V, e, Q=36).solve(q, rc.alpha0(), 240, ell=info["ell"]).alpha
        s = stage_values(fs, forc, z0, q=q, alpha=a, V=V, rc=rc, ell=info["ell"], eps_Q=info["eps_Q"], cid=cid)
        f, red = s["full"], s["reduced"]["eta_aug"]
        assert s["lower"] <= f["etaS"] <= red and f["etaS"] <= f["etaX"]
        for eps in (0.3 * s["lower"], 0.5 * (s["lower"] + f["etaS"]), 1.01 * f["etaX"], 1.01 * red):
            r = certify_three_stage(fs, forc, z0, eps, q=q, alpha=a, V=V, rc=rc, ell=info["ell"], eps_Q=info["eps_Q"],
                                    cid=cid, reduced=s["reduced"], tbar=s["tbar"], full=f)
            if red <= eps:
                assert r.status == "reduced_accept"
            elif s["lower"] > eps:
                assert r.status == "reject_cheap"
            elif f["etaX"] <= eps:
                assert r.status == "banded_accept"
            else:
                assert r.status in ("reject_banded", "exact_accept", "exact_reject")
            # never accepts above the exact bound's level, never rejects something the exact stage would accept
            if r.status == "reject_cheap":
                assert f["etaS"] > eps
