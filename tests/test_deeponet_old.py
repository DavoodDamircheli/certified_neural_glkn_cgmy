"""Plan 2.5: the v8 DeepONet checkpoint loads (v8 loader, hard compatibility check) and is fed the right inputs."""
import sys

import numpy as np
import pytest

from conftest import ROOT

pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not (ROOT / "v8_frozen" / "models" / "deeponet_final.pt").exists(),
                                reason="v8_frozen not present")


def test_v8_deeponet_prediction_inputs():
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_deeponet_old import v8_predictor, N, NT
    from cgmy_cert.params import Query, Mesh
    from cgmy_cert.direct import initial_raw_coefficients
    predict, meta = v8_predictor()
    assert meta["n_train"] == 800 and meta["seed"] == 0
    q = Query(0.4, 1.7, 6.3, 1.45, 0.03, T=0.7)
    C, c0, _ = predict(q)
    assert C.shape == (NT + 1, N + 2)
    assert np.max(np.abs(c0 - initial_raw_coefficients(q, Mesh(N=N)))) < 1e-12 * np.abs(c0).max()
    assert np.max(np.abs(C[0] - c0)) < 1e-12 * np.abs(c0).max()          # exact initial-condition embedding
