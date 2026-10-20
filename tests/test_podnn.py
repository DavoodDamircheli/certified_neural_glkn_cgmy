"""Plan 2.4: POD-NN pieces (the training itself is scripts/train_podnn.py; its done-when is reported in
results/training_log.json and results/E2_4_podnn.json)."""
import sys

import numpy as np
import pytest

from conftest import ROOT

torch = pytest.importorskip("torch")
sys.path.insert(0, str(ROOT / "scripts"))
from train_podnn import prep_file, rc_file, load_prep  # noqa: E402
from cgmy_cert.reduced_cert import load_rc  # noqa: E402
from cgmy_cert.podnn import PODNN, TorchReducedCert, CELL_IDS  # noqa: E402

pytestmark = pytest.mark.skipif(not prep_file("val30", 800).exists() or not rc_file(800).exists(),
                                reason="run scripts/train_podnn.py prep first")


@pytest.fixture(scope="module")
def data():
    torch.set_num_threads(1)
    rc = load_rc(rc_file(800))
    dv, extra = load_prep(prep_file("val30", 800))
    return rc, dv, extra


def test_torch_matches_numpy(data):
    """Torch eta^{X-bar} (residual part) == numpy to 1e-10 on 5 random reduced trajectories (follow-up 2.4)."""
    rc, dv, _ = data
    tc = TorchReducedCert(rc)
    rng = np.random.default_rng(5)
    T = lambda a: torch.as_tensor(a, dtype=torch.float64)
    for i in range(5):
        al = dv.alpha[i] + 1e-2 * rng.standard_normal(dv.alpha[i].shape) * np.abs(dv.alpha[i]).max()
        X = np.vstack([(al[:, 1:] - al[:, :-1]) / dv.dt[i], ((dv.bL[i, 1:] - dv.bL[i, :-1]) / dv.dt[i])[None]])
        Y = np.vstack([0.5 * (al[:, 1:] + al[:, :-1]), (0.5 * (dv.bL[i, 1:] + dv.bL[i, :-1]))[None]])
        ref = dv.dt[i] * rc.per_step(CELL_IDS[dv.cell[i]], dv.theta[i], X, Y).sum()
        tv = tc.eta2(T(al[None]), T(dv.theta[i:i + 1]), torch.as_tensor(dv.cell[i:i + 1]), T(dv.dt[i:i + 1]),
                     T(dv.bL[i:i + 1])).item()
        assert abs(np.sqrt(tv) - np.sqrt(ref)) <= 1e-10 * np.sqrt(ref)


def test_batched_cells_match_single(data):
    """A batch mixing cells gives the same per-sample values as one-by-one evaluation."""
    rc, dv, _ = data
    tc = TorchReducedCert(rc)
    T = lambda a: torch.as_tensor(a, dtype=torch.float64)
    sl = slice(0, 12)
    b = tc.eta2(T(dv.alpha[sl]), T(dv.theta[sl]), torch.as_tensor(dv.cell[sl]), T(dv.dt[sl]), T(dv.bL[sl]))
    for i in range(12):
        s = tc.eta2(T(dv.alpha[i:i + 1]), T(dv.theta[i:i + 1]), torch.as_tensor(dv.cell[i:i + 1]),
                    T(dv.dt[i:i + 1]), T(dv.bL[i:i + 1]))
        assert abs(b[i].item() - s.item()) <= 1e-9 * s.item()       # summation-order floor (cancellation)


def test_model_alpha0_fixed_and_state_roundtrip(data):
    rc, dv, _ = data
    a = dv.alpha[:, :, 1:]
    m = PODNN(48, 240, rc.alpha0(), dv.X.mean(0), dv.X.std(0), a.mean(0), a.std(0) + 1e-9, seed=0)
    out = m.predict(dv.X[:3])
    assert out.shape == (3, 48, 241)
    assert np.array_equal(out[:, :, 0], np.broadcast_to(rc.alpha0(), (3, 48)))
    m2 = PODNN.from_state(m.state())
    assert np.array_equal(m2.predict(dv.X[:3]), out)
    assert 2.5e6 < m.n_params() < 3.5e6          # 6-256-256-256-11520: ~3.1 M (the brief's "~0.5 M" is not this net)
