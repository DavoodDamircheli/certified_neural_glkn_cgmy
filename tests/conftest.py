import json, os, sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures"

from cgmy_cert.params import Query  # noqa: E402

FULL = os.environ.get("CGMY_FULL_TESTS", "0") == "1"


@pytest.fixture(scope="session")
def populations():
    raw = json.loads((FIX / "queries.json").read_text())["populations"]
    return {name: [Query(C=d["C"], G=d["G"], M=d["M"], Y=d["Y"], r=d["r"], T=d["T"], K=d["K"],
                         query_id=d["query_id"]) for d in pop] for name, pop in raw.items()}


@pytest.fixture(scope="session")
def audit48(populations):
    """v8 Table 5 protocol: first 16 of each stratum in file order."""
    pop = populations["audit459"]
    ids = [f"Q_audit_{i:04d}" for i in list(range(139, 155)) + list(range(395, 411)) + list(range(0, 16))]
    lookup = {q.query_id: q for q in pop}
    return [lookup[i] for i in ids]


@pytest.fixture(scope="session")
def realline():
    path = FIX / "realline_40digit.npz"
    if not path.exists():
        pytest.skip("realline_40digit.npz not built (scripts/build_fixtures.py realline)")
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    entries = {}
    for e in meta["entries"]:
        key = e["key"]
        entries[key] = dict(lags=d[key + "|lags"], values=d[key + "|values"], **e)
    return dict(meta=meta, entries=entries)


@pytest.fixture(scope="session")
def eig24():
    return json.loads((FIX / "eig24_40digit.json").read_text())["rows"]


def query_from_meta(meta, qid):
    d = meta["queries"][qid]
    return Query(C=d["C"], G=d["G"], M=d["M"], Y=d["Y"], r=d["r"], T=d["T"], K=d["K"], query_id=qid)
