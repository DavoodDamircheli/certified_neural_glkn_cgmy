"""Parameter box, query populations and mesh conventions (v8 Section 2, Table 1).

Conventions (frozen, identical to v8):
    spline order p = 3 (quadratic), Omega = (log K - 8, log K + 8), K = 100,
    h = 16 / N, raw basis k = -(p-1), ..., N-1 (N + p - 1 functions),
    free basis k = 0, ..., N - p (N_h = N - p + 1 functions, contiguous),
    time nodes tau_n = n * (T / N_T), pure Crank--Nicolson.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import numpy as np
from scipy.special import gamma as _gamma
from scipy.stats import qmc

P_ORDER = 3
STRIKE = 100.0
X_WIDTH = 16.0
X_MIN = float(np.log(STRIKE) - 8.0)
X_MAX = float(np.log(STRIKE) + 8.0)

# v8 Table 1 (configs/base.yaml of the frozen workspace)
BOX = dict(C=(0.05, 1.0), G=(1.0, 5.0), M=(2.0, 9.0), Y=(1.10, 1.80), r=(0.01, 0.10), T=(0.1, 1.0))
BOX_KEYS = ("C", "G", "M", "Y", "r")


@dataclass(frozen=True)
class Query:
    C: float
    G: float
    M: float
    Y: float
    r: float
    T: float = 1.0
    K: float = STRIKE
    query_id: str = ""

    # --- scalar CGMY quantities (v8 eqs. (2.2)-(2.3)) -------------------
    @property
    def gammaY(self) -> float:
        return float(_gamma(-self.Y))

    @property
    def omega_S(self) -> float:
        g = self.gammaY
        return self.C * g * ((self.M - 1) ** self.Y - self.M ** self.Y + (self.G + 1) ** self.Y - self.G ** self.Y)

    @property
    def lambda0(self) -> float:
        return self.r + self.C * self.gammaY * (self.M ** self.Y + self.G ** self.Y)

    @property
    def mu(self) -> tuple:
        return (self.C, self.G, self.M, self.Y, self.r)

    def validate(self) -> None:
        if not (self.C > 0 and self.G > 0 and self.M > 1 and 0 < self.Y < 2 and abs(self.Y - 1) > 1e-10):
            raise ValueError(f"CGMY parameters out of range: {self}")

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_row(row) -> "Query":
        return Query(C=float(row["C"]), G=float(row["G"]), M=float(row["M"]), Y=float(row["Y"]),
                     r=float(row["r"]), T=float(row.get("T", 1.0)), K=float(row.get("K", STRIKE)),
                     query_id=str(row.get("query_id", "")))


@dataclass(frozen=True)
class Mesh:
    N: int
    p: int = P_ORDER
    x_min: float = X_MIN
    x_max: float = X_MAX

    @property
    def h(self) -> float:
        return (self.x_max - self.x_min) / self.N

    @property
    def n_raw(self) -> int:
        return self.N + self.p - 1

    @property
    def n_free(self) -> int:
        return self.N - self.p + 1

    @property
    def free_slice(self) -> slice:
        """Positions of the free coefficients inside the raw vector."""
        return slice(self.p - 1, self.n_raw - (self.p - 1))

    @property
    def nodes(self) -> np.ndarray:
        return self.x_min + self.h * np.arange(self.N + 1)


def mesh(N: int) -> Mesh:
    return Mesh(N=N)


# ---------------------------------------------------------------- populations

def box_vertices(include_T: bool = False) -> list[Query]:
    out = []
    keys = BOX_KEYS + (("T",) if include_T else ())
    for bits in range(2 ** len(keys)):
        vals = {k: BOX[k][(bits >> i) & 1] for i, k in enumerate(keys)}
        out.append(Query(**vals, query_id=f"vertex_{bits:03d}"))
    return out


def sobol_queries(n: int, seed: int = 0, include_T: bool = True, prefix: str = "sobol") -> list[Query]:
    keys = BOX_KEYS + (("T",) if include_T else ())
    s = qmc.Sobol(d=len(keys), scramble=True, seed=seed).random(n)
    lo = np.array([BOX[k][0] for k in keys]); hi = np.array([BOX[k][1] for k in keys])
    pts = lo + s * (hi - lo)
    return [Query(**dict(zip(keys, map(float, x))), query_id=f"{prefix}_{i:04d}") for i, x in enumerate(pts)]


def load_queries_json(path) -> dict[str, list[Query]]:
    with open(path) as f:
        raw = json.load(f)
    fields = set(Query.__dataclass_fields__)          # fixture rows may carry extra keys (e.g. 'source')
    return {name: [Query(**{k: v for k, v in q.items() if k in fields}) for q in pop]
            for name, pop in raw["populations"].items()}
