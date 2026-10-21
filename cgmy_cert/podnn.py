"""POD-NN candidate (plan 2.4): (mu, T) -> POD coefficients alpha^1..alpha^{N_T} (r = 48), alpha^0 fixed to
the M-orthogonal projection of z_h^0 (so beta_0 is the projection error only).

Network: MLP 6 -> 256 -> 256 -> 256 -> r N_T with SiLU, in float64.  float32 is not usable: the predicted
coefficients (|alpha| ~ 1e2) would carry ~1e-5 absolute quantisation noise, which the residual (time
difference / dt and the operator, ||A|| ~ h^{-Y}) amplifies to ~1e-3 in eta, the target tolerance itself.
Inputs are standardised with training-set statistics; outputs are de-standardised per (mode, time) entry.

Loss = relative least squares in reduced coordinates  ||alpha_hat - alpha_ref||_F^2 / ||alpha_ref||_F^2
       + lambda_R * eta^{X-bar}(alpha_hat)     (residual part, per sample, no allowance/conversion),
with eta^{X-bar} from the R-factor tensors of `reduced_cert` (N-independent, differentiable).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .params import Query, Mesh
from .reduced_cert import ReducedCertificate, cell_of, Partition
from .direct import lifting

CELL_IDS = ("r0C0", "r0C1", "r1C0", "r1C1")   # the plan's rC:2x2 cells (N = 512 pilot)


def cell_ids(rc: ReducedCertificate) -> tuple:
    """Cell order used for the integer cell index of the training data: rC files keep CELL_IDS, others the file order."""
    return CELL_IDS if rc.partition == "rC:2x2" else tuple(rc.cells)


# ---------------------------------------------------------------- data (numpy)

@dataclass
class PODNNData:
    X: np.ndarray            # (n, 6) raw inputs (C, G, M, Y, r, T)
    alpha: np.ndarray        # (n, r, N_T+1) projected reference coefficients
    theta: np.ndarray        # (n, Q+2) affine scalars of the reduced certificate
    ell: np.ndarray          # (n, Q)
    cell: np.ndarray         # (n,) index into CELL_IDS
    dt: np.ndarray           # (n,)
    bL: np.ndarray           # (n, N_T+1)
    query_ids: np.ndarray

    def queries(self) -> list[Query]:
        return [Query(C=x[0], G=x[1], M=x[2], Y=x[3], r=x[4], T=x[5], query_id=str(i))
                for x, i in zip(self.X, self.query_ids)]


def prepare(ref_file, V: np.ndarray, rc: ReducedCertificate, eim, n: int | None = None, chunk: int = 200) -> PODNNData:
    """Projected coefficients alpha = V^T M Z of the stored references plus the per-query affine data."""
    from .symbols import banded_sym_matvec, _poly_entries, mass_symbol
    d = np.load(ref_file, mmap_mode="r")
    n = d["mu"].shape[0] if n is None else n
    mu = np.asarray(d["mu"][:n]); T = np.asarray(d["T"][:n]); ids = np.asarray(d["query_ids"][:n])
    NT = d["Z"].shape[2] - 1
    Me = _poly_entries(lambda t: mass_symbol(t, eim.p), eim.p)
    alpha = np.zeros((n, V.shape[1], NT + 1))
    for i0 in range(0, n, chunk):
        Z = np.asarray(d["Z"][i0:i0 + chunk])                      # (c, n_free, NT+1)
        MZ = np.stack([banded_sym_matvec(Me, z) for z in Z])
        alpha[i0:i0 + chunk] = np.einsum("jr,cjt->crt", V, MZ)
    msh = Mesh(N=eim.N, p=eim.p)
    qs = [Query(C=m[0], G=m[1], M=m[2], Y=m[3], r=m[4], T=float(t), query_id=str(i)) for m, t, i in zip(mu, T, ids)]
    ell = np.array([eim.ell((q.G, q.M, q.Y), rc.Q) for q in qs])
    theta = np.array([rc.theta(q, l) for q, l in zip(qs, ell)])
    ids = cell_ids(rc); P = Partition(rc.partition)
    cell = np.array([ids.index(P.cell_of(q)) for q in qs])
    dt = T / NT
    bL = np.array([lifting(q, msh, np.arange(NT + 1) * (q.T / NT)) for q in qs])
    X = np.column_stack([mu, T])
    return PODNNData(X=X, alpha=alpha, theta=theta, ell=ell, cell=cell, dt=dt, bL=bL, query_ids=ids)


# ---------------------------------------------------------------- torch pieces

def _torch():
    import torch
    return torch


def make_mlp(r: int, NT: int, width: int = 256, depth: int = 3):
    torch = _torch()
    nn = torch.nn
    layers, d_in = [], 6
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), nn.SiLU()]
        d_in = width
    layers.append(nn.Linear(d_in, r * NT))
    return nn.Sequential(*layers).double()


class PODNN:
    """Network + standardisation + fixed alpha^0."""

    def __init__(self, r: int, NT: int, alpha0: np.ndarray, x_mean, x_std, a_mean, a_std, seed: int = 0,
                 time_basis: np.ndarray | None = None):
        """time_basis (N_T x r_t): space-time variant (follow-up 3 section 4): the network outputs B (r x r_t) and
        alpha^n = alpha^0 + (B Phi_t^T)_n, n = 1..N_T; a_mean/a_std are then statistics of B."""
        torch = _torch()
        torch.manual_seed(seed)
        self.r, self.NT = r, NT
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
        self.time_basis = None if time_basis is None else t(time_basis)
        self.n_out_t = NT if time_basis is None else time_basis.shape[1]
        self.net = make_mlp(r, self.n_out_t)
        self.alpha0, self.x_mean, self.x_std, self.a_mean, self.a_std = map(t, (alpha0, x_mean, x_std, a_mean, a_std))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def forward(self, X):
        """X (b, 6) raw -> alpha (b, r, N_T+1) with alpha^0 = the projection of z_h^0."""
        torch = _torch()
        out = self.net((X - self.x_mean) / self.x_std).reshape(-1, self.r, self.n_out_t)
        a = self.a_mean + self.a_std * out
        a0 = self.alpha0.expand(a.shape[0], self.r)[:, :, None]
        if self.time_basis is not None:
            a = a0 + torch.einsum("brk,nk->brn", a, self.time_basis)
        return torch.cat([a0, a], dim=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            return self.forward(torch.as_tensor(X, dtype=torch.float64)).numpy()

    def state(self) -> dict:
        return dict(net=self.net.state_dict(), alpha0=self.alpha0, x_mean=self.x_mean, x_std=self.x_std,
                    a_mean=self.a_mean, a_std=self.a_std, r=self.r, NT=self.NT, time_basis=self.time_basis)

    @classmethod
    def from_state(cls, st: dict) -> "PODNN":
        tb = st.get("time_basis")
        m = cls(st["r"], st["NT"], st["alpha0"].numpy(), st["x_mean"].numpy(), st["x_std"].numpy(),
                st["a_mean"].numpy(), st["a_std"].numpy(), time_basis=None if tb is None else tb.numpy())
        m.net.load_state_dict(st["net"])
        return m


class TorchReducedCert:
    """eta^{X-bar} residual part (dt sum_n u_n)^{1/2}, u_n = ||R_M x_n + R_theta y_n||^2, batched in torch
    (float64; same arithmetic as ReducedCertificate.per_step without the rounding term)."""

    def __init__(self, rc: ReducedCertificate):
        torch = _torch()
        # one tensor per cell (compressed cells have different row counts k), (Q+2, k, r+1) each
        self.R = [torch.as_tensor(rc.cells[c].R, dtype=torch.float64) for c in cell_ids(rc)]

    def eta2(self, alpha, theta, cell, dt, bL):
        """alpha (b, r, N_T+1), theta (b, Q+2), cell (b,) long, dt (b,), bL (b, N_T+1) -> dt sum_n u_n (b,)."""
        torch = _torch()
        X = torch.cat([(alpha[:, :, 1:] - alpha[:, :, :-1]), (bL[:, None, 1:] - bL[:, None, :-1])], dim=1) / dt[:, None, None]
        Y = 0.5 * torch.cat([alpha[:, :, 1:] + alpha[:, :, :-1], (bL[:, None, 1:] + bL[:, None, :-1])], dim=1)
        out = torch.zeros(alpha.shape[0], dtype=torch.float64)
        for c in torch.unique(cell):
            idx = (cell == c).nonzero(as_tuple=True)[0]
            Rc = self.R[int(c)]
            Rth = torch.einsum("bk,kmj->bmj", theta[idx], Rc)
            W = torch.einsum("mj,bjn->bmn", Rc[0], X[idx]) + torch.bmm(Rth, Y[idx])
            out[idx] = dt[idx] * (W * W).sum(dim=(1, 2))
        return out


def rel_ls(alpha_hat, alpha):
    d = alpha_hat[:, :, 1:] - alpha[:, :, 1:]
    return (d * d).sum(dim=(1, 2)) / (alpha[:, :, 1:] ** 2).sum(dim=(1, 2))
