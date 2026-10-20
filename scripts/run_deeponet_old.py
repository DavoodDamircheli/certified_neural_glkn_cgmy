#!/usr/bin/env python3
"""Plan 2.5 (first half): certify the v8 DeepONet checkpoint as a candidate of the *new* (exact-assembly)
system at N = 64, N_T = 60 on bench120 -> results/deeponet_old_vs_new.json.

The network inputs are built with the v8 code itself (v8_frozen/numerics/src, read-only): normalised mu
(log for C, G, M), payoff projection psi / K, and c^0 = v8 compute_initial_coefficients (asserted equal to the
v9 initial vector).  The checkpoint is loaded with v8's load_checkpoint (hard discretisation / box check);
production_dataset.h5 is not needed.  The prediction is the raw coefficient trajectory c(tau_n), n = 0..60;
the candidate is its free part.  Full-order certificate: two-stage with the exact operator A_I (no EIM, no
allowance), eps = 1e-3 x {0.1, 0.3, 1, 3, 10}; true error = max_n ||z^n - z_ref^n||_{M_I} against a dense-LU
reference.
"""
from __future__ import annotations

import json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V8SRC = ROOT / "v8_frozen" / "numerics" / "src"
V8MODEL = ROOT / "v8_frozen" / "models"
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, Mesh, load_queries_json  # noqa: E402
from cgmy_cert.direct import reference_trajectory, initial_raw_coefficients, mass_norm  # noqa: E402
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.certificate import certify  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

N, NT = 64, 60
EPS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
OUT = ROOT / "results" / "deeponet_old_vs_new.json"


def v8_predictor():
    sys.path.insert(0, str(V8SRC)); sys.path.insert(0, str(V8MODEL / "training_code_at_checkpoint_commit_13bf0d5"))
    import torch
    torch.set_default_dtype(torch.float64)
    from cgmy.reference_solver import Mesh as V8Mesh, TimeGrid, MeshInfo, CGMYParameters, EuropeanPutPayoff
    from cgmy.operators import OperatorCache
    from data.sampling import load_parameter_box
    from data import payoff_family
    from common.checkpoint import load_checkpoint
    from correction.krylov import compute_initial_coefficients
    from learning.normalization import normalize_mu, normalize_psi, stats_from_box
    from deeponet_coeff import DeepONetCoeffNet
    box = load_parameter_box(ROOT / "v8_frozen" / "numerics" / "configs" / "base.yaml")
    msh = Mesh(N=N)
    v8mesh = V8Mesh(N=N, p=3, x_min=msh.x_min, x_max=msh.x_max)
    net, meta = load_checkpoint(V8MODEL / "deeponet_final.pt", DeepONetCoeffNet, mesh=v8mesh,
                                time_grid=TimeGrid(n_steps=NT), box=box, payoff_strike=100.0)
    ops = OperatorCache(v8mesh, v8mesh.x_min, v8mesh.x_max)
    minfo = MeshInfo(N=N, p=3, n_dof=ops.n_dof, h=ops.h, x_min=msh.x_min, x_max=msh.x_max, nodes=msh.nodes)
    stats = stats_from_box(box)
    psi = payoff_family.european_put(100.0, ops)

    def predict(q: Query):
        mu = CGMYParameters(C=q.C, G=q.G, M=q.M, Y=q.Y, r=q.r, T=q.T)
        c0, _, _ = compute_initial_coefficients(mu, EuropeanPutPayoff(strike=100.0), minfo, ops)
        tg = np.arange(NT + 1) * (q.T / NT)
        t0 = time.perf_counter()
        C = net.predict_trajectory(normalize_mu(mu, stats), psi, c0, tg, q.T)    # psi raw; /psi_scale inside
        return C, c0, time.perf_counter() - t0

    return predict, {k: meta[k] for k in ("git_sha", "git_dirty", "seed", "n_train", "val_l2_error",
                                           "dataset_manifest_hash")}


_PRED = None


def _one(qd):
    global _PRED
    if _PRED is None:
        _PRED = v8_predictor()[0]
    q = Query(**qd)
    C, c0_v8, t_pred = _PRED(q)
    msh = Mesh(N=N)
    c0_v9 = initial_raw_coefficients(q, msh)
    fs, forc, tr = reference_trajectory(q, N, NT, solver="lu")
    Z = C.T[msh.free_slice]                                      # free part, n_free x (N_T+1)
    z0 = tr.Z[:, 0]
    spec = spectral_data(q, N)
    err = max(mass_norm(fs, Z[:, k] - tr.Z[:, k]) for k in range(NT + 1))
    ref = max(mass_norm(fs, tr.Z[:, k]) for k in range(NT + 1))
    # boundary entries of the prediction vs the v9 prescribed values (the certificate uses the v9 lifting)
    bL = q.K * np.exp(-q.r * np.arange(NT + 1) * (q.T / NT)) * np.sqrt(msh.h)
    bd_dev = float(max(np.max(np.abs(C[:, :2] - bL[:, None])), np.max(np.abs(C[:, -2:]))))
    certs = {f"{e:g}": certify(fs, forc, Z, z0, e, mode="two-stage", spec=spec) for e in EPS}
    cX = certify(fs, forc, Z, z0, 1e-3, mode="banded", spec=spec)
    cS = certify(fs, forc, Z, z0, 1e-3, mode="exact", spec=spec)
    return dict(query_id=q.query_id, mu=list(q.mu), T=q.T, true_err=err, rel_true_err=err / ref,
                beta0=cX.beta0, etaX_aug=cX.etaX_aug, etaS_aug=cS.etaS_aug, effectivity_S=cS.etaS_aug / err,
                status={k: c.status for k, c in certs.items()}, bound_two_stage={k: c.bound for k, c in certs.items()},
                c0_v8_vs_v9=float(np.max(np.abs(c0_v8 - c0_v9))), boundary_entry_dev=bd_dev, t_predict_s=t_pred)


def main():
    pops = load_queries_json(ROOT / "tests" / "fixtures" / "queries.json")
    qs = pops["bench120"]
    _, meta = v8_predictor()
    t0 = time.time()
    with Pool(12) as pool:
        rows = pool.map(_one, [q.as_dict() for q in qs], chunksize=2)
    err = np.array([r["true_err"] for r in rows]); eS = np.array([r["etaS_aug"] for r in rows])
    eX = np.array([r["etaX_aug"] for r in rows])
    counts = {k: {} for k in rows[0]["status"]}
    for r in rows:
        for k, s in r["status"].items():
            counts[k][s] = counts[k].get(s, 0) + 1
    summ = dict(n=len(rows), status_counts_two_stage=counts,
                median_bound_etaS=float(np.median(eS)), max_bound_etaS=float(eS.max()),
                median_bound_etaX=float(np.median(eX)), max_bound_etaX=float(eX.max()),
                median_true_err=float(np.median(err)), max_true_err=float(err.max()),
                median_rel_true_err=float(np.median([r["rel_true_err"] for r in rows])),
                median_beta0=float(np.median([r["beta0"] for r in rows])),
                median_effectivity_S=float(np.median([r["effectivity_S"] for r in rows])),
                max_c0_v8_vs_v9=max(r["c0_v8_vs_v9"] for r in rows),
                max_boundary_entry_dev=max(r["boundary_entry_dev"] for r in rows),
                false_accepts=int(sum(r["status"]["0.001"].endswith("accept") and r["true_err"] > 1e-3 for r in rows)))
    out = dict(meta=dict(description="Plan 2.5: v8 DeepONet checkpoint certified as a candidate of the v9 exact-"
                         "assembly system (A_I, no EIM/allowance) at N = 64, N_T = 60 on bench120. Bounds are augmented "
                         "(beta0 included), true error = max_n mass-norm error vs dense LU reference, same units.",
                         checkpoint="v8_frozen/models/deeponet_final.pt", checkpoint_meta=meta,
                         population="bench120 (tests/fixtures/queries.json)", eps_sweep=list(EPS), N=N, N_T=NT,
                         inputs="v8 code: normalize_mu (log C,G,M), psi = european_put projection, c0 = "
                                "compute_initial_coefficients; production_dataset.h5 not needed",
                         wall_s=time.time() - t0, provenance=provenance()),
               summary=summ, rows=rows)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
