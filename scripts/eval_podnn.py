#!/usr/bin/env python3
"""Follow-up 2 section 4 / follow-up 4 item 3: evaluate every POD-NN checkpoint of the production sweep on validation30
at the production resolution (results/PRODUCTION_CONFIG.json: N = 1024, N_T = 240, eps_N).

Per checkpoint and query:
  (i)   reduced online certificate: CYm:3x3x2 cells, SVD-compressed tensors at the operational Q(eps_N)
        (results/N1024.json -> eimq), eps_Q on the N-independent nodes, allowance included, no tightening
        (and, for reference, the tightened value with c_mu);
  (ii)  full-order banded certificate with A_I (no allowance);
  (iii) full-order exact certificate with A_I (no allowance);
plus beta_0, the true error max_n ||V alpha^n - Z_ref^n||_{M_I} against the stored references, effectivities, and
acceptance at eps in 1e-3 x {0.1, 0.3, 1, 3, 10}.  Also merges the per-run training logs into
results/training_log.json (with the cell partition the training loss used).  Writes results/E2_4_podnn.json.
"""
from __future__ import annotations

import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query  # noqa: E402
from cgmy_cert.symbols import free_matrices  # noqa: E402
from cgmy_cert.eim import load_eim  # noqa: E402
from cgmy_cert.reduced_cert import load_rc, Partition, conversion_factor  # noqa: E402
from cgmy_cert.direct import free_forcing, initial_free_vector, mass_norm, lifting  # noqa: E402
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.certificate import certify  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

_PCJ = json.loads((Path(__file__).resolve().parents[1] / "results" / "PRODUCTION_CONFIG.json").read_text())
N, NT, R = _PCJ["N"], _PCJ["N_T"], 48
Q_OP = json.loads((Path(__file__).resolve().parents[1] / "results" / "N1024.json").read_text())["eimq"]["per_target"]["eps_N"]["Q_op"]
DATA = ROOT / "data"; RES = ROOT / "results"; CK = ROOT / "checkpoints"
EPS_V8 = tuple(1e-3 * f for f in (0.1, 0.3, 1, 3, 10))
_PC = Path(__file__).resolve().parents[1] / "results" / "PRODUCTION_CONFIG.json"
EPS_N_SWEEP = tuple(json.loads(_PC.read_text())["eps_sweep"]) if _PC.exists() else ()
EPS = tuple(sorted(set(EPS_V8) | set(EPS_N_SWEEP)))
PARTITION = "CYm:3x3x2"
TRAIN_PARTITION = "CYm:3x3x2"


def val_refs():
    d = np.load(DATA / f"references_N{N}_NT{NT}_validation30.npz")
    qs = [Query(C=m[0], G=m[1], M=m[2], Y=m[3], r=m[4], T=float(t), query_id=str(i))
          for m, t, i in zip(d["mu"], d["T"], d["query_ids"])]
    return qs, d["Z"]


def predict(ckpt: Path, X: np.ndarray) -> np.ndarray:
    import torch
    torch.set_num_threads(1)
    from cgmy_cert.podnn import PODNN
    return PODNN.from_state(torch.load(ckpt, weights_only=True)).predict(X)


def _query(args):
    i, n, alphas = args          # alphas: {checkpoint name: (r, NT+1) array} for this query
    qs, Zs = val_refs()
    q, Zref = qs[i], Zs[i]
    e = load_eim(N, build_if_missing=False)
    rc = load_rc(DATA / f"redcert_N{N}_r{R}_n{n}_{PARTITION.replace(':', '_')}_Q{Q_OP}c.npz")
    V = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_{n}.npz")["V"][:, :R]
    fs = free_matrices(q, N); forc = free_forcing(fs, q.T, NT)
    spec = spectral_data(q, N)
    z0 = initial_free_vector(q, fs.mesh)
    Q = Q_OP
    info = e.eps_Q_graded(q, Q)
    cid = Partition(PARTITION).cell_of(q)
    cell = rc.cells[cid]
    c_mu = conversion_factor(q, N, cell.rbar, cell.betabar)
    dt = q.T / NT
    bL = lifting(q, fs.mesh, np.arange(NT + 1) * dt)
    out = {}
    for name, al in alphas.items():
        Z = V @ al
        err = max(mass_norm(fs, Z[:, k] - Zref[:, k]) for k in range(NT + 1))
        red = rc.certify(q, al, dt, bL, info["ell"], info["eps_Q"], cid=cid)
        red_t = rc.certify(q, al, dt, bL, info["ell"], info["eps_Q"], cid=cid, c_mu=c_mu)
        cX = certify(fs, forc, Z, z0, 1e-3, mode="banded", spec=spec)
        cS = certify(fs, forc, Z, z0, 1e-3, mode="exact", spec=spec)
        b = dict(reduced=red["eta_aug"], reduced_tightened=red_t["eta_aug"], banded_AI=cX.etaX_aug, exact_AI=cS.etaS_aug)
        out[name] = dict(query_id=q.query_id, cell=cid, Q=Q, eps_Q=info["eps_Q"], c_mu=c_mu, beta0=cX.beta0,
                         true_err=err, bounds=b, effectivity={k: v / err for k, v in b.items()},
                         accept={k: {f"{t:g}": bool(v <= t) for t in EPS} for k, v in b.items()})
    return out


def main():
    runs = [json.loads(p.read_text()) for p in sorted((RES / "training_runs").glob("podnn_*.json"))]
    qs, _ = val_refs()
    X = np.array([[q.C, q.G, q.M, q.Y, q.r, q.T] for q in qs])
    t0 = time.time()
    results = {}
    for n in (800, 4000):
        names = [r["name"] for r in runs if r["n_train"] == n and r.get("N") == N and (CK / f"{r['name']}.pt").exists()]
        if os.environ.get("EVAL_DRY") == "1":          # reference candidates only (pipeline check)
            names = []
        preds = {nm: predict(CK / f"{nm}.pt", X) for nm in names}
        # reference candidates on the same basis: projection of the references, POD-Galerkin
        refs = np.load(DATA / f"podnn_prep_N{N}_val30_basis{n}.npz")
        preds["projection"] = refs["alpha"]
        from cgmy_cert.pod import AffineROM
        e = load_eim(N, build_if_missing=False)
        V = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_{n}.npz")["V"][:, :R]
        a0 = refs["alpha"][0, :, 0]
        preds["pod_galerkin"] = np.array([AffineROM(V, e, Q=Q_OP).solve(q, a0, NT, with_eps=False).alpha for q in qs])
        with Pool(10) as pool:
            per_q = pool.map(_query, [(i, n, {k: v[i] for k, v in preds.items()}) for i in range(len(qs))])
        for name in preds:
            rows = [pq[name] for pq in per_q]
            summ = dict(n_train=n, median_true_err=float(np.median([r["true_err"] for r in rows])),
                        median_beta0=float(np.median([r["beta0"] for r in rows])))
            for b in ("reduced", "reduced_tightened", "banded_AI", "exact_AI"):
                v = np.array([r["bounds"][b] for r in rows]); ef = np.array([r["effectivity"][b] for r in rows])
                summ[b] = dict(median=float(np.median(v)), max=float(v.max()), median_effectivity=float(np.median(ef)),
                               accepted={f"{t:g}": int(sum(r["accept"][b][f"{t:g}"] for r in rows)) for t in EPS},
                               false_accepts={f"{t:g}": int(sum(r["accept"][b][f"{t:g}"] and r["true_err"] > t
                                                               for r in rows)) for t in EPS})
            results[f"{name}|n{n}" if name in ("projection", "pod_galerkin") else name] = dict(summary=summ, rows=rows)
            print(name, n, {b: (round(summ[b]["median"], 6), summ[b]["accepted"]["0.001"]) for b in
                            ("reduced", "banded_AI", "exact_AI")}, flush=True)
    eps_N = _PCJ["eps_N"]
    passed = {k: dict(reduced=v["summary"]["reduced"]["median"] <= eps_N, exact_AI=v["summary"]["exact_AI"]["median"] <= eps_N,
                      reduced_at_1e3=v["summary"]["reduced"]["median"] <= 1e-3, exact_AI_at_1e3=v["summary"]["exact_AI"]["median"] <= 1e-3)
              for k, v in results.items() if k.startswith("podnn")}
    out = dict(meta=dict(description=f"POD-NN checkpoints on validation30, N = {N}. Bounds "
                         f"augmented (beta0 incl.). reduced = online certificate (CYm:3x3x2 cells, compressed, Q = {Q_OP} "
                         "matched to eps_N, graded eps_Q, allowance, no tightening); reduced_tightened = same / c_mu; "
                         "banded_AI / exact_AI = full-order with the exact operator, no allowance. projection and "
                         "pod_galerkin are reference candidates on the same basis.",
                         eps_sweep=list(EPS), eps_v8_sweep=list(EPS_V8), eps_N_sweep=list(EPS_N_SWEEP),
                         Q=Q_OP,
                         done_when_eps_provisional=1e-3, partition=PARTITION,
                         training_loss_partition=TRAIN_PARTITION, wall_s=time.time() - t0,
                         provenance=provenance()),
               done_when=dict(rule="median certified bound on validation30 <= eps_N (also reported at 1e-3) for at least one "
                                   "lambda_R", per_checkpoint=passed,
                              any_passed_eps_N=any(v["reduced"] or v["exact_AI"] for v in passed.values()),
                              any_passed_1e3=any(v["reduced_at_1e3"] or v["exact_AI_at_1e3"] for v in passed.values())),
               results=results)
    (RES / f"E2_4_podnn_N{N}.json").write_text(json.dumps(out, indent=1, default=float))
    # merged training log
    chk = RES / "training_runs" / "torch_numpy_check.json"
    log = dict(description=f"POD-NN training at the production resolution (r = 48, N = {N}, N_T = {NT}), trimmed design "
                           "(follow-up 4 item 3): 800/4000 trajectories x lambda_R in {0, 1e-2, 1e-1, 1}, seed 0; best "
                           "(size, lambda_R) by validation median full-order exact bound, seeds 1, 2; space-time variant "
                           "(r x r_t) at the best setting, seeds 0-2. Per run: per-epoch train losses, validation metric "
                           f"every val_every epochs (median rigorous reduced bound, CYm:3x3x2 cells, Q = {Q_OP}), selected "
                           f"epoch. Evaluation: E2_4_podnn_N{N}.json.",
               pilot=dict(file="results/pilot_N512.json", design="N = 512, 24 runs planned (2 sizes x 4 lambda_R x 3 seeds, "
                          "2000 epochs); stopped after the production N was fixed at 1024: 6 runs (n = 4000, lambda_R in {1, 0.1}, "
                          "seeds 0-2) reached ~1200 epochs, best validation median reduced bound 5.8-7.7 (rC cells)"),
               training_loss_cell_partition=PARTITION, evaluation_cell_partition=PARTITION,
               torch_numpy_check=json.loads(chk.read_text()) if chk.exists() else None, runs=runs,
               summary=[dict(name=r["name"], n_train=r["n_train"], lambda_R=r["lambda_R"], seed=r["seed"],
                             selected_epoch=r["selected_epoch"], selected_median_eta_aug_oldcells=r["selected_median_eta_aug"],
                             wall_s=r["wall_s"]) for r in runs])
    (RES / "training_log.json").write_text(json.dumps(log, indent=1, default=float))


if __name__ == "__main__":
    main()
