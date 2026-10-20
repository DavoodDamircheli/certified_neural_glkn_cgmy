#!/usr/bin/env python3
"""Plan 2.3: reduced online certificate eta^{X-bar}.

Steps
  cells   beta-bar per cell (0.99 x min spectral_data(q, 512).beta over 5x5 (r, C) x 27 (G, M, Y) points)
  build   R-factor tensors per cell for the POD bases (r = 48; n = 800, 4000) at N = 512
          -> data/redcert_N512_r48_n{n}.npz (sha256 -> results/MANIFEST.json)
  check   online eta^{X-bar} vs the full-order rho^T X-bar^{-1} rho on 20 validation queries
          (POD-Galerkin and perturbed candidates), conversion factor c_mu, validity vs true error
  timing  online time at r = 48, Q = 29 for N in {256, 1024, 4096}, single thread, rule-5 protocol
Writes results/E2_3_reduced_cert.json.
Usage: OMP_NUM_THREADS=1 python3 scripts/run_reduced_cert.py [cells] [build] [check] [timing] [--nproc 20]
"""
from __future__ import annotations

import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, Mesh, sobol_queries  # noqa: E402
from cgmy_cert.symbols import free_matrices  # noqa: E402
from cgmy_cert.eim import load_eim, sha256  # noqa: E402
from cgmy_cert.pod import AffineROM, pod_basis  # noqa: E402
from cgmy_cert.reduced_cert import (ReducedCertificate, cells, cell_beta_queries, cell_of,  # noqa: E402
                                    xbar_entries, conversion_factor, save_rc, load_rc)
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.certificate import cn_residuals, banded_dual_norms  # noqa: E402
from cgmy_cert.direct import free_forcing, mass_norm, reference_trajectory  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

N, NT, R = 512, 240, 48
DATA = ROOT / "data"; RES = ROOT / "results"
OUT = RES / "E2_3_reduced_cert.json"; MANIFEST = RES / "MANIFEST.json"


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1, default=float))


def rc_path(n):
    return DATA / f"redcert_N{N}_r{R}_n{n}.npz"


def cell_params(res):
    return {c["id"]: (c["rbar"], c["betabar"]) for c in res["cells"]}


# ---------------------------------------------------------------- cells

def _beta(qd):
    return spectral_data(Query(**qd), N).beta


def do_cells(res, nproc):
    out = []
    for c in cells():
        qs = cell_beta_queries(c)
        t0 = time.time()
        with Pool(nproc) as pool:
            b = np.array(pool.map(_beta, [q.as_dict() for q in qs], chunksize=4))
        out.append(dict(id=c["id"], r=list(c["r"]), C=list(c["C"]), rbar=c["r"][0], betabar=0.99 * float(b.min()),
                        beta_min=float(b.min()), beta_median=float(np.median(b)), n_points=len(qs),
                        wall_s=time.time() - t0))
        print(f"cell {c['id']}: r {c['r']}, C {c['C']}: beta min {b.min():.4e} median {np.median(b):.4e}", flush=True)
    res["cells"] = out
    res["cells_meta"] = dict(rule="betabar = 0.99 x min spectral_data(q, 512).beta over 5x5 geometric (r, C) grid "
                             "of the cell x 3x3x3 (min, mid, max) grid in (G, M, Y); rbar = cell r_min",
                             N=N, provenance=provenance(nproc=nproc))
    save(res)


# ---------------------------------------------------------------- build

def do_build(res, sets=(800, 4000)):
    man = json.loads(MANIFEST.read_text())
    e = load_eim(N, build_if_missing=False)
    res.setdefault("build", {})
    for n in sets:
        V = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_{n}.npz")["V"][:, :R]
        rc = ReducedCertificate.build(V, e, cell_params(res))
        path = rc_path(n); save_rc(rc, path)
        man["files"][path.name] = dict(file=str(path.relative_to(ROOT)), sha256=sha256(path), N=N, r=R, Q=rc.Q,
                                       basis=f"pod_N{N}_NT{NT}_seed0_{n}.npz", cells=cell_params(res),
                                       wall_s=rc.time_offline_s, provenance=provenance())
        res["build"][str(n)] = dict(file=path.name, Q=rc.Q, r=R, rows_m=int(rc.cells["r0C0"].R.shape[1]),
                                    offline_s=rc.time_offline_s)
        print(f"redcert n={n}: Q={rc.Q}, m={rc.cells['r0C0'].R.shape[1]}, {rc.time_offline_s:.1f}s", flush=True)
    MANIFEST.write_text(json.dumps(man, indent=1))
    save(res)


# ---------------------------------------------------------------- check

def _check_one(args):
    i, cp = args
    d = np.load(DATA / f"references_N{N}_NT{NT}_validation30.npz")
    m = d["mu"][i]
    q = Query(C=m[0], G=m[1], M=m[2], Y=m[3], r=m[4], T=float(d["T"][i]), query_id=str(d["query_ids"][i]))
    Zref = d["Z"][i]
    e = load_eim(N, build_if_missing=False)
    rc = load_rc(rc_path(800))
    V = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_800.npz")["V"][:, :R]
    rom = AffineROM(V, e)
    fs_Q, info = e.affine_free_system(q)
    fs = free_matrices(q, N)
    forc = free_forcing(fs_Q, q.T, NT)
    sol = rom.solve(q, rc.alpha0(), NT, ell=info["ell"])
    cid = cell_of(q); rbar, bbar = cp[cid]
    out = dict(query_id=q.query_id, cell=cid, eps_Q=info["eps_Q"], cands={})
    rng = np.random.default_rng(i)
    pert = sol.alpha + 1e-3 * rng.standard_normal(sol.alpha.shape) * np.abs(sol.alpha).max()
    for name, al in (("pod_galerkin", sol.alpha), ("perturbed", pert)):
        Z = V @ al
        Rr = cn_residuals(fs_Q, Z, forc)
        full, _ = banded_dual_norms(xbar_entries(rbar, bbar), fs.n, 2, Rr)
        tim = {}
        cert = rc.certify(q, al, sol.dt, sol.bL, info["ell"], info["eps_Q"], cid=cid, timings=tim)
        u = cert["per_step"]
        err = max(mass_norm(fs, Z[:, k] - Zref[:, k]) for k in range(NT + 1))
        out["cands"][name] = dict(
            eta2_rel_diff=float(abs(u.sum() - full.sum()) / full.sum()),
            eta_rel_diff=float(abs(np.sqrt(u.sum()) - np.sqrt(full.sum())) / np.sqrt(full.sum())),
            per_step_max_rel_diff=float(np.max(np.abs(u - full) / full)),
            rounding_over_sqrt_u_max=float(np.max(cert["rounding"] / np.sqrt(u))),
            eta_aug=cert["eta_aug"], beta0=cert["beta0"], c_mu=cert["c_mu"], true_err=err,
            effectivity=cert["eta_aug"] / err, valid=bool(cert["eta_aug"] >= err),
            t_form_s=tim["form"], t_conversion_s=tim["conversion"])
    return out


def do_check(res, nproc, n_q=20):
    cp = cell_params(res)
    t0 = time.time()
    with Pool(nproc) as pool:
        rows = pool.map(_check_one, [(i, cp) for i in range(n_q)], chunksize=1)
    summ = {}
    for name in ("pod_galerkin", "perturbed"):
        R_ = [r["cands"][name] for r in rows]
        summ[name] = dict(max_eta_rel_diff=max(x["eta_rel_diff"] for x in R_),
                          max_eta2_rel_diff=max(x["eta2_rel_diff"] for x in R_),
                          max_per_step_rel_diff=max(x["per_step_max_rel_diff"] for x in R_),
                          all_valid=all(x["valid"] for x in R_),
                          median_effectivity=float(np.median([x["effectivity"] for x in R_])),
                          min_c_mu=min(x["c_mu"] for x in R_), max_c_mu=max(x["c_mu"] for x in R_),
                          max_rounding_over_sqrt_u=max(x["rounding_over_sqrt_u_max"] for x in R_))
        print(name, json.dumps(summ[name]), flush=True)
    res["check"] = dict(summary=summ, rows=rows, n_queries=n_q, population="validation30[:20]",
                        basis=f"pod_N{N}_NT{NT}_seed0_800.npz r={R}", wall_s=time.time() - t0,
                        note="full-order reference = banded_dual_norms(X-bar, rho) with rho = cn_residuals on "
                             "(A_Q, f_Q); eta_aug includes beta0, allowance and the conversion 1/c_mu",
                        provenance=provenance(nproc=nproc))
    save(res)


# ---------------------------------------------------------------- timing (rule 5: one thread, warm-up, 5 reps)

def _median_time(fn, reps=5, inner=20):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        ts.append((time.perf_counter() - t0) / inner)
    return float(np.median(ts)), ts


def do_graded(res, Ns=(64, 256, 512, 1024, 4096), n_q=200):
    """Follow-up 2, 2b: eps_Q on the N-independent node set (GRADED_L0 + GRADED_NFINE refinement) vs the
    L-grid version: rigour (graded >= scale * max over the L-grid of |e|, the quantity that bounds
    ||A_I - A_Q||_2) and loss of tightness."""
    from cgmy_cert.eim import eim_test_queries, GRADED_L0, GRADED_NFINE
    qs = eim_test_queries(n_q)
    out = {}
    for Nn in Ns:
        e = load_eim(Nn, build_if_missing=False)
        if Nn >= 512:
            e.Q = 29
        g = e.graded()
        rg, ro = [], []
        for q in qs:
            old = e.eps_Q(q); new = e.eps_Q_graded(q, ell=old["ell"])
            rg.append(new["eps_Q"] / old["grid"]); ro.append(new["eps_Q"] / old["eps_Q"])
        rg, ro = np.array(rg), np.array(ro)
        out[str(Nn)] = dict(Q=e.Q, n_nodes=int(g.theta.size), min_graded_over_Lgrid_max=float(rg.min()),
                            median_graded_over_old_epsQ=float(np.median(ro)), max_graded_over_old_epsQ=float(ro.max()))
        print(f"N={Nn}: {out[str(Nn)]}", flush=True)
    res["graded_eps_Q"] = dict(per_N=out, L0=GRADED_L0, n_fine=GRADED_NFINE, n_queries=n_q,
                               population="eim_test_queries (sobol seed 2026)",
                               rule="sup_theta |e| bounded cell-wise on the node set (endpoint values + half "
                                    "width x max endpoint |e'|), +-pi as one-sided limits plus the L-grid "
                                    "convention value at -pi", provenance=provenance())
    save(res)


def do_timing(res, Ns=(256, 1024, 4096), Q=29, n_train=20, n_q=10, partition="CYm:3x3x2"):
    """Online time of the reduced certificate at r = 48, Q = 29 (follow-up 2, 2b): the quadratic form, ell(nu),
    eps_Q on the N-independent node set, and the optional tightening factor c_mu; all should be flat in N."""
    import random
    from cgmy_cert.reduced_cert import Partition
    assert os.environ.get("OMP_NUM_THREADS") == "1", "run with OMP_NUM_THREADS=1"
    cells_res = json.loads((RES / "E2_3_cells.json").read_text())
    cp = {k: (v["rbar"], v["betabar"]) for k, v in cells_res["partitions"][partition].items()}
    P = Partition(partition)
    res.setdefault("timing", {})
    train = sobol_queries(n_train, seed=11)
    qs = sobol_queries(n_q, seed=12)
    for Nn in Ns:
        e = load_eim(Nn, build_if_missing=False)
        if e.magic.size < Q:
            raise RuntimeError(f"EIM at N={Nn} has only {e.magic.size} terms")
        e.Q = Q
        fs0 = free_matrices(train[0], Nn)
        S = np.hstack([reference_trajectory(t, Nn, NT, solver="gmres")[2].Z for t in train])
        V, _ = pod_basis(S, fs0, R)
        rc = ReducedCertificate.build(V, e, cp, Q=Q, partition=partition)
        rom = AffineROM(V, e, Q=Q)
        rows = []
        for q in qs:
            info = e.eps_Q_graded(q, Q)
            sol = rom.solve(q, rc.alpha0(), NT, ell=info["ell"])
            cid = P.cell_of(q)
            f_form = lambda: rc.certify(q, sol.alpha, sol.dt, sol.bL, info["ell"], info["eps_Q"], cid=cid)
            f_solve = lambda: rom.solve(q, rc.alpha0(), NT, ell=info["ell"])
            f_eps = lambda: e.eps_Q_graded(q, Q, ell=info["ell"])
            f_ell = lambda: e.ell((q.G, q.M, q.Y), Q)
            f_conv = lambda: conversion_factor(q, Nn, *cp[cid])
            f_form(); f_solve()                                  # warm-up
            order = [("form", f_form, 20), ("reduced_solve", f_solve, 20), ("ell", f_ell, 20),
                     ("eps_Q_graded", f_eps, 20), ("c_mu_optional", f_conv, 1)]
            random.Random(hash(q.query_id) % 2 ** 32).shuffle(order)
            row = {}
            for name, fn, inner in order:
                row[name] = _median_time(fn, 5, inner)[0]
            rows.append(row)
        med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
        med["total_online_certificate"] = med["form"] + med["ell"] + med["eps_Q_graded"]
        res.setdefault("timing_v2", {})
        res["timing_v2"][str(Nn)] = dict(N=Nn, partition=partition, n_free=Mesh(N=Nn).n_free, r=R, Q=Q,
                                         rows_m=int(next(iter(rc.cells.values())).R.shape[1]),
                                      median_s=med, per_query=rows, n_queries=n_q,
                                      basis=f"POD of {n_train} trajectories (sobol seed 11) at N={Nn}")
        print(f"N={Nn}: " + ", ".join(f"{k} {v * 1e3:.2f} ms" for k, v in med.items()), flush=True)
    res["timing_v2_meta"] = dict(note="follow-up 2, 2b: eps_Q on the N-independent node set, new cells, c_mu "
                                 "optional (not part of the default online path); the earlier 'timing' block "
                                 "(L-grid eps_Q, rC cells, c_mu always) is kept for comparison",
                                 load_note="measured while the 2.4 training sweep occupied ~24 threads; Phase 3 "
                                           "(E2) repeats on an idle machine", protocol="OMP_NUM_THREADS=1, one warm-up per query, 5 repetitions (inner loops of 20 "
                              "for the fast parts), shuffled order, median of per-query medians",
                              implementation="numpy (no compilation)", provenance=provenance())
    save(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["cells", "build", "check", "graded", "timing"])
    ap.add_argument("--nproc", type=int, default=20)
    a = ap.parse_args()
    res = load()
    res["meta"] = dict(description="Plan 2.3: reduced online certificate eta^{X-bar} (R-factor form, see "
                       "cgmy_cert/reduced_cert.py and DEVIATIONS.md)", N=N, N_T=NT, r=R)
    if "cells" in a.steps:
        do_cells(res, a.nproc)
    if "build" in a.steps:
        do_build(res)
    if "check" in a.steps:
        do_check(res, a.nproc)
    if "graded" in a.steps:
        do_graded(res)
    if "timing" in a.steps:
        do_timing(res)


if __name__ == "__main__":
    main()
