#!/usr/bin/env python3
"""Follow-up 4, item 2: the production pipeline at N = 1024, N_T = 240 (results/PRODUCTION_CONFIG.json).

Steps (results/N1024.json; data/*_N1024_*; sha256 -> results/MANIFEST.json):
  pod      mass-weighted POD from the stored 800 and 4000 references (r_max = 62, singular values kept)
  eimq     eps-matched Q (EIM.Q_for_target) on validation30 at eps_N and eps_N/10; operational Q = max over queries
  cells    3 x 3 x 2 cells in (log C, Y, log min(G, M)): corner-rule beta-bar at N = 1024 (33-point Y grid),
           validity (c_mu >= 1 on 400 Sobol queries), cell vs own-norm effectivity (POD-Galerkin r = 48)
  tensors  SVD-compressed reduced-certificate tensors: r = 48 for both bases at both operational Q, and the POD-
           Galerkin ranks 8..62 (basis 800) at Q(eps_N); MB per cell
  checks   (a) full basis (r = n_free) reduced solve == dense-LU direct with A_I (1e-10); (b) reduced online vs
           full-order rho^T X-bar^{-1} rho on 20 validation queries (uncompressed: equality floor; compressed:
           rigorous bracket); (c) online timing at N = 1024, one thread, idle machine
  table    POD-Galerkin r = 8..62 on validation30: full-order banded and exact (A_I, no allowance), reduced online
           (Q(eps_N) tensors, allowance, untightened and tightened), true error, acceptance at eps_N x {0.1..10}
Usage: OMP_NUM_THREADS=1 python3 scripts/run_N1024.py [steps] [--nproc 20]
"""
from __future__ import annotations

import argparse, json, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, Mesh, sobol_queries, load_queries_json  # noqa: E402
from cgmy_cert.symbols import free_matrices, _dense_banded  # noqa: E402
from cgmy_cert.eim import load_eim, sha256  # noqa: E402
from cgmy_cert.pod import AffineROM, mass_cholesky, pod_galerkin  # noqa: E402
from cgmy_cert.reduced_cert import (ReducedCertificate, Partition, beta_bar_rigorous, conversion_factor,  # noqa: E402
                                    xbar_entries, save_rc, load_rc)
from cgmy_cert.direct import reference_trajectory, free_forcing, initial_free_vector, mass_norm  # noqa: E402
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.certificate import certify, cn_residuals, banded_dual_norms  # noqa: E402
from cgmy_cert.params import BOX  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

PC = json.loads((ROOT / "results" / "PRODUCTION_CONFIG.json").read_text())
N, NT = PC["N"], PC["N_T"]
EPS_N = PC["eps_N"]; EPS_SWEEP = tuple(PC["eps_sweep"])
DATA = ROOT / "data"; RES = ROOT / "results"
OUT = RES / "N1024.json"; MANIFEST = RES / "MANIFEST.json"
SPEC = "CYm:3x3x2"; RBAR = BOX["r"][0]; R48 = 48
RANKS = (8, 16, 24, 32, 40, 48, 56, 62)
SVD_TOL = 1e-12


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1, default=float))


def ref_file(tag):
    return DATA / f"references_N{N}_NT{NT}_{tag}.npz"


def pod_file(n):
    return DATA / f"pod_N{N}_NT{NT}_seed0_{n}.npz"


def rc_file(n, r, Q):
    return DATA / f"redcert_N{N}_r{r}_n{n}_{SPEC.replace(':', '_')}_Q{Q}c.npz"


def manifest_add(path, **meta):
    man = json.loads(MANIFEST.read_text())
    man["files"][path.name] = dict(file=str(path.relative_to(ROOT)), sha256=sha256(path), bytes=path.stat().st_size,
                                   **meta, provenance=provenance())
    MANIFEST.write_text(json.dumps(man, indent=1))


def val_queries():
    d = np.load(ref_file("validation30"))
    return [Query(C=m[0], G=m[1], M=m[2], Y=m[3], r=m[4], T=float(t), query_id=str(i))
            for m, t, i in zip(d["mu"], d["T"], d["query_ids"])]


# ---------------------------------------------------------------- pod

def do_pod(res):
    fs = free_matrices(Query(0.5, 3.0, 5.0, 1.5, 0.05), N)
    Lc = mass_cholesky(fs)
    res.setdefault("pod", {})
    for n in (800, 4000):
        t0 = time.time()
        Z = np.load(ref_file("seed0_4000"), mmap_mode="r")["Z"]
        W = np.empty((fs.n, n * (NT + 1)))
        for i0 in range(0, n, 200):
            blk = np.asarray(Z[i0:i0 + 200])
            W[:, i0 * (NT + 1):(i0 + blk.shape[0]) * (NT + 1)] = np.einsum("ij,qjt->iqt", Lc.T, blk).reshape(fs.n, -1)
        U, s, _ = np.linalg.svd(W, full_matrices=False)
        del W
        V = solve_triangular(Lc.T, U[:, :62], lower=False)
        orth = float(np.abs(V.T @ _dense_banded(fs.M_entries, fs.n) @ V - np.eye(62)).max())
        np.savez(pod_file(n), V=V, sv=s, n_traj=n, N=N, N_T=NT, source=ref_file("seed0_4000").name)
        manifest_add(pod_file(n), n_traj=n, r_max=62, construction="mass-weighted POD of all N_T+1 snapshots",
                     mass_orthonormality_err=orth)
        res["pod"][str(n)] = dict(sv_ratio={str(r): float(s[r] / s[0]) for r in RANKS}, singular_values=s[:200].tolist(),
                                  mass_orthonormality_err=orth, wall_s=time.time() - t0)
        print(f"POD n={n}: " + " ".join(f"{r}:{s[r] / s[0]:.1e}" for r in RANKS) + f" ({time.time() - t0:.0f}s)", flush=True)
        save(res)


# ---------------------------------------------------------------- eim Q

def _qt(args):
    qd, eps = args
    return load_eim(N, build_if_missing=False).Q_for_target(Query(**qd), eps)


def do_eimq(res, nproc):
    e = load_eim(N, build_if_missing=False)
    qs = val_queries()
    out = {}
    with Pool(nproc) as pool:
        for tag, eps in (("eps_N", EPS_N), ("eps_N/10", EPS_N / 10)):
            Qs = pool.map(_qt, [(q.as_dict(), eps) for q in qs])
            out[tag] = dict(eps=eps, Q_per_query=Qs, median=float(np.median(Qs)), max=int(max(Qs)), Q_op=int(max(Qs)))
            print(tag, out[tag]["median"], out[tag]["max"], flush=True)
    res["eimq"] = dict(per_target=out, q_max=int(e.magic.size), Q_vs_tol=e.meta.get("Q_vs_tol"),
                       test_err_at={str(Q): float(e.test_err[Q]) for Q in range(25, e.magic.size + 1)},
                       rule="operational Q = max over validation30 of EIM.Q_for_target(q, eps) (eps_Q ||cbar|| "
                            "sqrt(T/alpha) <= eps/10)", provenance=provenance(nproc=nproc))
    save(res)


def Q_ops(res):
    return res["eimq"]["per_target"]["eps_N"]["Q_op"], res["eimq"]["per_target"]["eps_N/10"]["Q_op"]


# ---------------------------------------------------------------- cells

def _cmu(args):
    qd, cp = args
    q = Query(**qd)
    return conversion_factor(q, N, *cp[Partition(SPEC).cell_of(q)])


def cell_params(res):
    return {k: (v["rbar"], v["betabar"]) for k, v in res["cells"]["betabar"].items()}


def _effect_one(args):
    i, cp = args
    d = np.load(ref_file("validation30"))
    q = val_queries()[i]
    e = load_eim(N, build_if_missing=False)
    V = np.load(pod_file(800))["V"][:, :R48]
    fs = free_matrices(q, N); forc = free_forcing(fs, q.T, NT)
    z0 = initial_free_vector(q, fs.mesh)
    a0 = V.T @ fs.M_matvec(z0)
    sol = AffineROM(V, e).solve(q, a0, NT, with_eps=False)
    Z = V @ sol.alpha
    Rr = cn_residuals(fs, Z, forc)
    sp = spectral_data(q, N)
    cX = certify(fs, forc, Z, z0, 1.0, mode="banded", spec=sp)
    cS = certify(fs, forc, Z, z0, 1.0, mode="exact", spec=sp)
    qf, _ = banded_dual_norms(xbar_entries(*cp[Partition(SPEC).cell_of(q)]), fs.n, 2, Rr)
    ex = float(np.sqrt(forc.dt * np.sum(qf)))
    return dict(query_id=q.query_id, over_etaX=ex / cX.etaX, over_etaS=ex / cS.etaS_upper)


def do_cells(res, nproc):
    out = {}
    for c in Partition(SPEC).cells():
        r = beta_bar_rigorous(c, N, nY=33, nproc=nproc)
        r["rbar"] = RBAR
        out[c["id"]] = r
        print(c["id"], round(r["betabar"], 4), r["argmin"], flush=True)
    res["cells"] = dict(partition=SPEC, betabar=out, rule="corner rule (follow-up 3 item 2) at N = 1024, 33-point Y grid")
    cp = cell_params(res)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qs = sobol_queries(400, seed=21, include_T=False)
    with Pool(nproc) as pool:
        c = np.array(pool.map(_cmu, [(q.as_dict(), cp) for q in qs], chunksize=4))
        eff = pool.map(_effect_one, [(i, cp) for i in range(30)], chunksize=1)
    a = np.array([x["over_etaX"] for x in eff]); b = np.array([x["over_etaS"] for x in eff])
    res["cells"].update(validity=dict(min_c_mu=float(c.min()), median_c_mu=float(np.median(c)), n_below_1=int((c < 1).sum()),
                                      n=400, population="sobol_queries(400, seed=21)"),
                        effectivity=dict(median_over_etaX=float(np.median(a)), max_over_etaX=float(a.max()),
                                         median_over_etaS=float(np.median(b)), max_over_etaS=float(b.max()), rows=eff,
                                         description="validation30, POD-Galerkin r = 48 (basis 800), residual parts with A_I"),
                        provenance=provenance(nproc=nproc))
    print("validity", res["cells"]["validity"], "effectivity", {k: v for k, v in res["cells"]["effectivity"].items() if k != "rows"})
    save(res)


# ---------------------------------------------------------------- tensors

def build_tensor(res, n, r, Q):
    e = load_eim(N, build_if_missing=False)
    V = np.load(pod_file(n))["V"][:, :r]
    rc = ReducedCertificate.build(V, e, cell_params(res), Q=Q, partition=SPEC, svd_tol=SVD_TOL)
    path = rc_file(n, r, Q)
    save_rc(rc, path)
    ks = [int(c.R.shape[1]) for c in rc.cells.values()]
    mb = [c.R.nbytes / 2 ** 20 for c in rc.cells.values()]
    manifest_add(path, N=N, r=r, Q=Q, partition=SPEC, svd_tol=SVD_TOL, basis=pod_file(n).name)
    return dict(file=path.name, n_basis=n, r=r, Q=Q, rows_k=[min(ks), max(ks)], MB_per_cell=[min(mb), max(mb)],
                total_MB=path.stat().st_size / 2 ** 20, offline_s=rc.time_offline_s)


def do_tensors(res):
    Qa, Qb = Q_ops(res)
    res.setdefault("tensors", {})
    jobs = [(n, R48, Q) for n in (800, 4000) for Q in sorted({Qa, Qb})] + [(800, r, Qa) for r in RANKS if r != R48]
    for n, r, Q in jobs:
        b = build_tensor(res, n, r, Q)
        res["tensors"][b["file"]] = b
        print(b, flush=True)
        save(res)


# ---------------------------------------------------------------- checks

def _check_one(args):
    i, Q, n = args
    q = val_queries()[i]
    Zref = np.load(ref_file("validation30"))["Z"][i]
    e = load_eim(N, build_if_missing=False)
    V = np.load(pod_file(n))["V"][:, :R48]
    rc = load_rc(rc_file(n, R48, Q))
    cid = Partition(SPEC).cell_of(q); c = rc.cells[cid]
    unc = ReducedCertificate.build(V, e, {cid: (c.rbar, c.betabar)}, Q=Q, partition=SPEC)
    fs_Q, info = e.affine_free_system(q, Q)
    forc_Q = free_forcing(fs_Q, q.T, NT)
    sol = AffineROM(V, e, Q=Q).solve(q, rc.alpha0(), NT, with_eps=False)
    out = {}
    rng = np.random.default_rng(i)
    for name, al in (("pod_galerkin", sol.alpha), ("perturbed", sol.alpha + 1e-3 * rng.standard_normal(sol.alpha.shape) * np.abs(sol.alpha).max())):
        Rr = cn_residuals(fs_Q, V @ al, forc_Q)
        full, _ = banded_dual_norms(xbar_entries(c.rbar, c.betabar), fs_Q.n, 2, Rr)
        X = np.vstack([(al[:, 1:] - al[:, :-1]) / sol.dt, ((sol.bL[1:] - sol.bL[:-1]) / sol.dt)[None]])
        Y = np.vstack([0.5 * (al[:, 1:] + al[:, :-1]), (0.5 * (sol.bL[1:] + sol.bL[:-1]))[None]])
        th = rc.theta(q, sol.ell)
        uu = unc.per_step(cid, th, X, Y)
        uc, rnd = rc.per_step(cid, th, X, Y, with_rounding=True)
        eta = lambda u: np.sqrt(np.sum(u))
        # per-step bracket: compressed main <= full-order value, full-order value <= main + tail + rounding; both
        # sides are compared with the double-precision full-order value, whose own floor is ~eps * kappa per step
        lower_excess = float(np.max(np.sqrt(uc) / np.sqrt(full) - 1))         # should be <= floor (~1e-7)
        upper_slack = float(np.min((np.sqrt(uc) + rnd) / np.sqrt(full) - 1))  # should be >= -floor
        out[name] = dict(uncompressed_eta_rel_diff=float(abs(eta(uu) - eta(full)) / eta(full)),
                         max_rel_lower_excess=lower_excess, min_rel_upper_slack=upper_slack,
                         per_step_uncompressed_max_rel_diff=float(np.max(np.abs(np.sqrt(uu) / np.sqrt(full) - 1))),
                         bracket_ok_to_floor=bool(lower_excess <= 1e-6 and upper_slack >= -1e-6),
                         compressed_bound_over_full=float(np.sqrt(np.sum((np.sqrt(uc) + rnd) ** 2)) / eta(full)))
    return out


def _median_time(fn, reps=5, inner=20):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        ts.append((time.perf_counter() - t0) / inner)
    return float(np.median(ts))


def do_checks(res, nproc):
    Qa, Qb = Q_ops(res)
    out = {}
    # (a) full basis
    rows = []
    for q in val_queries()[:3]:
        fs, forc, tr = reference_trajectory(q, N, NT, solver="lu")
        V = solve_triangular(mass_cholesky(fs).T, np.eye(fs.n), lower=False)
        Zg = pod_galerkin(fs, forc, V, tr.Z[:, 0])
        rel = max(mass_norm(fs, Zg[:, k] - tr.Z[:, k]) for k in range(NT + 1)) / max(mass_norm(fs, tr.Z[:, k]) for k in range(NT + 1))
        rows.append(dict(query_id=q.query_id, rel_err=float(rel)))
    out["full_basis_vs_direct"] = dict(rows=rows, max_rel_err=max(r["rel_err"] for r in rows), r=Mesh(N=N).n_free,
                                       pass_1e10=bool(max(r["rel_err"] for r in rows) < 1e-10),
                                       note="r = n_free (the analogue of the r = 62 = n_free check at N = 64)")
    print("full basis", out["full_basis_vs_direct"]["max_rel_err"], flush=True)
    # (b) online vs full-order
    with Pool(nproc) as pool:
        ch = pool.map(_check_one, [(i, Qa, 800) for i in range(20)], chunksize=1)
    for name in ("pod_galerkin", "perturbed"):
        out[f"online_vs_full_{name}"] = dict(
            max_uncompressed_eta_rel_diff=max(x[name]["uncompressed_eta_rel_diff"] for x in ch),
            all_brackets_ok_to_floor=all(x[name]["bracket_ok_to_floor"] for x in ch),
            max_rel_lower_excess=max(x[name]["max_rel_lower_excess"] for x in ch),
            min_rel_upper_slack=min(x[name]["min_rel_upper_slack"] for x in ch),
            max_per_step_uncompressed_rel_diff=max(x[name]["per_step_uncompressed_max_rel_diff"] for x in ch),
            max_compressed_bound_over_full=max(x[name]["compressed_bound_over_full"] for x in ch), Q=Qa, n_queries=20)
        print(name, out[f"online_vs_full_{name}"], flush=True)
    # (c) timing (one thread, idle machine), production tensors r = 48, basis 800, both Q
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    e = load_eim(N, build_if_missing=False)
    V = np.load(pod_file(800))["V"][:, :R48]
    tim = {}
    for Q in sorted({Qa, Qb}):
        rc = load_rc(rc_file(800, R48, Q)); rom = AffineROM(V, e, Q=Q)
        rows = []
        for q in val_queries()[:10]:
            cid = Partition(SPEC).cell_of(q)
            info = e.eps_Q_graded(q, Q)
            sol = rom.solve(q, rc.alpha0(), NT, ell=info["ell"])
            f_ell = lambda: e.ell((q.G, q.M, q.Y), Q)
            f_eps = lambda: e.eps_Q_graded(q, Q, ell=info["ell"])
            f_form = lambda: rc.certify(q, sol.alpha, sol.dt, sol.bL, info["ell"], info["eps_Q"], cid=cid)
            f_solve = lambda: rom.solve(q, rc.alpha0(), NT, ell=info["ell"])
            f_all = lambda: (f_ell(), f_eps(), f_form())
            for f in (f_ell, f_eps, f_form, f_solve):
                f()
            rows.append(dict(ell=_median_time(f_ell), eps_Q=_median_time(f_eps), form=_median_time(f_form),
                             reduced_solve=_median_time(f_solve), online_certificate_total=_median_time(f_all)))
        tim[str(Q)] = {k: float(np.median([r[k] for r in rows]) * 1e3) for k in rows[0]}
        tim[str(Q)]["rows_k"] = [int(min(c.R.shape[1] for c in rc.cells.values())), int(max(c.R.shape[1] for c in rc.cells.values()))]
        print("timing Q", Q, tim[str(Q)], flush=True)
    import subprocess
    load_avg = subprocess.run(["cat", "/proc/loadavg"], capture_output=True, text=True).stdout.strip()
    out["timing_ms"] = dict(per_Q=tim, unit="ms, median of per-query medians (10 validation queries)", loadavg=load_avg,
                            note="one thread (OMP_NUM_THREADS=1, taskset); the tensors are N-specific; the online cost "
                                 "is O((Q+2) k r + k r N_T) with k <= rows of the compressed factor, independent of N by "
                                 "construction (k ~ 350-420 at every N >= 512 for Q = 29, follow-up 3)")
    res["checks"] = dict(out, provenance=provenance(nproc=nproc))
    save(res)


# ---------------------------------------------------------------- table

def _table_one(args):
    i, Qa = args
    q = val_queries()[i]
    Zref = np.load(ref_file("validation30"))["Z"][i]
    e = load_eim(N, build_if_missing=False)
    Vall = np.load(pod_file(800))["V"]
    fs = free_matrices(q, N); forc = free_forcing(fs, q.T, NT)
    sp = spectral_data(q, N)
    z0 = initial_free_vector(q, fs.mesh)
    info = e.eps_Q_graded(q, Qa)
    cid = Partition(SPEC).cell_of(q)
    out = dict(query_id=q.query_id, cell=cid, eps_Q=info["eps_Q"], ranks={})
    for r in RANKS:
        V = Vall[:, :r]
        rc = load_rc(rc_file(800, r, Qa))
        c_mu = conversion_factor(q, N, rc.cells[cid].rbar, rc.cells[cid].betabar)
        t0 = time.perf_counter()
        sol = AffineROM(V, e, Q=Qa).solve(q, rc.alpha0(), NT, ell=info["ell"])
        t_solve = time.perf_counter() - t0
        Z = V @ sol.alpha
        err = max(mass_norm(fs, Z[:, k] - Zref[:, k]) for k in range(NT + 1))
        cX = certify(fs, forc, Z, z0, 1.0, mode="banded", spec=sp)
        cS = certify(fs, forc, Z, z0, 1.0, mode="exact", spec=sp)
        red = rc.certify(q, sol.alpha, sol.dt, sol.bL, info["ell"], info["eps_Q"], cid=cid)
        redt = rc.certify(q, sol.alpha, sol.dt, sol.bL, info["ell"], info["eps_Q"], cid=cid, c_mu=c_mu)
        b = dict(banded_AI=cX.etaX_aug, exact_AI=cS.etaS_aug, reduced=red["eta_aug"], reduced_tightened=redt["eta_aug"])
        out["ranks"][str(r)] = dict(true_err=err, beta0=cX.beta0, bounds=b, effectivity={k: v / err for k, v in b.items()},
                                    accept={k: {f"{t:.4g}": bool(v <= t) for t in EPS_SWEEP} for k, v in b.items()},
                                    c_mu=c_mu, t_reduced_solve_s=t_solve, t_full_banded_s=cX.time_banded + cX.time_residual,
                                    t_full_exact_s=cS.time_exact + cS.time_residual)
    return out


def do_table(res, nproc):
    Qa, _ = Q_ops(res)
    with Pool(nproc) as pool:
        rows = pool.map(_table_one, [(i, Qa) for i in range(30)], chunksize=1)
    summ = {}
    for r in RANKS:
        R_ = [x["ranks"][str(r)] for x in rows]
        s = dict(median_true_err=float(np.median([x["true_err"] for x in R_])), max_true_err=float(max(x["true_err"] for x in R_)))
        for b in ("banded_AI", "exact_AI", "reduced", "reduced_tightened"):
            v = np.array([x["bounds"][b] for x in R_]); ef = np.array([x["effectivity"][b] for x in R_])
            s[b] = dict(median=float(np.median(v)), max=float(v.max()), median_effectivity=float(np.median(ef)),
                        accepted={f"{t:.4g}": int(sum(x["accept"][b][f"{t:.4g}"] for x in R_)) for t in EPS_SWEEP},
                        false_accepts={f"{t:.4g}": int(sum(x["accept"][b][f"{t:.4g}"] and x["true_err"] > t for x in R_))
                                       for t in EPS_SWEEP})
        summ[str(r)] = s
        print(f"r={r}: err {s['median_true_err']:.2e} | " + " | ".join(
            f"{b} med {s[b]['median']:.2e} acc@eps_N {s[b]['accepted'][f'{EPS_N:.4g}']}" for b in ("exact_AI", "banded_AI", "reduced")), flush=True)
    res["pod_galerkin_table"] = dict(summary=summ, rows=rows, Q=Qa, eps_N=EPS_N, eps_sweep=list(EPS_SWEEP), basis="800",
                                     description="validation30, POD-Galerkin (reduced solve with A_Q at Q(eps_N)); bounds "
                                     "augmented: banded_AI / exact_AI = full-order with A_I (no allowance); reduced = online "
                                     "(CYm:3x3x2, compressed, allowance, untightened); reduced_tightened = / c_mu; true error "
                                     "= max_n mass-norm vs the stored GMRES references",
                                     provenance=provenance(nproc=nproc))
    save(res)


# ---------------------------------------------------------------- follow-up 5: extended ranks + three-stage rule

RANKS_EXT = (8, 16, 24, 32, 40, 48, 56, 62, 64, 80, 96, 128)


def pod128_file(n):
    return DATA / f"pod_N{N}_NT{NT}_seed0_{n}_r128.npz"


def rc_ext_file(n, r, Q):
    return DATA / f"redcert_N{N}_r{r}_n{n}_{SPEC.replace(':', '_')}_Q{Q}c_b128.npz"


def do_podext(res):
    """POD with r_max = 128 (same construction); compares the first 62 columns with the stored r_max = 62 bases."""
    fs = free_matrices(Query(0.5, 3.0, 5.0, 1.5, 0.05), N)
    Lc = mass_cholesky(fs)
    res.setdefault("podext", {})
    for n in (800, 4000):
        t0 = time.time()
        Z = np.load(ref_file("seed0_4000"), mmap_mode="r")["Z"]
        W = np.empty((fs.n, n * (NT + 1)))
        for i0 in range(0, n, 200):
            blk = np.asarray(Z[i0:i0 + 200])
            W[:, i0 * (NT + 1):(i0 + blk.shape[0]) * (NT + 1)] = np.einsum("ij,qjt->iqt", Lc.T, blk).reshape(fs.n, -1)
        U, s, _ = np.linalg.svd(W, full_matrices=False)
        del W
        V = solve_triangular(Lc.T, U[:, :128], lower=False)
        old = np.load(pod_file(n))["V"]
        same = float(np.abs(V[:, :62] - old).max())
        np.savez(pod128_file(n), V=V, sv=s, n_traj=n, N=N, N_T=NT, source=ref_file("seed0_4000").name)
        manifest_add(pod128_file(n), n_traj=n, r_max=128, construction="mass-weighted POD of all N_T+1 snapshots")
        res["podext"][str(n)] = dict(sv_ratio={str(r): float(s[r] / s[0]) for r in RANKS_EXT},
                                     max_diff_first62_vs_stored=same, wall_s=time.time() - t0)
        print(f"POD128 n={n}: diff first 62 vs stored {same:.1e}; " + " ".join(f"{r}:{s[r] / s[0]:.1e}" for r in RANKS_EXT[8:]), flush=True)
        save(res)


def do_tensors_ext(res):
    Qa, _ = Q_ops(res)
    e = load_eim(N, build_if_missing=False)
    res.setdefault("tensors_ext", {})
    for n in (800, 4000):
        Vall = np.load(pod128_file(n))["V"]
        for r in RANKS_EXT:
            path = rc_ext_file(n, r, Qa)
            if path.exists():
                continue
            rc = ReducedCertificate.build(Vall[:, :r], e, cell_params(res), Q=Qa, partition=SPEC, svd_tol=SVD_TOL)
            save_rc(rc, path)
            ks = [int(c.R.shape[1]) for c in rc.cells.values()]; mb = [c.R.nbytes / 2 ** 20 for c in rc.cells.values()]
            manifest_add(path, N=N, r=r, Q=Qa, partition=SPEC, svd_tol=SVD_TOL, basis=pod128_file(n).name)
            res["tensors_ext"][path.name] = dict(n_basis=n, r=r, Q=Qa, rows_k=[min(ks), max(ks)],
                                                 MB_per_cell=[min(mb), max(mb)], total_MB=path.stat().st_size / 2 ** 20,
                                                 offline_s=rc.time_offline_s)
            print(path.name, res["tensors_ext"][path.name], flush=True)
            save(res)


def _table_ext_one(args):
    i, Qa, n = args
    from cgmy_cert.three_stage import theta_bar, full_order_values, certify_three_stage
    q = val_queries()[i]
    Zref = np.load(ref_file("validation30"))["Z"][i]
    e = load_eim(N, build_if_missing=False)
    Vall = np.load(pod128_file(n))["V"]
    fs = free_matrices(q, N); forc = free_forcing(fs, q.T, NT)
    sp = spectral_data(q, N)
    z0 = initial_free_vector(q, fs.mesh)
    info = e.eps_Q_graded(q, Qa)
    cid = Partition(SPEC).cell_of(q)
    out = dict(query_id=q.query_id, cell=cid, eps_Q=info["eps_Q"], ranks={})
    tbar = None
    for r in RANKS_EXT:
        V = Vall[:, :r]
        rc = load_rc(rc_ext_file(n, r, Qa))
        cell = rc.cells[cid]
        if tbar is None:
            tbar = theta_bar(q, N, cell.rbar, cell.betabar)               # depends on (query, cell) only
            c_mu = conversion_factor(q, N, cell.rbar, cell.betabar)
        sol = AffineROM(V, e, Q=Qa).solve(q, rc.alpha0(), NT, ell=info["ell"])
        Z = V @ sol.alpha
        err = max(mass_norm(fs, Z[:, k] - Zref[:, k]) for k in range(NT + 1))
        red = rc.certify(q, sol.alpha, sol.dt, sol.bL, info["ell"], info["eps_Q"], cid=cid)
        full = full_order_values(fs, forc, z0, Z, q, sp)
        b = dict(reduced=red["eta_aug"], reduced_tightened=float(np.sqrt(red["beta0"] ** 2 + (red["eta_aug"] ** 2 - red["beta0"] ** 2) / c_mu)),
                 banded_AI=full["etaX"], exact_AI=full["etaS"])
        stat = {}
        for t in EPS_SWEEP:
            ts = certify_three_stage(fs, forc, z0, t, q=q, alpha=sol.alpha, V=V, rc=rc, ell=info["ell"], eps_Q=info["eps_Q"],
                                     cid=cid, reduced=red, tbar=tbar, full=full)
            stat[f"{t:.4g}"] = ts.status
        lo = np.maximum((np.sqrt(red["per_step"]) - red["rounding"]) / np.sqrt(tbar) - red["allowance"], 0.0)
        out["ranks"][str(r)] = dict(true_err=err, beta0=red["beta0"], bounds=b, effectivity={k: v / err for k, v in b.items()},
                                    lower_from_reduced=float(np.sqrt(red["beta0"] ** 2 + forc.dt * np.sum(lo ** 2))),
                                    theta_bar=tbar, Theta=full["Theta"], three_stage=stat,
                                    accept={k: {f"{t:.4g}": bool(v <= t) for t in EPS_SWEEP} for k, v in b.items()})
    return out


def do_table_ext(res, nproc):
    Qa, _ = Q_ops(res)
    from cgmy_cert.three_stage import STATUSES
    res.setdefault("table_ext", {})
    for n in (800, 4000):
        with Pool(nproc) as pool:
            rows = pool.map(_table_ext_one, [(i, Qa, n) for i in range(30)], chunksize=1)
        summ = {}
        for r in RANKS_EXT:
            R_ = [x["ranks"][str(r)] for x in rows]
            s = dict(median_true_err=float(np.median([x["true_err"] for x in R_])),
                     max_true_err=float(max(x["true_err"] for x in R_)),
                     lower_le_exact_all=bool(all(x["lower_from_reduced"] <= x["bounds"]["exact_AI"] * (1 + 1e-12) for x in R_)))
            for b in ("reduced", "reduced_tightened", "banded_AI", "exact_AI"):
                v = np.array([x["bounds"][b] for x in R_]); ef = np.array([x["effectivity"][b] for x in R_])
                s[b] = dict(median=float(np.median(v)), max=float(v.max()), median_effectivity=float(np.median(ef)),
                            max_effectivity=float(ef.max()),
                            accepted={f"{t:.4g}": int(sum(x["accept"][b][f"{t:.4g}"] for x in R_)) for t in EPS_SWEEP},
                            false_accepts={f"{t:.4g}": int(sum(x["accept"][b][f"{t:.4g}"] and x["true_err"] > t for x in R_))
                                           for t in EPS_SWEEP})
            s["three_stage"] = {f"{t:.4g}": {st: int(sum(x["three_stage"][f"{t:.4g}"] == st for x in R_)) for st in STATUSES}
                                for t in EPS_SWEEP}
            summ[str(r)] = s
            k = f"{EPS_N:.4g}"
            print(f"n={n} r={r}: err {s['median_true_err']:.2e} | acc@eps_N red {s['reduced']['accepted'][k]} band "
                  f"{s['banded_AI']['accepted'][k]} exact {s['exact_AI']['accepted'][k]} | 3-stage {s['three_stage'][k]}", flush=True)
        res["table_ext"][str(n)] = dict(summary=summ, rows=rows, Q=Qa, basis=pod128_file(n).name, ranks=list(RANKS_EXT))
        save(res)
    res["table_ext_meta"] = dict(description="follow-up 5 item 1-2: POD-Galerkin r = 8..128, both bases (r_max = 128 POD), "
                                 "validation30; bounds augmented; three-stage statuses at every eps of the sweep",
                                 eps_sweep=list(EPS_SWEEP), provenance=provenance(nproc=nproc))
    save(res)


def do_timing_ext(res, n_q=10):
    """Idle single-thread timing per (basis, r): reduced solve, online certificate (ell + eps_Q + form), and the
    retained row count rho of the compressed factor (range over the 18 cells and the timed queries' cells); the
    ranks where the certificate time crosses 5 ms and 50 ms (log-log interpolation)."""
    assert os.environ.get("OMP_NUM_THREADS") == "1", "run with OMP_NUM_THREADS=1 on an idle machine"
    Qa, _ = Q_ops(res)
    e = load_eim(N, build_if_missing=False)
    qs = val_queries()[:n_q]
    out = {}
    for n in (800, 4000):
        Vall = np.load(pod128_file(n))["V"]
        per = {}
        for r in RANKS_EXT:
            rc = load_rc(rc_ext_file(n, r, Qa)); V = Vall[:, :r]; rom = AffineROM(V, e, Q=Qa)
            rows, rho_q = [], []
            for q in qs:
                cid = Partition(SPEC).cell_of(q)
                info = e.eps_Q_graded(q, Qa)
                sol = rom.solve(q, rc.alpha0(), NT, ell=info["ell"])
                f_solve = lambda: rom.solve(q, rc.alpha0(), NT, ell=info["ell"])
                f_cert = lambda: (e.ell((q.G, q.M, q.Y), Qa), e.eps_Q_graded(q, Qa, ell=info["ell"]),
                                  rc.certify(q, sol.alpha, sol.dt, sol.bL, info["ell"], info["eps_Q"], cid=cid))
                f_form = lambda: rc.certify(q, sol.alpha, sol.dt, sol.bL, info["ell"], info["eps_Q"], cid=cid)
                for f in (f_solve, f_cert, f_form):
                    f()
                rows.append(dict(solve=_median_time(f_solve), certificate=_median_time(f_cert), form=_median_time(f_form)))
                rho_q.append(int(rc.cells[cid].R.shape[1]))
            ks = [int(c.R.shape[1]) for c in rc.cells.values()]
            per[str(r)] = dict({k: float(np.median([x[k] for x in rows]) * 1e3) for k in rows[0]},
                               rho_cells=[min(ks), max(ks)], rho_timed=[min(rho_q), max(rho_q)],
                               columns=(Qa + 2) * (r + 1))
            print(f"n={n} r={r}: solve {per[str(r)]['solve']:.2f} ms, certificate {per[str(r)]['certificate']:.2f} ms "
                  f"(form {per[str(r)]['form']:.2f}), rho {per[str(r)]['rho_cells']}", flush=True)
        rs = np.array(RANKS_EXT, float); ct = np.array([per[str(r)]["certificate"] for r in RANKS_EXT])
        cross = {}
        for lim in (5.0, 50.0):
            j = np.nonzero(ct > lim)[0]
            if j.size == 0:
                cross[f"{lim:g}ms"] = None
            elif j[0] == 0:
                cross[f"{lim:g}ms"] = float(rs[0])
            else:
                a, b = j[0] - 1, j[0]
                t = (np.log(lim) - np.log(ct[a])) / (np.log(ct[b]) - np.log(ct[a]))
                cross[f"{lim:g}ms"] = float(np.exp(np.log(rs[a]) + t * (np.log(rs[b]) - np.log(rs[a]))))
        out[str(n)] = dict(per_rank=per, crossing_rank=cross)
    import subprocess
    res["timing_ext"] = dict(per_basis=out, Q=Qa, unit="ms, median of per-query medians", n_queries=n_q,
                             loadavg=subprocess.run(["cat", "/proc/loadavg"], capture_output=True, text=True).stdout.strip(),
                             note="one thread, idle machine; certificate = ell(nu) + eps_Q (161 fixed nodes) + reduced form; "
                                  "rho = rows of the SVD-compressed factor; form cost ~ (Q+2) rho (r+1) + rho (r+1) N_T",
                             provenance=provenance())
    save(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["pod", "eimq", "cells", "tensors", "checks", "table"])
    ap.add_argument("--nproc", type=int, default=20)
    a = ap.parse_args()
    res = load()
    res["meta"] = dict(description="Production pipeline at N = 1024 (follow-up 4, item 2)", N=N, N_T=NT, eps_N=EPS_N,
                       production_config="results/PRODUCTION_CONFIG.json")
    for st in a.steps:
        dict(pod=lambda: do_pod(res), eimq=lambda: do_eimq(res, a.nproc), cells=lambda: do_cells(res, a.nproc),
             tensors=lambda: do_tensors(res), checks=lambda: do_checks(res, a.nproc), table=lambda: do_table(res, a.nproc),
             podext=lambda: do_podext(res), tensors_ext=lambda: do_tensors_ext(res),
             table_ext=lambda: do_table_ext(res, a.nproc), timing_ext=lambda: do_timing_ext(res))[st]()


if __name__ == "__main__":
    main()
