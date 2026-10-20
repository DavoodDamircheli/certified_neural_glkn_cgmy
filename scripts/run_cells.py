#!/usr/bin/env python3
"""Follow-up 2, 2c: cells in (log C, Y, log min(G, M)) for the reduced certificate's comparison norm X-bar.

    table       beta~(G, M, Y) = spectral_data(q with C = 1, N = 512).beta on a 9 x 10 x 13 grid
                (beta_mu = C beta~(G, M, Y): s_mu - r m is proportional to C); cell boundaries are grid nodes
    partitions  per partition: betabar(cell) = 0.99 * C_lo(cell) * min of the table over the cell (edges incl.),
                rbar = r_min of the box
    validity    min over a 400-query Sobol sample of c_mu = min_theta (s_mu - eps)/xbar (X-bar <= S_mu iff >= 1)
    effect      validation30, POD-Galerkin r = 48 (basis 800, A_I): residual parts of eta^{X-bar} vs the query's
                own eta^X and eta^S, for every partition (old rC:2x2 included)
    build       R-factor tensors (Q = q_max of the N = 512 EIM) for the chosen partition, bases 800 and 4000
                -> data/redcert_N512_r48_n{n}_{spec}.npz
Writes results/E2_3_cells.json.
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, BOX, sobol_queries, load_queries_json  # noqa: E402
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.reduced_cert import Partition, xbar_entries, conversion_factor, ReducedCertificate, save_rc  # noqa: E402
from cgmy_cert.eim import load_eim, sha256  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

N, NT, R = 512, 240, 48
DATA = ROOT / "data"; RES = ROOT / "results"
OUT = RES / "E2_3_cells.json"; MANIFEST = RES / "MANIFEST.json"
PARTITIONS = ("rC:2x2", "CYm:2x2x2", "CYm:3x3x2")
RBAR = BOX["r"][0]


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1, default=float))


def table_axes():
    G = np.geomspace(*BOX["G"], 9)
    M = np.unique(np.concatenate([np.geomspace(*BOX["M"], 9), [np.sqrt(5.0)]]))       # sqrt(5): min(G,M) split
    Y = np.linspace(*BOX["Y"], 13)                                                     # contains 1.333, 1.45, 1.567
    return G, M, Y


def _beta1(args):
    G, M, Y = args
    return spectral_data(Query(C=1.0, G=G, M=M, Y=Y, r=0.05), N).beta


def do_table(res, nproc):
    G, M, Y = table_axes()
    pts = [(g, m, y) for g in G for m in M for y in Y]
    t0 = time.time()
    with Pool(nproc) as pool:
        b = np.array(pool.map(_beta1, pts, chunksize=4)).reshape(G.size, M.size, Y.size)
    mono = {ax: dict(nonincreasing=bool(np.all(np.diff(b, axis=i) <= 1e-12 * np.abs(b).max())),
                     nondecreasing=bool(np.all(np.diff(b, axis=i) >= -1e-12 * np.abs(b).max())))
            for i, ax in enumerate("GMY")}
    res["table"] = dict(G=G.tolist(), M=M.tolist(), Y=Y.tolist(), beta_C1=b.tolist(), monotonicity=mono,
                        min=float(b.min()), max=float(b.max()), wall_s=time.time() - t0,
                        note="beta~ = spectral_data(q, 512).beta at C = 1 (already contains the 0.99 safety); "
                             "beta_mu = C beta~", provenance=provenance(nproc=nproc))
    print("beta~ range", b.min(), b.max(), "monotone:", mono, flush=True)
    save(res)


def cell_params(res, spec):
    P = Partition(spec)
    t = res["table"]; G, M, Y = map(np.array, (t["G"], t["M"], t["Y"])); b = np.array(t["beta_C1"])
    gg, mm, yy = np.meshgrid(G, M, Y, indexing="ij")
    mn = np.minimum(gg, mm)
    out = {}
    tol = 1e-12
    for c in P.cells():
        if P.kind == "rC":
            sel = np.ones_like(b, dtype=bool)
        else:
            sel = ((yy >= c["Y"][0] - tol) & (yy <= c["Y"][1] + tol)
                   & (mn >= c["m"][0] * (1 - tol)) & (mn <= c["m"][1] * (1 + tol)))
        out[c["id"]] = (c["r"][0] if P.kind == "rC" else RBAR, 0.99 * c["C"][0] * float(b[sel].min()))
    return out


def do_partitions(res):
    res["partitions"] = {spec: {cid: dict(rbar=rb, betabar=bb) for cid, (rb, bb) in cell_params(res, spec).items()}
                         for spec in PARTITIONS}
    for spec, cp in res["partitions"].items():
        print(spec, {k: round(v["betabar"], 4) for k, v in cp.items()}, flush=True)
    save(res)


def _cmu(args):
    qd, spec, cp = args
    q = Query(**qd)
    cid = Partition(spec).cell_of(q)
    return conversion_factor(q, N, *cp[cid])


def do_validity(res, nproc, n=400):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qs = sobol_queries(n, seed=21, include_T=False)
    res.setdefault("validity", {})
    for spec in PARTITIONS:
        cp = cell_params(res, spec)
        with Pool(nproc) as pool:
            c = np.array(pool.map(_cmu, [(q.as_dict(), spec, cp) for q in qs], chunksize=4))
        res["validity"][spec] = dict(min_c_mu=float(c.min()), median_c_mu=float(np.median(c)), n=n,
                                     n_below_1=int((c < 1).sum()), population="sobol_queries(400, seed=21)")
        print(spec, res["validity"][spec], flush=True)
    save(res)


def _effect_one(args):
    i, specs_cp = args
    from cgmy_cert.direct import free_forcing
    from cgmy_cert.certificate import cn_residuals, banded_dual_norms, certify
    from cgmy_cert.symbols import free_matrices
    from cgmy_cert.pod import AffineROM
    from cgmy_cert.reduced_cert import load_rc
    d = np.load(DATA / f"references_N{N}_NT{NT}_validation30.npz")
    m = d["mu"][i]
    q = Query(C=m[0], G=m[1], M=m[2], Y=m[3], r=m[4], T=float(d["T"][i]), query_id=str(d["query_ids"][i]))
    e = load_eim(N, build_if_missing=False)
    V = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_800.npz")["V"][:, :R]
    rom = AffineROM(V, e)
    fs = free_matrices(q, N); forc = free_forcing(fs, q.T, NT)
    a0 = load_rc(DATA / f"redcert_N{N}_r{R}_n800.npz").alpha0()
    sol = rom.solve(q, a0, NT, with_eps=False)
    Z = V @ sol.alpha
    Rr = cn_residuals(fs, Z, forc)
    spec_q = spectral_data(q, N)
    cX = certify(fs, forc, Z, Z[:, 0], 1.0, mode="banded", spec=spec_q)
    cS = certify(fs, forc, Z, Z[:, 0], 1.0, mode="exact", spec=spec_q)
    out = dict(query_id=q.query_id, etaX=cX.etaX, etaS=cS.etaS_upper, cells={})
    for spec, cp in specs_cp.items():
        cid = Partition(spec).cell_of(q)
        qf, _ = banded_dual_norms(xbar_entries(*cp[cid]), fs.n, 2, Rr)
        ex = float(np.sqrt(forc.dt * np.sum(qf)))
        out["cells"][spec] = dict(cell=cid, eta_xbar=ex, over_etaX=ex / cX.etaX, over_etaS=ex / cS.etaS_upper)
    return out


def do_effect(res, nproc):
    specs_cp = {spec: cell_params(res, spec) for spec in PARTITIONS}
    with Pool(nproc) as pool:
        rows = pool.map(_effect_one, [(i, specs_cp) for i in range(30)], chunksize=1)
    summ = {}
    for spec in PARTITIONS:
        a = np.array([r["cells"][spec]["over_etaX"] for r in rows]); b = np.array([r["cells"][spec]["over_etaS"] for r in rows])
        summ[spec] = dict(median_over_etaX=float(np.median(a)), max_over_etaX=float(a.max()),
                          median_over_etaS=float(np.median(b)), max_over_etaS=float(b.max()))
        print(spec, summ[spec], flush=True)
    res["effectivity"] = dict(summary=summ, rows=rows,
                              description="validation30, POD-Galerkin r = 48 (basis 800, reduced solve with A_Q); "
                              "residual parts (no beta0, no allowance) of the dual norms, residual with A_I: "
                              "eta^{X-bar}(cell norm) / eta^X(own X_I) and / eta^S(exact, PCG upper bracket)",
                              provenance=provenance(nproc=nproc))
    save(res)


def do_build(res, spec, sets=(800, 4000)):
    man = json.loads(MANIFEST.read_text())
    e = load_eim(N, build_if_missing=False)
    cp = cell_params(res, spec)
    res.setdefault("build", {})
    for n in sets:
        V = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_{n}.npz")["V"][:, :R]
        rc = ReducedCertificate.build(V, e, cp, Q=int(e.magic.size), partition=spec)
        path = DATA / f"redcert_N{N}_r{R}_n{n}_{spec.replace(':', '_')}.npz"
        save_rc(rc, path)
        mb = rc.cells[next(iter(rc.cells))].R.nbytes / 2 ** 20
        man["files"][path.name] = dict(file=str(path.relative_to(ROOT)), sha256=sha256(path), N=N, r=R, Q_max=rc.Q,
                                       partition=spec, cells={k: list(v) for k, v in cp.items()},
                                       basis=f"pod_N{N}_NT{NT}_seed0_{n}.npz", provenance=provenance())
        res["build"][f"{spec}|{n}"] = dict(file=path.name, Q_max=rc.Q, n_cells=len(rc.cells), MB_per_cell=mb,
                                          total_MB=path.stat().st_size / 2 ** 20, offline_s=rc.time_offline_s)
        print(spec, n, res["build"][f"{spec}|{n}"], flush=True)
    MANIFEST.write_text(json.dumps(man, indent=1))
    save(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["table", "partitions", "validity", "effect"])
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--build", default=None, help="partition spec to build tensors for")
    a = ap.parse_args()
    res = load()
    res["meta"] = dict(description="Follow-up 2, 2c: comparison-norm cells for the reduced certificate", N=N,
                       partitions=list(PARTITIONS), rbar=RBAR)
    for st in a.steps:
        dict(table=lambda: do_table(res, a.nproc), partitions=lambda: do_partitions(res),
             validity=lambda: do_validity(res, a.nproc), effect=lambda: do_effect(res, a.nproc))[st]()
    if a.build:
        do_build(res, a.build)


if __name__ == "__main__":
    main()
