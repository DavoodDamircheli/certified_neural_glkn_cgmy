#!/usr/bin/env python3
"""Phase 3.1: resolution and tolerance calibration.

ref30 + the two v8 corner queries (Q_audit_0397 = Corner A, Q_audit_0401 = Corner B, tab_corner.tex of v8):
discrete price V_h(T, S) vs COS at N in {128, 256, 512, 1024}, N_T in {60, 240}, spots S = 0.8K..1.2K.
COS: v8_frozen/numerics/validation/cos_pricer.py (cos_put_price, 8192 modes, range parameter L = 30),
cross-checked against the v8 Carr-Madan pricer on 10 (query, spot) rows.  The Galerkin price is the
raw coefficient vector at tau = T (free part from a dense-LU CN solve, prescribed boundary values) evaluated
at x = log S.
Production N = smallest N with max |V_h - V_COS| <= 1 cent over queries and spots (at the chosen N_T);
eps_N = tau_$ / max_s beta_I(x_s) (direct.spot_price_factor), tau_$ = 1 cent; also tau_$ = 0.1 cent at N = 1024.
Outputs: results/E3_1_calibration.json, tables_v9/tab_budget_v9.tex, results/PRODUCTION_CONFIG.json.
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v8_frozen" / "numerics"))
from cgmy_cert.params import Query, Mesh, load_queries_json  # noqa: E402
from cgmy_cert.direct import reference_trajectory, raw_basis_eval, lifting, spot_price_factor  # noqa: E402
from cgmy_cert.symbols import free_matrices  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

RES = ROOT / "results"; TAB = ROOT / "tables_v9"
OUT = RES / "E3_1_calibration.json"
NS = (128, 256, 512, 1024)
NTS = (60, 240)
SPOTS = tuple(100.0 * f for f in (0.8, 0.9, 1.0, 1.1, 1.2))
COS_N, COS_L = 8192, 30.0
CORNERS = ("Q_audit_0397", "Q_audit_0401")
SWEEP = (0.1, 0.3, 1.0, 3.0, 10.0)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1, default=float))


def queries():
    pops = load_queries_json(ROOT / "tests" / "fixtures" / "queries.json")
    audit = {q.query_id: q for q in pops["audit459"]}
    return pops["ref30"] + [audit[c] for c in CORNERS]


def _mkt(q):
    from validation.cgmy_reference_pricing import CGMYMarketParams
    return CGMYMarketParams(C=q.C, G=q.G, M=q.M, Y=q.Y, r=q.r, T=q.T)


def _cos(qd):
    from validation.cos_pricer import cos_put_price
    q = Query(**qd)
    t0 = time.time()
    out = {f"{S:g}": cos_put_price(_mkt(q), S, q.K, N=COS_N, L=COS_L)["price"] for S in SPOTS}
    return q.query_id, out, time.time() - t0


def _cm(args):
    qd, S = args
    from validation.independent_fourier_integration import carr_madan_put_price
    q = Query(**qd)
    r = carr_madan_put_price(_mkt(q), S, q.K)
    return q.query_id, S, r["price"], r["cutoff_cap_hit"]


def _galerkin(args):
    qd, N, NT = args
    q = Query(**qd)
    fs, forc, tr = reference_trajectory(q, N, NT, solver="lu")
    msh = fs.mesh
    c = np.zeros(msh.n_raw)
    c[msh.free_slice] = tr.Z[:, -1]
    c[: msh.p - 1] = lifting(q, msh, np.array([q.T]))[0]
    x = np.log(np.array(SPOTS))
    V = raw_basis_eval(x, msh) @ c
    beta = spot_price_factor(fs, x)
    return q.query_id, N, NT, {f"{S:g}": float(v) for S, v in zip(SPOTS, V)}, {f"{S:g}": float(b) for S, b in zip(SPOTS, beta)}


def do_cos(res, nproc):
    qs = queries()
    t0 = time.time()
    with Pool(nproc) as pool:
        out = pool.map(_cos, [q.as_dict() for q in qs], chunksize=1)
    res["cos"] = {qid: dict(prices=p, wall_s=w) for qid, p, w in out}
    res["cos_meta"] = dict(pricer="v8_frozen/numerics/validation/cos_pricer.py: cos_put_price", modes=COS_N,
                           range_parameter_L=COS_L, wall_s=time.time() - t0, provenance=provenance(nproc=nproc))
    save(res)
    print(f"COS: {len(out)} queries, {time.time() - t0:.0f}s", flush=True)


def do_cm(res, nproc, n_rows=10):
    qs = queries()
    rows = [(qs[(7 * i) % len(qs)].as_dict(), SPOTS[i % len(SPOTS)]) for i in range(n_rows)]
    with Pool(nproc) as pool:
        out = pool.map(_cm, rows, chunksize=1)
    chk = []
    for qid, S, p, cap in out:
        c = res["cos"][qid]["prices"][f"{S:g}"]
        chk.append(dict(query_id=qid, spot=S, carr_madan=p, cos=c, abs_diff=abs(p - c), cutoff_cap_hit=cap))
    md = max(r["abs_diff"] for r in chk)
    res["cos_vs_carr_madan"] = dict(rows=chk, max_abs_diff=md, within_1e12=bool(md <= 1e-12),
                                    pricer="v8 carr_madan_put_price (mpmath, dps 40, max_v 400)")
    save(res)
    print(f"COS vs Carr-Madan, {n_rows} rows: max |diff| = {md:.2e}", flush=True)


def do_galerkin(res, nproc):
    qs = queries()
    jobs = [(q.as_dict(), N, NT) for q in qs for N in NS for NT in NTS]
    t0 = time.time()
    with Pool(nproc) as pool:
        out = pool.map(_galerkin, jobs, chunksize=1)
    g = {}
    for qid, N, NT, V, b in out:
        g.setdefault(qid, {})[f"{N}|{NT}"] = dict(V=V, beta=b)
    res["galerkin"] = g
    res["galerkin_meta"] = dict(solver="reference_trajectory(q, N, N_T, solver='lu') (exact assembly, K = 100)",
                                wall_s=time.time() - t0, provenance=provenance(nproc=nproc))
    save(res)
    print(f"Galerkin: {len(out)} solves, {time.time() - t0:.0f}s", flush=True)


def do_summary(res):
    qs = queries()
    ids_ref = [q.query_id for q in qs if q.query_id not in CORNERS]
    table = {}
    for N in NS:
        for NT in NTS:
            k = f"{N}|{NT}"
            err = {qid: max(abs(res["galerkin"][qid][k]["V"][f"{S:g}"] - res["cos"][qid]["prices"][f"{S:g}"])
                            for S in SPOTS) for qid in res["galerkin"]}
            beta_max = max(res["galerkin"][qid][k]["beta"][f"{S:g}"] for qid in res["galerkin"] for S in SPOTS)
            table[k] = dict(N=N, N_T=NT, max_err_all=max(err.values()), max_err_ref30=max(err[i] for i in ids_ref),
                            err_cornerA=err[CORNERS[0]], err_cornerB=err[CORNERS[1]],
                            argmax_query=max(err, key=err.get), max_beta=beta_max,
                            eps_1cent=0.01 / beta_max, eps_0p1cent=0.001 / beta_max)
    res["table"] = table
    # production choice: smallest N with max error <= 1 cent; prefer N_T = 240 (the provisional choice), and
    # report whether 60 steps would also do
    prod = None
    for N in NS:
        if table[f"{N}|240"]["max_err_all"] <= 0.01:
            prod = (N, 240)
            break
    if prod is None:
        res["production"] = dict(found=False, note="no N in {128,...,1024} meets 1 cent at N_T = 240")
    else:
        N, NT = prod
        t = table[f"{N}|{NT}"]
        eps_N = t["eps_1cent"]
        cfg = dict(N=N, N_T=NT, tau_dollar=0.01, eps_N=eps_N, eps_sweep=[eps_N * f for f in SWEEP],
                   sweep_factors=list(SWEEP), max_beta_I=t["max_beta"], max_price_err=t["max_err_all"],
                   N_T_60_also_within_1cent=bool(table[f"{N}|60"]["max_err_all"] <= 0.01),
                   eps_0p1cent_at_N1024=table["1024|240"]["eps_0p1cent"],
                   matches_provisional=bool(N == 512 and NT == 240),
                   source="results/E3_1_calibration.json", provenance=provenance())
        res["production"] = cfg
        (RES / "PRODUCTION_CONFIG.json").write_text(json.dumps(cfg, indent=1, default=float))
    save(res)
    for k, t in table.items():
        print(f"N={t['N']:5d} N_T={t['N_T']:3d}: max err {t['max_err_all']:.2e} (ref30 {t['max_err_ref30']:.2e}, "
              f"A {t['err_cornerA']:.2e}, B {t['err_cornerB']:.2e}) beta_max {t['max_beta']:.3f} eps_1c {t['eps_1cent']:.3e}")
    print("production:", res["production"])


def fmt(x):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.2f}\\times10^{{{e}}}" if not -2 <= e <= 2 else f"{x:.3g}"


def do_table(res):
    t = res["table"]
    lines = [r"\begin{table}[!htbp]", r"\centering\small",
             r"\caption{Resolution and tolerance calibration (Phase 3.1): maximum price error $\max|V_h(T,S)-V_{\mathrm{COS}}|$ "
             r"over the 30 ref queries and the two v8 corner queries, spots $S\in\{0.8,\dots,1.2\}K$; COS with 8192 modes, "
             r"range parameter 30. $\varepsilon_N=\tau_\$/\max_s\beta_I(x_s)$ with $\tau_\$=1$ cent.}",
             r"\label{tab:budget}", r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$N$ & $N_T$ & ref30 & Corner A & Corner B & $\max_s\beta_I$ & $\varepsilon_N$\\", r"\midrule"]
    for k, r in t.items():
        lines.append(f"{r['N']} & {r['N_T']} & ${fmt(r['max_err_ref30'])}$ & ${fmt(r['err_cornerA'])}$ & "
                     f"${fmt(r['err_cornerB'])}$ & ${r['max_beta']:.3f}$ & ${fmt(r['eps_1cent'])}$\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TAB.mkdir(exist_ok=True)
    (TAB / "tab_budget_v9.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TAB / "tab_budget_v9.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["cos", "cm", "galerkin", "summary", "table"])
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    res = load()
    res["meta"] = dict(description="Phase 3.1 calibration: prices in currency units (K = 100); errors absolute",
                       population="ref30 + Q_audit_0397 (Corner A) + Q_audit_0401 (Corner B)", N=list(NS),
                       N_T=list(NTS), spots=list(SPOTS))
    for st in a.steps:
        dict(cos=lambda: do_cos(res, a.nproc), cm=lambda: do_cm(res, a.nproc), galerkin=lambda: do_galerkin(res, a.nproc),
             summary=lambda: do_summary(res), table=lambda: do_table(res))[st]()


if __name__ == "__main__":
    main()
