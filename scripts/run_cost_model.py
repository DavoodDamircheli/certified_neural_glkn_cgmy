#!/usr/bin/env python3
"""Phase 3, item 0 (plan 2.6): cost model with measured constants -> results/cost_model.json, tables_v9/tab_cost_v9.tex.

Per N in {256, 512, 1024, 2048, 4096} (5 bench120 queries, timing protocol of CLAUDE_PHASE3.md):
  assembly        free_matrices (symbol on the L-grid, L (2K+1) alias terms)          ops = L (2K+1)
  spectral_data   graded grid (fixed size), all spectral quantities                     ops = n_grid (2K+1)
  direct solve    CN march, circulant-preconditioned GMRES                              ops = (sum_n iters_n + N_T) m2 log2 m2
  residual        CN residual of a candidate (batched FFT pair)                         ops = (N_T + 1) m2 log2 m2
  banded eta^X    banded Cholesky + N_T banded solves                                   ops = N_T n_free
  exact eta^S     N_T PCG brackets                                                      ops = sum_n pcg_n m2 log2 m2
with m2 = 2^ceil(log2(2 n_free - 1)) the FFT length of the Toeplitz products.  At N = 1024 (the tensors are N-specific):
  ell + eps_Q     161 nodes x (2K+1) aliases + Q magic points                           ops = 161 (2K+1)  (N-independent)
  reduced solve   (Q+2) r^2 assembly + r^3 LU + N_T r^2 march                            ops = (Q+2) r^2 + r^3 + 3 N_T r^2
  reduced form    rho (Q+2)(r+1) contraction + 2 rho (r+1) N_T products                 ops = rho (r+1) (Q+2 + 2 N_T)
One constant per component, c = argmin sum (log t - log c ops)^2 (geometric-mean fit); fit residual = max and rms of
|log(t / (c ops))|.  rho(N) for N = 256..4096 from E2_3_reduced_cert.json (timing_v3).
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, Mesh, load_queries_json  # noqa: E402
from cgmy_cert.symbols import free_matrices, graded_theta_grid  # noqa: E402
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.direct import free_forcing, initial_free_vector, cn_march  # noqa: E402
from cgmy_cert.certificate import certify, cn_residuals  # noqa: E402
from cgmy_cert.eim import load_eim  # noqa: E402
from cgmy_cert.pod import AffineROM  # noqa: E402
from cgmy_cert.reduced_cert import load_rc, Partition  # noqa: E402
from cgmy_cert.timing import pin, check_single_thread, hardware, loadavg, protocol  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

PC = json.loads((ROOT / "results" / "PRODUCTION_CONFIG.json").read_text())
NT, K = PC["N_T"], 100
NS = (256, 512, 1024, 2048, 4096)
RANKS = (16, 32, 48, 64, 80, 128)
RES = ROOT / "results"; TAB = ROOT / "tables_v9"; DATA = ROOT / "data"
OUT = RES / "cost_model.json"


def m2_of(n):
    return 1 << int(np.ceil(np.log2(2 * n - 1)))


def fit(times, ops, calls=None):
    """One constant: t = c ops (geometric-mean fit).  With `calls` also the two-constant model t = c ops + d calls
    (per-step / per-iteration interpreter overhead), by non-negative least squares on relative residuals."""
    t, o = np.asarray(times, float), np.asarray(ops, float)
    c = float(np.exp(np.mean(np.log(t / o))))
    lr = np.log(t / (c * o))
    out = dict(constant_s_per_op=c, fit_residual_max_log=float(np.max(np.abs(lr))),
               fit_residual_rms_log=float(np.sqrt(np.mean(lr ** 2))), fit_max_rel=float(np.max(np.abs(np.exp(lr) - 1))))
    if calls is not None:
        from scipy.optimize import nnls
        k = np.asarray(calls, float)
        A = np.column_stack([o / t, k / t])                     # relative residual (c o + d k)/t - 1
        (c2, d2), _ = nnls(A, np.ones_like(t))
        pred = c2 * o + d2 * k
        rel = pred / t - 1
        out["two_constant"] = dict(c_s_per_op=float(c2), d_s_per_call=float(d2), fit_max_rel=float(np.max(np.abs(rel))),
                                   fit_rms_rel=float(np.sqrt(np.mean(rel ** 2))))
    return out


def full_order_rows(queries, warm):
    rows = []
    for N in NS:
        msh = Mesh(N=N); n = msh.n_free; m2 = m2_of(n)
        ctx = {}

        def setup(q):
            fs = free_matrices(q, N)
            forc = free_forcing(fs, q.T, NT)
            z0 = initial_free_vector(q, fs.mesh)
            sp = spectral_data(q, N)
            tr = cn_march(fs, z0, q.T, NT, solver="gmres", forcing=forc)
            Z = tr.Z + 1e-4 * np.random.default_rng(1).standard_normal(tr.Z.shape) * np.abs(tr.Z).max()
            cS = certify(fs, forc, Z, z0, 0.0, mode="exact", spec=sp)
            ctx[q.query_id] = dict(fs=fs, forc=forc, z0=z0, sp=sp, Z=Z, iters=int(np.sum(tr.iters)), pcg=int(np.sum(cS.pcg_iters)),
                                   L=fs.sym.L, n_grid=int(graded_theta_grid(q, N).size))

        for q in [warm] + queries:
            setup(q)
        meth = dict(
            assembly=lambda q: free_matrices(q, N),
            spectral_data=lambda q: spectral_data(q, N),
            direct_solve=lambda q: cn_march(ctx[q.query_id]["fs"], ctx[q.query_id]["z0"], q.T, NT, solver="gmres",
                                            forcing=ctx[q.query_id]["forc"]),
            residual=lambda q: cn_residuals(ctx[q.query_id]["fs"], ctx[q.query_id]["Z"], ctx[q.query_id]["forc"]),
            banded=lambda q: certify(ctx[q.query_id]["fs"], ctx[q.query_id]["forc"], ctx[q.query_id]["Z"], ctx[q.query_id]["z0"],
                                     0.0, mode="banded", spec=ctx[q.query_id]["sp"]),
            exact=lambda q: certify(ctx[q.query_id]["fs"], ctx[q.query_id]["forc"], ctx[q.query_id]["Z"], ctx[q.query_id]["z0"],
                                    0.0, mode="exact", spec=ctx[q.query_id]["sp"]))
        t = protocol(meth, queries, warm, reps=5, seed=N)
        for i, q in enumerate(queries):
            c = ctx[q.query_id]
            lg = m2 * np.log2(m2)
            ops = dict(assembly=c["L"] * (2 * K + 1), spectral_data=c["n_grid"] * (2 * K + 1),
                       direct_solve=(c["iters"] + NT) * lg, residual=(NT + 1) * lg, banded=NT * n, exact=c["pcg"] * lg)
            # banded/exact timings include the residual; the component time is the difference
            tb = max(t["banded"][i] - t["residual"][i], 1e-9); te = max(t["exact"][i] - t["residual"][i], 1e-9)
            comp = dict(assembly=t["assembly"][i], spectral_data=t["spectral_data"][i], direct_solve=t["direct_solve"][i],
                        residual=t["residual"][i], banded=tb, exact=te)
            rows.append(dict(N=N, n_free=n, m2=m2, query_id=q.query_id, gmres_iters_total=c["iters"],
                             gmres_iters_mean=c["iters"] / NT, pcg_iters_total=c["pcg"], pcg_iters_mean=c["pcg"] / NT,
                             L=c["L"], n_graded=c["n_grid"], times_s=comp, ops=ops))
        print(f"N={N}: " + ", ".join(f"{k} {np.median([r['times_s'][k] for r in rows if r['N'] == N]) * 1e3:.1f} ms"
                                     for k in rows[-1]["times_s"]), flush=True)
    return rows


def reduced_rows(queries, warm, Q):
    N = 1024
    e = load_eim(N, build_if_missing=False)
    Vall = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_800_r128.npz")["V"]
    rows = []
    P = Partition("CYm:3x3x2")
    eps_t = protocol(dict(ell_epsQ=lambda q: e.eps_Q_graded(q, Q)), queries, warm, reps=5, seed=7)   # computes ell(nu) inside
    for r in RANKS:
        rc = load_rc(DATA / f"redcert_N{N}_r{r}_n800_CYm_3x3x2_Q{Q}c_b128.npz")
        rom = AffineROM(Vall[:, :r], e, Q=Q)
        ctx = {}
        for q in [warm] + queries:
            info = e.eps_Q_graded(q, Q)
            sol = rom.solve(q, rc.alpha0(), NT, ell=info["ell"])
            cid = P.cell_of(q)
            ctx[q.query_id] = dict(info=info, sol=sol, cid=cid, rho=int(rc.cells[cid].R.shape[1]))
        meth = dict(reduced_solve=lambda q: rom.solve(q, rc.alpha0(), NT, ell=ctx[q.query_id]["info"]["ell"]),
                    reduced_form=lambda q: rc.certify(q, ctx[q.query_id]["sol"].alpha, ctx[q.query_id]["sol"].dt,
                                                      ctx[q.query_id]["sol"].bL, ctx[q.query_id]["info"]["ell"],
                                                      ctx[q.query_id]["info"]["eps_Q"], cid=ctx[q.query_id]["cid"]))
        t = protocol(meth, queries, warm, reps=5, seed=r)
        for i, q in enumerate(queries):
            rho = ctx[q.query_id]["rho"]
            rows.append(dict(N=N, r=r, Q=Q, query_id=q.query_id, rho=rho,
                             times_s=dict(reduced_solve=t["reduced_solve"][i], reduced_form=t["reduced_form"][i],
                                          ell_epsQ=eps_t["ell_epsQ"][i]),
                             ops=dict(reduced_solve=(Q + 2) * r * r + r ** 3 + 3 * NT * r * r,
                                      reduced_form=rho * (r + 1) * (Q + 2 + 2 * NT), ell_epsQ=161 * (2 * K + 1))))
        print(f"r={r}: solve {np.median(t['reduced_solve']) * 1e3:.2f} ms, form {np.median(t['reduced_form']) * 1e3:.2f} ms", flush=True)
    return rows


def main():
    check_single_thread(); pin()
    la0 = loadavg()
    pops = load_queries_json(ROOT / "tests" / "fixtures" / "queries.json")
    queries = pops["bench120"][:5]
    warm = pops["validation30"][0]
    Q = json.loads((RES / "N1024.json").read_text())["eimq"]["per_target"]["eps_N"]["Q_op"]
    t0 = time.time()
    fo = full_order_rows(queries, warm)
    red = reduced_rows(queries, warm, Q)
    fits = {}
    calls_fo = dict(direct_solve=lambda r: r["gmres_iters_total"] + NT, exact=lambda r: r["pcg_iters_total"] + NT,
                    residual=lambda r: 1, banded=lambda r: NT, assembly=lambda r: 1, spectral_data=lambda r: 1)
    for comp in ("assembly", "spectral_data", "direct_solve", "residual", "banded", "exact"):
        fits[comp] = fit([r["times_s"][comp] for r in fo], [r["ops"][comp] for r in fo],
                         [calls_fo[comp](r) for r in fo] if comp in ("direct_solve", "exact", "banded") else None)
    for comp in ("reduced_solve", "reduced_form", "ell_epsQ"):
        fits[comp] = fit([r["times_s"][comp] for r in red], [r["ops"][comp] for r in red],
                         [NT for r in red] if comp in ("reduced_solve", "reduced_form") else None)
    v3 = json.loads((RES / "E2_3_reduced_cert.json").read_text())["timing_v3"]["per_N"]
    rho_N = {k: dict(rows_before=v["rows_before"], rows_after_range=v["rows_after_range"], Q=v["Q"])
             for k, v in v3.items() if k.isdigit()}
    med = {}
    for N in NS:
        med[str(N)] = {k: float(np.median([r["times_s"][k] for r in fo if r["N"] == N])) for k in fo[0]["times_s"]}
        med[str(N)].update(gmres_iters_mean=float(np.median([r["gmres_iters_mean"] for r in fo if r["N"] == N])),
                           pcg_iters_mean=float(np.median([r["pcg_iters_mean"] for r in fo if r["N"] == N])))
    medr = {str(r): {k: float(np.median([x["times_s"][k] for x in red if x["r"] == r])) for k in red[0]["times_s"]}
            | dict(rho_median=float(np.median([x["rho"] for x in red if x["r"] == r]))) for r in RANKS}
    out = dict(meta=dict(description=__doc__.strip().splitlines()[0], N_list=list(NS), ranks=list(RANKS), Q=Q, N_T=NT, K=K,
                         population="bench120[:5]", warmup="validation30[0]", unit="seconds", wall_s=time.time() - t0,
                         loadavg_before=la0, loadavg_after=loadavg(), hardware=hardware(), provenance=provenance(),
                         reduced_cost_note="the reduced solve and certificate do not depend on N by construction: their cost "
                                           "is in (Q, r, rho, N_T), and rho saturates (rows_after_range, Q = 29) for N >= 512"),
               fits=fits, median_times_full_order=med, median_times_reduced=medr, rho_vs_N=rho_N,
               rows_full_order=fo, rows_reduced=red)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    write_table(out)
    print(json.dumps(fits, indent=1))


def write_table(out):
    f = out["fits"]; med = out["median_times_full_order"]; mr = out["median_times_reduced"]
    lines = [r"\begin{table}[!htbp]", r"\centering\small",
             r"\caption{Cost model with measured constants (one thread, idle machine; median over 5 bench queries). "
             r"Times in ms; $c$ = fitted seconds per operation of the stated count (one constant), fit residual = max relative "
             r"deviation; for components with a per-step interpreter overhead also $t=c'\,\mathrm{ops}+d\,\mathrm{calls}$.}",
             r"\label{tab:cost}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"component & op count & " + " & ".join(f"$N={N}$" for N in NS) + r" & $c$ (fit res.)\\", r"\midrule"]
    names = dict(assembly=("assembly", r"$L(2K+1)$"), spectral_data=("spectral data", r"$n_g(2K+1)$"),
                 direct_solve=("direct solve", r"$(\sum_n i_n+N_T)\,m\log m$"), residual=("residual", r"$N_T\,m\log m$"),
                 banded=(r"banded $\eta^X$", r"$N_T N_h$"), exact=(r"exact $\eta^S$", r"$\sum_n p_n\,m\log m$"))
    for k, (lab, opc) in names.items():
        tc = f[k].get("two_constant")
        extra = f"; $c'={tc['c_s_per_op']:.1e}$, $d={tc['d_s_per_call']:.1e}$ ({tc['fit_max_rel'] * 100:.0f}\\%)" if tc else ""
        lines.append(f"{lab} & {opc} & " + " & ".join(f"{med[str(N)][k] * 1e3:.2f}" for N in NS)
                     + f" & ${f[k]['constant_s_per_op']:.2e}$ ({f[k]['fit_max_rel'] * 100:.0f}\\%){extra}\\\\")
    lines += [r"\midrule", r"\multicolumn{8}{l}{reduced model at $N=1024$, $Q=" + str(out['meta']['Q']) + r"$ (independent of $N$ by construction)}\\",
              r"component & op count & " + " & ".join(f"$r={r}$" for r in list(mr)[:5]) + r" & $c$ (fit res.)\\"]
    for k, (lab, opc) in dict(reduced_solve=("reduced solve", r"$(Q{+}2)r^2+r^3+3N_Tr^2$"),
                              reduced_form=("reduced certificate", r"$\rho(r{+}1)(Q{+}2{+}2N_T)$"),
                              ell_epsQ=(r"$\ell(\nu)+\varepsilon_Q$", r"$161(2K+1)$")).items():
        tc = f[k].get("two_constant")
        extra = f"; $c'={tc['c_s_per_op']:.1e}$, $d={tc['d_s_per_call']:.1e}$ ({tc['fit_max_rel'] * 100:.0f}\\%)" if tc else ""
        lines.append(f"{lab} & {opc} & " + " & ".join(f"{mr[r][k] * 1e3:.2f}" for r in list(mr)[:5])
                     + f" & ${f[k]['constant_s_per_op']:.2e}$ ({f[k]['fit_max_rel'] * 100:.0f}\\%){extra}\\\\")
    lines.append(r"$\rho$ (retained rows) & & " + " & ".join(f"{mr[r]['rho_median']:.0f}" for r in list(mr)[:5]) + r" & \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TAB.mkdir(exist_ok=True)
    (TAB / "tab_cost_v9.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
