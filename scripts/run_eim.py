#!/usr/bin/env python3
"""Phase 2.1: build the EIM of the fractional symbol family for every production N, validate the
operator-norm bound eps_Q on the 1000 Sobol test queries (seed 2026) at N = 64 and 512, write
results/eim.json and figures/F4_eim.png.

The operational rank is Q = max(Q_tol, Q_op) (step `select`, see DEVIATIONS.md).
Usage: python3 scripts/run_eim.py [build] [select] [validate] [figure] [--N 64 128 256 512 1024] [--nproc 8]
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.figio import save_fig  # noqa: E402
from cgmy_cert.eim import (build_eim, load_eim, eim_path, sha256, eim_test_queries, family_grid,  # noqa: E402
                           TOL_SWEEP, TRAIN_SHAPE)
from cgmy_cert.symbols import free_matrices  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

RES = ROOT / "results"; FIG = ROOT / "figures"
OUT = RES / "eim.json"
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]   # Okabe-Ito, fixed order
N_DEFAULT = (64, 128, 256, 512, 1024)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1, default=float))


def do_build(Ns, nproc, res):
    res.setdefault("build", {})
    for N in Ns:
        t0 = time.time()
        e = build_eim(N, nproc=nproc)
        path = e.save(eim_path(N))
        wall = time.time() - t0
        res["build"][str(N)] = dict(
            file=str(path.relative_to(ROOT)), sha256=sha256(path), Q=e.Q, q_max=int(e.magic.size), tol=e.tol,
            L=e.L, n_free=e.n, Q_vs_tol=e.meta["Q_vs_tol"], test_err=e.test_err.tolist(),
            train_err=e.train_err.tolist(), singular_values=e.sv[:80].tolist(),
            nus_selected=e.nus[:e.Q].tolist(), lags_rel_imag_first_Q=e.meta["lags_rel_imag_first_Q"],
            wall_s=wall, nproc=nproc, provenance=provenance(nproc=nproc))
        print(f"N={N}: Q={e.Q} (tol {e.tol:g}), Q vs tol {e.meta['Q_vs_tol']}, {wall:.0f}s", flush=True)
        save(res)


# ---------------------------------------------------------------- operator-norm rank selection

OP_TOL = 1e-8          # done-when: eps_Q(mu) <= 1e-8 ||A_I(mu)||_2


def _curve_query(args):
    N, qd, Qs = args
    from cgmy_cert.params import Query
    q = Query(**qd)
    e = load_eim(N, build_if_missing=False)
    nA = float(np.linalg.norm(free_matrices(q, N).A.dense(), 2))
    return nA, e.eps_Q_curve(q, Qs).tolist()


def do_select(Ns, res, nproc, n_queries=1000, extra_terms=8):
    """Q = max(Q_tol, Q_op): Q_tol from the plan's stopping rule (test sup-error <= 1e-9), Q_op = smallest
    Q with max over the test queries of eps_Q / ||A_I||_2 <= 1e-8 (the done-when bound)."""
    qs = eim_test_queries(n_queries)
    res.setdefault("select", {})
    for N in Ns:
        e = load_eim(N, build_if_missing=False)
        Q_tol = int(e.meta.get("Q_tol", e.Q))
        Qs = list(range(max(Q_tol - 3, 1), min(Q_tol + extra_terms, e.magic.size) + 1))
        jobs = [(N, dict(C=q.C, G=q.G, M=q.M, Y=q.Y, r=q.r, T=q.T, K=q.K, query_id=q.query_id), Qs) for q in qs]
        t0 = time.time()
        with Pool(nproc) as pool:
            out = pool.map(_curve_query, jobs, chunksize=4)
        nA = np.array([o[0] for o in out]); eps = np.array([o[1] for o in out])
        ratio = eps / nA[:, None]
        worst = ratio.max(axis=0)
        ok = [Q for Q, w in zip(Qs, worst) if w <= OP_TOL]
        if not ok:
            raise RuntimeError(f"N={N}: no Q in {Qs} meets eps_Q <= {OP_TOL:g} ||A_I||")
        Q_op = ok[0]
        Q = max(Q_tol, Q_op)
        e.set_Q(Q, Q_tol=Q_tol, Q_op=Q_op, op_tol=OP_TOL, n_queries=n_queries)
        path = e.save(eim_path(N))
        res["select"][str(N)] = dict(Q_tol=Q_tol, Q_op=Q_op, Q=Q, Qs=Qs, max_eps_over_normA=worst.tolist(),
                                     n_violations=[int(v) for v in (ratio > OP_TOL).sum(axis=0)],
                                     wall_s=time.time() - t0, provenance=provenance(nproc=nproc))
        res["build"][str(N)].update(Q=Q, Q_tol=Q_tol, Q_op=Q_op, sha256=sha256(path))
        print(f"N={N}: Q_tol={Q_tol}, Q_op={Q_op} -> Q={Q}; max eps/||A|| per Q: "
              + ", ".join(f"{Qi}:{w:.1e}" for Qi, w in zip(Qs, worst)), flush=True)
        save(res)


# ---------------------------------------------------------------- validation (done-when of 2.1)

def _check_query(args):
    N, qd, dense, fine_factor = args
    from cgmy_cert.params import Query
    q = Query(**qd)
    e = load_eim(N, build_if_missing=False)
    t0 = time.perf_counter()
    info = e.eps_Q(q)
    t_eps = time.perf_counter() - t0
    t0 = time.perf_counter()
    AQ = e.A_Q(q, ell=info["ell"])
    t_AQ = time.perf_counter() - t0
    fs = free_matrices(q, N)
    out = dict(query_id=q.query_id, eps_Q=info["eps_Q"], eps_grid=info["grid"], eps_deriv=info["deriv"],
               t_eps_s=t_eps, t_AQ_s=t_AQ)
    # ||A_I||_2 (dense at every N used here: n_free <= 1022)
    A = fs.A.dense()
    out["normA"] = float(np.linalg.norm(A, 2))
    if dense:
        out["dense_err"] = float(np.linalg.norm(A - AQ.dense(), 2))
    if fine_factor:
        Lf = fine_factor * e.L
        _CACHE.setdefault((N, Lf), e.basis_on(Lf))
        xif = _CACHE[(N, Lf)]
        phi, _ = family_grid(Lf, (q.G, q.M, q.Y), e.h, e.K)
        out["fine_sup_err"] = info["scale"] * float(np.max(np.abs(phi - info["ell"] @ xif)))
    return out


_CACHE: dict = {}


def do_validate(res, nproc, Ns=(64, 512), n_queries=1000, n_dense_512=50):
    qs = eim_test_queries(n_queries)
    res.setdefault("validate", {})
    for N in [N for N in (64, 512) if N in Ns]:
        jobs = []
        for i, q in enumerate(qs):
            qd = dict(C=q.C, G=q.G, M=q.M, Y=q.Y, r=q.r, T=q.T, K=q.K, query_id=q.query_id)
            dense = N == 64 or i < n_dense_512
            jobs.append((N, qd, dense, 4 if N == 512 else 0))
        t0 = time.time()
        with Pool(nproc) as pool:
            rows = pool.map(_check_query, jobs, chunksize=4)
        e = load_eim(N, build_if_missing=False)
        eps = np.array([r["eps_Q"] for r in rows]); nA = np.array([r["normA"] for r in rows])
        summ = dict(N=N, Q=e.Q, n_queries=len(rows), population=f"sobol_queries({n_queries}, seed=2026, "
                    "include_T=False) over the box (C, G, M, Y, r)",
                    max_eps_over_normA=float(np.max(eps / nA)), median_eps_over_normA=float(np.median(eps / nA)),
                    bound_1e8_holds=bool(np.all(eps <= 1e-8 * nA)),
                    median_deriv_over_grid=float(np.median([r["eps_deriv"] / r["eps_grid"] for r in rows])),
                    median_t_eps_s=float(np.median([r["t_eps_s"] for r in rows])),
                    median_t_AQ_s=float(np.median([r["t_AQ_s"] for r in rows])), wall_s=time.time() - t0,
                    timing_note="t_eps/t_AQ measured with nproc workers in parallel; not paper timings",
                    provenance=provenance(nproc=nproc))
        dn = [r for r in rows if "dense_err" in r]
        summ.update(n_dense=len(dn), max_dense_over_eps=float(max(r["dense_err"] / r["eps_Q"] for r in dn)),
                    max_dense_over_grid=float(max(r["dense_err"] / r["eps_grid"] for r in dn)),
                    dense_violations=int(sum(r["dense_err"] > r["eps_Q"] for r in dn)))
        fn = [r for r in rows if "fine_sup_err" in r]
        if fn:
            summ.update(fine_grid_factor=4, max_fine_over_eps=float(max(r["fine_sup_err"] / r["eps_Q"] for r in fn)),
                        fine_violations=int(sum(r["fine_sup_err"] > r["eps_Q"] for r in fn)))
        res["validate"][str(N)] = dict(summary=summ, rows=rows)
        print(json.dumps(summ, indent=1), flush=True)
        save(res)


# ---------------------------------------------------------------- follow-up 2, 2d

def do_retol(res, Ns):
    """Q vs tolerance down to 1e-11 from the stored test-error curves (no rebuild: the greedy already stored
    every term down to a training error of 1e-13)."""
    for N in Ns:
        e = load_eim(N, build_if_missing=False)
        qv = {f"{t:g}": (int(np.nonzero(e.test_err <= t)[0][0]) if np.any(e.test_err <= t) else None) for t in TOL_SWEEP}
        res["build"][str(N)]["Q_vs_tol"] = qv
        print(f"N={N}: q_max={e.magic.size}, Q vs tol {qv}", flush=True)
    save(res)


def _qtarget(args):
    qd, eps = args
    from cgmy_cert.params import Query
    e = load_eim(512, build_if_missing=False)
    return e.Q_for_target(Query(**qd), eps)


def do_qtarget(res, nproc, eps_sweep=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2)):
    """Q matched to the target eps (eps_Q ||cbar||_2 sqrt(T/alpha) <= eps/10, default ||cbar|| and alpha of
    EIM.Q_for_target) for the validation30 queries at N = 512."""
    from cgmy_cert.params import load_queries_json
    qs = load_queries_json(ROOT / "tests" / "fixtures" / "queries.json")["validation30"]
    out = {}
    with Pool(nproc) as pool:
        for eps in eps_sweep:
            Qs = pool.map(_qtarget, [(q.as_dict(), eps) for q in qs])
            out[f"{eps:g}"] = dict(median_Q=float(np.median(Qs)), max_Q=int(max(Qs)), min_Q=int(min(Qs)), Q=Qs)
            print(f"eps={eps:g}: Q median {np.median(Qs)} max {max(Qs)}", flush=True)
    res["Q_for_target"] = dict(N=512, population="validation30", rule="smallest stored Q with eps_Q_graded(mu) "
                               "* 1.1||c_raw^0||_2 * sqrt(T / (r min m)) <= eps / 10", per_eps=out,
                               q_max=int(load_eim(512, build_if_missing=False).magic.size), provenance=provenance())
    save(res)


# ---------------------------------------------------------------- figure F4

def do_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = res["build"]
    Ns = sorted(int(k) for k in b)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for c, mk, N in zip(PAL, "osD^v", Ns):
        d = b[str(N)]
        sv = np.array(d["singular_values"]); te = np.array(d["test_err"])
        ax[0].semilogy(np.arange(1, sv.size + 1), sv, color=c, lw=1.2, ls=":")
        ax[0].semilogy(np.arange(te.size), te, color=c, lw=2, marker="o", ms=3, label=f"N = {N}")
        tq = [(float(t), q) for t, q in d["Q_vs_tol"].items() if q is not None]
        ax[1].semilogx([t for t, _ in tq], [q for _, q in tq], color=c, lw=2, marker=mk, ms=7 - Ns.index(N),
                   mfc="none" if N == Ns[-1] else c, label=f"N = {N}")
    ax[0].axhline(1e-9, color="0.5", lw=0.8, ls="--")
    ax[0].set_xlim(0, 45); ax[0].set_ylim(1e-15, 3)
    ax[0].set_xlabel("Q (number of EIM terms) / singular value index")
    ax[0].set_ylabel("relative sup-norm error")
    ax[0].set_title("(a) EIM test error (solid) and normalised\nsingular values of the training set (dotted)", fontsize=9)
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].invert_xaxis()
    ax[1].set_xlabel("tolerance (relative sup-norm, 1000 Sobol test parameters)")
    ax[1].set_ylabel("Q")
    ax[1].set_title("(b) EIM rank needed vs tolerance", fontsize=9)
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    save_fig(fig, FIG / "F4_eim.png")
    print("wrote", FIG / "F4_eim.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["build", "select", "validate", "figure"])
    ap.add_argument("--N", nargs="*", type=int, default=list(N_DEFAULT))
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--n-queries", type=int, default=1000)
    a = ap.parse_args()
    res = load()
    res["meta"] = dict(description="Plan 2.1: EIM of the fractional symbol family Phi_nu, nu = (G, M, Y), "
                       "per N on the L-grid of symbols(); errors are relative sup-norms of the stacked (Re, Im) "
                       "samples; eps_Q, dense_err, fine_sup_err are absolute operator-norm quantities "
                       "(same units as A_I).", training_grid=f"{TRAIN_SHAPE} tensor grid in (log G, log M, Y)",
                       tol_sweep=list(TOL_SWEEP), K=100, provenance=provenance(nproc=a.nproc))
    if "build" in a.steps:
        do_build(a.N, a.nproc, res)
    if "select" in a.steps:
        do_select(a.N, res, a.nproc, a.n_queries)
    if "retol" in a.steps:
        do_retol(res, a.N)
    if "qtarget" in a.steps:
        do_qtarget(res, a.nproc)
    if "validate" in a.steps:
        do_validate(res, a.nproc, a.N, a.n_queries)
    if "figure" in a.steps:
        do_figure(res)
    save(res)


if __name__ == "__main__":
    main()
