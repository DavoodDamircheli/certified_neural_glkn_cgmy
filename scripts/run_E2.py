#!/usr/bin/env python3
"""Phase 3, E2 (plan 3.3): matched benchmark on bench120 at the production configuration, and the refinement (F6b).

Methods (all at N = 1024, N_T = 240):
  1 direct           circulant-preconditioned GMRES CN march, uncertified -- the baseline of every ratio
  2 direct_banded    direct + banded full-order certificate eta^X (spectral data, residual, banded solves)
  3 podg64           POD-Galerkin r = 64 (basis 800) + three-stage rule + correction fallback when rejected
  4 podg80           POD-Galerkin r = 80 (basis 800) + three-stage rule (+ fallback, expected unused)
  5 podnn_st         space-time POD-NN (best seed by validation median exact bound, E1) + three-stage + fallback
Correction fallback: full-order CN march from z_h^0, GMRES warm-started at the candidate's z^{n+1} (same tolerance as
the direct solve), then the banded certificate on the corrected trajectory.
Setups: cold = per query the mesh, z_h^0, assembly, spectral data, factorisations and all symbol work inside the timer;
reusable = only query-independent data cached (mesh, z_h^0 -- the strike is fixed --, POD bases, offline tensors, EIM
columns and the 161-node EIM data, the network).  Offline products (reduced blocks, tensors, network) are cached in both.
Timing protocol of CLAUDE_PHASE3.md (one performance core, idle machine, warm-up, 5 repetitions, shuffled order,
medians of per-query medians, geometric-mean ratios to method 1 with 95% query-bootstrap CIs).
Tolerances eps_N x {0.3, 1, 3} (methods 3-5 depend on eps; 1-2 do not).
Usage: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_E2.py [bench] [refine] [figs]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, Mesh, load_queries_json  # noqa: E402
from cgmy_cert.symbols import free_matrices  # noqa: E402
from cgmy_cert.eim import load_eim  # noqa: E402
from cgmy_cert.pod import AffineROM  # noqa: E402
from cgmy_cert.reduced_cert import Partition, load_rc  # noqa: E402
from cgmy_cert.direct import free_forcing, initial_free_vector, cn_march, mass_norm, lifting  # noqa: E402
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.certificate import certify  # noqa: E402
from cgmy_cert.three_stage import certify_three_stage  # noqa: E402
from cgmy_cert.timing import pin, check_single_thread, hardware, loadavg, protocol, geo_ratio_ci  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

PC = json.loads((ROOT / "results" / "PRODUCTION_CONFIG.json").read_text())
N, NT, EPS_N = PC["N"], PC["N_T"], PC["eps_N"]
EPS = tuple(EPS_N * f for f in (0.3, 1.0, 3.0))
DATA = ROOT / "data"; RES = ROOT / "results"; FIG = ROOT / "figures_v9"; TAB = ROOT / "tables_v9"; CK = ROOT / "checkpoints"
OUT = RES / "E2.json"
SPEC = "CYm:3x3x2"


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1, default=float))


_QE = {}


def Q_used(eps):
    """Q matched to eps (E1, 2d rule); E1.json is read once and cached (the lookup sits inside timed calls)."""
    if "per_eps" not in _QE:
        _QE["per_eps"] = json.loads((RES / "E1.json").read_text())["Q_eps"]["per_eps"]
    e1 = _QE["per_eps"]
    # eps of E2 are a subset of the sweep (0.3, 1, 3) x eps_N
    return min(e1.values(), key=lambda v: abs(v["eps"] - eps))["Q_used"]


class Pipeline:
    """All methods, with the query-independent data loaded once (reusable) and an option to rebuild the
    query-independent cheap pieces per query (cold)."""

    def __init__(self):
        self.e = load_eim(N, build_if_missing=False)
        self.e.graded()                                          # 161-node EIM data (offline)
        self.P = Partition(SPEC)
        self.msh = Mesh(N=N)
        self.z0 = initial_free_vector(Query(0.5, 3.0, 5.0, 1.5, 0.05), self.msh)
        Vb = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_800_r128.npz")["V"]
        self.Qs = sorted({Q_used(t) for t in EPS})
        self.pod = {}
        for r in (64, 80):
            V = Vb[:, :r]
            self.pod[r] = dict(V=V, rom={Q: AffineROM(V, self.e, Q=Q) for Q in self.Qs},
                               rc={Q: load_rc(DATA / f"redcert_N{N}_r{r}_n800_{SPEC.replace(':', '_')}_Q{Q}c_b128.npz") for Q in self.Qs})
        e1 = json.loads((RES / "E1.json").read_text())["eval"]["summary"]
        st = [k for k, v in e1.items() if v["kind"] == "podnn_spacetime"]
        self.nn_name = min(st, key=lambda k: e1[k]["exact"]["median"])
        import torch
        torch.set_num_threads(1)
        from cgmy_cert.podnn import PODNN
        self.nn = PODNN.from_state(torch.load(CK / f"{self.nn_name}.pt", weights_only=True))
        Vn = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_4000.npz")["V"][:, :48]
        self.nnV = Vn
        self.nnrc = {Q: load_rc(DATA / f"redcert_N{N}_r48_n4000_{SPEC.replace(':', '_')}_Q{Q}c.npz") for Q in self.Qs}

    # ---- pieces
    def setup(self, cold):
        if cold:
            msh = Mesh(N=N)
            return msh, initial_free_vector(Query(0.5, 3.0, 5.0, 1.5, 0.05), msh)
        return self.msh, self.z0

    def direct(self, q, cold=False):
        t0 = time.perf_counter()
        msh, z0 = self.setup(cold)
        fs = free_matrices(q, N); forc = free_forcing(fs, q.T, NT)
        tr = cn_march(fs, z0, q.T, NT, solver="gmres", forcing=forc)
        return dict(Z=tr.Z, t_total=time.perf_counter() - t0)

    def direct_banded(self, q, cold=False):
        t0 = time.perf_counter()
        msh, z0 = self.setup(cold)
        fs = free_matrices(q, N); forc = free_forcing(fs, q.T, NT)
        ta = time.perf_counter()
        tr = cn_march(fs, z0, q.T, NT, solver="gmres", forcing=forc)
        t1 = time.perf_counter()
        sp = spectral_data(q, N, mode="fast")                     # online path: fast mode
        ts = time.perf_counter()
        c = certify(fs, forc, tr.Z, z0, EPS_N, mode="banded", spec=sp)
        t2 = time.perf_counter()
        return dict(Z=tr.Z, bound=c.etaX_aug, t_solve=t1 - t0, t_cert=t2 - t1, t_total=t2 - t0,
                    split=dict(setup_assembly_forcing=ta - t0, march=t1 - ta, spectral=ts - t1,
                               residual=c.time_residual, banded_solves=c.time_banded))

    def certified(self, q, method, eps, cold=False):
        """Candidate -> three-stage -> (correction + recertify if rejected)."""
        Q = Q_used(eps)
        t0 = time.perf_counter()
        msh, z0 = self.setup(cold)
        info = self.e.eps_Q_graded(q, Q)
        if method == "podnn_st":
            V, rc = self.nnV, self.nnrc[Q]
            alpha = self.nn.predict(np.array([[q.C, q.G, q.M, q.Y, q.r, q.T]]))[0]
        else:
            p = self.pod[int(method[4:])]
            V, rc = p["V"], p["rc"][Q]
            alpha = p["rom"][Q].solve(q, rc.alpha0(), NT, ell=info["ell"]).alpha
        t_cand = time.perf_counter() - t0
        cid = self.P.cell_of(q)
        # the full-order system is assembled only if the rule leaves the reduced stage
        lazy = {}

        class _FS:                                            # lazy assembly proxy for the three-stage rule
            pass

        t1 = time.perf_counter()
        dt = q.T / NT
        bL = lifting(q, msh, np.arange(NT + 1) * dt)
        red = rc.certify(q, alpha, dt, bL, info["ell"], info["eps_Q"], cid=cid)
        t_red = time.perf_counter() - t1
        split = dict(candidate=t_cand, reduced_stage=t_red, theta_bar=0.0, assembly=0.0, spectral=0.0, residual=0.0,
                     banded_solves=0.0, pcg=0.0, correction=0.0, recertify=0.0)
        if red["eta_aug"] <= eps:
            return dict(status="reduced_accept", corrected=False, Z=V @ alpha, t_candidate=t_cand, t_certificate=t_red,
                        t_correction=0.0, t_recertify=0.0, t_total=time.perf_counter() - t0, split=split)
        ta = time.perf_counter()
        fs = free_matrices(q, N); forc = free_forcing(fs, q.T, NT)
        split["assembly"] = time.perf_counter() - ta
        ts = certify_three_stage(fs, forc, z0, eps, q=q, alpha=alpha, V=V, rc=rc, ell=info["ell"], eps_Q=info["eps_Q"],
                                 cid=cid, reduced=red)
        split["theta_bar"] = ts.times.get("theta_bar", 0.0)
        if ts.full is not None:
            split.update(spectral=ts.full["t_spectral"], residual=ts.full["t_residual"], banded_solves=ts.full["t_banded_solves"],
                         pcg=ts.full.get("t_pcg", 0.0))
        t_cert = time.perf_counter() - t1 - split["assembly"]
        out = dict(status=ts.status, corrected=False, Z=V @ alpha, t_candidate=t_cand, t_certificate=t_cert,
                   t_correction=0.0, t_recertify=0.0, split=split)
        if not ts.accepted:
            t2 = time.perf_counter()
            tr = cn_march(fs, z0, q.T, NT, solver="gmres", forcing=forc, x0_traj=V @ alpha)
            t3 = time.perf_counter()
            sp = ts.full["spec"] if ts.full is not None else spectral_data(q, N, mode="fast")
            c = certify(fs, forc, tr.Z, z0, eps, mode="banded", spec=sp)
            t4 = time.perf_counter()
            split.update(correction=t3 - t2, recertify=t4 - t3)
            out.update(corrected=True, Z=tr.Z, t_correction=t3 - t2, t_recertify=t4 - t3,
                       corrected_bound=c.etaX_aug, corrected_accepted=bool(c.etaX_aug <= eps))
        out["t_total"] = time.perf_counter() - t0
        return out


def bench_queries():
    pops = load_queries_json(ROOT / "tests" / "fixtures" / "queries.json")
    return pops["bench120"], pops["validation30"][0]


def do_bench(res):
    check_single_thread(); pin()
    la0 = loadavg()
    pl = Pipeline()
    qs, warm = bench_queries()
    refs = np.load(DATA / f"references_N{N}_NT{NT}_bench120.npz")
    Zref = {str(i): j for j, i in enumerate(refs["query_ids"])}
    Zall = np.asarray(refs["Z"])                              # read once (NpzFile re-reads the array on every access)
    fsM = free_matrices(qs[0], N)                              # mass matrix (query-independent) for errors
    last = {}

    def wrap(name, f):
        def g(q):
            last[(name, q.query_id)] = f(q)
        return g

    out = {}
    for setup in ("reusable", "cold"):
        cold = setup == "cold"
        meth = {"direct": wrap("direct", lambda q: pl.direct(q, cold)),
                "direct_banded": wrap("direct_banded", lambda q: pl.direct_banded(q, cold))}
        for eps in EPS:
            for m in ("podg64", "podg80", "podnn_st"):
                meth[f"{m}@{eps:.4g}"] = wrap(f"{m}@{eps:.4g}", (lambda mm, ee: (lambda q: pl.certified(q, mm, ee, cold)))(m, eps))
        t0 = time.time()
        times = protocol(meth, qs, warm, reps=5, seed=11 if cold else 7)
        per = {}
        for name in meth:
            rows = []
            for i, q in enumerate(qs):
                r = last[(name, q.query_id)]
                Z = r.pop("Z")
                Zr = Zall[Zref[q.query_id]]
                err = max(mass_norm(fsM, Z[:, k] - Zr[:, k]) for k in range(NT + 1))
                rows.append(dict(query_id=q.query_id, time_s=times[name][i], true_err=err, **{k: v for k, v in r.items()}))
            per[name] = rows
        summ = {}
        tref = np.array(times["direct"])
        for name, rows in per.items():
            t = np.array([r["time_s"] for r in rows])
            s = dict(median_time_ms=float(np.median(t) * 1e3), ratio_to_direct=geo_ratio_ci(t, tref))
            if name == "direct_banded":
                frac = np.array([r["t_cert"] / r["t_total"] for r in rows])
                proper = np.array([(r["split"]["residual"] + r["split"]["banded_solves"]) / r["t_solve"] for r in rows])
                with_sym = np.array([(r["split"]["residual"] + r["split"]["banded_solves"] + r["split"]["spectral"]) / r["t_solve"]
                                     for r in rows])
                s.update(certificate_fraction_median=float(np.median(frac)),
                         certificate_over_solve_median=float(np.median([r["t_cert"] / r["t_solve"] for r in rows])),
                         certificate_proper_over_solve_median=float(np.median(proper)),
                         certificate_incl_symbol_work_over_solve_median=float(np.median(with_sym)),
                         split_median_ms={k: float(np.median([r["split"][k] for r in rows]) * 1e3) for k in rows[0]["split"]})
            if "@" in name:
                eps = float(name.split("@")[1])
                st = [r["status"] for r in rows]
                s.update(eps=eps, status_counts={k: st.count(k) for k in sorted(set(st))},
                         corrected=int(sum(r["corrected"] for r in rows)),
                         false_accepts=int(sum((not r["corrected"]) and r["status"].endswith("accept") and r["true_err"] > eps
                                               for r in rows)),
                         final_max_true_err=float(max(r["true_err"] for r in rows)),
                         time_split_median_ms={k: float(np.median([r[k] for r in rows]) * 1e3)
                                               for k in ("t_candidate", "t_certificate", "t_correction", "t_recertify")},
                         component_split_mean_ms={k: float(np.mean([r["split"][k] for r in rows]) * 1e3) for k in rows[0]["split"]},
                         symbol_work_mean_ms=float(np.mean([r["split"]["assembly"] + r["split"]["spectral"] + r["split"]["theta_bar"]
                                                           for r in rows]) * 1e3))
            summ[name] = s
        out[setup] = dict(summary=summ, rows=per, wall_s=time.time() - t0)
        print(setup, {k: (round(v["median_time_ms"], 1), round(v["ratio_to_direct"]["geo_mean_ratio"], 3),
                          v.get("status_counts"), v.get("false_accepts")) for k, v in summ.items()}, flush=True)
        res["bench"] = dict(per_setup=out, eps=list(EPS), nn_candidate=pl.nn_name, Q_used={f"{t:.4g}": Q_used(t) for t in EPS},
                            population="bench120", warmup="validation30[0]", loadavg_before=la0, loadavg_after=loadavg(),
                            hardware=hardware(), provenance=provenance(),
                            setups="cold: mesh and z_h^0 rebuilt per query; reusable: cached. Query-dependent work (assembly, "
                                   "spectral data, eps_Q, factorisations, solves) inside the timer in both; offline products "
                                   "(reduced blocks, tensors, network, EIM data) cached in both")
        save(res)


def do_refine(res, n_q=20, Ns=(64, 128, 256, 512, 1024, 2048, 4096)):
    check_single_thread(); pin()
    la0 = loadavg()
    qs, warm = bench_queries()
    pop = qs[:n_q]
    out = {}
    for N_ in Ns:
        ctx = {}
        for q in [warm] + pop:
            fs = free_matrices(q, N_); forc = free_forcing(fs, q.T, NT); z0 = initial_free_vector(q, fs.mesh)
            ctx[q.query_id] = dict(fs=fs, forc=forc, z0=z0, Z=cn_march(fs, z0, q.T, NT, solver="gmres", forcing=forc).Z)

        def solve(q):
            fs = free_matrices(q, N_); forc = free_forcing(fs, q.T, NT)
            cn_march(fs, ctx[q.query_id]["z0"], q.T, NT, solver="gmres", forcing=forc)

        def banded(q):
            x = ctx[q.query_id]
            certify(x["fs"], x["forc"], x["Z"], x["z0"], EPS_N, mode="banded", spec=spectral_data(q, N_, mode="fast"))

        t = protocol(dict(direct_solve=solve, banded_certificate=banded), pop, warm, reps=5, seed=N_)
        ds, bc = np.array(t["direct_solve"]), np.array(t["banded_certificate"])
        out[str(N_)] = dict(direct_solve_ms=float(np.median(ds) * 1e3), banded_certificate_ms=float(np.median(bc) * 1e3),
                            ratio_cert_over_solve=float(np.median(bc / ds)))
        print(N_, out[str(N_)], flush=True)
    e1 = json.loads((RES / "E1.json").read_text())
    red = e1["timing"]["per_rank"]
    res["refine"] = dict(per_N=out, n_queries=n_q, population="bench120[:20]", N_T=NT,
                         reduced_certificate_ms_N1024={r: red[r]["reduced_certificate"] for r in ("64", "80")},
                         note="direct solve includes assembly; banded certificate includes spectral data (fast mode), residual "
                              "and banded solves; the reduced online certificate (E1 timing, N = 1024) does not depend on N",
                         loadavg_before=la0, loadavg_after=loadavg(), hardware=hardware(), provenance=provenance())
    save(res)


def do_figs(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    b = res["bench"]["per_setup"]["reusable"]["summary"]
    labels = {"direct_banded": "direct + banded cert.", "podg64": "POD-G r=64 (3-stage)", "podg80": "POD-G r=80 (3-stage)",
              "podnn_st": "space-time POD-NN (3-stage)"}
    fig, axs = plt.subplots(1, len(EPS), figsize=(4.0 * len(EPS), 3.2), sharey=True)
    for ax, eps in zip(axs, EPS):
        names = ["direct_banded"] + [f"{m}@{eps:.4g}" for m in ("podg64", "podg80", "podnn_st")]
        y = np.arange(len(names))
        for j, (nm, c) in enumerate(zip(names, PAL)):
            rr = b[nm]["ratio_to_direct"]
            ax.errorbar(rr["geo_mean_ratio"], j, xerr=[[rr["geo_mean_ratio"] - rr["ci95"][0]], [rr["ci95"][1] - rr["geo_mean_ratio"]]],
                        fmt="o", color=c, ms=6, capsize=3)
            ax.text(rr["geo_mean_ratio"] * 1.15, j + 0.18, f"{rr['geo_mean_ratio']:.3g}", fontsize=7)
        ax.axvline(1.0, color="0.4", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([labels[n.split("@")[0]] for n in names], fontsize=8)
        ax.set_title(f"$\\varepsilon$ = {eps / EPS_N:g} $\\varepsilon_N$", fontsize=9)
        ax.set_xlabel("time / uncertified direct solve (geo-mean, 95% CI)", fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "F6a_benchmark.png", dpi=200); plt.close(fig)
    if "refine" in res:
        r = res["refine"]; Ns = [int(k) for k in r["per_N"]]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        ax[0].loglog(Ns, [r["per_N"][str(n)]["direct_solve_ms"] for n in Ns], color=PAL[0], marker="o", lw=2, label="direct solve")
        ax[0].loglog(Ns, [r["per_N"][str(n)]["banded_certificate_ms"] for n in Ns], color=PAL[1], marker="s", lw=2,
                     label="banded full-order certificate")
        for c, (rk, v) in zip((PAL[2], PAL[3]), r["reduced_certificate_ms_N1024"].items()):
            ax[0].axhline(v, color=c, ls="--", lw=1.2, label=f"reduced online certificate, r = {rk} (independent of N)")
        ax[0].set_xlabel("N"); ax[0].set_ylabel("time per query (ms)"); ax[0].legend(fontsize=7, frameon=False)
        ax[1].semilogx(Ns, [r["per_N"][str(n)]["ratio_cert_over_solve"] for n in Ns], color=PAL[1], marker="s", lw=2)
        ax[1].set_xlabel("N"); ax[1].set_ylabel("banded certificate / direct solve")
        for a in ax:
            a.grid(alpha=0.25, lw=0.5, which="both")
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(FIG / "F6b_refinement.png", dpi=200); plt.close(fig)
    # table
    lines = [r"\begin{table}[!htbp]", r"\centering\small",
             r"\caption{Matched benchmark on bench120 at $N=1024$, $N_T=240$ (one thread, idle machine, reusable setup): "
             r"geometric-mean time ratio to the uncertified direct solve (95\% query-bootstrap CI), three-stage outcome "
             r"(reduced accept / reject cheap / banded or exact accept / corrected), false accepts.}", r"\label{tab:benchmark}",
             r"\begin{tabular}{llrcr}", r"\toprule", r"$\varepsilon$ & method & time ratio (CI) & outcome & f.a.\\", r"\midrule"]
    s = b["direct_banded"]
    lines.append(rf"-- & direct + banded & {s['ratio_to_direct']['geo_mean_ratio']:.2f} ({s['ratio_to_direct']['ci95'][0]:.2f}--{s['ratio_to_direct']['ci95'][1]:.2f}) & cert.\ fraction {s['certificate_fraction_median']:.2f} & --\\")
    for eps in EPS:
        for m in ("podg64", "podg80", "podnn_st"):
            s = b[f"{m}@{eps:.4g}"]; sc = s["status_counts"]
            mid = sc.get("banded_accept", 0) + sc.get("exact_accept", 0)
            lines.append(rf"${eps / EPS_N:g}\varepsilon_N$ & {labels[m]} & {s['ratio_to_direct']['geo_mean_ratio']:.3g} "
                         rf"({s['ratio_to_direct']['ci95'][0]:.3g}--{s['ratio_to_direct']['ci95'][1]:.3g}) & "
                         rf"{sc.get('reduced_accept', 0)}/{sc.get('reject_cheap', 0)}/{mid}/{s['corrected']} & {s['false_accepts']}\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_benchmark_v9.tex").write_text("\n".join(lines) + "\n")
    print("E2 figures/table written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["bench", "refine", "figs"])
    a = ap.parse_args()
    res = load()
    res["meta"] = dict(description="E2 (plan 3.3): matched benchmark on bench120 and refinement", N=N, N_T=NT, eps=list(EPS))
    for st in a.steps:
        dict(bench=lambda: do_bench(res), refine=lambda: do_refine(res), figs=lambda: do_figs(res))[st]()


if __name__ == "__main__":
    main()
