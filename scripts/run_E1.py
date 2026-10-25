#!/usr/bin/env python3
"""Phase 3, E1 (plan 3.2): the reduced model and the candidates on validation30 at the production configuration.

Steps
  qeps     operational Q per eps of the sweep (2d rule, max over validation30); Q_used(eps) = max(Q_2d(eps), Q(eps_N))
           (the 2d rule is an inequality, any larger Q satisfies it; this keeps two tensor sets, Q = 36 and 37)
  tensors  Q = 37 compressed tensors for every rank and both bases (Q = 36 exist from follow-up 5)
  eval     every candidate x query: beta_0, reduced eta (at Q_used(eps)), eta^X, eta^S (A_I), true error vs the stored
           reference and the reference's own certified bound, effectivities, three-stage status + stopping stage at
           every eps.  Candidates: POD-Galerkin r in RANKS (bases 800, 4000), all standard POD-NN checkpoints
           (lambda_R x seeds) and the space-time POD-NN (3 seeds)
  select   POD-NN bound-selected and error-selected checkpoint per setting (size, representation)
  timing   F5c data: online solve, reduced certificate and three-stage total at eps_N per rank, and the uncertified
           direct solve at N = 1024 (timing protocol, one performance core, idle machine)
  figs     F5a (update), F5b, F5c; tables_v9/tab_pod_ranks_v9.tex
Writes results/E1.json.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.figio import save_fig  # noqa: E402
from cgmy_cert.params import Query, Mesh  # noqa: E402
from cgmy_cert.symbols import free_matrices  # noqa: E402
from cgmy_cert.eim import load_eim  # noqa: E402
from cgmy_cert.pod import AffineROM  # noqa: E402
from cgmy_cert.reduced_cert import ReducedCertificate, Partition, load_rc, save_rc  # noqa: E402
from cgmy_cert.direct import free_forcing, initial_free_vector, mass_norm, cn_march  # noqa: E402
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.certificate import certify  # noqa: E402
from cgmy_cert.three_stage import theta_bar, full_order_values, certify_three_stage, STATUSES  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

PC = json.loads((ROOT / "results" / "PRODUCTION_CONFIG.json").read_text())
N, NT, EPS_N = PC["N"], PC["N_T"], PC["eps_N"]
EPS = tuple(PC["eps_sweep"])
DATA = ROOT / "data"; RES = ROOT / "results"; FIG = ROOT / "figures_v9"; TAB = ROOT / "tables_v9"; CK = ROOT / "checkpoints"
OUT = RES / "E1.json"
SPEC = "CYm:3x3x2"
RANKS = (8, 16, 24, 32, 40, 48, 56, 62, 64, 80, 96, 128)
STAGE_OF = dict(reduced_accept=1, reject_cheap=1, banded_accept=2, reject_banded=2, exact_accept=3, exact_reject=3)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1, default=float))


def n1024():
    return json.loads((RES / "N1024.json").read_text())


def cell_params():
    return {k: (v["rbar"], v["betabar"]) for k, v in n1024()["cells"]["betabar"].items()}


def val_queries():
    d = np.load(DATA / f"references_N{N}_NT{NT}_validation30.npz")
    return [Query(C=m[0], G=m[1], M=m[2], Y=m[3], r=m[4], T=float(t), query_id=str(i))
            for m, t, i in zip(d["mu"], d["T"], d["query_ids"])]


def pod_file(n):
    return DATA / f"pod_N{N}_NT{NT}_seed0_{n}_r128.npz"


def rc_file(n, r, Q):
    return DATA / f"redcert_N{N}_r{r}_n{n}_{SPEC.replace(':', '_')}_Q{Q}c_b128.npz"


def rc_file_r48(n, Q):
    """POD-NN tensors: the r_max = 62 bases the networks were trained on (podnn prep), r = 48."""
    return DATA / f"redcert_N{N}_r48_n{n}_{SPEC.replace(':', '_')}_Q{Q}c.npz"


# ---------------------------------------------------------------- qeps / tensors

def _qt(args):
    qd, eps = args
    return load_eim(N, build_if_missing=False).Q_for_target(Query(**qd), eps)


def do_qeps(res, nproc):
    qs = val_queries()
    out = {}
    with Pool(nproc) as pool:
        for eps in EPS:
            Qs = pool.map(_qt, [(q.as_dict(), eps) for q in qs])
            out[f"{eps:.4g}"] = dict(eps=eps, Q_2d_max=int(max(Qs)), Q_2d_median=float(np.median(Qs)))
    QN = out[f"{EPS_N:.4g}"]["Q_2d_max"]
    for v in out.values():
        v["Q_used"] = max(v["Q_2d_max"], QN)
    res["Q_eps"] = dict(per_eps=out, rule="Q_used(eps) = max(Q_2d(eps), Q_2d(eps_N)); Q_2d = max over validation30 of "
                        "EIM.Q_for_target(q, eps); any Q >= Q_2d satisfies the 2d inequality", provenance=provenance())
    print(json.dumps(out), flush=True)
    save(res)


def do_tensors(res):
    e = load_eim(N, build_if_missing=False)
    cp = cell_params()
    Qs = sorted({v["Q_used"] for v in res["Q_eps"]["per_eps"].values()})
    built = {}
    for Q in Qs:
        for n in (800, 4000):
            Vall = np.load(pod_file(n))["V"]
            for r in RANKS:
                p = rc_file(n, r, Q)
                if not p.exists():
                    rc = ReducedCertificate.build(Vall[:, :r], e, cp, Q=Q, partition=SPEC, svd_tol=1e-12)
                    save_rc(rc, p)
                built[p.name] = p.stat().st_size / 2 ** 20
            p = rc_file_r48(n, Q)
            if not p.exists():
                V = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_{n}.npz")["V"][:, :48]
                save_rc(ReducedCertificate.build(V, e, cp, Q=Q, partition=SPEC, svd_tol=1e-12), p)
            built[p.name] = p.stat().st_size / 2 ** 20
    res["tensors"] = dict(files_MB=built, Q_values=Qs)
    print(f"tensors: {len(built)} files, {sum(built.values()) / 1024:.1f} GB", flush=True)
    save(res)


# ---------------------------------------------------------------- eval

def candidates():
    out = [dict(kind="pod_galerkin", name=f"pod_galerkin_r{r}_n{n}", n=n, r=r) for n in (800, 4000) for r in RANKS]
    log = json.loads((RES / "training_log.json").read_text())["runs"]
    for run in log:
        if run.get("N") != N or not (CK / f"{run['name']}.pt").exists():
            continue
        out.append(dict(kind="podnn_spacetime" if run["name"].endswith("spacetime") else "podnn", name=run["name"],
                        n=run["n_train"], r=48, lambda_R=run["lambda_R"], seed=run["seed"]))
    return out


def _predict_podnn(name, X):
    import torch
    torch.set_num_threads(1)
    from cgmy_cert.podnn import PODNN
    return PODNN.from_state(torch.load(CK / f"{name}.pt", weights_only=True)).predict(X)


_QCACHE = {}


def _query_ctx(i, Qs):
    """Per-query data shared by all candidates evaluated in this worker."""
    if i not in _QCACHE:
        q = val_queries()[i]
        d = np.load(DATA / f"references_N{N}_NT{NT}_validation30.npz")
        e = load_eim(N, build_if_missing=False)
        fs = free_matrices(q, N); forc = free_forcing(fs, q.T, NT)
        sp = spectral_data(q, N)
        z0 = initial_free_vector(q, fs.mesh)
        Zref = np.asarray(d["Z"][i])
        ref = certify(fs, forc, Zref, z0, 0.0, mode="banded", spec=sp)
        cid = Partition(SPEC).cell_of(q)
        cp = cell_params()[cid]
        _QCACHE[i] = dict(q=q, fs=fs, forc=forc, sp=sp, z0=z0, Zref=Zref, ref_bound=ref.etaX_aug, cid=cid,
                          infos={Q: e.eps_Q_graded(q, Q) for Q in Qs}, tbar=theta_bar(q, N, *cp))
    return _QCACHE[i]


def _eval_cand(args):
    c, Qmap = args
    Qs = sorted(set(Qmap.values()))
    e = load_eim(N, build_if_missing=False)
    if c["kind"] == "pod_galerkin":
        V = np.load(pod_file(c["n"]))["V"][:, :c["r"]]
        rcs = {Q: load_rc(rc_file(c["n"], c["r"], Q)) for Q in Qs}
        rom = AffineROM(V, e, Q=max(Qs))
    else:
        V = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_{c['n']}.npz")["V"][:, :48]
        rcs = {Q: load_rc(rc_file_r48(c["n"], Q)) for Q in Qs}
        Xall = np.array([[q.C, q.G, q.M, q.Y, q.r, q.T] for q in val_queries()])
        A_nn = _predict_podnn(c["name"], Xall)
    rows = []
    for i in range(30):
        x = _query_ctx(i, Qs)
        q, fs, forc, z0, cid, infos = x["q"], x["fs"], x["forc"], x["z0"], x["cid"], x["infos"]
        alpha = (rom.solve(q, rcs[max(Qs)].alpha0(), NT, ell=infos[max(Qs)]["ell"]).alpha
                 if c["kind"] == "pod_galerkin" else A_nn[i])
        Z = V @ alpha
        err = max(mass_norm(fs, Z[:, k] - x["Zref"][:, k]) for k in range(NT + 1))
        reds = {Q: rcs[Q].certify(q, alpha, forc.dt, forc.bL, infos[Q]["ell"], infos[Q]["eps_Q"], cid=cid) for Q in Qs}
        full = full_order_values(fs, forc, z0, Z, q, x["sp"])
        status = {}
        for t in EPS:
            Q = Qmap[f"{t:.4g}"]
            st = certify_three_stage(fs, forc, z0, t, q=q, alpha=alpha, V=V, rc=rcs[Q], ell=infos[Q]["ell"],
                                     eps_Q=infos[Q]["eps_Q"], cid=cid, reduced=reds[Q], tbar=x["tbar"], full=full)
            status[f"{t:.4g}"] = dict(status=st.status, stage=STAGE_OF[st.status], lower_from_reduced=st.lower_from_reduced)
        b = {f"reduced_Q{Q}": reds[Q]["eta_aug"] for Q in Qs}
        b.update(banded=full["etaX"], exact=full["etaS"])
        rows.append(dict(query_id=q.query_id, cell=cid, reference_bound_banded=x["ref_bound"],
                         eps_Q={str(Q): v["eps_Q"] for Q, v in infos.items()}, beta0=full["beta0"], true_err=err,
                         bounds=b, effectivity={k: v / err for k, v in b.items()}, theta_bar=x["tbar"],
                         Theta=full["Theta"], three_stage=status))
    return c["name"], rows


def do_eval(res, nproc):
    Qmap = {k: v["Q_used"] for k, v in res["Q_eps"]["per_eps"].items()}
    cands = candidates()
    t0 = time.time()
    with Pool(nproc) as pool:
        per_cand = dict(pool.map(_eval_cand, [(c, Qmap) for c in cands], chunksize=1))
    summ = {}
    for c in cands:
        R_ = per_cand[c["name"]]
        s = dict(c, median_true_err=float(np.median([x["true_err"] for x in R_])),
                 max_true_err=float(max(x["true_err"] for x in R_)),
                 median_beta0=float(np.median([x["beta0"] for x in R_])))
        for key in ("banded", "exact"):
            v = np.array([x["bounds"][key] for x in R_]); ef = np.array([x["effectivity"][key] for x in R_])
            s[key] = dict(median=float(np.median(v)), max=float(v.max()), median_effectivity=float(np.median(ef)),
                          accepted={f"{t:.4g}": int((v <= t).sum()) for t in EPS},
                          false_accepts={f"{t:.4g}": int(sum(x["bounds"][key] <= t and x["true_err"] > t for x in R_)) for t in EPS})
        red_acc, red_fa, red_med, red_eff = {}, {}, {}, {}
        for t in EPS:
            Q = Qmap[f"{t:.4g}"]; k = f"reduced_Q{Q}"
            v = np.array([x["bounds"][k] for x in R_])
            red_acc[f"{t:.4g}"] = int((v <= t).sum())
            red_fa[f"{t:.4g}"] = int(sum(x["bounds"][k] <= t and x["true_err"] > t for x in R_))
            red_med[f"{t:.4g}"] = float(np.median(v))
            red_eff[f"{t:.4g}"] = float(np.median([x["effectivity"][k] for x in R_]))
        s["reduced"] = dict(median_at_eps=red_med, accepted=red_acc, false_accepts=red_fa, median_effectivity_at_eps=red_eff,
                            median=red_med[f"{EPS_N:.4g}"], median_effectivity=red_eff[f"{EPS_N:.4g}"])
        s["three_stage"] = {f"{t:.4g}": {st: int(sum(x["three_stage"][f"{t:.4g}"]["status"] == st for x in R_)) for st in STATUSES}
                            for t in EPS}
        s["lower_le_exact_all"] = bool(all((x["three_stage"][f"{EPS_N:.4g}"]["lower_from_reduced"] or 0) <= x["bounds"]["exact"] * (1 + 1e-12)
                                           for x in R_))
        summ[c["name"]] = s
        print(f"{c['name']}: err {s['median_true_err']:.2e} exact {s['exact']['median']:.2e} acc@eps_N "
              f"red/band/ex {s['reduced']['accepted'][f'{EPS_N:.4g}']}/{s['banded']['accepted'][f'{EPS_N:.4g}']}/"
              f"{s['exact']['accepted'][f'{EPS_N:.4g}']}", flush=True)
    res["eval"] = dict(summary=summ, rows=per_cand, Q_map=Qmap, eps_sweep=list(EPS), wall_s=time.time() - t0,
                       reference_bound_note="reference_bound_banded = banded full-order bound of the stored GMRES reference "
                                            "itself (its own residual-derived error allowance)",
                       max_reference_bound=float(max(r["reference_bound_banded"] for v in per_cand.values() for r in v)),
                       provenance=provenance(nproc=nproc))
    save(res)


def do_select(res):
    summ = res["eval"]["summary"]
    out = {}
    for kind in ("podnn", "podnn_spacetime"):
        for n in (800, 4000):
            runs = [s for s in summ.values() if s["kind"] == kind and s["n"] == n]
            if not runs:
                continue
            by_lam = {}
            for s in runs:
                by_lam.setdefault(s["lambda_R"], []).append(s)
            agg = {lam: dict(median_exact=float(np.median([x["exact"]["median"] for x in v])),
                             median_true_err=float(np.median([x["median_true_err"] for x in v])),
                             seeds=sorted(x["seed"] for x in v), names=[x["name"] for x in v]) for lam, v in by_lam.items()}
            lb = min(agg, key=lambda l: agg[l]["median_exact"]); le = min(agg, key=lambda l: agg[l]["median_true_err"])
            out[f"{kind}|n{n}"] = dict(bound_selected=dict(lambda_R=lb, **agg[lb]), error_selected=dict(lambda_R=le, **agg[le]),
                                       per_lambda=agg)
    res["selection"] = dict(per_setting=out, note="selection across lambda_R within a setting (data size, representation); "
                            "within-run checkpoints were saved by the validation bound only (no per-epoch weights), so an "
                            "error-selected epoch inside a run is not available without retraining")
    print(json.dumps({k: (v["bound_selected"]["lambda_R"], v["error_selected"]["lambda_R"]) for k, v in out.items()}), flush=True)
    save(res)


# ---------------------------------------------------------------- timing (F5c)

def do_timing(res, n_q=10, n=800):
    """Idle machine, one performance core: per rank, the reduced solve, the reduced certificate (ell + eps_Q + form) and the
    three-stage total at eps_N (the lazy path the rule actually takes, nothing reused), on validation30[1:1+n_q]
    (warm-up: validation30[0]); and the uncertified direct solve at N = 1024 (cold: assembly + forcing + z0 + GMRES march;
    warm: GMRES march only)."""
    from cgmy_cert.timing import pin, check_single_thread, hardware, loadavg, protocol
    check_single_thread(); pin()
    la0 = loadavg()
    e = load_eim(N, build_if_missing=False)
    Q = res["Q_eps"]["per_eps"][f"{EPS_N:.4g}"]["Q_used"]
    qs = val_queries(); warm, pop = qs[0], qs[1:1 + n_q]
    P = Partition(SPEC)
    Vall = np.load(pod_file(n))["V"]
    ctxq = {}
    for q in [warm] + pop:
        fs = free_matrices(q, N)
        ctxq[q.query_id] = dict(fs=fs, forc=free_forcing(fs, q.T, NT), z0=initial_free_vector(q, fs.mesh), cid=P.cell_of(q))

    def direct_cold(q):
        fs = free_matrices(q, N); forc = free_forcing(fs, q.T, NT)
        cn_march(fs, initial_free_vector(q, fs.mesh), q.T, NT, solver="gmres", forcing=forc)

    def direct_warm(q):
        x = ctxq[q.query_id]
        cn_march(x["fs"], x["z0"], q.T, NT, solver="gmres", forcing=x["forc"])

    td = protocol(dict(direct_cold=direct_cold, direct_warm=direct_warm), pop, warm, reps=5, seed=1)
    per = {}
    for r in RANKS:
        V = Vall[:, :r]
        rc = load_rc(rc_file(n, r, Q)); rom = AffineROM(V, e, Q=Q)
        sols = {}
        for q in [warm] + pop:
            info = e.eps_Q_graded(q, Q)
            sols[q.query_id] = (info, rom.solve(q, rc.alpha0(), NT, ell=info["ell"]))

        def f_solve(q):
            rom.solve(q, rc.alpha0(), NT, ell=sols[q.query_id][0]["ell"])

        def f_cert(q):
            info = e.eps_Q_graded(q, Q, ell=e.ell((q.G, q.M, q.Y), Q))
            sol = sols[q.query_id][1]
            rc.certify(q, sol.alpha, sol.dt, sol.bL, info["ell"], info["eps_Q"], cid=ctxq[q.query_id]["cid"])

        def f_three(q):
            x = ctxq[q.query_id]
            info = e.eps_Q_graded(q, Q, ell=e.ell((q.G, q.M, q.Y), Q))
            sol = sols[q.query_id][1]
            certify_three_stage(x["fs"], x["forc"], x["z0"], EPS_N, q=q, alpha=sol.alpha, V=V, rc=rc, ell=info["ell"],
                                eps_Q=info["eps_Q"], cid=x["cid"])

        t = protocol(dict(reduced_solve=f_solve, reduced_certificate=f_cert, three_stage_total=f_three), pop, warm, reps=5, seed=r)
        statuses = []
        for q in pop:
            x = ctxq[q.query_id]; info, sol = sols[q.query_id]
            statuses.append(certify_three_stage(x["fs"], x["forc"], x["z0"], EPS_N, q=q, alpha=sol.alpha, V=V, rc=rc,
                                                ell=info["ell"], eps_Q=info["eps_Q"], cid=x["cid"]).status)
        per[str(r)] = dict({k: float(np.median(v)) * 1e3 for k, v in t.items()},
                           rho=[int(min(c.R.shape[1] for c in rc.cells.values())), int(max(c.R.shape[1] for c in rc.cells.values()))],
                           statuses={s: statuses.count(s) for s in set(statuses)})
        print(f"r={r}: {per[str(r)]}", flush=True)
    res["timing"] = dict(per_rank=per, direct_ms={k: float(np.median(v)) * 1e3 for k, v in td.items()}, basis=n, Q=Q,
                         population=f"validation30[1:{1 + n_q}]", warmup="validation30[0]", unit="ms (median of per-query medians)",
                         loadavg_before=la0, loadavg_after=loadavg(), hardware=hardware(), provenance=provenance())
    print("direct", res["timing"]["direct_ms"], flush=True)
    save(res)


# ---------------------------------------------------------------- figures / table

PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def do_figs(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summ, rows = res["eval"]["summary"], res["eval"]["rows"]
    kN = f"{EPS_N:.4g}"
    # ---- F5b: effectivity histograms per certificate
    cands = [(f"pod_galerkin_r{r}_n800", f"POD-Galerkin r = {r}") for r in (56, 64, 80)]
    st = [k for k, v in summ.items() if v["kind"] == "podnn_spacetime"]
    if st:
        cands.append((min(st, key=lambda k: summ[k]["exact"]["median"]), "space-time POD-NN"))
    QN = res["Q_eps"]["per_eps"][kN]["Q_used"]
    certs = ((f"reduced_Q{QN}", "reduced online"), ("banded", "banded full-order"), ("exact", "exact full-order"))
    fig, ax = plt.subplots(1, len(cands), figsize=(3.2 * len(cands), 3.0), sharey=True)
    bins = np.logspace(0, 3, 31)
    for a, (name, title) in zip(np.atleast_1d(ax), cands):
        for c, (k, lab) in zip(PAL, certs):
            ef = np.array([x["effectivity"][k] for x in rows[name]])
            a.hist(ef, bins=bins, histtype="step", lw=2, color=c, label=f"{lab} (med {np.median(ef):.1f})")
        a.set_xscale("log"); a.set_title(title, fontsize=9); a.set_xlabel("effectivity bound / true error", fontsize=8)
        a.legend(fontsize=6.5, frameon=False, loc="upper right")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    np.atleast_1d(ax)[0].set_ylabel("queries (validation30)", fontsize=8)
    fig.tight_layout(); save_fig(fig, FIG / "F5b_effectivity.png"); plt.close(fig)
    # ---- F5c: online time vs r
    if "timing" in res:
        t = res["timing"]; rs = [int(r) for r in t["per_rank"]]
        fig, a = plt.subplots(figsize=(5.4, 3.6))
        for c, mk, (k, lab) in zip(PAL, "osD", (("reduced_solve", "reduced solve"), ("reduced_certificate", "reduced certificate"),
                                                 ("three_stage_total", f"three-stage total at $\\varepsilon_N$"))):
            a.loglog(rs, [t["per_rank"][str(r)][k] for r in rs], color=c, marker=mk, lw=2, label=lab)
        a.axhline(t["direct_ms"]["direct_cold"], color="0.3", ls="--", lw=1.2,
                  label=f"direct solve, uncertified ({t['direct_ms']['direct_cold']:.0f} ms)")
        a.set_xlabel("POD rank r"); a.set_ylabel("online time per query (ms)")
        a.set_title(f"Online cost at N = {N} (one thread, idle machine)", fontsize=9)
        a.legend(fontsize=7, frameon=False); a.grid(alpha=0.25, lw=0.5, which="both")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        fig.tight_layout(); save_fig(fig, FIG / "F5c_online_time.png"); plt.close(fig)
    # ---- table of ranks
    lines = [r"\begin{table}[!htbp]", r"\centering\small",
             r"\caption{POD-Galerkin on validation30 at $N=1024$ (basis 800): median true error, accepted queries (of 30) at "
             r"$\varepsilon_N$ under the reduced / banded / exact certificates, three-stage outcome (reduced accept / reject at "
             r"the reduced stage / banded / exact), and median effectivities.}", r"\label{tab:pod_ranks}",
             r"\begin{tabular}{rrcccrrr}", r"\toprule",
             r"$r$ & true error & accepted (red/band/ex) & three-stage & & $\mathrm{eff}_{\mathrm{red}}$ & "
             r"$\mathrm{eff}_{\mathrm{band}}$ & $\mathrm{eff}_{\mathrm{ex}}$\\", r"\midrule"]
    for r in RANKS:
        s = summ[f"pod_galerkin_r{r}_n800"]; ts = s["three_stage"][kN]
        lines.append(f"{r} & ${s['median_true_err']:.1e}$ & {s['reduced']['accepted'][kN]}/{s['banded']['accepted'][kN]}/"
                     f"{s['exact']['accepted'][kN]} & {ts['reduced_accept']}/{ts['reject_cheap']}/"
                     f"{ts['banded_accept'] + ts['reject_banded']}/{ts['exact_accept'] + ts['exact_reject']} & & "
                     f"{s['reduced']['median_effectivity']:.1f} & {s['banded']['median_effectivity']:.1f} & "
                     f"{s['exact']['median_effectivity']:.1f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    # tab_pod_ranks_v9.tex is written by scripts/make_tables_phase2.py (the manuscript's layout); this E1 variant is kept
    # under its own name
    (TAB / "tab_pod_ranks_E1_v9.tex").write_text("\n".join(lines) + "\n")
    print("figures and table written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["qeps", "tensors", "eval", "select"])
    ap.add_argument("--nproc", type=int, default=20)
    a = ap.parse_args()
    res = load()
    res["meta"] = dict(description="E1 (plan 3.2): reduced model and candidates on validation30 at the production configuration",
                       N=N, N_T=NT, eps_N=EPS_N, eps_sweep=list(EPS), cells=SPEC,
                       true_error="max_n ||z^n - z_ref^n||_{M_I} vs the stored GMRES references")
    for st in a.steps:
        dict(qeps=lambda: do_qeps(res, a.nproc), tensors=lambda: do_tensors(res), eval=lambda: do_eval(res, a.nproc),
             select=lambda: do_select(res), timing=lambda: do_timing(res), figs=lambda: do_figs(res))[st]()


if __name__ == "__main__":
    main()
