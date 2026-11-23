#!/usr/bin/env python3
"""Phase 3, E3 (plan 3.4): out-of-distribution detection at the production configuration.

Queries: 200 Sobol points (seed 7) of the extended box C in [0.05, 2], G in [0.5, 5], M in [2, 15], Y in [0.9, 1.95]
(r, T from the box) that lie outside the box in at least one of (C, G, M, Y), excluding |Y - 1| < 0.02 (Gamma(-Y) is
singular at Y = 1); plus the 64 vertices of the extended box (no vertex has Y = 1, so the 0.98 / 1.02 substitution of
the brief is not triggered).  Rules recorded in the JSON:
  * the reduced (cell) stage applies only to queries inside the box (the cell beta-bar is exact by monotonicity up to the
    box maxima G_max, M_max; a query with M = 15 has min(G, M) in a cell but a smaller beta~); otherwise the three-stage rule
    starts at the banded stage with the query's own spectral_data;
  * eps_Q is evaluated at the query (rigorous anywhere) and enters the reduced-model allowance;
  * G < 1: the L-grid aliasing |t_m| ~ e^{-G h m} is checked by comparing the Toeplitz entries at L and 2L; L is
    doubled for a query if the difference exceeds 1e-10 relative;
  * references by direct GMRES at the query (N = 1024, N_T = 240), true error max_n mass norm.
Candidates: POD-Galerkin r = 64 and 80 (basis 800), space-time POD-NN (3 seeds).  Proxy: seed-ensemble disagreement
max_n ||z_seed - z_mean||_{M_I} > eps_N.  Writes results/E3.json and figures/F7_ood.png.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.figio import save_fig  # noqa: E402
from cgmy_cert.params import Query, BOX  # noqa: E402
from cgmy_cert.symbols import free_matrices  # noqa: E402
from cgmy_cert.eim import load_eim  # noqa: E402
from cgmy_cert.pod import AffineROM  # noqa: E402
from cgmy_cert.reduced_cert import Partition, load_rc  # noqa: E402
from cgmy_cert.direct import free_forcing, initial_free_vector, cn_march, mass_norm, lifting  # noqa: E402
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.certificate import certify  # noqa: E402
from cgmy_cert.three_stage import theta_bar  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

PC = json.loads((ROOT / "results" / "PRODUCTION_CONFIG.json").read_text())
N, NT, EPS_N = PC["N"], PC["N_T"], PC["eps_N"]
EPS = tuple(PC["eps_sweep"])
DATA = ROOT / "data"; RES = ROOT / "results"; FIG = ROOT / "figures"; CK = ROOT / "checkpoints"
OUT = RES / "E3.json"
SPEC = "CYm:3x3x2"
EXT = dict(C=(0.05, 2.0), G=(0.5, 5.0), M=(2.0, 15.0), Y=(0.9, 1.95), r=BOX["r"], T=BOX["T"])
KEYS = ("C", "G", "M", "Y", "r", "T")
ALIAS_TOL = 1e-10


def in_box(q: Query) -> bool:
    return all(BOX[k][0] <= getattr(q, k) <= BOX[k][1] for k in ("C", "G", "M", "Y", "r", "T"))


def ood_queries(n=200, seed=7):
    lo = np.array([EXT[k][0] for k in KEYS]); hi = np.array([EXT[k][1] for k in KEYS])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = qmc.Sobol(d=6, scramble=True, seed=seed)
        out = []
        while len(out) < n:
            for x in lo + S.random(256) * (hi - lo):
                q = Query(**dict(zip(KEYS, map(float, x))), query_id=f"ood_{len(out):04d}")
                if not in_box(q) and abs(q.Y - 1.0) >= 0.02:
                    out.append(q)
                    if len(out) == n:
                        break
    verts = []
    for bits in range(64):
        v = {k: EXT[k][(bits >> i) & 1] for i, k in enumerate(KEYS)}
        if abs(v["Y"] - 1.0) < 0.02:
            v["Y"] = 0.98 if v["Y"] < 1 else 1.02
        verts.append(Query(**v, query_id=f"extvertex_{bits:02d}"))
    return out + verts


def choose_L(q):
    """Default L-grid, doubled until the Toeplitz entries change by <= ALIAS_TOL relative (only needed for G < 1)."""
    fs = free_matrices(q, N)
    L = fs.sym.L
    info = dict(L_default=L, L_used=L, alias_rel_change=None)
    if q.G < 1.0:
        fs2 = free_matrices(q, N, L=2 * L)
        d = float(np.max(np.abs(fs2.A.t.real - fs.A.t.real)) / np.max(np.abs(fs.A.t.real)))
        info["alias_rel_change"] = d
        while d > ALIAS_TOL and info["L_used"] < 16 * L:
            info["L_used"] *= 2
            fs, fs2 = fs2, free_matrices(q, N, L=2 * info["L_used"])
            d = float(np.max(np.abs(fs2.A.t.real - fs.A.t.real)) / np.max(np.abs(fs.A.t.real)))
        info["alias_rel_change_final"] = d
    return fs, info


def _predict(names, X):
    import torch
    torch.set_num_threads(1)
    from cgmy_cert.podnn import PODNN
    return {n: PODNN.from_state(torch.load(CK / f"{n}.pt", weights_only=True)).predict(X)[0] for n in names}


def _one(args):
    qd, nn_names, Q = args
    q = Query(**qd)
    t0 = time.time()
    out = dict(query_id=q.query_id, mu=[q.C, q.G, q.M, q.Y, q.r], T=q.T, in_box=in_box(q))
    try:
        fs, linfo = choose_L(q)
        out["L"] = linfo
        forc = free_forcing(fs, q.T, NT); z0 = initial_free_vector(q, fs.mesh)
        ref = cn_march(fs, z0, q.T, NT, solver="gmres", forcing=forc)
        out["gmres_max_iters"] = int(max(ref.iters))
        sp = spectral_data(q, N)
        out.update(alpha_lb=sp.alpha_lb, beta=sp.beta, Theta=sp.Theta)
        ref_c = certify(fs, forc, ref.Z, z0, 0.0, mode="banded", spec=sp)
        out["reference_bound_banded"] = ref_c.etaX_aug
    except Exception as ex:                        # e.g. beta <= 0 or alpha_lb <= 0 outside the box: uncertifiable
        out["error"] = f"{type(ex).__name__}: {ex}"
        return out
    e = load_eim(N, build_if_missing=False)
    info = e.eps_Q_graded(q, Q)
    out["eps_Q"] = info["eps_Q"]
    out["eps_Q_rel"] = info["eps_Q"] / float(np.linalg.norm(fs.A.dense(), 2)) if N <= 1024 else None
    use_red = out["in_box"]
    cid = Partition(SPEC).cell_of(q) if use_red else None
    cands = {}
    Vb = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_800_r128.npz")["V"]
    for r in (64, 80):
        V = Vb[:, :r]
        rc = load_rc(DATA / f"redcert_N{N}_r{r}_n800_{SPEC.replace(':', '_')}_Q{Q}c_b128.npz")
        cands[f"podg{r}"] = (V, rc, AffineROM(V, e, Q=Q).solve(q, rc.alpha0(), NT, ell=info["ell"]).alpha)
    Vn = np.load(DATA / f"pod_N{N}_NT{NT}_seed0_4000.npz")["V"][:, :48]
    rcn = load_rc(DATA / f"redcert_N{N}_r48_n4000_{SPEC.replace(':', '_')}_Q{Q}c.npz")
    preds = _predict(nn_names, np.array([[q.C, q.G, q.M, q.Y, q.r, q.T]]))
    for nm, a in preds.items():
        cands[nm] = (Vn, rcn, a)
    out["cands"] = {}
    tb = None
    for name, (V, rc, alpha) in cands.items():
        Z = V @ alpha
        err = max(mass_norm(fs, Z[:, k] - ref.Z[:, k]) for k in range(NT + 1))
        cX = certify(fs, forc, Z, z0, 0.0, mode="banded", spec=sp)
        cS = certify(fs, forc, Z, z0, 0.0, mode="exact", spec=sp)
        b = dict(banded=cX.etaX_aug, exact=cS.etaS_aug)
        row = dict(true_err=err, bounds=b)
        if use_red:
            red = rc.certify(q, alpha, forc.dt, forc.bL, info["ell"], info["eps_Q"], cid=cid)
            b["reduced"] = red["eta_aug"]
        # three-stage status per eps (reduced stage only in the box)
        st = {}
        for t in EPS:
            if use_red and b["reduced"] <= t:
                s = "reduced_accept"
            elif use_red and (tb := tb or theta_bar(q, N, rc.cells[cid].rbar, rc.cells[cid].betabar)) and \
                    np.sqrt(red["beta0"] ** 2 + forc.dt * np.sum(np.maximum((np.sqrt(red["per_step"]) - red["rounding"]) / np.sqrt(tb)
                                                                             - red["allowance"], 0) ** 2)) > t:
                s = "reject_cheap"
            elif b["banded"] <= t:
                s = "banded_accept"
            elif np.sqrt(cX.beta0 ** 2 + cX.etaX ** 2 / sp.Theta) > t:
                s = "reject_banded"
            else:
                s = "exact_accept" if b["exact"] <= t else "exact_reject"
            st[f"{t:.4g}"] = s
        row["three_stage"] = st
        row["Z"] = Z
        out["cands"][name] = row
    # ensemble proxy
    Zs = np.stack([out["cands"][n]["Z"] for n in nn_names])
    zm = Zs.mean(axis=0)
    for i, n in enumerate(nn_names):
        out["cands"][n]["ensemble_disagreement"] = float(max(mass_norm(fs, Zs[i][:, k] - zm[:, k]) for k in range(NT + 1)))
    for row in out["cands"].values():
        row.pop("Z")
    out["wall_s"] = time.time() - t0
    return out


def do_run(res, nproc):
    qs = ood_queries()
    e1 = json.loads((RES / "E1.json").read_text())
    st = sorted(k for k, v in e1["eval"]["summary"].items() if v["kind"] == "podnn_spacetime")
    Q = e1["Q_eps"]["per_eps"][f"{EPS_N:.4g}"]["Q_used"]
    t0 = time.time()
    with Pool(nproc) as pool:
        rows = pool.map(_one, [(q.as_dict(), st, Q) for q in qs], chunksize=1)
    res["rows"] = rows
    res["setup"] = dict(nn_seeds=st, Q=Q, n_queries=len(qs), n_sobol=200, n_vertices=64, extended_box=EXT,
                        rules=dict(reduced_stage="only for queries inside the box (cell beta-bar valid up to G_max, M_max)",
                                   eps_Q="evaluated at the query (graded 161 nodes, rigorous anywhere)",
                                   aliasing=f"G < 1: L doubled until Toeplitz entries change by <= {ALIAS_TOL:g} relative",
                                   references="GMRES at the query, N = 1024, N_T = 240"),
                        wall_s=time.time() - t0, provenance=provenance(nproc=nproc))
    save(res)


def do_summary(res):
    rows = [r for r in res["rows"] if "error" not in r]
    bad = [dict(query_id=r["query_id"], mu=r["mu"], error=r["error"]) for r in res["rows"] if "error" in r]
    names = list(rows[0]["cands"])
    nn = res["setup"]["nn_seeds"]
    summ = {}
    for nm in names:
        s = {}
        for cert in ("reduced", "banded", "exact"):
            R_ = [r for r in rows if cert in r["cands"][nm]["bounds"]]
            fa = {f"{t:.4g}": int(sum(r["cands"][nm]["bounds"][cert] <= t and r["cands"][nm]["true_err"] > t for r in R_)) for t in EPS}
            fr = {f"{t:.4g}": int(sum(r["cands"][nm]["bounds"][cert] > t and r["cands"][nm]["true_err"] <= t for r in R_)) for t in EPS}
            acc = {f"{t:.4g}": int(sum(r["cands"][nm]["bounds"][cert] <= t for r in R_)) for t in EPS}
            s[cert] = dict(n=len(R_), accepted=acc, false_accepts=fa, false_rejects=fr)
        st = [r["cands"][nm]["three_stage"][f"{EPS_N:.4g}"] for r in rows]
        s["three_stage_at_eps_N"] = {k: st.count(k) for k in sorted(set(st))}
        s["three_stage_false_accepts"] = {f"{t:.4g}": int(sum(r["cands"][nm]["three_stage"][f"{t:.4g}"].endswith("accept")
                                                               and r["cands"][nm]["true_err"] > t for r in rows)) for t in EPS}
        s["median_true_err_in_box"] = float(np.median([r["cands"][nm]["true_err"] for r in rows if r["in_box"]])) if any(r["in_box"] for r in rows) else None
        s["median_true_err_out_of_box"] = float(np.median([r["cands"][nm]["true_err"] for r in rows if not r["in_box"]]))
        summ[nm] = s
    # proxy: ensemble disagreement > eps_N predicts true error > eps_N (pooled over seeds)
    tp = fp = fn = tn = 0
    for r in rows:
        for n in nn:
            pred = r["cands"][n]["ensemble_disagreement"] > EPS_N; lab = r["cands"][n]["true_err"] > EPS_N
            tp += pred and lab; fp += pred and not lab; fn += (not pred) and lab; tn += (not pred) and (not lab)
    proxy = dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
                 threshold=EPS_N, rule="flag if max_n ||z_seed - z_mean||_M > eps_N; label: true error > eps_N")
    e1 = json.loads((RES / "E1.json").read_text())
    in_eps = [v for r in e1["eval"]["rows"][next(iter(e1["eval"]["rows"]))] for k, v in r["eps_Q"].items() if k == str(res["setup"]["Q"])]
    dist = dict(eps_Q_out=dict(median=float(np.median([r["eps_Q"] for r in rows])), max=float(max(r["eps_Q"] for r in rows))),
                eps_Q_in_validation30=dict(median=float(np.median(in_eps)), max=float(max(in_eps))),
                alpha_lb_out=dict(min=float(min(r["alpha_lb"] for r in rows)), median=float(np.median([r["alpha_lb"] for r in rows]))),
                eps_Q_rel_out=dict(median=float(np.median([r["eps_Q_rel"] for r in rows])), max=float(max(r["eps_Q_rel"] for r in rows))))
    L_raised = [dict(query_id=r["query_id"], G=r["mu"][1], **r["L"]) for r in rows if r["L"]["L_used"] != r["L"]["L_default"]]
    G_small = [r for r in rows if r["mu"][1] < 1]
    alias = dict(n_G_below_1=len(G_small), n_L_raised=len(L_raised), raised=L_raised,
                 max_alias_change_default=float(max((r["L"]["alias_rel_change"] or 0) for r in G_small)) if G_small else None)
    res["summary"] = dict(per_candidate=summ, proxy=proxy, distributions=dist, aliasing=alias, uncertifiable=bad,
                          n_certifiable=len(rows), n_in_box=int(sum(r["in_box"] for r in rows)))
    save(res)
    print(json.dumps({k: (v["banded"]["false_accepts"], v["exact"]["false_accepts"], v["three_stage_at_eps_N"]) for k, v in summ.items()}),
          "\nproxy", proxy, "\ndist", dist, "\nalias", {k: v for k, v in alias.items() if k != "raised"}, "\nuncertifiable", len(bad), flush=True)


def shells_of(r):
    """Coordinates by which a query leaves the box (Sobol queries; the extended-box vertices form their own group)."""
    if r["query_id"].startswith("extvertex"):
        return ["vertices"]
    C, G, M, Y, _ = r["mu"]
    out = []
    if C > BOX["C"][1]: out.append("C>1")
    if Y < BOX["Y"][0]: out.append("Y<1.1")
    if Y > BOX["Y"][1]: out.append("Y>1.8")
    if G < BOX["G"][0]: out.append("G<1")
    if M > BOX["M"][1]: out.append("M>9")
    return out


def do_breakdown(res):
    """(a) per candidate: out-of-box queries with true error <= eps_N and how many of them the exact and the three-stage
    certificates accept; (b) the same per shell, with median true error and median eps_Q; (c) precision / recall of the naive
    rule 'reject everything outside the box' and of each certificate, as detectors of 'true error > eps_N'."""
    rows = [r for r in res["rows"] if "error" not in r]
    names = list(rows[0]["cands"])
    k = f"{EPS_N:.4g}"
    SH = ("C>1", "Y<1.1", "Y>1.8", "G<1", "M>9", "vertices")

    def counts(R, nm):
        good = [r for r in R if r["cands"][nm]["true_err"] <= EPS_N]
        acc_ex = sum(r["cands"][nm]["bounds"]["exact"] <= EPS_N for r in good)
        acc_3 = sum(r["cands"][nm]["three_stage"][k].endswith("accept") for r in good)
        return dict(n_queries=len(R), n_true_err_le_eps_N=len(good), exact_accepts=int(acc_ex), three_stage_accepts=int(acc_3),
                    median_true_err=float(np.median([r["cands"][nm]["true_err"] for r in R])) if R else None,
                    median_eps_Q=float(np.median([r["eps_Q"] for r in R])) if R else None)

    def detector(R, nm, flag):
        """flag(r) True = predicted 'true error > eps_N'."""
        tp = sum(flag(r) and r["cands"][nm]["true_err"] > EPS_N for r in R)
        fp = sum(flag(r) and r["cands"][nm]["true_err"] <= EPS_N for r in R)
        fn = sum((not flag(r)) and r["cands"][nm]["true_err"] > EPS_N for r in R)
        return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=tp / (tp + fp) if tp + fp else None,
                    recall=tp / (tp + fn) if tp + fn else None)

    out = {}
    for nm in names:
        per_shell = {sh: counts([r for r in rows if sh in shells_of(r)], nm) for sh in SH}
        out[nm] = dict(all=counts(rows, nm), per_shell=per_shell,
                       naive_reject_outside_box=detector(rows, nm, lambda r: True),
                       exact_certificate=detector(rows, nm, lambda r, nm=nm: r["cands"][nm]["bounds"]["exact"] > EPS_N),
                       three_stage=detector(rows, nm, lambda r, nm=nm: not r["cands"][nm]["three_stage"][k].endswith("accept")))
        a_ = out[nm]["all"]
        print(f"{nm[-30:]}: works on {a_['n_true_err_le_eps_N']}/264, exact finds {a_['exact_accepts']}, three-stage "
              f"{a_['three_stage_accepts']} | naive precision {out[nm]['naive_reject_outside_box']['precision']} | exact "
              f"precision {out[nm]['exact_certificate']['precision']}", flush=True)
    res["breakdown"] = dict(per_candidate=out, eps=EPS_N, shells=list(SH),
                            note="shells overlap (a Sobol query outside in several coordinates counts in each); vertices are the "
                                 "64 extended-box vertices; detectors predict 'true error > eps_N': the naive rule flags every "
                                 "out-of-box query, the certificates flag a rejection (the reduced stage is not used outside the box)")
    save(res)


def do_table_ood(res):
    """tables/tab_ood_v9.tex (layout and caption as edited in the manuscript; numbers from E3.json summary/breakdown)."""
    bd = res["breakdown"]["per_candidate"]
    st = [k for k in bd if "spacetime" in k]
    e_in = res["summary"]["distributions"]["eps_Q_in_validation30"]["median"]

    def tex_e(x):
        m, e = f"{x:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"

    lines = [r"% Numbers from results/E3.json (summary, breakdown) and results/NOTES_FOR_PAPER.md item 27 (checked against the Phase 3 zip, 2026-10-23).",
             r"\begin{table}[!htbp]", r"\centering\small",
             r"\caption{Out-of-distribution study: 264 queries outside the parameter box (200 Sobol points in the",
             r"shells $C\in[1,2]$, $Y\in[0.9,1.1)\cup(1.8,1.95]$, $G\in[0.5,1)$, $M\in[9,15]$, and 64 vertices of the",
             r"extended box) at the production configuration.  Top: queries on which the candidate's true error",
             r"is below $\varepsilon_N$ (``works''), queries the exact certificate accepts (``certified''), and",
             r"precision of two rules as detectors of ``true error $>\varepsilon_N$'': the certificate (recall 1 by",
             r"theorem) and the naive rule that rejects every query outside the training box (recall 1, but it",
             r"discards every working query).  Bottom: POD--Galerkin $r=80$ by shell (a query leaving the box in",
             r"several coordinates counts in each), with the median true error and the median EIM error",
             r"$\varepsilon_Q$ (inside the box: " + tex_e(e_in) + r").}",
             r"\label{tab:ood}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"candidate & works & certified & precision (certificate) & precision (naive rule)\\", r"\midrule"]
    for key, lab in (("podg64", r"POD--Galerkin $r=64$"), ("podg80", r"POD--Galerkin $r=80$")):
        a = bd[key]
        lines.append(f"{lab} & {a['all']['n_true_err_le_eps_N']} & {a['all']['exact_accepts']} & "
                     f"{a['exact_certificate']['precision']:.2f} & {a['naive_reject_outside_box']['precision']:.2f}\\\\")
    a = bd[st[0]]["all"]
    lines.append(f"space--time POD-NN & {a['n_true_err_le_eps_N']} & {a['exact_accepts']} & -- & --\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip", r"\begin{tabular}{lrrrrr}", r"\toprule",
              r"shell ($r=80$) & queries & works & certified & median error & median $\varepsilon_Q$\\", r"\midrule"]
    lab = {"C>1": r"$C>1$", "Y<1.1": r"$Y<1.1$", "Y>1.8": r"$Y>1.8$", "G<1": r"$G<1$", "M>9": r"$M>9$", "vertices": "vertices"}
    for sh in res["breakdown"]["shells"]:
        v = bd["podg80"]["per_shell"][sh]
        lines.append(f"{lab[sh]} & {v['n_queries']} & {v['n_true_err_le_eps_N']} & {v['exact_accepts']} & "
                     f"{tex_e(v['median_true_err'])} & {tex_e(v['median_eps_Q'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ROOT / "tables" / "tab_ood_v9.tex").write_text("\n".join(lines) + "\n")


def do_fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in res["rows"] if "error" not in r]
    names = list(rows[0]["cands"])
    # in-box comparison points: the same candidates on validation30 (E1)
    e1 = json.loads((RES / "E1.json").read_text())["eval"]
    QN = json.loads((RES / "E1.json").read_text())["Q_eps"]["per_eps"][f"{EPS_N:.4g}"]["Q_used"]
    map1 = {"podg64": "pod_galerkin_r64_n800", "podg80": "pod_galerkin_r80_n800"}
    inbox = []
    for nm in names:
        for x in e1["rows"][map1.get(nm, nm)]:
            b = dict(banded=x["bounds"]["banded"], exact=x["bounds"]["exact"], reduced=x["bounds"][f"reduced_Q{QN}"])
            inbox.append(dict(true_err=x["true_err"], bounds=b))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, cert, title in zip(axs, ("reduced", "banded", "exact"), ("reduced online (in-box queries only)", "banded full-order", "exact full-order")):
        out_pts = [r["cands"][n] for r in rows for n in names if cert in r["cands"][n]["bounds"]]
        for pts, col, lab in ((inbox, "#0072B2", "in box (validation30, E1)"), (out_pts, "#D55E00", "out of box (E3)")):
            x = [p["true_err"] for p in pts if cert in p["bounds"]]
            y = [p["bounds"][cert] for p in pts if cert in p["bounds"]]
            if x:
                ax.loglog(x, y, "o", ms=3, alpha=0.6, color=col, label=f"{lab} ({len(x)})")
        lim = [1e-8, 1e3]
        ax.plot(lim, lim, color="0.3", lw=1, label="bound = error")
        ax.axhline(EPS_N, color="0.5", ls="--", lw=1); ax.axvline(EPS_N, color="0.5", ls="--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("true error (max_n mass norm)"); ax.set_title(title, fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("certified bound"); axs[1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(); save_fig(fig, FIG / "F7_ood.png"); plt.close(fig)
    print("F7 written")


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["run", "summary", "fig"])
    ap.add_argument("--nproc", type=int, default=20)
    a = ap.parse_args()
    res = load()
    res["meta"] = dict(description="E3 (plan 3.4): out-of-distribution detection", N=N, N_T=NT, eps_N=EPS_N, eps_sweep=list(EPS))
    for st in a.steps:
        dict(run=lambda: do_run(res, a.nproc), summary=lambda: do_summary(res), fig=lambda: do_fig(res),
             breakdown=lambda: do_breakdown(res))[st]()


if __name__ == "__main__":
    main()
