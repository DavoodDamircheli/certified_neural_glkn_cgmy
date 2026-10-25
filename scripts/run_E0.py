#!/usr/bin/env python3
"""Phase 1.8: verification experiments E0.1-E0.5.  Writes results/E0.json, tables_v9/*.tex and
figures_v9/*.png.  Usage: python3 scripts/run_E0.py [E0.1 E0.2 E0.3 E0.4 E0.5] [--nproc 2]"""
from __future__ import annotations

import json, zlib, sys, time, argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy.linalg import norm, eigvalsh, eigh, toeplitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.figio import save_fig  # noqa: E402
from cgmy_cert.params import Query, Mesh, sobol_queries  # noqa: E402
from cgmy_cert.symbols import free_matrices, legacy_principal_period_free, _dense_banded  # noqa: E402
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.riesz import pcg_bracket, circulant_preconditioner, plain_cg_iterations  # noqa: E402
from cgmy_cert.direct import reference_trajectory, cn_march, mass_norm, free_basis_eval  # noqa: E402
from cgmy_cert.certificate import certify  # noqa: E402

RES = ROOT / "results"; FIG = ROOT / "figures_v9"; TAB = ROOT / "tables_v9"
for d in (RES, FIG, TAB):
    d.mkdir(exist_ok=True)
OUT = RES / "E0.json"
FIXJ = ROOT / "tests" / "fixtures" / "queries.json"
N_T = 60
# Okabe-Ito, fixed order
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.write_text(json.dumps(res, indent=1, default=float))


def pops():
    raw = json.loads(FIXJ.read_text())["populations"]
    return {k: [Query(C=d["C"], G=d["G"], M=d["M"], Y=d["Y"], r=d["r"], T=d["T"], K=d["K"], query_id=d["query_id"])
                for d in v] for k, v in raw.items()}


def qdict(q):
    return dict(C=q.C, G=q.G, M=q.M, Y=q.Y, r=q.r, T=q.T, K=q.K, query_id=q.query_id)


def qfrom(d):
    return Query(**d)


def fmt(x, digits=2):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    if -2 <= e <= 3:
        return f"{x:.{max(digits - e, 0)}f}" if e < digits else f"{x:.0f}"
    m = x / 10 ** e
    return f"${m:.{digits - 1}f}\\times10^{{{e}}}$"


# ====================================================================== E0.1
def e01(res, nproc):
    """Assembly table vs the 40-digit fixture."""
    d = np.load(ROOT / "tests" / "fixtures" / "realline_40digit.npz")
    meta = json.loads(str(d["meta"]))
    rows = {}
    for e in meta["entries"]:
        if e["which"] != "a" or not e["query_id"].startswith("Q_scale"):
            continue
        key = e["key"]; N = e["N"]
        qd = meta["queries"][e["query_id"]]
        q = Query(C=qd["C"], G=qd["G"], M=qd["M"], Y=qd["Y"], r=qd["r"], T=qd["T"], K=qd["K"], query_id=e["query_id"])
        n = Mesh(N).n_free
        lags = d[key + "|lags"]; vals = d[key + "|values"]
        A_ex = toeplitz(vals[(lags >= 0) & (lags < n)].real, vals[(lags <= 0) & (lags > -n)][::-1].real)
        nA = norm(A_ex, 2)
        row = dict(query_id=e["query_id"], N=N, normA=nA)
        row["pp"] = norm(legacy_principal_period_free(q, N)["A"] - A_ex, 2) / nA
        for K in (5, 20, 100):
            fs = free_matrices(q, N, K=K)
            row[f"K{K}"] = norm(fs.A.dense() - A_ex, 2) / nA
            row[f"eps{K}"] = fs.eps_K / nA
        rows.setdefault(N, []).append(row)
        print(f"  {key}: pp {row['pp']:.3f}  K5 {row['K5']:.2e} K20 {row['K20']:.2e} K100 {row['K100']:.2e} (eps100 {row['eps100']:.2e})")
    res["E0.1"] = {str(N): v for N, v in rows.items()}
    plot_e01(res)


# ====================================================================== E0.2
def _e02_one(args):
    qd, N = args
    q = qfrom(qd)
    fs = free_matrices(q, N, K=100)
    sp = spectral_data(q, N, K=100)
    dd = fs.dense()
    lmin = float(eigvalsh(dd["S"]).min())
    X = _dense_banded(sp.X_entries, fs.n)
    ev = eigh(dd["S"], X, eigvals_only=True)
    return dict(query_id=q.query_id, N=N, lmin=lmin, s_min=sp.s_min, alpha_lb=sp.alpha_lb, r_mmin=sp.r_mmin,
                beta=sp.beta, Theta=sp.Theta, gen_min=float(ev.min()), gen_max=float(ev.max()), eps_K=sp.eps_K)


def e02(res, nproc):
    P = pops()["spectral288"]
    worst = Query(0.05, 5.0, 9.0, 1.1, 0.01, query_id="worst_audit_vertex")
    out = {}
    for N in (64, 256, 1024):
        tasks = [(qdict(q), N) for q in P] + [(qdict(worst), N)]
        t0 = time.time()
        with Pool(nproc) as pool:
            rows = pool.map(_e02_one, tasks, chunksize=4)
        r1 = np.array([r["lmin"] / r["s_min"] for r in rows]); r2 = np.array([r["lmin"] / r["r_mmin"] for r in rows])
        viol = [r["query_id"] for r in rows if r["lmin"] < r["alpha_lb"]]
        gviol = [r["query_id"] for r in rows if r["gen_min"] < 1 - 1e-9 or r["gen_max"] > r["Theta"] * (1 + 1e-9)]
        print(f"  N={N}: {len(rows)} queries in {time.time()-t0:.0f}s; lmin/s_min in [{r1.min():.4f}, {r1.max():.2f}], "
              f"lmin/(r m_min) in [{r2.min():.2f}, {r2.max():.1f}]; violations {viol}; gen-eig violations {gviol}")
        out[str(N)] = dict(rows=rows, ratio_symbol=[float(r1.min()), float(np.median(r1)), float(r1.max())],
                           ratio_rm=[float(r2.min()), float(np.median(r2)), float(r2.max())], violations=viol,
                           gen_violations=gviol)
    res["E0.2"] = out
    plot_e02(res)


# ====================================================================== E0.3
def _e03_one(args):
    qd, N = args
    sp = spectral_data(qfrom(qd), N, K=100)
    return dict(query_id=qd["query_id"], N=N, sqrtTheta=float(np.sqrt(sp.Theta)), beta=sp.beta)


def _pod_candidate(fs, Z, r):
    M = _dense_banded(fs.M_entries, fs.n)
    Lc = np.linalg.cholesky(M)
    U, s, Vt = np.linalg.svd(Lc.T @ Z, full_matrices=False)
    P = np.linalg.solve(Lc.T, U[:, :r])
    return P @ (P.T @ (M @ Z))


def _coarse_candidate(q, fs_fine, Z_coarse, msh_coarse):
    """Prolong a coarse exact trajectory to the fine mesh by L2 projection (a smooth candidate with
    structured, discretisation-type error)."""
    from scipy.linalg import solve_banded
    msh = fs_fine.mesh
    # evaluate coarse solution at fine Gauss points and project
    xg, wg = np.polynomial.legendre.leggauss(6)
    nodes = msh.nodes
    b = np.zeros((msh.n_free, Z_coarse.shape[1]))
    Zraw = np.zeros((msh_coarse.n_raw, Z_coarse.shape[1]))
    Zraw[msh_coarse.free_slice] = Z_coarse
    for j in range(msh.N):
        a, c = nodes[j], nodes[j + 1]
        x = 0.5 * (c - a) * xg + 0.5 * (c + a); w = 0.5 * (c - a) * wg
        V = (free_basis_eval(x, msh_coarse) @ Z_coarse)
        Phi = free_basis_eval(x, msh)
        b += Phi.T @ (w[:, None] * V)
    bw = msh.p - 1
    return solve_banded((bw, bw), fs_fine.M_banded(), b)


_POD_BASIS = {}


def _snapshot_one(args):
    qd, N = args
    fs, forc, tr = reference_trajectory(qfrom(qd), N, N_T, solver="gmres")
    return tr.Z


def build_pod_bases(nproc, Ns=(64, 512), n_train=200, r_max=48):
    """Mass-weighted POD from n_train Sobol trajectories (seed 1) at each N (Phase-1 stand-in for
    the Phase-2 reduced model: same construction, smaller training set, no EIM)."""
    from cgmy_cert.pod import pod_basis
    train = sobol_queries(n_train, seed=1)
    for N in Ns:
        t0 = time.time()
        with Pool(nproc) as pool:
            snaps = pool.map(_snapshot_one, [(qdict(q), N) for q in train], chunksize=4)
        S = np.hstack(snaps)
        fs = free_matrices(train[0], N, K=100)
        V, sv = pod_basis(S, fs, r_max)
        _POD_BASIS[N] = (V, sv / sv[0])
        print(f"  POD basis N={N}: {len(train)} trajectories, {S.shape[1]} snapshots, {time.time()-t0:.0f}s; "
              f"sigma_r/sigma_1 at r=16,32,48: {sv[16]/sv[0]:.1e} {sv[32]/sv[0]:.1e} {sv[48]/sv[0]:.1e}")


def _e03_eff_one(args):
    qd, N = args
    q = qfrom(qd)
    fs, forc, tr = reference_trajectory(q, N, N_T, solver="gmres")
    sp = spectral_data(q, N, K=100)
    z0 = tr.Z[:, 0]
    from cgmy_cert.pod import pod_galerkin
    V = _POD_BASIS[N][0]
    cands = {f"podgal{r}": pod_galerkin(fs, forc, V[:, :r], z0) for r in (16, 32, 48)}
    Nc = N // 4
    fsc, forcc, trc = reference_trajectory(q, Nc, N_T, solver="lu")
    cands["coarse_N/4"] = _coarse_candidate(q, fs, trc.Z, fsc.mesh)
    rng = np.random.default_rng(zlib.crc32(q.query_id.encode()))
    dz = rng.standard_normal(fs.n); dz *= 1e-4 / mass_norm(fs, dz)
    cands["perturbed_z0"] = cn_march(fs, z0 + dz, q.T, N_T, solver="gmres", forcing=forc).Z
    out = dict(query_id=q.query_id, N=N, sqrtTheta=float(np.sqrt(sp.Theta)), cands={})
    for name, Z in cands.items():
        err = max(mass_norm(fs, Z[:, k] - tr.Z[:, k]) for k in range(N_T + 1))
        cX = certify(fs, forc, Z, z0, 1e-3, mode="banded", spec=sp)
        cS = certify(fs, forc, Z, z0, 1e-3, mode="exact", spec=sp)
        out["cands"][name] = dict(err=err, etaX=cX.etaX, etaS=cS.etaS_upper, etaX_aug=cX.etaX_aug, etaS_aug=cS.etaS_aug,
                                  eff_X=cX.etaX_aug / err, eff_S=cS.etaS_aug / err,
                                  ratio=(cX.etaX / cS.etaS_upper if cS.etaS_upper > 0 else np.nan),
                                  pcg_mean=float(np.mean(cS.pcg_iters)), t_banded=cX.time_banded, t_exact=cS.time_exact,
                                  t_residual=cX.time_residual)
    return out


def e03(res, nproc):
    P = pops()["spectral288"]
    Ns = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    tasks = [(qdict(q), N) for N in Ns for q in P]
    t0 = time.time()
    with Pool(nproc) as pool:
        rows = pool.map(_e03_one, tasks, chunksize=8)
    print(f"  Theta: {len(rows)} evaluations in {time.time()-t0:.0f}s")
    theta = {}
    for N in Ns:
        a = np.array([r["sqrtTheta"] for r in rows if r["N"] == N])
        theta[str(N)] = dict(median=float(np.median(a)), max=float(a.max()), min=float(a.min()), q90=float(np.quantile(a, 0.9)))
        print(f"    N={N}: sqrt(Theta) median {np.median(a):.2f} max {a.max():.2f}")
    res["E0.3"] = dict(sqrtTheta=theta, theta_rows=rows); save(res)
    # observed effectivity on candidates at N = 64 and 512 (40 Sobol of the spectral set + 8 vertices)
    sub = [q for q in P if q.query_id.startswith("sobol")][:40] + [q for q in P if q.query_id.startswith("vertex")][:8]
    build_pod_bases(nproc)
    eff = {}
    for N in (64, 512):
        t0 = time.time()
        with Pool(nproc) as pool:
            er = pool.map(_e03_eff_one, [(qdict(q), N) for q in sub], chunksize=2)
        eff[str(N)] = er
        print(f"  effectivity N={N}: {len(er)} queries in {time.time()-t0:.0f}s")
        for name in er[0]["cands"]:
            rat = np.array([r["cands"][name]["ratio"] for r in er]); effX = np.array([r["cands"][name]["eff_X"] for r in er])
            effS = np.array([r["cands"][name]["eff_S"] for r in er])
            print(f"    {name:12s}: eta^X/eta^S median {np.nanmedian(rat):.2f} max {np.nanmax(rat):.2f} | eff_S med {np.median(effS):.2f} max {effS.max():.2f} | eff_X med {np.median(effX):.2f} max {effX.max():.2f}")
    res["E0.3"] = dict(sqrtTheta=theta, theta_rows=rows, effectivity=eff,
                       pod_sv={str(N): _POD_BASIS[N][1][:49].tolist() for N in _POD_BASIS},
                       note="candidates: POD-Galerkin (mass-weighted basis from 200 Sobol trajectories, seed 1; rank 16/32/48; "
                            "full operator, no EIM), the L2-prolonged N/4 exact solution, and exact dynamics from perturbed z^0")
    save(res)
    plot_e03(res)


# ====================================================================== E0.4
def _e04_one(args):
    qd, N = args
    q = qfrom(qd)
    fs = free_matrices(q, N, K=100)
    sp = spectral_data(q, N, K=100)
    C = circulant_preconditioner(sym_vals=fs.sym.s.astype(complex), n=fs.n, kind="symbol")
    rng = np.random.default_rng(zlib.crc32(q.query_id.encode()))
    e = np.cumsum(rng.standard_normal(fs.n)); e -= e.mean()
    rho = fs.A.matvec(e)
    pcg = pcg_bracket(fs.S, rho, sp.alpha_lb, precond=C, tol=1e-12, maxit=500).iters
    cg = plain_cg_iterations(fs.S, rho, tol=1e-12, maxit=5000)
    fs2, forc, tr = reference_trajectory(q, N, N_T, solver="gmres")
    return dict(query_id=q.query_id, N=N, pcg=pcg, cg=cg, gmres_mean=float(np.mean(tr.iters)), gmres_max=int(max(tr.iters)))


def e04(res, nproc):
    qs = sobol_queries(20, seed=11)
    Ns = [64, 128, 256, 512, 1024, 2048, 4096]
    with Pool(nproc) as pool:
        rows = pool.map(_e04_one, [(qdict(q), N) for N in Ns for q in qs], chunksize=2)
    out = {}
    for N in Ns:
        rr = [r for r in rows if r["N"] == N]
        cg = np.array([r["cg"] for r in rr]); pcg = np.array([r["pcg"] for r in rr])
        gm = np.array([r["gmres_mean"] for r in rr]); gx = np.array([r["gmres_max"] for r in rr])
        out[str(N)] = dict(cg=[float(np.median(cg)), int(cg.max())], pcg=[float(np.median(pcg)), int(pcg.max())],
                           gmres=[float(np.median(gm)), int(gx.max())], rows=rr)
        print(f"  N={N}: CG {np.median(cg):.0f}/{cg.max()}  PCG {np.median(pcg):.0f}/{pcg.max()}  GMRES {np.median(gm):.1f}/{gx.max()}")
    res["E0.4"] = out
    plot_e04(res)


# ====================================================================== E0.5
def e05(res, nproc):
    out = []
    for Y in (0.5, 0.8, 0.95):
        q = Query(0.5, 3.0, 5.0, Y, 0.05, T=0.5, query_id=f"Y{Y}")
        N = 64
        fs, forc, tr = reference_trajectory(q, N, N_T, solver="gmres")
        trL = cn_march(fs, tr.Z[:, 0], q.T, N_T, solver="lu", forcing=forc)
        dLG = max(mass_norm(fs, tr.Z[:, k] - trL.Z[:, k]) for k in range(N_T + 1))
        sp = spectral_data(q, N, K=100)
        lmin = float(eigvalsh(fs.S.dense()).min())
        rng = np.random.default_rng(3)
        dz = rng.standard_normal(fs.n); dz *= 1e-4 / mass_norm(fs, dz)
        trP = cn_march(fs, tr.Z[:, 0] + dz, q.T, N_T, solver="lu", forcing=forc)
        err = max(mass_norm(fs, trP.Z[:, k] - trL.Z[:, k]) for k in range(N_T + 1))
        cX = certify(fs, forc, trP.Z, tr.Z[:, 0], 1e-3, mode="banded", spec=sp)
        cS = certify(fs, forc, trP.Z, tr.Z[:, 0], 1e-3, mode="exact", spec=sp)
        Zc = _pod_candidate(fs, trL.Z, 12)
        errc = max(mass_norm(fs, Zc[:, k] - trL.Z[:, k]) for k in range(N_T + 1))
        c2 = certify(fs, forc, Zc, tr.Z[:, 0], 1e-3, mode="two-stage", spec=sp)
        row = dict(Y=Y, alpha_lb=sp.alpha_lb, s_min=sp.s_min, lmin=lmin, r_mmin=sp.r_mmin, beta=sp.beta,
                   sqrtTheta=float(np.sqrt(sp.Theta)), gmres_max=int(max(tr.iters)), lu_vs_gmres=dLG,
                   perturbed=dict(err=err, eff_X=cX.bound / err, eff_S=cS.bound / err),
                   pod12=dict(err=errc, status=c2.status, bound=c2.bound, holds=bool(c2.bound >= errc)),
                   gammaY=q.gammaY, omega_S=q.omega_S)
        print(f"  Y={Y}: alpha_lb {sp.alpha_lb:.5f} (lmin {lmin:.5f}, r m_min {sp.r_mmin:.5f}), sqrtTheta {np.sqrt(sp.Theta):.2f}, "
              f"GMRES max {max(tr.iters)}, perturbed eff {cX.bound/err:.6f}/{cS.bound/err:.6f}, pod12 {c2.status} bound {c2.bound:.2e} err {errc:.2e}")
        out.append(row)
    res["E0.5"] = out


# ====================================================================== plot steps (JSON only)

def plot_e01(res):
    """tables_v9/tab_assembly_v9.tex from E0.json -> E0.1 (no computation)."""
    rows = {int(N): v for N, v in res["E0.1"].items()}
    lines = [r"\begin{table}[!htbp]", r"\centering\small",
             r"\caption{Assembly verification against the exact Galerkin matrix (40-digit real-line "
             r"fixture): relative operator-norm error of the free matrix $A_I$, median / max over the "
             r"ten cost-study queries ($N\le256$; three at $N=512$). The principal-period rule of the earlier "
             r"version is the alias sum with $K=0$; $\varepsilon_K/\|A_I\|_2$ is the bound of Lemma~\ref{lem:tail}.}",
             r"\label{tab:assembly-v9}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$N$ & queries & principal period & $K=5$ & $K=20$ & $K=100$ & $\varepsilon_{100}/\|A_I\|_2$ (max)\\", r"\midrule"]
    for N in sorted(rows):
        rr = rows[N]
        def mm(k):
            a = np.array([r[k] for r in rr]); return f"{fmt(np.median(a))} / {fmt(a.max())}"
        lines.append(f"{N} & {len(rr)} & {mm('pp')} & {mm('K5')} & {mm('K20')} & {mm('K100')} & {fmt(max(r['eps100'] for r in rr))}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (TAB / "tab_assembly_v9.tex").write_text("\n".join(lines) + "\n")


def plot_e02(res):
    """figures_v9/F2_spectral_bound.{png,pdf} from E0.json -> E0.2."""
    out = res["E0.2"]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.3))
    for ax, key, lab in ((axs[0], "ratio_symbol", r"$\lambda_{\min}(S_I)/s_{\min}(\mu)$"),
                         (axs[1], "ratio_rm", r"$\lambda_{\min}(S_I)/(r\,m_{\min})$")):
        data = []
        for N in ("64", "256", "1024"):
            rows = out[N]["rows"]
            data.append([r["lmin"] / (r["s_min"] if key == "ratio_symbol" else r["r_mmin"]) for r in rows[:-1]])
        bp = ax.boxplot(data, tick_labels=["64", "256", "1024"], widths=0.5, patch_artist=True,
                        medianprops=dict(color="black", lw=1.2), flierprops=dict(marker=".", ms=3, alpha=0.5))
        for b in bp["boxes"]:
            b.set(facecolor=PAL[0], alpha=0.35, edgecolor=PAL[0])
        for i, N in enumerate(("64", "256", "1024")):
            w = out[N]["rows"][-1]
            ax.plot(i + 1, w["lmin"] / (w["s_min"] if key == "ratio_symbol" else w["r_mmin"]), marker="*", ms=11,
                    color=PAL[3], ls="none", label="worst audit vertex" if i == 0 else None)
        ax.set_yscale("log"); ax.axhline(1.0, color="gray", lw=0.8, ls="--"); ax.set_xlabel("$N$"); ax.set_ylabel(lab)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Rigorous spectral lower bound over 288 queries (32 vertices + 256 Sobol), $K=100$", fontsize=10)
    fig.tight_layout(); save_fig(fig, FIG / "F2_spectral_bound.png"); plt.close(fig)


def plot_e03(res):
    """figures_v9/F3_theta_and_effectivity.{png,pdf} from E0.json -> E0.3."""
    theta, eff = res["E0.3"]["sqrtTheta"], res["E0.3"]["effectivity"]
    Ns = sorted(int(N) for N in theta)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.3))
    ax = axs[0]
    ax.plot(Ns, [theta[str(N)]["median"] for N in Ns], marker="o", ms=4, color=PAL[0], label="median")
    ax.plot(Ns, [theta[str(N)]["q90"] for N in Ns], marker="s", ms=4, color=PAL[1], label="90th percentile")
    ax.plot(Ns, [theta[str(N)]["max"] for N in Ns], marker="^", ms=4, color=PAL[3], label="max")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("$N$"); ax.set_ylabel(r"$\sqrt{\Theta_\mu}$")
    ax.grid(True, which="both", lw=0.3, alpha=0.4); ax.legend(frameon=False, fontsize=8)
    ax.set_title("A-priori effectivity bound of $\\eta^X$ (288 queries)", fontsize=9)
    ax = axs[1]
    names = list(eff["64"][0]["cands"].keys())
    for i, N in enumerate(("64", "512")):
        data = [[r["cands"][nm]["ratio"] for r in eff[N]] for nm in names]
        pos = np.arange(len(names)) + (i - 0.5) * 0.36
        bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, medianprops=dict(color="black", lw=1.1),
                        flierprops=dict(marker=".", ms=3, alpha=0.5))
        for b in bp["boxes"]:
            b.set(facecolor=PAL[i], alpha=0.4, edgecolor=PAL[i])
        ax.plot([], [], color=PAL[i], lw=6, alpha=0.4, label=f"$N={N}$")
    ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel(r"observed $\eta^X/\eta^S$"); ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.legend(frameon=False, fontsize=8); ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax.set_title("Observed ratio on candidates (48 queries)", fontsize=9)
    fig.tight_layout(); save_fig(fig, FIG / "F3_theta_and_effectivity.png"); plt.close(fig)


def plot_e04(res):
    """tables_v9/tab_pcg_v9.tex from E0.json -> E0.4."""
    out = res["E0.4"]
    Ns = sorted(int(N) for N in out)
    lines = [r"\begin{table}[!htbp]", r"\centering\small",
             r"\caption{Iteration counts (median / max over 20 Sobol queries) to relative residual $10^{-12}$: "
             r"conjugate gradients for the Riesz system $S_Iv=\rho$ with and without the circulant preconditioner "
             r"built from the symbol samples, and circulant-preconditioned GMRES per Crank--Nicolson step for "
             r"$B_+=M_I+\tfrac{\Delta\tau}{2}A_I$ (60 steps, warm start, mean over steps / max).}",
             r"\label{tab:pcg-v9}", r"\begin{tabular}{rrrr}", r"\toprule",
             r"$N$ & plain CG & circulant PCG & circulant GMRES ($B_+$)\\", r"\midrule"]
    for N in Ns:
        rr = out[str(N)]["rows"]
        cg = np.array([r["cg"] for r in rr]); pcg = np.array([r["pcg"] for r in rr])
        gm = np.array([r["gmres_mean"] for r in rr]); gx = np.array([r["gmres_max"] for r in rr])
        lines.append(f"{N} & {np.median(cg):.0f} / {cg.max()} & {np.median(pcg):.0f} / {pcg.max()} & {np.median(gm):.1f} / {gx.max()}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_pcg_v9.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*", default=["E0.1", "E0.2", "E0.3", "E0.4", "E0.5"],
                    help="E0.x computes and plots; 'plot' regenerates the tables/figures from results/E0.json only")
    ap.add_argument("--nproc", type=int, default=2)
    a = ap.parse_args()
    res = load()
    if a.items == ["plot"]:
        plot_e01(res); plot_e02(res); plot_e03(res); plot_e04(res)
        raise SystemExit(0)
    for item in a.items:
        print(f"=== {item}")
        t0 = time.time()
        {"E0.1": e01, "E0.2": e02, "E0.3": e03, "E0.4": e04, "E0.5": e05}[item](res, a.nproc)
        res.setdefault("timing_s", {})[item] = time.time() - t0
        save(res)
        print(f"=== {item} done in {time.time()-t0:.0f}s")
