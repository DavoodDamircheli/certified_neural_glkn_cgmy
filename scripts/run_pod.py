#!/usr/bin/env python3
"""Plan 2.2: POD bases from the stored references and POD-Galerkin with the affine (EIM) operator,
certified by the full-order certificate with the Theorem 5.2 allowance.  Writes
data/pod_N512_NT240_seed0_{800,4000}.npz (sha256 -> results/MANIFEST.json) and results/E2_2_pod.json.

Usage: python3 scripts/run_pod.py [basis] [validate] [--nproc 20] [--sets 800 4000]
"""
from __future__ import annotations

import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, Mesh  # noqa: E402
from cgmy_cert.symbols import free_matrices, _dense_banded  # noqa: E402
from cgmy_cert.eim import load_eim, sha256  # noqa: E402
from cgmy_cert.pod import AffineROM, mass_cholesky, allowance_vectors  # noqa: E402
from cgmy_cert.direct import initial_free_vector, free_forcing, mass_norm  # noqa: E402
from cgmy_cert.spectral import spectral_data  # noqa: E402
from cgmy_cert.certificate import certify  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

N, NT = 512, 240
DATA = ROOT / "data"; RES = ROOT / "results"
OUT = RES / "E2_2_pod.json"; MANIFEST = RES / "MANIFEST.json"
RANKS = (8, 16, 24, 32, 40, 48, 56, 62)
EPS_SWEEP = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)     # provisional (v8 eps = 1e-3 x {0.1,...,10}); E1 uses eps_N
R_MAX = 62


def ref_file(tag):
    return DATA / f"references_N{N}_NT{NT}_{tag}.npz"


def pod_file(n):
    return DATA / f"pod_N{N}_NT{NT}_seed0_{n}.npz"


def load_refs(tag, mmap=True):
    return np.load(ref_file(tag), mmap_mode="r" if mmap else None, allow_pickle=False)


def queries_from(d):
    return [Query(C=m[0], G=m[1], M=m[2], Y=m[3], r=m[4], T=float(T), query_id=str(i))
            for m, T, i in zip(d["mu"], d["T"], d["query_ids"])]


def check_manifest(tag):
    man = json.loads(MANIFEST.read_text())
    ent = man["files"][ref_file(tag).name]
    return ent["sha256"]


# ---------------------------------------------------------------- POD basis

def do_basis(sets):
    man = json.loads(MANIFEST.read_text())
    fs = free_matrices(Query(0.5, 3.0, 5.0, 1.5, 0.05), N)      # M_I does not depend on mu
    Lc = mass_cholesky(fs)
    for n in sets:
        t0 = time.time()
        d = load_refs("seed0_4000")
        Z = np.asarray(d["Z"][:n])                                  # (n, n_free, NT+1)
        # all snapshots, mass-weighted (L^T Z), thin SVD
        W = np.einsum("ij,qjt->iqt", Lc.T, Z, optimize=True).reshape(fs.n, -1)
        U, s, _ = np.linalg.svd(W, full_matrices=False)
        from scipy.linalg import solve_triangular
        V = solve_triangular(Lc.T, U[:, :R_MAX], lower=False)
        orth = float(np.abs(V.T @ _dense_banded(fs.M_entries, fs.n) @ V - np.eye(R_MAX)).max())
        np.savez(pod_file(n), V=V, sv=s, n_traj=n, N=N, N_T=NT, source=ref_file("seed0_4000").name)
        man["files"][pod_file(n).name] = dict(
            file=str(pod_file(n).relative_to(ROOT)), sha256=sha256(pod_file(n)), n_traj=n, r_max=R_MAX,
            source=ref_file("seed0_4000").name, source_sha256=check_manifest("seed0_4000"),
            construction="mass-weighted POD (V^T M_I V = I) of all N_T+1 snapshots of the first n trajectories",
            mass_orthonormality_err=orth, wall_s=time.time() - t0, provenance=provenance())
        MANIFEST.write_text(json.dumps(man, indent=1))
        print(f"POD n={n}: sigma_r/sigma_1 at r=8..62: "
              + " ".join(f"{r}:{s[r] / s[0]:.1e}" for r in RANKS if r < s.size) + f"  ({time.time() - t0:.0f}s)", flush=True)
        del W, Z


# ---------------------------------------------------------------- validation30

def _validate_one(args):
    i, n_basis = args
    d = load_refs("validation30")
    q = queries_from(d)[i]
    Zref = np.asarray(d["Z"][i])
    e = load_eim(N, build_if_missing=False)
    V = np.load(pod_file(n_basis))["V"]
    fs_Q, info = e.affine_free_system(q)
    fs = free_matrices(q, N)
    forc_Q = free_forcing(fs_Q, q.T, NT)
    forc = free_forcing(fs, q.T, NT)
    spec = spectral_data(q, N)
    z0 = initial_free_vector(q, fs.mesh)
    assert np.allclose(z0, Zref[:, 0], rtol=0, atol=1e-12 * np.abs(z0).max())
    ref_norm = max(mass_norm(fs, Zref[:, k]) for k in range(NT + 1))
    out = dict(query_id=q.query_id, mu=list(q.mu), T=q.T, eps_Q=info["eps_Q"], alpha_lb=spec.alpha_lb,
               x_min=spec.x_min, sqrt_Theta=float(np.sqrt(spec.Theta)), ref_max_mass_norm=ref_norm, ranks={})
    for r in RANKS:
        rom = AffineROM(V[:, :r], e)
        t0 = time.perf_counter()
        sol = rom.solve(q, rom.project(z0), NT, ell=info["ell"])
        t_solve = time.perf_counter() - t0
        Z = V[:, :r] @ sol.alpha
        err = max(mass_norm(fs, Z[:, k] - Zref[:, k]) for k in range(NT + 1))
        aS, aX, raw, free = allowance_vectors(Z, sol.bL, fs.mesh.p, info["eps_Q"], spec.alpha_lb, spec.x_min)
        cX = certify(fs_Q, forc_Q, Z, z0, 1e-3, mode="banded", spec=spec, allowance=aS, allowance_X=aX)
        cS = certify(fs_Q, forc_Q, Z, z0, 1e-3, mode="exact", spec=spec, allowance=aS, allowance_X=aX)
        # the same candidate certified with the exact operator A_I (no allowance), for comparison
        cXI = certify(fs, forc, Z, z0, 1e-3, mode="banded", spec=spec)
        resX = float(np.sqrt(forc.dt * np.sum(cX.per_step_X)))
        resS = float(np.sqrt(forc.dt * np.sum(cS.per_step_u)))
        out["ranks"][str(r)] = dict(
            true_err=err, beta0=cX.beta0, etaX_aug=cX.etaX_aug, etaS_aug=cS.etaS_aug, etaX_aug_AI=cXI.etaX_aug,
            residual_X=resX, residual_S=resS,
            allowance_X_l2=float(np.sqrt(forc.dt * np.sum(aX ** 2))),
            allowance_S_l2=float(np.sqrt(forc.dt * np.sum(aS ** 2))),
            allowance_X_over_residual=float(np.sqrt(forc.dt * np.sum(aX ** 2)) / resX) if resX > 0 else None,
            raw_over_free_norm_max=float(np.max(raw / free)),
            eff_X=cX.etaX_aug / err, eff_S=cS.etaS_aug / err,
            accept_X={f"{t:g}": bool(cX.etaX_aug <= t) for t in EPS_SWEEP},
            accept_S={f"{t:g}": bool(cS.etaS_aug <= t) for t in EPS_SWEEP},
            truly_within={f"{t:g}": bool(err <= t) for t in EPS_SWEEP},
            t_reduced_solve_s=t_solve, pcg_mean=float(np.mean(cS.pcg_iters)))
    return out


def do_validate(nproc, n_basis=800):
    t0 = time.time()
    with Pool(nproc) as pool:
        rows = pool.map(_validate_one, [(i, n_basis) for i in range(30)], chunksize=1)
    summ = {}
    for r in RANKS:
        R = [x["ranks"][str(r)] for x in rows]
        s = dict(median_true_err=float(np.median([x["true_err"] for x in R])),
                 max_true_err=float(np.max([x["true_err"] for x in R])),
                 median_etaX=float(np.median([x["etaX_aug"] for x in R])),
                 median_etaS=float(np.median([x["etaS_aug"] for x in R])),
                 median_eff_X=float(np.median([x["eff_X"] for x in R])),
                 median_eff_S=float(np.median([x["eff_S"] for x in R])),
                 min_eff_S=float(np.min([x["eff_S"] for x in R])),
                 max_allowance_X_over_residual=float(np.max([x["allowance_X_over_residual"] for x in R])),
                 max_rel_diff_etaX_AQ_vs_AI=float(np.max([abs(x["etaX_aug"] - x["etaX_aug_AI"]) / x["etaX_aug_AI"] for x in R])),
                 max_raw_over_free_norm=float(np.max([x["raw_over_free_norm_max"] for x in R])))
        for t in EPS_SWEEP:
            k = f"{t:g}"
            tw = np.array([x["truly_within"][k] for x in R])
            for mode in ("X", "S"):
                ac = np.array([x[f"accept_{mode}"][k] for x in R])
                s[f"eps={k}:{mode}"] = dict(accepted=int(ac.sum()), truly_within=int(tw.sum()),
                                            false_accepts=int((ac & ~tw).sum()), missed=int((~ac & tw).sum()))
        summ[str(r)] = s
        print(f"r={r}: true err med {s['median_true_err']:.2e} max {s['max_true_err']:.2e} | etaS med "
              f"{s['median_etaS']:.2e} eff_S med {s['median_eff_S']:.1f} | eps=1e-3 X {s['eps=0.001:X']} "
              f"S {s['eps=0.001:S']} | allow/res max {s['max_allowance_X_over_residual']:.1e}", flush=True)
    res = dict(meta=dict(description="Plan 2.2: POD-Galerkin with the affine EIM operator A_Q on validation30, "
                         "certified with the full-order certificate on (A_Q, f_Q) plus the Theorem 5.2 allowance "
                         "eps_Q ||cbar_raw||_2 / sqrt(alpha) (alpha = alpha_lb exact mode, x_min banded mode). "
                         "true_err = max_n ||Z^n - Z_ref^n||_{M_I} against the stored GMRES references; "
                         "bounds are augmented (beta0 included), same units.",
                         N=N, N_T=NT, Q=load_eim(N, build_if_missing=False).Q, ranks=list(RANKS),
                         eps_sweep=list(EPS_SWEEP),
                         eps_note="provisional sweep around the v8 tolerance 1e-3; E1 repeats at eps_N from 3.1",
                         population="validation30 (tests/fixtures/queries.json)",
                         basis=pod_file(n_basis).name, references=ref_file("validation30").name,
                         references_sha256=check_manifest("validation30"),
                         timing_note="t_reduced_solve_s measured in a 20-process pool; not paper timings",
                         wall_s=time.time() - t0, provenance=provenance(nproc=nproc)),
               summary=summ, rows=rows)
    OUT.write_text(json.dumps(res, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["basis", "validate"])
    ap.add_argument("--nproc", type=int, default=20)
    ap.add_argument("--sets", nargs="*", type=int, default=[800, 4000])
    a = ap.parse_args()
    if "basis" in a.steps:
        do_basis(a.sets)
    if "validate" in a.steps:
        do_validate(a.nproc)


if __name__ == "__main__":
    main()
