#!/usr/bin/env python3
"""Plan 2.4: train one POD-NN (r = 48) on the stored references and evaluate the selected checkpoint.

    python3 scripts/train_podnn.py run --n 800 --lam 0.1 --seed 0 [--epochs 2000] [--threads 4]
    python3 scripts/train_podnn.py prep                       # cache the projected data (run once)
    python3 scripts/train_podnn.py check                      # torch vs numpy eta^{X-bar}, 5 random trajectories
    python3 scripts/train_podnn.py collect                    # merge per-run logs into results/training_log.json

Checkpoint selection: every --val-every epochs the median rigorous reduced bound eta_aug (beta_0, EIM
allowance, conversion 1/c_mu, rounding term) on validation30 is computed; the best one is kept.
Provisional acceptance target eps = 1e-3 (the v8 tolerance) until 3.1 fixes eps_N; E1 repeats at eps_N.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query  # noqa: E402
from cgmy_cert.eim import load_eim, sha256  # noqa: E402
from cgmy_cert.reduced_cert import load_rc, conversion_factor  # noqa: E402
from cgmy_cert.podnn import prepare, PODNN, TorchReducedCert, rel_ls, CELL_IDS, PODNNData  # noqa: E402
from cgmy_cert.provenance import provenance  # noqa: E402

N, NT, R = 512, 240, 48
DATA = ROOT / "data"; RES = ROOT / "results"; CK = ROOT / "checkpoints"
RUNS = RES / "training_runs"
EPS_PROVISIONAL = 1e-3


def ref_file(tag):
    return DATA / f"references_N{N}_NT{NT}_{tag}.npz"


def basis(n):
    return np.load(DATA / f"pod_N{N}_NT{NT}_seed0_{n}.npz")["V"][:, :R]


def rc_file(n):
    return DATA / f"redcert_N{N}_r{R}_n{n}.npz"


def prep_file(tag, n_basis):
    return DATA / f"podnn_prep_{tag}_basis{n_basis}.npz"


def save_prep(d: PODNNData, path, **extra):
    np.savez(path, X=d.X, alpha=d.alpha, theta=d.theta, ell=d.ell, cell=d.cell, dt=d.dt, bL=d.bL,
             query_ids=d.query_ids, **extra)


def load_prep(path):
    z = np.load(path)
    d = PODNNData(X=z["X"], alpha=z["alpha"], theta=z["theta"], ell=z["ell"], cell=z["cell"], dt=z["dt"],
                  bL=z["bL"], query_ids=z["query_ids"])
    return d, {k: z[k] for k in z.files if k not in PODNNData.__dataclass_fields__}


def do_prep():
    e = load_eim(N, build_if_missing=False)
    for n in (800, 4000):
        V = basis(n); rc = load_rc(rc_file(n))
        t0 = time.time()
        save_prep(prepare(ref_file("seed0_4000"), V, rc, e, n=n), prep_file(f"train{n}", n))
        dv = prepare(ref_file("validation30"), V, rc, e)
        qs = dv.queries()
        eps = np.array([e.eps_Q(q)["eps_Q"] for q in qs])
        c_mu = np.array([conversion_factor(q, N, rc.cells[CELL_IDS[c]].rbar, rc.cells[CELL_IDS[c]].betabar)
                         for q, c in zip(qs, dv.cell)])
        save_prep(dv, prep_file("val30", n), eps_Q=eps, c_mu=c_mu)
        print(f"prep basis {n}: {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- rigorous validation metric (numpy)

def val_bounds(alpha, dv: PODNNData, extra, rc):
    out = []
    for i, q in enumerate(dv.queries()):
        c = rc.certify(q, alpha[i], dv.dt[i], dv.bL[i], dv.ell[i], float(extra["eps_Q"][i]),
                       c_mu=float(extra["c_mu"][i]), cid=CELL_IDS[dv.cell[i]])
        out.append(c["eta_aug"])
    return np.array(out)


# ---------------------------------------------------------------- check torch vs numpy

def do_check():
    import torch
    torch.set_num_threads(1)
    rc = load_rc(rc_file(800)); tc = TorchReducedCert(rc)
    dv, extra = load_prep(prep_file("val30", 800))
    rng = np.random.default_rng(5)
    rows = []
    for i in range(5):
        al = dv.alpha[i] + 1e-2 * rng.standard_normal(dv.alpha[i].shape) * np.abs(dv.alpha[i]).max()
        q = dv.queries()[i]
        cid = CELL_IDS[dv.cell[i]]
        Xn = np.vstack([(al[:, 1:] - al[:, :-1]) / dv.dt[i], ((dv.bL[i, 1:] - dv.bL[i, :-1]) / dv.dt[i])[None]])
        Yn = np.vstack([0.5 * (al[:, 1:] + al[:, :-1]), (0.5 * (dv.bL[i, 1:] + dv.bL[i, :-1]))[None]])
        ref = dv.dt[i] * rc.per_step(cid, dv.theta[i], Xn, Yn).sum()
        T = lambda a: torch.as_tensor(a, dtype=torch.float64)
        tv = tc.eta2(T(al[None]), T(dv.theta[i:i + 1]), torch.as_tensor(dv.cell[i:i + 1]), T(dv.dt[i:i + 1]),
                     T(dv.bL[i:i + 1])).item()
        rows.append(dict(query_id=q.query_id, eta_numpy=math.sqrt(ref), eta_torch=math.sqrt(tv),
                         rel_diff=abs(math.sqrt(tv) - math.sqrt(ref)) / math.sqrt(ref)))
    ok = all(r["rel_diff"] <= 1e-10 for r in rows)
    print(json.dumps(rows, indent=1), "\nall <= 1e-10:", ok)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "torch_numpy_check.json").write_text(json.dumps(dict(rows=rows, all_within_1e10=ok,
        description="eta^{X-bar} residual part, torch (TorchReducedCert) vs numpy (ReducedCertificate.per_step) on 5 "
                    "random reduced trajectories (validation30 projections + 1e-2 noise)",
        provenance=provenance()), indent=1))
    return ok


# ---------------------------------------------------------------- training

def run_name(n, lam, seed):
    return f"podnn_r{R}_n{n}_lamR{lam:g}_seed{seed}"


def do_run(n, lam, seed, epochs, threads, val_every, batch=64, lr0=1e-3, lr1=1e-5):
    import torch
    torch.set_num_threads(threads)
    name = run_name(n, lam, seed)
    rc = load_rc(rc_file(n)); tc = TorchReducedCert(rc)
    dtr, _ = load_prep(prep_file(f"train{n}", n))
    dv, extra = load_prep(prep_file("val30", n))
    x_mean, x_std = dtr.X.mean(0), dtr.X.std(0)
    a = dtr.alpha[:, :, 1:]
    a_mean, a_std = a.mean(0), a.std(0) + 1e-12 * np.abs(a).max()
    model = PODNN(R, NT, rc.alpha0(), x_mean, x_std, a_mean, a_std, seed=seed)
    T = lambda z: torch.as_tensor(z, dtype=torch.float64)
    Xt, At, Tht, Ct, Dt, Bt = T(dtr.X), T(dtr.alpha), T(dtr.theta), torch.as_tensor(dtr.cell), T(dtr.dt), T(dtr.bL)
    opt = torch.optim.Adam(model.net.parameters(), lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr1)
    g = torch.Generator().manual_seed(seed)
    log = dict(name=name, n_train=n, lambda_R=lam, seed=seed, epochs=epochs, batch=batch, lr=[lr0, lr1],
               optimizer="Adam, cosine annealing per epoch", n_params=model.n_params(), threads=threads,
               dtype="float64", val_every=val_every, selection="min median rigorous eta_aug (reduced) on validation30",
               train_loss=[], train_rel_ls=[], train_eta=[], val=[], provenance=provenance(torch=torch.__version__))
    best = (np.inf, -1); t_start = time.time()
    CK.mkdir(exist_ok=True)
    for ep in range(epochs):
        perm = torch.randperm(Xt.shape[0], generator=g)
        s_loss = s_ls = s_eta = 0.0
        for i0 in range(0, Xt.shape[0], batch):
            idx = perm[i0:i0 + batch]
            ah = model.forward(Xt[idx])
            ls = rel_ls(ah, At[idx]).mean()
            loss = ls
            if lam > 0:
                eta = torch.sqrt(tc.eta2(ah, Tht[idx], Ct[idx], Dt[idx], Bt[idx]) + 1e-300).mean()
                loss = loss + lam * eta
                s_eta += eta.item() * idx.numel()
            opt.zero_grad(); loss.backward(); opt.step()
            s_loss += loss.item() * idx.numel(); s_ls += ls.item() * idx.numel()
        sched.step()
        nn_ = Xt.shape[0]
        log["train_loss"].append(s_loss / nn_); log["train_rel_ls"].append(s_ls / nn_)
        log["train_eta"].append(s_eta / nn_ if lam > 0 else None)
        if (ep + 1) % val_every == 0 or ep == epochs - 1:
            al = model.predict(dv.X)
            b = val_bounds(al, dv, extra, rc)
            vls = float(np.mean(np.sum((al[:, :, 1:] - dv.alpha[:, :, 1:]) ** 2, axis=(1, 2))
                                / np.sum(dv.alpha[:, :, 1:] ** 2, axis=(1, 2))))
            med = float(np.median(b))
            log["val"].append(dict(epoch=ep + 1, median_eta_aug=med, max_eta_aug=float(b.max()), val_rel_ls=vls,
                                   accepted_at_eps_provisional=int((b <= EPS_PROVISIONAL).sum()), wall_s=time.time() - t_start))
            if med < best[0]:
                best = (med, ep + 1)
                torch.save(model.state(), CK / f"{name}.pt")
            print(f"{name} ep {ep + 1}: loss {s_loss / nn_:.3e} relLS {s_ls / nn_:.3e} | val med eta {med:.3e} "
                  f"relLS {vls:.3e} | {time.time() - t_start:.0f}s", flush=True)
    log.update(selected_epoch=best[1], selected_median_eta_aug=best[0], wall_s=time.time() - t_start,
               checkpoint=f"checkpoints/{name}.pt")
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{name}.json").write_text(json.dumps(log, indent=1, default=float))
    return log


def do_collect():
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("podnn_*.json"))]
    chk = RUNS / "torch_numpy_check.json"
    out = dict(description="Plan 2.4 POD-NN training (r = 48, N = 512, N_T = 240); per-epoch losses, validation "
                           "metric every val_every epochs, selected epoch. Provisional acceptance eps = 1e-3 "
                           "(v8 tolerance) until 3.1; E1 repeats at eps_N.",
               torch_numpy_check=json.loads(chk.read_text()) if chk.exists() else None, runs=runs,
               summary=[dict(name=r["name"], n_train=r["n_train"], lambda_R=r["lambda_R"], seed=r["seed"],
                             selected_epoch=r["selected_epoch"], median_eta_aug=r["selected_median_eta_aug"],
                             wall_s=r["wall_s"]) for r in runs])
    (RES / "training_log.json").write_text(json.dumps(out, indent=1, default=float))
    for s in out["summary"]:
        print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "check", "run", "collect"])
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=10)
    a = ap.parse_args()
    if a.step == "prep":
        do_prep()
    elif a.step == "check":
        sys.exit(0 if do_check() else 1)
    elif a.step == "run":
        do_run(a.n, a.lam, a.seed, a.epochs, a.threads, a.val_every)
    else:
        do_collect()


if __name__ == "__main__":
    main()
