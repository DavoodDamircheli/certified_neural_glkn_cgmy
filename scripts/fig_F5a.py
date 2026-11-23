#!/usr/bin/env python3
"""Figure F5a (E1): acceptance vs eps for the three certificates, and the three-stage status counts per rank.

Top row: POD-Galerkin (basis 800, validation30, N = 1024) at the transition ranks; accepted fraction vs eps for the
reduced online, banded full-order and exact full-order certificates (eps_N marked).  Bottom: three-stage status
counts at eps_N for every rank (stacked).  POD-NN candidates are added as further top-row panels once
results/E2_4_podnn_N1024.json exists (best seed-0 run and the space-time variant).
Usage: python3 scripts/fig_F5a.py [--basis 800]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from cgmy_cert.figio import save_fig  # noqa: E402
RES = ROOT / "results"; FIG = ROOT / "figures"
PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]       # Okabe-Ito, fixed order
CERTS = (("reduced", "reduced online"), ("banded_AI", "banded full-order"), ("exact_AI", "exact full-order"))
CERT_ALIAS = dict(banded_AI="banded", exact_AI="exact")             # E1.json key names
STAGES = (("reduced_accept", "reduced accept"), ("reject_cheap", "reject (reduced stage)"),
          ("banded_accept", "banded accept"), ("reject_banded", "reject (banded stage)"),
          ("exact_accept", "exact accept"), ("exact_reject", "exact reject"))
STAGE_COL = ["#0072B2", "#56B4E9", "#E69F00", "#F0E442", "#009E73", "#D55E00"]
PANEL_RANKS = (48, 56, 62, 64, 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="800")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pc = json.loads((RES / "PRODUCTION_CONFIG.json").read_text())
    eps_N = pc["eps_N"]
    e1p = RES / "E1.json"
    if e1p.exists():                     # Phase 3: the consolidated E1 evaluation (Q matched to eps)
        E = json.loads(e1p.read_text())["eval"]; S = E["summary"]
        keys = list(S[f"pod_galerkin_r48_n{a.basis}"]["exact"]["accepted"])
        eps = np.array([float(k) for k in keys]); n_q = 30
        summ = {str(r): S[f"pod_galerkin_r{r}_n{a.basis}"] for r in (8, 16, 24, 32, 40, 48, 56, 62, 64, 80, 96, 128)}
        sel = json.loads(e1p.read_text())["selection"]["per_setting"]
        panels = [(f"POD-Galerkin r = {r}", summ[str(r)]) for r in PANEL_RANKS]
        for key, lab in (("podnn|n4000", "POD-NN bound-selected"), ("podnn|n4000", "POD-NN error-selected"),
                         ("podnn_spacetime|n4000", "space-time POD-NN")):
            which = "error_selected" if "error" in lab else "bound_selected"
            names = sel[key][which]["names"]
            best = min(names, key=lambda k: S[k]["exact"]["median"])
            panels.append((f"{lab} ($\\lambda_R$={sel[key][which]['lambda_R']:g})", S[best]))
    else:
        d = json.loads((RES / "N1024.json").read_text())
        t = d["table_ext"][a.basis]
        summ, keys = t["summary"], list(next(iter(t["summary"].values()))["reduced"]["accepted"])
        eps = np.array([float(k) for k in keys])
        n_q = len(t["rows"])
        panels = [(f"POD-Galerkin r = {r}", summ[str(r)]) for r in PANEL_RANKS]
    ev = RES / "E2_4_podnn_N1024.json"
    if ev.exists() and not e1p.exists():
        e = json.loads(ev.read_text())["results"]
        nn = {k: v for k, v in e.items() if k.startswith("podnn") and v["summary"]["exact_AI"]["accepted"]}
        if nn:
            std = [k for k in nn if not k.endswith("spacetime")]
            st = [k for k in nn if k.endswith("spacetime")]
            for group, label in ((std, "POD-NN (best)"), (st, "POD-NN space-time (best)")):
                if group:
                    best = min(group, key=lambda k: nn[k]["summary"]["exact_AI"]["median"])
                    panels.append((label, nn[best]["summary"]))
    fig = plt.figure(figsize=(2.6 * len(panels), 6.4))
    gs = fig.add_gridspec(2, len(panels), height_ratios=[1, 1.05], hspace=0.45, wspace=0.12)
    for j, (title, s) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        for c, mk, ms, (key, lab) in zip(PAL, "osD", (8, 6, 4), CERTS):
            kk = key if key in s else CERT_ALIAS[key]
            got = {float(k2): v for k2, v in s[kk]["accepted"].items()}
            acc = np.array([next(v for e2, v in got.items() if abs(e2 - float(k)) <= 1e-3 * float(k)) for k in keys]) / n_q
            ax.semilogx(eps, acc, color=c, lw=2, marker=mk, ms=ms, label=lab)      # shapes/sizes keep overlaps visible
        ax.axvline(eps_N, color="0.5", lw=0.8, ls="--")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$\varepsilon$", fontsize=9)
        if j == 0:
            ax.set_ylabel("accepted fraction (validation30)", fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc="upper left")
        else:
            ax.set_yticklabels([])
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax = fig.add_subplot(gs[1, :])
    ranks = [int(r) for r in summ]
    ke = next(k for k in keys if abs(float(k) - eps_N) < 1e-3 * eps_N)
    x = np.arange(len(ranks)); bottom = np.zeros(len(ranks))
    for (st, lab), c in zip(STAGES, STAGE_COL):
        v = np.array([summ[str(r)]["three_stage"][ke][st] for r in ranks], float)
        if v.sum() == 0:
            continue
        ax.bar(x, v, bottom=bottom, color=c, edgecolor="white", linewidth=1.0, width=0.75, label=lab)
        bottom += v
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("POD rank r (POD-Galerkin)", fontsize=9)
    ax.set_ylabel(f"queries (of {n_q})", fontsize=9)
    ax.set_title(f"Three-stage rule at $\\varepsilon_N$ = {eps_N:.3g}: where each query is settled", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    FIG.mkdir(exist_ok=True)
    out = FIG / "F5a_acceptance.png"
    save_fig(fig, out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
