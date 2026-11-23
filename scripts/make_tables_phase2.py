#!/usr/bin/env python3
"""Tables for Section 5 from the Phase 2 JSONs: tab_pod_ranks_v9.tex, tab_podnn_v9.tex, tab_training_v9.tex."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"; TAB = ROOT / "tables"

def fmt(x, d=2):
    if x == 0: return "0"
    e = int(np.floor(np.log10(abs(x))))
    if -2 <= e <= 2: return f"{x:.{max(d - e, 0)}f}" if e < d else f"{x:.0f}"
    return f"${x / 10 ** e:.{d-1}f}\\times10^{{{e}}}$"

# ------------------------------------------------ POD ranks (validation30, N = 1024; Phase 3 E1 evaluation,
# the same evaluation that Figures 5-6 read)
E1 = json.load(open(RES / "E1.json"))["eval"]["summary"]
eN = "0.001185"
lines = [r"\begin{table}[!htbp]", r"\centering\small",
 r"\caption{POD--Galerkin candidates on the 30 validation queries at the production configuration "
 r"($N=1024$, $N_T=240$, $Q$ matched to $\varepsilon$): median maximum-in-time mass-norm error against the direct "
 r"solution, median effectivity (bound/error) of the three certificates, queries accepted at "
 r"$\varepsilon_N=1.18\times10^{-3}$ under each certificate, and the stage at which the three-stage rule of "
 r"Proposition~\ref{prop:threestage} settles the query (reduced accept / cheap reject / banded accept / exact accept / "
 r"exact reject). Basis from 800 training trajectories; the 4000-trajectory basis gives the same picture "
 r"(Section~\ref{ssec:e1}). No false accepts occur under any certificate at any tolerance of the sweep.}",
 r"\label{tab:pod-ranks}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{rrrrrrr}", r"\toprule",
 r"$r$ & error & eff.\ $\eta^S$ & eff.\ $\eta^X$ & eff.\ $\eta^{\bar X}$ & accepted $\bar X$/$X$/$S$ & three-stage\\", r"\midrule"]
for r in [8, 16, 24, 32, 40, 48, 56, 62, 64, 80, 96, 128]:
    s_ = E1[f"pod_galerkin_r{r}_n800"]; ts = s_["three_stage"][eN]
    acc = f"{s_['reduced']['accepted'][eN]}/{s_['banded']['accepted'][eN]}/{s_['exact']['accepted'][eN]}"
    st = "/".join(str(ts.get(k, 0)) for k in ("reduced_accept", "reject_cheap", "banded_accept", "exact_accept", "exact_reject"))
    lines.append(f"{r} & {fmt(s_['median_true_err'])} & {s_['exact']['median_effectivity']:.1f} & {s_['banded']['median_effectivity']:.1f} & "
                 f"{s_['reduced']['median_effectivity_at_eps'][eN]:.0f} & {acc} & {st}\\\\")
lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
(TAB / "tab_pod_ranks_v9.tex").write_text("\n".join(lines) + "\n")

# ------------------------------------------------ POD-NN (validation30, N = 1024)
P = json.load(open(RES / "E2_4_podnn_N1024.json"))["results"]
def row(name, label):
    s = P[name]["summary"]
    return (f"{label} & {fmt(s['median_true_err'])} & {fmt(s['exact_AI']['median'])} & {fmt(s['banded_AI']['median'])} & "
            f"{fmt(s['reduced']['median'])} & {s['exact_AI']['median_effectivity']:.1f}\\\\")
lines = [r"\begin{table}[!htbp]", r"\centering\small",
 r"\caption{Neural reduced candidates (POD-NN, $r=48$) on the 30 validation queries at the production configuration: "
 r"median true error and median augmented bounds of the three certificates, with the effectivity of the exact one. "
 r"Standard output: $r\times N_T$ coefficients; space--time output: $r\times r_t$ coefficients of a temporal POD "
 r"($r_t=23$). Checkpoints selected by the validation median of the reduced certificate. The projection of the "
 r"reference trajectories onto the same basis and the POD--Galerkin solution are the attainable floor. "
 r"At $\varepsilon_N=1.18\times10^{-3}$ the certificate rejects every POD-NN query, each by the cheap rejection of "
 r"the first stage.}",
 r"\label{tab:podnn}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrr}", r"\toprule",
 r"candidate & error & $\eta^S_{\rm aug}$ & $\eta^X_{\rm aug}$ & $\eta^{\bar X}_{\rm aug}$ & eff.\ $\eta^S$\\", r"\midrule",
 row("podnn_N1024_r48_n800_lamR0_seed0", "800 traj., $\\lambda_R=0$"),
 row("podnn_N1024_r48_n800_lamR1_seed0", "800 traj., $\\lambda_R=1$"),
 row("podnn_N1024_r48_n4000_lamR0_seed0", "4000 traj., $\\lambda_R=0$"),
 row("podnn_N1024_r48_n4000_lamR0.01_seed0", "4000 traj., $\\lambda_R=10^{-2}$"),
 row("podnn_N1024_r48_n4000_lamR0.1_seed0", "4000 traj., $\\lambda_R=10^{-1}$"),
 row("podnn_N1024_r48_n4000_lamR1_seed0", "4000 traj., $\\lambda_R=1$, seed 0"),
 row("podnn_N1024_r48_n4000_lamR1_seed1", "\\quad seed 1"),
 row("podnn_N1024_r48_n4000_lamR1_seed2", "\\quad seed 2"),
 row("podnn_N1024_r48_n4000_lamR1_seed0_spacetime", "space--time, 4000 traj., $\\lambda_R=1$, seed 0"),
 row("podnn_N1024_r48_n4000_lamR1_seed1_spacetime", "\\quad seed 1"),
 row("podnn_N1024_r48_n4000_lamR1_seed2_spacetime", "\\quad seed 2"),
 r"\midrule",
 row("projection|n4000", "projection of the references"),
 row("pod_galerkin|n4000", "POD--Galerkin, $r=48$"),
 r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
(TAB / "tab_podnn_v9.tex").write_text("\n".join(lines) + "\n")

# ------------------------------------------------ training details
L = json.load(open(RES / "training_log.json"))
runs = {r["name"]: r for r in L["runs"]}
def hours(names): 
    w = [runs[n]["wall_s"] / 3600 for n in names]; return f"{min(w):.1f}--{max(w):.1f}"
std4000 = [n for n in runs if "n4000" in n and "spacetime" not in n]; std800 = [n for n in runs if "n800" in n]
st = [n for n in runs if "spacetime" in n]
lines = [r"\begin{table}[!htbp]", r"\centering\small",
 r"\caption{POD-NN training details (production configuration). MLP $6\to256\to256\to256\to$ output with SiLU, "
 r"double precision, inputs $(\mu,T)$ standardised, $\alpha^0$ fixed to the projection of $z_h^0$; loss = relative "
 r"least squares in the reduced coordinates $+\lambda_R\,\eta^{\bar X}$ (reduced online certificate, differentiable); "
 r"Adam, learning rate $10^{-3}$ with cosine decay to $10^{-5}$, 2000 epochs, batch 64, validation every 10 epochs "
 r"on the 30-query set, checkpoint = epoch with the smallest median reduced bound. Intel Core Ultra 9 285K, 3 threads per run.}",
 r"\label{tab:training}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrr}", r"\toprule",
 r"variant & output & parameters & runs & wall time per run (h)\\", r"\midrule",
 f"standard, 800 trajectories & $r\\times N_T=11\\,520$ & $3.09$~M & $\\lambda_R\\in\\{{0,10^{{-2}},10^{{-1}},1\\}}$, seed 0 & {hours(std800)}\\\\",
 f"standard, 4000 trajectories & $r\\times N_T=11\\,520$ & $3.09$~M & $\\lambda_R\\in\\{{0,10^{{-2}},10^{{-1}},1\\}}$, seed 0; $\\lambda_R=1$, seeds 1--2 & {hours(std4000)}\\\\",
 f"space--time, 4000 trajectories & $r\\times r_t=1\\,104$ & $0.42$~M & $\\lambda_R=1$, seeds 0--2 & {hours(st)}\\\\",
 r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
(TAB / "tab_training_v9.tex").write_text("\n".join(lines) + "\n")
print(open(TAB / "tab_pod_ranks_v9.tex").read()[-900:])
