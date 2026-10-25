#!/usr/bin/env python3
"""Explicit coercivity constant of Appendix A.1, eq. (c1-explicit):
    c_1 = min_{mu in box} min{ r / (1 + 2^{1+Y/2} a_*^Y),  C kappa_Y / 2^{Y/2} },
    a_* = max(M_max, G_max),  kappa_Y = 2 e^{-pi} / ((2 - Y) pi^Y).
Both branches are monotone in (C, r) and the minimum over Y is taken on a fine grid.
Writes results/coercivity_c1_provable.json (replaces an earlier, looser formula)."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
BOX = dict(C=(0.05, 1.0), G=(1.0, 5.0), M=(2.0, 9.0), Y=(1.10, 1.80), r=(0.01, 0.10))
a = max(BOX["M"][1], BOX["G"][1])
Y = np.linspace(*BOX["Y"], 7001)
low = BOX["r"][0] / (1 + 2 ** (1 + Y / 2) * a ** Y)
kap = 2 * np.exp(-np.pi) / ((2 - Y) * np.pi ** Y)
high = BOX["C"][0] * kap / 2 ** (Y / 2)
c = np.minimum(low, high)
j = int(np.argmin(c))
out = dict(formula="c1 = min_box min{ r/(1+2^{1+Y/2} a^Y), C kappa_Y / 2^{Y/2} }, a = max(M_max, G_max), "
                   "kappa_Y = 2 e^{-pi} / ((2-Y) pi^Y)  (Appendix A.1, eq. c1-explicit)",
           c1_provable=float(c[j]), Y_at_min=float(Y[j]), branch="low-frequency" if low[j] <= high[j] else "high-frequency",
           low_branch_min=float(low.min()), high_branch_min=float(high.min()), a_star=a,
           c1_sharp_at_xi0=float(BOX["r"][0] / (1 + BOX["M"][1] ** BOX["Y"][1] + BOX["G"][1] ** BOX["Y"][1])),
           note="supersedes an earlier evaluation (3.06e-5) of a looser bound w <= 1 + 2^{Y/2}(M^Y+G^Y) + 2^{1+Y/2} a^Y")
(ROOT / "results" / "coercivity_c1_provable.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
