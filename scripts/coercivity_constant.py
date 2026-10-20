#!/usr/bin/env python3
"""Plan 1.6: the explicit continuum coercivity constant c_1 on the box from the Levy-Khintchine form,
c_1 = inf_{mu in box, xi} Re A_mu(xi) / (1 + |M - i xi|^Y + |G + i xi|^Y)   (v8 eq. (2.9) with q = Y/2),
evaluated numerically (vertices + Sobol, fine xi grid), together with the analytic lower bound of
Remark 3.3 of the note.  Writes results/coercivity_c1.json."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from cgmy_cert.params import Query, box_vertices, sobol_queries, BOX
from cgmy_cert.symbols import symbol_A

xi = np.concatenate([[0.0], np.geomspace(1e-4, 1e6, 20001)])

def ratio(q):
    A = symbol_A(xi, q).real
    w = 1.0 + np.abs(q.M - 1j * xi) ** q.Y + np.abs(q.G + 1j * xi) ** q.Y
    r = A / w
    j = int(np.argmin(r))
    return float(r[j]), float(xi[j]), float(A.min())

def analytic_lb(q):
    """Remark 3.3: Re A >= r + C [I_G + I_M] with I_a(xi) >= (2/pi^2) gamma_Y a^{Y-2} xi^2 (|xi|<=a),
    I_a(xi) >= 2 e^{-pi} / ((2-Y) pi^Y) |xi|^Y (|xi| >= a); minimise the resulting ratio over xi."""
    Y = q.Y
    gY = quad(lambda u: u ** (1 - Y) * np.exp(-u), 0, np.pi)[0]
    def I_lb(a):
        return np.where(np.abs(xi) <= a, 2 / np.pi ** 2 * gY * a ** (Y - 2) * xi ** 2,
                        2 * np.exp(-np.pi) / ((2 - Y) * np.pi ** Y) * np.abs(xi) ** Y)
    lb = q.r + q.C * (I_lb(q.G) + I_lb(q.M))
    w = 1.0 + np.abs(q.M - 1j * xi) ** Y + np.abs(q.G + 1j * xi) ** Y
    return float(np.min(lb / w))

qs = box_vertices() + sobol_queries(512, seed=3, include_T=False)
rows = [(ratio(q), analytic_lb(q), q) for q in qs]
worst = min(rows, key=lambda t: t[0][0])
worst_lb = min(rows, key=lambda t: t[1])
print(f"numerical c_1 over {len(qs)} queries: {worst[0][0]:.4e} at xi={worst[0][1]:.3g}, mu={worst[2].mu}")
print(f"analytic (Remark 3.3) c_1: {worst_lb[1]:.4e} at mu={worst_lb[2].mu}   (v8 Appendix A: 1.6e-8)")
# check ReA >= r everywhere (Lemma 3.1)
assert all(t[0][2] >= t[2].r - 1e-12 for t in rows)
json.dump(dict(c1_numerical=worst[0][0], xi_at_min=worst[0][1], mu_at_min=worst[2].mu,
               c1_analytic_remark33=worst_lb[1], mu_at_min_analytic=worst_lb[2].mu, v8_c1=1.6e-8,
               n_queries=len(qs), xi_grid="0 and 20001 geometric points in [1e-4, 1e6]"),
          open(ROOT / "results" / "coercivity_c1.json", "w"), indent=1)
