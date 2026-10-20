"""exp06 — false accept/reject rates across a sweep of tolerances.

For each tolerance tol: true_accept_i = (E_i <= tol), cert_accept_i =
(eta_guar_i <= tol). false_accept = cert_accept AND NOT true_accept (MUST
be 0 for the guaranteed certificate -- it follows structurally from
eta_guar_i >= E_i for every i, already verified in exp02: if eta_guar <=
tol then E_i <= eta_guar <= tol, so true_accept holds automatically).
false_reject = NOT cert_accept AND true_accept (allowed; expected, since
I_eff = eta_guar/E_i > 1 typically -- many false rejects indicate an
inefficient, overly conservative certificate, not a wrong one).

Reuses results/exp02_effectivity.h5's already-computed (E_i, eta_guar)
arrays rather than resolving every query from scratch -- same underlying
computation, no need to repeat the expensive PCG certificate runs.

Outputs: results/exp06_false_accept_reject.h5,
figures/exp06_false_accept_reject.png,
tables/exp06_false_accept_reject.tex.
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import io as common_io
from exp_common import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, fmt, write_h5, write_latex_table

EXP02_PATH = RESULTS_DIR / "exp02_effectivity.h5"
N_TOL = 60


def main() -> None:
    if not EXP02_PATH.exists():
        raise FileNotFoundError(f"{EXP02_PATH} not found -- run experiments/exp02_effectivity.py first.")

    arrays = common_io.read_arrays(EXP02_PATH)
    E_i = arrays["E_i"]
    eta_guar = arrays["eta_guar"]
    eta_sharp = arrays["eta_sharp"]
    n = len(E_i)

    tol_lo = min(E_i.min(), eta_guar.min()) * 0.5
    tol_hi = max(E_i.max(), eta_guar.max()) * 2.0
    tols = np.geomspace(tol_lo, tol_hi, N_TOL)

    n_true_accept, n_cert_accept = [], []
    n_false_accept, n_false_reject = [], []
    n_false_accept_sharp = []  # eta_sharp is NOT guaranteed -- expect this to be > 0 sometimes

    for tol in tols:
        true_accept = E_i <= tol
        cert_accept = eta_guar <= tol
        sharp_accept = eta_sharp <= tol

        n_true_accept.append(int(true_accept.sum()))
        n_cert_accept.append(int(cert_accept.sum()))
        n_false_accept.append(int(np.sum(cert_accept & ~true_accept)))
        n_false_reject.append(int(np.sum(~cert_accept & true_accept)))
        n_false_accept_sharp.append(int(np.sum(sharp_accept & ~true_accept)))

    n_true_accept = np.array(n_true_accept)
    n_cert_accept = np.array(n_cert_accept)
    n_false_accept = np.array(n_false_accept)
    n_false_reject = np.array(n_false_reject)
    n_false_accept_sharp = np.array(n_false_accept_sharp)

    total_false_accept_guar = int(n_false_accept.sum())
    total_false_accept_sharp = int(n_false_accept_sharp.sum())

    print("=" * 78)
    if (n_false_accept == 0).all():
        print(f"GUARANTEED CERTIFICATE: 0 false accepts at every one of {N_TOL} tolerances. PASS.")
    else:
        print("!" * 78)
        print(f"!!! GUARANTEED CERTIFICATE FALSE-ACCEPTED at "
              f"{int((n_false_accept > 0).sum())}/{N_TOL} tolerance levels "
              f"(total {total_false_accept_guar} false accepts) !!!")
        print("!!! This directly contradicts exp02's eta_guar >= E_i check -- if exp02 passed")
        print("!!! and this fails, the two scripts disagree and BOTH need re-auditing before")
        print("!!! trusting either. Do not report this certificate as guaranteed. !!!")
        print("!" * 78)
    print("=" * 78)
    print(f"(informational, not guaranteed) eta_sharp false accepts across the sweep: "
          f"{total_false_accept_sharp}")
    print(f"max false-reject count (most conservative tol): {n_false_reject.max()}/{n}")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    write_h5(
        RESULTS_DIR / "exp06_false_accept_reject.h5",
        {
            "tol": tols, "n_true_accept": n_true_accept, "n_cert_accept": n_cert_accept,
            "n_false_accept_guar": n_false_accept, "n_false_reject_guar": n_false_reject,
            "n_false_accept_sharp": n_false_accept_sharp,
        },
        attrs={"n_queries": n, "total_false_accept_guar": total_false_accept_guar,
               "total_false_accept_sharp": total_false_accept_sharp},
    )

    # ── Figure: ROC-style curve over tol ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(tols, n_true_accept / n, label="true accept rate", lw=2)
    ax1.plot(tols, n_cert_accept / n, label=r"$\eta_{guar}$ accept rate", ls="--")
    ax1.plot(tols, n_false_reject / n, label="false reject rate", ls=":", color="C3")
    ax1.set_xscale("log")
    ax1.set_xlabel("tolerance")
    ax1.set_ylabel("fraction of queries")
    ax1.set_title("Accept rates vs. tolerance")
    ax1.legend(fontsize=8)

    # Genuinely ROC-style: certificate-accept rate (y) vs true-accept rate (x)
    # as tol sweeps -- since eta_guar >= E_i always, this curve must stay on
    # or below the true-accept rate at every tol (equivalently, never above
    # the y=x diagonal would mean 0 false accepts; it can fall below the
    # diagonal, reflecting false rejects).
    order = np.argsort(n_true_accept)
    ax2.plot(n_true_accept[order] / n, n_cert_accept[order] / n, "o-", ms=3)
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label=r"$y=x$ (zero false accept/reject)")
    ax2.set_xlabel("true accept rate")
    ax2.set_ylabel(r"$\eta_{guar}$ accept rate")
    ax2.set_title("Certificate accept rate vs. true accept rate")
    ax2.legend(fontsize=8)

    fig.suptitle(f"exp06: false accept/reject over {N_TOL} tolerances (n={n} queries)")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp06_false_accept_reject.png"
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")

    # ── LaTeX table (summary at a few representative tolerances) ───────────
    header = ["tol", "true accept", "eta_guar accept", "false accept (guar)", "false reject (guar)"]
    rows = []
    for idx in np.linspace(0, N_TOL - 1, 8, dtype=int):
        rows.append([
            fmt(tols[idx]), str(n_true_accept[idx]), str(n_cert_accept[idx]),
            str(n_false_accept[idx]), str(n_false_reject[idx]),
        ])
    tex_path = TABLES_DIR / "exp06_false_accept_reject.tex"
    write_latex_table(tex_path, header, rows)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
