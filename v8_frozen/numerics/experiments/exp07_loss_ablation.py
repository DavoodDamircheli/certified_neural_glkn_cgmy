"""exp07 — the Stage G loss-ablation comparison, written up for the paper.

Repackages results/lossablation/ablation.csv (already computed and
verified in Stage G -- see docs/issues_report.md items G2/G3 for the
methodology fix that produced these numbers: the first lambda_R sweep was
meaningless because L_res outweighs L_data by ~3.4e6x at initialization,
so the sweep had to be rebracketed around the actual balance point) into
the paper's results/figures/tables convention. No new training is done
here.

Outputs: results/exp07_loss_ablation.h5, figures/exp07_loss_ablation.png,
tables/exp07_loss_ablation.tex.
"""

from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exp_common import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, fmt, write_h5, write_latex_table

ABLATION_CSV = RESULTS_DIR / "lossablation" / "ablation.csv"


def main() -> None:
    if not ABLATION_CSV.exists():
        raise FileNotFoundError(
            f"{ABLATION_CSV} not found -- run scripts/lossablation.py (Stage G) first."
        )

    rows = list(csv.DictReader(open(ABLATION_CSV)))

    regimes = [r["regime"] for r in rows]
    lambda_strs = [r["lambda_R"] for r in rows]
    test_l2 = np.array([float(r["test_l2_mean"]) for r in rows])
    stress_l2 = np.array([float(r["stress_l2_mean"]) for r in rows])
    train_l2 = np.array([float(r["train_l2_mean"]) for r in rows])
    test_eta_sharp = np.array([float(r["test_eta_sharp"]) for r in rows])
    train_time = np.array([float(r["train_time_s"]) for r in rows])
    gen_gap = np.array([float(r["generalization_gap"]) for r in rows])

    # ── HDF5 (numeric columns only; lambda_R stored as NaN for the two
    #    regimes without a numeric weight, with the regime name array
    #    carrying the authoritative label) ─────────────────────────────
    lambda_num = np.array([float(s) if s not in ("0.0", "n/a (no data term)") else
                            (0.0 if s == "0.0" else np.nan) for s in lambda_strs])
    write_h5(
        RESULTS_DIR / "exp07_loss_ablation.h5",
        {
            "regime": np.array(regimes, dtype=object).astype(str),
            "lambda_R": lambda_num,
            "train_l2_mean": train_l2, "test_l2_mean": test_l2, "stress_l2_mean": stress_l2,
            "test_eta_sharp": test_eta_sharp, "train_time_s": train_time,
            "generalization_gap": gen_gap,
        },
        attrs={"source": str(ABLATION_CSV)},
    )

    # ── Figure: test/stress L2 error vs lambda_R, with data_only/residual_only
    #    as reference points on either side of the hybrid sweep ──────────
    hybrid_mask = np.array([s not in ("0.0", "n/a (no data term)") for s in lambda_strs])
    hybrid_lambda = lambda_num[hybrid_mask]
    order = np.argsort(hybrid_lambda)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(hybrid_lambda[order], test_l2[hybrid_mask][order], "o-", label="test L2")
    ax1.plot(hybrid_lambda[order], stress_l2[hybrid_mask][order], "s-", label="stress L2")
    data_only_idx = regimes.index("data_only")
    residual_only_idx = regimes.index("residual_only")
    ax1.axhline(test_l2[data_only_idx], color="C0", ls="--", alpha=0.5, label="data_only test L2")
    ax1.axhline(test_l2[residual_only_idx], color="C2", ls=":", alpha=0.5, label="residual_only test L2")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\lambda_R$")
    ax1.set_ylabel("mean L2 error")
    ax1.set_title("Accuracy vs. residual-loss weight")
    ax1.legend(fontsize=8)

    ax2.plot(hybrid_lambda[order], train_time[hybrid_mask][order], "o-", color="C3")
    ax2.axhline(train_time[data_only_idx], color="C0", ls="--", alpha=0.5, label="data_only")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\lambda_R$")
    ax2.set_ylabel("training time (s)")
    ax2.set_title("Training cost of residual supervision")
    ax2.legend(fontsize=8)

    fig.suptitle("exp07: data-only vs. residual-only vs. hybrid training")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp07_loss_ablation.png"
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")

    # ── LaTeX table ──────────────────────────────────────────────────────
    header = ["regime", r"$\lambda_R$", "train $L_2$", "test $L_2$", "stress $L_2$", r"test $\eta_{sharp}$", "train (s)"]
    table_rows = []
    for r in rows:
        table_rows.append([
            r["regime"], r["lambda_R"], fmt(float(r["train_l2_mean"])), fmt(float(r["test_l2_mean"])),
            fmt(float(r["stress_l2_mean"])), fmt(float(r["test_eta_sharp"])), f"{float(r['train_time_s']):.1f}",
        ])
    tex_path = TABLES_DIR / "exp07_loss_ablation.tex"
    write_latex_table(tex_path, header, table_rows)
    print(f"wrote {tex_path}")

    best_idx = int(np.argmin(test_l2))
    print(f"\nbest test-L2 regime: {regimes[best_idx]} (lambda_R={lambda_strs[best_idx]}), "
          f"test_L2={test_l2[best_idx]:.4e} vs data_only={test_l2[data_only_idx]:.4e}")


if __name__ == "__main__":
    main()
