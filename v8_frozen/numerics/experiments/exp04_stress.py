"""exp04 — parameter-edge / stress tests, focused on Y (jump activity, the
central CGMY difficulty). 2D slices (Y,C), (Y,G), (Y,M), (Y,T); at each
grid point, records true error, certificate, effectivity, and accept/
reject. Heat maps.

Grid note: the OTHER three parameters (not in the current slice's pair)
are held at the parameter box's midpoint.

Accept/reject tolerance note: configs/base.yaml's frozen
certification.acceptance_tolerance (1e-3) is far below the error/
certificate magnitudes actually observed anywhere in this project so far
(E_i, eta ~ 0.1-20 in exp01-03) -- using it alone would show "reject"
everywhere, which is honest but not informative for a heat map. This
script reports BOTH: the frozen config tolerance (as configured, no
silent override), and a separate, clearly-labeled "calibrated" tolerance
(the median eta_guar observed in this experiment) purely for a legible
illustration of the accept/reject boundary. The calibrated value is NOT
written back to configs/base.yaml.

Scale note: 6x6 grid per slice (36 points x 4 slices = 144 reference
solves), reduced from a paper-scale resolution for wall-clock reasons.

Outputs: results/exp04_stress.h5, figures/exp04_stress.png,
tables/exp04_stress.tex.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cgmy.reference_solver import CGMYParameters
from exp_common import (
    FIGURES_DIR,
    FINAL_MODEL_NAME,
    RESULTS_DIR,
    TABLES_DIR,
    compute_certificates,
    fmt,
    get_box_and_stats,
    get_ops,
    load_models,
    predict,
    solve_reference,
    true_error_Ei,
    write_h5,
    write_latex_table,
)

GRID_N = 6
SLICE_DIMS = [("C", "Y"), ("G", "Y"), ("M", "Y"), ("T", "Y")]  # (x, y) axis of each 2D slice
FROZEN_TOL = None  # filled from configs/base.yaml in main()


def main() -> None:
    ops = get_ops()
    box, stats = get_box_and_stats()
    models = load_models(ops, stats)
    model = models[FINAL_MODEL_NAME]

    import yaml
    from exp_common import CONFIG_PATH
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    frozen_tol = cfg["certification"]["acceptance_tolerance"]

    midpoint = {
        "C": 0.5 * (box.C[0] + box.C[1]), "G": 0.5 * (box.G[0] + box.G[1]),
        "M": 0.5 * (box.M[0] + box.M[1]), "Y": 0.5 * (box.Y[0] + box.Y[1]),
        "r": 0.5 * (box.r[0] + box.r[1]), "T": 0.5 * (box.maturity[0] + box.maturity[1]),
    }
    bounds = {"C": box.C, "G": box.G, "M": box.M, "Y": box.Y, "r": box.r, "T": box.maturity}

    slice_results = {}
    all_eta_guar = []

    print(f"exp04: {len(SLICE_DIMS)} slices x {GRID_N}x{GRID_N} grid points...")
    for xdim, ydim in SLICE_DIMS:
        xs = np.linspace(*bounds[xdim], GRID_N)
        ys = np.linspace(*bounds[ydim], GRID_N)
        E_i_grid = np.full((GRID_N, GRID_N), np.nan)
        eta_guar_grid = np.full((GRID_N, GRID_N), np.nan)
        I_eff_grid = np.full((GRID_N, GRID_N), np.nan)
        spd_grid = np.zeros((GRID_N, GRID_N), dtype=bool)

        for i, xv in enumerate(xs):
            for j, yv in enumerate(ys):
                params = dict(midpoint)
                params[xdim] = xv
                params[ydim] = yv
                mu = CGMYParameters(C=params["C"], G=params["G"], M=params["M"],
                                     Y=params["Y"], r=params["r"], T=params["T"])
                try:
                    traj = solve_reference(mu)
                except Exception as exc:  # noqa: BLE001 -- log, don't crash the sweep
                    print(f"  [{xdim},{ydim}] x={xv:.3f} y={yv:.3f}: solve failed: {exc}")
                    continue

                pred = predict(model, mu, ops, stats, traj.initial_coefficients, traj.time_grid)
                E_i = true_error_Ei(pred, traj.coefficients, ops)
                cert = compute_certificates(traj, pred, mu, ops)

                E_i_grid[i, j] = E_i
                if cert.is_spd:
                    spd_grid[i, j] = True
                    eta_guar_grid[i, j] = cert.eta_guar
                    I_eff_grid[i, j] = cert.eta_guar / E_i if E_i > 0 else np.nan
                    all_eta_guar.append(cert.eta_guar)

        slice_results[(xdim, ydim)] = {
            "xs": xs, "ys": ys, "E_i": E_i_grid, "eta_guar": eta_guar_grid,
            "I_eff": I_eff_grid, "is_spd": spd_grid,
        }
        print(f"  slice ({xdim},{ydim}) done: "
              f"{int(spd_grid.sum())}/{GRID_N * GRID_N} SPD, "
              f"mean E_i={np.nanmean(E_i_grid):.3e}, mean eta_guar={np.nanmean(eta_guar_grid):.3e}")

    calibrated_tol = float(np.median(all_eta_guar)) if all_eta_guar else float("nan")
    print(f"\nfrozen config tolerance: {frozen_tol:.3e}  (everything will show 'reject' at this scale)")
    print(f"calibrated illustrative tolerance (median eta_guar across this experiment): {calibrated_tol:.3e}")

    # ── HDF5 ─────────────────────────────────────────────────────────────
    arrays = {}
    for (xdim, ydim), res in slice_results.items():
        key = f"{xdim}_{ydim}"
        arrays[f"{key}_xs"] = res["xs"]
        arrays[f"{key}_ys"] = res["ys"]
        arrays[f"{key}_E_i"] = res["E_i"]
        arrays[f"{key}_eta_guar"] = res["eta_guar"]
        arrays[f"{key}_I_eff"] = res["I_eff"]
        arrays[f"{key}_is_spd"] = res["is_spd"].astype(np.int8)
    write_h5(
        RESULTS_DIR / "exp04_stress.h5", arrays,
        attrs={"grid_n": GRID_N, "model": FINAL_MODEL_NAME, "frozen_tol": frozen_tol,
               "calibrated_tol": calibrated_tol},
    )

    # ── Figure: 4 slices x (E_i, eta_guar, accept/reject@calibrated) heat maps ──
    fig, axes = plt.subplots(len(SLICE_DIMS), 3, figsize=(12, 3.2 * len(SLICE_DIMS)))
    for row, (xdim, ydim) in enumerate(SLICE_DIMS):
        res = slice_results[(xdim, ydim)]
        extent = [res["xs"][0], res["xs"][-1], res["ys"][0], res["ys"][-1]]

        ax = axes[row, 0]
        im = ax.imshow(res["E_i"].T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax.set_title(f"true error $E_i$  ({xdim},{ydim})")
        ax.set_xlabel(xdim); ax.set_ylabel(ydim)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[row, 1]
        im = ax.imshow(res["eta_guar"].T, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax.set_title(r"$\eta_{guar}$" + f"  ({xdim},{ydim})")
        ax.set_xlabel(xdim); ax.set_ylabel(ydim)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[row, 2]
        accept = (res["eta_guar"] <= calibrated_tol).astype(float)
        accept[~res["is_spd"]] = np.nan
        im = ax.imshow(accept.T, origin="lower", extent=extent, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"accept @ calibrated tol={calibrated_tol:.2e}")
        ax.set_xlabel(xdim); ax.set_ylabel(ydim)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"exp04: stress heat maps ({FINAL_MODEL_NAME}, {GRID_N}x{GRID_N} grid/slice)")
    fig.tight_layout()
    fig_path = FIGURES_DIR / "exp04_stress.png"
    fig.savefig(fig_path, dpi=130)
    print(f"wrote {fig_path}")

    # ── LaTeX table: per-slice summary ──────────────────────────────────────
    header = ["slice", "mean $E_i$", "mean $\\eta_{guar}$", "mean $I_{eff}$", "frac accept (calibrated)", "frac SPD"]
    rows = []
    for (xdim, ydim), res in slice_results.items():
        frac_accept = float(np.nanmean((res["eta_guar"] <= calibrated_tol).astype(float)))
        rows.append([
            f"({xdim},{ydim})", fmt(np.nanmean(res["E_i"])), fmt(np.nanmean(res["eta_guar"])),
            fmt(np.nanmean(res["I_eff"])), fmt(frac_accept), fmt(res["is_spd"].mean()),
        ])
    tex_path = TABLES_DIR / "exp04_stress.tex"
    write_latex_table(tex_path, header, rows)
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
