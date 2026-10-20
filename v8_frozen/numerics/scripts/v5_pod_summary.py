"""Summaries for an M3 POD study run, rebuilt only from its raw CSVs and basis
artifacts (no hand-entered numbers).

    PYTHONPATH=src:experiments:scripts python3 scripts/v5_pod_summary.py <run_dir>

Writes <run_dir>/rank_summary.csv, <run_dir>/training_projection.csv,
<run_dir>/recovery_summary.csv and <run_dir>/summary.md.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "experiments", "scripts"):
    sys.path.insert(0, str(REPO_ROOT / sub))

from common.methods import NUMERICAL_SCOPE  # noqa: E402

RANKS = (8, 16, 32, 48, 62)
METRICS = ("projection_error", "initial_error", "evolution_error", "total_error", "E_rel",
           "terminal_price_error_vs_stored_reference", "cand_augmented_certificate")


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def training_projection(run_dir: Path) -> list[dict]:
    """Aggregate training-snapshot projection error in the M_I norm, both bases
    (the quantity mass-weighted POD minimises)."""
    from cgmy.operators import OperatorCache
    from certification.constrained import FreeOperatorCache, free_space_for_mesh
    from common import run_manifest as rm
    from data.sampling import load_mesh_and_time_grid
    from generate_production_dataset import load_all_train_samples
    from learning import pod_free as pf
    from pipeline.predict_certify_correct import build_mesh_info

    mesh, _ = load_mesh_and_time_grid(rm.BASE_CONFIG)
    fo = FreeOperatorCache(OperatorCache(mesh, mesh.x_min, mesh.x_max), free_space_for_mesh(build_mesh_info(mesh)))
    idx = fo.fs.free_idx
    M = fo.ops.M_h[idx, :][:, idx].toarray()
    trajs = load_all_train_samples(REPO_ROOT / "data" / "production_dataset.h5", 800)
    Z = np.concatenate([t.coefficients[:, idx] for _, _, t in trajs], axis=0).T
    tot = float(np.sum(Z * (M @ Z)))
    z0 = Z[:, 0]
    rows = []
    for name, btype in (("pod_basis_mass_weighted.npz", pf.BASIS_MASS_WEIGHTED),
                        ("pod_basis_euclidean_diagnostic.npz", pf.BASIS_EUCLIDEAN_DIAGNOSTIC)):
        full, _ = pf.load_pod_artifact(run_dir / "basis" / name, fo, rank=fo.n_dof, expected_basis_type=btype)
        for r in RANKS:
            V = full.modes[:, :r]
            R = Z - V @ (V.T @ (M @ Z))
            e = float(np.sum(R * (M @ R)))
            d0 = z0 - V @ (V.T @ (M @ z0))
            rows.append({"basis_type": btype, "rank": r, "training_sum_sq_mass_proj_error": e,
                         "training_rel_mass_proj_error": float(np.sqrt(e / tot)),
                         "initial_payoff_vector_mass_proj_error": float(np.sqrt(d0 @ M @ d0))})
    return rows


def main(run_dir: Path) -> None:
    rank_rows = []
    status_lines = []
    for sub in ("sweep_mass_weighted", "sweep_euclidean_diagnostic"):
        rows = read(run_dir / sub / "pod_rank_sweep.csv")
        btype = rows[0]["basis_type"]
        for r in RANKS:
            rr = [x for x in rows if int(x["rank"]) == r]
            out = {"basis_type": btype, "rank": r, "n_queries": len(rr)}
            for m in METRICS:
                v = np.array([float(x[m]) for x in rr if x[m] != ""])
                out[f"{m}_median"] = float(np.median(v))
                out[f"{m}_max"] = float(np.max(v))
            st = Counter(x["final_status"] for x in rr)
            out["n_accepted_direct"] = st.get("accepted_direct", 0)
            out["n_accepted_after_correction"] = st.get("accepted_after_correction", 0)
            out["n_other_status"] = len(rr) - out["n_accepted_direct"] - out["n_accepted_after_correction"]
            out["n_E_rel_zero_denominator"] = sum(1 for x in rr if x["E_rel_flag"])
            out["n_delivered"] = sum(1 for x in rr if x["output_delivered"] == "True")
            for t in ("t_predict_s", "t_certify_candidate_s", "t_correct_s", "t_recertify_s", "t_total_s"):
                out[f"{t}_median"] = float(np.median([float(x[t]) for x in rr]))
            out["pcg_iterations_total_median"] = float(np.median([float(x["pcg_iterations_total"]) for x in rr]))
            out["correction_gmres_iterations_median"] = float(np.median([float(x["correction_gmres_iterations"]) for x in rr]))
            out["reference_allowance_eta_aug_max"] = max(float(x["reference_allowance_eta_aug"]) for x in rr)
            out["reference_allowance_price_max"] = max(float(x["reference_allowance_price"]) for x in rr)
            rank_rows.append(out)
            status_lines.append(f"{btype} r={r}: {dict(st)}")
    write(run_dir / "rank_summary.csv", rank_rows)
    tp = training_projection(run_dir)
    write(run_dir / "training_projection.csv", tp)

    rec = read(run_dir / "recovery" / "full_rank_recovery.csv")
    groups = {}
    for x in rec:
        groups.setdefault((x["population"], x["source"]), []).append(x)
    rec_rows = []
    for (pop, src), xs in groups.items():
        ratio = [float(x["mass_max_error"]) / float(x["allowance_from_residuals"]) for x in xs]
        rec_rows.append({
            "population": pop, "source": src, "n": len(xs),
            "n_within_allowance": sum(x["within_allowance"] == "True" for x in xs),
            "max_coef_max_abs_error": max(float(x["coef_max_abs_error"]) for x in xs),
            "max_coef_rel_error": max(float(x["coef_rel_error"]) for x in xs),
            "max_mass_max_error": max(float(x["mass_max_error"]) for x in xs),
            "max_mass_rel_error": max(float(x["mass_rel_error"]) for x in xs),
            "max_error_over_allowance": max(ratio),
            "max_direct_residual_max_abs": max(float(x["direct_residual_max_abs"]) for x in xs),
            "max_direct_eta_sharp": max(float(x["direct_eta_sharp"]) for x in xs),
            "max_cond_F": max(float(x["cond_F"]) for x in xs), "cond_V": float(xs[0]["cond_V"]),
        })
    write(run_dir / "recovery_summary.csv", rec_rows)

    def fmt(v):
        return f"{v:.3e}" if isinstance(v, float) else str(v)

    md = [f"# M3 POD study {run_dir.name}", "", f"Scope: {NUMERICAL_SCOPE}", "",
          "All numbers below are rebuilt by scripts/v5_pod_summary.py from the raw CSVs in this directory.",
          "Errors are vs the STORED reference (trusted GMRES solve, projected); its own augmented "
          "certificate is the reference allowance column. E_rel = max_n ||z_pod - z_ref||_{M_I} / "
          "max_n ||z_ref||_{M_I}; not comparable to the legacy time-averaged absolute rank-sweep error.", "",
          "## Rank sweep (validation30, tolerance from configs/v5_free_workflow.yaml)", "",
          "| basis | rank | proj (med/max) | init β₀ (med) | evol (med/max) | E_rel (med/max) | price err $ (med/max) | cand η_aug (med) | direct acc. | corrected acc. | other | t_total s (med) | ref allowance η / $ (max) |",
          "|" + "---|" * 13]
    for o in rank_rows:
        md.append("| " + " | ".join([
            "mass-weighted" if o["basis_type"] == "mass_weighted_pod" else "Euclidean diag.", str(o["rank"]),
            f"{fmt(o['projection_error_median'])} / {fmt(o['projection_error_max'])}", fmt(o["initial_error_median"]),
            f"{fmt(o['evolution_error_median'])} / {fmt(o['evolution_error_max'])}",
            f"{fmt(o['E_rel_median'])} / {fmt(o['E_rel_max'])}",
            f"{fmt(o['terminal_price_error_vs_stored_reference_median'])} / {fmt(o['terminal_price_error_vs_stored_reference_max'])}",
            fmt(o["cand_augmented_certificate_median"]), str(o["n_accepted_direct"]),
            str(o["n_accepted_after_correction"]), str(o["n_other_status"]), fmt(o["t_total_s_median"]),
            f"{fmt(o['reference_allowance_eta_aug_max'])} / {fmt(o['reference_allowance_price_max'])}"]) + " |")
    md += ["", "## Training-snapshot projection error in the M_I norm (what mass-weighted POD minimises)", "",
           "| basis | rank | relative aggregate error | error on the (common) initial payoff vector z_h^0 |", "|---|---|---|---|"]
    for o in tp:
        md.append(f"| {o['basis_type']} | {o['rank']} | {fmt(o['training_rel_mass_proj_error'])} | {fmt(o['initial_payoff_vector_mass_proj_error'])} |")
    md += ["", "## Full-rank (62) recovery vs dense direct free march", "",
           "| population | source | n | within residual allowance | max coef rel err | max M_I err | max err/allowance | max direct residual | max cond(F) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for o in rec_rows:
        md.append(f"| {o['population']} | {o['source']} | {o['n']} | {o['n_within_allowance']} | {fmt(o['max_coef_rel_error'])} | "
                  f"{fmt(o['max_mass_max_error'])} | {fmt(o['max_error_over_allowance'])} | {fmt(o['max_direct_residual_max_abs'])} | {fmt(o['max_cond_F'])} |")
    md += ["", "## Status counts", "", *[f"- {s}" for s in status_lines]]
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
