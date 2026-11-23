# legacy_v8: read-only reference from the earlier (v8) implementation

Slimmed in the Phase 5a cleanup to what this repository uses (the full snapshot `v8_frozen/` is in the git tag
`pre-cleanup`):

| Kept | Used by |
|---|---|
| `numerics/vendor/` (v8 B-spline assembly) | `tests/test_symbols.py` (mass/derivative regression) |
| `numerics/src/`, `numerics/configs/`, `models/deeponet_final.pt`, `models/checkpoint_metadata.json`, `models/training_code_at_checkpoint_commit_13bf0d5/` | `scripts/run_deeponet_old.py`, `tests/test_deeponet_old.py`, claim `s43_deeponet_params` |
| `numerics/validation/` (COS pricer) | `scripts/run_E3_1.py` (calibration reference) |
| `query_sets/`, `fixtures_source/` | `scripts/build_fixtures.py`, `tests/test_symbols.py` |
| `runs/20261007T175335Z_m6_audit459/audit_free/positivity.csv` | `tests/test_symbols.py::test_legacy_regression_audit459` |
| `manuscript_v8/provenance/frozen_tables/tab_corner.tex` | claim `s56_corners_2dollar` |

The original handoff notes follow; paths in them refer to the full snapshot.

---

# Phase 0 handoff bundle (items 0.1 and 0.2)

Built 2026-10-19 from `certified_galkn_cgmy` at branch `v5/m10-manuscript`, HEAD `568e1b5`.
Every file is a copy. Nothing in the source repo was modified.

## READ FIRST: things that are not what the request assumed

1. **There is no v7 or v8 on this machine.** The newest manuscript is **v6**
   (`manuscript/v6/`), and `v7_build_manuscript_assets.py` does not exist. The
   closest script is `scripts/v6_build_manuscript_assets.py`, which is included.
   So "Tables 6–7" and "Table 9" use **your v8 numbering, which I can't check here**.
   In the v6 PDF (copy in `03_tables_data/manuscript_v6/`):
   - T5 Riesz brackets vs 40-digit, T6 augmented bound, T7 trajectory workload (120 queries),
     T8 amortised, T9 terminal-price workload, T10 repaired POD, T12 assembly checks.
   - I included the runs behind all of T4–T10 and T12. Please match them to your v8 numbers.

2. **The "40-digit" data is not all 40 digits, and none of it is stored as 40-digit strings.**
   - **24 eigenvalues** (`05_fixtures_40digit/alpha_check_24_eigenvalues.csv`): computed at
     mpmath `dps=40` (Cholesky/eigsy of the same assembled double `S_I`) in
     `scripts/v5_m6_riesz.py`, but **written to CSV as `float(...)`**, so only about 17
     significant digits survive. To get a fixture straight from the source at full precision,
     re-run that script's eigenvalue step and write `mp.nstr(x, 40)`. That takes minutes on CPU.
     I have not done this.
   - **Real-line Galerkin coefficients** (`runs/20261008T141446Z_m8_coefficient_convention/convention/coefficients.csv`,
     56 rows): column `real_line_quadpack` is **double-precision QUADPACK** (Fourier-weighted on
     [1, inf)). The only mpmath column, `real_line_mpmath_K20`, uses **dps=20** and covers lags
     0, 1, 10 only. Script: `scripts/v5_m8_coefficient_convention_check.py`. This is not a
     40-digit table.
   - **Principal-period "40-digit quadrature"** (independent assembly, v6 T12): the manuscript
     says 40-digit, but `audit/independent_assembly.py:118` sets **`mp.mp.dps = 30`**.
     The code and the manuscript disagree. Reported, not resolved.
   - `spectra.csv` in the same run holds 8 rows × 3 λ_min values (= 24 numbers) from
     **float64 `np.linalg.eigvalsh`**. Don't confuse these with the 24 mpmath eigenvalues above.

3. **The reference trajectories are not stored as arrays.** The benchmark stores only a
   sha256 per trajectory (`runs/*gate_d_v5_full*/reference/stored_reference.csv`,
   column `stored_reference_trajectory_sha256`) and recomputes them with the trusted solver
   (`solve_cgmy`, Toeplitz GMRES rtol 1e-10). You can use the hashes to check a bit-for-bit
   reproduction.

4. **DeepONet: the checkpoint exists, but there is no per-epoch training log.** What exists:
   - `02_deeponet_and_models/deeponet_final.pt`. Embedded metadata (also in
     `checkpoint_metadata.json`): seed 0, n_train 800, train_time 372 s, final train loss
     3.71e-6, val rel-L2 0.430, git sha `13bf0d5` (dirty), dataset manifest hash `d11dfb1a…`,
     N=64, 60 CN steps.
   - The training script **as of commit 13bf0d5** (`training_code_at_checkpoint_commit_13bf0d5/`)
     and the current one (`01_solver_code/scripts/train_final_models.py`). Three later commits
     touched training/provenance (50402d8, 302f4ca, ced996a). The "dirty" flag means the exact
     training source can't be fully guaranteed.
   - No notebook and no loss-history file. The training set `production_dataset.h5` is included.
   - So "retrain in Phase 3" is optional: the checkpoint loads and Tables 7/9 used it.

5. **GPU:** never used. torch is `2.14.0+cpu`, `cuda_available False`. The DeepONet was trained
   in 372 s on CPU (Intel Core Ultra 9 285K, 24 threads).

## Layout

| Dir | Contents |
|---|---|
| `01_solver_code/` | `src/` (operators, CN reference solver, lifting/BC, constrained free problem, residuals, certificate/PCG bracket, Riesz solver, POD, learning, pipeline), `vendor/cgmy_bspline/` (trusted solver, **principal-period assembly**: `toeplitz.py`, `connection_coefficients.py`, `matrices.py`), `audit/`, `validation/` (COS / Carr–Madan), `scripts/` (incl. `v6_build_manuscript_assets.py`, all `v5_*`), `benchmarks/` (`run_gate_d_benchmark_v5.py`), `experiments/`, `configs/`, `tests/` |
| `02_deeponet_and_models/` | `deeponet_final.pt`, `mlp_final.pt`, `projected_fno_final.pt`, legacy `pod_final.npz`, checkpoint metadata, training code at the checkpoint commit, `training/` CSVs |
| `03_tables_data/runs/` | `gate_d_v5_full` (reusable-setup), `gate_d_v5_full_cold` (cold), `m6_riesz_bracket`, `m6_audit459` (positivity + augmented), `m3_pod_study` (**POD bases**: `basis/pod_basis_mass_weighted.npz`, the one pinned in `configs/v5_free_workflow.yaml`, sha256 `4aa18e2e…`) |
| `03_tables_data/datasets/` | `production_dataset.h5`: the DeepONet training set and POD snapshot set. Not included: `operator_dataset.h5` (43 MB, only used by the operator-generalisation experiment) and `debug_dataset.h5`. |
| `03_tables_data/manuscript_v6/` | v6 `main.tex`, PDF, generated `tables/`, `evidence_v6.json`, evidence manifest |
| `04_environment/` | `requirements.txt`, `pyproject.toml`, full `pip_freeze.txt`, `versions.txt` (Python 3.10.12, NumPy 1.26.4, SciPy 1.15.3, torch 2.14.0+cpu, h5py 3.16.0, mpmath 1.3.0) |
| `05_fixtures_40digit/` | see item 2. `scripts/` holds the producing scripts. `runs/` holds the M8 convention and independent-assembly runs. `kappa_investigation.json` (60-digit mpmath κ check). |
| `06_query_sets/` | `Q_audit.csv` (459), `Q_bench120.csv`, `Q_ref30.csv`, `Q_scale.csv`, `Q_ref100.csv`, `Q_validation30.csv`, `Q_price_diag8.csv`, `v5_query_populations.json`, `generate_query_sets.py` |

## Query sets: provenance check

Each file's sha256 was recomputed and matches `v5_query_populations.json`:
audit459 `9d0ef945…`, bench120 `0fff434a…`, ref30 `7a2d94c3…`, scale30 `dffccd3d…`.

- **validation30** is not a stored file. It is *generated* by
  `data.sampling.sample_sobol(load_parameter_box(configs/base.yaml), 30, seed=999_999)`.
  I regenerated it here, and `sha256(queries_csv_text(...))` = `2ca78e02…` matches the
  declaration. It depends on `scipy.stats.qmc.Sobol` (scipy 1.15.3).
- **8 price-diagnostic queries** = the first 8 rows of `Q_ref30` (declared in
  `scripts/v5_m8_refinement.py:8`). These were extracted into `Q_price_diag8.csv`. If your v8
  "8 price-diagnostic" means something else, tell me.
- The 459 audit set is stratified: 139 known_failure, 256 sobol_main_box, 64 exact_vertex.
  The 139 failures were deliberately selected and are not an unbiased failure-rate sample.

## Split into two zips (each < 30 MB)

- `..._part1_of_2.zip`: everything except `03_tables_data/datasets/`.
- `..._part2_of_2.zip`: `03_tables_data/datasets/production_dataset.h5` (DeepONet training
  set / POD snapshot set) plus this README.

Both parts use the top-level folder `phase0_handoff/`. Unzip both into the same directory to
rebuild the full tree. Each part is a normal standalone zip, not a multi-volume archive.
