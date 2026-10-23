# certified-CGMY-Galerkin, v9 reproducibility archive

*Symbol-based residual certification of reduced and neural surrogates for tempered-fractional CGMY pricing.*
Production configuration (`results/PRODUCTION_CONFIG.json`): N = 1024, N_T = 240, K = 100, eps_N = 1.18e-3.

## Layout

```
cgmy_cert/        the package: params, symbols (assembly), spectral (spectral bounds; fine and fast mode), riesz, direct,
                  certificate, eim, pod, reduced_cert, three_stage, podnn, timing, provenance, figio, highprec
tests/            pytest suite (+ tests/fixtures: 40-digit real-line fixture, query populations)
scripts/          compute scripts (run_*.py, gen_references.py, train_podnn.py, sweep_N1024.py, eval_podnn.py, ...)
                  and the JSON-only plot/table step build_outputs.py (called by build.sh)
results/          every number: *.json (self-describing, with provenance), DEVIATIONS.md, NOTES_FOR_PAPER.md,
                  PHASE{2,3,4}_REPORT.md, MANIFEST.json, ARCHIVE_CHECK.json, NUMBER_CHECK.md
figures_v9/       F2-F8, PNG (dpi 200) and PDF (vector, fonts embedded)
tables_v9/        LaTeX tables
v8_frozen/        read-only v8 reference material (only in the git repository, not in the zip)
manuscript_v9/    the paper (read-only here; only in the git repository)
data/, checkpoints/   NOT in the archive (27 GB): hashes, sizes and regeneration commands in results/MANIFEST.json -> excluded
```

## Environment

Python 3.10.12, numpy 1.26.4, scipy 1.15.3, mpmath 1.3.0, matplotlib 3.10.9, pandas 2.3.3, torch 2.14.0 (CPU), PyYAML 6.0.2,
h5py 3.16.0 (`requirements.txt`). Results were produced on an Intel Core Ultra 9 285K (24 cores; performance cores 0-7),
Linux 6.8. All timings: one thread (`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(1)`),
pinned to performance core 3, idle machine (load averages stored in each JSON), protocol in `cgmy_cert/timing.py`.

```
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Commands

| What | Command |
|---|---|
| every table and figure from `results/*.json` (no solver, no data) | `bash build.sh` (or `make all`) |
| quick tests / full tests | `python3 -m pytest tests -q` / `CGMY_FULL_TESTS=1 python3 -m pytest tests -q` |
| clean-clone check (build + E0 + tests) | `python3 scripts/archive_check.py` |
| number check against the manuscript | `python3 scripts/check_numbers.py` |
| E0 (verification) | `python3 scripts/run_E0.py --nproc 1` |
| EIM (Phase 2.1) | `python3 scripts/run_eim.py build select validate figure retol qtarget` |
| references | `python3 scripts/gen_references.py --N 1024 --NT 240 --extra bench120 ref30` |
| N = 1024 pipeline (POD, cells, tensors, checks) | `python3 scripts/run_N1024.py pod eimq cells tensors checks table podext tensors_ext table_ext timing_ext` |
| calibration (3.1) | `python3 scripts/run_E3_1.py cos cm galerkin summary table` |
| POD-NN training sweep | `CGMY_N=1024 python3 scripts/train_podnn.py prep && python3 scripts/sweep_N1024.py` |
| cost model | `python3 scripts/run_cost_model.py` |
| E1 / E2 / E3 / E4 | `python3 scripts/run_E1.py qeps tensors eval select timing figs` / `run_E2.py bench refine figs` / `run_E3.py run summary fig breakdown` / `run_E4.py run summary` |

## Paper item -> result file -> script

| Output | Result file | Compute script | Plot/table step (in `build_outputs.py`) |
|---|---|---|---|
| `tables_v9/tab_assembly_v9.tex` (tab:assembly-v9) | `E0.json` (E0.1) | `run_E0.py E0.1` | `run_E0.plot_e01` |
| `figures_v9/F2_spectral_bound` (fig:spectral-bound) | `E0.json` (E0.2) | `run_E0.py E0.2` | `run_E0.plot_e02` |
| `figures_v9/F3_theta_and_effectivity` (fig:theta) | `E0.json` (E0.3) | `run_E0.py E0.3` | `run_E0.plot_e03` |
| `tables_v9/tab_pcg_v9.tex` (tab:pcg-v9) | `E0.json` (E0.4) | `run_E0.py E0.4` | `run_E0.plot_e04` |
| `figures_v9/F4_eim` | `eim.json` | `run_eim.py` | `run_eim.do_figure` |
| `tables_v9/tab_budget_v9.tex` (tab:budget) | `E3_1_calibration.json` | `run_E3_1.py` | `run_E3_1.do_table` |
| `tables_v9/tab_cost_v9.tex` (tab:cost) | `cost_model.json` | `run_cost_model.py` | `run_cost_model.write_table` |
| `figures_v9/F5a_acceptance` | `E1.json` | `run_E1.py eval select` | `fig_F5a.main` |
| `figures_v9/F5b_effectivity`, `F5c_online_time`, `tables_v9/tab_pod_ranks_v9.tex` (tab:pod_ranks) | `E1.json` | `run_E1.py eval timing` | `run_E1.do_figs` |
| `figures_v9/F6a_benchmark`, `F6b_refinement`, `tables_v9/tab_benchmark_v9.tex` (tab:benchmark) | `E2.json` | `run_E2.py bench refine` | `run_E2.do_figs` |
| `figures_v9/F7_ood` | `E3.json`, `E1.json` | `run_E3.py run summary` | `run_E3.do_fig` |
| `tables_v9/tab_ood_v9.tex` (tab:ood) | `E3.json` (breakdown) | `run_E3.py breakdown` | `run_E3.do_table_ood` |
| `figures_v9/F8_amortisation` | `E4.json`, `E4_progress.json` | `run_E4.py run summary` | `run_E4.plot_f8` |

Paper table/figure numbers: to be filled in against the current manuscript (the copy in `manuscript_v9/` here is the
Phase 1 skeleton; see `results/PHASE4_REPORT.md`).
