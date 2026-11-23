# Symbol-based residual certification of reduced and neural surrogates for tempered-fractional CGMY pricing

Code and results for the article by **Davood Damircheli** (Louisiana State University),
[arXiv:2611.03463](https://arxiv.org/abs/2611.03463), submitted to the SIAM Journal on Scientific Computing.

- Software (this repository, archived on Zenodo): DOI-SOFTWARE
- Large data files (Zenodo): DOI-DATA

## What this repository contains

The package `cgmy_cert` computes certified error bounds for approximate European-put price trajectories under the
Carr–Geman–Madan–Yor (CGMY) model, discretised by a B-spline Galerkin method with Crank–Nicolson time stepping. The
Galerkin matrices are assembled exactly from a periodised Fourier symbol, which also gives a closed-form lower bound
on the spectrum of the symmetric part, so the residual certificate needs no eigenvalue computation. A banded
comparison norm gives an O(N) full-order certificate, and empirical interpolation of the symbol makes the operator
affine, so reduced-order candidates (POD–Galerkin and POD-NN networks) are certified at a cost independent of N; a
three-stage rule escalates to the full-order certificate only when needed. The repository reproduces every table,
figure and number of the paper from stored result files, and contains the scripts that produced those files.

## Quick start

Tested with Python 3.10.12 on Linux x86_64; the pinned versions (NumPy 1.26.4, SciPy 1.15.3, mpmath 1.3.0,
matplotlib 3.10.9, pandas 2.3.3, PyTorch 2.14.0 CPU, ...) are in `requirements.txt`.

```bash
git clone https://github.com/DavoodDamircheli/certified_neural_glkn_cgmy.git
cd certified_neural_glkn_cgmy
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

python -m pytest -q                      # 67 tests, about 2.5 min (51 passed, 16 skipped without the large data)
bash build.sh                            # every table (tables/) and figure (figures/) from results/*.json
python scripts/make_paper_claims.py && python scripts/check_numbers.py   # 294 numbers of the paper vs result files
python scripts/run_E0.py --nproc 1       # rerun the verification studies (about 32 min on one core)
```

None of these commands needs the large data. `bash build.sh` reads only `results/*.json` and rewrites `tables/` and
`figures/` byte- and pixel-identically; `check_numbers.py` writes `results/NUMBER_CHECK.md` and exits with code 0 when
every number agrees. Thirteen tests need the large data and are skipped without it (with it: 63 passed, 3 skipped,
1 xpassed); `CGMY_FULL_TESTS=1 python -m pytest -q` also runs the three slow tests. The paper itself builds with
`bash paper/build_paper.sh` (pdflatex).

Timings in the paper were measured on an Intel Core Ultra 9 285K, Linux 6.8, one thread pinned to one performance core
of an idle machine (protocol in `cgmy_cert/timing.py`; hardware and load averages are stored in each result file).

## Reproducing the paper

Numbering as in `paper/main.pdf` and `paper/supplement.pdf` (supplementary materials, prefix SM). Table SM1.1
(notation), Fig. 3.1 (the online rule), Appendix A (proofs) and the cost-model formulas of Section SM5 have no result
file; the parameter box and the query populations are given in equation (2.9) and Section 5.1. All scripts are in
`scripts/`, all result files in `results/`; the plot/table step is the function called by `scripts/build_outputs.py`.

| Paper item | Output | Result file | Compute script | Plot/table step |
|---|---|---|---|---|
| Fig. 3.1 (online rule) | none | none | none | none |
| Fig. 4.1 (fig:eim) | `figures/F4_eim` | `eim.json` | `run_eim.py build select validate` | `run_eim.do_figure` |
| Table 5.1 (tab:budget) | `tables/tab_budget_v9.tex` | `E3_1_calibration.json` | `run_E3_1.py` | `run_E3_1.do_table` |
| Table 5.2 (tab:pod-ranks) | `tables/tab_pod_ranks_v9.tex` | `E1.json` (eval summary) | `run_E1.py eval` | `make_tables_phase2.py` |
| Fig. 5.1 (fig:acceptance) | `figures/F5a_acceptance` | `E1.json` | `run_E1.py eval select` | `fig_F5a.main` |
| Fig. 5.2 (fig:benchmark) | `figures/F6a_benchmark`, `F6b_refinement` | `E2.json` | `run_E2.py bench refine` | `run_E2.do_figs` |
| Table 5.3 (tab:benchmark) | `tables/tab_benchmark_v9.tex` | `E2.json` | `run_E2.py bench` | `run_E2.do_figs` |
| Table 5.4 (tab:ood) | `tables/tab_ood_v9.tex` | `E3.json` (summary, breakdown) | `run_E3.py run summary breakdown` | `run_E3.do_table_ood` |
| Fig. 5.3 (fig:ood) | `figures/F7_ood` | `E3.json`, `E1.json` | `run_E3.py run summary` | `run_E3.do_fig` |
| Fig. 5.4 (fig:online-time) | `figures/F5c_online_time` | `E1.json` (timing), `E2.json` (direct-solve line) | `run_E1.py timing`, `run_E2.py bench` | `run_E1.do_figs` |
| Fig. 5.5 (fig:amortisation) | `figures/F8_amortisation` | `E4.json`, `E4_progress.json` | `run_E4.py run summary` | `run_E4.plot_f8` |
| Table SM2.1 (tab:assembly-v9) | `tables/tab_assembly_v9.tex` | `E0.json` (E0.1) | `run_E0.py E0.1` | `run_E0.plot_e01` |
| Fig. SM2.1 (fig:spectral-bound) | `figures/F2_spectral_bound` | `E0.json` (E0.2) | `run_E0.py E0.2` | `run_E0.plot_e02` |
| Fig. SM2.2 (fig:theta) | `figures/F3_theta_and_effectivity` | `E0.json` (E0.3) | `run_E0.py E0.3` | `run_E0.plot_e03` |
| Table SM2.2 (tab:pcg-v9) | `tables/tab_pcg_v9.tex` | `E0.json` (E0.4) | `run_E0.py E0.4` | `run_E0.plot_e04` |
| Section SM3 (DeepONet, text) | none | `deeponet_old_vs_new.json` | `run_deeponet_old.py` | none |
| Table SM4.1 (tab:podnn) | `tables/tab_podnn_v9.tex` | `E2_4_podnn_N1024.json` | `eval_podnn.py` (via `sweep_N1024.py`) | `make_tables_phase2.py` |
| Table SM4.2 (tab:training) | `tables/tab_training_v9.tex` | `training_log.json` | `train_podnn.py` (via `sweep_N1024.py`) | `make_tables_phase2.py` |
| Table SM5.1 (tab:cost) | `tables/tab_cost_v9.tex` | `cost_model.json` | `run_cost_model.py` | `run_cost_model.write_table` |
| not in the paper | `figures/F5b_effectivity` | `E1.json` | `run_E1.py eval` | `run_E1.do_figs` |

Numbers in the text: `results/paper_claims.json` lists each with the result file and path that supports it (written by
`scripts/make_paper_claims.py`); `scripts/check_numbers.py` checks them. The production configuration is
`results/PRODUCTION_CONFIG.json` (N = 1024, N_T = 240, K = 100, eps_N = 1.18e-3).

**Which steps need the large data.** Rebuilding tables and figures, the tests, the number check, `run_E0.py`,
`run_eim.py`, `run_E3_1.py` and `run_deeponet_old.py` need no download. Rerunning `run_N1024.py`, `run_cells.py`,
`run_reduced_cert.py`, `run_pod.py`, `run_E1.py`, `run_E2.py`, `run_E3.py`, `run_cost_model.py`, `train_podnn.py`,
`eval_podnn.py` or `validate_spectral_fast.py` reads reference trajectories, POD bases, offline tensors or
checkpoints from `data/` and `checkpoints/`: download them (below) or regenerate them with the command listed for each
file in `results/MANIFEST.json` (key `excluded`).

## Large data

The data record (DOI-DATA) holds 116 files (29.0 GB) in five uncompressed tar archives, with `MANIFEST_DATA.sha256`
and `README_DATA.md` (contents, producing and consuming scripts):

| Archive | Size | Contents |
|---|---|---|
| `trajectories.tar` | 14.6 GB | `data/references_*.npz`: full-order reference trajectories |
| `offline_tensors.tar` | 13.1 GB | `data/redcert_*.npz` (reduced-certificate tensors), `data/eim/*.npz` (EIM of the symbol) |
| `podnn_training_data.tar` | 0.9 GB | `data/podnn_prep_*.npz`: POD-NN training and validation data |
| `checkpoints.tar` | 0.4 GB | `checkpoints/*.pt`: POD-NN networks |
| `pod_bases.tar` | 3.8 MB | `data/pod_*.npz`, `data/podnn_tbasis_*.npz`: POD bases |

Download the files into the repository root (the record number is the last part of DOI-DATA), verify, and unpack:

```bash
RECORD=<record number>
for f in trajectories offline_tensors podnn_training_data checkpoints pod_bases; do
  curl -L -C - -o $f.tar "https://zenodo.org/records/$RECORD/files/$f.tar?download=1"
done
curl -L -o MANIFEST_DATA.sha256 "https://zenodo.org/records/$RECORD/files/MANIFEST_DATA.sha256?download=1"
sha256sum -c MANIFEST_DATA.sha256
for f in trajectories offline_tensors podnn_training_data checkpoints pod_bases; do tar -xf $f.tar; done
```

The archives unpack into `data/` and `checkpoints/` (paths relative to the repository root). To check every unpacked
file against its SHA-256 in `results/MANIFEST.json`:

```bash
python -c "
import json, hashlib, os
m = json.load(open('results/MANIFEST.json'))['excluded']['files']
present = {p: d for p, d in m.items() if os.path.exists(p)}
bad = [p for p, d in present.items() if hashlib.sha256(open(p, 'rb').read()).hexdigest() != d['sha256']]
print(len(present) - len(bad), 'ok,', len(bad), 'bad,', len(m) - len(present), 'not present')"
```

## Repository layout

```
cgmy_cert/     the package: symbols (exact assembly), spectral, riesz, certificate, eim, pod, reduced_cert,
               three_stage, podnn, direct solver, timing protocol, provenance
tests/         pytest suite and fixtures (40-digit real-line fixture, query populations)
scripts/       compute scripts (run_*.py, gen_references.py, train_podnn.py, ...), build_outputs.py (tables and
               figures from result files), make_paper_claims.py / check_numbers.py, archive_check.py, zenodo_data.py
results/       result files (*.json, self-describing, with provenance), run logs, MANIFEST.json (hashes of every
               file, regeneration commands for the large data), NUMBER_CHECK.md, ARCHIVE_CHECK.json,
               DEVIATIONS.md and NOTES_FOR_PAPER.md (records kept during the computations)
tables/        LaTeX tables written by build.sh
figures/       figures written by build.sh (PNG, 200 dpi, and PDF)
paper/         the submitted manuscript: main.tex, supplement.tex, cover_letter.tex, theory/, tables/, figures/,
               compiled PDFs, build_paper.sh
legacy_v8/     read-only material from the earlier implementation, used by regression tests, the DeepONet stress
               test and the COS reference pricer
data/, checkpoints/   large files, not in git (Zenodo record DOI-DATA)
```

## Citation

```bibtex
@article{Damircheli2026CGMY,
  author        = {Damircheli, Davood},
  title         = {Symbol-based residual certification of reduced and neural surrogates for tempered-fractional
                   {CGMY} pricing},
  year          = {2026},
  eprint        = {2611.03463},
  archivePrefix = {arXiv},
  note          = {Submitted to SIAM Journal on Scientific Computing}
}

@software{Damircheli2026cgmycert,
  author    = {Damircheli, Davood},
  title     = {cgmy\_cert: code and results for "Symbol-based residual certification of reduced and neural
               surrogates for tempered-fractional {CGMY} pricing"},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0-sisc},
  doi       = {DOI-SOFTWARE}
}

@dataset{Damircheli2026cgmydata,
  author    = {Damircheli, Davood},
  title     = {Data for "Symbol-based residual certification of reduced and neural surrogates for
               tempered-fractional {CGMY} pricing"},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {DOI-DATA}
}
```

Citation metadata for GitHub and Zenodo: `CITATION.cff`, `.zenodo.json`.

## License and contact

Code: MIT License (`LICENSE`). Result files, tables, figures and the large data files: Creative Commons Attribution 4.0
International, CC BY 4.0 (`LICENSE-DATA`).

Davood Damircheli, Louisiana State University, ddamircheli@lsu.edu.
