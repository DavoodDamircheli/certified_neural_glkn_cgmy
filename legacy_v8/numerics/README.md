# certified_galkn_cgmy

## Galerkin-Certified Neural Operators for Tempered-Fractional CGMY Pricing PIDEs

Numerical package for a research project combining a trusted **B-spline
Galerkin solver** for the tempered-fractional CGMY pricing PIDE with
**operator learning** and **a posteriori certification**. The Galerkin
solver is treated as an external, already-validated reference: this
repository wraps it, it does not reimplement it.

---

## Six-stage dependency chain

Every later stage consumes the output of the one before it. Nothing in
stage *n* may be validated using anything from a stage after it.

| Stage | Name | Package | Produces | Depends on |
|---|---|---|---|---|
| 1 | Galerkin solver | `src/cgmy/` | Reference solutions (`GalerkinSolution`): B-spline coefficients on a frozen discretization | The pre-existing, externally validated solver |
| 2 | Discrete residual | `src/certification/residuals.py` | `R(u) = f - A(params) @ u` for any candidate coefficient vector | Stage 1's operator/load definitions (`cgmy.operators`) |
| 3 | Certificate | `src/certification/{riesz_solver,constants,guaranteed_certificate,mass_certificate}.py` | Guaranteed bound `eta(u) >= \|\|u - u_true\|\|` | Stage 2's residual, certified stability constants |
| 4 | Neural model | `src/learning/` | Fast prediction of a coefficient vector from (CGMY params, contract data) | Stage 1's datasets (`src/data/`) for training |
| 5 | Predict-certify-correct | `src/correction/krylov.py` | Accepted or Krylov-corrected coefficient vector | Stage 3's acceptance decision, Stage 4's prediction as an iterative seed |
| 6 | Experiments | `experiments/` | Convergence, certification-quality, generalization, and speedup results | All prior stages |

```
Galerkin solver -> discrete residual -> certificate -> neural model -> predict-certify-correct -> experiments
```

---

## Validation gates

Each gate must pass before the corresponding stage is considered trustworthy
enough to build on. A gate failing invalidates every downstream stage's
results until it is re-passed.

| Gate | Checks | Blocks progress on |
|---|---|---|
| G1 — Solver fidelity | Reference solver reproduces known/benchmark CGMY prices; spatial and temporal convergence rates match theory | Generating any training data (Stage 4) |
| G2 — Residual consistency | Residual of the *reference* solution is at machine-precision / discretization-error level (near-zero) | Trusting residuals computed for neural predictions (Stage 2) |
| G3 — Certificate reliability | `eta(u_theta) >= \|\|u_theta - u_reference\|\|` holds empirically across the sampled parameter box; effectivity index is bounded and stable | Using the certificate to accept/reject predictions (Stage 3) |
| G4 — Certificate computability | `eta(u_theta)` costs materially less than a full reference solve | Claiming any net speedup (Stage 6) |
| G5 — Neural approximation quality | Training/validation/test errors meet target tolerances on the frozen parameter box | Reporting certification statistics as meaningful (Stage 4) |
| G6 — Correction convergence | Krylov correction reduces a rejected prediction's residual below the acceptance tolerance within a bounded iteration budget | Claiming the predict-certify-correct loop is well-defined (Stage 5) |
| G7 — End-to-end speedup | `t_NN + t_cert (+ t_correct)  <<  t_Galerkin`, measured, not assumed | Reporting Stage 6 many-query experiments |

---

## Repository layout

```text
certified_galkn_cgmy/
├── src/
│   ├── cgmy/            # Stage 1: reference solver wrapper, operators, lifting, splines
│   ├── certification/   # Stages 2-3: residuals, certificates, stability constants
│   ├── learning/        # Stage 4: neural architectures, losses
│   ├── data/             # Sampling, payoff families, dataset generation
│   ├── correction/       # Stage 5: Krylov correction
│   └── common/           # HDF5 I/O and other cross-cutting utilities
├── tests/                 # pytest suite
├── experiments/           # Stage 6: experiment scripts/notebooks
├── configs/               # base.yaml: frozen discretization + parameter box
├── scripts/               # CLI entry points (dataset generation, training, evaluation)
├── data/                  # Generated datasets (gitignored contents)
├── checkpoints/           # Trained model checkpoints (gitignored contents)
├── results/               # Raw experiment outputs (gitignored contents)
├── figures/               # Generated figures for the manuscript
├── tables/                # Generated tables for the manuscript
├── vendor/
│   └── cgmy_bspline/      # The trusted external solver, vendored -- see its own README.md
└── notes/                 # Task write-ups (derivations, diagnostic studies, decisions surfaced)
```

---

## The trusted reference solver (`cgmy_bspline`)

The B-spline Galerkin solver this whole project certifies against
(CLAUDE.md Hard Invariant #1) lives in a separate, independently-versioned
repository, not a PyPI package. It is **vendored** into this repository at
`vendor/cgmy_bspline/` — a clean clone of `certified_galkn_cgmy` alone is
sufficient to import it; no separate checkout, network access, or
`CGMY_BSPLINE_PATH` environment variable is required by default. See
`vendor/cgmy_bspline/README.md` for exactly what was vendored, its
provenance (base commit + a patch of uncommitted upstream changes this
project depends on, applied on top), and why vendoring was chosen over a
git submodule or a `pip install git+...` dependency.

To point at a different `cgmy_bspline` checkout instead (e.g. a local
clone of the upstream repo, for working against a newer version), set
`CGMY_BSPLINE_PATH` to the directory *containing* a `cgmy_bspline/`
package directory before running anything — see `src/cgmy/_external.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Runtime/dev dependencies are pinned in `pyproject.toml`/`requirements.txt`
to the exact versions this project was developed and verified against:
`numpy==1.26.4`, `scipy==1.15.3`, `h5py==3.16.0`, `pyyaml==5.4.1`,
`matplotlib==3.10.9`, `torch==2.14.0`, `pytest==9.1.1`. Ask before adding
anything beyond this list.

## Testing and linting

```bash
pytest
ruff check .
```

**Reproducibility.** Verified in a fresh virtualenv from a clean clone
(`git clone` into an empty directory, `pip install -e ".[dev]"`, `pytest`,
no `CGMY_BSPLINE_PATH` set, no pre-existing external checkout on the
machine) (Task 6, working prompt; re-verified after Tasks 7-14 and the
final experiment matrix, working prompt):

- **115 passed, 1 failed** with zero other local setup. The one failure
  (`tests/test_exp10_smoke.py`) needs a trained `checkpoints/mlp_final.pt`
  (and its siblings) on disk -- `checkpoints/`/`data/` contents are
  gitignored by design (large binaries; `*.pt`, `*.npz`, `*.h5`), so a bare
  clone has none. This is expected, not a bug: generate them with
  `scripts/generate_production_dataset.py` then
  `scripts/train_final_models.py` (real compute time; not run as part of
  this check), or copy an existing `checkpoints/`/`data/production_dataset.h5`
  from a machine that already has them.
- **116 passed, 0 failed** once `checkpoints/`/`data/production_dataset.h5`
  are present (verified by copying this repository's own, provenance-patched
  checkpoints into the fresh clone — see `notes/05_provenance.md`).

Re-verify with `pytest -q | tail -1` after any change that touches `src/`,
`vendor/`, or `tests/`, and update this section if the counts change.

## Configuration

`configs/base.yaml` freezes the discretization (spline degree, element
count, domain truncation, time steps) and the CGMY/contract parameter box
used across data generation, training, and certification. Changing it
invalidates previously generated datasets and certificates.
