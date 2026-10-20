# certified-CGMY-Galerkin, v9 workspace

Target: *Symbol-based residual certification of reduced and neural surrogates for
tempered-fractional CGMY pricing* (v8 -> v9).  Plan: `step_by_step_workplan_v9.md` (project).

```
v8_frozen/           the v8 numerical workspace and manuscript bundle (tag v8-frozen; read-only)
cgmy_cert/           the v9 package (Phase 1): symbols, spectral, riesz, direct, certificate, pod, highprec
tests/               pytest suite (plan 1.1-1.5 done-when criteria); tests/fixtures/ (plan 0.2)
scripts/             build_fixtures.py, run_E0.py, coercivity_constant.py, assemble_v9_skeleton.py
results/             E0.json, coercivity_c1.json, logs
figures_v9/, tables_v9/   outputs of run_E0.py
manuscript_v9/       main_v9.tex (skeleton, plan 0.3) + theory/*.tex (one file per result)
data/                production_dataset.h5 (v8 DeepONet / POD data; not in git, sha256 in v8_frozen/)
```

Quick start (Python >= 3.10, numpy, scipy, mpmath, pandas, matplotlib, pytest):

    python3 -m pytest tests -q                 # ~5 min (quick subsets)
    CGMY_FULL_TESTS=1 python3 -m pytest tests  # full populations (~40 min)
    python3 scripts/run_E0.py                  # E0.1-E0.5 (~1 h on 2 cores)
    cd manuscript_v9 && pdflatex main_v9 (x3)

Conventions: p = 3, Omega = (log 100 - 8, log 100 + 8), free indices k = 0..N-3, K = 100 aliases,
L = multiple of N_h symbol samples, tau_n = n T / N_T, pure CN.  See module docstrings.
