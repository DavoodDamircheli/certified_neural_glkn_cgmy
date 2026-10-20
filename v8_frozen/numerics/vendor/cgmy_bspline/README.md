# Vendored `cgmy_bspline` — the trusted external Galerkin solver

Task 6 (working prompt): this project's certification stack has no
meaning without a fixed, reproducible version of the trusted external
B-spline Galerkin solver (CLAUDE.md Hard Invariant #1). Previously, `src/
cgmy/_external.py` located it via a hardcoded, machine-specific absolute
path (`/home/davood/projects/cgmy_bspline`) with no pinning at all — a
clean clone on any other machine would fail to import `cgmy_bspline`
entirely.

## Why vendoring, not a git submodule or a `pip install git+...` dependency

Both alternatives require pinning to a specific, retrievable **commit** of
the upstream repository (`https://github.com/DavoodDamircheli/cgmy_bspline`).
Investigating which commit to pin to surfaced a real problem: at the time
this was done, that repository's working tree had **uncommitted local
changes to exactly the functions this project depends on** —
`solver_cn.solve_cn`'s `record_history=True` mode (which
`cgmy.reference_solver.solve_cgmy`, this project's entire reference-solve
path, requires: `dt`, `bc_dofs`, `c_init`, `coeffs_history`,
`forcing_history`, `step_metadata` are all part of the uncommitted diff)
and `projection.assemble_payoff_rhs` (used directly by this project's own
`src/`). The last actual commit (`6df8ad1588c8fe761ea5e4e5ea61e63d3fbe1b4b`)
predates all of this — pinning to it via a submodule or a git dependency
would produce an archive that fails immediately (`solve_cn() got an
unexpected keyword argument 'record_history'`), not a merely-outdated one.

Committing (and pushing) that fix on the user's behalf, in a *different*
repository than the one this task concerns, is not a call to make
unilaterally. Vendoring sidesteps the dilemma entirely: it captures the
EXACT working-tree state this project's test suite already depends on,
entirely within `certified_galkn_cgmy`'s own repository, with no
dependency on `cgmy_bspline`'s git history being in any particular state.
**This is explicitly one of the three options the working prompt itself
offers** ("vendor it in, add it as a pinned git submodule, or declare it
as a git dependency") and the most defensible one given what was found.

**Recommendation, not an action taken here:** the uncommitted changes
described above look complete and deliberate (backward-compatible,
documented, not WIP) — committing them in `cgmy_bspline` itself would let
a future re-vendor (or a switch to a submodule/git-dependency pin) happen
cleanly. Left for the user to do in that repository.

## What was vendored, and why only this

Only the 11 top-level modules `certified_galkn_cgmy` actually imports
(directly or transitively) — traced by grepping every `from cgmy_bspline`/
`import cgmy_bspline` in this project's `src/`, `experiments/`, `scripts/`,
`tests/`, then following each vendored module's own internal
`cgmy_bspline.*` imports to closure:

    __init__.py  boundary.py  bspline_basis.py  connection_coefficients.py
    greeks.py  grid.py  matrices.py  parameters.py  projection.py
    solver_cn.py  toeplitz.py

**Not vendored** (present in the upstream repo, not needed here):
`calibration.py` and `payoff.py` (the latter is empty; neither is imported
anywhere in this project), `experiments/`, `results/` (a separate paper's
figures/LaTeX/PDF reports), `src/` (an older, unrelated American-option
solver tree — LSM, PSOR, COS pricers), and `tests/` (upstream's own test
suite — not required for this project's own tests to pass; re-vendor it
too if deeper validation of the solver itself is ever wanted).

## Provenance

- Base commit: `6df8ad1588c8fe761ea5e4e5ea61e63d3fbe1b4b` (upstream
  `https://github.com/DavoodDamircheli/cgmy_bspline`, branch state at
  vendor time).
- Plus the uncommitted working-tree changes to `projection.py`,
  `solver_cn.py`, `toeplitz.py` present at vendor time — saved verbatim as
  `UNCOMMITTED_CHANGES_AT_VENDOR_TIME.patch` in this directory (`git diff`
  output, applies cleanly against the base commit above).
- Vendored: 2026-09-18 (UTC), by this working prompt's Task 6.
- CLAUDE.md Hard Invariant #1 still holds: this project never modifies
  these files. Re-vendoring (replacing this directory's contents wholesale
  from a fresh, clean checkout of the upstream repo, base commit + patch,
  or a future clean commit) is the only sanctioned way to update it.

## How this is imported

`src/cgmy/_external.py` puts this directory's parent (`vendor/`) on
`sys.path` by default, so `import cgmy_bspline` resolves here. Set the
`CGMY_BSPLINE_PATH` environment variable to point at a different
`cgmy_bspline` checkout instead (e.g. a local clone of the upstream repo
after committing the fix described above) — see that module's docstring.
