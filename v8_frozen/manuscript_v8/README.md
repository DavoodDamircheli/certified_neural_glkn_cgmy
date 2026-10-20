# CGMY manuscript v8

Open `main.tex` in Overleaf, or run `bash build.sh` with TeX Live and Python 3. The compiled manuscript is `main_v8.pdf`. Its bibliography is embedded. This is one manuscript with one mathematical appendix; no separate supplement is required.

## Contents

- `main.tex`, `main_v8.pdf`: manuscript source and compiled paper.
- `tables/`, `figures/`: all displayed numerical tables and the two current plots.
- `scripts/verify_bundle.py`: checks retained numerical cells against frozen inputs, plot hashes, citations, labels and the build log.
- `scripts/v7_build_manuscript_assets.py`: preserved numerical-asset generator, requiring the separate numerical source and raw results.
- `provenance/`: input manifests and frozen numerical table sources for asset traceability; these are not additional manuscript appendices.
- `verification/`: local editorial checks and final build records.
- `REVISION_NOTES.md`, `SOURCE_DIFF.patch`, `SHA256SUMS`: change summary, complete source diff and package checksums.

The PDF has 23 pages, including references and the mathematical appendix, versus 51 pages in the supplied v7 PDF. It retains the 11 pt document class and one-inch margins. Broad tables use readable panels rather than reducing the body font.

## Numerical reproduction scope

This archive is self-contained for manuscript compilation and asset-fidelity checks. It does not contain the solver, raw campaign, training script, datasets or checkpoints. No numerical campaign was rerun for this editorial revision.

Numerical regeneration additionally requires the original `v5_M3-M7_handoff_20261008.zip`, `v5_M8-M9_addendum_20261008.zip` and `v5_M10_table_generation_supplement_20261008.zip`, assembled into a numerical workspace. With that workspace available, use:

```bash
python3 scripts/v7_build_manuscript_assets.py \
  --data-root /absolute/path/ws/results/free_integrated_v5 \
  --source-root /absolute/path/ws \
  --output-root /absolute/path/fresh_assets
```

The generator requires NumPy, SciPy, Matplotlib, h5py, PyYAML and the aggregation module from the numerical source. It regenerates the frozen numerical assets, not the condensed v8 table layouts. All retained numerical cells are checked against their frozen sources by `verify_bundle.py`; the Euclidean POD rows are summarised in the manuscript rather than repeated in its rank table.

The numerical evidence retains its original scope: exact-arithmetic discrete certificates evaluated in ordinary floating point; conditional spectral lower bounds; empirical continuum-price diagnostics. Raw-input and original-asset hashes remain in `provenance/evidence_v6.json`. The full numerical regeneration and training reproducibility checks documented in the supplied archive have not been newly executed here.
