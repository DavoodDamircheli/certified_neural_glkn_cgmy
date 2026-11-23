# Claude Code brief: Phase 5a, clean the repository and write the README (before the Zenodo upload)

Pause the Zenodo work after step 0. Do this brief first, then return to `CLAUDE_PHASE5_zenodo.md`
at step 2.3. The GitHub release that Zenodo archives must be a clean repository.

Ground rules:
- Do not rerun any experiments, and do not change any number in the paper.
- Nothing that `bash build.sh`, `pytest`, `check_numbers.py` or the scripts in `scripts/` need may be
  lost. Such files can be moved, provided every path that points to them is updated.
- Never rewrite git history and never force-push without my explicit "yes". Deleting files in a
  normal commit is enough: Zenodo archives the release tree, not the history.
- Do not touch `zenodo_data/` (the five tar archives, the manifest and README_DATA.md); they stay
  local and git-ignored until the upload is finished.
- **ASK** before any step marked ASK. One commit per step.

## 0. Safety net
1. Make sure `git status` is clean, then create the annotated tag `pre-cleanup` on the current
   head and push it. Everything removed below stays recoverable from that tag.
2. Discard the unpublished sandbox draft 647467 on sandbox.zenodo.org
   (`DELETE /api/deposit/depositions/647467` with the sandbox token, if it is still exported;
   otherwise tell me and I will delete it in the browser).

## 1. Inventory: report, do not delete yet
List every tracked and untracked file or folder that belongs to one of these groups, with its size
and whether any script, test or build file references it (`grep -r`):
- **Zip files:** in the root and anywhere else (handoff zips, `phase*_*.zip`, `manuscript_v9_bundle.zip`,
  `v8_frozen.zip`, `SISC_CGMY_v12_source.zip`, ...).
- **Claude brief and report files:** `CLAUDE_*.md`, `PHASE*_REPORT.md`, `DEVIATIONS.md`, `NOTES_FOR_PAPER.md`,
  and similar.
- **Old manuscript versions:** `manuscript_v8*`, `manuscript_v9/`, `amsart_version/`,
  `main_v9*.tex`, old `tables_v9/` and `figures_v9/` copies, and any other duplicates of v12.
- **Legacy code:** `v8_frozen/`. Note exactly which tests and fixtures use it; the 459-coordinate
  regression suite and the 40-digit fixture must keep working.
- **LaTeX and Python build artefacts:** `*.aux`, `*.log`, `*.out`, `*.synctex.gz`, `__pycache__`,
  `.pytest_cache`, `.ipynb_checkpoints`, and so on.
- Anything else that looks like scratch.

Show me the list as a table with a proposed action for each row (keep / move / delete / git-ignore),
and **ASK** before acting.

## 2. Target layout (default, adjust to what the inventory shows)
```
README.md            CITATION.cff   .zenodo.json   LICENSE   LICENSE-DATA   .gitignore
paper/               the submitted manuscript (contents of manuscript_v12/: main.tex, supplement.tex,
                     references.tex, xref files, siamart251216.cls, theory/, tables/, figures/,
                     cover_letter.tex, SMindex.txt/html, build script for the PDFs)
cgmy_cert/           the package
tests/               tests and fixtures (move what the regression tests need from v8_frozen here,
                     e.g. tests/fixtures/legacy_v8/, if that is smaller than keeping v8_frozen/)
scripts/             experiment, table, figure, claims and Zenodo scripts
results/             result JSON files, zenodo_ids.json, reports that the README links to
docs/                optional: a short reproducibility guide; no Claude briefs
```
- Move `manuscript_v12/` to `paper/`, and update `build.sh`, `check_numbers.py`,
  `make_paper_claims.py` and the README mapping.
- Delete the old manuscript versions, the zips and the Claude brief files (they are kept in the
  `pre-cleanup` tag and in my Claude project).
- Put the build artefacts, `zenodo_data/`, `*.zip` and the large data paths in `.gitignore`.
- Locally, delete untracked zips and build artefacts too, but **not** `zenodo_data/`, the large
  data files listed in the SHA-256 manifest, or the checkpoints.

**ASK** me to confirm the final layout before the move.

## 3. Verify that nothing broke
- `python -m pytest -q`: all tests pass (including the v8 regression tests), with the same count as
  before.
- `make_paper_claims.py && check_numbers.py` exits with code 0, with 294 claims.
- `bash build.sh` in a **fresh clone** of the cleaned repository: every table byte-identical and
  every figure pixel-identical.
- The paper still builds from `paper/` (main 26 pages, supplement 8, cover letter 2) with no
  undefined references.
- `git grep` finds no path to a deleted or moved file.

## 4. A clear README.md
Rewrite `README.md` for a reader of the paper. Use these sections:
1. **Title** of the paper, the author, and a link to arXiv:2611.03463. Add badges or lines for the
   Zenodo DOIs (software and data) as placeholders `DOI-SOFTWARE` / `DOI-DATA`, which step 4 of the
   Zenodo brief fills in.
2. **What this repository contains:** a 4–6 sentence summary of the method and of what the code
   reproduces.
3. **Quick start:** environment (Python 3.10, NumPy 1.26, SciPy 1.15, mpmath 1.3, PyTorch 2.14 CPU;
   provide or point to `requirements.txt` or `environment.yml`, creating one from the versions in the
   result files if missing), install, run the tests, rebuild all tables and figures
   (`bash build.sh`), and rerun the verification studies (`run_E0.py`, about 32 min on one core).
4. **Reproducing the paper:** the table "paper table/figure → result file → script" (the
   existing v12 mapping, including Fig. 3.1 with no result file), and which results need the
   large data.
5. **Large data:** what is on Zenodo (the five archives and their sizes), how to download them, how to
   verify them against `MANIFEST_DATA.sha256`, and how to unpack them into the expected paths.
6. **Repository layout:** a short tree with one line per folder.
7. **Citation:** BibTeX for the paper (arXiv for now) and for the software and data records
   (DOI placeholders).
8. **License:** MIT for the code, CC-BY-4.0 for the data and results. **Contact:** the author's name and
   affiliation; **ASK** before adding an e-mail address.

Keep it factual and readable: no marketing language, and every command copy-pasteable and tested.
Make sure `CITATION.cff` and `.zenodo.json` agree with the README (title, author, license).

## 5. Push
Show me `git status`, the diff summary and the README rendered as text, then **ASK** before pushing
to `main`. After the push, confirm that the repository page on GitHub shows the new README and that
none of the deleted files remains on `main`.

## 6. Then resume the Zenodo brief
Return to `CLAUDE_PHASE5_zenodo.md`, step 2.3 (production token).
- The README now carries the DOI placeholders; step 2.5 writes the reserved data DOI into them.
- The upload is about 29 GB. Run it in the background (`nohup` or `tmux`) with resumable,
  per-file retries and a log in `results/zenodo_upload.log`, so that a closed terminal or an
  idle-session timeout does not interrupt it. Tell me how to check progress, and keep the machine
  awake (for example `caffeinate -i` on macOS, `systemd-inhibit` on Linux).
- Verify the MD5 checksums after the upload, then stop at the draft as the brief says.
