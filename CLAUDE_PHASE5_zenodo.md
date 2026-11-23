# Claude Code brief: Phase 5 — archive code and data on Zenodo, put the DOIs in the paper

Read this whole file before acting, then work through the steps in order, one commit per step.
Ground rules:
- Do not rerun experiments, and do not change any numbers in the manuscript.
- Never publish a Zenodo record and never push a tag or release without my explicit "yes" in the chat.
- Stop and ask whenever a step says **ASK**.
- Put the Zenodo token only in the environment variable `ZENODO_TOKEN`. Never write it to a file and never echo it.

## 0. Sync the submitted manuscript (v12) into the repository

`SISC_CGMY_v12_source.zip` is in the repository root (I downloaded it from Claude).

1. Unpack it into `manuscript_v12/`.
2. Build it: `pdflatex main; pdflatex supplement`, twice. Check that the main article has 26 pages and the
   supplement 8, with no undefined references.
3. Mirror the two edited table captions in the table generators, so that `bash build.sh` still
   reproduces the tables byte-identically:
   - `tab_assembly`: the caption now reads "The principal-period rule is the alias sum with $K=0$;"
     (the words "of the earlier version" are gone);
   - `tab_podnn`: the last sentence now reads "At $\varepsilon_N=1.18\times10^{-3}$ the certificate
     rejects every POD-NN query, each by the cheap rejection of the first stage."
4. Point `scripts/check_numbers.py` at `manuscript_v12/` (`main.tex`, `supplement.tex`, `theory/*.tex`).
   Re-point every claim whose sentence was reworded (the values did not change), and add a claim for
   "414 queries with a reference solution" (= 150 + 264, in the abstract). The number 3.9 ms (online
   certificate at r = 48) is no longer in the text, only in the online-time figure: re-point or drop
   that claim.
5. Update the README mapping "paper table/figure → result file → script" from the compiled v12 PDFs.
   Read the numbering from the PDFs, because the tables and supplement sections were renumbered: the
   notation table and the cost-model formulas moved to the supplement, and there is a new Appendix A.

Done when `make_paper_claims.py && check_numbers.py` exits with code 0 and `bash build.sh` is
byte- and pixel-identical.

## 1. Repository hygiene for archiving

1. **ASK** whether to rename the GitHub repository from `ertified_neural_glkn_cgmy` to
   `certified_neural_glkn_cgmy`. If I say yes, run `gh repo rename certified_neural_glkn_cgmy` and
   update the local remote.
2. Add `LICENSE` (MIT for the code; **ASK** if unsure) and `LICENSE-DATA` (CC-BY-4.0) for the result
   files.
3. Add `CITATION.cff` (author Davood Damircheli, Louisiana State University; **ASK** for my ORCID) and
   a `.zenodo.json` with:
   - upload_type "software";
   - title "cgmy_cert: code and results for 'Symbol-based residual certification of reduced and
     neural surrogates for tempered-fractional CGMY pricing'";
   - creators, description, license and keywords;
   - related_identifiers: `arXiv:2611.03463` (`isSupplementTo`, scheme `arxiv`). The dataset DOI is
     added in step 2.
4. Check that the repository contains no secrets, tokens, absolute personal paths or files over
   100 MB. Report what you find before changing anything.

## 2. Data record (Zenodo REST API): reserve the DOI first, do not publish

The large files are those listed in the SHA-256 manifest of files too large to archive (reference
trajectories, POD bases, offline tensors, network checkpoints; about 27 GB). Find them through the
manifest.

1. Verify every file against its SHA-256 hash, and report any that are missing or mismatched.
2. Pack them into at most about 10 archives, grouped by kind: `trajectories.tar`,
   `pod_bases.tar`, `offline_tensors.tar`, `checkpoints.tar` (compress only if it helps).
   - The total must be under 50 GB and under 100 files.
   - Write `MANIFEST_DATA.sha256` for the archives.
   - Write `README_DATA.md`: what each archive contains, which script produces it and which script
     consumes it, and how to unpack it into the repository layout.
3. **ASK** me to create a Zenodo personal access token (scopes `deposit:write`, `deposit:actions`)
   and to export it as `ZENODO_TOKEN`. Test the whole flow first on `https://sandbox.zenodo.org` with
   a sandbox token and one small file, then repeat on `https://zenodo.org`.
4. Create the deposition with a reserved DOI:
   `POST /api/deposit/depositions` with `{"metadata": {"prereserve_doi": true}}`. Record the
   deposition id, `links.bucket` and `metadata.prereserve_doi.doi` in `results/zenodo_ids.json`.
5. Write the reserved data DOI into `README_DATA.md`, the repository `README.md` and `.zenodo.json`
   (`isSupplementedBy`, scheme `doi`).
6. Upload the archives, the manifest and the README with `curl --upload-file <file>
   "$BUCKET/<name>?access_token=$ZENODO_TOKEN"`, resuming or retrying on failure. Afterwards, compare
   Zenodo's MD5 checksums with local ones.
7. Set the metadata with `PUT /api/deposit/depositions/<id>`:
   - upload_type "dataset";
   - title "Data for 'Symbol-based residual certification … CGMY pricing'";
   - creators, description, license `cc-by-4.0`, keywords;
   - related_identifiers: arXiv:2611.03463 `isSupplementTo`. The software DOI is added in step 4.
8. Do **not** publish. Give me the link to the draft, the reserved DOI and the list of files, then
   stop.

## 3. Software record (GitHub release → Zenodo)

1. **ASK** me to confirm that I have enabled the repository on the Zenodo GitHub page
   (zenodo.org → GitHub → Sync → toggle On). This must happen before the release.
2. Commit everything from steps 0–2 and check that `git status` is clean.
3. **ASK**, then run:
   `gh release create v1.0-sisc --title "Version submitted to SISC" --notes "<short notes with the data DOI>"`.
4. Poll the Zenodo API (`GET /api/records?q=...` or `GET /api/deposit/depositions`) until the software
   record appears. Record its version DOI and concept DOI in `results/zenodo_ids.json`.

## 4. Link the two records and finish

1. Add the software version DOI to the data draft (`isRequiredBy` / `references`, scheme `doi`).
2. **ASK** for my "yes", then publish the data record:
   `POST /api/deposit/depositions/<id>/actions/publish`. Check that its DOI resolves.
3. In `manuscript_v12/main.tex`, replace the GitHub URL in the Reproducibility subsection with both
   DOIs, for example "The code and result files are archived at
   \url{https://doi.org/<software version DOI>} and the large data files at
   \url{https://doi.org/<data DOI>}". Add two entries to `references.tex`, in alphabetical order:
   - D. Damircheli, *cgmy_cert …* (software), Zenodo, 2026;
   - D. Damircheli, *Data for …* (dataset), Zenodo, 2026.
4. Make the same replacement in `manuscript_v12/cover_letter.tex`, in the Reproducibility paragraph.
5. Rebuild the main article, the supplement and the cover letter. The main article must stay at 26
   pages. If the references spill onto page 27, **ASK** before cutting anything.
6. Refresh `xref_main.tex` / `xref_supplement.tex` from the `.aux` files. Remove the
   hyperref anchor field, i.e. the 4th group of each `\newlabel`.
7. Zip `manuscript_v12/` as `SISC_CGMY_v12_submission.zip`.
8. Append to `results/PHASE5_REPORT.md`: the DOIs, the commit hash, and the claim-check and
   build-check results.

## Report back

When you are done, send me:
- both DOIs (software version and concept DOI, and the data DOI);
- the release URL;
- the page counts;
- the exit code of `check_numbers.py`;
- anything that still needs my action.
