#!/usr/bin/env bash
# Build the submitted manuscript: main article, supplementary materials and cover letter (pdflatex, two passes each;
# cross-references between main and supplement come from the frozen xref_main.tex / xref_supplement.tex).
# Usage: bash paper/build_paper.sh
set -euo pipefail
cd "$(dirname "$0")"
for doc in main supplement cover_letter; do
  pdflatex -interaction=nonstopmode -halt-on-error "$doc" > /dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$doc" > /dev/null
  echo "$doc.pdf: $(pdfinfo "$doc.pdf" 2>/dev/null | awk '/^Pages:/ {print $2}') pages"
done
grep -l -E "Reference .* undefined|Citation .* undefined" main.log supplement.log cover_letter.log && exit 1 || true
