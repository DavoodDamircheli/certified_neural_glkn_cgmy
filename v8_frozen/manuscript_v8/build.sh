#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cgmy_build_dir=$(mktemp -d /tmp/cgmy-v8-build.XXXXXX)
trap 'rm -rf "$cgmy_build_dir"' EXIT
for pass in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$cgmy_build_dir" -jobname=cgmy_v8_clean main.tex > "$cgmy_build_dir/build_pass_${pass}.log"
done
python3 - "$cgmy_build_dir" <<'PY'
from pathlib import Path
import os,sys,tempfile
source=Path(sys.argv[1])
for name,dest in [('cgmy_v8_clean.pdf',Path('main_v8.pdf')),('cgmy_v8_clean.log',Path('verification/latex_final.log'))]+[(f'build_pass_{i}.log',Path(f'verification/build_pass_{i}.log')) for i in range(1,4)]:
 dest.parent.mkdir(exist_ok=True)
 fd,p=tempfile.mkstemp(prefix='final-',suffix='.tmp',dir=dest.parent)
 with os.fdopen(fd,'wb') as f:
  f.write((source/name).read_bytes());f.flush();os.fsync(f.fileno())
 os.replace(p,dest)
PY
python3 scripts/verify_bundle.py
