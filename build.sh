#!/usr/bin/env bash
# Regenerate every table (tables/) and figure (figures/, PNG + PDF) from results/*.json only.
# No solver call and no data file are needed.  Usage: bash build.sh
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
python3 scripts/build_outputs.py
