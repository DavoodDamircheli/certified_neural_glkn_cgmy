#!/usr/bin/env python3
"""Phase 4, item 3: phase4_archive_<date>.zip = the reproducibility archive of item 1 (cgmy_cert, tests with fixtures,
scripts, results incl. MANIFEST.json, ARCHIVE_CHECK.json, NUMBER_CHECK.md and PHASE4_REPORT.md, figures_v9 PNG + PDF,
tables_v9, README.md, requirements.txt, build.sh, Makefile).  No data files (data/) and no checkpoints (*.pt): their
SHA-256, sizes and regeneration commands are in results/MANIFEST.json -> excluded.  The zip must stay below 100 MB.
The file list is results/MANIFEST.json -> archive (scripts/make_manifest.py); every file is re-hashed and must match the
manifest, so the zip and the manifest describe the same archive.  Usage: python3 scripts/make_manifest.py &&
python3 scripts/make_phase4_zip.py [--suffix _final]
"""
from __future__ import annotations

import argparse, json, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.eim import sha256  # noqa: E402

MAN = ROOT / "results" / "MANIFEST.json"
LIMIT_ZIP = 100 * 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="appended to the file name, e.g. _final")
    out = ROOT / f"phase4_archive_{time.strftime('%Y%m%d')}{ap.parse_args().suffix}.zip"
    arch = json.loads(MAN.read_text())["archive"]["files"]
    stale = [k for k, v in arch.items() if not (ROOT / k).is_file() or sha256(ROOT / k) != v["sha256"]]
    if stale:
        raise SystemExit(f"MANIFEST.json is stale for {len(stale)} files (e.g. {stale[:3]}); run scripts/make_manifest.py")
    files = [ROOT / k for k in arch] + [MAN]
    bad = [p for p in files if p.suffix == ".pt" or p.relative_to(ROOT).parts[0] in ("data", "checkpoints")]
    if bad:
        raise SystemExit(f"data/checkpoint files in the archive list: {bad[:3]}")
    for must in ("results/MANIFEST.json", "results/ARCHIVE_CHECK.json", "results/NUMBER_CHECK.md",
                 "results/PHASE4_REPORT.md", "build.sh", "README.md"):
        if ROOT / must not in files:
            raise SystemExit(f"missing from the archive (not tracked?): {must}")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, Path("cgmy_v9_archive") / p.relative_to(ROOT))
    size = out.stat().st_size
    n_png = sum(p.suffix == ".png" and "figures_v9" in p.parts for p in files)
    n_pdf = sum(p.suffix == ".pdf" and "figures_v9" in p.parts for p in files)
    print(f"wrote {out.name} ({size / 2 ** 20:.1f} MB, {len(files)} files; figures {n_png} PNG + {n_pdf} PDF)")
    if size >= LIMIT_ZIP:
        raise SystemExit("zip exceeds 100 MB")


if __name__ == "__main__":
    sys.exit(main())
