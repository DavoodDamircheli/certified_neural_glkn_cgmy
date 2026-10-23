#!/usr/bin/env python3
"""Phase 3 deliverable (CLAUDE_PHASE3.md section 6): phase3_results_<date>.zip with cgmy_cert, tests, scripts, results,
figures_v9, tables_v9 (DEVIATIONS.md, NOTES_FOR_PAPER.md, MANIFEST.json and the reports are in results/), and no
checkpoints (*.pt) and no data files (data/): their sha256 are listed in results/MANIFEST.json -> excluded_from_phase3_zip.
The test fixtures (tests/fixtures/) are kept; the tests need them.  The zip must stay below 100 MB.
"""
from __future__ import annotations

import json, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.eim import sha256  # noqa: E402

INCLUDE_DIRS = ("cgmy_cert", "tests", "scripts", "results", "figures_v9", "tables_v9")
SKIP_PARTS = ("__pycache__", ".pytest_cache")
LIMIT_ZIP = 100 * 2 ** 20


def main():
    out = ROOT / f"phase3_results_{time.strftime('%Y%m%d')}.zip"
    man_path = ROOT / "results" / "MANIFEST.json"
    man = json.loads(man_path.read_text())
    excl = {}
    for p in sorted(ROOT.joinpath("checkpoints").rglob("*.pt")):
        excl[str(p.relative_to(ROOT))] = dict(sha256=sha256(p), bytes=p.stat().st_size, reason="checkpoint")
    for p in sorted(ROOT.joinpath("data").rglob("*.npz")):
        known = man["files"].get(p.name, {}).get("sha256")
        excl[str(p.relative_to(ROOT))] = dict(sha256=known or sha256(p), bytes=p.stat().st_size, reason="data file")
    man["excluded_from_phase3_zip"] = excl
    man_path.write_text(json.dumps(man, indent=1))
    files = []
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not any(s in p.parts for s in SKIP_PARTS) and p.suffix != ".pt":
                files.append(p)
    for extra in ("README.md", "PHASE1_REPORT.md", "CLAUDE_PHASE2.md", "CLAUDE_PHASE3.md"):
        if (ROOT / extra).exists():
            files.append(ROOT / extra)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT))
    size = out.stat().st_size
    print(f"wrote {out} ({size / 2 ** 20:.1f} MB, {len(files)} files); {len(excl)} checkpoints/data files listed in MANIFEST")
    if size >= LIMIT_ZIP:
        raise SystemExit("zip exceeds 100 MB")


if __name__ == "__main__":
    main()
