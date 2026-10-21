#!/usr/bin/env python3
"""Phase 2 deliverable (CLAUDE_PHASE2.md, "Deliverable"): phase2_results_<date>.zip with cgmy_cert, tests, scripts,
results, figures_v9, tables_v9, the production checkpoints (<= 50 MB each), DEVIATIONS.md, NOTES_FOR_PAPER.md,
MANIFEST.json, PHASE2_REPORT.md.  Excluded (hash listed in results/MANIFEST.json -> excluded_from_zip): data/
(trajectories, bases, tensors), the pilot checkpoints and any file > 50 MB.
"""
from __future__ import annotations

import json, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.eim import sha256  # noqa: E402

LIMIT = 50 * 2 ** 20
INCLUDE_DIRS = ("cgmy_cert", "tests", "scripts", "results", "figures_v9", "tables_v9")
SKIP_PARTS = ("__pycache__", ".pytest_cache")


def main():
    date = time.strftime("%Y%m%d")
    out = ROOT / f"phase2_results_{date}.zip"
    man_path = ROOT / "results" / "MANIFEST.json"
    man = json.loads(man_path.read_text())
    excluded = {}
    files = []
    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if not p.is_file() or any(s in p.parts for s in SKIP_PARTS):
                continue
            if p.stat().st_size > LIMIT:
                excluded[str(p.relative_to(ROOT))] = dict(sha256=sha256(p), bytes=p.stat().st_size, reason="> 50 MB")
                continue
            files.append(p)
    for p in sorted((ROOT / "checkpoints").glob("*.pt")):          # production checkpoints (pilot is a subdirectory)
        if p.stat().st_size <= LIMIT:
            files.append(p)
        else:
            excluded[str(p.relative_to(ROOT))] = dict(sha256=sha256(p), bytes=p.stat().st_size, reason="> 50 MB")
    for p in sorted((ROOT / "checkpoints" / "pilot_N512").glob("*.pt")):
        excluded[str(p.relative_to(ROOT))] = dict(sha256=sha256(p), bytes=p.stat().st_size, reason="pilot checkpoint")
    for p in sorted((ROOT / "data").rglob("*.npz")):
        excluded[str(p.relative_to(ROOT))] = dict(sha256=man["files"].get(p.name, {}).get("sha256") or sha256(p),
                                                 bytes=p.stat().st_size, reason="data (trajectories / bases / tensors)")
    man["excluded_from_zip"] = excluded
    man_path.write_text(json.dumps(man, indent=1))
    for extra in ("PHASE1_REPORT.md", "README.md", "CLAUDE_PHASE2.md"):
        if (ROOT / extra).exists():
            files.append(ROOT / extra)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT))
    print(f"wrote {out} ({out.stat().st_size / 2 ** 20:.1f} MB, {len(files)} files); excluded {len(excluded)} files "
          f"(hashes in results/MANIFEST.json -> excluded_from_zip)")


if __name__ == "__main__":
    main()
