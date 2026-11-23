#!/usr/bin/env python3
"""Phase 4, item 1.5: results/MANIFEST.json -> 'archive' section: SHA-256 and size of every tracked file in the
archive (code, tests, scripts, results, figures, tables, paper, legacy_v8, fixtures, top-level files) and of every excluded data file and checkpoint, with
the command that regenerates each.  The earlier per-file provenance entries ('files') are kept.
"""
from __future__ import annotations

import json, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cgmy_cert.eim import sha256  # noqa: E402

ARCHIVE_DIRS = ("cgmy_cert", "tests", "scripts", "results", "figures", "tables", "paper", "legacy_v8")
TOP = ("README.md", "requirements.txt", "build.sh", "Makefile", "LICENSE", "LICENSE-DATA", "CITATION.cff", ".zenodo.json",
       ".gitignore")
SKIP = ("__pycache__", ".pytest_cache")
MAN = ROOT / "results" / "MANIFEST.json"

REGEN = [  # (regex on the relative path, command)
    (r"data/references_N(\d+)_NT240_(seed0_4000|seed0_800|validation30)\.npz",
     "python3 scripts/gen_references.py --N {0} --NT 240 --nproc 20   (the 800 file is the first 800 of the 4000 set)"),
    (r"data/references_N1024_NT240_(bench120|ref30)\.npz", "python3 scripts/gen_references.py --N 1024 --NT 240 --extra bench120 ref30"),
    (r"data/pod_N512_.*\.npz", "python3 scripts/run_pod.py basis"),
    (r"data/pod_N1024_.*_r128\.npz", "python3 scripts/run_N1024.py podext"),
    (r"data/pod_N1024_.*\.npz", "python3 scripts/run_N1024.py pod"),
    (r"data/eim/eim_N(\d+)_K100\.npz", "python3 scripts/run_eim.py build select --N {0}"),
    (r"data/redcert_N512_r48_n\d+\.npz", "python3 scripts/run_reduced_cert.py build"),
    (r"data/redcert_N512_.*CYm.*\.npz", "python3 scripts/run_cells.py --build CYm:3x3x2 [--Q 29 37]"),
    (r"data/redcert_N1024_.*_b128\.npz", "python3 scripts/run_N1024.py tensors_ext ; python3 scripts/run_E1.py tensors (Q = 37)"),
    (r"data/redcert_N1024_.*\.npz", "python3 scripts/run_N1024.py tensors ; python3 scripts/run_E1.py tensors (Q = 37)"),
    (r"data/podnn_prep_N1024_.*\.npz", "CGMY_N=1024 python3 scripts/train_podnn.py prep"),
    (r"data/podnn_prep_.*\.npz", "python3 scripts/train_podnn.py prep"),
    (r"data/podnn_tbasis_N1024_basis(\d+)\.npz", "CGMY_N=1024 python3 scripts/train_podnn.py tbasis --n {0}"),
    (r"checkpoints/pilot_N512/.*\.pt", "pilot (N = 512 sweep stopped by decision; not regenerated): scripts/train_podnn.py run ..."),
    (r"checkpoints/podnn_N1024_.*\.pt", "python3 scripts/sweep_N1024.py (CGMY_N=1024 scripts/train_podnn.py run ...)"),
]


def regen(rel):
    for pat, cmd in REGEN:
        m = re.fullmatch(pat, rel)
        if m:
            return cmd.format(*m.groups())
    return None


def main():
    man = json.loads(MAN.read_text())
    arch = {}
    tracked = subprocess.run(["git", "ls-files", "--", *ARCHIVE_DIRS], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split("\n")
    for rel in sorted(t for t in tracked if t):
        p = ROOT / rel
        if p.is_file() and not any(s in p.parts for s in SKIP) and p.resolve() != MAN.resolve():
            arch[rel] = dict(sha256=sha256(p), bytes=p.stat().st_size)
    for t in TOP:
        p = ROOT / t
        if p.exists():
            arch[t] = dict(sha256=sha256(p), bytes=p.stat().st_size)
    excl = {}
    known = man.get("files", {})
    for pat in ("data/**/*.npz", "checkpoints/**/*.pt"):
        for p in sorted(ROOT.glob(pat)):
            rel = str(p.relative_to(ROOT))
            excl[rel] = dict(sha256=known.get(p.name, {}).get("sha256") or sha256(p), bytes=p.stat().st_size,
                             regenerate=regen(rel) or "see the producing script in results/MANIFEST.json -> files")
    missing = [k for k, v in excl.items() if v["regenerate"].startswith("see")]
    man["archive"] = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), files=arch, n_files=len(arch),
                          total_bytes=int(sum(v["bytes"] for v in arch.values())),
                          note="results/MANIFEST.json itself is not hashed (it contains this list)")
    man["excluded"] = dict(files=excl, n_files=len(excl), total_bytes=int(sum(v["bytes"] for v in excl.values())),
                           without_specific_command=missing)
    MAN.write_text(json.dumps(man, indent=1))
    print(f"archive: {len(arch)} files, {man['archive']['total_bytes'] / 2 ** 20:.1f} MB; excluded: {len(excl)} files, "
          f"{man['excluded']['total_bytes'] / 2 ** 30:.1f} GB; without a specific command: {len(missing)}")


if __name__ == "__main__":
    main()
