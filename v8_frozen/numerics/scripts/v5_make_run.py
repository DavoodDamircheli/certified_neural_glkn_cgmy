"""Create a v5 run directory with manifest (roadmap §1, M0 / gate G0).

    PYTHONPATH=src:experiments python3 scripts/v5_make_run.py --label g0_dryrun \
        [--population validation30] [--evidence-baseline PATH] [--test-log PATH]

Writes results/free_integrated_v5/<run_id>/ with manifest.json,
resolved_config.json, queries.csv (the declared population(s) used),
source_hashes.json, evidence_snapshot.json, run.log, summary.md. No numerical
study is executed; every study is recorded as "not run". If --evidence-baseline
is given, the current evidence snapshot is compared against it and any
modified/deleted file aborts the run (G0: old evidence unchanged).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from common import run_manifest as rm  # noqa: E402
from common.methods import NUMERICAL_SCOPE  # noqa: E402


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--population", action="append", default=None)
    ap.add_argument("--evidence-baseline", type=Path, default=None)
    ap.add_argument("--test-log", type=Path, default=None)
    ap.add_argument("--study-status", action="append", default=[],
                    help="name=status, e.g. g0_dry_run=completed")
    args = ap.parse_args(argv)

    resolved = rm.resolve_config()
    wf = resolved["workflow_config"]
    populations = rm.declare_populations()
    used = args.population or []
    for name in used:
        if name not in populations:
            raise SystemExit(f"undeclared population {name!r}")

    evidence = rm.evidence_snapshot()
    evidence_diff = None
    if args.evidence_baseline is not None:
        evidence_diff = rm.compare_evidence(json.loads(args.evidence_baseline.read_text()), evidence)
        if evidence_diff["modified"] or evidence_diff["deleted"]:
            raise SystemExit(f"G0 violation: pre-existing evidence changed: {evidence_diff}")

    run_id = rm.new_run_id(args.label)
    run_dir = rm.create_run_dir(run_id)
    studies = dict(s.split("=", 1) for s in args.study_status)
    manifest = rm.build_manifest(
        run_id, method_id=wf["method_id"], studies=studies, seeds=wf["seeds"],
        resolved_config=resolved, populations=populations, population_used=used,
    )
    manifest["evidence_snapshot_sha256"] = evidence["tree_sha256"]
    manifest["evidence_vs_baseline"] = evidence_diff if evidence_diff is not None else "no baseline given"

    rm.write_json(run_dir / "manifest.json", manifest)
    rm.write_json(run_dir / "resolved_config.json", resolved)
    rm.write_json(run_dir / "source_hashes.json", rm.source_hashes())
    rm.write_json(run_dir / "evidence_snapshot.json", evidence)
    rows = [dict(r, population=name) for name in used for r in populations[name]["rows"]]
    if rows:
        import csv
        with open(run_dir / "queries.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    if args.test_log is not None:
        shutil.copyfile(args.test_log, run_dir / "test_log.txt")

    log = [
        f"run_id={run_id}",
        f"method_id={manifest['method_id']}",
        f"git_head={manifest['git']['head']} dirty={manifest['git']['dirty']}",
        f"source_tree_sha256={manifest['source_tree_sha256']}",
        f"populations_verified={sorted(populations)} used={used}",
        f"evidence_files={evidence['n_files']} evidence_vs_baseline={manifest['evidence_vs_baseline']}",
        f"threads={manifest['threads']}",
    ]
    (run_dir / "run.log").write_text("\n".join(log) + "\n")
    summary = [
        f"# v5 run {run_id}", "",
        f"- method: `{manifest['method_id']}` ({manifest['coordinate_space']})",
        f"- numerical scope: {NUMERICAL_SCOPE}",
        f"- source: `{manifest['git']['head']}` (dirty={manifest['git']['dirty']}), "
        f"tree sha256 `{manifest['source_tree_sha256'][:16]}…`",
        f"- populations verified against frozen hashes: {', '.join(sorted(populations))}; used: {used or 'none'}",
        "", "| study | status |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in manifest["studies"].items()],
    ]
    (run_dir / "summary.md").write_text("\n".join(summary) + "\n")
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
