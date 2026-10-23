#!/usr/bin/env python3
"""Phase 4, item 2: check the numbers in the manuscript against the result files.

Claims file results/paper_claims.json: {"claims": [{"id", "text", "where" (optional: section label or file), "json",
"path", "transform" (optional), "tol", "value" (optional: the number the text states, if it cannot be parsed from
"text")}]}.
  * "text" must occur verbatim in manuscript_v9/main_v9.tex or manuscript_v9/theory/*.tex (whitespace-normalised);
  * "path" is a JSON path into results/<json>: keys separated by '/', list indices as integers, and the selectors
    "*" (all children, gives a list), "@min", "@max", "@median", "@len" applied to the list built so far (plain
    components are always keys, so a key named "median" is addressed as "median");
  * "transform" (applied to the JSON value): "percent" (x 100), "ratio:<path>" (value / value at <path> in the same
    file), "inv" (1/x), "round:<k>" (round to k significant digits for display only), "log10";
  * the stated number is parsed from "value" or else from "text" (first number, LaTeX \\times10^{e} and 1e-3 forms);
    PASS if |stated - json| <= tol |json| (tol relative; "abs:<t>" for an absolute tolerance).
Prints PASS/FAIL per claim, writes results/NUMBER_CHECK.md, exit code 1 on any FAIL.
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MS = ROOT / "manuscript_v9"
CLAIMS = RES / "paper_claims.json"
OUT = RES / "NUMBER_CHECK.md"


def tex_corpus():
    files = [MS / "main_v9.tex"] + sorted((MS / "theory").glob("*.tex"))
    return {str(f.relative_to(ROOT)): re.sub(r"\s+", " ", f.read_text()) for f in files if f.exists()}


def resolve(obj, path):
    cur = [obj]
    multi = False
    for part in [p for p in path.split("/") if p != ""]:
        if part == "*":
            nxt = []
            for c in cur:
                nxt.extend(c.values() if isinstance(c, dict) else c)
            cur, multi = nxt, True
            continue
        if part in ("@min", "@max", "@median", "@len"):
            vals = [float(v) for v in cur]
            cur = [dict(min=min, max=max, median=lambda v: float(np.median(v)), len=len)[part[1:]](vals)]
            multi = False
            continue
        nxt = []
        for c in cur:
            if isinstance(c, list):
                nxt.append(c[int(part)])
            else:
                nxt.append(c[part])
        cur = nxt
    return cur if multi else cur[0]


NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*\\times\s*10\^\{?[-+]?\d+\}?|[eE][-+]?\d+)?")


def parse_number(s):
    m = NUM.search(s.replace("{,}", "").replace("\\,", ""))
    if not m:
        raise ValueError(f"no number in {s!r}")
    t = m.group(0)
    mm = re.match(r"([-+]?\d+(?:\.\d+)?)\s*\\times\s*10\^\{?([-+]?\d+)\}?", t)
    return float(mm.group(1)) * 10 ** int(mm.group(2)) if mm else float(t)


def evaluate(claim, cache):
    f = claim["json"]
    if f not in cache:
        cache[f] = json.loads((RES / f).read_text())
    v = resolve(cache[f], claim["path"])
    for tr in [t for t in claim.get("transform", "").split(";") if t]:
        if tr == "percent":
            v = 100 * v
        elif tr == "inv":
            v = 1 / v
        elif tr == "log10":
            v = float(np.log10(v))
        elif tr.startswith("ratio:"):
            v = v / resolve(cache[f], tr[6:])
        elif tr.startswith("round:"):
            pass
        else:
            raise ValueError(f"unknown transform {tr}")
    return float(v)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default=str(CLAIMS))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out_path = Path(a.out)
    corpus = tex_corpus()
    claims = json.loads(Path(a.claims).read_text())["claims"]
    cache, rows = {}, []
    for c in claims:
        text = re.sub(r"\s+", " ", c["text"])
        where = [k for k, v in corpus.items() if text in v]
        try:
            stated = float(c["value"]) if "value" in c else parse_number(text)
            got = evaluate(c, cache)
            tol = c.get("tol", 0.01)
            ok_num = (abs(stated - got) <= float(tol[4:])) if isinstance(tol, str) and tol.startswith("abs:") \
                else abs(stated - got) <= float(tol) * abs(got)
            err = None
        except Exception as ex:                                     # noqa: BLE001
            stated = got = None; ok_num = False; err = f"{type(ex).__name__}: {ex}"
        status = "PASS" if (ok_num and where) else "FAIL"
        reason = [] if status == "PASS" else ([] if where else ["text not found in the manuscript"]) + \
            ([] if ok_num else [err or "value differs"])
        rows.append(dict(id=c.get("id"), status=status, text=c["text"], stated=stated, json_value=got, json=c["json"],
                         path=c["path"], where=where, reason="; ".join(reason)))
        print(f"{status}  {c.get('id', '')}: stated {stated}  json {got}  ({c['json']}:{c['path']})" + (f"  -- {'; '.join(reason)}" if reason else ""))
    fails = [r for r in rows if r["status"] == "FAIL"]
    lines = [f"# Number check: manuscript vs result files", "",
             f"Claims checked: {len(rows)}; passed: {len(rows) - len(fails)}; failed: {len(fails)}.",
             "Manuscript files: " + ", ".join(corpus), ""]
    if fails:
        lines += ["| id | text in the manuscript | stated | result file says | where | reason |", "|---|---|---|---|---|---|"]
        for r in fails:
            lines.append(f"| {r['id']} | `{r['text']}` | {r['stated']} | {r['json_value']} ({r['json']}: `{r['path']}`) | "
                         f"{', '.join(r['where']) or '-'} | {r['reason']} |")
    out_path.write_text("\n".join(lines) + "\n")
    out_path.with_suffix(".json").write_text(json.dumps(dict(n=len(rows), n_fail=len(fails), rows=rows), indent=1))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
