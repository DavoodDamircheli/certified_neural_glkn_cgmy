#!/usr/bin/env python3
"""Phase 4, item 2: check the numbers in the manuscript against the result files.

Claims file results/paper_claims.json: {"claims": [{"id", "section", "text", "json", "path" | "expr", "transform"
(optional), "value" (optional), "tol" (optional), "rel" (optional), "note" (optional)}]}.
  * "text" must occur verbatim in manuscript_v9/main_v9.tex or manuscript_v9/theory/*.tex (whitespace-normalised);
  * "path" is a JSON path into results/<json>: keys separated by '/', list indices as integers, and the selectors
    "*" (all children, gives a list), "@min", "@max", "@median", "@len" applied to the list built so far (plain
    components are always keys, so a key named "median" is addressed as "median");
  * "expr" (instead of "path"): a Python expression evaluated with J = the loaded result file, R(name) = any other
    result file, np = numpy; used for counts, ratios and derived quantities;
  * "transform" (applied to the value): "percent" (x 100), "ratio:<path>" (value / value at <path> in the same
    file), "inv" (1/x), "log10";
  * the stated number is "value" (a number or a string such as "1.7e-9") or else the first number in "text"
    (LaTeX \\times10^{e} and 1e-3 forms);
  * "tol": relative tolerance (default 0.01), "abs:<t>" absolute, or "round" = half a unit of the last stated digit
    (the stated number is the result rounded);
  * "rel": "eq" (default), "le" (the text says the quantity is at most / below the stated number: PASS if
    result <= stated, up to the tolerance), "ge" (at least);
  * "json": null marks a number that no result file holds (status FAIL, kind "no source"; "note" says where the
    number comes from, if anywhere).
A FAIL is classified as "rounding" (off by at most one unit of the last stated digit), "substantive", "no source" or
"text" (the quoted text is not in the manuscript).  Prints PASS/FAIL per claim, writes results/NUMBER_CHECK.md and
.json, exit code 1 on any FAIL.
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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


def _load(name, cache):
    if name not in cache:
        cache[name] = json.loads((RES / name).read_text())
    return cache[name]


def evaluate(claim, cache):
    J = _load(claim["json"], cache)
    if "expr" in claim:
        v = eval(claim["expr"], {"J": J, "R": lambda n: _load(n, cache), "np": np, "len": len, "sum": sum,
                                 "min": min, "max": max, "abs": abs, "sorted": sorted, "re": re})          # noqa: S307
    else:
        v = resolve(J, claim["path"])
    for tr in [t for t in claim.get("transform", "").split(";") if t]:
        if tr == "percent":
            v = 100 * v
        elif tr == "inv":
            v = 1 / v
        elif tr == "log10":
            v = float(np.log10(v))
        elif tr.startswith("ratio:"):
            v = v / resolve(J, tr[6:])
        else:
            raise ValueError(f"unknown transform {tr}")
    return float(v)


def last_digit_unit(s):
    """Unit of the last stated digit of a number written as '0.33', '1.7e-9', '48800', '1.7\\times10^{-9}'."""
    s = str(s).replace("{,}", "").replace("\\,", "").replace(",", "")
    m = re.match(r"\s*[-+]?(\d+)(?:\.(\d+))?\s*(?:(?:[eE]|\\times\s*10\^\{?)([-+]?\d+)\}?)?", s)
    if not m:
        raise ValueError(f"cannot read the digits of {s!r}")
    dec = len(m.group(2) or "")
    exp = int(m.group(3) or 0)
    return 10.0 ** (exp - dec)


def stated_value(c, text):
    if "value" in c:
        v = c["value"]
        return (float(v) if not isinstance(v, str) else parse_number(v)), (v if isinstance(v, str) else None)
    m = NUM.search(text.replace("{,}", "").replace("\\,", ""))
    return parse_number(text), (m.group(0) if m else None)


def check(c, stated, raw, got):
    tol = c.get("tol", 0.01)
    if tol == "round":
        if raw is None:
            raise ValueError("tol 'round' needs the stated digits (give value as a string)")
        t_abs = 0.5 * last_digit_unit(raw) * (1 + 1e-9)
    elif isinstance(tol, str) and tol.startswith("abs:"):
        t_abs = float(tol[4:])
    else:
        t_abs = float(tol) * abs(got)
    rel = c.get("rel", "eq")
    if rel == "le":
        return got <= stated + t_abs
    if rel == "ge":
        return got >= stated - t_abs
    return abs(stated - got) <= t_abs


def kind_of_fail(c, stated, raw, got):
    if raw is None or c.get("rel", "eq") != "eq":
        return "substantive"
    try:
        return "rounding" if abs(stated - got) <= 1.0 * last_digit_unit(raw) * (1 + 1e-9) else "substantive"
    except ValueError:
        return "substantive"


def fmt(x):
    return "-" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))


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
        stated = got = None
        kind, err = None, None
        try:
            stated, raw = stated_value(c, text)
            if c.get("json") is None:
                ok_num, kind = False, "no source"
            else:
                got = evaluate(c, cache)
                ok_num = check(c, stated, raw, got)
                if not ok_num:
                    kind = kind_of_fail(c, stated, raw, got)
        except Exception as ex:                                     # noqa: BLE001
            ok_num, kind, err = False, "substantive", f"{type(ex).__name__}: {ex}"
        if not where:
            kind = "text"
        status = "PASS" if (ok_num and where) else "FAIL"
        src = "-" if c.get("json") is None else f"{c['json']}: {c.get('path') or c.get('expr')}"
        rows.append(dict(id=c.get("id"), section=c.get("section", ""), status=status, kind=kind if status == "FAIL" else None,
                         text=c["text"], stated=stated, json_value=got, rel=c.get("rel", "eq"), tol=c.get("tol", 0.01),
                         source=src, where=where, note=c.get("note", ""), error=err))
        print(f"{status}  {c.get('id', '')}: stated {fmt(stated)}  result {fmt(got)}  ({src})"
              + (f"  -- {kind}" if status == "FAIL" else "") + (f" [{err}]" if err else ""))
    fails = [r for r in rows if r["status"] == "FAIL"]
    kinds = {k: [r for r in fails if r["kind"] == k] for k in ("substantive", "no source", "rounding", "text")}
    lines = ["# Number check: manuscript vs result files", "",
             "Generated by `scripts/check_numbers.py` from `results/paper_claims.json`; do not edit by hand.", "",
             f"Claims checked: {len(rows)}; passed: {len(rows) - len(fails)}; failed: {len(fails)} "
             f"({len(kinds['substantive'])} substantive, {len(kinds['no source'])} without a result file, "
             f"{len(kinds['rounding'])} rounding-level, {len(kinds['text'])} text not found).", "",
             "Manuscript files: " + ", ".join(f"`{k}`" for k in corpus), "",
             "A claim passes when the stated number agrees with the result file within its tolerance (`round` = half a unit "
             "of the last stated digit; `le`/`ge` = the text states a bound). Rounding-level = off by at most one unit of "
             "the last stated digit.", ""]
    heads = {"substantive": "Substantive mismatches", "no source": "Numbers that no result file holds",
             "rounding": "Rounding-level mismatches", "text": "Quoted text not found in the manuscript"}
    for k in ("substantive", "no source", "rounding", "text"):
        if not kinds[k]:
            continue
        lines += [f"## {heads[k]} ({len(kinds[k])})", "",
                  "| id | section | text in the manuscript | stated | result file | source | note |", "|---|---|---|---|---|---|---|"]
        for r in kinds[k]:
            rel = {"le": " (bound: <=)", "ge": " (bound: >=)"}.get(r["rel"], "")
            lines.append(f"| {r['id']} | {r['section']} | `{r['text']}` | {fmt(r['stated'])}{rel} | {fmt(r['json_value'])} | "
                         f"`{r['source']}` | {(r['note'] + (' ' + r['error'] if r['error'] else '')).strip()} |")
        lines.append("")
    lines += [f"## Passed ({len(rows) - len(fails)})", "", "| id | section | stated | result file | source |", "|---|---|---|---|---|"]
    for r in rows:
        if r["status"] == "PASS":
            lines.append(f"| {r['id']} | {r['section']} | {fmt(r['stated'])} | {fmt(r['json_value'])} | `{r['source']}` |")
    out_path.write_text("\n".join(lines) + "\n")
    out_path.with_suffix(".json").write_text(json.dumps(dict(n=len(rows), n_fail=len(fails),
                                                             n_fail_by_kind={k: len(v) for k, v in kinds.items()},
                                                             rows=rows), indent=1))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
