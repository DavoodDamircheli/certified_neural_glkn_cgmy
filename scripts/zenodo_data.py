"""Zenodo REST helper for the Phase 5 data record (never publishes on its own).

The token is read from the environment variable ZENODO_TOKEN only; it is never written to disk or printed, and for
uploads it is handed to curl as a header on stdin (so it is not in argv or in any URL).

  python3 scripts/zenodo_data.py --sandbox create            # deposition with reserved DOI -> results/zenodo_ids.json
  python3 scripts/zenodo_data.py --sandbox upload FILE...    # curl --upload-file into the bucket, with retries
  python3 scripts/zenodo_data.py --sandbox check FILE...     # compare Zenodo's MD5 with the local MD5
  python3 scripts/zenodo_data.py --sandbox metadata META.json
  python3 scripts/zenodo_data.py --sandbox show
  python3 scripts/zenodo_data.py --sandbox publish           # only after an explicit "yes" from the author
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
IDS = ROOT / "results" / "zenodo_ids.json"
BASES = {"sandbox": "https://sandbox.zenodo.org", "zenodo": "https://zenodo.org"}


def token():
    t = os.environ.get("ZENODO_TOKEN")
    if not t:
        sys.exit("ZENODO_TOKEN is not set")
    return t


def hdr():
    return {"Authorization": f"Bearer {token()}", "Content-Type": "application/json"}


def load_ids():
    return json.loads(IDS.read_text()) if IDS.exists() else {}


def save_ids(ids):
    IDS.write_text(json.dumps(ids, indent=2) + "\n")


def dep_url(base, rec):
    return f"{BASES[base]}/api/deposit/depositions/{rec['deposition_id']}"


def ok(r):
    if r.status_code >= 400:
        sys.exit(f"HTTP {r.status_code}: {r.text[:2000]}")
    return r.json() if r.text else {}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cmd_create(base, ids, a):
    if base in ids and ids[base].get("data", {}).get("deposition_id") and not a.force:
        sys.exit(f"{base}: data deposition already recorded in {IDS} (use --force for a new one)")
    d = ok(requests.post(f"{BASES[base]}/api/deposit/depositions", headers=hdr(),
                         json={"metadata": {"prereserve_doi": True}}))
    rec = dict(deposition_id=d["id"], bucket=d["links"]["bucket"], reserved_doi=d["metadata"]["prereserve_doi"]["doi"],
               html=d["links"]["html"], created=d["created"])
    ids.setdefault(base, {})["data"] = rec
    save_ids(ids)
    print(json.dumps(rec, indent=2))


def cmd_upload(base, ids, a):
    rec = ids[base]["data"]
    for f in a.files:
        p = Path(f)
        url = f"{rec['bucket']}/{p.name}"
        for attempt in range(1, a.retries + 1):
            print(f"[{time.strftime('%H:%M:%S')}] {p.name} ({p.stat().st_size / 1e9:.2f} GB) attempt {attempt}", flush=True)
            r = subprocess.run(["curl", "-sS", "--fail-with-body", "-H", "@-", "--upload-file", str(p), url,
                                "-o", "/dev/null", "-w", "%{http_code} %{speed_upload}\n"],
                               input=f"Authorization: Bearer {token()}\n", text=True, capture_output=True)
            print("  ", r.stdout.strip(), r.stderr.strip()[:500], flush=True)
            if r.returncode == 0:
                break
            time.sleep(min(60 * attempt, 300))
        else:
            sys.exit(f"upload of {p.name} failed after {a.retries} attempts")


def cmd_check(base, ids, a):
    rec = ids[base]["data"]
    files = {f["key"]: f for f in ok(requests.get(rec["bucket"], headers=hdr())).get("contents", [])}
    bad = 0
    for f in a.files:
        p = Path(f)
        remote = files.get(p.name, {}).get("checksum", "missing")
        local = "md5:" + md5(p)
        flag = "OK" if remote == local else "MISMATCH"
        bad += flag != "OK"
        print(f"{flag:8s} {p.name:32s} local {local} zenodo {remote}")
    extra = sorted(set(files) - {Path(f).name for f in a.files})
    if extra:
        print("on Zenodo but not checked:", extra)
    sys.exit(1 if bad else 0)


def cmd_metadata(base, ids, a):
    meta = json.loads(Path(a.meta).read_text())
    rec = ids[base]["data"]
    cur = ok(requests.get(dep_url(base, rec), headers=hdr()))["metadata"]
    meta["prereserve_doi"] = cur.get("prereserve_doi", True)
    d = ok(requests.put(dep_url(base, rec), headers=hdr(), json={"metadata": meta}))
    print(json.dumps(d["metadata"], indent=2))


def cmd_show(base, ids, a):
    d = ok(requests.get(dep_url(base, ids[base]["data"]), headers=hdr()))
    print(json.dumps({k: d.get(k) for k in ("id", "state", "submitted", "doi", "links")}, indent=2))
    for f in d.get("files", []):
        print(f"  {f['filename']:32s} {int(f['filesize']) / 1e9:8.3f} GB  md5:{f['checksum']}")


def cmd_publish(base, ids, a):
    rec = ids[base]["data"]
    d = ok(requests.post(dep_url(base, rec) + "/actions/publish", headers=hdr()))
    rec.update(published_doi=d.get("doi"), record_url=d["links"].get("record_html") or d["links"].get("html"))
    save_ids(ids)
    print(json.dumps(rec, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("create"); c.add_argument("--force", action="store_true")
    u = sp.add_parser("upload"); u.add_argument("files", nargs="+"); u.add_argument("--retries", type=int, default=5)
    k = sp.add_parser("check"); k.add_argument("files", nargs="+")
    m = sp.add_parser("metadata"); m.add_argument("meta")
    sp.add_parser("show"); sp.add_parser("publish")
    a = ap.parse_args()
    base = "sandbox" if a.sandbox else "zenodo"
    globals()[f"cmd_{a.cmd}"](base, load_ids(), a)


if __name__ == "__main__":
    main()
