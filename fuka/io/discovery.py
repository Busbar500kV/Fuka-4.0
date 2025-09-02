# fuka/io/discovery.py
from __future__ import annotations
import argparse, json, os, subprocess, sys, urllib.request
from typing import Dict, Iterable, List, Optional

HTTP_TIMEOUT = 8.0

def _load_json(url: str) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

def _is_httpish(u: str) -> bool:
    return isinstance(u, str) and (
        u.startswith("http://") or u.startswith("https://") or u.startswith("gs://")
    )

def _to_https(u: str) -> str:
    if u.startswith("gs://"):
        return u.replace("gs://", "https://storage.googleapis.com/")
    return u

def _bucket_from_prefix(prefix: str) -> str:
    # e.g. https://storage.googleapis.com/fuka4-runs -> fuka4-runs
    return prefix.rstrip("/").split("/")[-1]

def files_from_index(prefix: str, run_id: str, table: str) -> List[str]:
    """Read *_index.json and keep only http(s)/gs entries -> HTTPS."""
    url = f"{prefix.rstrip('/')}/runs/{run_id}/shards/{table}_index.json"
    j = _load_json(url)
    if not j or "files" not in j or not isinstance(j["files"], list):
        return []
    return [_to_https(x) for x in j["files"] if _is_httpish(x)]

def files_from_manifest(prefix: str, run_id: str, table: str) -> List[str]:
    """Read manifest.json and convert data/ relative paths to HTTPS."""
    url = f"{prefix.rstrip('/')}/runs/{run_id}/manifest.json"
    m = _load_json(url)
    if not m:
        return []
    out: List[str] = []
    for sh in m.get("shards", []):
        if isinstance(sh, dict) and sh.get("table") == table:
            p = (sh.get("path") or "").lstrip("/")  # "data/runs/<...>/shards/..."
            if p.startswith("data/"):
                p = p[len("data/"):]
            if p:
                out.append(f"{prefix.rstrip('/')}/{p}")
    # Keep http(s)/gs only, and normalize to https
    out = [_to_https(x) for x in out if _is_httpish(x)]
    return sorted(out)

def files_from_gsutil(prefix: str, run_id: str, table: str) -> List[str]:
    """Optional fallback: list directly via gsutil (requires VM creds)."""
    bucket = _bucket_from_prefix(prefix)
    pat = f"gs://{bucket}/runs/{run_id}/shards/{table}_*.parquet"
    try:
        p = subprocess.run(["gsutil", "ls", pat], capture_output=True, text=True)
        if p.returncode != 0:
            return []
        return [_to_https(u.strip()) for u in p.stdout.splitlines() if u.strip()]
    except Exception:
        return []

def discover_table(prefix: str, run_id: str, table: str, allow_gsutil: bool=False) -> List[str]:
    """Order: index (filtered) -> manifest -> gsutil (optional)."""
    for fn in (files_from_index, files_from_manifest):
        lst = fn(prefix, run_id, table)
        if lst:
            return lst
    if allow_gsutil or os.environ.get("FUKA_ALLOW_GSUTIL") in ("1","true","True"):
        lst = files_from_gsutil(prefix, run_id, table)
        if lst:
            return lst
    return []

def discover(prefix: str, run_id: str, tables: Iterable[str], allow_gsutil: bool=False) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for t in tables:
        out[t] = discover_table(prefix, run_id, t, allow_gsutil=allow_gsutil)
    return out

# ---- CLI ----
def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser("fuka.io.discovery")
    ap.add_argument("--prefix", required=True, help="HTTPS prefix, e.g. https://storage.googleapis.com/fuka4-runs")
    ap.add_argument("--run_id", required=True, help="Run id, e.g. FUKA_4_0_3D_YYYYMMDDTHHMMSSZ")
    ap.add_argument("--tables", default="state,edges", help="Comma-separated tables to discover")
    ap.add_argument("--allow-gsutil", action="store_true", help="Enable gsutil fallback if index/manifest empty")
    return ap.parse_args(argv)

def main(argv: Optional[List[str]]=None) -> int:
    ns = _parse_args(argv or sys.argv[1:])
    tabs = [t.strip() for t in ns.tables.split(",") if t.strip()]
    res = discover(ns.prefix, ns.run_id, tabs, allow_gsutil=ns.allow_gsutil)
    # Always print a valid JSON object (never empty output)
    sys.stdout.write(json.dumps(res, indent=2))
    sys.stdout.flush()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())