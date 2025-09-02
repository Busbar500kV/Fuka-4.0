from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import argparse, subprocess, urllib.request, json

TABLES = ["events","spectra","state","ledger","edges","env","catalysts"]

def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

def list_local_files(data_root: Path, run_id: str, table: str) -> List[str]:
    root = data_root / "runs" / run_id / "shards"
    return sorted([str(p) for p in root.glob(f"{table}_*.parquet")])

def list_gcs_objects(bucket: str, run_id: str, table: str) -> List[str]:
    out = subprocess.run(
        ["gsutil", "ls", f"gs://{bucket}/runs/{run_id}/shards/{table}_*.parquet"],
        capture_output=True, text=True, check=False
    ).stdout.strip().splitlines()
    return [u for u in out if u]

def to_https(gs_urls: List[str]) -> List[str]:
    return [u.replace("gs://", "https://storage.googleapis.com/") for u in gs_urls]

def publish_gcs_json(bucket: str, run_id: str, table: str, files_https: List[str]) -> None:
    tmp = Path(f"/tmp/{table}_index.json")
    _write_json(tmp, {"table": table, "run_id": run_id, "files": files_https})
    subprocess.run(["gsutil", "cp", str(tmp), f"gs://{bucket}/runs/{run_id}/shards/{table}_index.json"], check=True)

def fetch_runs_index(bucket: str) -> List[str]:
    url = f"https://storage.googleapis.com/{bucket}/runs/index.json"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
            return list({str(x) for x in data.get("runs", [])})
    except Exception:
        return []

def publish_runs_index(bucket: str, run_id: str) -> None:
    runs = set(fetch_runs_index(bucket))
    runs.add(run_id)
    tmp = Path("/tmp/runs_index.json")
    _write_json(tmp, {"runs": sorted(runs)})
    subprocess.run(["gsutil", "cp", str(tmp), f"gs://{bucket}/runs/index.json"], check=True)

def publish_manifest_gcs(bucket: str, run_id: str) -> None:
    shards = []
    for t in TABLES:
        objs = list_gcs_objects(bucket, run_id, t)
        for u in objs:
            fname = u.split("/")[-1]
            shards.append({"table": t, "path": f"data/runs/{run_id}/shards/{fname}"})
    tmp = Path("/tmp/manifest.json")
    _write_json(tmp, {"run_id": run_id, "shards": sorted(shards, key=lambda s: (s["table"], s["path"]))})
    subprocess.run(["gsutil", "cp", str(tmp), f"gs://{bucket}/runs/{run_id}/manifest.json"], check=True)

def write_local_indices_and_manifest(data_root: Path, run_id: str) -> None:
    out_dir = data_root / "runs" / run_id / "shards"
    for t in TABLES:
        loc = list_local_files(data_root, run_id, t)
        _write_json(out_dir / f"{t}_index.json", {"table": t, "run_id": run_id, "files": loc})
    # manifest: relative paths
    shards_dir = data_root / "runs" / run_id / "shards"
    shards = []
    for t in TABLES:
        for p in sorted(shards_dir.glob(f"{t}_*.parquet")):
            shards.append({"table": t, "path": f"data/runs/{run_id}/shards/{p.name}"})
    _write_json(data_root / "runs" / run_id / "manifest.json",
                {"run_id": run_id, "shards": sorted(shards, key=lambda s: (s['table'], s['path']))})

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--bucket")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--to_gcs", action="store_true")
    ap.add_argument("--publish_runs_index", action="store_true")
    ap.add_argument("--publish_manifest", action="store_true")
    args = ap.parse_args()

    if args.to_gcs:
        if not args.bucket:
            raise SystemExit("--to_gcs requires --bucket")
        for t in TABLES:
            gs = list_gcs_objects(args.bucket, args.run_id, t)
            publish_gcs_json(args.bucket, args.run_id, t, to_https(gs))
        if args.publish_runs_index:
            publish_runs_index(args.bucket, args.run_id)
        if args.publish_manifest:
            publish_manifest_gcs(args.bucket, args.run_id)
    else:
        write_local_indices_and_manifest(Path(args.data_root), args.run_id)

if __name__ == "__main__":
    main()