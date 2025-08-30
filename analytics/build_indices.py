"""
Build and publish JSON indices for all tables so the Streamlit UI can load
Parquet files over HTTPS without wildcards.

Two modes:
  1) Local -> write indices to local run folder (no GCS)
  2) GCS   -> use gsutil to list objects and upload indices to the bucket

Usage (Colab, recommended):
  !python analytics/build_indices.py --bucket fuka4-runs --run_id FUKA_4_0_SMOKE --to_gcs

If you want local-only:
  !python analytics/build_indices.py --data_root data --run_id FUKA_4_0_SMOKE
"""

from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from typing import List


TABLES = ["events", "spectra", "state", "ledger", "edges", "env"]


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def list_local_files(data_root: Path, run_id: str, table: str) -> List[str]:
    root = data_root / "runs" / run_id / "shards"
    return sorted([str(p) for p in root.glob(f"{table}_*.parquet")])


def list_gcs_objects(bucket: str, run_id: str, table: str) -> List[str]:
    # returns gs:// URLs; convert to https later
    out = subprocess.run(
        ["gsutil", "ls", f"gs://{bucket}/runs/{run_id}/shards/{table}_*.parquet"],
        capture_output=True, text=True, check=False
    ).stdout.strip().splitlines()
    return [u for u in out if u]


def to_https(gs_urls: List[str]) -> List[str]:
    return [u.replace("gs://", "https://storage.googleapis.com/") for u in gs_urls]


def publish_gcs_json(bucket: str, run_id: str, table: str, files_https: List[str]):
    tmp = Path(f"/tmp/{table}_index.json")
    _write_json(tmp, {"table": table, "run_id": run_id, "files": files_https})
    subprocess.run(["gsutil", "cp", str(tmp), f"gs://{bucket}/runs/{run_id}/shards/{table}_index.json"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--bucket", help="GCS bucket name")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--to_gcs", action="store_true", help="Publish indices to GCS (requires --bucket)")
    ap.add_argument("--publish_runs_index", action="store_true", help="Also publish /runs/index.json")
    args = ap.parse_args()

    if args.to_gcs:
        if not args.bucket:
            raise SystemExit("--to_gcs requires --bucket")
        # per-table indices to GCS
        for t in TABLES:
            gs = list_gcs_objects(args.bucket, args.run_id, t)
            https = to_https(gs)
            publish_gcs_json(args.bucket, args.run_id, t, https)
            print(f"[GCS] {t}: {len(https)} files indexed")

        if args.publish_runs_index:
            # naïve: just ensure run_id is present
            tmp = Path("/tmp/runs_index.json")
            _write_json(tmp, {"runs": [args.run_id]})
            subprocess.run(["gsutil", "cp", str(tmp), f"gs://{args.bucket}/runs/index.json"], check=True)
            print("[GCS] runs/index.json published")
    else:
        # local-only indices
        data_root = Path(args.data_root)
        out_dir = data_root / "runs" / args.run_id / "shards"
        for t in TABLES:
            loc = list_local_files(data_root, args.run_id, t)
            payload = {"table": t, "run_id": args.run_id, "files": loc}
            _write_json(out_dir / f"{t}_index.json", payload)
            print(f"[LOCAL] {t}: {len(loc)} files indexed at {out_dir}/{t}_index.json")


if __name__ == "__main__":
    main()