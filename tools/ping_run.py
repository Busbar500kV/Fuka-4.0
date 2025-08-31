#!/usr/bin/env python3
"""
Ping: create a tiny test run locally, upload to GCS, write *_index.json and runs/index.json.

Usage:
  python tools/ping_run.py --bucket fuka4-runs --run_id FUKA_4_0_PING
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import pandas as pd
from google.cloud import storage


def make_local_shards(base: Path, run_id: str) -> None:
    ldir = base / "data" / "runs" / run_id / "shards"
    ldir.mkdir(parents=True, exist_ok=True)

    # --- state
    rows = []
    for step in (0, 50, 100):
        for x in range(0, 4):
            for y in range(0, 3):
                for z in range(0, 2):
                    rows.append({
                        "run_id": run_id, "step": step, "conn_id": 10 + x + y + z,
                        "x": x, "y": y, "z": z,
                        "m": 0.1 * (1 + x + y + z),
                        "T_eff": 280 + step * 0.01,
                        "theta_thr": 0.5,
                    })
    pd.DataFrame(rows).to_parquet(ldir / "state_000.parquet", index=False)

    # --- events
    erows = []
    for step in (0, 50, 100):
        for i in range(6):
            erows.append({
                "run_id": run_id, "step": step, "conn_id": i,
                "x": float(i), "y": float(i % 3), "z": float(i % 2),
                "dF": 0.001 * i, "dm": 0.0001 * i,
                "w_sel": 1.2, "A_sel": 0.8, "phi_sel": 0.3,
                "T_eff": 280 + step * 0.01, "theta_thr": 0.4,
            })
    pd.DataFrame(erows).to_parquet(ldir / "events_000.parquet", index=False)

    # --- ledger
    ldgr = [{
        "run_id": run_id, "step": s, "dF": 0.01, "c2dm": 0.009,
        "Q": 0.0, "W_cat": 0.001, "net_flux": 0.0, "balance_error": 1e-6
    } for s in (0, 50, 100)]
    pd.DataFrame(ldgr).to_parquet(ldir / "ledger_000.parquet", index=False)

    # --- env
    env = []
    for step in (0, 100):
        for x in range(0, 4):
            for y in range(0, 3):
                for z in range(0, 2):
                    env.append({
                        "run_id": run_id, "step": step,
                        "x": x, "y": y, "z": z, "value": 0.01 * (x + y + z),
                    })
    pd.DataFrame(env).to_parquet(ldir / "env_000.parquet", index=False)

    # --- manifest (relative paths like recorder writes)
    manifest = {
        "run_id": run_id,
        "created_at": pd.Timestamp.now().timestamp(),
        "shards": [
            {"table": "state",  "path": f"data/runs/{run_id}/shards/state_000.parquet"},
            {"table": "events", "path": f"data/runs/{run_id}/shards/events_000.parquet"},
            {"table": "ledger", "path": f"data/runs/{run_id}/shards/ledger_000.parquet"},
            {"table": "env",    "path": f"data/runs/{run_id}/shards/env_000.parquet"},
        ],
    }
    (base / "data" / "runs" / run_id / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"✅ Local shards & manifest written under {ldir}")


def upload_and_index(base: Path, bucket_name: str, run_id: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # rsync-like: upload the run dir recursively
    local_run = base / "data" / "runs" / run_id
    for path in local_run.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(base)
        blob = bucket.blob(str(rel).replace("\\", "/"))
        blob.upload_from_filename(str(path))
    print(f"✅ Uploaded run to gs://{bucket_name}/runs/{run_id}/")

    # write per-table *_index.json (exact HTTPS URLs)
    https = f"https://storage.googleapis.com/{bucket_name}"
    tables = {
        "events": ["events_000.parquet"],
        "state":  ["state_000.parquet"],
        "ledger": ["ledger_000.parquet"],
        "env":    ["env_000.parquet"],
    }
    for t, files in tables.items():
        index = {
            "table": t,
            "run_id": run_id,
            "files": [f"{https}/runs/{run_id}/shards/{f}" for f in files],
        }
        bucket.blob(f"runs/{run_id}/shards/{t}_index.json").upload_from_string(
            json.dumps(index, indent=2), content_type="application/json"
        )
    print("✅ Wrote per-table index JSONs")

    # update runs/index.json (append run_id if missing)
    runs_blob = bucket.blob("runs/index.json")
    try:
        runs = json.loads(runs_blob.download_as_text()).get("runs", [])
    except Exception:
        runs = []
    runs = sorted(set([*map(str, runs), run_id]))
    runs_blob.upload_from_string(
        json.dumps({"runs": runs}, indent=2), content_type="application/json"
    )
    print("✅ Updated runs/index.json:", runs)

    # Print handy URLs
    print("\nURLs:")
    print(f"  Manifest: {https}/runs/{run_id}/manifest.json")
    print(f"  State index:  {https}/runs/{run_id}/shards/state_index.json")
    print(f"  Events index: {https}/runs/{run_id}/shards/events_index.json")
    print(f"  Ledger index: {https}/runs/{run_id}/shards/ledger_index.json")
    print(f"  Env index:    {https}/runs/{run_id}/shards/env_index.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="GCS bucket name, e.g. fuka4-runs")
    ap.add_argument("--run_id", required=True, help="Run ID to create, e.g. FUKA_4_0_PING")
    ap.add_argument("--base", default="/root/Fuka-4.0", help="Repo base directory on VM")
    args = ap.parse_args()

    base = Path(args.base)
    make_local_shards(base, args.run_id)
    upload_and_index(base, args.bucket, args.run_id)
    print("\nDone. Set DATA_URL_PREFIX to the bucket root (e.g., https://storage.googleapis.com/fuka4-runs) and select the run in the UI.")


if __name__ == "__main__":
    main()