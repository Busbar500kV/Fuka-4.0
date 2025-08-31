#!/usr/bin/env python3
"""
Ping-run generator: writes a tiny, valid run locally and uploads it to GCS.

Creates (locally first):
  data/runs/<RUN_ID>/shards/
    - state_000.parquet
    - events_000.parquet
    - ledger_000.parquet
    - env_000.parquet
  data/runs/<RUN_ID>/manifest.json

Then uploads those files to:
  gs://<BUCKET>/runs/<RUN_ID>/...

Also writes *_index.json (events/state/ledger/env) and updates runs/index.json.

Dependencies: pandas, pyarrow, google-cloud-storage
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage


def write_local_run(base: Path, run_id: str) -> Dict[str, List[Path]]:
    """Create very small shards + manifest under base/data/runs/<run_id>/..."""
    run_dir = base / "data" / "runs" / run_id
    shards = run_dir / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    # state_000: tiny 3D lattice at steps 0,50,100
    rows = []
    for step in (0, 50, 100):
        for x in range(4):
            for y in range(3):
                for z in range(2):
                    rows.append(
                        dict(
                            run_id=run_id,
                            step=step,
                            conn_id=10 + x + y + z,
                            x=float(x),
                            y=float(y),
                            z=float(z),
                            m=0.1 * (1 + x + y + z),
                            T_eff=280.0 + step * 0.01,
                            theta_thr=0.5,
                        )
                    )
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), shards / "state_000.parquet")

    # events_000: a few events per step
    erows = []
    for step in (0, 50, 100):
        for i in range(6):
            erows.append(
                dict(
                    run_id=run_id,
                    step=step,
                    conn_id=i,
                    x=float(i),
                    y=float(i % 3),
                    z=float(i % 2),
                    dF=0.001 * i,
                    dm=0.0001 * i,
                    w_sel=1.2,
                    A_sel=0.8,
                    phi_sel=0.3,
                    T_eff=280.0 + step * 0.01,
                    theta_thr=0.4,
                )
            )
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(erows), preserve_index=False), shards / "events_000.parquet")

    # ledger_000: aggregate lines
    lrows = [
        dict(run_id=run_id, step=s, dF=0.01, c2dm=0.009, Q=0.0, W_cat=0.001, net_flux=0.0, balance_error=1e-6)
        for s in (0, 50, 100)
    ]
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(lrows), preserve_index=False), shards / "ledger_000.parquet")

    # env_000: tiny field
    frows = []
    for step in (0, 100):
        for x in range(4):
            for y in range(3):
                for z in range(2):
                    frows.append(dict(run_id=run_id, step=step, x=float(x), y=float(y), z=float(z), value=0.01 * (x + y + z)))
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(frows), preserve_index=False), shards / "env_000.parquet")

    # manifest.json with *relative* paths (like your recorder)
    manifest = dict(
        run_id=run_id,
        created_at=pd.Timestamp.now().timestamp(),
        shards=[
            dict(table="state", path=f"data/runs/{run_id}/shards/state_000.parquet"),
            dict(table="events", path=f"data/runs/{run_id}/shards/events_000.parquet"),
            dict(table="ledger", path=f"data/runs/{run_id}/shards/ledger_000.parquet"),
            dict(table="env", path=f"data/runs/{run_id}/shards/env_000.parquet"),
        ],
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "state": [shards / "state_000.parquet"],
        "events": [shards / "events_000.parquet"],
        "ledger": [shards / "ledger_000.parquet"],
        "env": [shards / "env_000.parquet"],
        "manifest": [run_dir / "manifest.json"],
    }


def upload_file(bucket: storage.Bucket, src: Path, dst: str, content_type: str | None = None):
    blob = bucket.blob(dst)
    blob.upload_from_filename(str(src), content_type=content_type)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True, help="GCS bucket name (no gs://)")
    p.add_argument("--run_id", required=True, help="Run ID (e.g., FUKA_4_0_PING)")
    p.add_argument("--prefix", default="runs", help='Key prefix in bucket (default: "runs")')
    p.add_argument("--base", default="/root/Fuka-4.0", help="Repo base on VM (for local writes)")
    args = p.parse_args()

    base = Path(args.base)
    files = write_local_run(base, args.run_id)

    client = storage.Client()
    bucket = client.bucket(args.bucket)

    # Upload shards + manifest
    for table, paths in files.items():
        for pth in paths:
            rel = f"{args.prefix}/{args.run_id}/" + str(pth).split(f"data/runs/{args.run_id}/")[-1].lstrip("/").replace("\\", "/")
            upload_file(
                bucket=bucket,
                src=pth,
                dst=rel,
                content_type="application/json" if pth.name.endswith(".json") else "application/octet-stream",
            )
            print("Uploaded:", rel)

    # Write simple *_index.json (exact HTTPS URLs)
    def https_for(name: str) -> str:
        return f"https://storage.googleapis.com/{args.bucket}/{args.prefix}/{args.run_id}/shards/{name}"

    indices = {
        "events_index.json": {"table": "events", "run_id": args.run_id, "files": [https_for("events_000.parquet")]},
        "state_index.json": {"table": "state", "run_id": args.run_id, "files": [https_for("state_000.parquet")]},
        "ledger_index.json": {"table": "ledger", "run_id": args.run_id, "files": [https_for("ledger_000.parquet")]},
        "env_index.json": {"table": "env", "run_id": args.run_id, "files": [https_for("env_000.parquet")]},
    }
    for fname, payload in indices.items():
        tmp = base / f".tmp_{fname}"
        tmp.write_text(json.dumps(payload, indent=2))
        upload_file(bucket, tmp, f"{args.prefix}/{args.run_id}/shards/{fname}", content_type="application/json")
        tmp.unlink(missing_ok=True)
        print("Wrote index:", f"{args.prefix}/{args.run_id}/shards/{fname}")

    # Update runs/index.json
    runs_blob = bucket.blob(f"{args.prefix}/index.json")
    try:
        current = json.loads(runs_blob.download_as_text())
        runs = set(map(str, current.get("runs", [])))
    except Exception:
        runs = set()
    runs.add(args.run_id)
    runs_blob.upload_from_string(json.dumps({"runs": sorted(runs)}, indent=2), content_type="application/json")
    print("Updated runs index:", sorted(runs))

    # Friendly URLs
    print("\n==== READY ====")
    print("Run:", args.run_id)
    print("State:", https_for("state_000.parquet"))
    print("Events:", https_for("events_000.parquet"))
    print("Ledger:", https_for("ledger_000.parquet"))
    print("Env:", https_for("env_000.parquet"))
    print("\nIn Streamlit, set:")
    print('  DATA_URL_PREFIX = "https://storage.googleapis.com/{bucket}"'.format(bucket=args.bucket))
    print("Then pick run:", args.run_id)


if __name__ == "__main__":
    main()