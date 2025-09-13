# analytics/build_indices.py
import argparse, json, os, pathlib, glob, datetime, subprocess

def to_https(gs_url: str) -> str:
    return gs_url.replace("gs://", "https://storage.googleapis.com/")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--prefer_gcs_for_manifest", action="store_true")
    args = ap.parse_args()

    data_root = pathlib.Path(args.data_root)
    run_id = args.run_id

    # Local run paths
    run_dir = data_root / "runs" / run_id
    shards_dir = run_dir / "shards"
    if not shards_dir.exists():
        print(f"[build_indices] ERROR: Shards directory not found: {shards_dir}")
        return 2

    # Tables present (auto-detect)
    parquet_files = list(shards_dir.glob("*.parquet"))
    tables = sorted({p.stem.split("_")[0] for p in parquet_files})

    # Per-table indices
    indices = {}
    for t in tables:
        files = sorted(shards_dir.glob(f"{t}_*.parquet"))
        indices[t] = [str(p) for p in files]
        idx_path = shards_dir / f"{t}_index.json"
        idx_path.write_text(json.dumps({"table": t, "run_id": run_id, "files": [str(p) for p in files]}, indent=2))

    # Manifest (relative paths for UI)
    shards = []
    for t, files in indices.items():
        for f in files:
            # Make relative to repo data root for UI
            rel = str(pathlib.Path("data") / "runs" / run_id / "shards" / pathlib.Path(f).name)
            shards.append({"table": t, "path": rel})

    manifest = {
        "run_id": run_id,
        "root": f"data/runs/{run_id}",
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
        "shards": sorted(shards, key=lambda s: (s["table"], s["path"])),
        "tables": tables,
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # If a bucket is defined, mirror to GCS and update top-level runs index
    bucket = os.environ.get("FUKA_GCS_BUCKET")
    if bucket:
        # sync shards + indices + manifest
        subprocess.run(["gsutil","-m","rsync","-r", str(run_dir), f"{bucket}/runs/{run_id}"], check=False)

        # update runs index.json
        runs_idx_gs = f"{bucket}/runs/index.json"
        existing = []
        try:
            out = subprocess.run(["gsutil","cat", runs_idx_gs], capture_output=True, text=True)
            if out.returncode == 0:
                existing = json.loads(out.stdout).get("runs", [])
        except Exception:
            existing = []
        if run_id not in existing:
            existing.append(run_id)
        existing = sorted(set(existing))
        tmp = pathlib.Path("/tmp/fuka_runs_index.json")
        tmp.write_text(json.dumps({"runs": existing}, indent=2))
        subprocess.run(["gsutil","cp", str(tmp), runs_idx_gs], check=False)

        # Optionally advertise HTTPS location used by UI (not strictly needed)
        # print("[build_indices] manifest:", to_https(f"{bucket}/runs/{run_id}/manifest.json"))

    print("[build_indices] OK.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())