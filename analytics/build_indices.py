# analytics/build_indices.py
# FUKA 4.0: canonical index + manifest builder (local + GCS publish)
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Tuple

# ------------------------------- Config -------------------------------

TABLE_PREFIXES = {
    "state":   "state_",
    "edges":   "edges_",
    "env":     "env_",
    "events":  "events_",
    "spectra": "spectra_",
    "ledger":  "ledger_",
}

# ------------------------------ Helpers -------------------------------

def p(*args, **kw):
    print(*args, **kw, flush=True)

def run_ok(cmd: List[str]) -> Tuple[int, str, str]:
    p(f"[RUNNER] exec: {' '.join(cmd)}")
    c = subprocess.run(cmd, capture_output=True, text=True)
    return c.returncode, c.stdout, c.stderr

def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def is_parquet_name(name: str) -> bool:
    return name.endswith(".parquet")

def detect_table(fname: str) -> str:
    for t, pref in TABLE_PREFIXES.items():
        if fname.startswith(pref):
            return t
    return "unknown"

def http_from_gs(gs_url: str) -> str:
    # gs://bucket/path → https://storage.googleapis.com/bucket/path
    return gs_url.replace("gs://", "https://storage.googleapis.com/")

def list_gs(bucket: str, run_id: str) -> List[str]:
    # Return full gs:// URLs for all shards under run
    pat = f"{bucket.rstrip('/')}/runs/{run_id}/shards/*.parquet"
    code, out, err = run_ok(["gsutil", "ls", pat])
    if code != 0:
        raise RuntimeError(f"gsutil ls failed: {err.strip()}")
    return [u.strip() for u in out.splitlines() if u.strip()]

def gs_cp(local: pathlib.Path, dest_gs: str) -> None:
    code, out, err = run_ok(["gsutil", "cp", str(local), dest_gs])
    if code != 0:
        raise RuntimeError(f"gsutil cp failed: {err.strip()}")

def gs_cat_json(gs_url: string) -> Dict:
    code, out, err = run_ok(["gsutil", "cat", gs_url])
    if code != 0:
        raise RuntimeError(f"gsutil cat failed: {err.strip()}")
    try:
        return json.loads(out)
    except Exception:
        return {}

# ----------------------------- Core logic -----------------------------

@dataclasses.dataclass
class Context:
    data_root: pathlib.Path
    run_id: str
    # Optional bucket in form "gs://bucket-name"
    gcs_bucket: str | None

    @property
    def run_dir(self) -> pathlib.Path:
        return self.data_root / "runs" / self.run_id

    @property
    def shards_dir(self) -> pathlib.Path:
        return self.run_dir / "shards"

    @property
    def local_manifest(self) -> pathlib.Path:
        return self.run_dir / "manifest.json"

    def gcs_runs_prefix(self) -> str:
        assert self.gcs_bucket, "gcs_bucket not set"
        return f"{self.gcs_bucket.rstrip('/')}/runs/{self.run_id}"

    def gcs_shards_prefix(self) -> str:
        return self.gcs_runs_prefix() + "/shards"

def find_local_shards(ctx: Context) -> List[pathlib.Path]:
    if not ctx.shards_dir.exists():
        return []
    return sorted([p for p in ctx.shards_dir.glob("*.parquet") if p.is_file()])

def build_manifest_from_local(ctx: Context) -> Dict:
    shard_files = find_local_shards(ctx)
    shards = []
    for path in shard_files:
        fname = path.name
        table = detect_table(fname)
        shards.append({"table": table, "path": f"data/runs/{ctx.run_id}/shards/{fname}"})
    manifest = {
        "run_id": ctx.run_id,
        "root": f"data/runs/{ctx.run_id}",
        "shards": shards,
        "tables": sorted({s["table"] for s in shards if s["table"] != "unknown"}),
    }
    return manifest

def build_manifest_from_gcs(ctx: Context) -> Dict:
    # Fall back to GCS listing if local shards are absent
    urls = list_gs(ctx.gcs_bucket, ctx.run_id)  # type: ignore[arg-type]
    shards = []
    for u in sorted(urls):
        fname = u.split("/")[-1]
        table = detect_table(fname)
        shards.append({"table": table, "path": f"data/runs/{ctx.run_id}/shards/{fname}"})
    manifest = {
        "run_id": ctx.run_id,
        "root": f"data/runs/{ctx.run_id}",
        "shards": shards,
        "tables": sorted({s["table"] for s in shards if s["table"] != "unknown"}),
    }
    return manifest

def write_local_manifest(ctx: Context, manifest: Dict) -> None:
    ensure_dir(ctx.run_dir)
    ctx.local_manifest.write_text(json.dumps(manifest, indent=2))
    p(f"[RUNNER] Local manifest written: {ctx.local_manifest}")

def publish_manifest_gcs(ctx: Context, manifest: Dict) -> None:
    tmp = pathlib.Path(f"/tmp/{ctx.run_id}_manifest.json")
    tmp.write_text(json.dumps(manifest, indent=2))
    gs_cp(tmp, f"{ctx.gcs_runs_prefix()}/manifest.json")  # type: ignore[union-attr]
    p(f"[RUNNER] Uploaded manifest.json -> {ctx.gcs_runs_prefix()}/manifest.json")  # type: ignore[union-attr]

def write_local_table_indexes(ctx: Context, manifest: Dict) -> Dict[str, pathlib.Path]:
    """Write per-table *_index.json locally referencing HTTPS (if bucket present) else relative paths."""
    out_paths: Dict[str, pathlib.Path] = {}
    by_table: Dict[str, List[str]] = {}
    for s in manifest.get("shards", []):
        t = s["table"]; rel = s["path"]
        by_table.setdefault(t, []).append(rel)

    for table, rels in by_table.items():
        # local index stores HTTPS if publishing to GCS, else relative paths
        if ctx.gcs_bucket:
            # Convert rel to gs:// and then to https
            https_files = []
            for rel in rels:
                fname = rel.split("/")[-1]
                https = http_from_gs(f"{ctx.gcs_shards_prefix()}/{fname}")  # type: ignore[union-attr]
                https_files.append(https)
            payload = {"table": table, "run_id": ctx.run_id, "files": https_files}
        else:
            payload = {"table": table, "run_id": ctx.run_id, "files": rels}

        out = ctx.shards_dir / f"{table}_index.json"
        ensure_dir(ctx.shards_dir)
        out.write_text(json.dumps(payload, indent=2))
        out_paths[table] = out
        p(f"[RUNNER] Local index: {out}")

    return out_paths

def publish_table_indexes_gcs(ctx: Context, local_paths: Dict[str, pathlib.Path]) -> None:
    for table, path_ in local_paths.items():
        gs_cp(path_, f"{ctx.gcs_shards_prefix()}/{table}_index.json")  # type: ignore[union-attr]
        p(f"[RUNNER] Uploaded {table}_index.json -> {ctx.gcs_shards_prefix()}/{table}_index.json")  # type: ignore[union-attr]

def publish_runs_index_gcs(ctx: Context) -> None:
    # Maintain runs/index.json list
    runs_index_gs = f"{ctx.gcs_bucket.rstrip('/')}/runs/index.json"  # type: ignore[union-attr]
    try:
        current = gs_cat_json(runs_index_gs)
        runs = set(current.get("runs", []))
    except Exception:
        runs = set()
    runs.add(ctx.run_id)
    tmp = pathlib.Path("/tmp/runs_index.json")
    tmp.write_text(json.dumps({"runs": sorted(runs)}, indent=2))
    gs_cp(tmp, runs_index_gs)
    p(f"[RUNNER] Updated runs/index.json")

# ------------------------------- CLI ----------------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build local indices + manifest; publish to GCS if configured.")
    ap.add_argument("--data_root", required=True, help="Root of local data dir (contains runs/<RUN_ID>)")
    ap.add_argument("--run_id", required=True, help="Run ID to index")
    ap.add_argument("--prefer_gcs_for_manifest", action="store_true",
                    help="If set and GCS is configured, derive manifest from GCS listing (helps when local shards are pruned).")
    args = ap.parse_args(argv)

    data_root = pathlib.Path(args.data_root).resolve()
    gcs_bucket = os.environ.get("FUKA_GCS_BUCKET")  # e.g. gs://fuka4-runs
    ctx = Context(data_root=data_root, run_id=args.run_id, gcs_bucket=gcs_bucket)

    # --------- Build manifest (local preferred; can fall back to GCS) ----------
    if ctx.gcs_bucket and args.prefer_gcs_for_manifest:
        try:
            manifest = build_manifest_from_gcs(ctx)
            p("[RUNNER] Manifest derived from GCS listing.")
        except Exception as e:
            p(f"[RUNNER] WARN: GCS listing failed ({e}); falling back to local shards.")
            manifest = build_manifest_from_local(ctx)
    else:
        manifest = build_manifest_from_local(ctx)
        if not manifest.get("shards") and ctx.gcs_bucket:
            p("[RUNNER] No local shards; deriving manifest from GCS.")
            manifest = build_manifest_from_gcs(ctx)

    # ---------------- Write local manifest + indexes -------------------
    write_local_manifest(ctx, manifest)
    local_idx_paths = write_local_table_indexes(ctx, manifest)

    # -------------------------- GCS publish ----------------------------
    if ctx.gcs_bucket:
        try:
            publish_table_indexes_gcs(ctx, local_idx_paths)
            publish_manifest_gcs(ctx, manifest)
            publish_runs_index_gcs(ctx)
            p("[RUNNER] GCS publish OK.")
        except Exception as e:
            p(f"[RUNNER] ERROR: GCS publish failed: {e}")
            return 2
    else:
        p("[RUNNER] FUKA_GCS_BUCKET not set; skipped GCS publish.")

    p("[RUNNER] Local indices/manifest complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())