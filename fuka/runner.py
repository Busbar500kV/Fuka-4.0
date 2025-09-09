# fuka/runner.py

from __future__ import annotations
import os, json, subprocess, sys
from pathlib import Path
from typing import Dict, List

from .engine import Engine
from .recorder import ParquetRecorder  # import the class directly


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _debug(msg: str) -> None:
    print(f"[RUNNER] {msg}", flush=True)


def _run_subprocess(cmd: List[str], name: str) -> bool:
    try:
        _debug(f"exec: {' '.join(cmd)}")
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            _debug(f"{name} FAILED (code {res.returncode})\nSTDERR:\n{res.stderr.strip()}")
            return False
        _debug(f"{name} OK")
        return True
    except Exception as e:
        _debug(f"{name} EXCEPTION: {e}")
        return False


def _relative_https(bucket_https_base: str, p: Path) -> str:
    # p is an absolute or relative local path like "<data_root>/runs/<RUN>/shards/state_000.parquet"
    # We want HTTPS pointing to "runs/<RUN>/shards/state_000.parquet"
    # Streamlit app prepends DATA_URL_PREFIX; but our *_index.json on GCS should hold absolute HTTPS.
    # We'll build absolute HTTPS here: base + relative under bucket root.
    # Expect layout: <data_root>/runs/<RUN>/...
    # Find the "runs" component and take relative from there.
    parts = list(p.as_posix().split("/"))
    try:
        k = parts.index("runs")
        rel = "/".join(parts[k:])  # e.g., "runs/<RUN>/shards/file.parquet"
    except ValueError:
        rel = p.as_posix()
    return f"{bucket_https_base}/{rel}"


def _build_local_table_indices(run_dir: Path) -> Dict[str, List[str]]:
    """
    Build per-table indices under: <run_dir>/shards/<table>_index.json
    Returns a dict table -> [relative paths under 'data/...' or absolute].
    """
    shards_dir = run_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    by_table: Dict[str, List[str]] = {}
    for f in sorted(shards_dir.glob("*.parquet")):
        name = f.name
        if "_" not in name:
            continue
        table = name.split("_", 1)[0]
        by_table.setdefault(table, []).append(f)

    # Write local indices (relative 'data/...' paths so local UI can also read via manifest_to_https)
    out: Dict[str, List[str]] = {}
    for table, files in by_table.items():
        # local JSON uses relative paths (the Streamlit manifest_to_https will convert to HTTPS if needed)
        rels = [f"data/runs/{run_dir.name}/shards/{f.name}" for f in files]
        idx_path = shards_dir / f"{table}_index.json"
        idx_path.write_text(json.dumps({"files": rels}, indent=2))
        out[table] = rels
    return out


def _ensure_manifest_from_shards(run_dir: Path) -> None:
    """
    Ensure manifest.json exists and contains all shard parquet entries.
    """
    manifest_path = run_dir / "manifest.json"
    shards_dir = run_dir / "shards"
    shards = []
    for f in sorted(shards_dir.glob("*.parquet")):
        name = f.name
        if "_" not in name:
            continue
        table = name.split("_", 1)[0]
        # manifest expects "path" relative under "data/..."
        path_rel = f"data/runs/{run_dir.name}/shards/{name}"
        shards.append({"table": table, "path": path_rel})
    payload = {
        "run_id": run_dir.name,
        "shards": shards,
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(manifest_path)


def _build_local_runs_index(runs_root: Path) -> Path:
    """
    Build runs/index.json listing all subdirectories under runs_root.
    """
    runs_index = runs_root / "index.json"
    runs = []
    for p in sorted(runs_root.glob("*/")):
        if (p / "shards").exists():
            runs.append(p.name)
    tmp = runs_index.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"runs": runs}, indent=2))
    tmp.replace(runs_index)
    return runs_index


def _maybe_publish_to_gcs(bucket: str, runs_root: Path, run_id: str) -> None:
    """
    Upload manifest + per-table indices + runs/index.json to GCS.
    Optional public read if FUKA_PUBLISH_PUBLIC=1.
    """
    make_public = os.environ.get("FUKA_PUBLISH_PUBLIC", "") == "1"
    bucket_uri = f"gs://{bucket}"
    bucket_https = f"https://storage.googleapis.com/{bucket}"

    run_dir = runs_root / run_id
    shards_dir = run_dir / "shards"

    # Upload manifest
    _run_subprocess(["gsutil", "cp", str(run_dir / "manifest.json"),
                     f"{bucket_uri}/runs/{run_id}/manifest.json"], "GCS manifest upload")

    # Upload per-table indices
    for idx in shards_dir.glob("*_index.json"):
        _run_subprocess(["gsutil", "cp", str(idx),
                         f"{bucket_uri}/runs/{run_id}/shards/{idx.name}"], f"GCS {idx.name} upload")

    # Upload runs/index.json
    _run_subprocess(["gsutil", "cp", str(runs_root / "index.json"),
                     f"{bucket_uri}/runs/index.json"], "GCS runs/index.json upload")

    if make_public:
        _run_subprocess(["gsutil", "acl", "ch", "-u", "AllUsers:R",
                         f"{bucket_uri}/runs/{run_id}/manifest.json"], "GCS ACL manifest public")
        for idx in shards_dir.glob("*_index.json"):
            _run_subprocess(["gsutil", "acl", "ch", "-u", "AllUsers:R",
                             f"{bucket_uri}/runs/{run_id}/shards/{idx.name}"], f"GCS ACL {idx.name} public")
        _run_subprocess(["gsutil", "acl", "ch", "-u", "AllUsers:R",
                         f"{bucket_uri}/runs/index.json"], "GCS ACL runs/index.json public")


def _publish_with_script_or_fallback(data_root: Path, run_id: str) -> None:
    """
    Try analytics/build_indices.py first (legacy path).
    If missing/fails, build indices/manifest/runs-index locally in Python.
    Then optionally publish to GCS if FUKA_GCS_BUCKET is set.
    """
    runs_root = data_root / "runs"
    repo_root = Path(__file__).resolve().parents[1]
    build_py = repo_root / "analytics" / "build_indices.py"

    used_script = False
    if build_py.exists():
        # Local indices build
        ok_local = _run_subprocess(
            ["python", str(build_py), "--data_root", str(data_root), "--run_id", run_id],
            "build_indices.py (local)"
        )
        used_script = True
        if not ok_local:
            _debug("build_indices.py local failed; falling back to internal Python builder.")
    else:
        _debug("analytics/build_indices.py not found; using internal Python builder.")

    # Fallback: build indices/manifest locally in Python if needed
    try:
        # Always ensure *_index.json exist (idempotent)
        _build_local_table_indices(runs_root / run_id)
        # Always ensure manifest exists and reflects shards
        _ensure_manifest_from_shards(runs_root / run_id)
        # Always rebuild runs/index.json from local runs dir
        _build_local_runs_index(runs_root)
        _debug("Local indices/manifest/runs-index built OK.")
    except Exception as e:
        _debug(f"Local index build EXCEPTION: {e}")

    # Publish to GCS if configured
    bucket = os.environ.get("FUKA_GCS_BUCKET", "").strip()
    if bucket:
        if build_py.exists():
            ok_gcs = _run_subprocess(
                ["python", str(build_py), "--bucket", bucket, "--run_id", run_id, "--to_gcs",
                 "--publish_runs_index", "--publish_manifest"],
                "build_indices.py (GCS publish)"
            )
            if not ok_gcs:
                _debug("build_indices.py GCS publish failed; falling back to direct gsutil publish.")
                _maybe_publish_to_gcs(bucket, runs_root, run_id)
        else:
            _maybe_publish_to_gcs(bucket, runs_root, run_id)
    else:
        _debug("FUKA_GCS_BUCKET not set; skipping GCS publish.")


def run_headless(config_path: str, data_root: str) -> None:
    cfg = _load_cfg(config_path)

    run_id = str(cfg.get("run_name") or cfg.get("run_id") or "FUKA_4_0_RUN")
    steps  = int(cfg.get("steps", 0))
    flush_every = int(cfg.get("io", {}).get("flush_every", 1000))

    # Recorder writes under <data_root>/runs/<run_id>/
    runs_root = Path(data_root) / "runs"
    rec = ParquetRecorder(
        data_root=str(runs_root),
        run_id=run_id,
        flush_every=flush_every,
    )

    # Run engine (pass full cfg so physics/world/io/catalysts are available)
    eng = Engine(recorder=rec, steps=steps, cfg=cfg)
    eng.run()

    # Post: build indices/manifest locally (or via legacy script), and publish if configured
    _publish_with_script_or_fallback(Path(data_root), run_id)

    _debug(f"Headless run complete: {run_id}")