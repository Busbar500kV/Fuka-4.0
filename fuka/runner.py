from __future__ import annotations
import os, json, subprocess
from pathlib import Path

from .engine import Engine
from .recorder import ParquetRecorder  # import the class directly


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def run_headless(config_path: str, data_root: str) -> None:
    cfg = _load_cfg(config_path)

    run_id = str(cfg.get("run_name") or cfg.get("run_id") or "FUKA_4_0_RUN")
    steps  = int(cfg.get("steps", 0))
    flush_every = int(cfg.get("io", {}).get("flush_every", 1000))

    # Recorder writes under <data_root>/runs/<run_id>/
    runs_root = os.path.join(data_root, "runs")
    rec = ParquetRecorder(
        data_root=runs_root,
        run_id=run_id,
        flush_every=flush_every,
    )

    # Run engine (pass full cfg so physics/world/io are available)
    eng = Engine(recorder=rec, steps=steps, cfg=cfg)
    eng.run()

    # Post: always ensure local indices+manifest exist for offline UI
    try:
        repo_root = Path(__file__).resolve().parents[1]  # project root
        build_py = repo_root / "analytics" / "build_indices.py"
        if build_py.exists():
            subprocess.run(
                ["python", str(build_py), "--data_root", data_root, "--run_id", run_id],
                check=False
            )
    except Exception:
        pass

    # Optional: publish to GCS if env vars are set
    bucket = os.environ.get("FUKA_GCS_BUCKET")
    if bucket:
        try:
            repo_root = Path(__file__).resolve().parents[1]
            build_py = repo_root / "analytics" / "build_indices.py"
            subprocess.run(
                ["python", str(build_py), "--bucket", bucket, "--run_id", run_id, "--to_gcs",
                 "--publish_runs_index", "--publish_manifest"],
                check=False
            )
        except Exception:
            pass