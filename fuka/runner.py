# fuka/runner.py
from __future__ import annotations

"""
Headless runner for Fuka.

Responsibilities
- Load a config, choose/normalize a run_id, and ensure a clean run directory.
- Invoke the engine's headless execution (compatible with both old and new signatures).
- Always rebuild per-table indexes and the run-level manifest on completion.
- Emit clear, machine-parseable logs and non-zero exit codes on hard failures.

This file does not render or upload. Rendering/publishing is handled by external scripts.
"""

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Engine entrypoint (we support both signatures shown in historical code)
try:
    # Typical location in this repo
    from core.engine import run_headless as _engine_run_headless  # type: ignore
except Exception:
    # Fallback if module path differs (older repo layouts)
    from engine import run_headless as _engine_run_headless  # type: ignore

# Index/manifest builder (File 2 you just installed)
try:
    from analytics.build_indices import rebuild_indices as _rebuild_indices  # type: ignore
except Exception as _e:
    _rebuild_indices = None  # type: ignore


# ------------------------------ helpers ------------------------------

def _stamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_run_id(prefix: str = "FUKA_4_0_3D") -> str:
    return f"{prefix}_{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"


def _echo(level: str, msg: str) -> None:
    print(f"[{_stamp()}] [{level}] {msg}", flush=True)


def _ensure_run_dir(data_root: Path, run_id: str) -> Path:
    run_dir = data_root / "runs" / run_id
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_meta(run_dir: Path, meta: dict) -> None:
    meta_path = run_dir / "run_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _find_latest_run(data_root: Path) -> Optional[str]:
    runs_dir = data_root / "runs"
    if not runs_dir.is_dir():
        return None
    cands = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not cands:
        return None
    # newest by mtime
    latest = max(cands, key=lambda p: p.stat().st_mtime)
    return latest.name


def _call_engine(config_path: Path, data_root: Path, run_id: str) -> None:
    """
    Call engine.run_headless with best-effort signature compatibility:
      - Newer: run_headless(config_path: str, data_root: str, run_id: str)
      - Older: run_headless(config_path: str, data_root: str)
    """
    # Some engines honor RUN_ID via env; set it as a courtesy
    os.environ["FUKA_RUN_ID"] = run_id

    try:
        _echo("INFO", f"Calling engine.run_headless(config='{config_path}', data_root='{data_root}', run_id='{run_id}')")
        _engine_run_headless(str(config_path), str(data_root), str(run_id))  # type: ignore[arg-type]
        return
    except TypeError:
        # Fall back to legacy 2-arg signature
        _echo("WARN", "Engine run_headless() does not accept run_id; falling back to (config_path, data_root).")
        _engine_run_headless(str(config_path), str(data_root))  # type: ignore[arg-type]
        return


# ------------------------------ main flow ------------------------------

def run(
    config_path: Path,
    data_root: Path = Path("data"),
    run_id: Optional[str] = None,
    prefix: Optional[str] = None,
    skip_index: bool = False,
) -> str:
    """
    Execute a headless simulation and (optionally) rebuild indices/manifest.

    Returns the resolved run_id.
    """
    resolved_run_id = run_id or _ts_run_id()
    run_dir = _ensure_run_dir(data_root, resolved_run_id)

    _echo("RUN", f"run_id={resolved_run_id}")
    _echo("INFO", f"data_root={data_root}")
    _echo("INFO", f"config={config_path}")
    if prefix:
        _echo("INFO", f"public_prefix={prefix}")

    # Write a minimal meta upfront (helpful for debugging if a run crashes early)
    _write_meta(
        run_dir,
        {
            "run_id": resolved_run_id,
            "created_utc": _stamp(),
            "data_root": str(data_root),
            "config_path": str(config_path),
            "public_prefix": prefix,
            "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "pid": os.getpid(),
        },
    )

    # Execute engine
    try:
        _call_engine(config_path, data_root, resolved_run_id)
    except Exception as e:
        _echo("FATAL", f"Engine failed: {type(e).__name__}: {e}")
        raise

    # Engines that ignore run_id may write into a different/new folder;
    # try to detect that and correct our run_id so downstream steps work.
    if not (data_root / "runs" / resolved_run_id / "shards").glob("*.parquet"):
        latest = _find_latest_run(data_root)
        if latest and latest != resolved_run_id:
            _echo("WARN", f"No shards found under run_id '{resolved_run_id}'. Using latest found run '{latest}'.")
            resolved_run_id = latest
            run_dir = data_root / "runs" / resolved_run_id

    # Index/manifest
    if not skip_index:
        if _rebuild_indices is None:
            _echo("ERROR", "analytics.build_indices not available; skipping index/manifest rebuild.")
        else:
            try:
                _rebuild_indices(str(data_root), resolved_run_id, prefix)
                _echo("OK", f"indices + manifest rebuilt for run_id={resolved_run_id}")
            except Exception as e:
                _echo("ERROR", f"Index/manifest rebuild failed: {type(e).__name__}: {e}")
                # Do not raise — rendering may still proceed via direct shard paths.

    # Final meta (append/overwrite)
    _write_meta(
        run_dir,
        {
            "run_id": resolved_run_id,
            "completed_utc": _stamp(),
            "data_root": str(data_root),
            "config_path": str(config_path),
            "public_prefix": prefix,
        },
    )

    _echo("DONE", f"run_id={resolved_run_id}")
    return resolved_run_id


# ------------------------------ CLI ------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fuka headless runner")
    ap.add_argument("--config", required=True, help="Path to simulation config (JSON/YAML)")
    ap.add_argument("--data_root", default="data", help="Data root (default: data)")
    ap.add_argument("--run_id", default=None, help="Override run id (default: timestamped)")
    ap.add_argument(
        "--prefix",
        default=os.environ.get("DATA_URL_PREFIX", "").strip() or None,
        help="Public HTTPS prefix for indices (e.g., https://storage.googleapis.com/fuka4-runs)",
    )
    ap.add_argument("--skip_index", action="store_true", help="Skip index/manifest rebuild")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run(
            config_path=Path(args.config),
            data_root=Path(args.data_root),
            run_id=args.run_id,
            prefix=args.prefix,
            skip_index=bool(args.skip_index),
        )
    except Exception:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())