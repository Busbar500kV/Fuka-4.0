# fuka/runner.py
from __future__ import annotations
import os, json
from pathlib import Path

from .engine import Engine
from .recorder import ParquetRecorder  # import the class directly

def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def run_headless(config_path: str, data_root: str) -> None:
    cfg = _load_cfg(config_path)

    # pull run_id & steps (adjust keys if your config uses different names)
    run_id = cfg.get("run_name", "RUN")
    steps  = int(cfg.get("steps", 0))

    # optional: io.flush_every in config
    flush_every = int(cfg.get("io", {}).get("flush_every", 1000))

    # ParquetRecorder(data_root: str, run_id: str, flush_every: int = 1000)
    rec = ParquetRecorder(
        data_root=os.path.join(data_root, "runs"),
        run_id=run_id,
        flush_every=flush_every,
    )

    # Engine(recorder, steps, ...)
    eng = Engine(recorder=rec, steps=steps)
    eng.run()