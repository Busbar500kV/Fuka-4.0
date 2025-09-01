from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
from .engine import Engine
from .recorder import ParquetRecorder
from .catalog import update_catalog

import os, json

def _load_cfg(path):
    with open(path, "r") as f:
        return json.load(f)

def run_headless(config_path: str, data_root: str):
    cfg = _load_cfg(config_path)

    # pull run_id & steps from your config (adjust keys if different)
    run_id = cfg.get("run_name", "RUN")
    steps  = int(cfg.get("steps", 0))                # Engine expects an int here

    # ParquetRecorder wants a *data_root* (directory) + *run_id*
    # Give it the base “runs” directory; it will manage subfolders.
    rec = ParquetRecorder(
        data_root=os.path.join(data_root, "runs"),
        run_id=run_id,
        flush_every=cfg.get("io", {}).get("flush_every", 1000),
    )

    # Engine expects recorder first and steps second. Use keywords for safety.
    eng = Engine(recorder=rec, steps=steps)
    eng.run()

def run_headless(config_path: str, data_root: str):
    cfg = json.loads(Path(config_path).read_text())
    run_id = cfg["run_name"]
    rec = Recorder(run_id, Path(data_root))
    eng = Engine(cfg, rec)

    for _ in range(cfg["steps"]):
        eng.step()
        if (eng.step_idx % cfg["io"]["flush_every"]) == 0:
            rec.flush_all()

    rec.finalize()
    # update catalog
    update_catalog(Path(data_root) / "catalog.duckdb",
                   Path(data_root) / "runs" / run_id)
                
            # At the end of fuka/recorder.py

__all__ = ["ParquetRecorder", "Recorder"]