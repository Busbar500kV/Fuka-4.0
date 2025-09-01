from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
from .engine import Engine
from .recorder import ParquetRecorder as Recorder
from .catalog import update_catalog

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
Recorder = ParquetRecorder
__all__ = ["ParquetRecorder", "Recorder"]