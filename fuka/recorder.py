from __future__ import annotations
import time, json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class Recorder:
    def __init__(self, run_id: str, root: Path):
        self.run_id = run_id
        self.root = Path(root)
        self.run_dir = self.root / "runs" / run_id
        (self.run_dir / "shards").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self._events_buf: List[Dict[str, Any]] = []
        self._spectra_buf: List[Dict[str, Any]] = []
        self._state_buf: List[Dict[str, Any]] = []
        self._ledger_buf: List[Dict[str, Any]] = []

        self._counts = {"events":0, "spectra":0, "state":0, "ledger":0}
        self._shard_idx = {"events":0, "spectra":0, "state":0, "ledger":0}

        # write manifest skeleton
        (self.run_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_id, "created_at": time.time(), "shards":[]
        }, indent=2))

    def log_event(self, **row):
        row["run_id"] = self.run_id
        self._events_buf.append(row)
        self._counts["events"] += 1

    def log_spectra(self, **row):
        row["run_id"] = self.run_id
        self._spectra_buf.append(row)
        self._counts["spectra"] += 1

    def log_state(self, **row):
        row["run_id"] = self.run_id
        self._state_buf.append(row)
        self._counts["state"] += 1

    def log_ledger(self, **row):
        row["run_id"] = self.run_id
        self._ledger_buf.append(row)
        self._counts["ledger"] += 1

    def _flush(self, name: str, buf: List[Dict[str, Any]]):
        if not buf: return None
        df = pd.DataFrame(buf)
        shard_idx = self._shard_idx[name]
        shard_path = self.run_dir / "shards" / f"{name}_{shard_idx:03d}.parquet"
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, shard_path)
        self._shard_idx[name] += 1
        buf.clear()
        return str(shard_path)

    def flush_all(self):
        written = {}
        written["events"] = self._flush("events", self._events_buf)
        written["spectra"] = self._flush("spectra", self._spectra_buf)
        written["state"] = self._flush("state", self._state_buf)
        written["ledger"] = self._flush("ledger", self._ledger_buf)

        # append to manifest
        manifest_path = self.run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for k, p in written.items():
            if p:
                manifest["shards"].append({"table": k, "path": p})
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def finalize(self):
        self.flush_all()