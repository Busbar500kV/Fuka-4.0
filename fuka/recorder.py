from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import time
import json

TABLES = ("events", "spectra", "state", "ledger", "edges", "env")


@dataclass
class ParquetRecorder:
    """
    Writes table buffers to Parquet shards under:
        <data_root>/<run_id>/shards/<table>_000.parquet

    Also maintains a manifest.json with the schema expected by the Streamlit UI:
        {
          "run_id": "...",
          "created_at": <epoch_seconds_float>,
          "created_at_iso": "YYYY-MM-DDTHH:MM:SSZ",
          "shards": [{"table":"events","path":"data/runs/<RUN_ID>/shards/events_000.parquet"}, ...]
        }

    Notes:
    - The "path" entries are RELATIVE under "data/..." so the UI can prepend DATA_URL_PREFIX.
    - Every row you log must include a "step" integer; the UI discovers min/max by scanning it.
    """
    data_root: str
    run_id: str
    flush_every: int = 1000

    def __post_init__(self) -> None:
        # Normalize dirs
        self.root = Path(self.data_root)
        self.run_dir = self.root / self.run_id
        self.shards_dir = self.run_dir / "shards"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        # in-memory buffers and next shard index per table
        self._buf: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TABLES}
        self._next_idx: Dict[str, int] = {t: 0 for t in TABLES}

        # manifest
        now = time.time()
        self.manifest_path = self.run_dir / "manifest.json"
        self._manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "created_at": float(now),
            "created_at_iso": pd.Timestamp.utcfromtimestamp(now).isoformat() + "Z",
            "shards": []
        }
        self._write_manifest()  # create early so the UI can see a stub

    # ---------------- logging APIs (engine calls these) ----------------
    def log_event(self, **row: Any) -> None:
        self._buf["events"].append(self._with_common(row)); self._maybe_flush("events")

    def log_spectrum(self, **row: Any) -> None:
        self._buf["spectra"].append(self._with_common(row)); self._maybe_flush("spectra")

    def log_state(self, **row: Any) -> None:
        self._buf["state"].append(self._with_common(row)); self._maybe_flush("state")

    def log_ledger(self, **row: Any) -> None:
        self._buf["ledger"].append(self._with_common(row)); self._maybe_flush("ledger")

    def log_edge(self, **row: Any) -> None:
        self._buf["edges"].append(self._with_common(row)); self._maybe_flush("edges")

    def log_env(self, **row: Any) -> None:
        self._buf["env"].append(self._with_common(row)); self._maybe_flush("env")

    # ---------------- housekeeping ----------------
    def _with_common(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Require "step" for all rows
        if "step" not in row:
            raise ValueError("All rows must include 'step'")
        return row

    def _maybe_flush(self, table: str) -> None:
        if len(self._buf[table]) >= self.flush_every:
            self._flush_table(table)

    def _flush_table(self, table: str) -> None:
        buf = self._buf[table]
        if not buf:
            return
        idx = self._next_idx[table]
        fname = f"{table}_{idx:03d}.parquet"
        path = self.shards_dir / fname

        # to DataFrame with stable column ordering (step first if present)
        df = pd.DataFrame(buf)
        if "step" in df.columns:
            cols = ["step"] + [c for c in df.columns if c != "step"]
            df = df[cols]

        df.to_parquet(path, index=False)

        # update manifest (relative path under data/)
        rel = f"data/runs/{self.run_id}/shards/{fname}"
        self._manifest["shards"].append({"table": table, "path": rel})
        self._write_manifest()

        # advance
        self._buf[table].clear()
        self._next_idx[table] = idx + 1

    def _write_manifest(self) -> None:
        # write atomically
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2))
        tmp.replace(self.manifest_path)

    def finalize(self) -> None:
        # flush all tables
        for t in TABLES:
            if self._buf[t]:
                self._flush_table(t)
        # one last manifest write
        self._write_manifest()