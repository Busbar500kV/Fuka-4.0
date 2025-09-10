# fuka/recorder.py
from __future__ import annotations

"""
Parquet-backed recorder for Fuka.

Features
- Deterministic folder layout: data/runs/<run_id>/{shards,logs}
- Buffered per-table writes to *.parquet shards (append-by-batch, rotate by row count)
- Stable schemas (first batch defines order; later batches permute to match)
- Writes run-level manifest.json and per-table *_index.json on finalize()
- Optional HTTPS prefix for indices via env DATA_URL_PREFIX or ctor arg
- Idempotent finalize() (safe to call more than once)

Tables commonly used:
  state:   step,int | x,y,z,int | value,float
  edges:   step,int | x0..z1,int | v0,v1,float | [deposit?,kappa?]
  events:  step,int | x,y,z,int | value,float
  catalysts, spectra, ledger, env: flexible (will persist provided fields)

Notes
- We avoid keeping files open between calls (safer for long headless runs).
- We don't depend on pyarrow schemas being predeclared; we normalize column order.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time
import threading

import pandas as pd


@dataclass
class _TableBuf:
    name: str
    schema: List[str] = field(default_factory=list)       # column order
    rows: List[Dict[str, Any]] = field(default_factory=list)
    shards: List[str] = field(default_factory=list)       # relative paths data/runs/<run_id>/shards/xxx.parquet
    written_rows: int = 0
    shard_seq: int = 0


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_dirs(run_dir: Path) -> None:
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)


def _rel_shard_path(run_id: str, fname: str) -> str:
    # manifest uses paths relative to repo root starting at "data/"
    return str(Path("data") / "runs" / run_id / "shards" / fname)


def _httpize(prefix: Optional[str], rel_data_path: str) -> str:
    if not prefix:
        return rel_data_path
    p = rel_data_path
    if p.startswith("data/"):
        p = p[len("data/"):]
    return prefix.rstrip("/") + "/" + p.lstrip("/")


class ParquetRecorder:
    """
    Minimal interface used by engine:
      add(table: str, rows: List[dict])
      finalize()

    Extra attributes:
      manifest_path -> Path to manifest.json (used by Engine for out_dir inference)
    """

    def __init__(
        self,
        data_root: str | Path = "data",
        run_id: Optional[str] = None,
        *,
        https_prefix: Optional[str] = None,
        max_rows_per_shard: int = 200_000,
        flush_hint_rows: int = 10_000,
    ) -> None:
        self.data_root = Path(data_root)
        self.run_id = run_id or f"FUKA_{int(time.time())}"
        self.run_dir = self.data_root / "runs" / self.run_id
        _ensure_dirs(self.run_dir)

        # public prefix for index URLs
        self.https_prefix = https_prefix or os.environ.get("DATA_URL_PREFIX") or None

        # shard policy
        self.max_rows_per_shard = int(max_rows_per_shard)
        self.flush_hint_rows = int(flush_hint_rows)

        # tables
        self._tables: Dict[str, _TableBuf] = {}

        # thread lock (in case add() is called cross-threads)
        self._lock = threading.Lock()

        # manifest location (exposed)
        self.manifest_path: Path = self.run_dir / "manifest.json"

        # write a tiny meta
        meta = {
            "run_id": self.run_id,
            "created_utc": _stamp(),
            "data_root": str(self.data_root),
            "https_prefix": self.https_prefix,
        }
        with (self.run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # ---------------------- public API ----------------------

    def add(self, table: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self._lock:
            T = self._tables.get(table)
            if T is None:
                T = _TableBuf(name=table)
                self._tables[table] = T

            # Update schema on first batch
            if not T.schema:
                # Preserve user-provided order, but ensure 'step' first when present
                first = list(rows[0].keys())
                if "step" in first:
                    first = ["step"] + [k for k in first if k != "step"]
                T.schema = first

            # Buffer rows
            T.rows.extend(rows)

            # Opportunistic flush for big buffers
            if len(T.rows) >= self.flush_hint_rows:
                self._flush_table(T)

    def finalize(self) -> None:
        # Flush any remaining buffers
        with self._lock:
            for T in self._tables.values():
                self._flush_table(T, force=True)

            # Write *_index.json per table and manifest.json
            self._write_indices_and_manifest()

    # ---------------------- internals ----------------------

    def _flush_table(self, T: _TableBuf, force: bool = False) -> None:
        if not T.rows:
            return

        # rotate shard if needed
        if (not force) and T.written_rows >= self.max_rows_per_shard:
            # safety: in practice written_rows will be reset below; this branch kept for clarity
            pass

        # Decide shard filename (rotate by sequence)
        if (T.written_rows == 0) or (T.written_rows >= self.max_rows_per_shard):
            T.shard_seq += 1
            T.written_rows = 0

        fname = f"{T.name}_{T.shard_seq:04d}.parquet"
        rel_path = _rel_shard_path(self.run_id, fname)
        abs_path = self.run_dir / "shards" / Path(fname).name

        # Normalize dataframe to stable column order (extend missing cols with NaN)
        df = pd.DataFrame.from_records(T.rows)
        # expand schema if new columns appear later
        new_cols = [c for c in df.columns if c not in T.schema]
        if new_cols:
            T.schema += new_cols
        # ensure all schema columns are present in df
        for col in T.schema:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[T.schema]

        # Fix dtypes for common fields
        for col in df.columns:
            if col == "step" or col.endswith(("x", "y", "z")) or col.endswith(("x0", "y0", "z0", "x1", "y1", "z1")):
                # integers but allow missing -> Int64 (nullable)
                try:
                    df[col] = df[col].astype("Int64")
                except Exception:
                    pass
            elif col in ("value", "v0", "v1", "deposit", "kappa"):
                try:
                    df[col] = df[col].astype("float32")
                except Exception:
                    pass

        # Append or write new shard
        # Strategy: each shard seq writes once (no appends) for simplicity and robustness.
        # If you want true appending, swap to pyarrow.dataset with append mode.
        df.to_parquet(abs_path, index=False)
        if rel_path not in T.shards:
            T.shards.append(rel_path)
        T.written_rows += len(T.rows)
        T.rows.clear()

    def _write_indices_and_manifest(self) -> None:
        shards_dir = self.run_dir / "shards"
        # Per-table indices
        for name, T in self._tables.items():
            files = [self._maybe_url(p) for p in T.shards]
            index_payload = {
                "table": name,
                "run_id": self.run_id,
                "files": files,
                "count": len(files),
            }
            with (shards_dir / f"{name}_index.json").open("w", encoding="utf-8") as f:
                json.dump(index_payload, f, indent=2)

        # Manifest (flat list of shards)
        shards = []
        for name, T in self._tables.items():
            for rel in T.shards:
                shards.append({"table": name, "path": rel})

        manifest = {
            "run_id": self.run_id,
            "root": f"data/runs/{self.run_id}",
            "created_utc": _stamp(),
            "tables": sorted(self._tables.keys()),
            "shards": shards,
        }
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _maybe_url(self, rel_path: str) -> str:
        return _httpize(self.https_prefix, rel_path)