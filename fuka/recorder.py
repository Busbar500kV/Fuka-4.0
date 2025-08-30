"""
Fuka 4.0 — Parquet recorder with shard indices.

Tables supported:
  - events(step, conn_id, x,y,z, dF, dm, w_sel, A_sel, phi_sel, T_eff, theta_thr, ...)
  - spectra(step, conn_id, w_dom, A_dom, phi_dom, F_local, E_sum, S_spec, ...)
  - state(step, conn_id, x,y,z, m, T_eff, theta_thr, ...)
  - ledger(step, dF, c2dm, Q, W_cat, net_flux, balance_error, ...)
  - edges(step, src_conn, dst_conn, weight, [optional: w, phi], [x,y,z optional])
  - env(step, x, y, z, value, [conn_id optional])

Each table is buffered in-memory and flushed to Parquet shards like:
  data/runs/<RUN_ID>/shards/<table>_000.parquet

We also maintain a manifest.json under:
  data/runs/<RUN_ID>/manifest.json

NOTE:
- We write simple, wide parquet files (one row = one log line).
- We convert "xmu" (t, x, y, z) tuples into scalar columns t/x/y/z before flush.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetRecorder:
    def __init__(self, data_root: str, run_id: str, flush_every: int = 1000):
        self.data_root = Path(data_root)
        self.run_id = run_id
        self.flush_every = int(flush_every)

        self.run_dir = self.data_root / "runs" / self.run_id
        self.shards_dir = self.run_dir / "shards"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        # table -> list[dict]
        self._buf: Dict[str, List[Dict[str, Any]]] = {
            "events": [], "spectra": [], "state": [], "ledger": [],
            "edges": [], "env": []
        }

        # table -> next shard index
        self._next_idx: Dict[str, int] = {k: 0 for k in self._buf.keys()}

        # manifest
        self.manifest_path = self.run_dir / "manifest.json"
        self._manifest = {
            "run_id": self.run_id,
            "created_at": float(pd.Timestamp.now().timestamp()),
            "shards": []  # list of {"table": <t>, "path": <relative>}
        }
        if self.manifest_path.exists():
            # resume / append mode
            try:
                self._manifest = json.loads(self.manifest_path.read_text())
                # best-effort: infer next indices from existing shard names
                for s in self._manifest.get("shards", []):
                    t = s.get("table")
                    p = Path(s.get("path", ""))
                    stem = p.stem  # e.g., "events_003"
                    if "_" in stem:
                        try:
                            idx = int(stem.split("_")[-1])
                            self._next_idx[t] = max(self._next_idx.get(t, 0), idx + 1)
                        except Exception:
                            pass
            except Exception:
                pass

        # simple counters to decide when to flush
        self._since_flush = 0

    # -------------- helpers --------------

    def _with_common(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize row: inject run_id and expand xmu->[t,x,y,z] if present."""
        row = dict(row)
        row["run_id"] = self.run_id
        if "xmu" in row:
            t, x, y, z = row["xmu"]
            row.pop("xmu", None)
            row["t"] = float(t)
            row["x"] = float(x); row["y"] = float(y); row["z"] = float(z)
        return row

    def _flush_table(self, table: str) -> Optional[Path]:
        buf = self._buf[table]
        if not buf:
            return None
        df = pd.DataFrame(buf)
        self._buf[table] = []

        shard_idx = self._next_idx[table]
        self._next_idx[table] += 1

        rel = Path("data") / "runs" / self.run_id / "shards" / f"{table}_{shard_idx:03d}.parquet"
        abs_path = self.data_root / "runs" / self.run_id / "shards" / f"{table}_{shard_idx:03d}.parquet"

        table_arrow = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table_arrow, abs_path)

        self._manifest["shards"].append({"table": table, "path": str(rel)})
        return abs_path

    def _maybe_flush(self):
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush_all()
            self._since_flush = 0

    def flush_all(self):
        written_any = False
        for t in self._buf.keys():
            p = self._flush_table(t)
            if p is not None:
                written_any = True
        if written_any:
            self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def finalize(self):
        self.flush_all()
        # final sync of manifest
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    # -------------- public loggers --------------

    def log_event(self, **row):
        self._buf["events"].append(self._with_common(row))
        self._maybe_flush()

    def log_spectra(self, **row):
        self._buf["spectra"].append(self._with_common(row))
        self._maybe_flush()

    def log_state(self, **row):
        self._buf["state"].append(self._with_common(row))
        self._maybe_flush()

    def log_ledger(self, **row):
        self._buf["ledger"].append(self._with_common(row))
        self._maybe_flush()

    # NEW
    def log_edge(self, **row):
        """Row should contain: src_conn, dst_conn, step, weight, [w, phi], [x,y,z optional]."""
        self._buf["edges"].append(self._with_common(row))
        self._maybe_flush()

    # NEW
    def log_env(self, **row):
        """Row should contain: step, x, y, z, value, [conn_id optional]."""
        self._buf["env"].append(self._with_common(row))
        self._maybe_flush()