# analytics/build_indices.py
from __future__ import annotations

"""
Rebuild per-table *_index.json files and a run-level manifest.json.

- Scans data/runs/<run_id>/shards for *.parquet
- Classifies shards into logical tables (state, edges, events, catalysts, spectra, ledger, env)
- Validates minimally (file is readable, has a step column)
- Writes:
    data/runs/<run_id>/shards/<table>_index.json
    data/runs/<run_id>/manifest.json

If --prefix is supplied (e.g., https://storage.googleapis.com/fuka4-runs),
the *_index.json files contain HTTPS URLs so the packer (and Manim) can stream directly.
Otherwise they contain relative file paths for local use.

Idempotent & safe: rerunning will refresh indexes without touching parquet shards.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb as ddb


# ----------------------------- helpers -----------------------------

@dataclass(frozen=True)
class TableSpec:
    name: str
    # minimal columns required to classify this table
    must_have_any: Tuple[Tuple[str, ...], ...]


TABLE_SPECS: List[TableSpec] = [
    # "state" typically: x,y,z,value (or legacy: x,y,z,m)
    TableSpec(
        "state",
        (
            ("x", "y", "z", "value"),
            ("x", "y", "z", "m"),
        ),
    ),
    # "edges": endpoints + values; deposit/kappa optional
    TableSpec(
        "edges",
        (
            ("x0", "y0", "z0", "x1", "y1", "z1", "v0", "v1"),
        ),
    ),
    # optional tables used by analytics or future physics
    TableSpec("events", (("event", "x", "y", "z"), ("evt", "kind", "x", "y", "z"),)),
    TableSpec("catalysts", (("id", "x", "y", "z"),)),
    TableSpec("spectra", (("k", "power"), ("freq", "power"),)),
    TableSpec("ledger", (("key", "value"),)),
    TableSpec("env", (("param", "value"),)),
]


def _duck_connect() -> ddb.DuckDBPyConnection:
    con = ddb.connect(":memory:")
    return con


def _columns_of(con: ddb.DuckDBPyConnection, pq_path: str) -> List[str]:
    # LIMIT 0 to get schema quickly
    try:
        df = con.execute("SELECT * FROM read_parquet(?) LIMIT 0", [pq_path]).df()
        return [str(c) for c in df.columns]
    except Exception:
        return []


def _has_step(con: ddb.DuckDBPyConnection, pq_path: str) -> bool:
    try:
        df = con.execute("SELECT step FROM read_parquet(?) LIMIT 1", [pq_path]).df()
        return "step" in df.columns
    except Exception:
        # If selection fails, try projecting nothing but check schema
        cols = _columns_of(con, pq_path)
        return "step" in cols


def _classify_table(cols: List[str]) -> str | None:
    s = set(c.lower() for c in cols)
    for spec in TABLE_SPECS:
        for candidate in spec.must_have_any:
            if all((c.lower() in s) for c in candidate):
                return spec.name
    return None


def _httpize(prefix: str, run_id: str, shards_rel_path: str) -> str:
    prefix = prefix.rstrip("/")
    # shards_rel_path like: data/runs/<run_id>/shards/<file>.parquet
    if shards_rel_path.startswith("data/"):
        shards_rel_path = shards_rel_path[len("data/"):]
    return f"{prefix}/{shards_rel_path.lstrip('/')}"


# ----------------------------- core -----------------------------

def rebuild_indices(data_root: str, run_id: str, prefix: str | None = None) -> Dict[str, List[str]]:
    """
    Returns a mapping {table: [files_or_urls,...]}.
    """
    run_dir = Path(data_root) / "runs" / run_id
    shards_dir = run_dir / "shards"
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"Shards directory not found: {shards_dir}")

    con = _duck_connect()

    # group candidate parquet files by table
    by_table: Dict[str, List[str]] = {}
    errors: List[str] = []

    for p in sorted(shards_dir.glob("*.parquet")):
        rel = str(Path("data") / "runs" / run_id / "shards" / p.name)
        cols = _columns_of(con, str(p))
        if not cols:
            errors.append(f"[warn] unreadable parquet (skipped): {p.name}")
            continue
        if not _has_step(con, str(p)):
            errors.append(f"[warn] parquet lacks required 'step' column (skipped): {p.name}")
            continue
        table = _classify_table(cols)
        if table is None:
            # lightweight heuristic: filename prefix before first underscore
            table = p.name.split("_", 1)[0].lower()
        by_table.setdefault(table, []).append(rel)

    # write per-table indices
    shards_dir.mkdir(parents=True, exist_ok=True)
    for table, rel_files in by_table.items():
        files = [(_httpize(prefix, run_id, f) if prefix else f) for f in rel_files]
        index_payload = {
            "table": table,
            "run_id": run_id,
            "files": files,
            "count": len(files),
        }
        out = shards_dir / f"{table}_index.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(index_payload, f, indent=2)

    # write manifest
    manifest = {
        "run_id": run_id,
        "root": f"data/runs/{run_id}",
        "shards": [{"table": t, "path": rel} for t, rels in by_table.items() for rel in rels],
        "tables": sorted(by_table.keys()),
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # optional diagnostics
    if errors:
        diag_path = run_dir / "index_warnings.txt"
        with diag_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(errors) + "\n")

    return by_table


# ----------------------------- CLI -----------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild Fuka run indices and manifest.")
    ap.add_argument("--data_root", default="data", help="Root data directory (default: data)")
    ap.add_argument("--run_id", required=True, help="Run identifier (folder under data/runs)")
    ap.add_argument("--prefix", default=os.environ.get("DATA_URL_PREFIX", "").strip() or None,
                    help="HTTPS prefix for public access (e.g., https://storage.googleapis.com/fuka4-runs)")
    args = ap.parse_args(argv)

    try:
        by_table = rebuild_indices(args.data_root, args.run_id, args.prefix)
    except FileNotFoundError as e:
        print(f"[build_indices] ERROR: {e}", file=sys.stderr)
        return 2

    summary = ", ".join(f"{t}:{len(v)} shards" for t, v in sorted(by_table.items()))
    pref = f" (urls under {args.prefix})" if args.prefix else ""
    print(f"[build_indices] OK: {args.run_id} -> {summary}{pref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())