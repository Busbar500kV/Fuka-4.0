from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any
import json
import duckdb


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs(
            run_id TEXT PRIMARY KEY,
            created_at DOUBLE,
            created_at_iso TEXT,
            notes TEXT
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS shards(
            run_id TEXT,
            table_name TEXT,
            shard_path TEXT
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS table_counts(
            run_id TEXT,
            table_name TEXT,
            shard_count BIGINT
        );
    """)


def _read_manifest(run_dir: Path) -> Dict[str, Any]:
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(f"manifest.json not found at {mpath}")
    return json.loads(mpath.read_text())


def _upsert_run(con: duckdb.DuckDBPyConnection, m: Dict[str, Any]) -> None:
    run_id = str(m.get("run_id", "UNKNOWN"))
    created_at = float(m.get("created_at", 0.0))
    created_at_iso = str(m.get("created_at_iso", ""))
    con.execute(
        "INSERT OR REPLACE INTO runs(run_id, created_at, created_at_iso, notes) VALUES (?, ?, ?, ?)",
        [run_id, created_at, created_at_iso, ""],
    )


def _replace_shards(con: duckdb.DuckDBPyConnection, run_id: str, shards: Iterable[Dict[str, Any]]) -> None:
    con.execute("DELETE FROM shards WHERE run_id = ?", [run_id])
    rows = []
    for s in shards:
        table_name = str(s.get("table", "unknown"))
        shard_path = str(s.get("path", ""))
        rows.append((run_id, table_name, shard_path))
    if rows:
        con.executemany("INSERT INTO shards(run_id, table_name, shard_path) VALUES (?, ?, ?)", rows)

    con.execute("DELETE FROM table_counts WHERE run_id = ?", [run_id])
    con.execute("""
        INSERT INTO table_counts
        SELECT run_id, table_name, COUNT(*) AS shard_count
        FROM shards
        WHERE run_id = ?
        GROUP BY run_id, table_name
    """, [run_id])


def update_catalog(catalog_path: Path, run_dir: Path) -> None:
    """
    Update (or create) a DuckDB catalog from <run_dir>/manifest.json.
    """
    catalog_path = Path(catalog_path)
    run_dir = Path(run_dir)
    con = duckdb.connect(str(catalog_path))
    try:
        _ensure_schema(con)
        m = _read_manifest(run_dir)
        run_id = str(m.get("run_id", "UNKNOWN"))
        _upsert_run(con, m)
        _replace_shards(con, run_id, m.get("shards", []))
    finally:
        con.close()