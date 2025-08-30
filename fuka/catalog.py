from __future__ import annotations
from pathlib import Path
import duckdb, json

def update_catalog(catalog_path: Path, run_dir: Path):
    con = duckdb.connect(str(catalog_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs(
            run_id TEXT PRIMARY KEY,
            started_at DOUBLE,
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
    manifest = json.loads((run_dir / "manifest.json").read_text())
    run_id = manifest["run_id"]
    con.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
                [run_id, manifest["created_at"], ""])
    for s in manifest["shards"]:
        con.execute("INSERT INTO shards VALUES (?, ?, ?)",
                    [run_id, s["table"], s["path"]])
    con.close()