# fuka/render/pack_npz.py
from __future__ import annotations

"""
Pack a Fuka run (served over HTTPS) into a single NPZ for Manim.

Design goals
- Zero local state required: consumes shards via HTTPS using DuckDB httpfs.
- Robust discovery: prefer per-table *_index.json; fallback to manifest.json.
- Canonical payload keys for renderer:
    steps (F,)
    state_x/state_y/state_z/state_value (Ns,)
    state_idx (F+1,)            # prefix-sum index into state_* for each frame
    edges_x0/edges_y0/edges_z0/edges_x1/edges_y1/edges_z1 (Ne,)
    edges_idx (F+1,)            # prefix-sum index into edges_* for each frame
    edge_value (Ne,)            # canonical colour scalar for edges: 0.5*(v0+v1)
    edge_strength (Ne,)         # |v1 - v0|
Optional:
    edge_deposit/edge_kappa (Ne,) if present in shards (propagated transparently)
"""

import argparse
import json
import sys
import urllib.request
from typing import Dict, List, Tuple

import duckdb as ddb
import numpy as np


# -------------------- HTTP helpers --------------------

def _http_json(url: str, timeout: float = 15.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

def _looks_http(path: str) -> bool:
    return isinstance(path, str) and path.startswith(("http://", "https://"))


# -------------------- Discovery --------------------

def _gather_parquet_urls(prefix: str, run_id: str, table: str, *, allow_manifest: bool = True) -> List[str]:
    """
    Try shards/<table>_index.json first (preferred).
    Fallback to manifest.json -> convert "data/..." to HTTPS URLs.
    """
    files: List[str] = []

    # 1) *_index.json (preferred)
    idx_url = f"{prefix}/runs/{run_id}/shards/{table}_index.json"
    idx = _http_json(idx_url)
    if idx and isinstance(idx.get("files"), list):
        fs = [f for f in idx["files"] if _looks_http(f) and f.endswith(".parquet")]
        if fs:
            return sorted(fs)

    if allow_manifest:
        # 2) manifest.json
        man_url = f"{prefix}/runs/{run_id}/manifest.json"
        man = _http_json(man_url)
        if man and isinstance(man.get("shards"), list):
            base = prefix.rstrip("/")
            for s in man["shards"]:
                if str(s.get("table")) != table:
                    continue
                p = str(s.get("path", ""))
                # manifest path is relative: "data/runs/<RUN_ID>/shards/xxx.parquet"
                if p.startswith("data/"):
                    p = p[len("data/"):]
                url = f"{base}/{p.lstrip('/')}"
                if url.endswith(".parquet"):
                    files.append(url)
            if files:
                return sorted(files)

    return []


# -------------------- Core logic --------------------

def _connect_duckdb() -> ddb.DuckDBPyConnection:
    con = ddb.connect(":memory:")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    return con

def _read_per_step_arrays(
    con: ddb.DuckDBPyConnection,
    files: List[str],
    table: str,
    col_expr: str,
    step_min: int,
    step_max: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read rows from HTTPS parquet shards and return a tuple:
      (concat_values, prefix_idx)
    where concat_values is a 1D or 2D array flattened later by the caller
    and prefix_idx is (F+1,) int64 cumulative counts per step.

    `col_expr` should select the desired columns in order, e.g.:
        "x,y,z,value"   for state
        "x0,y0,z0,x1,y1,z1,v0,v1,deposit,kappa"   for edges (extra cols may be absent)
    """
    steps = np.arange(int(step_min), int(step_max) + 1, dtype=np.int64)
    counts: List[int] = []
    chunks: List[np.ndarray] = []

    for s in steps:
        df = con.execute(f"SELECT {col_expr} FROM read_parquet(?) WHERE step = ?", [files, int(s)]).df()
        if df.empty:
            counts.append(0)
            # use a shape-compatible empty
            if not chunks:
                # create a dummy empty with the right number of columns
                ncols = len([c for c in col_expr.split(",") if c.strip()])
                chunks.append(np.zeros((0, ncols), dtype=np.float32))
            continue
        arr = df.to_numpy("float32")
        counts.append(len(arr))
        chunks.append(arr)

    if not chunks:
        values = np.zeros((0, len([c for c in col_expr.split(",") if c.strip()])), dtype=np.float32)
    else:
        values = np.concatenate(chunks, axis=0)
    prefix = np.zeros(len(steps) + 1, dtype=np.int64)
    if counts:
        prefix[1:] = np.cumsum(np.array(counts, dtype=np.int64), dtype=np.int64)
    return values, prefix


def pack(prefix: str, run_id: str, step_min: int, step_max: int, out_path: str) -> None:
    con = _connect_duckdb()

    # Discover shards
    state_files = _gather_parquet_urls(prefix, run_id, "state")
    edges_files = _gather_parquet_urls(prefix, run_id, "edges")

    if not state_files:
        print("[pack] ERROR: no 'state' shards found (index or manifest).", file=sys.stderr)
        sys.exit(2)

    steps = np.arange(int(step_min), int(step_max) + 1, dtype=np.int64)

    # Read states (3D schema preferred)
    try:
        S, s_idx = _read_per_step_arrays(con, state_files, "state", "x,y,z,value", step_min, step_max)
    except Exception:
        # Legacy fallback (m instead of value)
        S, s_idx = _read_per_step_arrays(con, state_files, "state", "x,y,z,m as value", step_min, step_max)

    # Read edges (optional)
    E = None
    e_idx = None
    extra_edge = {}  # deposit/kappa propagated if present
    if edges_files:
        # Try to read with optional deposit/kappa
        # Order: x0,y0,z0,x1,y1,z1,v0,v1,deposit,kappa
        try:
            cols = "x0,y0,z0,x1,y1,z1,v0,v1,deposit,kappa"
            E, e_idx = _read_per_step_arrays(con, edges_files, "edges", cols, step_min, step_max)
            have_deposit = True
            have_kappa = True
        except Exception:
            # minimal required columns
            cols = "x0,y0,z0,x1,y1,z1,v0,v1"
            E, e_idx = _read_per_step_arrays(con, edges_files, "edges", cols, step_min, step_max)
            have_deposit = False
            have_kappa = False

        if E is not None and E.size > 0:
            # Derive canonical edge fields
            # E columns (at least): 0..5 positions, 6=v0, 7=v1, (8 deposit?), (9 kappa?)
            v0 = E[:, 6]
            v1 = E[:, 7]
            edge_value = 0.5 * (v0 + v1)
            edge_strength = np.abs(v1 - v0)
            extra_edge["edge_value"] = edge_value.astype(np.float64)
            extra_edge["edge_strength"] = edge_strength.astype(np.float64)
            if have_deposit and E.shape[1] >= 9:
                extra_edge["edge_deposit"] = E[:, 8].astype(np.float64)
            if have_kappa and E.shape[1] >= 10:
                extra_edge["edge_kappa"] = E[:, 9].astype(np.float64)

    # Build payload with guaranteed keys (present even if empty)
    if S.size == 0:
        sx = sy = sz = sval = np.zeros((0,), dtype=np.float64)
    else:
        sx, sy, sz, sval = (S[:, 0].astype(np.float64),
                            S[:, 1].astype(np.float64),
                            S[:, 2].astype(np.float64),
                            S[:, 3].astype(np.float64))

    if E is None or E.size == 0:
        x0 = y0 = z0 = x1 = y1 = z1 = np.zeros((0,), dtype=np.float64)
        e_idx_u = np.zeros(len(steps) + 1, dtype=np.int64)
    else:
        x0, y0, z0, x1, y1, z1 = (E[:, 0].astype(np.float64),
                                   E[:, 1].astype(np.float64),
                                   E[:, 2].astype(np.float64),
                                   E[:, 3].astype(np.float64),
                                   E[:, 4].astype(np.float64),
                                   E[:, 5].astype(np.float64))
        e_idx_u = e_idx

    s_idx_u = s_idx

    payload: Dict[str, np.ndarray] = {
        "steps": steps.astype(np.int64),

        "state_x": sx,
        "state_y": sy,
        "state_z": sz,
        "state_value": sval,
        "state_idx": s_idx_u.astype(np.int64),

        "edges_x0": x0,
        "edges_y0": y0,
        "edges_z0": z0,
        "edges_x1": x1,
        "edges_y1": y1,
        "edges_z1": z1,
        "edges_idx": e_idx_u.astype(np.int64),
    }

    # Attach derived edge fields if we have edges
    payload.update(extra_edge)

    np.savez(out_path, **payload)
    print(f"[pack] wrote {out_path} | steps={steps.size} state_rows={sx.size} edge_rows={x0.size}", flush=True)


# -------------------- CLI --------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pack Fuka shards (HTTPS index/manifest) into a single NPZ for Manim.")
    ap.add_argument("--prefix", required=True, help="e.g. https://storage.googleapis.com/fuka4-runs")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--step_min", type=int, required=True)
    ap.add_argument("--step_max", type=int, required=True)
    ap.add_argument("--out", default="assets/fuka_anim.npz")
    args = ap.parse_args(argv)

    pack(prefix=args.prefix, run_id=args.run_id, step_min=args.step_min, step_max=args.step_max, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())