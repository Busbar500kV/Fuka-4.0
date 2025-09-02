# fuka/render/pack_npz.py
from __future__ import annotations
import argparse, os, sys
from typing import List, Optional, Tuple

import duckdb as ddb
import numpy as np
import pandas as pd

from fuka.io.discovery import discover

def _connect_duck() -> ddb.DuckDBPyConnection:
    con = ddb.connect(":memory:")
    try:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
    except Exception:
        pass
    return con

def _read_one_step(con: ddb.DuckDBPyConnection, files: List[str], step: int, cols: List[str]) -> Optional[pd.DataFrame]:
    if not files:
        return None
    q = f"SELECT {', '.join(cols)} FROM read_parquet(?) WHERE step = ?"
    try:
        return con.execute(q, [files, int(step)]).df()
    except Exception:
        return None

def pack_npz(prefix: str, run_id: str, step_min: int, step_max: int, out_npz: str,
             max_points: int = 15000, max_edges: int = 2000, allow_gsutil: bool = False) -> Tuple[int,int,int,int]:
    """
    Returns: (n_steps, sum_points, n_steps_edges, sum_edges)
    """
    tables = discover(prefix, run_id, tables=("state","edges"), allow_gsutil=allow_gsutil)
    state_files = tables.get("state", [])
    edges_files = tables.get("edges", [])

    if not state_files:
        raise RuntimeError(f"No state shards found for run {run_id} (prefix={prefix}).")

    con = _connect_duck()
    steps = list(range(int(step_min), int(step_max) + 1))

    states_seq: List[np.ndarray] = []
    edges_seq:  List[np.ndarray] = []

    for s in steps:
        # state: prefer 3D schema (value), fallback to legacy (m AS value)
        df = _read_one_step(con, state_files, s, ["x","y","z","value"])
        if df is None or df.empty:
            df = _read_one_step(con, state_files, s, ["x","y","z","m AS value"])
        if df is None or df.empty:
            s_arr = np.zeros((0,4), dtype="float32")
        else:
            if len(df) > max_points:
                df = df.sample(int(max_points), random_state=0)
            s_arr = df[["x","y","z","value"]].to_numpy("float32")
        states_seq.append(s_arr)

        # edges: prefer 3D schema
        if edges_files:
            ed = _read_one_step(con, edges_files, s, ["x0","y0","z0","x1","y1","z1","v0","v1"])
            if ed is None or ed.empty:
                e_arr = np.zeros((0,8), dtype="float32")
            else:
                if len(ed) > max_edges:
                    ed = ed.sample(int(max_edges), random_state=0)
                e_arr = ed[["x0","y0","z0","x1","y1","z1","v0","v1"]].to_numpy("float32")
        else:
            e_arr = np.zeros((0,8), dtype="float32")
        edges_seq.append(e_arr)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        steps=np.array(steps, dtype="int32"),
        states=np.array(states_seq, dtype=object),
        edges=np.array(edges_seq, dtype=object),
    )
    return len(steps), int(sum(len(a) for a in states_seq)), len(steps), int(sum(len(a) for a in edges_seq))

# ---- CLI ----
def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser("fuka.render.pack_npz")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--step_min", required=True, type=int)
    ap.add_argument("--step_max", required=True, type=int)
    ap.add_argument("--out", default="assets/fuka_anim.npz")
    ap.add_argument("--max_points", type=int, default=15000)
    ap.add_argument("--max_edges",  type=int, default=2000)
    ap.add_argument("--allow-gsutil", action="store_true")
    return ap.parse_args(argv)

def main(argv: Optional[List[str]]=None) -> int:
    ns = _parse_args(argv or sys.argv[1:])
    n_steps, sum_pts, _, sum_edges = pack_npz(
        prefix=ns.prefix,
        run_id=ns.run_id,
        step_min=ns.step_min,
        step_max=ns.step_max,
        out_npz=ns.out,
        max_points=ns.max_points,
        max_edges=ns.max_edges,
        allow_gsutil=ns.allow_gsutil,
    )
    print(f"wrote {ns.out} | steps={n_steps} Σpts={sum_pts} Σedges={sum_edges}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())