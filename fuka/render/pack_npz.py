# fuka/render/pack_npz.py
from __future__ import annotations

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
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _looks_http(x: str) -> bool:
    return isinstance(x, str) and (x.startswith("http://") or x.startswith("https://"))


# -------------------- discovery --------------------

def discover_files(prefix: str, run_id: str, table: str, allow_gsutil: bool = False) -> List[str]:
    """
    Try shards/<table>_index.json first (preferred).
    Fallback to manifest.json -> convert "data/..." to HTTPS URLs.
    Optionally fall back to `gsutil ls` if allow_gsutil=True (and index/manifest missing).
    """
    files: List[str] = []

    # 1) *_index.json (preferred)
    idx_url = f"{prefix}/runs/{run_id}/shards/{table}_index.json"
    idx = _http_json(idx_url)
    if idx and isinstance(idx.get("files"), list):
        # Some old indices had absolute local paths; keep only HTTP(S)
        fs = [f for f in idx["files"] if _looks_http(f)]
        if fs:
            return sorted(fs)

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

    # 3) optional gsutil (best effort)
    if allow_gsutil:
        try:
            import subprocess
            bucket = prefix.split("://", 1)[-1].split("/", 1)[0]
            pat = f"gs://{bucket}/runs/{run_id}/shards/{table}_*.parquet"
            p = subprocess.run(["gsutil", "ls", pat], capture_output=True, text=True)
            if p.returncode == 0:
                for line in p.stdout.strip().splitlines():
                    if line.startswith("gs://"):
                        files.append(line.replace("gs://", "https://storage.googleapis.com/"))
        except Exception:
            pass
        if files:
            return sorted(files)

    return []


# -------------------- duckdb helpers --------------------

def connect_duckdb() -> ddb.DuckDBPyConnection:
    con = ddb.connect(":memory:")
    try:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
    except Exception:
        # if already installed/loaded
        pass
    return con


def table_has_cols(con: ddb.DuckDBPyConnection, files: List[str], cols: List[str]) -> bool:
    if not files:
        return False
    try:
        # probe with LIMIT 0 so we don't scan
        qry = f"SELECT {', '.join(cols)} FROM read_parquet(?) LIMIT 0"
        con.execute(qry, [files])
        return True
    except Exception:
        return False


# -------------------- extraction --------------------

def extract_state(con: ddb.DuckDBPyConnection, state_files: List[str], step_min: int, step_max: int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (steps_sorted, sx, sy, sz, sval, state_idx)
    steps_sorted: shape (S,)
    state_idx:    prefix sums shape (S+1,)
    """
    if not state_files:
        return (np.zeros(0, dtype=np.int64),
                np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros(1, dtype=np.int64))

    # Detect viable schema
    # New: step, x, y, z, value
    if table_has_cols(con, state_files, ["step", "x", "y", "z", "value"]):
        base = """
            SELECT step::BIGINT AS step, x::DOUBLE x, y::DOUBLE y, z::DOUBLE z,
                   value::DOUBLE AS value
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """
    # Legacy: step, x, y, z, m
    elif table_has_cols(con, state_files, ["step", "x", "y", "z", "m"]):
        base = """
            SELECT step::BIGINT AS step, x::DOUBLE x, y::DOUBLE y, z::DOUBLE z,
                   m::DOUBLE AS value
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """
    # Legacy2: step, x, y, z, T_eff
    elif table_has_cols(con, state_files, ["step", "x", "y", "z", "T_eff"]):
        base = """
            SELECT step::BIGINT AS step, x::DOUBLE x, y::DOUBLE y, z::DOUBLE z,
                   T_eff::DOUBLE AS value
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """
    else:
        # fallback: no usable columns
        return (np.zeros(0, dtype=np.int64),
                np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros(1, dtype=np.int64))

    df = con.execute(base, [state_files, int(step_min), int(step_max)]).df()
    if df.empty:
        return (np.zeros(0, dtype=np.int64),
                np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros(1, dtype=np.int64))

    df = df.sort_values(["step"], kind="mergesort").reset_index(drop=True)
    steps = df["step"].to_numpy(dtype=np.int64)
    sx = df["x"].to_numpy(dtype=np.float64)
    sy = df["y"].to_numpy(dtype=np.float64)
    sz = df["z"].to_numpy(dtype=np.float64)
    sval = df["value"].to_numpy(dtype=np.float64)

    # build prefix sums per unique step
    uniq, counts = np.unique(steps, return_counts=True)
    pref = np.zeros(len(uniq) + 1, dtype=np.int64)
    pref[1:] = np.cumsum(counts, dtype=np.int64)

    return uniq, sx, sy, sz, sval, pref


def extract_edges(con: ddb.DuckDBPyConnection,
                  edges_files: List[str],
                  state_files: List[str],
                  step_min: int,
                  step_max: int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (steps_sorted, ex0, ey0, ez0, ex1, ey1, ez1, edges_idx)
    If no edges available/derivable, returns empty arrays (with idx shape (1,))
    """
    if not edges_files:
        return (np.zeros(0, dtype=np.int64),
                *(np.zeros(0) for _ in range(6)),
                np.zeros(1, dtype=np.int64))

    # 3D edges present directly?
    if table_has_cols(con, edges_files, ["step", "x0", "y0", "z0", "x1", "y1", "z1"]):
        qry = """
            SELECT step::BIGINT AS step,
                   x0::DOUBLE AS x0, y0::DOUBLE AS y0, z0::DOUBLE AS z0,
                   x1::DOUBLE AS x1, y1::DOUBLE AS y1, z1::DOUBLE AS z1
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """
        df = con.execute(qry, [edges_files, int(step_min), int(step_max)]).df()
        if df.empty:
            return (np.zeros(0, dtype=np.int64),
                    *(np.zeros(0) for _ in range(6)),
                    np.zeros(1, dtype=np.int64))
        df = df.sort_values(["step"], kind="mergesort").reset_index(drop=True)
        steps = df["step"].to_numpy(dtype=np.int64)
        x0 = df["x0"].to_numpy(dtype=np.float64)
        y0 = df["y0"].to_numpy(dtype=np.float64)
        z0 = df["z0"].to_numpy(dtype=np.float64)
        x1 = df["x1"].to_numpy(dtype=np.float64)
        y1 = df["y1"].to_numpy(dtype=np.float64)
        z1 = df["z1"].to_numpy(dtype=np.float64)

    # Legacy edges: src_conn/dst_conn at each step → join with state positions
    elif table_has_cols(con, edges_files, ["step", "src_conn", "dst_conn"]) and \
         state_files and table_has_cols(con, state_files, ["step", "conn_id", "x", "y", "z"]):
        qry = """
            WITH E AS (
              SELECT step::BIGINT AS step, src_conn::BIGINT AS s, dst_conn::BIGINT AS d
              FROM read_parquet(?)
              WHERE step BETWEEN ? AND ?
            ),
            S AS (
              SELECT step::BIGINT AS step, conn_id::BIGINT AS c, x::DOUBLE x, y::DOUBLE y, z::DOUBLE z
              FROM read_parquet(?)
              WHERE step BETWEEN ? AND ?
            )
            SELECT E.step,
                   S1.x AS x0, S1.y AS y0, S1.z AS z0,
                   S2.x AS x1, S2.y AS y1, S2.z AS z1
            FROM E
            JOIN S AS S1 ON S1.step = E.step AND S1.c = E.s
            JOIN S AS S2 ON S2.step = E.step AND S2.c = E.d
        """
        df = con.execute(qry, [edges_files, int(step_min), int(step_max),
                               state_files, int(step_min), int(step_max)]).df()
        if df.empty:
            return (np.zeros(0, dtype=np.int64),
                    *(np.zeros(0) for _ in range(6)),
                    np.zeros(1, dtype=np.int64))
        df = df.sort_values(["step"], kind="mergesort").reset_index(drop=True)
        steps = df["step"].to_numpy(dtype=np.int64)
        x0 = df["x0"].to_numpy(dtype=np.float64)
        y0 = df["y0"].to_numpy(dtype=np.float64)
        z0 = df["z0"].to_numpy(dtype=np.float64)
        x1 = df["x1"].to_numpy(dtype=np.float64)
        y1 = df["y1"].to_numpy(dtype=np.float64)
        z1 = df["z1"].to_numpy(dtype=np.float64)
    else:
        # No usable edges
        return (np.zeros(0, dtype=np.int64),
                *(np.zeros(0) for _ in range(6)),
                np.zeros(1, dtype=np.int64))

    # build prefix sums per unique step
    uniq, counts = np.unique(steps, return_counts=True)
    pref = np.zeros(len(uniq) + 1, dtype=np.int64)
    pref[1:] = np.cumsum(counts, dtype=np.int64)

    return uniq, x0, y0, z0, x1, y1, z1, pref


# -------------------- main pack --------------------

def pack_to_npz(prefix: str,
                run_id: str,
                step_min: int,
                step_max: int,
                out_path: str,
                allow_gsutil: bool = False) -> None:
    print(f"[pack] run={run_id} window=[{step_min},{step_max}] https={prefix} gsutil={int(allow_gsutil)}", flush=True)

    state_files = discover_files(prefix, run_id, "state", allow_gsutil=allow_gsutil)
    edges_files = discover_files(prefix, run_id, "edges", allow_gsutil=allow_gsutil)

    print(f"[pack] discovered: state_files={len(state_files)} edges_files={len(edges_files)}", flush=True)
    if not state_files and not edges_files:
        raise SystemExit("No state or edges files found (index/manifest).")

    con = connect_duckdb()

    s_steps, sx, sy, sz, sval, s_idx = extract_state(con, state_files, step_min, step_max)
    e_steps, x0, y0, z0, x1, y1, z1, e_idx = extract_edges(con, edges_files, state_files, step_min, step_max)

    # unify steps: we want a single 'steps' that covers either source
    if s_steps.size and e_steps.size:
        steps = np.union1d(s_steps, e_steps)
    elif s_steps.size:
        steps = s_steps
    else:
        steps = e_steps

    # Remap per-source prefix arrays onto unified 'steps'
    # Build new prefix of length len(steps)+1
    def remap_prefix(steps_src: np.ndarray, prefix_src: np.ndarray, total_len: int) -> np.ndarray:
        # prefix_src must be length len(steps_src)+1
        if prefix_src.size == 0:
            return np.zeros(steps.size + 1, dtype=np.int64)
        if prefix_src.size == len(steps_src):
            # older code might have stored length==len(steps_src); fix to +1
            fixed = np.zeros(len(steps_src) + 1, dtype=np.int64)
            fixed[1:] = prefix_src
            prefix_src = fixed
        out = np.zeros(steps.size + 1, dtype=np.int64)
        # walk both lists
        i_src = 0
        cur = 0
        for i_all, st in enumerate(steps):
            if i_src < len(steps_src) and steps_src[i_src] == st:
                nxt = int(prefix_src[i_src + 1])
                out[i_all + 1] = nxt
                cur = nxt
                i_src += 1
            else:
                out[i_all + 1] = cur
        # sanity
        if out[-1] != total_len:
            # Some inputs may not fully align; clamp to total length
            out[-1] = total_len
        return out

    s_idx_u = remap_prefix(s_steps, s_idx, total_len=sx.size)
    e_idx_u = remap_prefix(e_steps, e_idx, total_len=x0.size)

    # Save guaranteed keys. Missing arrays are empty but present.
    payload = {
        "steps": steps.astype(np.int64),

        "state_x": sx.astype(np.float64),
        "state_y": sy.astype(np.float64),
        "state_z": sz.astype(np.float64),
        "state_value": sval.astype(np.float64),
        "state_idx": s_idx_u.astype(np.int64),

        "edges_x0": x0.astype(np.float64),
        "edges_y0": y0.astype(np.float64),
        "edges_z0": z0.astype(np.float64),
        "edges_x1": x1.astype(np.float64),
        "edges_y1": y1.astype(np.float64),
        "edges_z1": z1.astype(np.float64),
        "edges_idx": e_idx_u.astype(np.int64),
    }

    np.savez(out_path, **payload)
    print(f"[pack] wrote {out_path} | steps={steps.size} state_rows={sx.size} edge_rows={x0.size}", flush=True)


# -------------------- CLI --------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pack Fuka shards (HTTPS index/manifest) into a single NPZ for Manim.")
    ap.add_argument("--prefix", required=True, help="HTTPS prefix, e.g. https://storage.googleapis.com/fuka4-runs")
    ap.add_argument("--run_id", required=True, help="Run id")
    ap.add_argument("--step_min", type=int, required=True)
    ap.add_argument("--step_max", type=int, required=True)
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--allow-gsutil", action="store_true", help="Allow gsutil listing fallback")
    args = ap.parse_args(argv)

    if args.step_max <= args.step_min:
        print("step_max must be > step_min", file=sys.stderr)
        return 2

    pack_to_npz(
        prefix=args.prefix,
        run_id=args.run_id,
        step_min=int(args.step_min),
        step_max=int(args.step_max),
        out_path=args.out,
        allow_gsutil=bool(args.allow_gsutil),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())