# fuka/render/pack_npz.py
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


"""
Canonical NPZ packer (Master Plan v2)

Reads shards under:
  {data_root}/runs/{run_id}/shards/{state|edges|env}_{step:06d}.parquet

Emits a single NPZ with keys:
  - steps:        int32[NumSteps]
  - state_idx:    int64[NumSteps+1]   # prefix-sum row offsets into state_* arrays
  - edges_idx:    int64[NumSteps+1]   # prefix-sum row offsets into edges_* arrays

  - state_x, state_y, state_z: int32[TotalStateRows]
  - state_value:               float32[TotalStateRows]

  - edges_x0, edges_y0, edges_z0,
    edges_x1, edges_y1, edges_z1:     int32[TotalEdgeRows]
  - edge_v0, edge_v1:                  float32[TotalEdgeRows]
  - edge_value:                        float32[TotalEdgeRows]   # alias of v1 for convenience
  - edge_strength:                     float32[TotalEdgeRows]   # |v1 - v0|

This schema is what the Manim scene and analytics expect.
"""

# ------------------------------ helpers ------------------------------ #

def _shard_paths(data_root: Path, run_id: str, step: int) -> Tuple[Path, Path, Path]:
    rd = data_root / "runs" / run_id / "shards"
    return (
        rd / f"state_{step:06d}.parquet",
        rd / f"edges_{step:06d}.parquet",
        rd / f"env_{step:06d}.parquet",
    )

def _discover_steps(data_root: Path, run_id: str, lo: int, hi: int) -> List[int]:
    steps: List[int] = []
    for s in range(int(lo), int(hi) + 1):
        state_p, _, _ = _shard_paths(data_root, run_id, s)
        if state_p.exists():
            steps.append(s)
    return steps

def _read_state(path: Path) -> pd.DataFrame:
    # Expect columns: step,x,y,z,value
    df = pd.read_parquet(path)
    # enforce dtypes
    for c in ("step", "x", "y", "z"):
        if df[c].dtype != np.int32:
            df[c] = df[c].astype(np.int32, copy=False)
    if df["value"].dtype != np.float32:
        df["value"] = df["value"].astype(np.float32, copy=False)
    return df

def _read_edges(path: Path) -> pd.DataFrame:
    # Expect columns: step,x0,y0,z0,x1,y1,z1,v0,v1
    if not path.exists():
        # allow missing edges shard: produce empty frame
        cols = ["step","x0","y0","z0","x1","y1","z1","v0","v1"]
        return pd.DataFrame({c: np.array([], dtype=np.int32 if c in cols[:7] else np.float32) for c in cols})
    df = pd.read_parquet(path)
    for c in ("step","x0","y0","z0","x1","y1","z1"):
        if df[c].dtype != np.int32:
            df[c] = df[c].astype(np.int32, copy=False)
    for c in ("v0","v1"):
        if df[c].dtype != np.float32:
            df[c] = df[c].astype(np.float32, copy=False)
    return df

# ------------------------------- pack -------------------------------- #

def pack(
    prefix: str = "",
    run_id: str = "",
    step_min: int = 0,
    step_max: int = 0,
    out: str = "assets/fuka_anim.npz",
    data_root: str | Path | None = None,
) -> bool:
    """
    Pack shards→NPZ. Returns True on success, False on any failure.
    """
    try:
        if not run_id:
            raise ValueError("pack: run_id is required")

        root = Path(data_root or os.environ.get("FUKA_DATA_ROOT", "data")).resolve()
        out_p = Path(out).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        steps = _discover_steps(root, run_id, step_min, step_max)
        if not steps:
            raise FileNotFoundError(
                f"pack: no state shards found under {root}/runs/{run_id}/shards/ "
                f"in window [{step_min},{step_max}]"
            )

        # Accumulators
        state_x: List[np.ndarray] = []
        state_y: List[np.ndarray] = []
        state_z: List[np.ndarray] = []
        state_val: List[np.ndarray] = []
        state_idx: List[int] = [0]

        edges_x0: List[np.ndarray] = []
        edges_y0: List[np.ndarray] = []
        edges_z0: List[np.ndarray] = []
        edges_x1: List[np.ndarray] = []
        edges_y1: List[np.ndarray] = []
        edges_z1: List[np.ndarray] = []
        edge_v0: List[np.ndarray] = []
        edge_v1: List[np.ndarray] = []
        edge_strength: List[np.ndarray] = []
        edges_idx: List[int] = [0]

        # Per-step loop
        for s in steps:
            sp, ep, _ = _shard_paths(root, run_id, s)

            sdf = _read_state(sp)
            state_x.append(sdf["x"].to_numpy(copy=False))
            state_y.append(sdf["y"].to_numpy(copy=False))
            state_z.append(sdf["z"].to_numpy(copy=False))
            state_val.append(sdf["value"].to_numpy(copy=False))
            state_idx.append(state_idx[-1] + len(sdf))

            edf = _read_edges(ep)
            if len(edf):
                edges_x0.append(edf["x0"].to_numpy(copy=False))
                edges_y0.append(edf["y0"].to_numpy(copy=False))
                edges_z0.append(edf["z0"].to_numpy(copy=False))
                edges_x1.append(edf["x1"].to_numpy(copy=False))
                edges_y1.append(edf["y1"].to_numpy(copy=False))
                edges_z1.append(edf["z1"].to_numpy(copy=False))
                v0 = edf["v0"].to_numpy(copy=False)
                v1 = edf["v1"].to_numpy(copy=False)
                edge_v0.append(v0)
                edge_v1.append(v1)
                edge_strength.append(np.abs(v1 - v0).astype(np.float32, copy=False))
            edges_idx.append(edges_idx[-1] + len(edf))

        # Concatenate
        def cat(parts: List[np.ndarray], dtype) -> np.ndarray:
            if not parts:
                return np.asarray([], dtype=dtype)
            return np.concatenate(parts).astype(dtype, copy=False)

        npz_payload: Dict[str, np.ndarray] = {
            "steps": np.asarray(steps, dtype=np.int32),
            "state_idx": np.asarray(state_idx, dtype=np.int64),
            "edges_idx": np.asarray(edges_idx, dtype=np.int64),

            "state_x": cat(state_x, np.int32),
            "state_y": cat(state_y, np.int32),
            "state_z": cat(state_z, np.int32),
            "state_value": cat(state_val, np.float32),

            "edges_x0": cat(edges_x0, np.int32),
            "edges_y0": cat(edges_y0, np.int32),
            "edges_z0": cat(edges_z0, np.int32),
            "edges_x1": cat(edges_x1, np.int32),
            "edges_y1": cat(edges_y1, np.int32),
            "edges_z1": cat(edges_z1, np.int32),
            "edge_v0": cat(edge_v0, np.float32),
            "edge_v1": cat(edge_v1, np.float32),
            "edge_value": cat(edge_v1, np.float32),          # alias
            "edge_strength": cat(edge_strength, np.float32),
        }

        # Save
        np.savez(out_p, **npz_payload)

        total_state = int(npz_payload["state_idx"][-1])
        total_edges = int(npz_payload["edges_idx"][-1])
        print(
            f"[pack] wrote {out_p} | steps={len(steps)} "
            f"state_rows={total_state} edge_rows={total_edges}"
        )
        return True

    except Exception as e:
        print(f"[pack] ERROR: {e}")
        return False


# ------------------------------- CLI --------------------------------- #

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pack Fuka shards into canonical NPZ")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--step_min", type=int, default=0)
    ap.add_argument("--step_max", type=int, default=300)
    ap.add_argument("--out", default="assets/fuka_anim.npz")
    ap.add_argument("--prefix", default=os.environ.get("DATA_URL_PREFIX", ""))
    ap.add_argument("--data_root", default=os.environ.get("FUKA_DATA_ROOT", "data"))
    return ap.parse_args()

def main() -> None:
    args = _parse_args()
    ok = pack(
        prefix=args.prefix,
        run_id=args.run_id,
        step_min=args.step_min,
        step_max=args.step_max,
        out=args.out,
        data_root=args.data_root,
    )
    if not ok:
        raise SystemExit(2)

if __name__ == "__main__":
    main()