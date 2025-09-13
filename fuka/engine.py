# fuka/engine.py
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ------------------------------- Config --------------------------------

@dataclass(frozen=True)
class EngineConfig:
    data_root: str
    run_id: str
    steps: int
    seed: int
    grid_shape: Tuple[int, int, int]  # (nx, ny, nz)
    url_prefix: str = ""
    # physics knobs (kept minimal & explicit)
    alpha: float = 0.12        # diffusion strength in [0,1]
    noise: float = 0.01        # per-step additive noise
    dt_seconds: float = 0.1    # timestep in seconds


# ------------------------------- Helpers -------------------------------

def _mk_dirs(root: Path) -> Path:
    shards = root / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    return shards


def _arrow_write(path: Path, frame: pd.DataFrame) -> None:
    """
    Write a pandas frame to Parquet with efficient defaults (float32 where applicable).
    """
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        where=str(path),
        compression="zstd",
        use_dictionary=True,
        data_page_size=1 << 16,
        write_statistics=True,
    )


def _neighbor_offsets() -> np.ndarray:
    # 6-neighborhood (von Neumann)
    return np.array(
        [[ 1, 0, 0], [-1, 0, 0],
         [ 0, 1, 0], [ 0,-1, 0],
         [ 0, 0, 1], [ 0, 0,-1]], dtype=np.int32
    )


# ------------------------------ Engine ---------------------------------

class Engine:
    """
    Deterministic headless engine.

    Writes canonical Parquet shards for each step:
      - state_{step:06d}.parquet : step,x,y,z,value (float32)
      - edges_{step:06d}.parquet : step,x0,y0,z0,x1,y1,z1,v0,v1 (float32)
      - env_{step:06d}.parquet   : step,dt_seconds,alpha,nx,ny,nz
    """

    def __init__(
        self,
        data_root: str,
        run_id: str,
        steps: int,
        seed: int,
        grid_shape: Tuple[int, int, int],
        physics_cfg: Dict = None,
        url_prefix: str = "",
    ) -> None:
        physics_cfg = physics_cfg or {}
        alpha = float(physics_cfg.get("alpha", 0.12))
        noise = float(physics_cfg.get("noise", 0.01))
        dt    = float(physics_cfg.get("dt_seconds", 0.1))

        self.cfg = EngineConfig(
            data_root=data_root,
            run_id=run_id,
            steps=int(steps),
            seed=int(seed),
            grid_shape=(int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])),
            url_prefix=url_prefix or "",
            alpha=alpha, noise=noise, dt_seconds=dt,
        )

        # IO paths
        self.run_root = Path(self.cfg.data_root) / "runs" / self.cfg.run_id
        self.shards_dir = _mk_dirs(self.run_root)

        # RNG & constants
        self.rng = np.random.default_rng(self.cfg.seed)
        self.nei = _neighbor_offsets()

        # state buffer
        nx, ny, nz = self.cfg.grid_shape
        self.state = np.zeros((nx, ny, nz), dtype=np.float32)

        # simple seeded impulse to kick dynamics (center + small noise)
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        self.state[cx, cy, cz] = 1.0
        self.state += (self.rng.standard_normal(self.state.shape).astype(np.float32)) * (self.cfg.noise * 0.5)

    # ------------------------------ Core update ------------------------------

    def _laplacian_3d(self, g: np.ndarray) -> np.ndarray:
        """
        3D 6-neighbor Laplacian with Neumann (reflecting) boundary conditions.
        """
        # pad reflect
        gp = np.pad(g, pad_width=1, mode="edge")
        center = gp[1:-1,1:-1,1:-1]

        lap = (
            gp[2:  ,1:-1,1:-1] + gp[0:-2,1:-1,1:-1] +
            gp[1:-1,2:  ,1:-1] + gp[1:-1,0:-2,1:-1] +
            gp[1:-1,1:-1,2:  ] + gp[1:-1,1:-1,0:-2] -
            6.0 * center
        )
        return lap.astype(np.float32, copy=False)

    def _advance(self) -> np.ndarray:
        """
        Advance state by one step using a stable explicit scheme:
          s_{t+1} = s_t + alpha * Laplacian(s_t) + noise
        """
        g = self.state
        lap = self._laplacian_3d(g)
        new = g + self.cfg.alpha * lap
        if self.cfg.noise > 0:
            new += self.rng.standard_normal(g.shape).astype(np.float32) * self.cfg.noise
        # clamp to a reasonable range to avoid float blow-ups
        np.clip(new, -10.0, 10.0, out=new)
        return new.astype(np.float32, copy=False)

    # ------------------------------ Edges ------------------------------------

    def _sample_edges(self, prev: np.ndarray, new: np.ndarray, max_edges: int) -> pd.DataFrame:
        """
        Build a sparse set of edges connecting a cell to one random neighbor
        (biased by magnitude of change). Returns a pandas DataFrame.
        """
        nx, ny, nz = prev.shape
        total = nx * ny * nz

        # weight by absolute change; sample without replacement
        delta = np.abs(new - prev).reshape(-1)
        w = delta / (delta.sum() + 1e-8)

        k = min(max_edges, total)
        if k <= 0:
            return pd.DataFrame(columns=["step","x0","y0","z0","x1","y1","z1","v0","v1"], dtype=np.float32)

        idxs = self.rng.choice(total, size=k, replace=False, p=w)
        x0 = (idxs // (ny * nz)).astype(np.int32)
        y0 = ((idxs // nz) % ny).astype(np.int32)
        z0 = (idxs % nz).astype(np.int32)

        # pick a random neighbor per sampled cell; clamp at borders
        off = self.nei[self.rng.integers(0, len(self.nei), size=k)]
        x1 = np.clip(x0 + off[:,0], 0, nx-1).astype(np.int32)
        y1 = np.clip(y0 + off[:,1], 0, ny-1).astype(np.int32)
        z1 = np.clip(z0 + off[:,2], 0, nz-1).astype(np.int32)

        v0 = prev[x0, y0, z0].astype(np.float32)
        v1 = new [x1, y1, z1].astype(np.float32)

        df = pd.DataFrame({
            "x0": x0, "y0": y0, "z0": z0,
            "x1": x1, "y1": y1, "z1": z1,
            "v0": v0, "v1": v1,
        })
        return df

    # ------------------------------ Writers ----------------------------------

    def _write_state(self, step: int, grid: np.ndarray) -> None:
        nx, ny, nz = grid.shape
        xs, ys, zs = np.indices((nx, ny, nz), dtype=np.int32)
        df = pd.DataFrame({
            "step": np.full(xs.size, step, dtype=np.int32),
            "x": xs.ravel(order="C"),
            "y": ys.ravel(order="C"),
            "z": zs.ravel(order="C"),
            "value": grid.ravel(order="C").astype(np.float32),
        })
        shard = self.shards_dir / f"state_{step:06d}.parquet"
        _arrow_write(shard, df)

    def _write_edges(self, step: int, df_edges: pd.DataFrame) -> None:
        if df_edges.empty:
            # still write an empty shard to keep indexing simple
            df = pd.DataFrame(
                {"step": np.array([], dtype=np.int32),
                 "x0": [], "y0": [], "z0": [],
                 "x1": [], "y1": [], "z1": [],
                 "v0": np.array([], dtype=np.float32),
                 "v1": np.array([], dtype=np.float32)}
            )
        else:
            df = df_edges.copy()
            df.insert(0, "step", np.full(len(df), step, dtype=np.int32))
            # enforce dtypes
            for c in ("x0","y0","z0","x1","y1","z1"):
                df[c] = df[c].astype(np.int32, copy=False)
            df["v0"] = df["v0"].astype(np.float32, copy=False)
            df["v1"] = df["v1"].astype(np.float32, copy=False)

        shard = self.shards_dir / f"edges_{step:06d}.parquet"
        _arrow_write(shard, df)

    def _write_env(self, step: int) -> None:
        nx, ny, nz = self.cfg.grid_shape
        df = pd.DataFrame({
            "step": [np.int32(step)],
            "dt_seconds": [np.float32(self.cfg.dt_seconds)],
            "alpha": [np.float32(self.cfg.alpha)],
            "nx": [np.int32(nx)], "ny": [np.int32(ny)], "nz": [np.int32(nz)],
        })
        shard = self.shards_dir / f"env_{step:06d}.parquet"
        _arrow_write(shard, df)

    # ------------------------------ Run --------------------------------------

    def run(self) -> None:
        """
        Execute steps and write shards.
        """
        nx, ny, nz = self.cfg.grid_shape
        prev = self.state

        # choose an edges budget that scales gently with grid size
        # (kept small for smoke/demo, linear-ish for larger grids)
        base = max(nx, ny, nz)
        max_edges = int(min(nx*ny*nz, max(256, base * base)))

        for step in range(self.cfg.steps):
            new = self._advance()

            # record
            self._write_state(step, new)
            edges_df = self._sample_edges(prev, new, max_edges=max_edges)
            self._write_edges(step, edges_df)
            self._write_env(step)

            # rotate
            prev = new
            self.state = new

        # small marker so the UI / scripts can assert completion without scanning shards
        (self.run_root / "_done.marker").write_text("ok", encoding="utf-8")