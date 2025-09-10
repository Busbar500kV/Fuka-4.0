# fuka/encoder.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np


@dataclass
class EncoderCfg:
    """
    Feature encoder for visualization/analysis.

    Guarantees:
      - On every call to step(), writes (optionally, controlled by `every`)
        a cumulative NPZ artifact containing **all encoded connections up to that step**.
      - On finalize(), writes a full roll-up NPZ for the entire run.

    Parameters
    ----------
    enabled : bool
        Master switch. If False, the encoder is inert.
    every : int
        Write frequency in steps. 1 = write every step (default).
    topk_edges : int
        Number of strongest neighbour edges to encode per step (global top-k over 6-neighbour pairs).
    cumulative : bool
        Always cumulative (required by design). Kept for future modes; must be True.
    out_subdir : str
        Directory name (under the run folder) where NPZ files are written.
    max_edges_total : int
        Safety cap on total encoded edges across the whole run.
    """
    enabled: bool = True
    every: int = 1
    topk_edges: int = 2000
    cumulative: bool = True
    out_subdir: str = "enc"
    max_edges_total: int = 5_000_000


class Encoder:
    """
    Headless-safe encoder that extracts strongest 6-neighbour connections
    and writes NPZ artifacts usable by renderers (e.g., Manim).

    NPZ schema per artifact (mirrors pack_npz canonical edge fields):
        steps:         (F,) int64
        edges_x0..z1:  (Ne,) float64
        edges_idx:     (F+1,) int64   # prefix-sum (cumulative by step)
        edge_value:    (Ne,) float64  # 0.5 * (v0 + v1)
        edge_strength: (Ne,) float64  # |v1 - v0|

    Notes
    -----
    - `step(E_in)` may be passed a snapshot of the field *before* physics mixing
      (as in the Engine), but will fall back to world.energy if None.
    - Maintains its own step counter (`self._step`) so it does not rely on global mutable state.
    """

    def __init__(self, world, cfg: EncoderCfg, run_dir: Path) -> None:
        self.world = world
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.out_dir = self.run_dir / cfg.out_subdir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # running state
        self._step: int = -1  # incremented to 0 on first step()
        self._steps: List[int] = []          # recorded step numbers
        # cumulative stores
        self._x0: List[float] = []
        self._y0: List[float] = []
        self._z0: List[float] = []
        self._x1: List[float] = []
        self._y1: List[float] = []
        self._z1: List[float] = []
        self._v0: List[float] = []
        self._v1: List[float] = []
        self._edges_idx: List[int] = [0]     # prefix sum (starts with 0)
        self._total_edges: int = 0

        # write a tiny meta file once
        meta = {
            "note": "Per-step cumulative edge encodings for this run.",
            "topk_edges": int(self.cfg.topk_edges),
            "every": int(self.cfg.every),
            "cumulative": bool(self.cfg.cumulative),
            "max_edges_total": int(self.cfg.max_edges_total),
        }
        (self.out_dir / "encoder_meta.json").write_text(_json_dumps(meta), encoding="utf-8")

    # ----------------------------- public API -----------------------------

    def step(self, E_in: Optional[np.ndarray] = None) -> None:
        if not self.cfg.enabled:
            return
        self._step += 1
        step = self._step

        # choose field to analyze
        E = E_in if E_in is not None else self.world.energy
        if not isinstance(E, np.ndarray) or E.ndim != 3:
            return  # safety no-op

        # compute strongest neighbour edges for this frame
        x0, y0, z0, x1, y1, z1, v0, v1 = _edge_pairs_topk_by_grad(E, self.cfg.topk_edges)

        # append to cumulative stores (respect max_edges_total)
        add = int(x0.size)
        if add > 0:
            room = max(0, self.cfg.max_edges_total - self._total_edges)
            if room <= 0:
                # stop encoding further edges; still advance bookkeeping
                self._edges_idx.append(self._total_edges)
            else:
                take = min(room, add)
                sl = slice(0, take)
                self._x0.extend(x0[sl].astype(np.float64).tolist())
                self._y0.extend(y0[sl].astype(np.float64).tolist())
                self._z0.extend(z0[sl].astype(np.float64).tolist())
                self._x1.extend(x1[sl].astype(np.float64).tolist())
                self._y1.extend(y1[sl].astype(np.float64).tolist())
                self._z1.extend(z1[sl].astype(np.float64).tolist())
                self._v0.extend(v0[sl].astype(np.float64).tolist())
                self._v1.extend(v1[sl].astype(np.float64).tolist())
                self._total_edges += take
                self._edges_idx.append(self._total_edges)
        else:
            # no edges this step; just repeat prefix sum
            self._edges_idx.append(self._total_edges)

        self._steps.append(step)

        # periodic write (cumulative up to this step)
        if self.cfg.every > 0 and (step % self.cfg.every) == 0:
            self._write_cumulative_npz(step)

    def save(self) -> None:
        """Write a final roll-up NPZ (all steps)."""
        if not self.cfg.enabled:
            return
        self._write_cumulative_npz(self._step, final=True)

    # ----------------------------- internals -----------------------------

    def _write_cumulative_npz(self, step: int, *, final: bool = False) -> None:
        # convert lists to numpy
        steps = np.asarray(self._steps, dtype=np.int64)
        x0 = np.asarray(self._x0, dtype=np.float64)
        y0 = np.asarray(self._y0, dtype=np.float64)
        z0 = np.asarray(self._z0, dtype=np.float64)
        x1 = np.asarray(self._x1, dtype=np.float64)
        y1 = np.asarray(self._y1, dtype=np.float64)
        z1 = np.asarray(self._z1, dtype=np.float64)
        v0 = np.asarray(self._v0, dtype=np.float64)
        v1 = np.asarray(self._v1, dtype=np.float64)
        eidx = np.asarray(self._edges_idx, dtype=np.int64)  # length F+1

        # canonical derived fields
        edge_value = 0.5 * (v0 + v1)
        edge_strength = np.abs(v1 - v0)

        # filename strategy
        if final:
            out = self.out_dir / "edges_all.npz"
        else:
            out = self.out_dir / f"edges_step_{step:04d}.npz"

        np.savez(
            out,
            steps=steps,
            edges_x0=x0, edges_y0=y0, edges_z0=z0,
            edges_x1=x1, edges_y1=y1, edges_z1=z1,
            edges_idx=eidx,
            edge_value=edge_value,
            edge_strength=edge_strength,
        )

# ----------------------------- utilities -----------------------------

def _edge_pairs_topk_by_grad(E: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-axis neighbour differences and select global top-k by |delta|.
    Returns arrays: x0,y0,z0,x1,y1,z1,v0,v1  (all 1D of length <= k)
    """
    nx, ny, nz = E.shape
    if min(nx, ny, nz) <= 0 or k <= 0:
        return _empty_edges()

    # Differences along axes
    dx = E[1:, :, :] - E[:-1, :, :]
    dy = E[:, 1:, :] - E[:, :-1, :]
    dz = E[:, :, 1:] - E[:, :, :-1]

    # Flatten magnitudes with offsets to recover coordinates
    mag_x = np.abs(dx).reshape(-1)
    mag_y = np.abs(dy).reshape(-1)
    mag_z = np.abs(dz).reshape(-1)

    sizes = (mag_x.size, mag_y.size, mag_z.size)
    total = sum(sizes)
    if total == 0:
        return _empty_edges()

    k = min(k, total)
    cat = np.concatenate([mag_x, mag_y, mag_z], axis=0)
    # top-k indices into concatenated array
    gidx = _topk_flat_indices(cat, k)

    # prepare outputs
    out_x0 = np.empty((k,), dtype=np.float32)
    out_y0 = np.empty((k,), dtype=np.float32)
    out_z0 = np.empty((k,), dtype=np.float32)
    out_x1 = np.empty((k,), dtype=np.float32)
    out_y1 = np.empty((k,), dtype=np.float32)
    out_z1 = np.empty((k,), dtype=np.float32)
    out_v0 = np.empty((k,), dtype=np.float32)
    out_v1 = np.empty((k,), dtype=np.float32)

    off_x = 0
    off_y = sizes[0]
    off_z = sizes[0] + sizes[1]

    for i, gi in enumerate(gidx):
        gi = int(gi)
        if gi < off_y:
            # X-edge
            li = gi - off_x
            ix = li // (ny * nz)
            rem = li % (ny * nz)
            iy = rem // nz
            iz = rem % nz
            x0 = ix; y0 = iy; z0 = iz
            x1 = ix + 1; y1 = iy; z1 = iz
        elif gi < off_z:
            # Y-edge
            li = gi - off_y
            ix = li // ((ny - 1) * nz)
            rem = li % ((ny - 1) * nz)
            iy = rem // nz
            iz = rem % nz
            x0 = ix; y0 = iy; z0 = iz
            x1 = ix; y1 = iy + 1; z1 = iz
        else:
            # Z-edge
            li = gi - off_z
            ix = li // (ny * (nz - 1))
            rem = li % (ny * (nz - 1))
            iy = rem // (nz - 1)
            iz = rem % (nz - 1)
            x0 = ix; y0 = iy; z0 = iz
            x1 = ix; y1 = iy; z1 = iz + 1

        out_x0[i] = x0; out_y0[i] = y0; out_z0[i] = z0
        out_x1[i] = x1; out_y1[i] = y1; out_z1[i] = z1
        out_v0[i] = E[int(x0), int(y0), int(z0)]
        out_v1[i] = E[int(x1), int(y1), int(z1)]

    return out_x0, out_y0, out_z0, out_x1, out_y1, out_z1, out_v0, out_v1


def _topk_flat_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k values of arr (1D), descending order."""
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    n = arr.size
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    if k >= n:
        order = np.argsort(-arr, kind="stable")
        return order
    idx = np.argpartition(arr, -k)[-k:]
    vals = arr[idx]
    order = np.argsort(-vals, kind="stable")
    return idx[order]


def _empty_edges() -> Tuple[np.ndarray, ...]:
    return tuple(np.zeros((0,), dtype=np.float32) for _ in range(8))


# ----------------------------- tiny utils -----------------------------

def _json_dumps(obj: Dict[str, Any]) -> str:
    import json
    return json.dumps(obj, indent=2)