# render/manim_fuka_scene.py
from __future__ import annotations

import os
import math
import numpy as np
from typing import Dict, Tuple

from manim import (
    Scene, ThreeDScene, ORIGIN, BLUE, YELLOW, WHITE,
    Dot3D, Line3D, VGroup, config, FadeIn, FadeOut
)

# -------- helpers --------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _downsample_idx(n: int, k: int, seed: int = 0) -> np.ndarray:
    """return indices [0..n) of size <= k (random sample if needed)."""
    n = int(n)
    k = int(k)
    if n <= k:
        return np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False).astype(np.int32))

def _value_to_color(v: np.ndarray) -> np.ndarray:
    """map scalar -> RGB via a simple blue→yellow ramp (0..1)."""
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return np.zeros((0, 3))
    m = np.nanmax(v)
    if not np.isfinite(m) or m <= 1e-12:
        t = np.zeros_like(v)
    else:
        t = np.clip(v / m, 0.0, 1.0)
    # blue (0,0,1) to yellow (1,1,0)
    r = t
    g = t
    b = 1.0 - t
    return np.stack([r, g, b], axis=1)

# -------- NPZ loader --------

class FukaFrames:
    """
    Lightweight reader for NPZ files produced by pack_npz.
    Supports:
      - v3D: arrays 'steps' (int), 'state_x','state_y','state_z','state_value',
             'state_idx' (exclusive prefix sums), optional edges_* and edges_idx
      - legacy: flat arrays with 'row_step' to filter by step
    """
    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        self.z = np.load(npz_path, allow_pickle=False)
        self.mode = self._detect_mode()
        self.steps = self._steps()

    def _has_keys(self, *keys: str) -> bool:
        return all(k in self.z.files for k in keys)

    def _detect_mode(self) -> str:
        if self._has_keys("steps", "state_x", "state_y", "state_z", "state_value"):
            return "v3d"
        elif self._has_keys("row_step", "x", "y", "z"):
            return "legacy"
        else:
            raise ValueError(
                f"Unrecognized NPZ schema for {self.npz_path}; keys={sorted(self.z.files)}"
            )

    def _steps(self) -> np.ndarray:
        if self.mode == "v3d":
            return self.z["steps"].astype(np.int64)
        else:
            return np.unique(self.z["row_step"].astype(np.int64))

    def nframes(self) -> int:
        return int(len(self.steps))

    def state_for_step(self, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.mode == "v3d":
            steps = self.z["steps"]
            idx = self.z.get("state_idx")  # optional prefix sums per frame
            x = self.z["state_x"]; y=self.z["state_y"]; z=self.z["state_z"]; v=self.z.get("state_value")
            v = v if v is not None else np.ones_like(x)
            if idx is not None:
                i = int(np.searchsorted(steps, step))
                if i >= len(steps) or steps[i] != step:
                    return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
                lo = int(idx[i]); hi = int(idx[i+1]) if i+1 < len(idx) else len(x)
                return x[lo:hi], y[lo:hi], z[lo:hi], v[lo:hi]
            else:
                # fallback: full arrays correspond to this lone step
                return x, y, z, v
        else:
            rs = self.z["row_step"].astype(np.int64)
            sel = (rs == int(step))
            x = self.z["x"][sel]; y=self.z["y"][sel]; z=self.z["z"][sel]
            v = self.z["value"][sel] if "value" in self.z.files else np.ones_like(x)
            return x, y, z, v

    def edges_for_step(self, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.mode == "v3d":
            if not self._has_keys("edges_x0","edges_y0","edges_z0","edges_x1","edges_y1","edges_z1"):
                return tuple(np.empty(0) for _ in range(6))
            steps = self.z["steps"]
            idx = self.z.get("edges_idx")
            x0=self.z["edges_x0"]; y0=self.z["edges_y0"]; z0=self.z["edges_z0"]
            x1=self.z["edges_x1"]; y1=self.z["edges_y1"]; z1=self.z["edges_z1"]
            if idx is not None:
                i = int(np.searchsorted(steps, step))
                if i >= len(steps) or steps[i] != step:
                    return tuple(np.empty(0) for _ in range(6))
                lo = int(idx[i]); hi = int(idx[i+1]) if i+1 < len(idx) else len(x0)
                return x0[lo:hi], y0[lo:hi], z0[lo:hi], x1[lo:hi], y1[lo:hi], z1[lo:hi]
            else:
                return x0, y0, z0, x1, y1, z1
        else:
            # legacy has no edges in the NPZ normally
            return tuple(np.empty(0) for _ in range(6))

# -------- Manim scene --------

class FukaWorldEdges3D(ThreeDScene):
    def construct(self):
        npz_path = os.environ.get("FUKA_NPZ", "")
        if not npz_path or not os.path.exists(npz_path):
            raise FileNotFoundError(f"FUKA_NPZ not found: {npz_path}")

        step_seconds = _env_float("FUKA_STEP_SECONDS", 0.20)
        max_points   = _env_int("FUKA_MAX_POINTS", 80000)
        max_edges    = _env_int("FUKA_MAX_EDGES", 20000)
        point_r      = _env_float("FUKA_POINT_RADIUS", 0.035)
        edge_w       = _env_float("FUKA_EDGE_WIDTH", 2.5)
        show_edges   = _env_bool("FUKA_SHOW_EDGES", True)

        frames = FukaFrames(npz_path)
        n = frames.nframes()
        if n == 0:
            return

        # camera defaults
        self.set_camera_orientation(phi=70*math.pi/180, theta=45*math.pi/180)

        for i, step in enumerate(frames.steps):
            # --- state points ---
            x, y, z, val = frames.state_for_step(int(step))
            pts = VGroup()
            if x.size:
                keep = _downsample_idx(x.size, max_points, seed=int(step) & 0xFFFF)
                xs, ys, zs = x[keep], y[keep], z[keep]
                cols = _value_to_color(val[keep])
                for j in range(xs.shape[0]):
                    d = Dot3D(point=(float(xs[j]), float(ys[j]), float(zs[j])))
                    rgb = cols[j]; d.set_color((float(rgb[0]), float(rgb[1]), float(rgb[2])))
                    d.radius = point_r
                    pts.add(d)

            # --- edges ---
            segs = VGroup()
            if show_edges:
                x0,y0,z0,x1,y1,z1 = frames.edges_for_step(int(step))
                if x0.size:
                    keep = _downsample_idx(x0.size, max_edges, seed=(int(step)+12345) & 0xFFFF)
                    for j in keep:
                        segs.add(Line3D(
                            start=(float(x0[j]), float(y0[j]), float(z0[j])),
                            end  =(float(x1[j]), float(y1[j]), float(z1[j])),
                            thickness=edge_w
                        ))

            # animate: fade in → hold → fade out
            grp = VGroup(pts, segs)
            self.add(grp)
            self.wait(step_seconds)
            self.remove(grp)