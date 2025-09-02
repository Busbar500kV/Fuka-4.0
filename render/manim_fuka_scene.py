# render/manim_fuka_scene.py
from __future__ import annotations

import os
import math
import random
import colorsys
from typing import Tuple, Iterable

import numpy as np
from manim import (
    Scene, ThreeDScene, Dot3D, Line3D, VGroup,
    ORIGIN, config
)

# -----------------------------
# Helpers
# -----------------------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

def _normalize(vals: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(vals)) if vals.size else 0.0
    vmax = float(np.nanmax(vals)) if vals.size else 1.0
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(vals, dtype=float)
    return (vals - vmin) / (vmax - vmin)

def _value_to_rgb_triplets(vals: np.ndarray) -> np.ndarray:
    """
    Map scalar in [0,1] to an RGB triplet using a simple HSV ramp
    (blue→cyan→green→yellow→red). Returns shape (N,3) floats in [0,1].
    """
    v = np.clip(vals, 0.0, 1.0).astype(float)
    # hue ∈ [0.0, 0.66] (blue→red)
    hues = 0.66 * (1.0 - v)
    rgbs = np.empty((v.shape[0], 3), dtype=float)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        rgbs[i, 0] = r
        rgbs[i, 1] = g
        rgbs[i, 2] = b
    return rgbs

def _thin_indices(n: int, k: int) -> np.ndarray:
    """Return up to k indices from range(n) deterministically (head+stride)."""
    if n <= k:
        return np.arange(n, dtype=np.int64)
    stride = max(1, n // k)
    return np.arange(0, n, stride, dtype=np.int64)[:k]

# -----------------------------
# Scene
# -----------------------------
class FukaWorldEdges3D(ThreeDScene):
    """
    Reads NPZ built by fuka.render.pack_npz and animates points + edges per step.

    Expected arrays (flat, row-per-record):
      steps:           (S,)
      state_step:      (Ns,)
      state_x/y/z/v:   (Ns,)
      edges_step:      (Ne,)
      x0,y0,z0,x1,y1,z1,v0,v1: (Ne,)
    """

    def construct(self):
        npz_path = os.environ.get("FUKA_NPZ", "").strip()
        if not npz_path or not os.path.isfile(npz_path):
            raise FileNotFoundError(f"FUKA_NPZ not set or file not found: {npz_path}")

        step_seconds = _env_float("FUKA_STEP_SECONDS", 0.2)
        max_points   = _env_int("FUKA_MAX_POINTS", 80000)
        max_edges    = _env_int("FUKA_MAX_EDGES", 20000)

        point_radius = _env_float("FUKA_POINT_RADIUS", 0.035)
        edge_width   = _env_float("FUKA_EDGE_WIDTH", 2.5)
        show_edges   = _env_bool("FUKA_SHOW_EDGES", True)

        # Make camera a bit nicer
        self.set_camera_orientation(phi=65 * math.pi/180, theta=45 * math.pi/180)

        data = np.load(npz_path, allow_pickle=False)

        steps = data["steps"].astype(np.int64)
        # state
        s_step = data["state_step"].astype(np.int64)
        sx = data["state_x"].astype(float)
        sy = data["state_y"].astype(float)
        sz = data["state_z"].astype(float)
        sv = data["state_value"].astype(float)
        # edges
        e_step = data["edges_step"].astype(np.int64)
        x0 = data["x0"].astype(float); y0 = data["y0"].astype(float); z0 = data["z0"].astype(float)
        x1 = data["x1"].astype(float); y1 = data["y1"].astype(float); z1 = data["z1"].astype(float)
        v0 = data["v0"].astype(float); v1 = data["v1"].astype(float)

        # Global normalization for stable colors across frames
        svn = _normalize(sv)
        v0n = _normalize(v0)

        # Build per-step index lookups
        from collections import defaultdict
        idx_state = defaultdict(list)
        for i, st in enumerate(s_step):
            idx_state[int(st)].append(i)
        idx_edges = defaultdict(list)
        for i, st in enumerate(e_step):
            idx_edges[int(st)].append(i)

        # Main animation: for each step, draw a cloud + optional lines; replace each frame
        cur_group = VGroup()

        for st in steps:
            st = int(st)
            self.clear()         # ensure previous frame is cleared from the scene graph
            self.remove(cur_group)
            pts = VGroup()
            seg = VGroup()

            # ---- STATE POINTS ----
            sidx = np.array(idx_state.get(st, []), dtype=np.int64)
            if sidx.size:
                # downsample if needed
                sidx = _thin_indices(int(sidx.size), max_points).astype(np.int64)
                xs = sx[sidx]; ys = sy[sidx]; zs = sz[sidx]
                valn = svn[sidx]  # normalized 0..1
                cols = _value_to_rgb_triplets(valn)

                # Draw points
                for j in range(xs.shape[0]):
                    d = Dot3D(point=(float(xs[j]), float(ys[j]), float(zs[j])))
                    rgb = tuple(map(float, cols[j]))   # (r,g,b) floats
                    d.set_fill(rgb, opacity=1.0)       # use fill for Dot3D
                    d.set_stroke(rgb, width=0.0, opacity=0.0)
                    d.radius = point_radius
                    pts.add(d)

            # ---- EDGES ----
            if show_edges:
                eidx = np.array(idx_edges.get(st, []), dtype=np.int64)
                if eidx.size:
                    eidx = _thin_indices(int(eidx.size), max_edges).astype(np.int64)
                    x0s = x0[eidx]; y0s = y0[eidx]; z0s = z0[eidx]
                    x1s = x1[eidx]; y1s = y1[eidx]; z1s = z1[eidx]
                    wv  = v0n[eidx]  # weight/color based on src value
                    cols = _value_to_rgb_triplets(wv)
                    for j in range(x0s.shape[0]):
                        line = Line3D(
                            start=(float(x0s[j]), float(y0s[j]), float(z0s[j])),
                            end=(float(x1s[j]), float(y1s[j]), float(z1s[j])),
                        )
                        rgb = tuple(map(float, cols[j]))
                        line.set_stroke(rgb, width=edge_width)
                        seg.add(line)

            cur_group = VGroup(pts, seg)
            self.add(cur_group)

            # hold for step duration (convert seconds → Manim run_time)
            self.wait(step_seconds)

        # End
        self.wait(0.1)