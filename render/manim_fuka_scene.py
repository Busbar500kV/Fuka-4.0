# render/manim_fuka_scene.py
# Manim scene that reads an NPZ created by fuka.render.pack_npz
# Expected NPZ keys:
#   steps: (F,)
#   state_x/state_y/state_z/state_value: (N_state_total,)
#   state_idx: (F+1,)
#   edges_x0/edges_y0/edges_z0/edges_x1/edges_y1/edges_z1: (N_edge_total,)
#   edges_idx: (F+1,)
#
# Optional env knobs:
#   FUKA_NPZ=</abs/path/to.npz>
#   FUKA_FPS=6
#   FUKA_STEP_SECONDS=0.2
#   FUKA_MAX_POINTS=8000
#   FUKA_MAX_EDGES=8000
#   FUKA_POINT_RADIUS=0.04
#   FUKA_EDGE_WIDTH=2.0
#   FUKA_PAD=0.5

from __future__ import annotations
import os
import math
from typing import Tuple
import numpy as np

from manim import (
    ThreeDScene, VGroup, Dot3D, Line3D,
    ORIGIN, config
)

# ---------- helpers ----------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)

def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)

def _load_npz() -> Tuple[np.ndarray, dict]:
    npz_path = os.environ.get("FUKA_NPZ", "").strip()
    if not npz_path:
        raise RuntimeError("FUKA_NPZ not set (absolute path to packed NPZ).")
    data = np.load(npz_path, allow_pickle=False)
    required = [
        "steps",
        "state_x","state_y","state_z","state_value","state_idx",
        "edges_x0","edges_y0","edges_z0","edges_x1","edges_y1","edges_z1","edges_idx",
    ]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"NPZ missing required keys: {missing}. Found: {list(data.files)}")
    return data["steps"], {k: data[k] for k in required if k != "steps"}

def _value_to_rgb(vals: np.ndarray) -> np.ndarray:
    """Map scalar -> RGB in [0,1]. Simple two-stop gradient (green→yellow)."""
    if vals.size == 0:
        return np.zeros((0,3), dtype=float)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    t = (vals - vmin) / (vmax - vmin)
    rgb = np.empty((len(vals),3), dtype=float)
    # green (low) to yellow (high): (0,1,0) → (1,1,0)
    rgb[:, 0] = t                 # R: 0→1
    rgb[:, 1] = 1.0               # G: 1
    rgb[:, 2] = 0.0               # B: 0
    return np.clip(rgb, 0.0, 1.0)

def _downsample(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=int)
    stride = max(1, n // max_n)
    idx = np.arange(0, n, stride, dtype=int)
    if idx.size > max_n:
        idx = idx[:max_n]
    return idx

# ---------- Scene ----------

class FukaWorldEdges3D(ThreeDScene):
    def construct(self):
        # timing & visual knobs
        fps = _env_int("FUKA_FPS", 6)
        step_secs = _env_float("FUKA_STEP_SECONDS", 0.2)
        max_points = _env_int("FUKA_MAX_POINTS", 8000)
        max_edges  = _env_int("FUKA_MAX_EDGES", 8000)
        point_radius = _env_float("FUKA_POINT_RADIUS", 0.04)
        edge_width   = _env_float("FUKA_EDGE_WIDTH", 2.0)
        pad          = _env_float("FUKA_PAD", 0.5)

        config.frame_rate = fps

        # load packed arrays
        steps, D = _load_npz()
        sx, sy, sz = D["state_x"], D["state_y"], D["state_z"]
        sv = D["state_value"]; sidx = D["state_idx"]
        ex0, ey0, ez0 = D["edges_x0"], D["edges_y0"], D["edges_z0"]
        ex1, ey1, ez1 = D["edges_x1"], D["edges_y1"], D["edges_z1"]
        eidx = D["edges_idx"]

        # bounds & camera
        if sx.size:
            xmin, xmax = float(np.min(sx)), float(np.max(sx))
            ymin, ymax = float(np.min(sy)), float(np.max(sy))
            zmin, zmax = float(np.min(sz)), float(np.max(sz))
        else:
            xmin = ymin = zmin = -1.0
            xmax = ymax = zmax =  1.0
        xrange = xmax - xmin; yrange = ymax - ymin; zrange = zmax - zmin
        xmin -= pad * xrange; xmax += pad * xrange
        ymin -= pad * yrange; ymax += pad * yrange
        zmin -= pad * zrange; zmax += pad * zrange

        self.set_camera_orientation(phi=70*math.pi/180, theta=45*math.pi/180, zoom=1.0)
        self.camera.frame.move_to(((xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2))

        # animate frames
        F = int(steps.shape[0])
        for fi in range(F):
            # state points for this frame
            s_lo = int(sidx[fi]); s_hi = int(sidx[fi+1])
            n_s = max(0, s_hi - s_lo)
            pts_group = VGroup()
            if n_s > 0:
                si = _downsample(n_s, max_points) + s_lo
                xs, ys, zs = sx[si], sy[si], sz[si]
                cols = _value_to_rgb(sv[si])
                for j in range(xs.shape[0]):
                    d = Dot3D(point=(float(xs[j]), float(ys[j]), float(zs[j])), radius=point_radius)
                    rgb = (float(cols[j,0]), float(cols[j,1]), float(cols[j,2]))
                    d.set_fill(rgb, opacity=1.0).set_stroke(rgb, opacity=1.0, width=0.0)
                    pts_group.add(d)

            # edges for this frame
            e_lo = int(eidx[fi]); e_hi = int(eidx[fi+1])
            n_e = max(0, e_hi - e_lo)
            edges_group = VGroup()
            if n_e > 0:
                ei = _downsample(n_e, max_edges) + e_lo
                for j in ei:
                    p0 = (float(ex0[j]), float(ey0[j]), float(ez0[j]))
                    p1 = (float(ex1[j]), float(ey1[j]), float(ez1[j]))
                    seg = Line3D(p0, p1, stroke_width=edge_width, stroke_opacity=0.85)
                    edges_group.add(seg)

            # composite frame
            frame_group = VGroup(edges_group, pts_group)
            self.add(frame_group)
            self.wait(step_secs)
            self.remove(frame_group)