# render/manim_fuka_scene.py
from __future__ import annotations
import os, math, colorsys
from typing import Sequence, Iterable, Dict
import numpy as np
from manim import ThreeDScene, Dot3D, Line3D, VGroup

# ---------- env helpers ----------
def _env_float(name: str, default: float) -> float:
    try: return float(os.environ.get(name, default))
    except Exception: return float(default)

def _env_int(name: str, default: int) -> int:
    try: return int(os.environ.get(name, default))
    except Exception: return int(default)

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name); 
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","y","on")

# ---------- array helpers ----------
def _normalize(vals: np.ndarray) -> np.ndarray:
    if vals.size == 0: return np.zeros_like(vals, dtype=float)
    vmin = float(np.nanmin(vals)); vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(vals, dtype=float)
    out = (vals - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)

def _value_to_rgb_triplets(vals: np.ndarray) -> np.ndarray:
    """Scalar in [0,1] -> RGB (blue→red) via HSV; returns (N,3) floats."""
    v = np.clip(vals.astype(float), 0.0, 1.0)
    hues = 0.66 * (1.0 - v)
    rgbs = np.empty((v.shape[0], 3), dtype=float)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        rgbs[i] = (r, g, b)
    return rgbs

def _thin_indices(n: int, k: int) -> np.ndarray:
    if n <= k: return np.arange(n, dtype=np.int64)
    stride = max(1, n // k)
    return np.arange(0, n, stride, dtype=np.int64)[:k]

def _pick(npz: Dict[str, np.ndarray], *names: str, required: bool = True) -> np.ndarray:
    """Pick the first existing array from a list of names, else raise with debug info."""
    for n in names:
        if n in npz: return npz[n]
    if required:
        raise KeyError(f"None of {names} found in NPZ. Available keys: {list(npz.files)}")
    return np.array([])

# ---------- Scene ----------
class FukaWorldEdges3D(ThreeDScene):
    """
    Robust loader for NPZ produced by pack_npz, tolerating different key names.
    Expected any of the following (first match wins):

      steps:            'steps' | 'frame_steps' | 'unique_steps'

      state step:       'state_step' | 'state_steps' | 's_step'
      state x:          'state_x' | 'sx' | 'x'
      state y:          'state_y' | 'sy' | 'y'
      state z:          'state_z' | 'sz' | 'z'
      state value:      'state_value' | 'value' | 'm' | 'val'

      edges step:       'edges_step' | 'edge_step' | 'e_step'
      edges x0..z1:     'x0','y0','z0','x1','y1','z1' (or legacy 'xs','ys','zs' + dst coords not supported)

      Optional weights: 'v0','v1' (falls back to zeros if absent)
    """
    def construct(self):
        npz_path = os.environ.get("FUKA_NPZ", "").strip()
        if not npz_path or not os.path.isfile(npz_path):
            raise FileNotFoundError(f"FUKA_NPZ not set or file not found: {npz_path}")

        step_seconds = _env_float("FUKA_STEP_SECONDS", 0.20)
        max_points   = _env_int("FUKA_MAX_POINTS",   80000)
        max_edges    = _env_int("FUKA_MAX_EDGES",    20000)
        point_radius = _env_float("FUKA_POINT_RADIUS", 0.035)
        edge_width   = _env_float("FUKA_EDGE_WIDTH",   2.5)
        show_edges   = _env_bool ("FUKA_SHOW_EDGES",   True)

        # Camera
        self.set_camera_orientation(phi=65*math.pi/180, theta=45*math.pi/180)

        data = np.load(npz_path, allow_pickle=False)

        # steps (global ordered)
        steps = _pick(data, "steps", "frame_steps", "unique_steps").astype(np.int64)

        # state
        s_step = _pick(data, "state_step", "state_steps", "s_step").astype(np.int64)
        sx     = _pick(data, "state_x", "sx", "x").astype(float)
        sy     = _pick(data, "state_y", "sy", "y").astype(float)
        sz     = _pick(data, "state_z", "sz", "z").astype(float)
        # value can be named a few ways
        sv     = _pick(data, "state_value", "value", "m", "val").astype(float)

        # edges
        e_step = _pick(data, "edges_step", "edge_step", "e_step").astype(np.int64)
        x0 = _pick(data, "x0").astype(float)
        y0 = _pick(data, "y0").astype(float)
        z0 = _pick(data, "z0").astype(float)
        x1 = _pick(data, "x1").astype(float)
        y1 = _pick(data, "y1").astype(float)
        z1 = _pick(data, "z1").astype(float)
        v0 = _pick(data, "v0", required=False).astype(float) if "v0" in data else np.zeros_like(x0)
        # v1 optional, unused for color; keep shape check
        _ =  _pick(data, "v1", required=False)

        # global normalization for consistent colors across frames
        svn = _normalize(sv)
        v0n = _normalize(v0)

        # per-step indices
        from collections import defaultdict
        idx_state = defaultdict(list)
        for i, st in enumerate(s_step): idx_state[int(st)].append(i)
        idx_edges = defaultdict(list)
        for i, st in enumerate(e_step): idx_edges[int(st)].append(i)

        cur = VGroup()

        for st in steps:
            st = int(st)
            # clear previous frame content
            self.clear()
            self.remove(cur)

            pts = VGroup(); seg = VGroup()

            # points
            sidx = np.array(idx_state.get(st, []), dtype=np.int64)
            if sidx.size:
                sidx = _thin_indices(int(sidx.size), max_points)
                xs, ys, zs = sx[sidx], sy[sidx], sz[sidx]
                cols = _value_to_rgb_triplets(svn[sidx])
                for j in range(xs.shape[0]):
                    d = Dot3D(point=(float(xs[j]), float(ys[j]), float(zs[j])))
                    rgb = tuple(map(float, cols[j]))
                    d.set_fill(rgb, opacity=1.0)
                    d.set_stroke(rgb, width=0.0, opacity=0.0)
                    d.radius = point_radius
                    pts.add(d)

            # edges
            if show_edges:
                eidx = np.array(idx_edges.get(st, []), dtype=np.int64)
                if eidx.size:
                    eidx = _thin_indices(int(eidx.size), max_edges)
                    x0s, y0s, z0s = x0[eidx], y0[eidx], z0[eidx]
                    x1s, y1s, z1s = x1[eidx], y1[eidx], z1[eidx]
                    cols = _value_to_rgb_triplets(v0n[eidx])
                    for j in range(x0s.shape[0]):
                        line = Line3D(
                            start=(float(x0s[j]), float(y0s[j]), float(z0s[j])),
                            end  =(float(x1s[j]), float(y1s[j]), float(z1s[j]))
                        )
                        rgb = tuple(map(float, cols[j]))
                        line.set_stroke(rgb, width=edge_width)
                        seg.add(line)

            cur = VGroup(pts, seg)
            self.add(cur)
            self.wait(step_seconds)

        self.wait(0.1)