# render/manim_fuka_scene.py
from __future__ import annotations
"""
Fuka 4.0 — Manim scene (OpenGL) for 3D points + edges.

Drop-in replacement focused on:
- Robust camera (no OpenGL .set_zoom dependency; uses attribute fallback)
- Env-driven camera + sizing:
    FUKA_CAM_PHI, FUKA_CAM_THETA, FUKA_CAM_ZOOM
    FUKA_FPS, FUKA_STEP_SECONDS, FUKA_POINT_RADIUS, FUKA_EDGE_WIDTH
    FUKA_MAX_POINTS, FUKA_MAX_EDGES
- Env-driven colour selection:
    FUKA_COLOR_KEY          -> edges colour field
    FUKA_POINTS_COLOR_KEY   -> points colour field (defaults to state_value)
- Optional debug axes/cube: FUKA_DEBUG_AXES=1
- Safe per-frame update (add/remove groups; avoids .become() crash on OpenGL 0.19)

NPZ requirements (packed by prep_fuka_npz / pack_npz):
  steps (F,)
  state_x/state_y/state_z/state_value (Ns,)
  state_idx (F+1,)
  edges_x0/edges_y0/edges_z0/edges_x1/edges_y1/edges_z1 (Ne,)
  edges_idx (F+1,)

Optional NPZ keys used for edge colouring if present:
  edge_deposit, edge_strength, edge_value, edge_kappa
"""

import math, os
from typing import Tuple, Dict
import numpy as np
from manim import ThreeDScene, VGroup, Dot3D, Line3D, config

# ---------- env helpers ----------
def _env_float(name: str, default: float) -> float:
    try: return float(os.environ.get(name, default))
    except Exception: return float(default)

def _env_int(name: str, default: int) -> int:
    try: return int(float(os.environ.get(name, default)))
    except Exception: return int(default)

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or str(v).strip()=="" else str(v).strip()

# ---------- io ----------
def _load_npz() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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
    D = {k: data[k] for k in required if k != "steps"}
    # Optional edge fields for colouring
    for k in ("edge_deposit","edge_strength","edge_value","edge_kappa"):
        if k in data.files:
            D[k] = data[k]
    return data["steps"], D

# ---------- colour + helpers ----------
def _value_to_rgb(vals: np.ndarray) -> np.ndarray:
    """Return Nx3 float array in [0,1] using a simple green→yellow ramp."""
    if vals.size == 0:
        return np.zeros((0,3), dtype=float)
    v = vals.astype(float)
    vmin = float(np.nanmin(v)); vmax = float(np.nanmax(v))
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    t = (v - vmin) / (vmax - vmin)
    rgb = np.empty((len(v),3), dtype=float)
    rgb[:,0] = t          # R grows with value
    rgb[:,1] = 1.0        # full green
    rgb[:,2] = 0.0        # zero blue
    return np.clip(rgb, 0.0, 1.0)

def _rgb01_to_hex(r: float, g: float, b: float) -> str:
    R = max(0, min(255, int(round(r*255))))
    G = max(0, min(255, int(round(g*255))))
    B = max(0, min(255, int(round(b*255))))
    return f"#{R:02X}{G:02X}{B:02X}"

def _downsample(n: int, max_n: int) -> np.ndarray:
    if n <= max_n: return np.arange(n, dtype=int)
    stride = max(1, n // max_n)
    idx = np.arange(0, n, stride, dtype=int)
    if idx.size > max_n:
        idx = idx[:max_n]
    return idx

def _edge_color_values(D: Dict[str, np.ndarray], key: str) -> np.ndarray:
    """Return per-edge scalar array matching edges_x0 length for colour mapping."""
    N = D["edges_x0"].shape[0]
    # Preferred explicit fields, if present
    for k in (key, "edge_deposit","edge_strength","edge_value","edge_kappa"):
        if k in D:
            v = D[k].astype(float)
            if v.shape[0] == N:
                return v
    # Fallback: segment length as a proxy
    dx = (D["edges_x1"] - D["edges_x0"]).astype(float)
    dy = (D["edges_y1"] - D["edges_y0"]).astype(float)
    dz = (D["edges_z1"] - D["edges_z0"]).astype(float)
    return np.sqrt(dx*dx + dy*dy + dz*dz)

# ---------- scene ----------
class FukaWorldEdges3D(ThreeDScene):
    def construct(self):
        # Controls
        fps          = _env_int("FUKA_FPS", 12)
        step_secs    = _env_float("FUKA_STEP_SECONDS", 0.20)
        max_points   = _env_int("FUKA_MAX_POINTS", 8000)
        max_edges    = _env_int("FUKA_MAX_EDGES", 8000)
        point_radius = _env_float("FUKA_POINT_RADIUS", 0.04)
        edge_width   = _env_float("FUKA_EDGE_WIDTH", 2.0)
        debug_axes   = _env_int("FUKA_DEBUG_AXES", 0)

        # Colour keys
        points_key = _env_str("FUKA_POINTS_COLOR_KEY", "state_value")  # (only 'state_value' exists in current NPZ)
        edges_key  = _env_str("FUKA_COLOR_KEY", "edge_deposit")        # try explicit; fall back internally

        # Camera
        cam_phi_deg   = _env_float("FUKA_CAM_PHI",   70.0)
        cam_theta_deg = _env_float("FUKA_CAM_THETA", 45.0)
        cam_zoom      = _env_float("FUKA_CAM_ZOOM",   1.0)

        config.frame_rate = fps

        steps, D = _load_npz()
        sx, sy, sz = D["state_x"], D["state_y"], D["state_z"]
        sv, sidx   = D["state_value"], D["state_idx"]
        ex0, ey0, ez0 = D["edges_x0"], D["edges_y0"], D["edges_z0"]
        ex1, ey1, ez1 = D["edges_x1"], D["edges_y1"], D["edges_z1"]
        eidx = D["edges_idx"]

        # bounds + recenter (avoid frame transforms on 0.19 OpenGL)
        if sx.size:
            xmin, xmax = float(sx.min()), float(sx.max())
            ymin, ymax = float(sy.min()), float(sy.max())
            zmin, zmax = float(sz.min()), float(sz.max())
        else:
            xmin = ymin = zmin = -1.0; xmax = ymax = zmax = 1.0
        cx, cy, cz = (xmin+xmax)/2.0, (ymin+ymax)/2.0, (zmin+zmax)/2.0

        # Camera: avoid .set_zoom() on OpenGL; use attribute fallback
        self.set_camera_orientation(phi=math.radians(cam_phi_deg),
                                    theta=math.radians(cam_theta_deg))
        try:
            cam = self.renderer.camera  # OpenGLCamera
            if hasattr(cam, "set_zoom"):
                cam.set_zoom(cam_zoom)           # Cairo style
            elif hasattr(cam, "zoom"):
                cam.zoom = float(cam_zoom)       # OpenGL attribute
        except Exception:
            pass  # keep defaults if anything goes wrong

        # Optional guides
        if debug_axes:
            try:
                from manim import ThreeDAxes, Cube
                self.add(ThreeDAxes(
                    x_range=[-3,3,1], y_range=[-3,3,1], z_range=[-3,3,1],
                    x_length=6, y_length=6, z_length=6,
                    axis_config={"stroke_opacity": 0.18, "stroke_width": 1},
                ))
                self.add(Cube(side_length=6.0, stroke_opacity=0.15, fill_opacity=0.0))
            except Exception:
                pass

        # Precompute colours
        # Points: only 'state_value' exists; keep hook for future fields.
        if points_key != "state_value":
            pvals = sv
        else:
            pvals = sv
        # Edges: choose available field; else length proxy
        evals = _edge_color_values(D, edges_key)

        F = int(steps.shape[0])
        for fi in range(F):
            # ----- points -----
            s_lo = int(sidx[fi]); s_hi = int(sidx[fi+1]); n_s = max(0, s_hi-s_lo)
            pts_group = VGroup()
            if n_s:
                si = _downsample(n_s, max_points) + s_lo
                xs, ys, zs = sx[si]-cx, sy[si]-cy, sz[si]-cz
                cols = _value_to_rgb(pvals[si])
                for j in range(len(si)):
                    hexc = _rgb01_to_hex(float(cols[j,0]), float(cols[j,1]), float(cols[j,2]))
                    d = Dot3D(point=(float(xs[j]), float(ys[j]), float(zs[j])),
                              radius=point_radius)
                    d.set_fill(color=hexc, opacity=1.0).set_stroke(color=hexc, opacity=1.0, width=0.0)
                    pts_group.add(d)

            # ----- edges -----
            e_lo = int(eidx[fi]); e_hi = int(eidx[fi+1]); n_e = max(0, e_hi-e_lo)
            edges_group = VGroup()
            if n_e:
                ei = _downsample(n_e, max_edges) + e_lo
                ecol = _value_to_rgb(evals[ei])
                for jj, j in enumerate(ei):
                    p0 = (float(ex0[j]-cx), float(ey0[j]-cy), float(ez0[j]-cz))
                    p1 = (float(ex1[j]-cx), float(ey1[j]-cy), float(ez1[j]-cz))
                    seg = Line3D(p0, p1, stroke_width=edge_width, stroke_opacity=0.95)
                    hexc = _rgb01_to_hex(float(ecol[jj,0]), float(ecol[jj,1]), float(ecol[jj,2]))
                    seg.set_stroke(color=hexc)
                    edges_group.add(seg)

            # Compose and show frame
            frame_group = VGroup(edges_group, pts_group)
            self.add(frame_group)
            self.wait(step_secs)
            self.remove(frame_group)
