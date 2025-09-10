# render/manim_fuka_scene.py
from __future__ import annotations
"""
ManimCE 3D scene for Fuka cumulative edge renders.

Reads a single NPZ (either the final roll-up or any per-step cumulative file)
with canonical keys:
  steps (F,)
  edges_x0..z1 (Ne,)
  edges_idx (F+1,)
  edge_value (Ne,)
  edge_strength (Ne,)

Environment knobs (optional):
  FUKA_COLOR_KEY       : one of ["edge_value","edge_strength"] (default: "edge_strength")
  FUKA_POINTS_COLOR_KEY: currently unused placeholder for future node/voxel coloring
  FUKA_MAX_EDGES       : hard cap on edges drawn per frame (default: 8000)
  FUKA_CAM_PHI         : camera polar angle deg (default: 65)
  FUKA_CAM_THETA       : camera azimuth deg (default: -45)
  FUKA_CAM_ZOOM        : zoom factor (default: 1.1)
  FUKA_DEBUG_AXES      : "1" to show 3D axes (default: off)
  FUKA_WIRE_ALPHA      : 0..1 opacity for edges (default: 0.9)
  FUKA_STRENGTH_GAMMA  : gamma for color/width mapping (default: 0.6)
  FUKA_WIDTH_MIN       : minimum stroke width (default: 1.2)
  FUKA_WIDTH_MAX       : maximum stroke width (default: 4.0)
  FUKA_CMAP            : "magma|viridis|plasma|inferno|turbo" (default: "magma")
  FUKA_STEP_MIN        : first frame index within npz to render (default: 0)
  FUKA_STEP_MAX        : last frame index within npz to render (default: last)
  FUKA_FPS_HINT        : metadata only; actual fps set via manim CLI

Usage example:
  manim -pqh render/manim_fuka_scene.py Fuka3DScene \
    --config_file=render/manim.cfg \
    -s  # or render animation with -qk/--fps via manim CLI

Note: Manim controls output fps; pass --fps and -q flags there.
"""

import json
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from manim import (
    ThreeDScene, VGroup, Line3D, ORIGIN, config, Axes3D, BLUE, YELLOW, RED
)

# -------------------------- utilities --------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except Exception:
        return default

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default

def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _gamma_map(x: np.ndarray, gamma: float) -> np.ndarray:
    x = _clamp01(x)
    g = max(1e-6, float(gamma))
    return x ** g

def _normalize(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    mn = float(np.min(v))
    mx = float(np.max(v))
    if not math.isfinite(mn) or not math.isfinite(mx) or mx <= mn:
        return np.zeros_like(v, dtype=float)
    return (v - mn) / (mx - mn)

# Simple colormap (no external deps); returns RGB triplets 0..1
def _cmap(name: str, x01: np.ndarray) -> np.ndarray:
    name = (name or "magma").lower()
    # Small procedural palettes (approximate)
    if name == "viridis":
        r = 0.267 + 2.227*x01 - 1.597*(x01**2)
        g = 0.005 + 3.134*x01 - 2.412*(x01**2)
        b = 0.329 + 1.579*x01 - 0.927*(x01**2)
    elif name == "plasma":
        r = 0.050 + 2.545*x01 - 1.476*(x01**2)
        g = 0.031 + 0.888*x01 + 0.102*(x01**2)
        b = 0.527 + 0.535*x01 - 0.053*(x01**2)
    elif name == "inferno":
        r = 0.001 + 2.433*x01 - 1.520*(x01**2)
        g = 0.000 + 0.970*x01 + 0.299*(x01**2)
        b = 0.014 + 0.153*x01 + 0.430*(x01**2)
    elif name == "turbo":
        # Fast approximate turbo-like
        r = _clamp01(1.0 - np.abs(2*x01 - 1.0))
        g = _clamp01(1.0 - np.abs(2*x01 - 0.5))
        b = _clamp01(1.0 - np.abs(2*x01))
    else:  # magma default
        r = 0.001 + 2.235*x01 - 1.272*(x01**2)
        g = 0.000 + 0.680*x01 + 0.270*(x01**2)
        b = 0.012 + 0.160*x01 + 0.530*(x01**2)
    rgb = np.stack([_clamp01(r), _clamp01(g), _clamp01(b)], axis=1)
    return rgb

def _choose_color_key(edge_value: np.ndarray, edge_strength: np.ndarray) -> np.ndarray:
    key = _env_str("FUKA_COLOR_KEY", "edge_strength").lower()
    if key == "edge_value":
        return edge_value
    # default
    return edge_strength

def _stroke_widths(strength01: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    return (wmin + (wmax - wmin) * strength01).astype(float)

def _load_npz() -> Tuple[dict, np.lib.npyio.NpzFile]:
    # Path to NPZ chosen by Manim config's media_dir or via argument?
    # We adopt a simple convention: look under config.assets_dir first, else CWD / assets
    candidates = []
    if config.assets_dir:
        candidates.append(Path(config.assets_dir) / "fuka_anim.npz")
        candidates.append(Path(config.assets_dir) / "edges_all.npz")
    candidates += [Path("assets/fuka_anim.npz"), Path("assets/edges_all.npz")]
    candidates += sorted(Path(".").glob("edges_step_*.npz"))[::-1]  # prefer latest
    for p in candidates:
        if p.exists():
            return {"path": str(p)}, np.load(str(p))
    raise FileNotFoundError(
        "Could not find NPZ. Place it at assets/fuka_anim.npz or assets/edges_all.npz or provide edges_step_*.npz in CWD."
    )

# -------------------------- scene --------------------------

class Fuka3DScene(ThreeDScene):
    def construct(self) -> None:
        meta, npz = _load_npz()

        steps = npz["steps"].astype(np.int64)
        x0 = npz["edges_x0"].astype(float)
        y0 = npz["edges_y0"].astype(float)
        z0 = npz["edges_z0"].astype(float)
        x1 = npz["edges_x1"].astype(float)
        y1 = npz["edges_y1"].astype(float)
        z1 = npz["edges_z1"].astype(float)
        eidx = npz["edges_idx"].astype(np.int64)
        ev  = npz.get("edge_value", np.zeros_like(x0))
        es  = npz.get("edge_strength", np.abs(ev))  # fallback: |value|

        # Camera & styling from env
        cam_phi   = math.radians(_env_float("FUKA_CAM_PHI", 65))
        cam_theta = math.radians(_env_float("FUKA_CAM_THETA", -45))
        cam_zoom  = _env_float("FUKA_CAM_ZOOM", 1.1)
        debug_axes= _env_int("FUKA_DEBUG_AXES", 0) == 1

        alpha     = float(np.clip(_env_float("FUKA_WIRE_ALPHA", 0.9), 0.0, 1.0))
        gamma     = float(max(1e-6, _env_float("FUKA_STRENGTH_GAMMA", 0.6)))
        wmin      = _env_float("FUKA_WIDTH_MIN", 1.2)
        wmax      = _env_float("FUKA_WIDTH_MAX", 4.0)
        cmap_name = _env_str("FUKA_CMAP", "magma")

        step_min_env = _env_int("FUKA_STEP_MIN", 0)
        step_max_env = _env_int("FUKA_STEP_MAX", len(steps)-1)
        step_min = max(0, min(step_min_env, len(steps)-1))
        step_max = max(step_min, min(step_max_env, len(steps)-1))

        hard_cap = _env_int("FUKA_MAX_EDGES", 8000)

        # Normalize color key across full dataset for consistency
        color_key_vec = _choose_color_key(ev, es)
        color01_full = _normalize(color_key_vec)
        color01_full = _gamma_map(color01_full, gamma)
        palette = _cmap(cmap_name, color01_full)

        # Set up 3D camera
        self.set_camera_orientation(phi=cam_phi, theta=cam_theta)
        self.camera.frame.set_zoom(cam_zoom)

        # Optional axes box
        if debug_axes:
            # auto-scale axes based on max coord
            maxx = max(np.max(x0), np.max(x1)) if x0.size else 1.0
            maxy = max(np.max(y0), np.max(y1)) if y0.size else 1.0
            maxz = max(np.max(z0), np.max(z1)) if z0.size else 1.0
            axes = Axes3D(
                x_range=[0, maxx, max(1, math.ceil(maxx/5))],
                y_range=[0, maxy, max(1, math.ceil(maxy/5))],
                z_range=[0, maxz, max(1, math.ceil(maxz/5))],
                axis_config={"stroke_width": 1.5},
                x_length=6, y_length=6, z_length=6,
            )
            self.add(axes)

        # Render frames from step_min..step_max (inclusive)
        for fidx in range(step_min, step_max + 1):
            start = int(eidx[0]) if fidx == 0 else int(eidx[fidx])
            stop  = int(eidx[fidx + 1])
            n = max(0, stop - start)

            group = VGroup()

            if n > 0:
                # hard cap per frame for performance
                if n > hard_cap:
                    start = stop - hard_cap
                    n = hard_cap

                # slice views
                sl = slice(start, stop)
                x0f, y0f, z0f = x0[sl], y0[sl], z0[sl]
                x1f, y1f, z1f = x1[sl], y1[sl], z1[sl]
                cols = palette[sl]
                # widths mapped from strength globally (so same look over time)
                s01 = _normalize(es)[sl]
                s01 = _gamma_map(s01, gamma)
                widths = _stroke_widths(s01, wmin, wmax)

                # Centering transform (optional): shift to origin by mean
                cx = np.mean(np.concatenate([x0f, x1f])) if x0f.size else 0.0
                cy = np.mean(np.concatenate([y0f, y1f])) if y0f.size else 0.0
                cz = np.mean(np.concatenate([z0f, z1f])) if z0f.size else 0.0

                for i in range(n):
                    p0 = (x0f[i] - cx, y0f[i] - cy, z0f[i] - cz)
                    p1 = (x1f[i] - cx, y1f[i] - cy, z1f[i] - cz)
                    r, g, b = cols[i]
                    # Manim expects opacity 0..1 and color tuple as a hex or Color; we use a Color-like tuple via rgb_to_color
                    color = (r, g, b)
                    seg = Line3D(
                        start=p0,
                        end=p1,
                        color=color,
                        stroke_width=float(widths[i]),
                        opacity=alpha,
                    )
                    group.add(seg)

            # Add the current frame’s group and hold a very short pause (1 frame)
            self.add(group)
            self.wait(1 / max(1, config.frame_rate))
            self.remove(group)

        # end for
        # Write a tiny render meta alongside video (useful for debugging)
        try:
            out_dir = Path(config.media_dir or ".")
            (out_dir / "fuka_render_meta.json").write_text(
                json.dumps({"npz": meta["path"], "frames": int(step_max - step_min + 1)}, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass