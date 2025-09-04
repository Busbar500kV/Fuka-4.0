# render/manim_fuka_scene.py
from __future__ import annotations
import os, math, json, re
from typing import Tuple, Dict, Any, Iterable
import numpy as np

from manim import (
    ThreeDScene, VGroup, Dot3D, Line3D, config,
)

# ========== ENV ==========
def _env_float(name: str, default: float) -> float:
    try: return float(os.environ.get(name, default))
    except Exception: return float(default)

def _env_int(name: str, default: int) -> int:
    try: return int(float(os.environ.get(name, default)))
    except Exception: return int(default)

def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default

NPZ_PATH     = _env_str("FUKA_NPZ", "")
FPS          = _env_int("FUKA_FPS", 12)
STEP_SECS    = _env_float("FUKA_STEP_SECONDS", 0.20)
MAX_POINTS   = _env_int("FUKA_MAX_POINTS", 80000)
MAX_EDGES    = _env_int("FUKA_MAX_EDGES", 60000)
EDGE_WIDTH   = _env_float("FUKA_EDGE_WIDTH", 3.0)
POINT_RADIUS = _env_float("FUKA_POINT_RADIUS", 0.05)
COLOR_MODE   = _env_str("FUKA_COLOR_MODE", "auto")  # auto|edge_strength|edge_deposit|edge_kappa|edge_value|point_value
COLOR_KEY    = _env_str("FUKA_COLOR_KEY", "")
POINTS_COLOR_KEY = _env_str("FUKA_POINTS_COLOR_KEY", "")
CAM_PHI      = math.radians(_env_float("FUKA_CAM_PHI", 65.0))
CAM_THETA    = math.radians(_env_float("FUKA_CAM_THETA", -45.0))
CAM_ZOOM     = _env_float("FUKA_CAM_ZOOM", 1.10)

# ========== COLOR MAP (viridis-ish) ==========
# 11-point viridis palette (hex to rgb 0..1)
_VIRIDIS = [
    (68,1,84),(72,35,116),(64,67,135),(52,94,141),(41,120,142),
    (32,144,140),(34,167,132),(68,190,112),(121,209,81),(189,223,38),(253,231,37)
]
_VIRIDIS = [tuple(c/255.0 for c in rgb) for rgb in _VIRIDIS]

def _interp_color01(x: float) -> Tuple[float,float,float]:
    x = max(0.0, min(1.0, float(x)))
    pos = x * (len(_VIRIDIS)-1)
    i = int(pos)
    if i >= len(_VIRIDIS)-1:
        return _VIRIDIS[-1]
    t = pos - i
    c0 = _VIRIDIS[i]; c1 = _VIRIDIS[i+1]
    return (c0[0]*(1-t)+c1[0]*t, c0[1]*(1-t)+c1[1]*t, c0[2]*(1-t)+c1[2]*t)

def _rgb_to_manim(c: Tuple[float,float,float]) -> str:
    # Manim accepts hex or rgb; we build hex
    r,g,b = (max(0,min(255,int(round(x*255)))) for x in c)
    return f"#{r:02x}{g:02x}{b:02x}"

# ========== NPZ HELPERS ==========
def _pick_key(keys: Iterable[str], want_order: list[str]) -> str | None:
    kset = {k.lower(): k for k in keys}
    # explicit
    for w in want_order:
        if w in kset: return kset[w]
    # fuzzy e* search
    for pref in ["e", "edge_"]:
        for w in ["strength","deposit","kappa","value","val","enc","weight","w","d"]:
            for k in keys:
                kl = k.lower()
                if kl.startswith(pref) and w in kl:
                    return k
    return None

def _normalize01(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return np.zeros_like(x, dtype=float)
    x = x.astype(float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)

def _build_indices(idx: np.ndarray, total: int) -> np.ndarray:
    """Ensure idx has one extra sentinel at the end."""
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.array([0, total], dtype=np.int64)
    if idx[-1] != total:
        idx = np.concatenate([idx, [total]])
    return idx

# ========== GEOM NORMALIZATION ==========
def _make_normalizer(points_xyz: np.ndarray, pad=0.5, target_half=3.0):
    if points_xyz.size == 0:
        ctr = np.array([0.0,0.0,0.0]); s = 1.0
    else:
        mins = points_xyz.min(axis=0); maxs = points_xyz.max(axis=0)
        span = float(np.max(maxs - mins)) or 1.0
        s = (target_half - pad) * 2.0 / span
        ctr = 0.5*(mins + maxs)

    def f(p: Tuple[float,float,float]) -> Tuple[float,float,float]:
        x,y,z = p
        x, y, z = (x-ctr[0])*s, (y-ctr[1])*s, (z-ctr[2])*s
        # map World(x,y,z) -> Manim(X=x, Y=z, Z=y)
        return (x, z, y)
    return f

# ========== SCENE ==========
class FukaScene(ThreeDScene):
    def construct(self):
        if not NPZ_PATH:
            self.add(*[])  # no-op
            return
        npz = np.load(NPZ_PATH)

        # ---- pull arrays (flexible naming) ----
        # Points per frame (optional)
        sx = npz.get("sx", np.zeros(0)); sy = npz.get("sy", np.zeros(0)); sz = npz.get("sz", np.zeros(0))
        sval = npz.get("sval", npz.get("s_value", npz.get("value", np.zeros(0))))
        sidx = _build_indices(npz.get("sidx", np.zeros(0)), sx.shape[0])

        # Edges per frame
        ex0 = npz.get("ex0", np.zeros(0)); ey0 = npz.get("ey0", np.zeros(0)); ez0 = npz.get("ez0", np.zeros(0))
        ex1 = npz.get("ex1", np.zeros(0)); ey1 = npz.get("ey1", np.zeros(0)); ez1 = npz.get("ez1", np.zeros(0))
        eidx = _build_indices(npz.get("eidx", np.zeros(0)), ex0.shape[0])

        # Optional edge coloring attributes — auto-detect unless explicit env given
        edge_keys = list(npz.keys())
        ecolor_key = COLOR_KEY or (
            _pick_key(edge_keys, [
                "edge_strength", "edge_deposit", "edge_kappa", "edge_value"
            ])
        )
        ecolor = npz.get(ecolor_key, None) if ecolor_key else None

        # Optional point coloring
        pcolor_key = POINTS_COLOR_KEY or ("sval" if "sval" in npz else None)
        pcolor = npz.get(pcolor_key, None) if pcolor_key else None

        # ---- compute normalization from all coords we'll render ----
        all_pts = []
        if sx.size: all_pts.append(np.c_[sx,sy,sz])
        if ex0.size: all_pts.append(np.c_[ex0,ey0,ez0])
        if ex1.size: all_pts.append(np.c_[ex1,ey1,ez1])
        if all_pts:
            all_pts = np.vstack(all_pts)
        else:
            all_pts = np.zeros((1,3))
        norm = _make_normalizer(all_pts, pad=0.5, target_half=3.0)

        # ---- camera ----
        self.set_camera_orientation(phi=CAM_PHI, theta=CAM_THETA)  # OpenGLCamera has no set_zoom()
        if abs(CAM_ZOOM - 1.0) > 1e-6:
            # Zoom in/out by scaling the view frame; >1.0 means "closer"
            try:
                self.renderer.camera.frame.scale(1.0 / CAM_ZOOM)
            except Exception:
                pass  # stay robust across manim variants

        # Optional bounds cube/axes (comment out if undesired)
        # from manim import Cube, ThreeDAxes
        # self.add(Cube(side_length=6.0, stroke_opacity=0.15, fill_opacity=0.0))
        # self.add(ThreeDAxes(x_range=[-3,3,1], y_range=[-3,3,1], z_range=[-3,3,1],
        #                     x_length=6, y_length=6, z_length=6,
        #                     axis_config={"stroke_opacity": 0.18, "stroke_width": 1}))

        # ---- helpers ----
        def downsample(n: int, k: int) -> np.ndarray:
            if n <= k: return np.arange(n, dtype=np.int64)
            # even reservoir style: pick approximately uniform indices
            return (np.linspace(0, n-1, num=k)).astype(np.int64)

        # pick color scaler
        def make_color_scaler(vec: np.ndarray | None):
            if vec is None or vec.size == 0:
                return lambda _: _rgb_to_manim(_interp_color01(0.0))
            v = vec.astype(float)
            v01 = _normalize01(v)
            def f(i: int) -> str:
                return _rgb_to_manim(_interp_color01(float(v01[i])))
            return f

        # edge color rule
        edge_color_by = (COLOR_MODE if COLOR_MODE != "auto" else
                         ("point_value" if COLOR_MODE == "point_value" else "edge"))
        edge_cfunc = make_color_scaler(ecolor)  # default

        # point color rule
        point_cfunc = make_color_scaler(pcolor)

        # ---- persistent groups ----
        g = VGroup()
        self.add(g)

        F = int(max(len(sidx), len(eidx)) - 1)
        for fi in range(F):
            # ----- points -----
            pts_group = VGroup()
            slo, shi = int(sidx[fi]), int(sidx[fi+1])
            if shi > slo and sx.size:
                pick = downsample(shi - slo, MAX_POINTS) + slo
                for j in pick:
                    X = norm((float(sx[j]), float(sy[j]), float(sz[j])))
                    dot = Dot3D(point=X, radius=POINT_RADIUS, stroke_width=0, fill_opacity=0.95)
                    dot.set_color(point_cfunc(j))
                    pts_group.add(dot)

            # ----- edges -----
            edges_group = VGroup()
            elo, ehi = int(eidx[fi]), int(eidx[fi+1])
            if ehi > elo and ex0.size:
                pick = downsample(ehi - elo, MAX_EDGES) + elo
                for j in pick:
                    p0 = norm((float(ex0[j]), float(ey0[j]), float(ez0[j])))
                    p1 = norm((float(ex1[j]), float(ey1[j]), float(ez1[j])))
                    seg = Line3D(p0, p1, stroke_width=EDGE_WIDTH, stroke_opacity=0.95)
                    seg.set_color(edge_cfunc(j))
                    edges_group.add(seg)

            new_frame = VGroup(edges_group, pts_group)
            g.become(new_frame)
            self.wait(STEP_SECS)
