# render/manim_fuka_scene.py
from __future__ import annotations
import os, math
from typing import Tuple, Iterable
import numpy as np

from manim import (
    ThreeDScene, VGroup, Dot3D, Line3D, config,
)

# ---------- ENV ----------
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
COLOR_KEY    = _env_str("FUKA_COLOR_KEY", "")            # explicit edge color key (optional)
POINTS_COLOR_KEY = _env_str("FUKA_POINTS_COLOR_KEY", "") # explicit point color key (optional)
CAM_PHI      = math.radians(_env_float("FUKA_CAM_PHI", 65.0))
CAM_THETA    = math.radians(_env_float("FUKA_CAM_THETA", -45.0))
CAM_ZOOM     = _env_float("FUKA_CAM_ZOOM", 1.10)
DEBUG_AXES   = _env_int("FUKA_DEBUG_AXES", 0)

# ---------- Color map (viridis-ish) ----------
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

def _rgb_to_hex(c: Tuple[float,float,float]) -> str:
    r,g,b = (max(0,min(255,int(round(x*255)))) for x in c)
    return f"#{r:02x}{g:02x}{b:02x}"

def _normalize01(x: np.ndarray) -> np.ndarray:
    if x is None or x.size == 0:
        return np.zeros(0, dtype=float)
    x = x.astype(float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)

def _build_indices(idx: np.ndarray, total: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.array([0, total], dtype=np.int64)
    if idx[-1] != total:
        idx = np.concatenate([idx, [total]])
    return idx

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
        # World(x,y,z) -> Manim(X=x, Y=z, Z=y)
        return (x, z, y)
    return f

def _pick_edge_color_key(keys: Iterable[str]) -> str | None:
    # try explicit COLOR_KEY first
    if COLOR_KEY:
        return COLOR_KEY if COLOR_KEY in keys else None
    pri = ["edge_strength","edge_deposit","edge_kappa","edge_value"]
    for k in pri:
        if k in keys: return k
    return None  # fall back to state_value-based nearest sampling

class FukaScene(ThreeDScene):
    def construct(self):
        if not NPZ_PATH:
            return
        npz = np.load(NPZ_PATH)

        # ---- tolerate both naming schemes ----
        # Points (state)
        sx = npz.get("sx", npz.get("state_x"))
        sy = npz.get("sy", npz.get("state_y"))
        sz = npz.get("sz", npz.get("state_z"))
        sidx = npz.get("sidx", npz.get("state_idx"))
        sval = (npz.get("sval", npz.get("s_value", npz.get("value")))
                if "sval" in npz or "s_value" in npz or "value" in npz else npz.get("state_value"))

        # Edges
        ex0 = npz.get("ex0", npz.get("edges_x0"))
        ey0 = npz.get("ey0", npz.get("edges_y0"))
        ez0 = npz.get("ez0", npz.get("edges_z0"))
        ex1 = npz.get("ex1", npz.get("edges_x1"))
        ey1 = npz.get("ey1", npz.get("edges_y1"))
        ez1 = npz.get("ez1", npz.get("edges_z1"))
        eidx = npz.get("eidx", npz.get("edges_idx"))

        # indices must be present even if empty
        sidx = _build_indices(sidx if sidx is not None else np.zeros(0), (sx.shape[0] if isinstance(sx, np.ndarray) else 0))
        eidx = _build_indices(eidx if eidx is not None else np.zeros(0), (ex0.shape[0] if isinstance(ex0, np.ndarray) else 0))

        # ---- choose edge color data if provided; else we'll derive from state_value ----
        ecolor_key = _pick_edge_color_key(npz.keys())
        ecolor = npz.get(ecolor_key, None) if ecolor_key else None  # may be None

        # ---- gather coords for normalization ----
        stacks = []
        for arrs in ((sx,sy,sz), (ex0,ey0,ez0), (ex1,ey1,ez1)):
            if isinstance(arrs[0], np.ndarray):
                stacks.append(np.c_[arrs[0], arrs[1], arrs[2]])
        all_pts = np.vstack(stacks) if stacks else np.zeros((1,3))
        norm = _make_normalizer(all_pts, pad=0.5, target_half=3.0)

        # ---- camera ----
        self.set_camera_orientation(phi=CAM_PHI, theta=CAM_THETA)  # OpenGLCamera has no set_zoom()
        if abs(CAM_ZOOM - 1.0) > 1e-6:
            try:
                self.renderer.camera.frame.scale(1.0 / CAM_ZOOM)
            except Exception:
                pass

        # Optional debug cube/axes
        if DEBUG_AXES:
            from manim import Cube, ThreeDAxes
            self.add(Cube(side_length=6.0, stroke_opacity=0.15, fill_opacity=0.0))
            self.add(ThreeDAxes(x_range=[-3,3,1], y_range=[-3,3,1], z_range=[-3,3,1],
                                x_length=6, y_length=6, z_length=6,
                                axis_config={"stroke_opacity": 0.18, "stroke_width": 1}))

        def downsample(n: int, k: int) -> np.ndarray:
            if n <= k: return np.arange(n, dtype=np.int64)
            return (np.linspace(0, n-1, num=k)).astype(np.int64)

        def make_color_scaler(vec: np.ndarray | None):
            if vec is None or not isinstance(vec, np.ndarray) or vec.size == 0:
                return lambda _: _rgb_to_hex(_interp_color01(0.0))
            v01 = _normalize01(vec)
            def f(i: int) -> str:
                return _rgb_to_hex(_interp_color01(float(v01[i])))
            return f

        # point color scaler
        pcolor_vec = (npz.get(POINTS_COLOR_KEY) if POINTS_COLOR_KEY else sval) if isinstance(sval, np.ndarray) else None
        point_cfunc = make_color_scaler(pcolor_vec)

        g = VGroup()
        self.add(g)

        # number of frames = max over sidx/eidx - 1
        F = int(max(len(sidx), len(eidx)) - 1)

        # Prepare KDTree function if we need edge colors from state_value
        use_kdtree = (ecolor is None) and isinstance(sval, np.ndarray) and isinstance(sx, np.ndarray)
        if use_kdtree:
            try:
                from scipy.spatial import cKDTree as KDTree
            except Exception:
                KDTree = None
                use_kdtree = False

        for fi in range(F):
            # ----- points (state) -----
            slo, shi = int(sidx[fi]), int(sidx[fi+1])
            pts_group = VGroup()
            if isinstance(sx, np.ndarray) and shi > slo:
                pick = downsample(shi - slo, MAX_POINTS) + slo
                for j in pick:
                    X = norm((float(sx[j]), float(sy[j]), float(sz[j])))
                    dot = Dot3D(point=X, radius=POINT_RADIUS, stroke_width=0, fill_opacity=0.95)
                    dot.set_color(point_cfunc(j))
                    pts_group.add(dot)

            # ----- edges -----
            elo, ehi = int(eidx[fi]), int(eidx[fi+1])
            edges_group = VGroup()
            if isinstance(ex0, np.ndarray) and ehi > elo:
                pick = downsample(ehi - elo, MAX_EDGES) + elo

                # edge color scaler if we have a direct per-edge vector
                edge_cfunc = make_color_scaler(ecolor) if isinstance(ecolor, np.ndarray) else None

                # If no per-edge data, build KDTree on this frame's points to color edges by endpoint state_value
                if edge_cfunc is None and use_kdtree and slo < shi:
                    P = np.c_[sx[slo:shi], sy[slo:shi], sz[slo:shi]]
                    V = sval[slo:shi] if isinstance(sval, np.ndarray) else None
                    tree = KDTree(P) if P.size and V is not None else None

                for j in pick:
                    p0 = norm((float(ex0[j]), float(ey0[j]), float(ez0[j])))
                    p1 = norm((float(ex1[j]), float(ey1[j]), float(ez1[j])))
                    seg = Line3D(p0, p1, stroke_width=EDGE_WIDTH, stroke_opacity=0.95)

                    if edge_cfunc is not None:
                        seg.set_color(edge_cfunc(j))
                    elif use_kdtree and slo < shi and tree is not None:
                        # nearest state_value at each endpoint, average -> color
                        # invert normalization for query: we need original coords (do simple inverse approx since scaling is uniform)
                        # Easier: query in original space (ex0/ey0/ez0 are original already)
                        # Find nearest indices in current frame slice
                        import numpy as _np
                        q0 = (ex0[j], ey0[j], ez0[j])
                        q1 = (ex1[j], ey1[j], ez1[j])
                        d0, i0 = tree.query([q0], k=1)
                        d1, i1 = tree.query([q1], k=1)
                        i0 = int(i0[0]); i1 = int(i1[0])
                        v0 = float(V[i0]); v1 = float(V[i1])
                        val = (v0 + v1) * 0.5
                        # we need a stable normalization per-frame; use V slice
                        v01 = _normalize01(V)
                        # map local indices back to 0..1
                        c0 = float(v01[i0]); c1 = float(v01[i1])
                        c = (c0 + c1) * 0.5
                        seg.set_color(_rgb_to_hex(_interp_color01(c)))
                    else:
                        seg.set_color(_rgb_to_hex(_interp_color01(0.0)))

                    edges_group.add(seg)

            new_frame = VGroup(edges_group, pts_group)
            g.become(new_frame)
            self.wait(STEP_SECS)
