# render/manim_fuka_scene.py
from __future__ import annotations
from manim import *
import numpy as np, os

def _cmap(vals: np.ndarray):
    if len(vals) == 0:
        return []
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax <= vmin:
        t = np.zeros_like(vals)
    else:
        t = (vals - vmin) / (vmax - vmin)
    pal = [BLUE_E, BLUE_D, BLUE_C, TEAL_E, GREEN_E, YELLOW_E]
    out = []
    for x in t:
        i = int(x * (len(pal) - 1))
        j = min(len(pal) - 1, i + 1)
        a = x * (len(pal) - 1) - i
        out.append(interpolate_color(pal[i], pal[j], a))
    return out

class FukaWorldEdges3D(ThreeDScene):
    def construct(self):
        npz = os.environ.get("FUKA_NPZ", "assets/fuka_anim.npz")
        step_sec = float(os.environ.get("FUKA_STEP_SECONDS", "0.05"))
        pr = float(os.environ.get("FUKA_POINT_RADIUS", "0.035"))
        ew = float(os.environ.get("FUKA_EDGE_WIDTH", "2.5"))
        show_edges = os.environ.get("FUKA_SHOW_EDGES", "1") not in ("0","false","False")

        data = np.load(npz, allow_pickle=True)
        steps = data["steps"]
        states = data["states"]
        edges = data["edges"]

        self.set_camera_orientation(phi=70*DEGREES, theta=30*DEGREES, zoom=1.25)
        axes = ThreeDAxes(x_range=[-1,1,1], y_range=[-1,1,1], z_range=[-1,1,1])
        self.add(axes)

        def make_frame(k: int):
            pts = VGroup()
            s = states[k]
            if len(s) > 0:
                colors = _cmap(s[:,3])
                for (x,y,z,v), c in zip(s, colors):
                    pts.add(Dot3D(point=[x,y,z], radius=pr, color=c))
            edg = VGroup()
            e = edges[k]
            if show_edges and len(e) > 0:
                for x0,y0,z0,x1,y1,z1,_,_ in e:
                    edg.add(Line3D([x0,y0,z0], [x1,y1,z1], stroke_width=ew, color=GRAY_B))
            return pts, edg

        pts, edg = make_frame(0)
        self.add(pts, edg)

        for k in range(1, len(steps)):
            npts, nedg = make_frame(k)
            self.play(
                ReplacementTransform(pts, npts),
                ReplacementTransform(edg, nedg) if show_edges else AnimationGroup(),
                run_time=step_sec, rate_func=linear
            )
            pts, edg = npts, nedg

        self.wait(0.25)