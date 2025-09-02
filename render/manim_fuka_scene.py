# render/manim_fuka_scene.py
from manim import *
import numpy as np
import os

def colormap_viridis(vals: np.ndarray) -> list[Color]:
    if len(vals) == 0:
        return []
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        t = np.zeros_like(vals, dtype=float)
    else:
        t = (vals - vmin) / (vmax - vmin)
    palette = [BLUE_E, BLUE_D, BLUE_C, TEAL_E, GREEN_E, YELLOW_E]
    out = []
    for x in t:
        x = float(np.clip(x, 0.0, 1.0))
        i = int(x * (len(palette)-1))
        j = min(len(palette)-1, i+1)
        alpha = x * (len(palette)-1) - i
        out.append(interpolate_color(palette[i], palette[j], alpha))
    return out

class FukaWorldEdges3D(ThreeDScene):
    """
    Render a 3D movie of world points + edges over steps.
    Expects NPZ path via env FUKA_NPZ or default 'assets/fuka_anim.npz'.
    """
    def construct(self):
        npz_path = os.environ.get("FUKA_NPZ", "assets/fuka_anim.npz")
        secs_per_step = float(os.environ.get("FUKA_STEP_SECONDS", "0.05"))  # 20 steps/sec
        point_radius  = float(os.environ.get("FUKA_POINT_RADIUS", "0.035"))
        edge_width    = float(os.environ.get("FUKA_EDGE_WIDTH", "2.5"))
        show_edges    = os.environ.get("FUKA_SHOW_EDGES", "1") not in ("0","false","False")

        data = np.load(npz_path, allow_pickle=True)
        steps  = data["steps"]
        states = data["states"]
        edges  = data["edges"]

        # Scene frame & camera
        self.set_camera_orientation(phi=70*DEGREES, theta=30*DEGREES, zoom=1.25)
        # auto bounds (if your data is normalized, keep [-1,1])
        axes = ThreeDAxes(x_range=[-1,1,1], y_range=[-1,1,1], z_range=[-1,1,1])
        self.add(axes)

        # Initial frame
        k0 = 0
        pts0 = states[k0]
        ed0  = edges[k0]

        pts_group = VGroup()
        if len(pts0) > 0:
            colors = colormap_viridis(pts0[:,3])
            for (x,y,z,val), c in zip(pts0, colors):
                pts_group.add(Dot3D(point=[x,y,z], radius=point_radius, color=c))
        self.add(pts_group)

        edge_group = VGroup()
        if show_edges and len(ed0) > 0:
            for (x0,y0,z0,x1,y1,z1,_,_) in ed0:
                edge_group.add(Line3D([x0,y0,z0], [x1,y1,z1], stroke_width=edge_width, color=GRAY_B))
        self.add(edge_group)

        # Animate: swap groups per step
        for k in range(1, len(steps)):
            pts = states[k]
            eds = edges[k]

            new_pts = VGroup()
            if len(pts) > 0:
                colors = colormap_viridis(pts[:,3])
                for (x,y,z,val), c in zip(pts, colors):
                    new_pts.add(Dot3D(point=[x,y,z], radius=point_radius, color=c))

            new_edges = VGroup()
            if show_edges and len(eds) > 0:
                for (x0,y0,z0,x1,y1,z1,_,_) in eds:
                    new_edges.add(Line3D([x0,y0,z0], [x1,y1,z1], stroke_width=edge_width, color=GRAY_B))

            # replace over a small duration; linear timing
            self.play(
                ReplacementTransform(pts_group, new_pts),
                ReplacementTransform(edge_group, new_edges) if show_edges else AnimationGroup(),
                run_time=secs_per_step, rate_func=linear
            )
            pts_group = new_pts
            edge_group = new_edges

        self.wait(0.25)