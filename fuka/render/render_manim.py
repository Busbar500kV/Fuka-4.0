# fuka/render/render_manim.py
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

def run(
    npz_path: str,
    quality: str = "med",              # low|med|high
    fps: int = 12,
    step_seconds: float = 0.20,
    max_points: int = 80000,
    max_edges: int = 60000,
    edge_width: float = 3.0,
    point_radius: float = 0.05,
    color_mode: str = "auto",          # auto|edge_strength|edge_deposit|edge_kappa|edge_value|point_value
    color_key: str | None = None,      # explicit NPZ key to color by (edges)
    points_color_key: str | None = None,# explicit NPZ key to color by (points)
    phi_deg: float = 65.0,
    theta_deg: float = -45.0,
    zoom: float = 1.10,
):
    """
    Shells out to manim to render render/manim_fuka_scene.py while passing options via env vars.
    """

    npz_path = str(Path(npz_path).expanduser().resolve())
    scene_py = Path(__file__).resolve().parents[2] / "render" / "manim_fuka_scene.py"
    if not scene_py.exists():
        raise SystemExit(f"Scene file not found: {scene_py}")

    env = os.environ.copy()
    env["FUKA_NPZ"] = npz_path
    env["FUKA_QUALITY"] = quality
    env["FUKA_FPS"] = str(int(fps))
    env["FUKA_STEP_SECONDS"] = str(float(step_seconds))
    env["FUKA_MAX_POINTS"] = str(int(max_points))
    env["FUKA_MAX_EDGES"] = str(int(max_edges))
    env["FUKA_EDGE_WIDTH"] = str(float(edge_width))
    env["FUKA_POINT_RADIUS"] = str(float(point_radius))
    env["FUKA_COLOR_MODE"] = color_mode
    if color_key:
        env["FUKA_COLOR_KEY"] = color_key
    if points_color_key:
        env["FUKA_POINTS_COLOR_KEY"] = points_color_key
    env["FUKA_CAM_PHI"] = str(float(phi_deg))
    env["FUKA_CAM_THETA"] = str(float(theta_deg))
    env["FUKA_CAM_ZOOM"] = str(float(zoom))

    # choose manim quality flags
    qflag = {"low": "-ql", "med": "-qm", "high": "-qh"}.get(quality, "-qm")

    cmd = [
        "manim",
        qflag,
        "-v", "WARNING",
        "--fps", str(int(fps)),
        str(scene_py),
        "FukaScene",
    ]
    print("[fuka:render] running:", " ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise SystemExit(rc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Path to packed NPZ from pack_npz.py")
    ap.add_argument("--quality", default="med", choices=["low", "med", "high"])
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--step-seconds", type=float, default=0.20)
    ap.add_argument("--max-points", type=int, default=80000)
    ap.add_argument("--max-edges", type=int, default=60000)
    ap.add_argument("--edge-width", type=float, default=3.0)
    ap.add_argument("--point-radius", type=float, default=0.05)
    ap.add_argument("--color-mode", default="auto",
                    choices=["auto","edge_strength","edge_deposit","edge_kappa","edge_value","point_value"])
    ap.add_argument("--color-key", default=None, help="Explicit NPZ key for edge colors")
    ap.add_argument("--points-color-key", default=None, help="Explicit NPZ key for point colors")
    ap.add_argument("--phi", type=float, default=65.0)
    ap.add_argument("--theta", type=float, default=-45.0)
    ap.add_argument("--zoom", type=float, default=1.10)
    args = ap.parse_args()

    run(
        npz_path=args.npz,
        quality=args.quality,
        fps=args.fps,
        step_seconds=args.step_seconds,
        max_points=args.max_points,
        max_edges=args.max_edges,
        edge_width=args.edge_width,
        point_radius=args.point_radius,
        color_mode=args.color_mode,
        color_key=args.color_key,
        points_color_key=args.points_color_key,
        phi_deg=args.phi,
        theta_deg=args.theta,
        zoom=args.zoom,
    )

if __name__ == "__main__":
    raise SystemExit(main())
