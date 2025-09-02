# fuka/render/render_manim.py
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

def run(npz_path: str,
        quality: str = "med",
        fps: int = 12,
        step_seconds: float = 0.20,
        max_points: int = 80000,
        max_edges: int = 20000,
        disable_cache: bool = False,
        scene_qual_name: str = "render.manim_fuka_scene.FukaWorldEdges3D",
        media_dir: str | None = None) -> None:

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    # Split scene module/class
    if scene_qual_name.count(".") < 1:
        raise ValueError("scene_qual_name must be <module>.<Class>")
    mod_name, cls_name = scene_qual_name.rsplit(".", 1)

    # Import module to locate file path
    mod = importlib.import_module(mod_name)
    scene_file = inspect.getsourcefile(mod)
    if not scene_file:
        raise RuntimeError(f"Cannot locate source file for module {mod_name}")

    repo_dir = os.environ.get("REPO_DIR", str(Path(scene_file).resolve().parents[1]))
    scene_path = str(Path(scene_file).resolve())
    media_dir = media_dir or str(Path(repo_dir) / "media_out")

    Path(media_dir).mkdir(parents=True, exist_ok=True)

    # Export env for the scene
    env = os.environ.copy()
    env["FUKA_NPZ"] = npz_path
    env["FUKA_STEP_SECONDS"] = str(step_seconds)
    env["FUKA_MAX_POINTS"] = str(max_points)
    env["FUKA_MAX_EDGES"] = str(max_edges)
    env.setdefault("FUKA_POINT_RADIUS", "0.035")
    env.setdefault("FUKA_EDGE_WIDTH", "2.5")
    env.setdefault("FUKA_SHOW_EDGES", "1")

    # quality flags
    qflag = {"low": "-ql", "med": "-qm", "high": "-qh"}.get(quality, "-qm")
    cache_flag = "--disable_caching" if disable_cache else ""

    # Output name based on NPZ
    stem = Path(npz_path).with_suffix("").name
    out_name = f"{stem}.mp4"

    cmd = [
        sys.executable, "-m", "manim", scene_path,
        qflag,
        cache_flag,
        "--fps", str(int(fps)),
        "--media_dir", media_dir,
        "-o", out_name,
        cls_name,
    ]
    # Remove empty args (cache_flag may be '')
    cmd = [c for c in cmd if c]

    print("[render_manim] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--quality", default="med", choices=["low","med","high"])
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--step_seconds", type=float, default=0.20)
    ap.add_argument("--max_points", type=int, default=80000)
    ap.add_argument("--max_edges", type=int, default=20000)
    ap.add_argument("--disable_cache", action="store_true")
    ap.add_argument("--scene", default="render.manim_fuka_scene.FukaWorldEdges3D")
    ap.add_argument("--media_dir", default=None)
    args = ap.parse_args(argv)

    run(
        npz_path=args.npz,
        quality=args.quality,
        fps=args.fps,
        step_seconds=args.step_seconds,
        max_points=args.max_points,
        max_edges=args.max_edges,
        disable_cache=args.disable_cache,
        scene_qual_name=args.scene,
        media_dir=args.media_dir,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())