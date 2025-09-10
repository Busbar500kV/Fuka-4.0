# analytics/streamlit_app.py
from __future__ import annotations
import os
import sys
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import duckdb as ddb

# Project imports (canonical pipeline)
from .build_indices import rebuild_indices
from fuka.render.pack_npz import pack as pack_npz


# ----------------------------- helpers -----------------------------

DATA_ROOT = Path(os.environ.get("FUKA_DATA_ROOT", "data"))
CONFIG_DIR = Path("configs")
ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def _runs() -> List[Path]:
    runs_dir = DATA_ROOT / "runs"
    if not runs_dir.is_dir():
        return []
    items = [p for p in runs_dir.iterdir() if p.is_dir()]
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return items

def _run_meta(run_dir: Path) -> Dict[str, str]:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text("utf-8"))
    except Exception:
        return {}

def _shards(run_dir: Path) -> List[Path]:
    sd = run_dir / "shards"
    return sorted(sd.glob("*.parquet")) if sd.is_dir() else []

def _table_counts(run_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in _shards(run_dir):
        table = p.name.split("_", 1)[0].lower()
        counts[table] = counts.get(table, 0) + 1
    return counts

def _prefix() -> Optional[str]:
    v = os.environ.get("DATA_URL_PREFIX", "").strip()
    return v or None

def _call_render_script(
    run_id: str, step_min: int, step_max: int, out_npz: str, quality: str, fps: int
) -> Tuple[int, str]:
    """Invoke scripts/render_fuka.sh and return (exit_code, combined_output)."""
    script = Path("scripts") / "render_fuka.sh"
    if not script.exists():
        return 2, "[ui] Missing scripts/render_fuka.sh"
    env = os.environ.copy()
    cmd = [
        "bash",
        str(script),
        *([ "--prefix", _prefix() ] if _prefix() else []),
        "--run_id", run_id,
        "--step_min", str(step_min),
        "--step_max", str(step_max),
        "--out_npz", out_npz,
        "--rebuild_indices", "1",     # ensure up-to-date indices for UI calls
        "--data_root", str(DATA_ROOT),
        "--quality", quality,
        "--fps", str(fps),
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        out = proc.stdout + "\n" + proc.stderr
        return proc.returncode, out
    except Exception as e:
        return 2, f"[ui] Render invocation failed: {e}"

def _human_size(bytes_: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    n = float(bytes_)
    for u in units:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def _get_config_files() -> List[Path]:
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(CONFIG_DIR.glob("*.json"))

def _read_steps_from_config(p: Path) -> int:
    try:
        o = json.loads(p.read_text("utf-8"))
        return int(((o.get("run") or {}).get("steps", 0)))
    except Exception:
        return 0


# ----------------------------- UI -----------------------------

st.set_page_config(page_title="Fuka 4.0 – Control Panel", layout="wide")

st.title("Fuka 4.0 — Headless Control & Viewer")

tab_runs, tab_new, tab_sql = st.tabs(["Runs", "New Run", "SQL Explorer"])

# ----------------------------- Runs tab -----------------------------

with tab_runs:
    colL, colR = st.columns([2, 3], gap="large")

    with colL:
        st.subheader("Available runs")
        run_paths = _runs()
        if not run_paths:
            st.info("No runs found under `data/runs/`. Launch a new run from the 'New Run' tab.")
        run_labels = [p.name for p in run_paths]
        selected = st.selectbox("Select a run", run_labels, index=0 if run_labels else None)

        if selected:
            run_dir = DATA_ROOT / "runs" / selected
            meta = _run_meta(run_dir)
            counts = _table_counts(run_dir)

            st.markdown(f"**Run ID:** `{selected}`")
            if meta:
                st.json(meta, expanded=False)

            # Shard table summary
            df = pd.DataFrame(
                [{"table": k, "shards": v} for k, v in sorted(counts.items())]
            )
            st.dataframe(df, use_container_width=True)

            # Actions
            st.divider()
            st.subheader("Actions")

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Rebuild Indices", type="primary"):
                    with st.spinner("Rebuilding indices..."):
                        try:
                            mapping = rebuild_indices(str(DATA_ROOT), selected, _prefix())
                            st.success(f"OK. Tables: {', '.join(sorted(mapping.keys()))}")
                        except Exception as e:
                            st.error(f"Index rebuild failed: {e}")

            with c2:
                st.write("**Pack NPZ**")
                step_max_guess = max(0, _read_steps_from_config(CONFIG_DIR / "config_default.json") - 1)
                step_min = st.number_input("Step min", min_value=0, value=0, step=1, key="pack_min")
                step_max = st.number_input("Step max", min_value=step_min, value=step_max_guess, step=1, key="pack_max")
                out_npz = st.text_input("Output NPZ", value=str(ASSETS_DIR / "fuka_anim.npz"))
                if st.button("Pack", key="pack_btn"):
                    with st.spinner("Packing NPZ..."):
                        try:
                            pack_npz(prefix=_prefix() or "", run_id=selected,
                                     step_min=int(step_min), step_max=int(step_max),
                                     out_path=out_npz)
                            st.success(f"Packed: {out_npz}")
                        except Exception as e:
                            st.error(f"Pack failed: {e}")

            with c3:
                st.write("**Render MP4**")
                quality = st.selectbox("Quality", ["l", "m", "h", "k"], index=3)
                fps = st.number_input("FPS", min_value=1, max_value=60, value=24, step=1)
                out_npz_r = st.text_input("NPZ path (render input)", value=str(ASSETS_DIR / "fuka_anim.npz"), key="npz_render")
                if st.button("Render", key="render_btn"):
                    with st.spinner("Rendering with Manim..."):
                        code, out = _call_render_script(selected, int(step_min), int(step_max), out_npz_r, quality, int(fps))
                        if code == 0:
                            st.success("Render complete.")
                        else:
                            st.error("Render failed.")
                        st.code(out, language="bash")

    with colR:
        st.subheader("Shard files")
        if selected:
            files = _shards(DATA_ROOT / "runs" / selected)
            if not files:
                st.info("No shard files found yet.")
            else:
                recs = []
                for p in files:
                    try:
                        sz = p.stat().st_size
                    except Exception:
                        sz = 0
                    recs.append({
                        "filename": p.name,
                        "table": p.name.split("_", 1)[0],
                        "size": _human_size(sz)
                    })
                df = pd.DataFrame(recs)
                st.dataframe(df, use_container_width=True, height=440)

# ----------------------------- New Run tab -----------------------------

with tab_new:
    st.subheader("Launch a new headless run")

    cfg_files = _get_config_files()
    if not cfg_files:
        st.warning("No config files in `configs/`. Add one first.")
    else:
        cfg_map = {p.name: p for p in cfg_files}
        chosen = st.selectbox("Config", list(cfg_map.keys()), index=0)
        rid = st.text_input("Run ID (optional)", value="")
        steps_override = st.number_input("Override steps (-1 = use config)", min_value=-1, value=-1, step=1)
        pub_prefix = st.text_input("Public prefix (optional)", value=_prefix() or "")

        if st.button("Start Run", type="primary"):
            # call the runner module as a subprocess to avoid module path headaches
            cmd = [
                sys.executable, "-m", "fuka.runner",
                "--config", str(cfg_map[chosen]),
                "--data_root", str(DATA_ROOT),
            ]
            if rid.strip():
                cmd += ["--run_id", rid.strip()]
            if pub_prefix.strip():
                cmd += ["--prefix", pub_prefix.strip()]

            # Optional steps override: patch a temp config if requested
            temp_cfg = None
            if int(steps_override) >= 0:
                # create a temp JSON with overridden steps
                try:
                    obj = json.loads(cfg_map[chosen].read_text("utf-8"))
                    obj.setdefault("run", {})["steps"] = int(steps_override)
                    temp_cfg = Path(".ui_tmp_cfg.json")
                    temp_cfg.write_text(json.dumps(obj, indent=2), encoding="utf-8")
                    cmd[cmd.index("--config") + 1] = str(temp_cfg)
                except Exception as e:
                    st.error(f"Failed to override steps: {e}")
                    temp_cfg = None

            with st.spinner("Running headless..."):
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    st.code(proc.stdout + "\n" + proc.stderr, language="bash")
                    if proc.returncode == 0:
                        st.success("Run completed.")
                    else:
                        st.error("Run failed.")
                finally:
                    if temp_cfg and temp_cfg.exists():
                        temp_cfg.unlink(missing_ok=True)

# ----------------------------- SQL Explorer tab -----------------------------

with tab_sql:
    st.subheader("DuckDB SQL explorer (read-only)")

    runs = [p.name for p in _runs()]
    if not runs:
        st.info("No runs yet.")
    else:
        sel = st.selectbox("Run", runs, index=0)
        run_dir = DATA_ROOT / "runs" / sel
        shards = _shards(run_dir)
        if not shards:
            st.info("No shards found for this run.")
        else:
            query = st.text_area(
                "SQL (tables by file path via read_parquet)",
                value="SELECT * FROM read_parquet('data/runs/{sel}/shards/state_*.parquet') LIMIT 20".format(sel=sel),
                height=120,
            )
            if st.button("Execute"):
                try:
                    con = ddb.connect(":memory:")
                    df = con.execute(query).df()
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Query failed: {e}")