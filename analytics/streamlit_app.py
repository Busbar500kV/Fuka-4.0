# analytics/streamlit_app.py
from __future__ import annotations
import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Robust import shim so this file works whether launched as:
#   - streamlit run analytics/streamlit_app.py             (file mode)
#   - python -m streamlit run analytics/streamlit_app.py   (module mode)
#   - python -m analytics.streamlit_app                    (package mode)
# -----------------------------------------------------------------------------
try:
    # Package-style (preferred)
    from analytics.build_indices import rebuild_indices
except Exception:
    # File-style: add repo root (parent-of-parent) to sys.path and retry
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from analytics.build_indices import rebuild_indices  # type: ignore

import json
import time
from typing import List, Dict, Any

import streamlit as st
import duckdb as ddb

from fuka.render.pack_npz import pack as pack_npz  # canonical packer

# ----------------------------------- UI --------------------------------------

st.set_page_config(page_title="Fuka 4.0 Control Panel", layout="wide")

DEFAULT_DATA_ROOT = os.environ.get("FUKA_DATA_ROOT", "data")
DEFAULT_RUN_ID = ""
DEFAULT_OUT_NPZ = "assets/fuka_anim.npz"
DEFAULT_PREFIX = os.environ.get("DATA_URL_PREFIX", "")

st.title("Fuka 4.0 — Control Panel")

with st.sidebar:
    st.header("Global Settings")
    data_root = st.text_input("Data root", value=DEFAULT_DATA_ROOT)
    url_prefix = st.text_input("Public HTTPS prefix (optional)", value=DEFAULT_PREFIX)
    st.caption("If set, indices & packer will embed this prefix (e.g., GCS https URL).")

# Tabs
tab_runs, tab_pack, tab_sql = st.tabs(["Runs / Indices", "Pack + Render", "SQL Explorer"])

# ---------------------------- Helpers ----------------------------------------

def _runs_list(root: str) -> List[str]:
    manifest = Path(root) / "runs" / "index.json"
    if manifest.is_file():
        try:
            return sorted(json.loads(manifest.read_text()).get("runs", []))
        except Exception:
            pass
    # fallback: list directories under data/runs
    runs_dir = Path(root) / "runs"
    if not runs_dir.is_dir():
        return []
    return sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])

def _run_dir(root: str, run_id: str) -> Path:
    return Path(root) / "runs" / run_id

# ----------------------------- Tab: Runs -------------------------------------

with tab_runs:
    st.subheader("Existing Runs")
    runs = _runs_list(data_root)
    sel = st.selectbox("Select a run", runs, index=0 if runs else None, placeholder="No runs found")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Rebuild Indices + Manifest", disabled=not bool(sel)):
            with st.spinner("Rebuilding indices..."):
                ok = rebuild_indices(data_root=data_root, run_id=sel, prefix=url_prefix)
            if ok:
                st.success(f"indices rebuilt for {sel}")
            else:
                st.error("Failed to rebuild indices (see server logs)")

    with col_b:
        # quick view of run_meta.json
        if sel:
            meta_p = _run_dir(data_root, sel) / "run_meta.json"
            if meta_p.is_file():
                st.caption("run_meta.json")
                st.code(meta_p.read_text()[:8000], language="json")
            else:
                st.caption("run_meta.json not found")

# --------------------------- Tab: Pack + Render ------------------------------

with tab_pack:
    st.subheader("Pack Canonical NPZ")
    run_id = st.text_input("Run ID", value=sel or DEFAULT_RUN_ID, placeholder="e.g., FUKA_2025...")
    step_min, step_max = st.slider("Step window", 0, 10000, (0, 300), help="Inclusive min, inclusive max")
    out_npz = st.text_input("Output NPZ path", value=DEFAULT_OUT_NPZ)

    do_pack = st.button("Pack NPZ", disabled=not bool(run_id))
    if do_pack:
        with st.spinner("Packing NPZ..."):
            ok = pack_npz(prefix=url_prefix, run_id=run_id, step_min=step_min, step_max=step_max, out=out_npz)
        if ok:
            st.success(f"NPZ written to {out_npz}")
        else:
            st.error("Pack failed; check server logs")

    st.divider()
    st.subheader("Render (Manim CLI)")
    st.caption("After packing NPZ, render via shell: `manim -qm --fps 24 render/manim_fuka_scene.py Fuka3DScene`")

# --------------------------- Tab: SQL Explorer -------------------------------

with tab_sql:
    st.subheader("DuckDB: Query Shards")
    st.caption("Use {data_root}/runs/<RUN_ID>/shards/*.parquet paths in your FROM clause.")
    default_sql = f"""
-- Example:
-- SELECT step, AVG(value) AS avg_val
-- FROM parquet_scan('{data_root}/runs/{sel or "<RUN_ID>"}/shards/state_*.parquet')
-- GROUP BY step ORDER BY step LIMIT 20;
"""
    q = st.text_area("SQL", height=180, value=default_sql)
    if st.button("Run SQL"):
        try:
            con = ddb.connect()
            t0 = time.time()
            df = con.execute(q).fetchdf()
            dt = time.time() - t0
            st.success(f"OK in {dt:.2f}s, {len(df):,} rows")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"SQL error: {e}")