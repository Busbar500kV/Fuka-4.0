# analytics/streamlit_app.py
# FUKA UI (HTTPS-only) — restores 3D world/edges, Events, Diagnostics
from __future__ import annotations

import json
import math
import os
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import plotly.graph_objects as go
import streamlit as st
import requests


# ------------------------------- Config -------------------------------

@st.cache_data(show_spinner=False)
def get_env(k: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit Cloud secrets or env
    return st.secrets.get(k, os.environ.get(k, default))  # type: ignore[arg-type]

DATA_URL_PREFIX = get_env("DATA_URL_PREFIX", "https://storage.googleapis.com/fuka4-runs")
DEFAULT_RUN_ID  = get_env("DEFAULT_RUN_ID", None)

# ----------------------------- URL helpers ----------------------------

def _ujoin(*parts: str) -> str:
    parts = [p.strip("/") for p in parts if p is not None]
    if not parts:
        return ""
    return parts[0] + "/" + "/".join(parts[1:])

def url_runs_index(prefix: str) -> str:
    return _ujoin(prefix, "runs", "index.json")

def url_manifest(prefix: str, run_id: str) -> str:
    return _ujoin(prefix, "runs", run_id, "manifest.json")

def url_table_index(prefix: str, run_id: str, table: str) -> str:
    return _ujoin(prefix, "runs", run_id, "shards", f"{table}_index.json")

# ------------------------------ IO (HTTP) -----------------------------

@st.cache_data(show_spinner=False, ttl=300)
def http_json(url: str) -> Dict:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=600)
def http_parquet(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    buf = pa.BufferReader(r.content)
    table = pq.read_table(buf)
    return table.to_pandas(types_mapper=pd.ArrowDtype)

# ------------------------------ Catalog -------------------------------

@st.cache_data(show_spinner=False, ttl=120)
def list_runs(prefix: str) -> List[str]:
    try:
        data = http_json(url_runs_index(prefix))
        runs = data.get("runs", [])
        # newest last or alphabetic — keep stable
        return list(runs)
    except Exception:
        # Fallback: allow manual input if index missing
        return []

@st.cache_data(show_spinner=False, ttl=300)
def load_manifest(prefix: str, run_id: str) -> Dict:
    return http_json(url_manifest(prefix, run_id))

@st.cache_data(show_spinner=False, ttl=300)
def list_shards_from_index(prefix: str, run_id: str, table: str) -> List[str]:
    """Prefer *_index.json (fast), else derive from manifest (slower)."""
    try:
        idx = http_json(url_table_index(prefix, run_id, table))
        files = idx.get("files", [])
        # Already HTTPS; return directly
        return files
    except Exception:
        # derive from manifest
        man = load_manifest(prefix, run_id)
        shards = man.get("shards", [])
        rels = [s["path"] for s in shards if s.get("table") == table]
        return [_ujoin(prefix, rel) for rel in rels]

# ------------------------------ Data loads ----------------------------

def _pick_first_n(urls: List[str], n: int) -> List[str]:
    return urls[:max(0, min(n, len(urls)))]

@st.cache_data(show_spinner=True, ttl=300)
def load_table(prefix: str, run_id: str, table: str, max_shards: int = 10) -> pd.DataFrame:
    urls = list_shards_from_index(prefix, run_id, table)
    urls = _pick_first_n(urls, max_shards)
    frames: List[pd.DataFrame] = []
    for u in urls:
        try:
            frames.append(http_parquet(u))
        except Exception as e:
            st.warning(f"Failed to load shard: {u}\n{e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df

# ------------------------------ Utilities -----------------------------

def find_xyz_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Try common variants
    candidates = [
        ("x", "y", "z"),
        ("X", "Y", "Z"),
        ("i", "j", "k"),
        ("pos_x", "pos_y", "pos_z"),
    ]
    for trip in candidates:
        if all(c in df.columns for c in trip):
            return trip
    return (None, None, None)

def find_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("node", "node_id", "id", "idx"):
        if c in df.columns:
            return c
    return None

def find_step_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("step", "t", "time", "iter"):
        if c in df.columns:
            return c
    return None

def find_edge_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    for s, d in (("src", "dst"), ("source", "target"), ("u", "v"), ("i", "j")):
        if s in df.columns and d in df.columns:
            return s, d
    return (None, None)

# ------------------------------- Layout -------------------------------

st.set_page_config(page_title="FUKA 4.0", layout="wide")
st.title("FUKA 4.0 — World • Edges • Events • Diagnostics")

with st.sidebar:
    st.subheader("Data Source")
    prefix = st.text_input("DATA_URL_PREFIX", value=DATA_URL_PREFIX or "")
    runs = list_runs(prefix)
    run_id = DEFAULT_RUN_ID or (runs[-1] if runs else "")
    run_id = st.text_input("Run ID", value=run_id)
    if runs:
        st.caption("Available runs (from runs/index.json):")
        st.code("\n".join(runs[-10:]) or "(none)", language="text")

    st.divider()
    st.subheader("Load limits")
    max_state_shards = st.number_input("Max state shards", 1, 1000, 10, 1)
    max_edges_shards = st.number_input("Max edge shards", 1, 1000, 10, 1)
    max_events_shards = st.number_input("Max event shards", 1, 1000, 10, 1)

    st.caption("Tip: increase if you need more history; this keeps loads fast.")

if not prefix or not run_id:
    st.warning("Set DATA_URL_PREFIX and Run ID to begin.")
    st.stop()

# Manifest preview
try:
    manifest = load_manifest(prefix, run_id)
    with st.expander("Manifest", expanded=False):
        st.json(manifest)
except Exception as e:
    st.error(f"Failed to fetch manifest for run '{run_id}' at {url_manifest(prefix, run_id)}\n{e}")
    st.stop()

tabs = st.tabs(["Overview", "3D World", "Edges (3D)", "Events", "Diagnostics"])

# ------------------------------ Overview ------------------------------
with tabs[0]:
    st.subheader("Quick stats")
    c1, c2, c3 = st.columns(3)
    try:
        state_idx = list_shards_from_index(prefix, run_id, "state")
        edges_idx = list_shards_from_index(prefix, run_id, "edges")
        env_idx   = list_shards_from_index(prefix, run_id, "env")
    except Exception:
        state_idx, edges_idx, env_idx = [], [], []

    c1.metric("State shards", f"{len(state_idx)}")
    c2.metric("Edge shards",  f"{len(edges_idx)}")
    c3.metric("Env shards",   f"{len(env_idx)}")

    st.caption("Below loads only a limited window so the app stays responsive.")

# ------------------------------ 3D World ------------------------------
with tabs[1]:
    st.subheader("World (3D scatter)")
    state_df = load_table(prefix, run_id, "state", max_shards=max_state_shards)
    if state_df.empty:
        st.info("No state shards found (or columns missing).")
    else:
        step_col = find_step_column(state_df)
        xcol, ycol, zcol = find_xyz_columns(state_df)
        id_col = find_id_column(state_df)

        if not (xcol and ycol and zcol):
            st.warning("Couldn’t find x/y/z columns in state table.")
        else:
            if step_col:
                min_step, max_step = int(state_df[step_col].min()), int(state_df[step_col].max())
                sel = st.slider("Step", min_value=min_step, max_value=max_step, value=min_step, step=1)
                sdf = state_df[state_df[step_col] == sel]
            else:
                st.caption("No step column; showing all points.")
                sdf = state_df

            fig = go.Figure()
            hover = id_col if id_col else None
            fig.add_trace(go.Scatter3d(
                x=sdf[xcol], y=sdf[ycol], z=sdf[zcol],
                mode="markers",
                marker=dict(size=3),
                text=(sdf[hover] if hover else None),
                name="nodes"
            ))
            fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0), height=700)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------ Edges 3D ------------------------------
with tabs[2]:
    st.subheader("Edges (3D lines)")
    edges_df = load_table(prefix, run_id, "edges", max_shards=max_edges_shards)
    if edges_df.empty:
        st.info("No edges shards found (or columns missing).")
    else:
        state_df = load_table(prefix, run_id, "state", max_shards=max_state_shards)
        scol, dcol = find_edge_columns(edges_df)
        xcol, ycol, zcol = find_xyz_columns(state_df)
        id_col = find_id_column(state_df)
        step_col = find_step_column(state_df)

        if not (scol and dcol):
            st.warning("Couldn’t find edge source/target columns.")
        elif state_df.empty or not (xcol and ycol and zcol and id_col):
            st.warning("State table missing (or lacks id/x/y/z) to position edges.")
        else:
            if step_col:
                min_step, max_step = int(state_df[step_col].min()), int(state_df[step_col].max())
                sel = st.slider("Step", min_value=min_step, max_value=max_step, value=min_step, step=1, key="edges_step")
                sdf = state_df[state_df[step_col] == sel][[id_col, xcol, ycol, zcol]].drop_duplicates(subset=[id_col])
            else:
                st.caption("No step column in state; joining without filtering.")
                sdf = state_df[[id_col, xcol, ycol, zcol]].drop_duplicates(subset=[id_col])

            # left-join twice to get coordinates for source/target
            src = edges_df.merge(sdf, how="left", left_on=scol, right_on=id_col, suffixes=("", "_src"))
            src.rename(columns={xcol: "sx", ycol: "sy", zcol: "sz"}, inplace=True)
            tgt = src.merge(sdf, how="left", left_on=dcol, right_on=id_col, suffixes=("", "_tgt"))
            tgt.rename(columns={xcol: "tx", ycol: "ty", zcol: "tz"}, inplace=True)

            # Build line segments for plotly (NaNs will be skipped)
            coords = []
            for _, r in tgt.iterrows():
                coords.append((r.get("sx"), r.get("sy"), r.get("sz")))
                coords.append((r.get("tx"), r.get("ty"), r.get("tz")))
                coords.append((None, None, None))  # separator

            if not coords:
                st.info("No edge coords could be derived.")
            else:
                xs, ys, zs = zip(*coords)
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=list(xs), y=list(ys), z=list(zs),
                    mode="lines",
                    line=dict(width=1),
                    name="edges"
                ))
                fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0), height=700)
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------ Events -------------------------------
with tabs[3]:
    st.subheader("Events")
    events_df = load_table(prefix, run_id, "events", max_shards=max_events_shards)
    if events_df.empty:
        st.info("No events table found.")
    else:
        # show a lightweight table; limit rows for responsiveness
        st.dataframe(events_df.head(200))

# ----------------------------- Diagnostics ---------------------------
with tabs[4]:
    st.subheader("Diagnostics")
    # Try some common diagnostics: spectra, ledger, env summaries if exist
    diag_tabs = st.tabs(["Env", "Spectra", "Ledger"])

    with diag_tabs[0]:
        env_df = load_table(prefix, run_id, "env", max_shards=5)
        if env_df.empty:
            st.info("No env table.")
        else:
            st.dataframe(env_df.head(300))

    with diag_tabs[1]:
        spectra_df = load_table(prefix, run_id, "spectra", max_shards=5)
        if spectra_df.empty:
            st.info("No spectra table.")
        else:
            # Auto-plot first two numeric columns vs step if present
            step_col = find_step_column(spectra_df)
            numeric_cols = [c for c in spectra_df.columns if pd.api.types.is_numeric_dtype(spectra_df[c])]
            ycols = [c for c in numeric_cols if c != (step_col or "")]
            if step_col and ycols:
                sel_y = st.multiselect("Y columns", ycols, default=ycols[:1])
                fig = go.Figure()
                for yc in sel_y:
                    fig.add_trace(go.Scatter(x=spectra_df[step_col], y=spectra_df[yc], mode="lines", name=yc))
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(spectra_df.head(300))

    with diag_tabs[2]:
        ledger_df = load_table(prefix, run_id, "ledger", max_shards=5)
        if ledger_df.empty:
            st.info("No ledger table.")
        else:
            st.dataframe(ledger_df.head(300))