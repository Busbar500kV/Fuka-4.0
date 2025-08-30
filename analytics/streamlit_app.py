# analytics/streamlit_app.py

import json, time, urllib.request
from urllib.error import HTTPError, URLError
from typing import Dict, List, Optional, Tuple

import duckdb as ddb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- Page / session ----------------
st.set_page_config(page_title="Fuka 4.0 – Analytics", layout="wide")

PREFIX_DEFAULT = st.secrets.get("DATA_URL_PREFIX", "").rstrip("/")
if not PREFIX_DEFAULT:
    st.warning("DATA_URL_PREFIX secret missing (e.g. https://storage.googleapis.com/fuka4-runs)")

# session defaults
st.session_state.setdefault("playing", False)
st.session_state.setdefault("frame_step", 0)
st.session_state.setdefault("override_files", None)  # dict: table -> [urls]

# ---------------- HTTP helpers ----------------
@st.cache_data(show_spinner=False, ttl=30)
def http_json(url: str) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

@st.cache_data(show_spinner=False, ttl=30)
def http_text(url: str) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

def runs_index_url(prefix: str) -> str:
    return f"{prefix}/runs/index.json"

def table_index_url(prefix: str, run_id: str, table: str) -> str:
    return f"{prefix}/runs/{run_id}/shards/{table}_index.json"

def manifest_url(prefix: str, run_id: str) -> str:
    return f"{prefix}/runs/{run_id}/manifest.json"

def manifest_to_https(prefix: str, manifest: dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for s in manifest.get("shards", []):
        t = s.get("table"); p = s.get("path","")
        if not t or not p: continue
        if p.startswith("data/"): p = p[len("data/"):]
        url = f"{prefix}/{p}"
        if url.endswith(".parquet"): out.setdefault(t, []).append(url)
    for k in out: out[k].sort()
    return out

@st.cache_data(show_spinner=False, ttl=30)
def list_runs(prefix: str) -> List[str]:
    data = http_json(runs_index_url(prefix))
    if data and "runs" in data:
        return sorted({str(x) for x in data["runs"]})
    return []

@st.cache_data(show_spinner=False, ttl=30)
def load_index(prefix: str, run_id: str, table: str) -> List[str]:
    data = http_json(table_index_url(prefix, run_id, table))
    if not data: return []
    return data.get("files", []) or []

def get_files(prefix: str, run_id: str, table: str) -> List[str]:
    ov = st.session_state.get("override_files")
    if isinstance(ov, dict) and table in ov:
        return ov[table]
    return load_index(prefix, run_id, table)

# ---------------- Sidebar ----------------
st.sidebar.title("Fuka 4.0 Analytics")
prefix = st.sidebar.text_input("Data URL prefix", PREFIX_DEFAULT)
runs = list_runs(prefix)
run_id = st.sidebar.selectbox("Run", runs, index=0 if runs else None, placeholder="Select…")
manual_run = st.sidebar.text_input("Or enter run id", value="" if run_id else "")
if not run_id and manual_run:
    run_id = manual_run.strip()
if not run_id:
    st.stop()

# ---------------- Header + animation ----------------
st.title(f"Fuka 4.0 — Run: {run_id}")

def discover_step_bounds(prefix: str, run_id: str) -> Tuple[int,int]:
    for table in ("ledger","events","state"):
        files = get_files(prefix, run_id, table)
        if files:
            con = ddb.connect(":memory:")
            try:
                df = con.execute("SELECT MIN(step) lo, MAX(step) hi FROM read_parquet(?)", [files]).df()
                if not df.empty and pd.notnull(df.iloc[0]["lo"]) and pd.notnull(df.iloc[0]["hi"]):
                    return int(df.iloc[0]["lo"]), int(df.iloc[0]["hi"])
            except Exception:
                pass
            finally:
                con.close()
    return 0, 10_000

lo_guess, hi_guess = discover_step_bounds(prefix, run_id)

c1,c2,c3,c4 = st.columns([1,1,1,2])
with c1:
    step_min = st.number_input("Step min", min_value=0, value=lo_guess, step=100)
with c2:
    step_max = st.number_input("Step max", min_value=1, value=max(hi_guess, lo_guess+1), step=500)
with c3:
    fps   = st.slider("FPS", 1, 24, 6)
    trail = st.slider("Trail (steps)", 1, 5000, 200)
with c4:
    frame_lo, frame_hi = int(step_min), int(step_max)
    # clamp existing frame in new bounds
    st.session_state.frame_step = max(frame_lo, min(frame_hi, int(st.session_state.frame_step)))
    frame = st.slider("Frame (step)", frame_lo, frame_hi, value=st.session_state.frame_step, key="frame_slider")
    # keep session value in sync with slider
    if frame != st.session_state.frame_step:
        st.session_state.frame_step = int(frame)
    # play/pause
    label = "⏵ Play" if not st.session_state.playing else "⏸ Pause"
    if st.button(label, use_container_width=True):
        st.session_state.playing = not st.session_state.playing

# animate by bumping frame then rerunning
if st.session_state.playing:
    time.sleep(1.0/max(1,fps))
    step = max(1, (frame_hi - frame_lo)//200 or 1)
    nxt = st.session_state.frame_step + step
    if nxt > frame_hi: nxt = frame_lo
    st.session_state.frame_step = nxt
    # Use modern API; some hosts don't expose experimental_rerun
    st.rerun()

frame = int(st.session_state.frame_step)
win_lo = max(frame_lo, frame - trail)
win_hi = min(frame_hi, frame)

st.caption(f"Window: [{win_lo}, {win_hi}] | frame={frame} | fps={fps} | trail={trail}")

# ---------------- DuckDB ----------------
con = ddb.connect(":memory:")

# ---------------- Tabs ----------------
tab_overview, tab_events, tab_spectra, tab_world3d, tab_edges3d, tab_diag = st.tabs(
    ["Overview", "Events", "Spectra", "World 3D", "Edges 3D", "Diagnostics"]
)

# ---- Overview ----
with tab_overview:
    st.subheader("Aggregates")
    files_ledger = get_files(prefix, run_id, "ledger")
    files_state  = get_files(prefix, run_id, "state")
    files_events = get_files(prefix, run_id, "events")
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("Ledger shards", len(files_ledger))
    with k2: st.metric("State shards",  len(files_state))
    with k3: st.metric("Event shards",  len(files_events))
    man = http_json(manifest_url(prefix, run_id))
    with k4: st.metric("Manifest entries", len(man.get("shards",[])) if man else 0)

    if files_ledger:
        df = con.execute("""
            SELECT MIN(step) AS min_step, MAX(step) AS max_step,
                   SUM(dF) AS sum_dF, SUM(c2dm) AS sum_c2dm,
                   SUM(balance_error) AS sum_balance_error
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """, [files_ledger, int(win_lo), int(win_hi)]).df()
        st.dataframe(df)

# ---- Events ----
with tab_events:
    st.subheader("Events in window")
    files = get_files(prefix, run_id, "events")
    if not files:
        st.info("No events_* shards yet.")
    else:
        df = con.execute("""
            SELECT step, conn_id, dm, dF, w_sel, A_sel, phi_sel, T_eff
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """, [files, int(win_lo), int(win_hi)]).df()
        st.write(f"{len(df):,} rows")
        st.dataframe(df.head(500))
        if not df.empty:
            df2 = con.execute("""
                SELECT step, SUM(dm) AS dm_sum
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
                GROUP BY step ORDER BY step
            """, [files, int(win_lo), int(win_hi)]).df()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df2["step"], y=df2["dm_sum"], mode="lines", name="Σ dm"))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                              xaxis_title="step", yaxis_title="sum(dm)")
            st.plotly_chart(fig, use_container_width=True)

# ---- Spectra ----
with tab_spectra:
    st.subheader("Spectral selection vs. step")
    files = get_files(prefix, run_id, "spectra")
    if not files:
        st.info("No spectra_* shards yet.")
    else:
        df = con.execute("""
            SELECT step, w_dom, A_dom, phi_dom, AVG(F_local) AS F_mean
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
            GROUP BY step, w_dom, A_dom, phi_dom
            ORDER BY step
        """, [files, int(win_lo), int(win_hi)]).df()
        st.dataframe(df.head(300))
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["step"], y=df["F_mean"], mode="lines+markers", name="F_mean"))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                              xaxis_title="step", yaxis_title="F_mean")
            st.plotly_chart(fig, use_container_width=True)

# ---- World 3D ----
with tab_world3d:
    st.subheader("3D world (env & state) — 1D lattice embedded in 3D")
    files_env   = get_files(prefix, run_id, "env")
    files_state = get_files(prefix, run_id, "state")

    env_df = pd.DataFrame()
    if files_env:
        env_df = con.execute("""
            SELECT step, t, x, y, z, value
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """, [files_env, int(win_lo), int(win_hi)]).df()

    state_df = pd.DataFrame()
    if files_state:
        state_df = con.execute("""
            SELECT step, x, y, z, m, T_eff
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """, [files_state, int(win_lo), int(win_hi)]).df()

    if env_df.empty and state_df.empty:
        st.info("No env/state data in the selected window.")
    else:
        fig = go.Figure()
        if not env_df.empty:
            fig.add_trace(go.Scatter3d(
                x=env_df["x"], y=env_df["y"], z=env_df["z"],
                mode="markers", marker=dict(size=2, opacity=0.45),
                name="env(value)"
            ))
        if not state_df.empty:
            size = 4 + 6*(state_df["m"] / max(1e-9, float(state_df["m"].max())))
            fig.add_trace(go.Scatter3d(
                x=state_df["x"], y=state_df["y"], z=state_df["z"],
                mode="markers",
                marker=dict(size=size, color=state_df["T_eff"], colorscale="Viridis", showscale=True),
                name="state(m,T)"
            ))
        fig.update_layout(height=600, scene=dict(aspectmode="data"),
                          margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---- Edges 3D ----
with tab_edges3d:
    st.subheader("Catalyst edges (transfers)")
    files_edges = get_files(prefix, run_id, "edges")
    files_state = get_files(prefix, run_id, "state")

    if not files_edges:
        st.info("No edges_* shards yet.")
    else:
        edges_df = con.execute("""
            SELECT step, src_conn, dst_conn, weight, t, x AS xs, y AS ys, z AS zs
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """, [files_edges, int(win_lo), int(win_hi)]).df()

        if files_state:
            sd = con.execute("""
                SELECT step, conn_id, x AS xd, y AS yd, z AS zd
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_state, int(win_lo), int(win_hi)]).df()
        else:
            sd = pd.DataFrame()

        if edges_df.empty or sd.empty:
            st.info("Edges exist but insufficient state to map coordinates in this window.")
        else:
            sd_latest = sd.sort_values("step").groupby("conn_id").tail(1).set_index("conn_id")
            seg_rows = []
            for _, r in edges_df.iterrows():
                if r["dst_conn"] in sd_latest.index:
                    dst = sd_latest.loc[r["dst_conn"]]
                    seg_rows.append((
                        float(r["xs"]), float(r["ys"]), float(r["zs"]),
                        float(dst["xd"]), float(dst["yd"]), float(dst["zd"]),
                        float(r["weight"])
                    ))
            if not seg_rows:
                st.info("No mappable edges in this window.")
            else:
                seg = pd.DataFrame(seg_rows, columns=["x1","y1","z1","x2","y2","z2","w"])
                fig = go.Figure()
                wmax = max(1e-9, float(seg["w"].max()))
                # draw a subset if extremely dense to keep UI responsive
                max_lines = 2000
                draw = seg if len(seg) <= max_lines else seg.sample(max_lines, random_state=0)
                for _, s in draw.iterrows():
                    fig.add_trace(go.Scatter3d(
                        x=[s.x1, s.x2], y=[s.y1, s.y2], z=[s.z1, s.z2],
                        mode="lines", line=dict(width=2 + 6*(s.w/wmax)),
                        showlegend=False
                    ))
                fig.update_layout(height=600, scene=dict(aspectmode="data"),
                                  margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)

# ---- Diagnostics ----
with tab_diag:
    st.subheader("Diagnostics")

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("Rebuild indices from manifest (session-only)"):
            man = http_json(manifest_url(prefix, run_id))
            if not man:
                st.error("Failed to load manifest.json")
            else:
                st.session_state.override_files = manifest_to_https(prefix, man)
                st.success("Using manifest-derived file lists for this session.")
    with c2:
        if st.button("Clear manifest override"):
            st.session_state.override_files = None
            st.info("Cleared override; using on-bucket *_index.json.")
    with c3:
        if st.button("Refresh caches"):
            http_json.clear(); http_text.clear()
            list_runs.clear(); load_index.clear()
            st.success("Cleared cache. Data will be reloaded on next render.")

    st.divider()
    st.subheader("Raw indices / file lists")
    for t in ["events","spectra","state","ledger","edges","env"]:
        files = get_files(prefix, run_id, t)
        st.write(f"**{t}**: {len(files)} files")
        if files:
            st.code("\n".join(files[:10]))

    st.divider()
    st.subheader("Manifest (truncated)")
    mtxt = http_text(manifest_url(prefix, run_id))
    if mtxt: st.code(mtxt[:4000])
    else:    st.info("manifest.json not found (yet).")