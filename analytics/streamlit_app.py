# analytics/streamlit_app.py
import json, time, urllib.request, math
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

# Session defaults
st.session_state.setdefault("playing", False)
st.session_state.setdefault("frame_step", 0)
st.session_state.setdefault("override_files", None)    # dict: table -> [urls]
st.session_state.setdefault("active_panel", "World 3D")

# ---------------- Tunables & caps ----------------
DEFAULT_FPS = 6
MAX_POINTS_PER_FRAME = 20000
MAX_EDGES_PER_FRAME  = 3000
WORLD_TRAIL_MAX      = 8   # how many previous steps to overlay (tiny to keep light)

# --------------- Cache helpers (short TTL for "live") ---------------
def _ttl(live: bool) -> int:
    return 5 if live else 30

# ---------------- HTTP helpers ----------------
@st.cache_data(show_spinner=False)
def http_json(url: str, ttl: int) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

@st.cache_data(show_spinner=False)
def http_text(url: str, ttl: int) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

def _with_buster(url: str, live_token: Optional[int]) -> str:
    if not live_token:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={live_token}"

def runs_index_url(prefix: str, live_token: Optional[int]) -> str:
    return _with_buster(f"{prefix}/runs/index.json", live_token)

def table_index_url(prefix: str, run_id: str, table: str, live_token: Optional[int]) -> str:
    return _with_buster(f"{prefix}/runs/{run_id}/shards/{table}_index.json", live_token)

def manifest_url(prefix: str, run_id: str, live_token: Optional[int]) -> str:
    return _with_buster(f"{prefix}/runs/{run_id}/manifest.json", live_token)

def manifest_to_https(prefix: str, manifest: dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for s in manifest.get("shards", []):
        t = s.get("table"); p = s.get("path","")
        if not t or not p:
            continue
        if p.startswith("data/"):
            p = p[len("data/"):]
        url = f"{prefix}/{p}"
        if url.endswith(".parquet"):
            out.setdefault(t, []).append(url)
    for k in list(out.keys()):
        out[k].sort()
    return out

@st.cache_data(show_spinner=False)
def list_runs(prefix: str, live_token: Optional[int], ttl: int) -> List[str]:
    data = http_json(runs_index_url(prefix, live_token), ttl)
    if data and "runs" in data:
        return sorted({str(x) for x in data["runs"]})
    return []

@st.cache_data(show_spinner=False)
def load_index(prefix: str, run_id: str, table: str, live_token: Optional[int], ttl: int) -> List[str]:
    data = http_json(table_index_url(prefix, run_id, table, live_token), ttl)
    if not data:
        return []
    return data.get("files", []) or []

def _looks_http(x: str) -> bool:
    return isinstance(x, str) and (x.startswith("http://") or x.startswith("https://"))

def get_files(prefix: str, run_id: str, table: str, prefer_manifest: bool,
              live_token: Optional[int], ttl: int) -> List[str]:
    """
    Returns a list of Parquet HTTPS URLs for the given table.
    Order of precedence:
      1) per-session override from Diagnostics ("Rebuild from manifest")
      2) *_index.json IF it exists AND all entries are HTTP(S)
      3) manifest.json (converted to HTTPS)
    """
    ov = st.session_state.get("override_files")
    if isinstance(ov, dict) and table in ov:
        return ov[table]

    files = load_index(prefix, run_id, table, live_token, ttl)
    if files and all(_looks_http(f) for f in files):
        return files

    if prefer_manifest or (not files) or any(not _looks_http(f) for f in files):
        man = http_json(manifest_url(prefix, run_id, live_token), ttl)
        if man:
            by_table = manifest_to_https(prefix, man)
            mf = by_table.get(table, [])
            if mf:
                return mf
    return files  # may be empty

# ---------------- DuckDB helper ----------------
@st.cache_resource
def connect_duckdb() -> ddb.DuckDBPyConnection:
    con = ddb.connect(":memory:")
    try:
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
    except Exception:
        pass
    return con

def query_df(sql: str, params: list) -> pd.DataFrame:
    con = connect_duckdb()
    return con.execute(sql, params).df()

# ---------------- Sidebar ----------------
st.sidebar.title("Fuka 4.0 Analytics")
prefix = st.sidebar.text_input("Data URL prefix", PREFIX_DEFAULT)

# Live discovery of new shards every ~10s to keep load low
live_refresh = st.sidebar.toggle("Live refresh (discover new shards)", value=True,
                                 help="Auto cache-bust indices/manifest every ~10s")
live_token = int(time.time()) // 10 if live_refresh else None
ttl_val = _ttl(live_refresh)

runs = list_runs(prefix, live_token, ttl_val)
run_id = st.sidebar.selectbox("Run", runs, index=0 if runs else None, placeholder="Select…")
manual_run = st.sidebar.text_input("Or enter run id", value="" if run_id else "")
if not run_id and manual_run:
    run_id = manual_run.strip()
prefer_manifest = st.sidebar.toggle("Prefer manifest when *_index.json is empty or local", value=True)

# Active panel (render one at a time for stability)
panel = st.sidebar.radio("Panel",
                         ["World 3D", "Edges 3D", "Events", "Spectra", "Overview", "Diagnostics"],
                         index=["World 3D","Edges 3D","Events","Spectra","Overview","Diagnostics"].index(
                             st.session_state.get("active_panel","World 3D")))
st.session_state.active_panel = panel

if not run_id:
    st.stop()

# ---------------- Header + animation controls ----------------
st.title(f"Fuka 4.0 — Run: {run_id}")

@st.cache_data(show_spinner=False)
def discover_step_bounds(prefix: str, run_id: str, prefer_manifest: bool,
                         live_token: Optional[int], ttl: int) -> Tuple[int,int]:
    lo, hi = math.inf, -math.inf
    con = connect_duckdb()
    for table in ("ledger","events","state"):
        files = get_files(prefix, run_id, table, prefer_manifest, live_token, ttl)
        if not files:
            continue
        try:
            df = con.execute("SELECT MIN(step) lo, MAX(step) hi FROM read_parquet(?)", [files]).df()
            if not df.empty and pd.notnull(df.iloc[0]["lo"]) and pd.notnull(df.iloc[0]["hi"]):
                lo = min(lo, int(df.iloc[0]["lo"]))
                hi = max(hi, int(df.iloc[0]["hi"]))
        except Exception:
            pass
    if lo is math.inf or hi is -math.inf:
        return 0, 10_000
    lo = max(0, lo)
    hi = max(lo+1, hi)
    return lo, hi

lo_guess, hi_guess = discover_step_bounds(prefix, run_id, prefer_manifest, live_token, ttl_val)
if lo_guess >= hi_guess:
    hi_guess = lo_guess + 1

c1,c2,c3,c4 = st.columns([1,1,1,2])
with c1:
    step_min = st.number_input("Step min", min_value=0, value=int(lo_guess), step=100)
with c2:
    step_max = st.number_input("Step max", min_value=max(1, step_min+1), value=int(max(hi_guess, step_min+1)), step=500)
with c3:
    fps   = st.slider("FPS", 1, 20, DEFAULT_FPS, help="Lower is lighter; 6 FPS is a good rough movie")
    stride = st.number_input("Δstep per frame", min_value=1,
                             value=max(1, (int(step_max)-int(step_min))//400 or 1), step=1,
                             help="How many simulation steps to advance per frame")
with c4:
    label = "⏵ Play" if not st.session_state.playing else "⏸ Pause"
    if st.button(label, use_container_width=True):
        st.session_state.playing = not st.session_state.playing

# Optional trail for World 3D (kept tiny)
trail_frames = st.sidebar.slider("World 3D trail (frames)", 0, WORLD_TRAIL_MAX, 2,
                                 help="Overlay a few recent frames with fading opacity")

frame_lo, frame_hi = int(step_min), int(step_max)

# Server-driven animation: bump frame and rerun ALWAYS when playing
if st.session_state.playing:
    time.sleep(1.0/max(1,fps))
    nxt = int(st.session_state.frame_step) + int(stride)
    if nxt > frame_hi:
        nxt = frame_lo
    st.session_state.frame_step = int(nxt)
    st.rerun()

# Frame, window, HUD
frame = int(st.session_state.frame_step)
frame = max(frame_lo, min(frame_hi, frame))
st.session_state.frame_step = frame

hud1, hud2 = st.columns([3,1])
with hud1:
    st.markdown(f"### ⏱️ Current step: **{frame}**  &nbsp;&nbsp;|&nbsp;&nbsp; FPS **{fps}**  |  Stride **{stride}**")
with hud2:
    pct = 0.0 if frame_hi == frame_lo else (frame - frame_lo)/(frame_hi - frame_lo)
    st.progress(pct, text=f"{int(100*pct)}% through range")

# ---------------- DuckDB ----------------
con = connect_duckdb()

# ---------- Shared helpers ----------
def cap_points(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) > max_points:
        return df.sample(int(max_points), random_state=0)
    return df

# ---------------- Single-panel placeholders ----------------
ph_chart = st.empty()
ph_table = st.empty()

# ---------------- PANEL RENDERERS ----------------
def render_overview():
    files_ledger = get_files(prefix, run_id, "ledger", prefer_manifest, live_token, ttl_val)
    files_state  = get_files(prefix, run_id, "state",  prefer_manifest, live_token, ttl_val)
    files_events = get_files(prefix, run_id, "events", prefer_manifest, live_token, ttl_val)

    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("Ledger shards", len(files_ledger))
    with k2: st.metric("State shards",  len(files_state))
    with k3: st.metric("Event shards",  len(files_events))
    mtxt = http_text(manifest_url(prefix, run_id, live_token), ttl_val)
    if mtxt:
        try:
            man = json.loads(mtxt)
            with k4: st.metric("Manifest entries", len(man.get("shards",[])))
        except Exception:
            with k4: st.metric("Manifest entries", 0)
    else:
        with k4: st.metric("Manifest entries", 0)

    if files_ledger:
        try:
            df = query_df("""
                SELECT MIN(step) AS min_step, MAX(step) AS max_step,
                       SUM(sum_energy) AS sum_energy,
                       AVG(mean_energy) AS mean_energy,
                       MAX(max_energy)  AS max_energy,
                       AVG(std_energy)  AS std_energy,
                       SUM(cat_deposit) AS sum_cat_deposit
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_ledger, int(frame_lo), int(frame_hi)])
        except Exception:
            df = query_df("""
                SELECT MIN(step) AS min_step, MAX(step) AS max_step,
                       SUM(dF) AS sum_dF, SUM(c2dm) AS sum_c2dm,
                       SUM(balance_error) AS sum_balance_error
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_ledger, int(frame_lo), int(frame_hi)])
        ph_table.dataframe(df)

def render_events():
    files_ev = get_files(prefix, run_id, "events", prefer_manifest, live_token, ttl_val)
    if not files_ev:
        ph_table.info("No events_* shards yet."); return
    # count events per step, but only plot up to current frame
    try:
        df = query_df("""
            SELECT step
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """, [files_ev, int(frame_lo), int(frame_hi)])
        if df.empty:
            ph_table.info("No events in range."); return
        cnt_all = df.groupby("step", as_index=False).size().rename(columns={"size":"events"})
        cur = cnt_all[cnt_all["step"] <= frame]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cur["step"], y=cur["events"], mode="lines", name="# events"))
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                          xaxis_title="step", yaxis_title="# events")
        ph_chart.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Legacy fallback: sum dm
        try:
            df = query_df("""
                SELECT step, dm
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_ev, int(frame_lo), int(frame_hi)])
            if df.empty:
                ph_table.info("No readable events in either schema for this range."); return
            agg = df.groupby("step", as_index=False)["dm"].sum().rename(columns={"dm":"dm_sum"})
            cur = agg[agg["step"] <= frame]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cur["step"], y=cur["dm_sum"], mode="lines", name="Σ dm"))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                              xaxis_title="step", yaxis_title="sum(dm)")
            ph_chart.plotly_chart(fig, use_container_width=True)
        except Exception:
            ph_table.info("No readable events in either schema for this range.")

def render_spectra():
    files_sp = get_files(prefix, run_id, "spectra", prefer_manifest, live_token, ttl_val)
    if not files_sp:
        ph_table.info("No spectra_* shards yet."); return
    # Show histogram for the *current* step (movie feel)
    try:
        sdf = query_df("""
            SELECT step, counts, bin_edges
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
            ORDER BY step
        """, [files_sp, int(frame_lo), int(frame_hi)])
        if sdf.empty or "counts" not in sdf.columns:
            raise RuntimeError("No counts column")
        row = sdf[sdf["step"] == frame]
        if row.empty:
            ph_table.info("No spectrum at current step."); return
        counts = list(row.iloc[0]["counts"] or [])
        edges  = list(row.iloc[0]["bin_edges"] or [])
        if edges and len(edges) >= 2:
            e = np.array(edges, dtype=float)
            x_bins = ((e[:-1] + e[1:]) / 2.0).tolist()
        else:
            x_bins = list(range(len(counts)))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_bins, y=counts, name=f"step {frame}"))
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10),
                          xaxis_title="bin", yaxis_title="count")
        ph_chart.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Legacy fallback: running mean up to current step
        try:
            df = query_df("""
                SELECT step, AVG(F_local) AS F_mean
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
                GROUP BY step ORDER BY step
            """, [files_sp, int(frame_lo), int(frame_hi)])
            cur = df[df["step"] <= frame]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cur["step"], y=cur["F_mean"], mode="lines+markers", name="F_mean"))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                              xaxis_title="step", yaxis_title="F_mean")
            ph_chart.plotly_chart(fig, use_container_width=True)
        except Exception:
            ph_table.info("No readable spectra in either schema for this range.")

def _read_state_at(step_val: int, files_state: List[str]) -> pd.DataFrame:
    # Try new schema
    df = query_df("""
        SELECT step, x, y, z, value
        FROM read_parquet(?)
        WHERE step = ?
    """, [files_state, int(step_val)])
    if not df.empty and {"x","y","z","value"}.issubset(df.columns):
        return df
    # Legacy
    df = query_df("""
        SELECT step, x, y, z, m AS value
        FROM read_parquet(?)
        WHERE step = ?
    """, [files_state, int(step_val)])
    return df

def render_world3d():
    files_state = get_files(prefix, run_id, "state", prefer_manifest, live_token, ttl_val)
    if not files_state:
        ph_table.info("No state_* shards yet."); return

    # Current frame cloud
    df0 = _read_state_at(frame, files_state)
    if df0.empty:
        ph_table.info("No state at current step."); return
    df0 = cap_points(df0, MAX_POINTS_PER_FRAME)

    # Optional small trail: fetch a few previous steps (lightweight)
    trail_steps = []
    if trail_frames > 0:
        for k in range(1, trail_frames+1):
            s = frame - k*int(stride)
            if s < frame_lo:
                break
            trail_steps.append(s)

    # Read trail frames (sampled, fewer points)
    trail_chunks = []
    per_trail = max(5000, MAX_POINTS_PER_FRAME // 8)
    for s in trail_steps:
        dfi = _read_state_at(s, files_state)
        if dfi.empty:
            continue
        dfi = cap_points(dfi, per_trail)
        dfi["__age"] = frame - s  # 1 = newest older frame
        trail_chunks.append(dfi)
    trail_df = pd.concat(trail_chunks, ignore_index=True) if trail_chunks else pd.DataFrame()

    # Build plot
    vmax = max(1e-9, float(df0["value"].max()))
    fig = go.Figure()

    # Current frame points
    size0 = 3 + 6*(df0["value"]/vmax)
    fig.add_trace(go.Scatter3d(
        x=df0["x"], y=df0["y"], z=df0["z"],
        mode="markers",
        marker=dict(size=size0, color=df0["value"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="value")),
        name=f"state(step={frame})"
    ))

    # Trail points with fading opacity (no colorbar)
    if not trail_df.empty:
        # normalize opacity by age (more recent = more opaque)
        ages = trail_df["__age"].astype(float).to_numpy()
        if len(ages) > 0:
            # map age 1..trail_frames -> opacity 0.5..0.15
            op = 0.5 - (ages-1) * (0.35/max(1.0, float(trail_frames)))
            op = np.clip(op, 0.15, 0.5)
        else:
            op = 0.2
        size_t = 2 + 4*(trail_df["value"]/vmax)
        fig.add_trace(go.Scatter3d(
            x=trail_df["x"], y=trail_df["y"], z=trail_df["z"],
            mode="markers",
            marker=dict(size=size_t, color=trail_df["value"], colorscale="Viridis",
                        showscale=False, opacity=float(np.mean(op)) if np.ndim(op)==1 else 0.2),
            name=f"trail({len(trail_steps)} frames)"
        ))

    fig.update_layout(height=650, scene=dict(aspectmode="data"),
                      margin=dict(l=10,r=10,t=30,b=10))
    ph_chart.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def render_edges3d():
    files_edges = get_files(prefix, run_id, "edges", prefer_manifest, live_token, ttl_val)
    if not files_edges:
        ph_table.info("No edges_* shards yet."); return
    # Current frame only (edges are heavy)
    drawn = False
    try:
        edf = query_df("""
            SELECT x0, y0, z0, x1, y1, z1, v0
            FROM read_parquet(?)
            WHERE step = ?
        """, [files_edges, int(frame)])
        if not edf.empty and {"x0","y0","z0","x1","y1","z1"}.issubset(edf.columns):
            if len(edf) > MAX_EDGES_PER_FRAME:
                edf = edf.sample(MAX_EDGES_PER_FRAME, random_state=0)
            x = []; y = []; z = []
            for _, r in edf.iterrows():
                x += [r["x0"], r["x1"], None]
                y += [r["y0"], r["y1"], None]
                z += [r["z0"], r["z1"], None]
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines",
                                       line=dict(width=3.0), showlegend=False))
            fig.update_layout(height=650, scene=dict(aspectmode="data"),
                              margin=dict(l=10,r=10,t=30,b=10))
            ph_chart.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            drawn = True
    except Exception:
        pass

    if not drawn:
        # Legacy fallback (needs state coords; we skip here for lightness)
        ph_table.info("No readable edges in 3D schema at current step.")

def render_diag():
    st.subheader("Diagnostics")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("Rebuild indices from manifest (session-only)"):
            m = http_json(manifest_url(prefix, run_id, live_token), ttl_val)
            if not m:
                st.error("Failed to load manifest.json")
            else:
                st.session_state.override_files = manifest_to_https(prefix, m)
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
    for t in ["events","spectra","state","ledger","edges","env","catalysts"]:
        files = get_files(prefix, run_id, t, prefer_manifest, live_token, ttl_val)
        st.write(f"**{t}**: {len(files)} files")
        if files:
            st.code("\n".join(files[:10]))

    st.divider()
    st.subheader("Manifest (truncated)")
    mtxt = http_text(manifest_url(prefix, run_id, live_token), ttl_val)
    if mtxt: st.code(mtxt[:4000])
    else:    st.info("manifest.json not found (yet).")

# ---------------- Render selected panel only ----------------
if panel == "Overview":
    render_overview()
elif panel == "World 3D":
    render_world3d()
elif panel == "Edges 3D":
    render_edges3d()
elif panel == "Events":
    render_events()
elif panel == "Spectra":
    render_spectra()
else:
    render_diag()