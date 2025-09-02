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

# session defaults
st.session_state.setdefault("playing", False)          # server-side play
st.session_state.setdefault("frame_step", 0)
st.session_state.setdefault("override_files", None)    # dict: table -> [urls]
st.session_state.setdefault("active_panel", "World 3D")  # which panel is being rendered

# --------------- Cache helpers (short TTL for "live") ---------------
def _ttl(live: bool) -> int:
    return 5 if live else 30

# ---------------- HTTP helpers ----------------
@st.cache_data(show_spinner=False)
def http_json(url: str, ttl: int) -> Optional[dict]:
    """Cache keyed by URL and TTL value to allow 5s vs 30s modes."""
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

# Live discovery of new shards every ~5s
live_refresh = st.sidebar.toggle("Live refresh (discover new shards)", value=True,
                                 help="Auto cache-bust indices/manifest every 5s")
live_token = int(time.time()) // 5 if live_refresh else None
ttl_val = _ttl(live_refresh)

runs = list_runs(prefix, live_token, ttl_val)
run_id = st.sidebar.selectbox("Run", runs, index=0 if runs else None, placeholder="Select…")
manual_run = st.sidebar.text_input("Or enter run id", value="" if run_id else "")
if not run_id and manual_run:
    run_id = manual_run.strip()
prefer_manifest = st.sidebar.toggle("Prefer manifest when *_index.json is empty or local", value=True)

# Active panel (render exactly one for stability)
panel = st.sidebar.radio("Panel", ["World 3D", "Edges 3D", "Events", "Spectra", "Overview", "Diagnostics"],
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
    fps   = st.slider("FPS", 1, 30, 12)
    stride = st.number_input("Δstep per frame", min_value=1,
                             value=max(1, (int(step_max)-int(step_min))//300 or 1), step=1)
with c4:
    label = "⏵ Play" if not st.session_state.playing else "⏸ Pause"
    if st.button(label, use_container_width=True):
        st.session_state.playing = not st.session_state.playing

frame_lo, frame_hi = int(step_min), int(step_max)

# Server-driven animation: bump frame and rerun only when playing
if st.session_state.playing:
    time.sleep(1.0/max(1,fps))
    nxt = int(st.session_state.frame_step) + int(stride)
    if nxt > frame_hi:
        nxt = frame_lo
    st.session_state.frame_step = int(nxt)
    # soft auto-refresh to catch new shards / update panel
    if live_refresh:
        st.rerun()   # <-- FIX: use st.rerun() (experimental_rerun removed)

frame = int(st.session_state.frame_step)

st.caption(f"Range: [{frame_lo}, {frame_hi}] | fps={fps} | stride={stride} | panel={panel} | live={live_refresh}")

# ---------- Shared helpers (downsampling & safe caps) ----------
MAX_POINTS_PER_FRAME = 40000
MAX_EDGES_PER_FRAME  = 4000

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
    try:
        df = query_df("""
            SELECT step
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """, [files_ev, int(frame_lo), int(frame_hi)])
        if df.empty:
            ph_table.info("No events in range."); return
        cnt = df.groupby("step", as_index=False).size().rename(columns={"size":"events"})
        # show data up to *current* frame for movie feel
        cur = cnt[cnt["step"] <= max(frame, frame_lo)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cur["step"], y=cur["events"], mode="lines", name="# events"))
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                          xaxis_title="step", yaxis_title="# events")
        ph_chart.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Legacy fallback
        try:
            df = query_df("""
                SELECT step, dm
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_ev, int(frame_lo), int(frame_hi)])
            if df.empty:
                ph_table.info("No readable events in either schema for this range."); return
            agg = df.groupby("step", as_index=False)["dm"].sum().rename(columns={"dm":"dm_sum"})
            cur = agg[agg["step"] <= max(frame, frame_lo)]
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
    try:
        sdf = query_df("""
            SELECT step, counts, bin_edges
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
            ORDER BY step
        """, [files_sp, int(frame_lo), int(frame_hi)])
        if sdf.empty or "counts" not in sdf.columns:
            raise RuntimeError("No counts column")
        # show the histogram for the *current* frame only
        row = sdf[sdf["step"] == max(frame, frame_lo)]
        if row.empty:
            ph_table.info("No spectrum at current frame."); return
        counts = list(row.iloc[0]["counts"] or [])
        edges  = list(row.iloc[0]["bin_edges"] or [])
        if edges and len(edges) >= 2:
            e = np.array(edges, dtype=float)
            x_bins = ((e[:-1] + e[1:]) / 2.0).tolist()
        else:
            x_bins = list(range(len(counts)))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_bins, y=counts, name=f"step {int(row.iloc[0]['step'])}"))
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=30,b=10),
                          xaxis_title="bin", yaxis_title="count")
        ph_chart.plotly_chart(fig, use_container_width=True)
    except Exception:
        # Legacy fallback: running mean plot up to current frame
        try:
            df = query_df("""
                SELECT step, AVG(F_local) AS F_mean
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
                GROUP BY step ORDER BY step
            """, [files_sp, int(frame_lo), int(frame_hi)])
            cur = df[df["step"] <= max(frame, frame_lo)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cur["step"], y=cur["F_mean"], mode="lines+markers", name="F_mean"))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                              xaxis_title="step", yaxis_title="F_mean")
            ph_chart.plotly_chart(fig, use_container_width=True)
        except Exception:
            ph_table.info("No readable spectra in either schema for this range.")

def render_world3d():
    files_state = get_files(prefix, run_id, "state", prefer_manifest, live_token, ttl_val)
    if not files_state:
        ph_table.info("No state_* shards yet."); return
    # Read ONLY current frame’s points
    df = pd.DataFrame()
    try:
        df = query_df("""
            SELECT step, x, y, z, value
            FROM read_parquet(?)
            WHERE step = ?
        """, [files_state, int(max(frame, frame_lo))])
        if df.empty or not {"x","y","z","value"}.issubset(df.columns):
            raise RuntimeError("try legacy")
    except Exception:
        try:
            df = query_df("""
                SELECT step, x, y, z, m AS value
                FROM read_parquet(?)
                WHERE step = ?
            """, [files_state, int(max(frame, frame_lo))])
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        ph_table.info("No state at current frame."); return
    df = cap_points(df, MAX_POINTS_PER_FRAME)
    vmax = max(1e-9, float(df["value"].max()))
    size = 3 + 6*(df["value"]/vmax)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"],
        mode="markers",
        marker=dict(size=size, color=df["value"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="value")),
        name="state(value)"
    ))
    fig.update_layout(height=600, scene=dict(aspectmode="data"),
                      margin=dict(l=10,r=10,t=30,b=10))
    ph_chart.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def render_edges3d():
    files_edges = get_files(prefix, run_id, "edges", prefer_manifest, live_token, ttl_val)
    if not files_edges:
        ph_table.info("No edges_* shards yet."); return
    # Read ONLY current frame’s edges
    drawn = False
    try:
        edf = query_df("""
            SELECT x0, y0, z0, x1, y1, z1, v0
            FROM read_parquet(?)
            WHERE step = ?
        """, [files_edges, int(max(frame, frame_lo))])
        if not edf.empty and {"x0","y0","z0","x1","y1","z1"}.issubset(edf.columns):
            if len(edf) > MAX_EDGES_PER_FRAME:
                edf = edf.sample(MAX_EDGES_PER_FRAME, random_state=0)
            # pack into polyline with None separators
            x = []; y = []; z = []
            for _, r in edf.iterrows():
                x += [r["x0"], r["x1"], None]
                y += [r["y0"], r["y1"], None]
                z += [r["z0"], r["z1"], None]
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines",
                                       line=dict(width=3.0), showlegend=False))
            fig.update_layout(height=600, scene=dict(aspectmode="data"),
                              margin=dict(l=10,r=10,t=30,b=10))
            ph_chart.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            drawn = True
    except Exception:
        pass

    if not drawn:
        # Legacy fallback: need state to map conn ids → coords
        try:
            files_state = get_files(prefix, run_id, "state", prefer_manifest, live_token, ttl_val)
            if not files_state:
                ph_table.info("State shards unavailable for legacy edges mapping."); return
            e = query_df("""
                SELECT src_conn, dst_conn, weight, t, x AS xs, y AS ys, z AS zs
                FROM read_parquet(?)
                WHERE step = ?
            """, [files_edges, int(max(frame, frame_lo))])
            s = query_df("""
                SELECT conn_id, x AS xd, y AS yd, z AS zd
                FROM read_parquet(?)
                WHERE step = ?
            """, [files_state, int(max(frame, frame_lo))])
            if e.empty or s.empty:
                ph_table.info("No edges/state at current frame."); return
            s = s.set_index("conn_id")
            rows = []
            for _, r in e.iterrows():
                if r["dst_conn"] in s.index:
                    dst = s.loc[r["dst_conn"]]
                    rows.append((float(r["xs"]), float(r["ys"]), float(r["zs"]),
                                 float(dst["xd"]), float(dst["yd"]), float(dst["zd"])))
            if not rows:
                ph_table.info("No mappable edges at current frame."); return
            seg = pd.DataFrame(rows, columns=["x1","y1","z1","x2","y2","z2"])
            if len(seg) > MAX_EDGES_PER_FRAME:
                seg = seg.sample(MAX_EDGES_PER_FRAME, random_state=0)
            x = []; y = []; z = []
            for _, srow in seg.iterrows():
                x += [srow.x1, srow.x2, None]
                y += [srow.y1, srow.y2, None]
                z += [srow.z1, srow.z2, None]
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines",
                                       line=dict(width=3.0), showlegend=False))
            fig.update_layout(height=600, scene=dict(aspectmode="data"),
                              margin=dict(l=10,r=10,t=30,b=10))
            ph_chart.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            ph_table.info("No readable edges in either schema at current frame.")

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