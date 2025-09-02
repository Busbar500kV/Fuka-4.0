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
st.session_state.setdefault("playing", False)          # server-side play (used only when client_anim=False)
st.session_state.setdefault("frame_step", 0)
st.session_state.setdefault("override_files", None)    # dict: table -> [urls]

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
    Returns a list of Parquet URLs for the given table.
    Order of precedence:
      1) per-session override from Diagnostics ("Rebuild from manifest")
      2) *_index.json IF it exists AND all entries are HTTP(S)
      3) manifest.json (converted to HTTPS)
    """
    # 1) explicit override (session-only)
    ov = st.session_state.get("override_files")
    if isinstance(ov, dict) and table in ov:
        return ov[table]

    # 2) try index
    files = load_index(prefix, run_id, table, live_token, ttl)
    if files and all(_looks_http(f) for f in files):
        return files

    # 3) fallback to manifest when index is empty or contains local paths
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

if not run_id:
    st.stop()

# ---------------- Header + animation controls ----------------
st.title(f"Fuka 4.0 — Run: {run_id}")

# Use client-side animation to avoid page reruns/flicker
client_anim = st.checkbox("Client-side animation (Plotly frames)", value=True,
                          help="Plays fully in the browser (smooth). Turn off to use server-side reruns.")

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
    fps   = st.slider("FPS", 1, 30, 10)
    stride = st.number_input("Frame stride (Δstep per frame)", min_value=1,
                             value=max(1, (int(step_max)-int(step_min))//200 or 1), step=1)
with c4:
    # server-side Play only used if client_anim=False
    label = "⏵ Play (server)" if not st.session_state.playing else "⏸ Pause (server)"
    if st.button(label, use_container_width=True):
        st.session_state.playing = not st.session_state.playing

# animate by bumping frame then rerunning (server-driven only)
frame_lo, frame_hi = int(step_min), int(step_max)
if (not client_anim) and st.session_state.playing:
    time.sleep(1.0/max(1,fps))
    nxt = int(st.session_state.frame_step) + int(stride)
    if nxt > frame_hi:
        nxt = frame_lo
    st.session_state.frame_step = int(nxt)
    st.rerun()
frame = int(st.session_state.frame_step)
win_lo = max(frame_lo, frame - int(0))  # window defined by min/max; we still keep vars for compat
win_hi = min(frame_hi, frame)

st.caption(f"Range: [{frame_lo}, {frame_hi}] | fps={fps} | stride={stride} | client_anim={client_anim} | live={live_refresh}")

# ---------------- DuckDB ----------------
con = connect_duckdb()

# ---------------- Tabs ----------------
tab_overview, tab_events, tab_spectra, tab_world3d, tab_edges3d, tab_diag = st.tabs(
    ["Overview", "Events (animated)", "Spectra (animated)", "World 3D (animated)", "Edges 3D (animated)", "Diagnostics"]
)

# ---- Overview ----
with tab_overview:
    st.subheader("Aggregates")
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
        # Try new 3D ledger schema first; fallback to legacy
        try:
            df = con.execute("""
                SELECT MIN(step) AS min_step, MAX(step) AS max_step,
                       SUM(sum_energy) AS sum_energy,
                       AVG(mean_energy) AS mean_energy,
                       MAX(max_energy)  AS max_energy,
                       AVG(std_energy)  AS std_energy,
                       SUM(cat_deposit) AS sum_cat_deposit
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_ledger, int(frame_lo), int(frame_hi)]).df()
        except Exception:
            df = con.execute("""
                SELECT MIN(step) AS min_step, MAX(step) AS max_step,
                       SUM(dF) AS sum_dF, SUM(c2dm) AS sum_c2dm,
                       SUM(balance_error) AS sum_balance_error
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_ledger, int(frame_lo), int(frame_hi)]).df()
        st.dataframe(df)

# ---- Events (animated) ----
with tab_events:
    st.subheader("Events per step")
    files_ev = get_files(prefix, run_id, "events", prefer_manifest, live_token, ttl_val)
    if not files_ev:
        st.info("No events_* shards yet.")
    else:
        # Prefer new schema (step,x,y,z,value); fallback legacy (step, dm, ...)
        done = False
        try:
            df = con.execute("""
                SELECT step
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_ev, int(frame_lo), int(frame_hi)]).df()
            if not df.empty:
                cnt = df.groupby("step", as_index=False).size().rename(columns={"size":"events"})
                steps_all = sorted(cnt["step"].tolist())
                # Build frames with growing time-series
                frames_seq = steps_all[::max(1, int(stride))] or steps_all
                if frames_seq and frames_seq[-1] != steps_all[-1]:
                    frames_seq.append(steps_all[-1])

                # create cumulative y for each frame
                def _series_upto(s):
                    m = cnt[cnt["step"] <= s]
                    return m["step"].tolist(), m["events"].tolist()

                x0,y0 = _series_upto(frames_seq[0])
                fig = go.Figure(
                    data=[go.Scatter(x=x0, y=y0, mode="lines", name="# events")],
                    layout=go.Layout(
                        height=300, margin=dict(l=10,r=10,t=30,b=10),
                        xaxis_title="step", yaxis_title="# events",
                        updatemenus=[dict(
                            type="buttons", showactive=True,
                            x=0.0, y=1.15, xanchor="left", yanchor="top",
                            buttons=[
                                dict(label="▶ Play", method="animate",
                                     args=[None, {"frame": {"duration": int(1000/max(1,fps))},
                                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                                dict(label="⏸ Pause", method="animate",
                                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                            ]
                        )],
                        sliders=[dict(
                            active=0, x=0.1, y=0.05, xanchor="left", len=0.8,
                            currentvalue=dict(prefix="step: "),
                            steps=[dict(label=str(s), method="animate",
                                        args=[[str(s)], {"mode": "immediate",
                                                         "frame": {"duration": 0}, "transition": {"duration": 0}}])
                                   for s in frames_seq]
                        )]
                    ),
                    frames=[
                        go.Frame(
                            name=str(s),
                            data=[go.Scatter(x=_series_upto(s)[0], y=_series_upto(s)[1], mode="lines")]
                        ) for s in frames_seq
                    ]
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                done = True
        except Exception:
            pass

        if not done:
            try:
                df = con.execute("""
                    SELECT step, dm
                    FROM read_parquet(?)
                    WHERE step BETWEEN ? AND ?
                """, [files_ev, int(frame_lo), int(frame_hi)]).df()
                if df.empty:
                    st.info("No readable events in either schema for this range.")
                else:
                    cnt = df.groupby("step", as_index=False)["dm"].sum().rename(columns={"dm":"dm_sum"})
                    steps_all = sorted(cnt["step"].tolist())
                    frames_seq = steps_all[::max(1, int(stride))] or steps_all
                    if frames_seq and frames_seq[-1] != steps_all[-1]:
                        frames_seq.append(steps_all[-1])
                    def _series_upto(s):
                        m = cnt[cnt["step"] <= s]
                        return m["step"].tolist(), m["dm_sum"].tolist()
                    x0,y0 = _series_upto(frames_seq[0])
                    fig = go.Figure(
                        data=[go.Scatter(x=x0, y=y0, mode="lines", name="Σ dm")],
                        layout=go.Layout(
                            height=300, margin=dict(l=10,r=10,t=30,b=10),
                            xaxis_title="step", yaxis_title="sum(dm)",
                            updatemenus=[dict(type="buttons", showactive=True, x=0.0, y=1.15,
                                              xanchor="left", yanchor="top",
                                              buttons=[dict(label="▶ Play", method="animate",
                                                            args=[None, {"frame": {"duration": int(1000/max(1,fps))},
                                                                         "fromcurrent": True, "transition": {"duration": 0}}]),
                                                       dict(label="⏸ Pause", method="animate",
                                                            args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
                            sliders=[dict(active=0, x=0.1, y=0.05, xanchor="left", len=0.8,
                                          currentvalue=dict(prefix="step: "),
                                          steps=[dict(label=str(s), method="animate",
                                                      args=[[str(s)], {"mode": "immediate",
                                                                       "frame": {"duration": 0}, "transition": {"duration": 0}}])
                                                 for s in frames_seq])]
                        ),
                        frames=[go.Frame(name=str(s),
                                         data=[go.Scatter(x=_series_upto(s)[0], y=_series_upto(s)[1], mode="lines")])
                                for s in frames_seq]
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                st.info("No readable events in either schema for this range.")

# ---- Spectra (animated) ----
with tab_spectra:
    st.subheader("Spectra over time")
    files_sp = get_files(prefix, run_id, "spectra", prefer_manifest, live_token, ttl_val)
    if not files_sp:
        st.info("No spectra_* shards yet.")
    else:
        done = False
        try:
            sdf = con.execute("""
                SELECT step, counts, bin_edges
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
                ORDER BY step
            """, [files_sp, int(frame_lo), int(frame_hi)]).df()
            if not sdf.empty and "counts" in sdf.columns:
                # Collapse DuckDB list columns to Python lists
                def safe_list(x):
                    try:
                        return list(x) if x is not None else []
                    except Exception:
                        return []
                sdf["counts"] = sdf["counts"].map(safe_list)
                if "bin_edges" in sdf.columns:
                    sdf["bin_edges"] = sdf["bin_edges"].map(safe_list)
                    # compute bin centers if possible
                    def centers(edges):
                        edges = list(edges)
                        if len(edges) >= 2:
                            e = np.array(edges, dtype=float)
                            return ((e[:-1] + e[1:]) / 2.0).tolist()
                        return list(range(len(edges)))
                    sdf["x_bins"] = sdf["bin_edges"].map(centers)
                else:
                    # fallback: index the counts
                    sdf["x_bins"] = sdf["counts"].map(lambda c: list(range(len(c))))

                steps_all = sorted(sdf["step"].unique().tolist())
                frames_seq = steps_all[::max(1,int(stride))] or steps_all
                if frames_seq and frames_seq[-1] != steps_all[-1]:
                    frames_seq.append(steps_all[-1])

                def row_for(s):
                    r = sdf[sdf["step"] == s].head(1)
                    if r.empty: return [], []
                    return r.iloc[0]["x_bins"], r.iloc[0]["counts"]

                x0, y0 = row_for(frames_seq[0])
                fig = go.Figure(
                    data=[go.Bar(x=x0, y=y0, name=f"step {frames_seq[0]}")],
                    layout=go.Layout(
                        height=360, margin=dict(l=10,r=10,t=30,b=10),
                        xaxis_title="bin", yaxis_title="count",
                        updatemenus=[dict(type="buttons", showactive=True, x=0.0, y=1.15,
                                          xanchor="left", yanchor="top",
                                          buttons=[dict(label="▶ Play", method="animate",
                                                        args=[None, {"frame": {"duration": int(1000/max(1,fps))},
                                                                     "fromcurrent": True, "transition": {"duration": 0}}]),
                                                   dict(label="⏸ Pause", method="animate",
                                                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
                        sliders=[dict(active=0, x=0.1, y=0.05, xanchor="left", len=0.8,
                                      currentvalue=dict(prefix="step: "),
                                      steps=[dict(label=str(s), method="animate",
                                                  args=[[str(s)], {"mode": "immediate",
                                                                   "frame": {"duration": 0}, "transition": {"duration": 0}}])
                                             for s in frames_seq])]
                    ),
                    frames=[go.Frame(name=str(s),
                                     data=[go.Bar(x=row_for(s)[0], y=row_for(s)[1], name=f"step {s}")])
                            for s in frames_seq]
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                done = True
        except Exception:
            pass

        if not done:
            # Legacy fallback: plot mean F per step (animated)
            try:
                df = con.execute("""
                    SELECT step, AVG(F_local) AS F_mean
                    FROM read_parquet(?)
                    WHERE step BETWEEN ? AND ?
                    GROUP BY step ORDER BY step
                """, [files_sp, int(frame_lo), int(frame_hi)]).df()
                if df.empty:
                    st.info("No readable spectra in either schema for this range.")
                else:
                    steps_all = df["step"].tolist()
                    frames_seq = steps_all[::max(1,int(stride))] or steps_all
                    if frames_seq and frames_seq[-1] != steps_all[-1]:
                        frames_seq.append(steps_all[-1])
                    def upto(s):
                        m = df[df["step"] <= s]
                        return m["step"].tolist(), m["F_mean"].tolist()
                    x0,y0 = upto(frames_seq[0])
                    fig = go.Figure(
                        data=[go.Scatter(x=x0, y=y0, mode="lines+markers", name="F_mean")],
                        layout=go.Layout(
                            height=300, margin=dict(l=10,r=10,t=30,b=10),
                            xaxis_title="step", yaxis_title="F_mean",
                            updatemenus=[dict(type="buttons", showactive=True, x=0.0, y=1.15,
                                              xanchor="left", yanchor="top",
                                              buttons=[dict(label="▶ Play", method="animate",
                                                            args=[None, {"frame": {"duration": int(1000/max(1,fps))},
                                                                         "fromcurrent": True, "transition": {"duration": 0}}]),
                                                       dict(label="⏸ Pause", method="animate",
                                                            args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
                            sliders=[dict(active=0, x=0.1, y=0.05, xanchor="left", len=0.8,
                                          currentvalue=dict(prefix="step: "),
                                          steps=[dict(label=str(s), method="animate",
                                                      args=[[str(s)], {"mode": "immediate",
                                                                       "frame": {"duration": 0}, "transition": {"duration": 0}}])
                                                 for s in frames_seq])]
                        ),
                        frames=[go.Frame(name=str(s),
                                         data=[go.Scatter(x=upto(s)[0], y=upto(s)[1], mode="lines+markers")])
                                for s in frames_seq]
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                st.info("No readable spectra in either schema for this range.")

# ---- World 3D (animated) ----
with tab_world3d:
    st.subheader("3D world (state)")
    files_state = get_files(prefix, run_id, "state", prefer_manifest, live_token, ttl_val)

    # New 3D state schema: (step, x, y, z, value); fallback maps m->value
    state_df = pd.DataFrame()
    if files_state:
        ok = False
        try:
            state_df = con.execute("""
                SELECT step, x, y, z, value
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_state, int(frame_lo), int(frame_hi)]).df()
            if not state_df.empty and {"x","y","z","value"}.issubset(state_df.columns):
                ok = True
        except Exception:
            pass

        if not ok:
            try:
                state_df = con.execute("""
                    SELECT step, x, y, z, m AS value
                    FROM read_parquet(?)
                    WHERE step BETWEEN ? AND ?
                """, [files_state, int(frame_lo), int(frame_hi)]).df()
            except Exception:
                state_df = pd.DataFrame()

    # Controls
    cA, cB = st.columns([1,1])
    with cA:
        color_by = st.selectbox("Color by", ["value"], index=0)
    with cB:
        max_points = st.number_input("Max points", min_value=2000, value=40000, step=5000,
                                     help="Downsample per frame to keep UI responsive")

    if state_df.empty:
        st.info("No state data in the selected range.")
    else:
        # Per-step downsampling for even density across frames
        def sample_per_step(df, per_step):
            out = []
            for s, g in df.groupby("step"):
                if len(g) > per_step:
                    out.append(g.sample(per_step, random_state=0))
                else:
                    out.append(g)
            return pd.concat(out, ignore_index=True)

        steps_all = sorted(state_df["step"].unique())
        frames_seq = steps_all[::max(1,int(stride))] or steps_all
        if frames_seq and frames_seq[-1] != steps_all[-1]:
            frames_seq.append(steps_all[-1])

        # target per-frame count
        per_step_quota = max(200, int(max_points / max(1, len(frames_seq))))
        sdf = sample_per_step(state_df, per_step_quota)

        vmax = max(1e-9, float(sdf[color_by].max()))
        size_base, size_scale = 3.0, 6.0

        # First frame
        f0 = sdf[sdf["step"] == frames_seq[0]]
        fig = go.Figure(
            data=[go.Scatter3d(
                x=f0["x"], y=f0["y"], z=f0["z"],
                mode="markers",
                marker=dict(
                    size=size_base + size_scale*(f0[color_by]/vmax),
                    color=f0[color_by], colorscale="Viridis", showscale=True,
                    colorbar=dict(title=color_by)
                ),
                name=f"state({color_by})"
            )],
            layout=go.Layout(
                height=600, margin=dict(l=10,r=10,t=30,b=10),
                scene=dict(aspectmode="data"),
                updatemenus=[dict(
                    type="buttons", showactive=True, x=0.0, y=1.15,
                    xanchor="left", yanchor="top",
                    buttons=[
                        dict(label="▶ Play", method="animate",
                             args=[None, {"frame": {"duration": int(1000/max(1,fps))},
                                          "fromcurrent": True, "transition": {"duration": 0}}]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                    ]
                )],
                sliders=[dict(
                    active=0, x=0.1, y=0.05, xanchor="left", len=0.8,
                    currentvalue=dict(prefix="step: "),
                    steps=[dict(label=str(s), method="animate",
                                args=[[str(s)], {"mode": "immediate",
                                                 "frame": {"duration": 0}, "transition": {"duration": 0}}])
                           for s in frames_seq]
                )]
            ),
            frames=[
                go.Frame(
                    name=str(s),
                    data=[go.Scatter3d(
                        x=(g := sdf[sdf["step"] == s])["x"],
                        y=g["y"], z=g["z"],
                        mode="markers",
                        marker=dict(
                            size=size_base + size_scale*(g[color_by]/vmax),
                            color=g[color_by], colorscale="Viridis", showscale=True
                        )
                    )]
                )
                for s in frames_seq
            ]
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---- Edges 3D (animated) ----
with tab_edges3d:
    st.subheader("Gradient edges")
    files_edges = get_files(prefix, run_id, "edges", prefer_manifest, live_token, ttl_val)

    if not files_edges:
        st.info("No edges_* shards yet.")
    else:
        # Prefer new 3D edges schema: (step, x0,y0,z0,v0,x1,y1,z1,v1)
        drawn = False
        try:
            edf = con.execute("""
                SELECT step, x0, y0, z0, x1, y1, z1, v0
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_edges, int(frame_lo), int(frame_hi)]).df()
            if not edf.empty and {"x0","y0","z0","x1","y1","z1"}.issubset(edf.columns):
                # pack segments of each frame into a single polyline trace using None separators
                steps_all = sorted(edf["step"].unique())
                frames_seq = steps_all[::max(1,int(stride))] or steps_all
                if frames_seq and frames_seq[-1] != steps_all[-1]:
                    frames_seq.append(steps_all[-1])

                max_lines = st.number_input("Max edges per frame", 200, 20000, 4000, step=500)

                def poly_from(df):
                    if len(df) > max_lines:
                        df = df.sample(int(max_lines), random_state=0)
                    x = []; y = []; z = []; w = []
                    v0max = max(1e-9, float(df["v0"].max())) if "v0" in df.columns and len(df) else 1.0
                    widths = []
                    for _, r in df.iterrows():
                        x += [r["x0"], r["x1"], None]
                        y += [r["y0"], r["y1"], None]
                        z += [r["z0"], r["z1"], None]
                        widths.append(2 + 6*((float(r.get("v0", 1.0)))/v0max))
                    # For 3D, Plotly can't vary width per segment in a single trace; use mean width
                    mean_w = float(np.mean(widths) if widths else 2.0)
                    return x,y,z,mean_w

                x0,y0,z0,w0 = poly_from(edf[edf["step"] == frames_seq[0]])
                fig = go.Figure(
                    data=[go.Scatter3d(x=x0, y=y0, z=z0, mode="lines",
                                       line=dict(width=w0), showlegend=False)],
                    layout=go.Layout(
                        height=600, margin=dict(l=10,r=10,t=30,b=10),
                        scene=dict(aspectmode="data"),
                        updatemenus=[dict(type="buttons", showactive=True, x=0.0, y=1.15,
                                          xanchor="left", yanchor="top",
                                          buttons=[dict(label="▶ Play", method="animate",
                                                        args=[None, {"frame": {"duration": int(1000/max(1,fps))},
                                                                     "fromcurrent": True, "transition": {"duration": 0}}]),
                                                   dict(label="⏸ Pause", method="animate",
                                                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
                        sliders=[dict(active=0, x=0.1, y=0.05, xanchor="left", len=0.8,
                                      currentvalue=dict(prefix="step: "),
                                      steps=[dict(label=str(s), method="animate",
                                                  args=[[str(s)], {"mode": "immediate",
                                                                   "frame": {"duration": 0}, "transition": {"duration": 0}}])
                                             for s in frames_seq])]
                    ),
                    frames=[go.Frame(name=str(s),
                                     data=[go.Scatter3d(
                                         x=poly_from(edf[edf["step"] == s])[0],
                                         y=poly_from(edf[edf["step"] == s])[1],
                                         z=poly_from(edf[edf["step"] == s])[2],
                                         mode="lines",
                                         line=dict(width=poly_from(edf[edf["step"] == s])[3]),
                                         showlegend=False
                                     )]) for s in frames_seq]
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                drawn = True
        except Exception:
            pass

        if not drawn:
            # Legacy fallback (as in your previous join approach)
            try:
                files_state = get_files(prefix, run_id, "state", prefer_manifest, live_token, ttl_val)
                if not files_state:
                    st.info("State shards unavailable for legacy edges mapping.")
                else:
                    # Build joined segments per step
                    edges_df = con.execute("""
                        SELECT step, src_conn, dst_conn, weight, t, x AS xs, y AS ys, z AS zs
                        FROM read_parquet(?)
                        WHERE step BETWEEN ? AND ?
                    """, [files_edges, int(frame_lo), int(frame_hi)]).df()
                    sd = con.execute("""
                        SELECT step, conn_id, x AS xd, y AS yd, z AS zd
                        FROM read_parquet(?)
                        WHERE step BETWEEN ? AND ?
                    """, [files_state, int(frame_lo), int(frame_hi)]).df()
                    if edges_df.empty or sd.empty:
                        st.info("Edges exist but insufficient state to map coordinates.")
                    else:
                        sd_latest = sd.sort_values("step").groupby("conn_id").tail(1).set_index("conn_id")
                        # Precompute per-step polylines
                        def seg_for_step(sstep):
                            e = edges_df[edges_df["step"] == sstep]
                            rows = []
                            for _, r in e.iterrows():
                                if r["dst_conn"] in sd_latest.index:
                                    dst = sd_latest.loc[r["dst_conn"]]
                                    rows.append((float(r["xs"]), float(r["ys"]), float(r["zs"]),
                                                 float(dst["xd"]), float(dst["yd"]), float(dst["zd"]),
                                                 float(r["weight"])))
                            if not rows:
                                return [], [], [], 2.0
                            seg = pd.DataFrame(rows, columns=["x1","y1","z1","x2","y2","z2","w"])
                            max_lines = st.number_input("Max edges per frame", 200, 20000, 3000, step=500, key="legacy_edges_max")
                            if len(seg) > max_lines:
                                seg = seg.sample(int(max_lines), random_state=0)
                            x = []; y = []; z = []
                            for _, s in seg.iterrows():
                                x += [s.x1, s.x2, None]
                                y += [s.y1, s.y2, None]
                                z += [s.z1, s.z2, None]
                            wmean = 2.0 + 6.0 * float(seg["w"].mean() / max(1e-9, float(seg["w"].max())))
                            return x,y,z,wmean

                        steps_all = sorted(edges_df["step"].unique())
                        frames_seq = steps_all[::max(1,int(stride))] or steps_all
                        if frames_seq and frames_seq[-1] != steps_all[-1]:
                            frames_seq.append(steps_all[-1])

                        x0,y0,z0,w0 = seg_for_step(frames_seq[0])
                        fig = go.Figure(
                            data=[go.Scatter3d(x=x0, y=y0, z=z0, mode="lines",
                                               line=dict(width=w0), showlegend=False)],
                            layout=go.Layout(
                                height=600, margin=dict(l=10,r=10,t=30,b=10),
                                scene=dict(aspectmode="data"),
                                updatemenus=[dict(type="buttons", showactive=True, x=0.0, y=1.15,
                                                  xanchor="left", yanchor="top",
                                                  buttons=[dict(label="▶ Play", method="animate",
                                                                args=[None, {"frame": {"duration": int(1000/max(1,fps))},
                                                                             "fromcurrent": True, "transition": {"duration": 0}}]),
                                                           dict(label="⏸ Pause", method="animate",
                                                                args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
                                sliders=[dict(active=0, x=0.1, y=0.05, xanchor="left", len=0.8,
                                              currentvalue=dict(prefix="step: "),
                                              steps=[dict(label=str(s), method="animate",
                                                          args=[[str(s)], {"mode": "immediate",
                                                                           "frame": {"duration": 0}, "transition": {"duration": 0}}])
                                                     for s in frames_seq])]
                            ),
                            frames=[go.Frame(name=str(s),
                                             data=[go.Scatter3d(
                                                 x=seg_for_step(s)[0], y=seg_for_step(s)[1], z=seg_for_step(s)[2],
                                                 mode="lines", line=dict(width=seg_for_step(s)[3]), showlegend=False)])
                                    for s in frames_seq]
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                st.info("No readable edges in either schema for this range.")

# ---- Diagnostics ----
with tab_diag:
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