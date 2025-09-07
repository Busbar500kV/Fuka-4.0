# analytics/streamlit_app.py
import json, urllib.request, math, time
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

# Session defaults (no animation; scrubber drives "frame_step")
st.session_state.setdefault("frame_step", 0)
st.session_state.setdefault("override_files", None)    # dict: table -> [urls]
st.session_state.setdefault("active_panel", "World 3D")

# ---------------- Tunables & caps ----------------
MAX_POINTS_PER_FRAME = 20000
MAX_EDGES_PER_FRAME  = 3000
WORLD_TRAIL_STEPS_DEFAULT = 0     # default: no trail (set >0 in sidebar if you want a short trail)

# --------------- Cache helpers (short TTL for “live”) ---------------
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

# Live discovery of new shards every ~10s
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

# Optional small trail for World 3D (in *steps*, not frames)
world_trail_steps = st.sidebar.number_input("World 3D trail (previous steps)", min_value=0,
                                            value=WORLD_TRAIL_STEPS_DEFAULT, step=1,
                                            help="Overlay a few previous steps with fading opacity")

if not run_id:
    st.stop()

# ---------------- Header + scrubber ----------------
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

c1,c2 = st.columns([1,1])
with c1:
    step_min = st.number_input("Step min", min_value=0, value=int(lo_guess), step=100)
with c2:
    step_max = st.number_input("Step max", min_value=max(1, step_min+1),
                               value=int(max(hi_guess, step_min+1)), step=500)

# Scrubber: the ONE control that drives updates
frame = st.slider("Scrub step (time)", min_value=int(step_min), max_value=int(step_max),
                  value=int(min(max(st.session_state.frame_step, step_min), step_max)),
                  step=1, help="Drag to change the current simulation step")
st.session_state.frame_step = int(frame)

# HUD
hud1, hud2 = st.columns([3,1])
with hud1:
    st.markdown(f"### ⏱️ Current step: **{frame}**")
with hud2:
    pct = 0.0 if step_max == step_min else (frame - step_min)/(step_max - step_min)
    st.progress(pct, text=f"{int(100*pct)}% through range")

# ---------------- Shared helpers ----------------
def cap_points(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) > max_points:
        return df.sample(int(max_points), random_state=0)
    return df

# ---------------- Panel placeholders ----------------
ph_chart = st.empty()
ph_table = st.empty()

# ---------- Encoded-edges helpers (NEW) ----------
import io
import colorsys
from pathlib import Path

@st.cache_data(show_spinner=False)
def http_bytes(url: str, ttl: int) -> Optional[bytes]:
    """Download raw bytes from URL (for .npz)."""
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read()
    except Exception:
        return None

def encoded_edges_url(prefix: str, run_id: str, live_token: Optional[int]) -> str:
    # We publish encoded edges under runs/<run_id>/observer/encoded_edges.npz
    base = f"{prefix}/runs/{run_id}/observer/encoded_edges.npz"
    return _with_buster(base, live_token)

@st.cache_data(show_spinner=False)
def load_encoded_edges_npz(prefix: str, run_id: str, live_token: Optional[int], ttl: int):
    """
    Returns dict with Gx/Gy/Gz, Ex/Ey/Ez, Sx/Sy/Sz, Px/Py/Pz, tau (float)
    or None if not available.
    """
    url = encoded_edges_url(prefix, run_id, live_token)
    raw = http_bytes(url, ttl)
    if not raw:
        return None
    try:
        with np.load(io.BytesIO(raw)) as z:
            out = {
                "Gx": z["Gx"], "Gy": z["Gy"], "Gz": z["Gz"],
                "Ex": z["Ex"], "Ey": z["Ey"], "Ez": z["Ez"],
                "Sx": z["Sx"], "Sy": z["Sy"], "Sz": z["Sz"],
                "Px": z["Px"], "Py": z["Py"], "Pz": z["Pz"],
                "tau": float(z["tau_encode"]) if "tau_encode" in z else 0.05
            }
            return out
    except Exception:
        return None

def _hsv_to_rgb_vec(H: np.ndarray, S: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Vectorized HSV->RGB for arrays in [0,1]. Uses colorsys per-item (fast enough for capped edges).
    Returns r,g,b in [0,1].
    """
    Hf = np.asarray(H).ravel(); Sf = np.asarray(S).ravel(); Vf = np.asarray(V).ravel()
    rgb = [colorsys.hsv_to_rgb(float(h), float(s), float(v)) for h,s,v in zip(Hf,Sf,Vf)]
    rgb = np.array(rgb, dtype=float)
    R = rgb[:,0].reshape(H.shape); G = rgb[:,1].reshape(H.shape); B = rgb[:,2].reshape(H.shape)
    return R,G,B

def _build_axis_edges(mask: np.ndarray, axis: str) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    From a boolean mask of encoded edges on one axis, build arrays of x0,y0,z0,x1,y1,z1
    using integer voxel coordinates (edge midpoints are nice, but lines look fine).
    """
    idx = np.argwhere(mask)
    if idx.size == 0:
        return (np.array([]),)*6
    if axis == "x":
        # mask shape (nx-1,ny,nz), edge from (x,y,z) -> (x+1,y,z)
        x0 = idx[:,0].astype(float); y0 = idx[:,1].astype(float); z0 = idx[:,2].astype(float)
        x1 = x0 + 1.0; y1 = y0;       z1 = z0
    elif axis == "y":
        # mask shape (nx,ny-1,nz), edge (x,y,z) -> (x,y+1,z)
        x0 = idx[:,0].astype(float); y0 = idx[:,1].astype(float); z0 = idx[:,2].astype(float)
        x1 = x0;       y1 = y0 + 1.0; z1 = z0
    else:  # "z"
        # mask shape (nx,ny,nz-1), edge (x,y,z) -> (x,y,z+1)
        x0 = idx[:,0].astype(float); y0 = idx[:,1].astype(float); z0 = idx[:,2].astype(float)
        x1 = x0;       y1 = y0;       z1 = z0 + 1.0
    return x0,y0,z0,x1,y1,z1

def _hsv_from_embeddings(G: np.ndarray, Eabs: np.ndarray, Esign: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Color recipe:
      Hue   ~ atan2(Esign, Eabs)  in [0,1]
      Sat   ~ normalized persistence
      Value ~ normalized G
    """
    # tiny eps to avoid div-by-zero
    eps = 1e-9
    H = (np.arctan2(Esign, Eabs + eps) / np.pi + 1.0) * 0.5
    S = (P - P.min()) / (P.max() - P.min() + eps)
    V = (G - G.min()) / (G.max() - G.min() + eps)
    return H,S,V

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
            """, [files_ledger, int(step_min), int(step_max)])
        except Exception:
            df = query_df("""
                SELECT MIN(step) AS min_step, MAX(step) AS max_step,
                       SUM(dF) AS sum_dF, SUM(c2dm) AS sum_c2dm,
                       SUM(balance_error) AS sum_balance_error
                FROM read_parquet(?)
                WHERE step BETWEEN ? AND ?
            """, [files_ledger, int(step_min), int(step_max)])
        ph_table.dataframe(df)

def render_events():
    files_ev = get_files(prefix, run_id, "events", prefer_manifest, live_token, ttl_val)
    if not files_ev:
        ph_table.info("No events_* shards yet."); return
    # Count events per step up to current frame
    try:
        df = query_df("""
            SELECT step
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
        """, [files_ev, int(step_min), int(step_max)])
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
            """, [files_ev, int(step_min), int(step_max)])
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
    # Show histogram for the *current* step
    try:
        sdf = query_df("""
            SELECT step, counts, bin_edges
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
            ORDER BY step
        """, [files_sp, int(step_min), int(step_max)])
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
            """, [files_sp, int(step_min), int(step_max)])
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

    # Current step cloud
    df0 = _read_state_at(frame, files_state)
    if df0.empty:
        ph_table.info("No state at current step."); return
    df0 = cap_points(df0, MAX_POINTS_PER_FRAME)

    # Tiny trail: previous N *steps* (not frames)
    trail_df = pd.DataFrame()
    if world_trail_steps > 0:
        chunks = []
        per_trail = max(4000, MAX_POINTS_PER_FRAME // 8)
        for s in range(1, world_trail_steps+1):
            step_prev = frame - s
            if step_prev < step_min: break
            dfi = _read_state_at(step_prev, files_state)
            if dfi.empty: continue
            dfi = cap_points(dfi, per_trail)
            dfi["__age"] = s
            chunks.append(dfi)
        if chunks:
            trail_df = pd.concat(chunks, ignore_index=True)

    vmax = max(1e-9, float(df0["value"].max()))
    fig = go.Figure()

    # Current step
    size0 = 3 + 6*(df0["value"]/vmax)
    fig.add_trace(go.Scatter3d(
        x=df0["x"], y=df0["y"], z=df0["z"],
        mode="markers",
        marker=dict(size=size0, color=df0["value"], colorscale="Viridis", showscale=True,
                    colorbar=dict(title="value")),
        name=f"state(step={frame})"
    ))

    # Trail (fading opacity)
    if not trail_df.empty:
        ages = trail_df["__age"].astype(float).to_numpy()
        op = 0.5 - (ages-1) * (0.35/max(1.0, float(world_trail_steps)))
        op = np.clip(op, 0.15, 0.5)
        size_t = 2 + 4*(trail_df["value"]/vmax)
        fig.add_trace(go.Scatter3d(
            x=trail_df["x"], y=trail_df["y"], z=trail_df["z"],
            mode="markers",
            marker=dict(size=size_t, color=trail_df["value"], colorscale="Viridis",
                        showscale=False, opacity=float(np.mean(op))),
            name=f"trail({world_trail_steps} steps)"
        ))

    fig.update_layout(height=650, scene=dict(aspectmode="data"),
                      margin=dict(l=10,r=10,t=30,b=10))
    ph_chart.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def render_edges3d():
    files_edges = get_files(prefix, run_id, "edges", prefer_manifest, live_token, ttl_val)
    if not files_edges:
        ph_table.info("No edges_* shards yet."); return
    # Current step only (edges are heavy)
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