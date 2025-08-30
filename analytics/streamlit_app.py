# analytics/streamlit_app.py
import os, json, urllib.request
from typing import Dict, List, Tuple

import duckdb as ddb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- Config ----------
st.set_page_config(page_title="Fuka 4.0 — Cloud Analytics", layout="wide")
st.title("Fuka 4.0 — Cloud Analytics (GCS)")

DEFAULT_PREFIX = "https://storage.googleapis.com/fuka4-runs"
DATA_URL_PREFIX = st.secrets.get("DATA_URL_PREFIX", os.getenv("DATA_URL_PREFIX", DEFAULT_PREFIX))
prefix = st.text_input("Bucket URL prefix", DATA_URL_PREFIX)
st.caption(f"Using: {prefix}")

con = ddb.connect()

# ---------- Helpers ----------
def http_get_json(url: str, timeout: float = 8.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)

def list_runs(prefix: str) -> List[str]:
    # Try /runs/index.json, fallback to wildcard scan
    try:
        data = http_get_json(f"{prefix}/runs/index.json")
        return sorted(data.get("runs", []))
    except Exception:
        pass
    # wildcard best-effort
    pat = f"{prefix}/runs/*/shards/*.parquet"
    try:
        rows = con.execute(
            f"SELECT DISTINCT REGEXP_EXTRACT(file,'.*/runs/([^/]+)/shards/.*') AS run_id "
            f"FROM parquet_scan('{pat}', hive_partitioning=0)"
        ).fetchall()
        return sorted([r[0] for r in rows if r and r[0]])
    except Exception:
        return []

def load_index(prefix: str, run_id: str, table: str) -> List[str]:
    url = f"{prefix}/runs/{run_id}/shards/{table}_index.json"
    data = http_get_json(url)
    files = [u for u in data.get("files", []) if u.endswith(".parquet")]
    return files

def read_parquet_files(files: List[str], cols: List[str] = None) -> pd.DataFrame:
    if not files:
        return pd.DataFrame(columns=cols or [])
    if cols:
        q = f"SELECT {', '.join(cols)} FROM read_parquet($files)"
    else:
        q = "SELECT * FROM read_parquet($files)"
    return con.execute(q, {"files": files}).df()

def fourway_metrics(counts: Dict[str,int]):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("events", counts.get("events", 0))
    c2.metric("spectra", counts.get("spectra", 0))
    c3.metric("state", counts.get("state", 0))
    c4.metric("ledger", counts.get("ledger", 0))

def count_shards(prefix: str, run_id: str) -> Dict[str,int]:
    counts={}
    for t in ["events","spectra","state","ledger","edges","env"]:
        try:
            files = load_index(prefix, run_id, t)
            counts[t]=len(files)
        except Exception:
            counts[t]=0
    return counts

# ---------- Run selection ----------
runs = list_runs(prefix)
if runs:
    run_id = st.selectbox("Run", runs, index=0)
else:
    st.info("No runs discovered; enter manually or publish /runs/index.json")
    run_id = st.text_input("Run ID")

# ---------- Diagnostics ----------
if run_id:
    st.caption("Shard indices discovered")
    counts = count_shards(prefix, run_id)
    fourway_metrics(counts)

st.divider()

# ---------- Tabs ----------
tab_world, tab_energy, tab_events, tab_spectra, tab_diag = st.tabs(
    ["🌍 World 3D", "⚡ Energy", "📇 Events", "🎚️ Spectra", "🛠️ Diagnostics"]
)

# ======== WORLD 3D ========
with tab_world:
    st.subheader("3D connections + environment")
    step_min = st.number_input("Step min", min_value=0, value=0, step=100)
    step_max = st.number_input("Step max", min_value=step_min, value=max(step_min+100, 5000), step=100)
    colour_by = st.selectbox("Colour nodes by", ["m (mass)", "w_dom (freq)", "A_dom (amp)", "T_eff"], index=0)

    # Load nodes (state) & optional spectra to colour
    state_files = []
    spectra_files = []
    env_files = []
    edges_files = []
    try: state_files = load_index(prefix, run_id, "state")
    except: pass
    try: spectra_files = load_index(prefix, run_id, "spectra")
    except: pass
    try: env_files = load_index(prefix, run_id, "env")
    except: pass
    try: edges_files = load_index(prefix, run_id, "edges")
    except: pass

    nodes = read_parquet_files(state_files, cols=["step","conn_id","x","y","z","m","T_eff","theta_thr"])
    # Pick the last snapshot within window per node
    if not nodes.empty:
        nodes = nodes[(nodes["step"]>=step_min)&(nodes["step"]<=step_max)]
        nodes = nodes.sort_values(["conn_id","step"]).groupby("conn_id").tail(1)

    spec = read_parquet_files(spectra_files, cols=["step","conn_id","w_dom","A_dom","phi_dom"])
    if not spec.empty:
        spec = spec[(spec["step"]>=step_min)&(spec["step"]<=step_max)]
        spec = spec.sort_values(["conn_id","step"]).groupby("conn_id").tail(1)

    df_nodes = nodes.merge(spec, on=["conn_id","step"], how="left") if not nodes.empty else pd.DataFrame()

    # Env samples (sparse point cloud)
    env = read_parquet_files(env_files, cols=["step","x","y","z","value"])
    env = env[(env["step"]>=step_min)&(env["step"]<=step_max)] if not env.empty else env

    # Edges during window
    edges = read_parquet_files(edges_files, cols=["step","src_conn","dst_conn","weight"])
    edges = edges[(edges["step"]>=step_min)&(edges["step"]<=step_max)] if not edges.empty else edges

    # Map conn_id -> position for edges
    pos = df_nodes.set_index("conn_id")[["x","y","z"]] if not df_nodes.empty else pd.DataFrame()

    # Plotly 3D
    fig = go.Figure()

    # environment points
    if not env.empty:
        fig.add_trace(go.Scatter3d(
            x=env["x"], y=env["y"], z=env["z"],
            mode="markers",
            marker=dict(size=2, color=env["value"], colorscale="Viridis", showscale=True),
            name="environment"
        ))

    # edges as line segments
    if not edges.empty and not pos.empty:
        xe, ye, ze = [], [], []
        wts = []
        for _, r in edges.iterrows():
            if r["src_conn"] in pos.index and r["dst_conn"] in pos.index:
                xs, ys, zs = pos.loc[r["src_conn"]]; xd, yd, zd = pos.loc[r["dst_conn"]]
                xe += [xs, xd, None]; ye += [ys, yd, None]; ze += [zs, zd, None]
                wts.append(r["weight"])
        if xe:
            fig.add_trace(go.Scatter3d(
                x=xe, y=ye, z=ze, mode="lines",
                line=dict(width=3), opacity=0.4, name="edges"
            ))

    # nodes
    if not df_nodes.empty:
        if colour_by.startswith("m"):
            color = df_nodes["m"]
        elif colour_by.startswith("w_dom"):
            color = df_nodes["w_dom"]
        elif colour_by.startswith("A_dom"):
            color = df_nodes["A_dom"]
        else:
            color = df_nodes["T_eff"]
        fig.add_trace(go.Scatter3d(
            x=df_nodes["x"], y=df_nodes["y"], z=df_nodes["z"],
            mode="markers",
            marker=dict(size=4, color=color, colorscale="Plasma", showscale=True),
            text=[f"id {i}" for i in df_nodes["conn_id"]],
            name="connections"
        ))

    fig.update_layout(height=650, scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig, use_container_width=True)

# ======== ENERGY ========
with tab_energy:
    st.subheader("Energy flow & conservation")
    step_min_e = st.number_input("Step min (energy)", min_value=0, value=0, step=100, key="emin")
    step_max_e = st.number_input("Step max (energy)", min_value=step_min_e, value=max(step_min_e+100, 5000), step=100, key="emax")

    ledger_files = []
    try: ledger_files = load_index(prefix, run_id, "ledger")
    except: pass

    led = read_parquet_files(ledger_files, cols=["step","dF","c2dm","Q","W_cat","net_flux","balance_error"])
    if not led.empty:
        led = led[(led["step"]>=step_min_e)&(led["step"]<=step_max_e)]
        g = led.groupby("step", as_index=False).sum(numeric_only=True)
        st.line_chart(g[["step","dF"]].rename(columns={"dF":"free_energy_drop"}), x="step", y="free_energy_drop")
        st.line_chart(g[["step","c2dm"]].rename(columns={"c2dm":"c2dm"}), x="step", y="c2dm")
        st.line_chart(g[["step","balance_error"]], x="step", y="balance_error")
        st.write("Totals:", g[["dF","c2dm","Q","W_cat","net_flux","balance_error"]].sum().to_frame("total").T)
    else:
        st.info("No ledger shards yet.")

# ======== EVENTS ========
with tab_events:
    st.subheader("Event explorer")
    step_min_ev = st.number_input("Step min (events)", min_value=0, value=0, step=100, key="vmin")
    step_max_ev = st.number_input("Step max (events)", min_value=step_min_ev, value=max(step_min_ev+100, 5000), step=100, key="vmax")

    event_files = []
    try: event_files = load_index(prefix, run_id, "events")
    except: pass

    ev = read_parquet_files(event_files, cols=["step","conn_id","x","y","z","dF","dm","w_sel","A_sel","phi_sel","T_eff","theta_thr"])
    if not ev.empty:
        ev = ev[(ev["step"]>=step_min_ev)&(ev["step"]<=step_max_ev)]
        st.write(ev.head(50))
        st.bar_chart(ev.groupby("step", as_index=False)["dm"].sum(), x="step", y="dm")
    else:
        st.info("No events yet in this window.")

# ======== SPECTRA ========
with tab_spectra:
    st.subheader("Spectral summary")
    spectra_files = []
    try: spectra_files = load_index(prefix, run_id, "spectra")
    except: pass
    sp = read_parquet_files(spectra_files, cols=["step","conn_id","w_dom","A_dom","phi_dom","F_local","E_sum","S_spec"])
    if not sp.empty:
        st.write(sp.head(50))
        st.line_chart(sp.groupby("step", as_index=False)["F_local"].mean(), x="step", y="F_local")
    else:
        st.info("No spectra shards.")

# ======== DIAGNOSTICS ========
with tab_diag:
    st.subheader("Raw indices")
    for t in ["events","spectra","state","ledger","edges","env"]:
        try:
            files = load_index(prefix, run_id, t)
            st.text(f"{t}: {len(files)} files")
            if files:
                st.code("\n".join(files[:5]))
        except Exception as e:
            st.text(f"{t}: (no index) {e}")