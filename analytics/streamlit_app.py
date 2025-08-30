# analytics/streamlit_app.py

import json
import math
import urllib.request
from urllib.error import HTTPError, URLError
from typing import Dict, List, Optional

import duckdb as ddb
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============== Config & helpers ==============

st.set_page_config(page_title="Fuka 4.0 – Analytics", layout="wide")

PREFIX_DEFAULT = st.secrets.get("DATA_URL_PREFIX", "").rstrip("/")
if not PREFIX_DEFAULT:
    st.warning("DATA_URL_PREFIX secret missing. Set it to your bucket root, e.g. "
               "`https://storage.googleapis.com/fuka4-runs`")

@st.cache_data(show_spinner=False, ttl=60)
def http_json(url: str) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            return json.loads(r.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None

@st.cache_data(show_spinner=False, ttl=60)
def http_text(url: str) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

def runs_index_url(prefix: str) -> str:
    # recorder cell publishes runs/index.json
    return f"{prefix}/runs/index.json"

def table_index_url(prefix: str, run_id: str, table: str) -> str:
    # recorder cell publishes per-table indices in .../runs/<RUN>/shards/<table>_index.json
    return f"{prefix}/runs/{run_id}/shards/{table}_index.json"

def manifest_url(prefix: str, run_id: str) -> str:
    return f"{prefix}/runs/{run_id}/manifest.json"

def manifest_to_https(prefix: str, manifest: dict) -> Dict[str, List[str]]:
    """
    Convert manifest entries (which store paths like 'data/runs/<RUN>/shards/..')
    into HTTPS URLs rooted at <prefix>/runs/<RUN>/shards/...
    """
    by_table: Dict[str, List[str]] = {}
    shards = manifest.get("shards", [])
    for s in shards:
        t = s.get("table")
        p = s.get("path", "")
        if not t or not p:
            continue
        # Strip 'data/' leading component because objects in GCS start from 'runs/...'
        if p.startswith("data/"):
            p = p[len("data/"):]
        url = f"{prefix}/{p}"
        if url.endswith(".parquet"):
            by_table.setdefault(t, []).append(url)
    # stable order
    for k in by_table:
        by_table[k] = sorted(by_table[k])
    return by_table

@st.cache_data(show_spinner=False, ttl=30)
def load_index(prefix: str, run_id: str, table: str) -> List[str]:
    """Try to load prebuilt *_index.json; returns list of HTTPS file URLs or []."""
    url = table_index_url(prefix, run_id, table)
    data = http_json(url)
    if not data:
        return []
    return data.get("files", []) or []

@st.cache_data(show_spinner=False, ttl=30)
def list_runs(prefix: str) -> List[str]:
    data = http_json(runs_index_url(prefix))
    if data and "runs" in data:
        return sorted(list({str(x) for x in data["runs"]}))
    return []

def get_file_list(prefix: str, run_id: str, table: str) -> List[str]:
    """
    Use session override if user rebuilt from manifest; else use *_index.json.
    """
    override = st.session_state.get("override_files")
    if isinstance(override, dict) and table in override:
        return override[table]
    return load_index(prefix, run_id, table)


# ============== Sidebar ==============

st.sidebar.title("Fuka 4.0 Analytics")
prefix = st.sidebar.text_input("Data URL prefix", PREFIX_DEFAULT, help="Bucket HTTPS root, e.g. https://storage.googleapis.com/fuka4-runs")

runs = list_runs(prefix)
run_id = st.sidebar.selectbox("Run", runs, index=0 if runs else None, placeholder="Select a run…")
manual_run = st.sidebar.text_input("Or enter run id", value="" if run_id else "")
if not run_id and manual_run:
    run_id = manual_run.strip()

st.sidebar.caption("Tip: Use the Diagnostics tab’s 'Rebuild from manifest' "
                   "to bypass stale indices during live runs.")

if not run_id:
    st.stop()


# ============== Header ==============

st.title(f"Fuka 4.0 — Run: {run_id}")

# Step filters (applied in all tabs)
colA, colB, colC = st.columns([1,1,2])
with colA:
    step_min = st.number_input("Step min", min_value=0, value=0, step=100)
with colB:
    step_max = st.number_input("Step max", min_value=1, value=5_000, step=1000)
with colC:
    st.write(" ")


# ============== DuckDB connection ==============
con = ddb.connect(":memory:")


# ============== TABS ==============
tab_overview, tab_events, tab_spectra, tab_world3d, tab_edges3d, tab_diag = st.tabs(
    ["Overview", "Events", "Spectra", "World 3D", "Edges 3D", "Diagnostics"]
)


# ============== Overview ==============
with tab_overview:
    st.subheader("Aggregate metrics")

    files_events = get_file_list(prefix, run_id, "events")
    files_state  = get_file_list(prefix, run_id, "state")
    files_ledger = get_file_list(prefix, run_id, "ledger")

    # Counts & quick KPIs
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        st.metric("Event shards", len(files_events))
    with kcol2:
        st.metric("State shards", len(files_state))
    with kcol3:
        st.metric("Ledger shards", len(files_ledger))
    with kcol4:
        man = http_json(manifest_url(prefix, run_id))
        st.metric("Manifest shards", len(man.get("shards", [])) if man else 0)

    # Totals over selected step window
    if files_ledger:
        q = """
        SELECT 
          MIN(step) AS min_step, 
          MAX(step) AS max_step, 
          SUM(dF)   AS sum_dF,
          SUM(c2dm) AS sum_c2dm,
          SUM(balance_error) AS sum_balance_error
        FROM read_parquet(?)
        WHERE step BETWEEN ? AND ?
        """
        df = con.execute(q, [files_ledger, int(step_min), int(step_max)]).df()
        st.write(df)
        if not df.empty:
            be = float(df["sum_balance_error"].iloc[0] or 0.0)
            st.caption(f"Global balance error over window: {be:.6g}")


# ============== Events ==============
with tab_events:
    st.subheader("Events in step window")

    files = get_file_list(prefix, run_id, "events")
    if not files:
        st.info("No events_* shards yet.")
    else:
        q = """
        SELECT step, conn_id, dm, dF, w_sel, A_sel, phi_sel, T_eff
        FROM read_parquet(?)
        WHERE step BETWEEN ? AND ?
        """
        df = con.execute(q, [files, int(step_min), int(step_max)]).df()
        st.write(f"{len(df):,} rows")
        st.dataframe(df.head(500))

        if not df.empty:
            # simple time series: sum(dm) by step
            q2 = """
            SELECT step, SUM(dm) AS dm_sum
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
            GROUP BY step
            ORDER BY step
            """
            df2 = con.execute(q2, [files, int(step_min), int(step_max)]).df()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df2["step"], y=df2["dm_sum"], mode="lines", name="Σ dm"))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                              xaxis_title="step", yaxis_title="sum(dm)")
            st.plotly_chart(fig, use_container_width=True)


# ============== Spectra ==============
with tab_spectra:
    st.subheader("Selected spectral component vs. step")
    files = get_file_list(prefix, run_id, "spectra")
    if not files:
        st.info("No spectra_* shards yet.")
    else:
        q = """
        SELECT step, w_dom, A_dom, phi_dom, AVG(F_local) AS F_mean
        FROM read_parquet(?)
        WHERE step BETWEEN ? AND ?
        GROUP BY step, w_dom, A_dom, phi_dom
        ORDER BY step
        """
        df = con.execute(q, [files, int(step_min), int(step_max)]).df()
        st.dataframe(df.head(200))

        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["step"], y=df["F_mean"], mode="lines+markers", name="F_mean"))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10),
                              xaxis_title="step", yaxis_title="F_mean")
            st.plotly_chart(fig, use_container_width=True)


# ============== World 3D (env + state) ==============
with tab_world3d:
    st.subheader("3D world (env and state)")

    files_env   = get_file_list(prefix, run_id, "env")
    files_state = get_file_list(prefix, run_id, "state")

    step_lo, step_hi = int(step_min), int(step_max)

    # env point cloud (sparse)
    env_df = pd.DataFrame()
    if files_env:
        q = """
        SELECT step, t, x, y, z, value
        FROM read_parquet(?)
        WHERE step BETWEEN ? AND ?
        """
        env_df = con.execute(q, [files_env, step_lo, step_hi]).df()

    # state nodes (downsampled)
    state_df = pd.DataFrame()
    if files_state:
        q = """
        SELECT step, x, y, z, m, T_eff
        FROM read_parquet(?)
        WHERE step BETWEEN ? AND ?
        """
        state_df = con.execute(q, [files_state, step_lo, step_hi]).df()

    if env_df.empty and state_df.empty:
        st.info("No env/state data in the selected step range yet.")
    else:
        fig = go.Figure()

        if not env_df.empty:
            fig.add_trace(go.Scatter3d(
                x=env_df["x"], y=env_df["y"], z=env_df["z"],
                mode="markers",
                marker=dict(size=2, opacity=0.5),
                name="env(value)"
            ))

        if not state_df.empty:
            # node size by mass, color by T_eff
            size = 4 + 6 * (state_df["m"] / (state_df["m"].max() or 1.0))
            fig.add_trace(go.Scatter3d(
                x=state_df["x"], y=state_df["y"], z=state_df["z"],
                mode="markers",
                marker=dict(size=size, color=state_df["T_eff"], colorscale="Viridis", showscale=True),
                name="state(m,T_eff)"
            ))

        fig.update_layout(
            height=600,
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="z",
                aspectmode="data"
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)


# ============== Edges 3D ==============
with tab_edges3d:
    st.subheader("Catalyst edges (transfers)")

    files_edges = get_file_list(prefix, run_id, "edges")
    if not files_edges:
        st.info("No edges_* shards yet (catalysts may still be propagating).")
    else:
        # We need coordinates; edges have (src_conn,dst_conn) and optional xmu for src.
        # We'll join with a state snapshot to get dst xyz too (approx), using nearest step.
        q_edges = """
        SELECT step, src_conn, dst_conn, weight, t, x, y, z
        FROM read_parquet(?)
        WHERE step BETWEEN ? AND ?
        """
        edges_df = con.execute(q_edges, [files_edges, int(step_min), int(step_max)]).df()

        files_state = get_file_list(prefix, run_id, "state")
        if files_state:
            q_state = """
            SELECT step, conn_id, x AS x2, y AS y2, z AS z2
            FROM read_parquet(?)
            WHERE step BETWEEN ? AND ?
            """
            sd = con.execute(q_state, [files_state, int(step_min), int(step_max)]).df()
        else:
            sd = pd.DataFrame(columns=["step","conn_id","x2","y2","z2"])

        if edges_df.empty or sd.empty:
            st.info("Not enough data to plot edges in 3D yet.")
        else:
            # Merge by approximate conn id: use latest state per conn_id in window
            sd_latest = sd.sort_values("step").groupby("conn_id").tail(1)
            # Map dst_conn -> (x2,y2,z2)
            sd_map = sd_latest.set_index("conn_id")[["x2","y2","z2"]]

            # Build line segments
            lines = []
            for _, row in edges_df.iterrows():
                src = (row["x"], row["y"], row["z"])
                dst_row = sd_map.loc[row["dst_conn"]] if row["dst_conn"] in sd_map.index else None
                if dst_row is None:
                    continue
                dst = (float(dst_row["x2"]), float(dst_row["y2"]), float(dst_row["z2"]))
                w = float(row["weight"])
                lines.append((*src, *dst, w))

            if not lines:
                st.info("Edges exist but could not map coordinates yet.")
            else:
                seg = pd.DataFrame(lines, columns=["x1","y1","z1","x2","y2","z2","w"])
                # Plot as many separate segments; a single trace with NaN breaks
                fig = go.Figure()
                for _, r in seg.iterrows():
                    fig.add_trace(go.Scatter3d(
                        x=[r.x1, r.x2], y=[r.y1, r.y2], z=[r.z1, r.z2],
                        mode="lines",
                        line=dict(width=2 + 6 * (r.w / (seg["w"].max() or 1.0))),
                        showlegend=False
                    ))
                fig.update_layout(
                    height=600,
                    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)


# ============== Diagnostics ==============
with tab_diag:
    st.subheader("Diagnostics")

    # 1) manifest rebuild (session only)
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Rebuild indices from manifest (session only)"):
            man = http_json(manifest_url(prefix, run_id))
            if not man:
                st.error("Could not load manifest.json")
            else:
                rebuilt = manifest_to_https(prefix, man)
                st.session_state["override_files"] = rebuilt
                st.success("Rebuilt from manifest for this session.")

    with c2:
        if st.button("Clear manifest override"):
            st.session_state.pop("override_files", None)
            st.info("Cleared session override; using on-bucket *_index.json again.")

    st.divider()
    st.subheader("Raw indices")
    for t in ["events","spectra","state","ledger","edges","env"]:
        files = get_file_list(prefix, run_id, t)
        st.write(f"**{t}**: {len(files)} files")
        if files:
            st.code("\n".join(files[:5]))

    st.divider()
    st.subheader("Manifest (for reference)")
    mtxt = http_text(manifest_url(prefix, run_id))
    if mtxt:
        st.code(mtxt[:4000])
    else:
        st.info("No manifest.json accessible (yet).")