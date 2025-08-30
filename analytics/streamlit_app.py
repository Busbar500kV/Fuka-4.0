import os, json, urllib.request
from typing import Dict, List
import duckdb as ddb
import streamlit as st

st.set_page_config(page_title="Fuka 4.0 — Cloud Analytics", layout="wide")
st.title("Fuka 4.0 — Cloud Analytics (GCS)")

DEFAULT_PREFIX = "https://storage.googleapis.com/fuka4-runs"
DATA_URL_PREFIX = st.secrets.get("DATA_URL_PREFIX", os.getenv("DATA_URL_PREFIX", DEFAULT_PREFIX))
st.caption(f"📦 Using DATA_URL_PREFIX = {DATA_URL_PREFIX}")

prefix = st.text_input("Data URL prefix (bucket root)", DATA_URL_PREFIX)
con = ddb.connect()

def list_runs_via_wildcard(prefix: str) -> List[str]:
    pat = f"{prefix}/runs/*/shards/*.parquet"
    return [
        r[0] for r in
        con.execute(
            f"""
            SELECT DISTINCT REGEXP_EXTRACT(file, '.*/runs/([^/]+)/shards/.*') AS run_id
            FROM parquet_scan('{pat}', hive_partitioning=0)
            """
        ).fetchall()
        if r and r[0]
    ]

def list_runs_via_index(prefix: str) -> List[str]:
    url = f"{prefix}/runs/index.json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
        return data.get("runs", [])
    except Exception as e:
        st.info(f"No index.json found at {url} ({e}).")
        return []

# Try wildcard first; if it 404s, fall back to index.json
run_ids: List[str] = []
try:
    with st.spinner("Scanning bucket for runs…"):
        run_ids = sorted(list_runs_via_wildcard(prefix))
except Exception as e:
    st.warning(f"Auto-scan via wildcard failed:\n{e}\nFalling back to index.json…")
    run_ids = list_runs_via_index(prefix)

if run_ids:
    run_id = st.selectbox("Choose a run ID", run_ids, index=0)
else:
    st.info("No runs discovered. Enter a run ID manually or publish /runs/index.json.")
    run_id = st.text_input("Run ID")

def count_shards(prefix: str, run_id: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for table in ["events", "spectra", "state", "ledger"]:
        pat = f"{prefix}/runs/{run_id}/shards/{table}_*.parquet"
        try:
            n = con.execute(f"SELECT COUNT(*) FROM parquet_scan('{pat}', hive_partitioning=0)").fetchone()[0]
        except Exception:
            n = 0
        counts[table] = int(n)
    return counts

if run_id:
    counts = count_shards(prefix, run_id)
    st.caption("📊 Shards available")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("events", counts["events"])
    c2.metric("spectra", counts["spectra"])
    c3.metric("state", counts["state"])
    c4.metric("ledger", counts["ledger"])

st.divider()
st.subheader("Diagnostics")

colA, colB = st.columns([2, 1])

with colA:
    step_min = st.number_input("Step min", 0, value=0)
    step_max = st.number_input("Step max", 10_000, value=5_000)
    events_pattern = f"{prefix}/runs/{run_id}/shards/events_*.parquet" if run_id else ""
    st.text_area("Events query pattern", events_pattern, height=60)

with colB:
    if run_id:
        test_url = f"{prefix}/runs/{run_id}/shards/events_000.parquet"
        if st.button("Test access to first event shard"):
            st.write(f"Trying: {test_url}")
            try:
                df_test = con.execute(f"SELECT COUNT(*) AS rows FROM read_parquet('{test_url}')").df()
                st.success(f"Read OK: rows={int(df_test.iloc[0,0])}")
            except Exception as e:
                st.error(f"Failed to read {test_url}\n\n{e}")

st.divider()
st.subheader("Events in step window")

if run_id and st.button("Load data"):
    if counts.get("events", 0) == 0:
        st.warning("No event shards found yet for this run.")
        st.stop()

    q = f"""
      SELECT step, conn_id, dF, dm, w_sel, A_sel, phi_sel
      FROM read_parquet('{events_pattern}')
      WHERE step BETWEEN {int(step_min)} AND {int(step_max)}
    """
    try:
        df = con.execute(q).df()
    except Exception as e:
        st.error(f"Could not load data from:\n{events_pattern}\n\n{e}")
        st.stop()

    if df.empty:
        st.warning("No events in this step range.")
    else:
        st.write("Sample rows:", df.head(50))
        st.line_chart(df.groupby("step", as_index=False)["dm"].sum(), x="step", y="dm")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Δm", f"{df['dm'].sum():.6g}")
        c2.metric("Events", f"{len(df):,}")
        c3.metric("Steps covered", f"{df['step'].nunique():,}")

st.caption("Tip: Publish /runs/index.json so the app can list runs even when wildcards are empty.")