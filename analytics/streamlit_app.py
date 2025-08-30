# analytics/streamlit_app.py
import os
import json
import urllib.request
from typing import Dict, List

import duckdb as ddb
import streamlit as st

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Fuka 4.0 — Cloud Analytics", layout="wide")
st.title("Fuka 4.0 — Cloud Analytics (GCS over HTTPS)")

# ----------------------------
# Resolve DATA_URL_PREFIX
# ----------------------------
DEFAULT_PREFIX = "https://storage.googleapis.com/fuka4-runs"
DATA_URL_PREFIX = st.secrets.get(
    "DATA_URL_PREFIX", os.getenv("DATA_URL_PREFIX", DEFAULT_PREFIX)
)
st.caption(f"📦 Using DATA_URL_PREFIX = {DATA_URL_PREFIX}")

# Allow quick override from the UI (handy for testing another bucket)
prefix = st.text_input("Bucket URL prefix", DATA_URL_PREFIX)

# Single DuckDB connection
con = ddb.connect()

# ----------------------------
# Helpers
# ----------------------------
def http_get_json(url: str, timeout: float = 6.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)

def list_runs_via_wildcard(prefix: str) -> List[str]:
    """Try to discover runs by scanning any parquet under runs/*/shards/."""
    pat = f"{prefix}/runs/*/shards/*.parquet"
    try:
        rows = con.execute(
            f"""
            SELECT DISTINCT REGEXP_EXTRACT(file, '.*/runs/([^/]+)/shards/.*') AS run_id
            FROM parquet_scan('{pat}', hive_partitioning=0)
            """
        ).fetchall()
        return sorted([r[0] for r in rows if r and r[0]])
    except Exception:
        return []

def list_runs_via_index(prefix: str) -> List[str]:
    """Fallback: fetch /runs/index.json which lists available run_ids."""
    url = f"{prefix}/runs/index.json"
    try:
        data = http_get_json(url)
        return sorted(data.get("runs", []))
    except Exception:
        return []

def count_shards_best_effort(prefix: str, run_id: str) -> Dict[str, int]:
    """Best-effort counters using wildcard scans (may be 0 if no matches)."""
    counts: Dict[str, int] = {}
    for table in ["events", "spectra", "state", "ledger"]:
        pat = f"{prefix}/runs/{run_id}/shards/{table}_*.parquet"
        try:
            n = con.execute(
                f"SELECT COUNT(*) FROM parquet_scan('{pat}', hive_partitioning=0)"
            ).fetchone()[0]
        except Exception:
            n = 0
        counts[table] = int(n)
    return counts

def load_event_files_from_index(prefix: str, run_id: str) -> List[str]:
    """Read shards_index.json to get explicit HTTPS URLs of event shards."""
    url = f"{prefix}/runs/{run_id}/shards/shards_index.json"
    data = http_get_json(url)  # raises if missing or not public
    files = data.get("files", [])
    # basic sanity
    return [u for u in files if u.endswith(".parquet")]

# ----------------------------
# Discover runs
# ----------------------------
with st.spinner("Scanning bucket for runs…"):
    run_ids = list_runs_via_wildcard(prefix)
    if not run_ids:
        # Fallback to index.json if wildcards return nothing
        run_ids = list_runs_via_index(prefix)

if run_ids:
    run_id = st.selectbox("Choose a run", run_ids, index=0)
else:
    st.info("No runs discovered automatically. Enter a run ID manually or publish /runs/index.json.")
    run_id = st.text_input("Run ID", "")

# ----------------------------
# Diagnostics / Shard counters
# ----------------------------
counts = {}
if run_id:
    counts = count_shards_best_effort(prefix, run_id)
    st.caption("📊 Shards discovered (best effort via wildcard scans)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("events", counts.get("events", 0))
    c2.metric("spectra", counts.get("spectra", 0))
    c3.metric("state", counts.get("state", 0))
    c4.metric("ledger", counts.get("ledger", 0))

st.divider()
st.subheader("Diagnostics")

colA, colB = st.columns([2, 1], vertical_alignment="top")

with colA:
    # Step window inputs (correct keyword args)
    step_min = st.number_input("Step min", min_value=0, value=0, step=100)
    step_max = st.number_input(
        "Step max",
        min_value=step_min,
        value=max(step_min + 100, 5_000),
        step=100
    )

    # Reference wildcard pattern (for your info only; not used for reads)
    events_pattern = f"{prefix}/runs/{run_id}/shards/events_*.parquet" if run_id else ""
    st.text_area("Reference (wildcard) events pattern", events_pattern, height=60)

with colB:
    # Test access: read first file listed in shards_index.json
    if run_id and st.button("Test access via shard index"):
        try:
            files = load_event_files_from_index(prefix, run_id)
            if not files:
                st.warning("shards_index.json loaded but no files listed yet.")
            else:
                test_url = files[0]
                df_test = con.execute(
                    "SELECT COUNT(*) AS rows FROM read_parquet($files)",
                    {"files": [test_url]},
                ).df()
                st.success(f"Read OK: {test_url}\nrows={int(df_test.iloc[0,0])}")
        except Exception as e:
            st.error(f"Index test failed:\n{e}")

# ----------------------------
# Load & visualize events (index-based, no HTTPS wildcards)
# ----------------------------
st.divider()
st.subheader("Events in step window")

if run_id and st.button("Load events"):
    # 1) Fetch explicit list of event shard URLs from the index
    try:
        files = load_event_files_from_index(prefix, run_id)
    except Exception as e:
        st.error(
            "Could not load shards_index.json for this run.\n\n"
            f"Expected at: {prefix}/runs/{run_id}/shards/shards_index.json\n\n{e}"
        )
        st.stop()

    if not files:
        st.warning("No event shards listed in shards_index.json yet.")
        st.stop()

    # 2) Read all listed event shards (pass the list; no wildcards)
    try:
        df = con.execute(
            """
            SELECT step, conn_id, dF, dm, w_sel, A_sel, phi_sel
            FROM read_parquet($files)
            """,
            {"files": files},
        ).df()
    except Exception as e:
        st.error(f"Failed to read {len(files)} event shards via index.\n\n{e}")
        st.stop()

    # 3) Filter and visualize
    df = df[(df["step"] >= int(step_min)) & (df["step"] <= int(step_max))]
    if df.empty:
        st.warning("No events in this step range.")
    else:
        st.write("Sample rows:", df.head(50))
        st.line_chart(
            df.groupby("step", as_index=False)["dm"].sum(),
            x="step",
            y="dm",
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Δm", f"{df['dm'].sum():.6g}")
        c2.metric("Events", f"{len(df):,}")
        c3.metric("Steps covered", f"{df['step'].nunique():,}")

st.caption(
    "Tip: publish /runs/index.json (list of run_ids) and per-run /shards/shards_index.json "
    "with explicit file URLs so the app avoids HTTP wildcards."
)