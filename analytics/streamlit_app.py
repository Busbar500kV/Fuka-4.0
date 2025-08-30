import os
from pathlib import Path
import duckdb as ddb
import streamlit as st

st.title("Fuka 4.0 — Read-only Analytics (GCS)")

# Accept either local path or HTTPS prefix
default_prefix = os.getenv("DATA_URL_PREFIX", "https://storage.googleapis.com/fuka4-runs")
prefix = st.text_input("Data URL prefix (or folder)", default_prefix)
run_id = st.text_input("Run ID", "FUKA_4_0_DEMO")
step_min = st.number_input("Step min", 0)
step_max = st.number_input("Step max", 5000)

if st.button("Load"):
    con = ddb.connect()
    # Works for both local and HTTPS. Wildcards are supported.
    pattern = f"{prefix}/runs/{run_id}/shards/events_*.parquet"
    q = f"""
      SELECT step, conn_id, dF, dm, w_sel, A_sel, phi_sel
      FROM read_parquet('{pattern}')
      WHERE step BETWEEN {int(step_min)} AND {int(step_max)}
    """
    df = con.execute(q).df()
    st.write(df.head(100))
    if not df.empty:
        st.line_chart(df.groupby("step", as_index=False)["dm"].sum(), x="step", y="dm")