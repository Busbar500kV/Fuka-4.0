import duckdb as ddb
import streamlit as st
from pathlib import Path

st.title("Fuka 4.0 — Read-only Analytics")

data_root = Path("data")
run_id = st.text_input("Run ID", "FUKA_4_0_DEMO")

step_min = st.number_input("Step min", 0)
step_max = st.number_input("Step max", 5000)

if st.button("Load"):
    pattern = str(data_root / "runs" / run_id / "shards" / "events_*.parquet")
    con = ddb.connect()
    df = con.execute(f"""
      SELECT step, conn_id, dF, dm, w_sel, A_sel, phi_sel
      FROM read_parquet('{pattern}')
      WHERE step BETWEEN {int(step_min)} AND {int(step_max)}
    """).df()
    st.write(df.head(50))
    if not df.empty:
        st.line_chart(df.groupby("step", as_index=False)["dm"].sum(), x="step", y="dm")