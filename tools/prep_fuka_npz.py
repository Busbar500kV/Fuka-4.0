# tools/prep_fuka_npz.py
import argparse, sys, json
import numpy as np
import duckdb as ddb

MAX_POINTS = 15000
MAX_EDGES  = 2000

def sample_df(df, n):
    if len(df) <= n:
        return df
    return df.sample(n, random_state=0)

def read_index_files(con, prefix: str, run_id: str, table: str):
    url = f"{prefix}/runs/{run_id}/shards/{table}_index.json"
    try:
        df = con.execute("SELECT * FROM read_json_auto(?)", [url]).df()
        if "files" in df.columns and len(df) > 0:
            files = [f for f in df["files"] if isinstance(f, str) and f.startswith(("http://","https://"))]
            return files
    except Exception:
        pass
    return []

def main():
    ap = argparse.ArgumentParser(description="Pack Fuka step slices into NPZ for Manim rendering.")
    ap.add_argument("--prefix", required=True, help="e.g. https://storage.googleapis.com/fuka4-runs")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--step_min", type=int, required=True)
    ap.add_argument("--step_max", type=int, required=True)
    ap.add_argument("--out", default="assets/fuka_anim.npz")
    ap.add_argument("--max_points", type=int, default=MAX_POINTS)
    ap.add_argument("--max_edges",  type=int, default=MAX_EDGES)
    args = ap.parse_args()

    con = ddb.connect(":memory:")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    state_files = read_index_files(con, args.prefix, args.run_id, "state")
    edges_files = read_index_files(con, args.prefix, args.run_id, "edges")

    if not state_files:
        print("No state_* index files found. Abort.", file=sys.stderr)
        sys.exit(2)

    steps = list(range(int(args.step_min), int(args.step_max) + 1))
    state_by_step = {}
    edges_by_step = {}

    for s in steps:
        # 3D schema preferred; legacy fallback m->value
        df = con.execute("""
            SELECT x, y, z, value FROM read_parquet(?) WHERE step = ?
        """, [state_files, s]).df()
        if df.empty:
            df = con.execute("""
                SELECT x, y, z, m AS value FROM read_parquet(?) WHERE step = ?
            """, [state_files, s]).df()
        if not df.empty:
            df = sample_df(df, args.max_points)
            arr = df[["x","y","z","value"]].to_numpy("float32")
            state_by_step[s] = arr

        if edges_files:
            ed = con.execute("""
                SELECT x0,y0,z0,x1,y1,z1,v0,v1 FROM read_parquet(?) WHERE step = ?
            """, [edges_files, s]).df()
            if not ed.empty:
                ed = sample_df(ed, args.max_edges)
                earr = ed[["x0","y0","z0","x1","y1","z1","v0","v1"]].to_numpy("float32")
                edges_by_step[s] = earr

    S, E = [], []
    for s in steps:
        S.append(state_by_step.get(s, np.zeros((0,4), dtype="float32")))
        E.append(edges_by_step.get(s,  np.zeros((0,8), dtype="float32")))

    np.savez_compressed(args.out,
        steps=np.array(steps, dtype="int32"),
        states=np.array(S, dtype=object),
        edges=np.array(E, dtype=object),
        meta=dict(prefix=args.prefix, run_id=args.run_id,
                  step_min=int(args.step_min), step_max=int(args.step_max),
                  max_points=int(args.max_points), max_edges=int(args.max_edges))
    )
    print(f"[prep] wrote {args.out}: steps {steps[0]}..{steps[-1]} "
          f"(Σpoints={sum(len(x) for x in S):,}, Σedges={sum(len(x) for x in E):,})")

if __name__ == "__main__":
    main()