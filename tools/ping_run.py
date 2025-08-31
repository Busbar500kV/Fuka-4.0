# tools/ping_run.py
import argparse, json
from pathlib import Path
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from google.cloud import storage

def make_local(base: Path, run_id: str) -> Path:
    shards = base / "data" / "runs" / run_id / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    # state
    rows = []
    for step in (0, 50, 100):
        for x in range(4):
            for y in range(3):
                for z in range(2):
                    rows.append(dict(run_id=run_id, step=step, conn_id=10+x+y+z,
                                     x=x, y=y, z=z, m=0.1*(1+x+y+z),
                                     T_eff=280+step*0.01, theta_thr=0.5))
    pd.DataFrame(rows).to_parquet(shards/"state_000.parquet", index=False)

    # events
    er = []
    for step in (0, 50, 100):
        for i in range(6):
            er.append(dict(run_id=run_id, step=step, conn_id=i,
                           x=float(i), y=float(i%3), z=float(i%2),
                           dF=0.001*i, dm=0.0001*i, w_sel=1.2, A_sel=0.8, phi_sel=0.3,
                           T_eff=280+step*0.01, theta_thr=0.4))
    pd.DataFrame(er).to_parquet(shards/"events_000.parquet", index=False)

    # ledger
    ld = [dict(run_id=run_id, step=s, dF=0.01, c2dm=0.009, Q=0.0, W_cat=0.001,
               net_flux=0.0, balance_error=1e-6) for s in (0,50,100)]
    pd.DataFrame(ld).to_parquet(shards/"ledger_000.parquet", index=False)

    # env
    env = []
    for step in (0, 100):
        for x in range(4):
            for y in range(3):
                for z in range(2):
                    env.append(dict(run_id=run_id, step=step, x=x, y=y, z=z, value=0.01*(x+y+z)))
    pd.DataFrame(env).to_parquet(shards/"env_000.parquet", index=False)

    # manifest
    manifest = dict(
        run_id=run_id,
        created_at=float(pd.Timestamp.now().timestamp()),
        shards=[
            dict(table="state",  path=f"data/runs/{run_id}/shards/state_000.parquet"),
            dict(table="events", path=f"data/runs/{run_id}/shards/events_000.parquet"),
            dict(table="ledger", path=f"data/runs/{run_id}/shards/ledger_000.parquet"),
            dict(table="env",    path=f"data/runs/{run_id}/shards/env_000.parquet"),
        ]
    )
    (base/"data"/"runs"/run_id/"manifest.json").write_text(json.dumps(manifest, indent=2))
    return shards

def upload_and_index(local_shards: Path, bucket: str, run_id: str):
    client = storage.Client()
    b = client.bucket(bucket)

    # upload all parquet + manifest
    for p in local_shards.parents[1].rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_shards.parents[2])  # data/runs/...
            b.blob(str(rel)).upload_from_filename(str(p))

    # per-table *_index.json
    url_base = f"https://storage.googleapis.com/{bucket}/runs/{run_id}/shards"
    for t in ("events","state","ledger","env"):
        data = {"table":t,"run_id":run_id,
                "files":[f"{url_base}/{t}_000.parquet"]}
        b.blob(f"runs/{run_id}/shards/{t}_index.json").upload_from_string(
            json.dumps(data, indent=2), content_type="application/json"
        )

    # runs/index.json
    idx = b.blob("runs/index.json")
    try:
        cur = json.loads(idx.download_as_text()).get("runs", [])
    except Exception:
        cur = []
    runs = sorted(set(cur) | {run_id})
    idx.upload_from_string(json.dumps({"runs":runs}, indent=2), content_type="application/json")
    print("runs/index.json now:", runs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--run_id", required=True)
    args = ap.parse_args()
    base = Path("/root/Fuka-4.0")
    shards = make_local(base, args.run_id)
    upload_and_index(shards, args.bucket, args.run_id)
    print(f"✅ Test run {args.run_id} uploaded to gs://{args.bucket}/runs/{args.run_id}")