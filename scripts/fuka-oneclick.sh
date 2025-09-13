#!/bin/bash
# Fuka 4.0 — one-click headless run + sync + index + manifest
# Idempotent, reusable, no manual patching needed.
set -euo pipefail

# ----------- SETTINGS (edit if needed) -----------
GIT_REMOTE="${GIT_REMOTE:-https://github.com/Busbar500kV/Fuka-4.0.git}"
REPO_DIR="${REPO_DIR:-/root/Fuka-4.0}"

# Write data in a neutral location (world-readable)
DATA_ROOT="${DATA_ROOT:-/srv/fuka/data}"

# Python venv
VENV_DIR="${VENV_DIR:-/opt/fuka-venv}"
PYTHON_BIN="$VENV_DIR/bin/python"

# GCS
GCP_PROJECT="${GCP_PROJECT:-eternal-sunset-239719}"
GCS_BUCKET="${GCS_BUCKET:-gs://fuka4-runs}"
HTTPS_PREFIX="${HTTPS_PREFIX:-https://storage.googleapis.com/fuka4-runs}"

# Entry / config
ENTRYPOINT_MOD="${ENTRYPOINT_MOD:-headless.run_headless}"
BASE_CONFIG="${BASE_CONFIG:-fuka/config_default.json}"

# Logs & PID
LOG_DIR="${LOG_DIR:-/var/log/fuka}"
RUN_LOG="$LOG_DIR/headless.log"
SYNC_LOG="$LOG_DIR/sync.log"
PID_FILE="${PID_FILE:-/var/run/fuka_headless.pid}"

SYNC_INTERVAL="${SYNC_INTERVAL:-30}"
TABLES="events spectra state ledger edges env"
# -------------------------------------------------

STAMP(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
need_cmd(){ command -v "$1" >/dev/null 2>&1; }

bootstrap_system() {
  mkdir -p "$LOG_DIR"; touch "$RUN_LOG" "$SYNC_LOG"
  echo "[$(STAMP)] [BOOT] Installing essentials..." | tee -a "$RUN_LOG"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null
  need_cmd git || apt-get install -y git >/dev/null
  dpkg -s python3-venv >/dev/null 2>&1 || apt-get install -y python3-venv >/dev/null
  dpkg -s python3-pip  >/dev/null 2>&1 || apt-get install -y python3-pip  >/dev/null
  need_cmd gcloud || echo "[$(STAMP)] [BOOT] WARNING: gcloud not found"
  need_cmd gsutil || echo "[$(STAMP)] [BOOT] WARNING: gsutil not found"

  # Ensure neutral data root exists and is world-readable (for non-root user access)
  mkdir -p "$DATA_ROOT"/runs
  chmod -R 755 /srv/fuka || true
}

ensure_auth() {
  gcloud config set project "$GCP_PROJECT" >/dev/null || true
}

stop_running() {
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" || true)"
    if [[ -n "${PID:-}" ]] && ps -p "$PID" -o comm= | grep -qE 'python|python3'; then
      echo "[$(STAMP)] [STOP] Stopping PID $PID" | tee -a "$RUN_LOG"
      kill "$PID" 2>/dev/null || true
      for _ in {1..25}; do sleep 0.2; ps -p "$PID" >/dev/null || break; done
      ps -p "$PID" >/devnull && { echo "[$(STAMP)] [STOP] Forcing kill $PID" | tee -a "$RUN_LOG"; kill -9 "$PID" 2>/dev/null || true; }
    fi
    rm -f "$PID_FILE"
  fi
  pkill -f "$ENTRYPOINT_MOD" 2>/dev/null || true
}

update_code() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    echo "[$(STAMP)] [GIT] Fetch & reset in $REPO_DIR" | tee -a "$RUN_LOG"
    git -C "$REPO_DIR" fetch --all --prune >/dev/null
    git -C "$REPO_DIR" reset --hard origin/main >/dev/null
  else
    echo "[$(STAMP)] [GIT] Clone $GIT_REMOTE -> $REPO_DIR" | tee -a "$RUN_LOG"
    rm -rf "$REPO_DIR"
    git clone "$GIT_REMOTE" "$REPO_DIR" >/dev/null
  fi
}

ensure_venv() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[$(STAMP)] [VENV] Creating venv at $VENV_DIR" | tee -a "$RUN_LOG"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
  if [[ -f "$REPO_DIR/requirements-render.txt" ]]; then
    echo "[$(STAMP)] [VENV] pip -r requirements-render.txt" | tee -a "$RUN_LOG"
    "$PYTHON_BIN" -m pip install -r "$REPO_DIR/requirements-render.txt" >/dev/null
  elif [[ -f "$REPO_DIR/requirements.txt" ]]; then
    echo "[$(STAMP)] [VENV] pip -r requirements.txt" | tee -a "$RUN_LOG"
    "$PYTHON_BIN" -m pip install -r "$REPO_DIR/requirements.txt" >/dev/null
  fi
}

make_timestamped_config() {
  local base_cfg="$1" out_cfg="$2"
  export BASE_CFG="$base_cfg"; export OUT_CFG="$out_cfg"
  export STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  "$PYTHON_BIN" - <<'PY' >/dev/null
import json, os
src=os.environ["BASE_CFG"]; dst=os.environ["OUT_CFG"]; stamp=os.environ["STAMP"]
cfg=json.load(open(src))
cfg["run_name"]=cfg.get("run_name","FUKA")+f"_{stamp}"
json.dump(cfg, open(dst,"w"), indent=2)
print(cfg["run_name"])
PY
  "$PYTHON_BIN" - <<'PY'
import json, os
print(json.load(open(os.environ["OUT_CFG"]))["run_name"])
PY
}

publish_indices_and_manifest() {
  local run_id="$1"
  echo "[$(STAMP)] [INDEX] Rebuilding indices for run_id=$run_id" | tee -a "$RUN_LOG"
  RUN_ID="$run_id" DATA_ROOT="$DATA_ROOT" HTTPS_PREFIX="$HTTPS_PREFIX" "$PYTHON_BIN" - <<'PY'
import os, json, subprocess, pathlib
root=os.environ["DATA_ROOT"]; run=os.environ["RUN_ID"]; prefix=os.environ["HTTPS_PREFIX"]
cmd=["python","-m","analytics.build_indices","--data_root",root,"--run_id",run,"--prefix",prefix]
r=subprocess.run(cmd, cwd="/root/Fuka-4.0")
print("[RUNNER] build_indices.py (local) rc=", r.returncode)
PY

  echo "[$(STAMP)] [INDEX] Syncing indices + shards to GCS" | tee -a "$RUN_LOG"
  gsutil -m rsync -r "$DATA_ROOT/runs/$run_id" "$GCS_BUCKET/runs/$run_id" >>"$SYNC_LOG" 2>&1 || true

  # also maintain runs index in bucket
  echo "[$(STAMP)] [INDEX] Updating runs/index.json in bucket" | tee -a "$RUN_LOG"
  "$PYTHON_BIN" - <<'PY'
import json, os, subprocess, pathlib, urllib.request
bucket=os.environ["GCS_BUCKET"].replace("gs://","")
root=os.environ["DATA_ROOT"]
# Gather local runs
runs_dir=pathlib.Path(root,"runs")
runs=sorted([p.name for p in runs_dir.iterdir() if p.is_dir() and (p/"shards").exists()])
tmp=pathlib.Path("/tmp/index.json"); tmp.write_text(json.dumps({"runs":runs}, indent=2))
subprocess.run(["gsutil","cp",str(tmp),f"gs://{bucket}/runs/index.json"], check=False)
PY
}

start_run() {
  bootstrap_system
  ensure_auth
  update_code
  ensure_venv

  local tmp_cfg run_id run_dir
  tmp_cfg="$(mktemp)"
  pushd "$REPO_DIR" >/dev/null
  run_id="$(make_timestamped_config "$BASE_CONFIG" "$tmp_cfg")"
  run_dir="$DATA_ROOT/runs/$run_id"
  mkdir -p "$run_dir"
  chmod -R 755 "$DATA_ROOT" || true
  echo "[$(STAMP)] [RUN] run_id=$run_id" | tee -a "$RUN_LOG"

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  set +e
  nohup "$PYTHON_BIN" -m "$ENTRYPOINT_MOD" \
        --config "$tmp_cfg" \
        --data_root "$DATA_ROOT" \
        >>"$RUN_LOG" 2>&1 &
  set -e
  local PID=$!
  echo "$PID" > "$PID_FILE"
  echo "[$(STAMP)] [RUN] PID=$PID" | tee -a "$RUN_LOG"
  popd >/dev/null

  # live sync while running
  (
    echo "[$(STAMP)] [SYNC] Live rsync $run_dir -> $GCS_BUCKET/runs/$run_id" | tee -a "$SYNC_LOG"
    while kill -0 "$PID" 2>/dev/null; do
      gsutil -m rsync -r "$run_dir" "$GCS_BUCKET/runs/$run_id" >>"$SYNC_LOG" 2>&1 || true
      sleep "$SYNC_INTERVAL"
    done
    echo "[$(STAMP)] [SYNC] Final rsync" | tee -a "$SYNC_LOG"
    gsutil -m rsync -r "$run_dir" "$GCS_BUCKET/runs/$run_id" >>"$SYNC_LOG" 2>&1 || true
  ) & disown

  # wait for completion
  while kill -0 "$PID" 2>/dev/null; do sleep 1; done
  echo "[$(STAMP)] [RUN] Completed: $run_id" | tee -a "$RUN_LOG"

  publish_indices_and_manifest "$run_id"
  echo "[$(STAMP)] [DONE] Run $run_id complete" | tee -a "$RUN_LOG"
}

status_run() {
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE" || true)"
    if [[ -n "${PID:-}" ]] && ps -p "$PID" >/dev/null; then
      echo "Headless running (PID $PID). Logs: $RUN_LOG"
      exit 0
    fi
  fi
  echo "Headless not running."; exit 1
}

tail_logs(){ tail -n 200 -f "$RUN_LOG" "$SYNC_LOG"; }

case "${1:-start}" in
  start|"") stop_running; start_run ;;
  restart)  stop_running; start_run ;;
  stop)     stop_running; echo "Stopped." ;;
  status)   status_run ;;
  tail)     tail_logs ;;
  *) echo "Usage: $0 [start|restart|stop|status|tail]"; exit 2 ;;
esac