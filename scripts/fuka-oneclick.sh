#!/bin/bash
# Fuka 4.0 — single-run headless + live sync + final publish
# Deterministic: one runner at a time, one sync loop per run, no cross-run churn.
set -euo pipefail

# ------------------ CONSTANTS (VM-ready) ------------------
GIT_REMOTE="${GIT_REMOTE:-https://github.com/Busbar500kV/Fuka-4.0.git}"
REPO_DIR="${REPO_DIR:-/root/Fuka-4.0}"

DATA_ROOT="${DATA_ROOT:-/srv/fuka/data}"          # canonical local data root
VENV_DIR="${VENV_DIR:-/opt/fuka-venv}"
PYTHON_BIN="$VENV_DIR/bin/python"

ENTRYPOINT_MOD="${ENTRYPOINT_MOD:-headless.run_headless}"
BASE_CONFIG="${BASE_CONFIG:-fuka/config_default.json}"

LOG_DIR="${LOG_DIR:-/var/log/fuka}"
RUN_LOG="$LOG_DIR/headless.log"
SYNC_LOG="$LOG_DIR/sync.log"

LOCK_FILE="/var/lock/fuka-oneclick.lock"
PID_FILE="/var/run/fuka_headless.pid"
RUNID_FILE="/var/run/fuka_run_id.txt"

# GCS publish (recommended)
FUKA_GCS_BUCKET="${FUKA_GCS_BUCKET:-gs://fuka4-runs}"
HTTPS_PREFIX="https://storage.googleapis.com/${FUKA_GCS_BUCKET#gs://}"

# Sync settings
SYNC_INTERVAL="${SYNC_INTERVAL:-30}"       # seconds
# Which per-table indexes to publish
TABLES="state edges env events spectra ledger"
# ----------------------------------------------------------

STAMP(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
need_cmd(){ command -v "$1" >/dev/null 2>&1; }
log(){ echo "[$(STAMP)] $*" | tee -a "$RUN_LOG"; }

with_lock() {
  exec 9>"$LOCK_FILE"
  flock -n 9 || { echo "Another headless run is active (lock: $LOCK_FILE)"; exit 1; }
}

bootstrap() {
  mkdir -p "$LOG_DIR"; touch "$RUN_LOG" "$SYNC_LOG"
  log "[BOOT] Installing essentials..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null
  need_cmd git || apt-get install -y git >/dev/null
  dpkg -s python3-venv >/dev/null 2>&1 || apt-get install -y python3-venv >/dev/null
  dpkg -s python3-pip  >/dev/null 2>&1 || apt-get install -y python3-pip  >/dev/null
  need_cmd gsutil || apt-get install -y google-cloud-sdk >/dev/null || true
}

update_code() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    log "[GIT] Fetch & reset in $REPO_DIR"
    git -C "$REPO_DIR" fetch --all --prune >/dev/null
    git -C "$REPO_DIR" reset --hard origin/main >/dev/null
  else
    log "[GIT] Clone $GIT_REMOTE -> $REPO_DIR"
    rm -rf "$REPO_DIR"
    git clone "$GIT_REMOTE" "$REPO_DIR" >/dev/null
  fi
}

ensure_venv() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    log "[VENV] Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
  if [[ -f "$REPO_DIR/requirements-render.txt" ]]; then
    log "[VENV] pip -r requirements-render.txt"
    "$PYTHON_BIN" -m pip install -r "$REPO_DIR/requirements-render.txt" >/dev/null
  fi
  if [[ -f "$REPO_DIR/requirements.txt" ]]; then
    "$PYTHON_BIN" -m pip install -r "$REPO_DIR/requirements.txt" >/dev/null
  fi
}

make_timestamped_config() {
  local base_cfg="$1" out_cfg="$2"
  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  "$PYTHON_BIN" - <<PY
import json
with open("$base_cfg") as f: cfg=json.load(f)
cfg["run_name"]=cfg.get("run_name","FUKA")+"_"+ "$stamp"
with open("$out_cfg","w") as f: json.dump(cfg,f, indent=2)
print(cfg["run_name"])
PY
}

stop_running() {
  if [[ -f "$PID_FILE" ]]; then
    local PID="$(cat "$PID_FILE" || true)"
    if [[ -n "${PID:-}" ]] && ps -p "$PID" -o comm= | grep -qE 'python|python3'; then
      log "[STOP] Stopping PID $PID"
      kill "$PID" 2>/dev/null || true
      for _ in {1..40}; do sleep 0.2; ps -p "$PID" >/dev/null || break; done
      ps -p "$PID" >/dev/null && { log "[STOP] Forcing kill $PID"; kill -9 "$PID" 2>/dev/null || true; }
    fi
    rm -f "$PID_FILE"
  fi
  # Kill any stale sync loops owned by previous runs
  pkill -f "gsutil -m rsync -r" 2>/dev/null || true
}

live_sync_loop() {
  local run_dir="$1" pid="$2" gcs_dest="$3"
  echo "[$(STAMP)] [SYNC] Live rsync $run_dir -> $gcs_dest" | tee -a "$SYNC_LOG"
  while kill -0 "$pid" 2>/dev/null; do
    gsutil -m rsync -r "$run_dir" "$gcs_dest" >>"$SYNC_LOG" 2>&1 || true
    sleep "$SYNC_INTERVAL"
  done
  echo "[$(STAMP)] [SYNC] Final rsync" | tee -a "$SYNC_LOG"
  gsutil -m rsync -r "$run_dir" "$gcs_dest" >>"$SYNC_LOG" 2>&1 || true
}

publish_all() {
  local run_id="$1"
  local gcs_bucket="$2"
  log "[INDEX] Syncing indices + shards to GCS"
  FUKA_GCS_BUCKET="$gcs_bucket" "$PYTHON_BIN" "$REPO_DIR/analytics/build_indices.py" \
      --data_root "$DATA_ROOT" \
      --run_id "$run_id" \
      --prefer_gcs_for_manifest
  log "[INDEX] Updating runs/index.json in bucket"
}

start_run() {
  with_lock
  bootstrap
  update_code
  ensure_venv

  mkdir -p "$DATA_ROOT/runs" "$LOG_DIR"

  local tmp_cfg run_id run_dir gcs_dest
  tmp_cfg="$(mktemp)"
  pushd "$REPO_DIR" >/dev/null
  run_id="$(make_timestamped_config "$BASE_CONFIG" "$tmp_cfg")"
  echo "$run_id" > "$RUNID_FILE"
  run_dir="$DATA_ROOT/runs/$run_id"
  mkdir -p "$run_dir"
  gcs_dest="${FUKA_GCS_BUCKET}/runs/${run_id}"
  log "[RUN] run_id=$run_id"

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  set +e
  nohup "$PYTHON_BIN" -m "$ENTRYPOINT_MOD" \
        --config "$tmp_cfg" \
        --data_root "$DATA_ROOT" \
        >>"$RUN_LOG" 2>&1 &
  local PID=$!
  set -e
  echo "$PID" > "$PID_FILE"
  log "[RUN] PID=$PID"
  popd >/dev/null

  # Ensure the sync loop is cleaned up on any exit
  trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

  # one (1) sync loop for this run
  ( live_sync_loop "$run_dir" "$PID" "$gcs_dest" ) &

  # wait for end of simulation, then publish
  while kill -0 "$PID" 2>/dev/null; do sleep 1; done
  publish_all "$run_id" "$FUKA_GCS_BUCKET"
  log "[DONE] Run $run_id complete"
}

status_run() {
  if [[ -f "$PID_FILE" ]]; then
    local PID="$(cat "$PID_FILE" || true)"
    if [[ -n "${PID:-}" ]] && ps -p "$PID" >/dev/null; then
      echo "Headless running (PID $PID), RUN_ID=$(cat "$RUNID_FILE" 2>/dev/null || echo '?'). Logs: $RUN_LOG"
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