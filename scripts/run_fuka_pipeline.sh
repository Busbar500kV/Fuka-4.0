#!/usr/bin/env bash
#
# Full Fuka pipeline:
#   1) Run headless simulation (python -m fuka.runner)
#   2) Rebuild indices + manifest (python -m analytics.build_indices)
#   3) Pack NPZ over HTTPS/local (python -m fuka.render.pack_npz)
#   4) Render with Manim (render/manim_fuka_scene.py)
#
# Hard requirements:
#   - python, manim
#   - Python deps: duckdb, numpy, pandas, pyarrow
#
# Optional env knobs (forwarded to the Manim scene):
#   FUKA_COLOR_KEY=edge_strength|edge_value
#   FUKA_MAX_EDGES=12000
#   FUKA_CAM_PHI=65 FUKA_CAM_THETA=-45 FUKA_CAM_ZOOM=1.1
#   FUKA_DEBUG_AXES=0|1
#   FUKA_WIRE_ALPHA=0.9 FUKA_STRENGTH_GAMMA=0.6
#   FUKA_WIDTH_MIN=1.2 FUKA_WIDTH_MAX=4.0
#   FUKA_STEP_MIN / FUKA_STEP_MAX (override auto step range)
#
# Example:
#   bash scripts/run_fuka_pipeline.sh \
#     --config configs/demo_3d.json \
#     --data_root data \
#     --run_id FUKA_4_0_3D_TEST \
#     --prefix https://storage.googleapis.com/fuka4-runs \
#     --out_npz assets/fuka_anim.npz \
#     --quality k --fps 24
#
set -euo pipefail

# ----------------------- defaults -----------------------
CONFIG=""
DATA_ROOT="data"
RUN_ID=""
PREFIX="${DATA_URL_PREFIX:-}"       # e.g. https://storage.googleapis.com/fuka4-runs
OUT_NPZ="assets/fuka_anim.npz"
QUALITY="k"                         # l/m/h/k (maps to manim -q)
FPS=24
STEP_MIN_OVERRIDE=""
STEP_MAX_OVERRIDE=""

# ----------------------- helpers ------------------------
die() { echo "[run_fuka_pipeline] ERROR: $*" >&2; exit 2; }
log() { echo "[run_fuka_pipeline] $*" >&2; }

need_bin() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

abspath() {
  python - <<'PY'
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
}

latest_run_id() {
  local root="${1:-data}"
  python - "$root" <<'PY'
import sys, os, pathlib, time, json
root = pathlib.Path(sys.argv[1]) / "runs"
if not root.is_dir():
    print("")
    raise SystemExit(0)
cands = [p for p in root.iterdir() if p.is_dir()]
if not cands:
    print("")
    raise SystemExit(0)
# newest by mtime
latest = max(cands, key=lambda p: p.stat().st_mtime)
print(latest.name)
PY
}

steps_from_config() {
  local cfg="${1:?}"
  python - "$cfg" <<'PY'
import sys, json
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    obj = json.load(f)
steps = int(((obj.get("run") or {}).get("steps", 0)))
print(steps if steps>0 else 0)
PY
}

# ----------------------- parse args ---------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)            CONFIG="${2:-}"; shift 2 ;;
    --data_root)         DATA_ROOT="${2:-data}"; shift 2 ;;
    --run_id)            RUN_ID="${2:-}"; shift 2 ;;
    --prefix)            PREFIX="${2:-}"; shift 2 ;;
    --out_npz)           OUT_NPZ="${2:-assets/fuka_anim.npz}"; shift 2 ;;
    --quality)           QUALITY="${2:-k}"; shift 2 ;;
    --fps)               FPS="${2:-24}"; shift 2 ;;
    --step_min)          STEP_MIN_OVERRIDE="${2:-}"; shift 2 ;;
    --step_max)          STEP_MAX_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '1,160p' "$0"; exit 0 ;;
    *)
      die "Unknown arg: $1 (use --help)";;
  esac
done

[[ -n "$CONFIG" ]] || die "--config is required"
[[ -f "$CONFIG" ]] || die "Config not found: $CONFIG"

need_bin python
need_bin manim

# ----------------------- run headless -------------------
log "Running headless simulation…"
python -m fuka.runner \
  --config "$CONFIG" \
  --data_root "$DATA_ROOT" \
  ${RUN_ID:+--run_id "$RUN_ID"} \
  ${PREFIX:+--prefix "$PREFIX"} || die "Headless runner failed"

# resolve run_id (if not provided)
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(latest_run_id "$DATA_ROOT")"
  [[ -n "$RUN_ID" ]] || die "Could not determine run_id after running the engine"
  log "Resolved run_id → $RUN_ID"
fi

# ----------------------- indices/manifest ----------------
log "Rebuilding indices for run_id=$RUN_ID"
python -m analytics.build_indices \
  --data_root "$DATA_ROOT" \
  --run_id "$RUN_ID" \
  ${PREFIX:+--prefix "$PREFIX"} || die "Index rebuild failed"

# ----------------------- step range ---------------------
if [[ -n "$STEP_MIN_OVERRIDE" ]]; then
  STEP_MIN="$STEP_MIN_OVERRIDE"
else
  STEP_MIN="0"
fi

if [[ -n "$STEP_MAX_OVERRIDE" ]]; then
  STEP_MAX="$STEP_MAX_OVERRIDE"
else
  # derive from config.run.steps (minus one)
  TOTAL_STEPS="$(steps_from_config "$CONFIG")"
  if [[ "$TOTAL_STEPS" =~ ^[0-9]+$ ]] && [[ "$TOTAL_STEPS" -gt 0 ]]; then
    STEP_MAX="$((TOTAL_STEPS - 1))"
  else
    # fallback: guess a range
    STEP_MAX="999"
  fi
fi

# ----------------------- pack + render -------------------
# Reuse the render helper (it handles NPZ packing & Manim)
[[ -f scripts/render_fuka.sh ]] || die "Missing helper: scripts/render_fuka.sh"
bash scripts/render_fuka.sh \
  ${PREFIX:+--prefix "$PREFIX"} \
  --run_id "$RUN_ID" \
  --step_min "$STEP_MIN" \
  --step_max "$STEP_MAX" \
  --out_npz "$OUT_NPZ" \
  --rebuild_indices 0 \
  --data_root "$DATA_ROOT" \
  --quality "$QUALITY" \
  --fps "$FPS" || die "Render pipeline failed"

log "All done. Run: $RUN_ID | NPZ: $OUT_NPZ"