#!/usr/bin/env bash
#
# Render a Fuka run to video using the HTTPS packer + Manim scene.
#
# Steps:
#   1) (Optional) rebuild indices/manifest for the run (local filesystem)
#   2) Pack shards over HTTPS (or local) into a single NPZ for Manim
#   3) Invoke Manim to render the 3D scene using canonical edge fields
#
# Requirements:
#   - Python 3.9+ with: duckdb, numpy, pandas, pyarrow, manim
#   - ManimCE in PATH (`manim`)
#
# Usage:
#   scripts/render_fuka.sh \
#     --prefix https://storage.googleapis.com/fuka4-runs \
#     --run_id FUKA_4_0_3D_20250906T234845Z \
#     --step_min 0 --step_max 999 \
#     --out_npz assets/fuka_anim.npz \
#     --rebuild_indices 0 \
#     --data_root data \
#     --quality k --fps 24
#
# Notes:
#   - If --prefix is omitted, the packer will still work if your *_index.json
#     contains HTTPS URLs; otherwise consider running analytics/build_indices.py
#     with --prefix first (or set DATA_URL_PREFIX).
#   - You can override Manim camera/colour knobs via environment variables:
#       FUKA_COLOR_KEY=edge_strength|edge_value
#       FUKA_MAX_EDGES=8000
#       FUKA_CAM_PHI=65 FUKA_CAM_THETA=-45 FUKA_CAM_ZOOM=1.1
#       FUKA_DEBUG_AXES=0|1
#       FUKA_WIRE_ALPHA=0.9 FUKA_STRENGTH_GAMMA=0.6
#       FUKA_WIDTH_MIN=1.2 FUKA_WIDTH_MAX=4.0
#       FUKA_STEP_MIN=0 FUKA_STEP_MAX=999
#
set -euo pipefail

# ----------------------- defaults -----------------------
PREFIX="${DATA_URL_PREFIX:-}"
RUN_ID=""
STEP_MIN=0
STEP_MAX=0
OUT_NPZ="assets/fuka_anim.npz"
REBUILD_INDICES=0
DATA_ROOT="data"
QUALITY="k"   # Manim quality: l (low), m, h, k (4K). We'll map to -q flag.
FPS=24

# ----------------------- helpers ------------------------
die() { echo "[render_fuka] ERROR: $*" >&2; exit 2; }
log() { echo "[render_fuka] $*" >&2; }

need_bin() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

abspath() {
  python - <<'PY'
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
}

# ----------------------- parse args ---------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)            PREFIX="${2:-}"; shift 2 ;;
    --run_id)            RUN_ID="${2:-}"; shift 2 ;;
    --step_min)          STEP_MIN="${2:-0}"; shift 2 ;;
    --step_max)          STEP_MAX="${2:-0}"; shift 2 ;;
    --out_npz)           OUT_NPZ="${2:-assets/fuka_anim.npz}"; shift 2 ;;
    --rebuild_indices)   REBUILD_INDICES="${2:-0}"; shift 2 ;;
    --data_root)         DATA_ROOT="${2:-data}"; shift 2 ;;
    --quality)           QUALITY="${2:-k}"; shift 2 ;;
    --fps)               FPS="${2:-24}"; shift 2 ;;
    -h|--help)
      sed -n '1,120p' "$0"; exit 0 ;;
    *)
      die "Unknown arg: $1 (use --help)";;
  esac
done

[[ -n "$RUN_ID" ]] || die "--run_id is required"
[[ "$STEP_MAX" =~ ^[0-9]+$ ]] || die "--step_max must be an integer"
[[ "$STEP_MIN" =~ ^[0-9]+$ ]] || die "--step_min must be an integer"
[[ "$STEP_MAX" -ge "$STEP_MIN" ]] || die "--step_max must be >= --step_min"

# ----------------------- checks -------------------------
need_bin python
need_bin manim

# Verify scene file exists
SCENE_FILE="render/manim_fuka_scene.py"
[[ -f "$SCENE_FILE" ]] || die "Scene not found: $SCENE_FILE"

# Ensure output folder for NPZ
OUT_DIR="$(dirname "$OUT_NPZ")"
mkdir -p "$OUT_DIR"

# ----------------------- optional: rebuild indices -------------------------
if [[ "$REBUILD_INDICES" == "1" || "$REBUILD_INDICES" == "true" ]]; then
  log "Rebuilding indices for run_id=$RUN_ID under $DATA_ROOT (prefix='${PREFIX}')"
  python -m analytics.build_indices --data_root "$DATA_ROOT" --run_id "$RUN_ID" ${PREFIX:+--prefix "$PREFIX"} || \
    die "Index rebuild failed"
fi

# ----------------------- pack NPZ from shards ------------------------------
log "Packing NPZ → $OUT_NPZ"
if [[ -n "${PREFIX}" ]]; then
  # HTTPS-aware packing (preferred)
  python -m fuka.render.pack_npz \
    --prefix "$PREFIX" \
    --run_id "$RUN_ID" \
    --step_min "$STEP_MIN" --step_max "$STEP_MAX" \
    --out "$OUT_NPZ" \
    || die "pack_npz failed"
else
  # No prefix provided. packer will try *_index.json first; make sure those contain URLs.
  log "No --prefix provided. pack_npz will rely on *_index.json/manifest paths; ensure they are URL-based."
  python -m fuka.render.pack_npz \
    --prefix "" \
    --run_id "$RUN_ID" \
    --step_min "$STEP_MIN" --step_max "$STEP_MAX" \
    --out "$OUT_NPZ" \
    || die "pack_npz failed"
fi

# ----------------------- render with Manim ---------------------------------
# Map QUALITY to manim -q flag
case "$QUALITY" in
  l|low)   QFLAG="-ql" ;;
  m|med)   QFLAG="-qm" ;;
  h|high)  QFLAG="-qh" ;;
  k|4k|ultra) QFLAG="-qk" ;;
  *)       QFLAG="-qk" ;;
esac

# Manim uses config.media_dir for outputs; you can supply render/manim.cfg if you have one.
# It also reads NPZ from assets/fuka_anim.npz by default (our OUT_NPZ).
if [[ "$OUT_NPZ" != "assets/fuka_anim.npz" ]]; then
  # If NPZ path differs, ensure the scene will find it: prefer assets/fuka_anim.npz
  mkdir -p assets
  ln -sf "$(abspath "$OUT_NPZ")" "assets/fuka_anim.npz"
fi

log "Rendering with Manim (quality=$QUALITY, fps=$FPS)"
manim $QFLAG --fps "$FPS" "$SCENE_FILE" Fuka3DScene || die "Manim render failed"

log "Done. NPZ: $OUT_NPZ"