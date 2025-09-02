#!/usr/bin/env bash
# scripts/render_fuka.sh
set -euo pipefail

# Defaults
QUALITY="low"     # low|med|high
OUT_NPZ="assets/fuka_anim.npz"
PREFIX=""
RUN_ID=""
STEP_MIN=""
STEP_MAX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)   PREFIX="$2"; shift 2;;
    --run_id)   RUN_ID="$2"; shift 2;;
    --step_min) STEP_MIN="$2"; shift 2;;
    --step_max) STEP_MAX="$2"; shift 2;;
    --out)      OUT_NPZ="$2"; shift 2;;
    --quality)  QUALITY="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

if [[ -z "$PREFIX" || -z "$RUN_ID" || -z "$STEP_MIN" || -z "$STEP_MAX" ]]; then
  echo "Usage: $0 --prefix URL --run_id ID --step_min N --step_max M [--out path.npz] [--quality low|med|high]"
  exit 2
fi

# Quality presets
case "$QUALITY" in
  low)   MANIM_Q="-pql"; STEP_SEC="0.04"; POINT_R="0.030"; EDGE_W="2.5";;
  med)   MANIM_Q="-pqm"; STEP_SEC="0.06"; POINT_R="0.032"; EDGE_W="3.0";;
  high)  MANIM_Q="-pqh"; STEP_SEC="0.08"; POINT_R="0.034"; EDGE_W="3.5";;
  *) echo "Unknown quality '$QUALITY'"; exit 2;;
esac

echo "[render] Packing NPZ → $OUT_NPZ"
python tools/prep_fuka_npz.py \
  --prefix "$PREFIX" \
  --run_id "$RUN_ID" \
  --step_min "$STEP_MIN" \
  --step_max "$STEP_MAX" \
  --out "$OUT_NPZ"

export FUKA_NPZ="$OUT_NPZ"
export FUKA_STEP_SECONDS="$STEP_SEC"
export FUKA_POINT_RADIUS="$POINT_R"
export FUKA_EDGE_WIDTH="$EDGE_W"
export FUKA_SHOW_EDGES="1"

echo "[render] Manim rendering…"
manim $MANIM_Q render/manim_fuka_scene.py FukaWorldEdges3D \
  --renderer=opengl --write_to_movie \
  --media_dir media_out

echo "[render] Done. See media_out/videos/manim_fuka_scene/"