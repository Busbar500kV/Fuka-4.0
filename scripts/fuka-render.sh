#!/bin/bash
# Fuka 4.0 — render MP4 for a given RUN_ID (isolated from headless)
set -euo pipefail

RUN_ID="${1:-}"
if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 <RUN_ID>"; exit 2
fi

REPO_DIR="${REPO_DIR:-/root/Fuka-4.0}"
VENV_DIR="${VENV_DIR:-/opt/fuka-venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
ASSETS_DIR="${ASSETS_DIR:-$REPO_DIR/assets}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/exports}"
HTTPS_PREFIX="${HTTPS_PREFIX:-https://storage.googleapis.com/fuka4-runs}"

mkdir -p "$ASSETS_DIR" "$OUT_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 1) Pack NPZ from shards (via HTTPS listing)
"$PYTHON_BIN" -m fuka.render.pack_npz \
  --run_id "$RUN_ID" \
  --https_prefix "$HTTPS_PREFIX" \
  --out "$ASSETS_DIR/fuka_anim_${RUN_ID}_0001_0010.npz" \
  --start 1 --end 10

# 2) Render MP4 with Manim
SCENE_FILE="${REPO_DIR}/fuka/render/scenes/fuka_world.py"
SCENE_NAME="FukaWorldScene"

if [[ ! -f "$SCENE_FILE" ]]; then
  echo "Scene file not found: $SCENE_FILE"; exit 3
fi

manim "$SCENE_FILE" "$SCENE_NAME" -q l -o "fuka_${RUN_ID}.mp4" --media_dir "$OUT_DIR"
echo "MP4 ready: $OUT_DIR/fuka_${RUN_ID}.mp4"