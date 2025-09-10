# Fuka-4.0
First Universal Kommon Ancestor - යසස් පොන්වීර 

# Fuka 4.0 — Headless Simulation → NPZ → Manim Render

This repo implements a fully headless physics sandbox for evolving 3D fields,
recording sparse connections, and rendering them as 3D animations.

---

## Pipeline overview

1. **Run simulation (headless)**
   - Config: JSON under `configs/`
   - Runner: `python -m fuka.runner`
   - Outputs: `data/runs/<RUN_ID>/shards/*.parquet`

2. **Rebuild indices/manifest**
   - `python -m analytics.build_indices`
   - Produces per-table `*_index.json` and `manifest.json`

3. **Pack to NPZ**
   - `python -m fuka.render.pack_npz`
   - Merges shards (via `duckdb` httpfs) into a single `assets/fuka_anim.npz`

4. **Render with Manim**
   - `manim render/manim_fuka_scene.py Fuka3DScene`
   - Reads canonical NPZ keys and produces MP4

---

## Quick start

1. Install deps
```bash
pip install -r requirements.txt


2. Run the full pipeline
export DATA_URL_PREFIX="https://storage.googleapis.com/fuka4-runs"

bash scripts/run_fuka_pipeline.sh \
  --config configs/demo_3d.json \
  --data_root data \
  --run_id FUKA_4_0_3D_DEMO \
  --prefix "$DATA_URL_PREFIX" \
  --out_npz assets/fuka_anim.npz \
  --quality k --fps 24

3. Render style knobs (via env vars)

export FUKA_COLOR_KEY=edge_strength   # edge_strength | edge_value
export FUKA_MAX_EDGES=10000
export FUKA_CAM_PHI=65
export FUKA_CAM_THETA=-45
export FUKA_CAM_ZOOM=1.1
export FUKA_DEBUG_AXES=0
export FUKA_WIRE_ALPHA=0.9
export FUKA_STRENGTH_GAMMA=0.6
export FUKA_WIDTH_MIN=1.2
export FUKA_WIDTH_MAX=4.0

4. Outputs
	•	Run data → data/runs/<RUN_ID>/shards/*.parquet
	•	Indices → data/runs/<RUN_ID>/shards/*_index.json
	•	Manifest → data/runs/<RUN_ID>/manifest.json
	•	Encoded edges → data/runs/<RUN_ID>/enc/edges_step_*.npz, edges_all.npz
	•	Animation → media/videos/render/manim_fuka_scene/Fuka3DScene.mp4

⸻

Example configs
	•	configs/demo_3d.json → small 48×48×32 grid, 400 steps, moving external source,
catalysts, bath scaling, observer logging, encoder writing cumulative NPZ.

⸻

Developer notes
	•	Engine is deterministic given config + seeds (World3D, catalysts, external source, guess field).
	•	All per-step encodings are cumulative (every frame includes connections up to that step).
	•	Recorder writes robust Parquet shards with per-table indices + manifest.
	•	Manim scene consumes canonical NPZ schema (edge_value, edge_strength).

⸻

Useful scripts
	•	scripts/render_fuka.sh — pack + render (given an existing run)
	•	scripts/run_fuka_pipeline.sh — full pipeline (simulate → index → pack → render)

⸻

