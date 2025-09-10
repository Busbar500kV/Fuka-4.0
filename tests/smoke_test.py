# tests/smoke_test.py
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

# Import from the repo
from fuka.engine import Engine
from fuka.recorder import ParquetRecorder
from analytics.build_indices import rebuild_indices
from fuka.render.pack_npz import pack as pack_npz


@pytest.mark.quick
def test_end_to_end_smoke(tmp_path: Path) -> None:
    """
    End-to-end smoke test:
      Engine.run() -> parquet shards
      -> rebuild_indices()
      -> pack_npz(prefix='.') to local NPZ
      -> assert canonical NPZ keys
    """
    data_root = tmp_path / "data"
    run_id = "FUKA_SMOKE_TEST"
    steps = 6

    # Minimal config (small grid, short run, encoder not required for NPZ pack)
    cfg = {
        "world": {"grid_shape": [12, 12, 6]},
        "physics": {
            "T": 0.0015,
            "flux_limit": 0.2,
            "boundary_leak": 0.01,
            "update_mode": "euler",
            "alpha": 0.18,
        },
        "bath": {"target_std": 0.2, "rate": 0.4},
        "external_source": {"kind": "pulse", "amplitude": 0.03, "every": 2, "coord": [6, 6, 3]},
        "guess_field": {"amplitude": 0.02, "k_fires": 1, "mode": "random", "decay": 0.9},
        "catalysts": {"spawn_rate": 0.2, "decay_p": 0.05, "deposit": 0.01, "walk_sigma": 0.7, "max_count": 500},
        "observer": {"log_every": 0},
        "encoder": {
            "enabled": True,
            "every": 2,
            "topk_edges": 300,   # helps generate edges parquet via engine's recorder too if you enable it there
            "cumulative": True,
            "out_subdir": "enc",
            "max_edges_total": 100000,
        },
        "io": {
            "flush_every": 5,
            "state_topk": 300,   # ensure we always have 'state' shards
            "edges_topk": 200,   # ensure we always have 'edges' shards
            "event_threshold_sigma": 2.5,
            "log_env_every": 3,
        },
        "time": {"dt_seconds": 0.001},
        "run": {"steps": steps},
    }

    # Wire recorder + engine directly (avoids relying on runner module pathing)
    rec = ParquetRecorder(data_root=str(data_root), run_id=run_id)
    engine = Engine(recorder=rec, steps=steps, cfg=cfg)
    engine.run()

    # Verify shards exist
    shards_dir = data_root / "runs" / run_id / "shards"
    assert shards_dir.is_dir(), "Shards directory was not created"
    parquet_files = list(shards_dir.glob("*.parquet"))
    assert parquet_files, "No parquet shards were written"

    # Rebuild indices (local relative paths)
    mapping = rebuild_indices(str(data_root), run_id, prefix=None)
    assert "state" in mapping, "State table missing in index"
    assert "edges" in mapping, "Edges table missing in index"
    assert len(mapping["state"]) > 0
    assert len(mapping["edges"]) > 0

    # Pack NPZ using local files (prefix='.' yields relative paths)
    out_npz = tmp_path / "fuka_anim_smoke.npz"
    pack_npz(prefix=".", run_id=run_id, step_min=0, step_max=steps - 1, out_path=str(out_npz))
    assert out_npz.exists(), "NPZ packing did not produce output"

    # Load NPZ and validate canonical keys + basic shape invariants
    z = np.load(out_npz)
    required_keys = [
        "steps",
        "state_x", "state_y", "state_z", "state_value", "state_idx",
        "edges_x0", "edges_y0", "edges_z0", "edges_x1", "edges_y1", "edges_z1", "edges_idx",
    ]
    for k in required_keys:
        assert k in z.files, f"Missing NPZ key: {k}"

    # Edge canonicals are optional when there are no edges; here edges_topk>0 so they should exist
    assert "edge_value" in z.files, "Missing edge_value (canonical)"
    assert "edge_strength" in z.files, "Missing edge_strength (canonical)"

    steps_arr = z["steps"]
    sidx = z["state_idx"]
    eidx = z["edges_idx"]
    assert steps_arr.ndim == 1 and steps_arr.size == steps, "Steps vector shape mismatch"
    assert sidx.ndim == 1 and sidx.size == steps + 1, "state_idx must be length steps+1"
    assert eidx.ndim == 1 and eidx.size == steps + 1, "edges_idx must be length steps+1"

    # prefix sums must be non-decreasing
    assert np.all(sidx[1:] >= sidx[:-1]), "state_idx is not non-decreasing"
    assert np.all(eidx[1:] >= eidx[:-1]), "edges_idx is not non-decreasing"