# fuka/observer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path


@dataclass
class ObserverCfg:
    """
    Optional observer for diagnostics.

    Parameters
    ----------
    log_every : int
        Interval of steps at which to compute stats (0 = off).
    out_dir : Optional[str]
        Directory to drop CSV/NPZ diagnostics. If None, in-memory only.
    """
    log_every: int = 0
    out_dir: Optional[str] = None


class Observer:
    """
    Lightweight, headless-safe observer.
    - At chosen intervals, computes global stats of world.energy.
    - Optionally appends them to a CSV in out_dir.
    - Keeps last_stats for programmatic access.
    """
    def __init__(self, world, cfg: ObserverCfg, run_dir: Path) -> None:
        self.world = world
        self.cfg = cfg
        self.run_dir = Path(cfg.out_dir or run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.last_stats: Dict[str, Any] = {}
        self._rows: list[Dict[str, Any]] = []
        self._csv_path = self.run_dir / "observer_stats.csv" if cfg.log_every > 0 else None

    def step(self) -> None:
        if self.cfg.log_every <= 0:
            return
        step = getattr(self.world, "step_counter", None)
        if step is None:
            # allow engine to pass step explicitly if needed
            return
        if (step % self.cfg.log_every) != 0:
            return

        E = self.world.energy
        stats = {
            "step": int(step),
            "mean": float(np.mean(E)),
            "std": float(np.std(E)),
            "min": float(np.min(E)),
            "max": float(np.max(E)),
        }
        self.last_stats = stats
        self._rows.append(stats)

    def finalize(self, extra: Dict[str, Any] | None = None) -> None:
        if self._csv_path is None or not self._rows:
            return
        import pandas as pd
        df = pd.DataFrame(self._rows)
        if extra:
            for k, v in extra.items():
                df[k] = v
        df.to_csv(self._csv_path, index=False)
        self._rows.clear()