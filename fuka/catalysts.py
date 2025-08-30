# fuka/catalysts.py
"""
Catalyst token field that diffuses / propagates locally and returns transfers.

We keep it intentionally simple and local:
  - self.field[i] holds "catalyst tokens" at connection i
  - On each step, we spread tokens to neighbors within 'hop' cells
  - We return a list of (src_idx, dst_idx, share) for UI edges
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class CatalystConfig:
    beta: float = 0.5          # growth factor
    lam: float = 0.1           # decay rate per step
    speed_cells_per_step: int = 1  # hop radius


class CatalystTokens:
    def __init__(self, size: int, cfg: CatalystConfig):
        self.size = int(size)
        self.cfg = cfg
        self.field = np.zeros(self.size, dtype=float)

    def seed(self, idx: int, amount: float):
        if 0 <= idx < self.size:
            self.field[idx] += float(amount)

    def step_with_transfers(self) -> list[tuple[int, int, float]]:
        """Spread locally; return list of transfers for edge logging."""
        hop = max(0, int(self.cfg.speed_cells_per_step))
        if hop == 0:
            # no spread, just decay/growth
            self.field = (1.0 - self.cfg.lam) * (self.field * (1.0 + self.cfg.beta))
            return []

        transfers: list[tuple[int, int, float]] = []
        new = np.zeros_like(self.field)

        for i, val in enumerate(self.field):
            if val <= 0.0:
                continue
            L = max(0, i - hop)
            R = min(self.size - 1, i + hop)
            span = R - L + 1
            share = val / span
            for j in range(L, R + 1):
                new[j] += share
                transfers.append((i, j, float(share)))

        # simple growth/decay on the moved mass
        new = (1.0 - self.cfg.lam) * (new * (1.0 + self.cfg.beta))
        self.field = new
        return transfers