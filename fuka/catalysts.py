from __future__ import annotations
import numpy as np

class CatalystTokens:
    """Lightweight token field on a 1D grid for demo."""
    def __init__(self, size: int, lam: float, hop: int):
        self.size = size
        self.decay = lam
        self.hop = hop
        self.field = np.zeros(size, dtype=float)

    def emit(self, idx: int, energy_drop: float, beta: float):
        self.field[idx] += beta * energy_drop

    def step(self):
        # finite-speed spread (hop cells left/right), then decay
        new = np.zeros_like(self.field)
        for i, val in enumerate(self.field):
            if val == 0: continue
            L = max(0, i - self.hop); R = min(self.size-1, i + self.hop)
            share = val / (R-L+1)
            new[L:R+1] += share
        self.field = new * (1.0 - self.decay)


def apply_catalyst_effects(theta_thr: float, T_eff: float, cat_val: float,
                           zeta: float=0.05, epsT: float=0.05):
    theta_new = max(0.0, theta_thr * (1.0 - zeta*cat_val))
    T_new = max(1e-6, T_eff * (1.0 - epsT*cat_val))
    return theta_new, T_new