# fuka/guess_field.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from .physics import World3D, _stable_seed_from


@dataclass
class GuessFieldCfg:
    """
    Top-down "reward/punish" stimulus layer.

    Parameters
    ----------
    amplitude : float
        Magnitude of injection (added to world.energy).
    k_fires : int
        Number of voxels to excite per step (default: 0 = off).
    seed : Optional[int]
        RNG seed for reproducibility; if None, derived deterministically.
    mode : str
        "random" -> excite random voxels
        "maxima" -> excite current highest-value voxels
        "minima" -> excite current lowest-value voxels
    decay : float
        Exponential decay factor applied to a persistent hidden mask (0..1).
        Allows the "guess" field to fade in/out over steps.
    """
    amplitude: float = 0.0
    k_fires: int = 0
    seed: Optional[int] = None
    mode: str = "random"
    decay: float = 0.95


class GuessField:
    """
    Maintains a hidden "guess" field overlay that can reward/punish
    particular voxels. Every step:
      - Choose k_fires voxels (mode dependent)
      - Add amplitude to world.energy
      - Update hidden mask with decay
    """
    def __init__(self, world: World3D, cfg: GuessFieldCfg) -> None:
        self.world = world
        self.cfg = cfg

        seed = cfg.seed
        if seed is None:
            seed = _stable_seed_from(("guess", cfg.amplitude, cfg.k_fires, cfg.mode,
                                      cfg.decay, world.nx, world.ny, world.nz))
        self.rng = np.random.default_rng(int(seed))

        self.hidden_mask = np.zeros(world.energy.shape, dtype=np.float32)

    def step(self, step: int, k_fires: Optional[int] = None) -> None:
        c = self.cfg
        if c.k_fires <= 0 and not k_fires:
            return
        amp = float(c.amplitude)
        if amp == 0.0:
            return
        k = int(k_fires if k_fires is not None else c.k_fires)
        if k <= 0:
            return

        E = self.world.energy
        nx, ny, nz = self.world.nx, self.world.ny, self.world.nz
        N = nx * ny * nz

        # flatten indices
        if c.mode == "random":
            idx = self.rng.choice(N, size=min(k, N), replace=False)
        elif c.mode == "maxima":
            flat = E.reshape(-1)
            idx = np.argpartition(flat, -k)[-k:]
        elif c.mode == "minima":
            flat = E.reshape(-1)
            idx = np.argpartition(flat, k)[:k]
        else:
            # unknown mode = fallback random
            idx = self.rng.choice(N, size=min(k, N), replace=False)

        x = (idx // (ny * nz)).astype(np.int64)
        rem = idx % (ny * nz)
        y = (rem // nz).astype(np.int64)
        z = (rem % nz).astype(np.int64)

        self.world.energy[x, y, z] += amp
        self.hidden_mask *= float(c.decay)
        self.hidden_mask[x, y, z] += amp

    def snapshot(self) -> np.ndarray:
        """
        Return a copy of the hidden guess mask for diagnostics/visualization.
        """
        return self.hidden_mask.copy()