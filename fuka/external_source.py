from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .physics import World3D

@dataclass
class ExternalSourceCfg:
    enabled: bool = True
    # pulses: list of (t0, t1, x, y, z, amp, sigma)
    pulses: List[Tuple[int,int,float,float,float,float,float]] = None
    clip: float = 2.0

class ExternalSource:
    def __init__(self, world: World3D, cfg: ExternalSourceCfg):
        self.world = world
        self.cfg = cfg
        self._pulses = cfg.pulses or []

    def _deposit_gaussian(self, center, amp, sigma):
        x0, y0, z0 = center
        nx, ny, nz = self.world.nx, self.world.ny, self.world.nz
        r = max(1.0, sigma) * 3.0
        xmin = int(max(0, np.floor(x0 - r))); xmax = int(min(nx-1, np.ceil(x0 + r)))
        ymin = int(max(0, np.floor(y0 - r))); ymax = int(min(ny-1, np.ceil(y0 + r)))
        zmin = int(max(0, np.floor(z0 - r))); zmax = int(min(nz-1, np.ceil(z0 + r)))
        if xmin>xmax or ymin>ymax or zmin>zmax: return 0.0
        xs = np.arange(xmin, xmax+1, dtype=np.float32)
        ys = np.arange(ymin, ymax+1, dtype=np.float32)
        zs = np.arange(zmin, zmax+1, dtype=np.float32)
        X,Y,Z = np.meshgrid(xs, ys, zs, indexing="ij")
        inv2s2 = 1.0 / (2.0 * (sigma**2 + 1e-12))
        G = np.exp(-((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2) * inv2s2).astype(np.float32)
        blob = (amp * G).astype(np.float32)
        if self.cfg.clip is not None:
            np.clip(blob, -float(self.cfg.clip), float(self.cfg.clip), out=blob)
        self.world.energy[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] += blob
        return float(np.sum(blob))

    def step(self, t: int) -> float:
        """Deposit all pulses active at step t; returns total added energy."""
        if not self.cfg.enabled or not self._pulses: return 0.0
        total = 0.0
        for (t0,t1,x,y,z,amp,sigma) in self._pulses:
            if t0 <= t <= t1:
                total += self._deposit_gaussian((x,y,z), float(amp), float(sigma))
        return total