from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .physics import World3D

@dataclass
class Sinusoid:
    center: Tuple[float, float, float]
    amp: float
    sigma: float
    # either freq_hz with dt_seconds, or freq_cps (cycles per step)
    freq_hz: Optional[float] = None
    freq_cps: Optional[float] = None
    phase_deg: float = 0.0
    t_on: int = 0
    t_off: int = 2**31 - 1  # large default

@dataclass
class ExternalSourceCfg:
    enabled: bool = True
    # Pulses: (t0, t1, x, y, z, amp, sigma)
    pulses: Optional[List[Tuple[int,int,float,float,float,float,float]]] = None
    # Functional sinusoids (sum of components)
    sinusoids: Optional[List[Dict[str, Any]]] = None
    clip: float = 2.0
    dt_seconds: float = 0.001  # default 1 ms per step if freq_hz is used

class ExternalSource:
    def __init__(self, world: World3D, cfg: ExternalSourceCfg):
        self.world = world
        self.cfg = cfg
        self._pulses = cfg.pulses or []
        self._sins: List[Sinusoid] = []
        if cfg.sinusoids:
            for s in cfg.sinusoids:
                self._sins.append(
                    Sinusoid(
                        center=tuple(map(float, s.get("center", (world.nx/2, world.ny/2, world.nz/2)))),
                        amp=float(s.get("amp", 0.0)),
                        sigma=float(s.get("sigma", 3.0)),
                        freq_hz=s.get("freq_hz", None),
                        freq_cps=s.get("freq_cps", None),
                        phase_deg=float(s.get("phase_deg", 0.0)),
                        t_on=int(s.get("t_on", 0)),
                        t_off=int(s.get("t_off", 2**31-1)),
                    )
                )

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

    def _sin_amp_at(self, s: Sinusoid, t_step: int) -> float:
        if not (s.t_on <= t_step <= s.t_off):
            return 0.0
        phase_rad = np.deg2rad(s.phase_deg)
        if s.freq_cps is not None:
            # cycles per step (native discrete frequency)
            omega = 2.0 * np.pi * float(s.freq_cps) * t_step
        elif s.freq_hz is not None:
            # physical Hz using dt_seconds
            omega = 2.0 * np.pi * float(s.freq_hz) * (t_step * float(self.cfg.dt_seconds))
        else:
            return 0.0
        return float(s.amp * np.sin(omega + phase_rad))

    def step(self, t: int) -> float:
        """Deposit pulses + sinusoidal components at time step t; returns total added energy."""
        if not self.cfg.enabled:
            return 0.0
        total = 0.0
        # Pulses
        for (t0,t1,x,y,z,amp,sigma) in self._pulses:
            if t0 <= t <= t1:
                total += self._deposit_gaussian((x,y,z), float(amp), float(sigma))
        # Sinusoidal components
        for s in self._sins:
            a = self._sin_amp_at(s, t)
            if a != 0.0:
                total += self._deposit_gaussian(s.center, a, s.sigma)
        return total