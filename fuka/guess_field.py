from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .physics import World3D

@dataclass
class GuessFieldCfg:
    enabled: bool = True
    eta: float = 0.2            # learning rate on K
    decay: float = 0.05         # K decay per step
    diffuse: float = 0.02       # K diffusion per step
    s0: float = 0.5             # deposit amplitude baseline
    sigma_c: float = 3.0        # deposit size (cells)
    sigma_reward: float = 1.5   # local smoothing scale (cells) for reward proxy
    beta_jitter: float = 2.0    # annealing strength

class GuessField:
    def __init__(self, world: World3D, cfg: GuessFieldCfg, harvest_mask: Optional[np.ndarray] = None):
        self.world = world
        self.cfg = cfg
        self.rng = world.rng
        self.K = np.zeros_like(world.energy, dtype=np.float32)  # “preference” field
        self.prev_local = np.zeros_like(world.energy, dtype=np.float32)
        self.mask = (harvest_mask.astype(np.float32) if harvest_mask is not None else None)

    def _softmax_choice(self) -> tuple[int,int,int]:
        K = self.K if self.mask is None else self.K * (0.1 + self.mask)
        flat = (K - np.max(K)).ravel()
        p = np.exp(flat).astype(np.float64)
        s = p.sum()
        if not np.isfinite(s) or s <= 0.0:
            x = self.rng.integers(0, self.world.nx)
            y = self.rng.integers(0, self.world.ny)
            z = self.rng.integers(0, self.world.nz)
            return int(x),int(y),int(z)
        idx = int(self.rng.choice(p.size, p=p/s))
        x = idx // (self.world.ny * self.world.nz)
        rem = idx % (self.world.ny * self.world.nz)
        y = rem // self.world.nz
        z = rem % self.world.nz
        return int(x),int(y),int(z)

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
        self.world.energy[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] += blob
        return float(np.sum(blob))

    def _local_smooth(self, E: np.ndarray, sigma: float) -> np.ndarray:
        # Fast separable 3x3x3 kernel approximating Gaussian
        k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        A = E
        for axis in (0,1,2):
            A = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, A)
        return A

    def step(self, t: int, k_fires: int = 1) -> dict:
        if not self.cfg.enabled: return {"fires": 0, "amp": 0.0}
        total_amp = 0.0
        for _ in range(max(1, k_fires)):
            x,y,z = self._softmax_choice()
            # Anneal amplitude by local post-bath energy (prev_local holds last post-bath E)
            loc = float(self.prev_local[x,y,z])
            amp = float(self.cfg.s0) * (1.0 + np.tanh(self.cfg.beta_jitter * loc))
            total_amp += abs(amp)
            self._deposit_gaussian((x,y,z), amp=amp, sigma=float(self.cfg.sigma_c))
        # Reward proxy: local smoothed energy after this step (will be set from engine)
        return {"fires": k_fires, "amp": total_amp}

    def post_bath_update(self) -> None:
        """Update K using existence-based local reward; then decay/diffuse K."""
        E = self.world.energy
        R = self._local_smooth(E, self.cfg.sigma_reward)
        # Reward: increase K where local E rose vs previous; penalty flip where it fell
        delta = (R - self.prev_local)
        self.prev_local = R.copy()
        self.K += float(self.cfg.eta) * np.maximum(0.0, delta)
        self.K -= float(self.cfg.eta) * 0.5 * np.maximum(0.0, -delta)  # mild penalty
        # Decay
        self.K *= (1.0 - float(self.cfg.decay))
        # Diffuse K (6-neighbour average blend)
        if self.cfg.diffuse > 0.0:
            a = float(self.cfg.diffuse)
            K = self.K
            nx,ny,nz = self.world.nx, self.world.ny, self.world.nz
            mean = np.zeros_like(K, dtype=np.float32)
            cnt = np.zeros_like(K, dtype=np.int32)
            def add(dx,dy,dz):
                xs=slice(max(0,dx), nx-max(0,-dx))
                ys=slice(max(0,dy), ny-max(0,-dy))
                zs=slice(max(0,dz), nz-max(0,-dz))
                xs2=slice(max(0,-dx), nx-max(0,dx))
                ys2=slice(max(0,-dy), ny-max(0,dy))
                zs2=slice(max(0,-dz), nz-max(0,dz))
                mean[xs,ys,zs] += K[xs2,ys2,zs2]; cnt[xs,ys,zs] += 1
            for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                add(*d)
            msk = cnt>0
            mean[msk] = mean[msk] / cnt[msk]
            self.K = (1.0 - a) * K
            self.K[msk] += a * mean[msk]