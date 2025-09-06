from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Any, List


# ----------------------------
# Core physics configuration
# ----------------------------
@dataclass
class PhysicsCfg:
    T: float = 0.0015           # white-noise std dev per step
    flux_limit: float = 0.20    # 6-neighbour diffusion blend in [0,1]
    boundary_leak: float = 0.01 # fractional loss at boundary faces
    radius: int = 1             # reserved (peak-locality), not used by the solver here
    seed: int | None = None     # RNG seed for reproducibility


class World3D:
    """
    A simple 3D scalar field 'energy' on an axis-aligned grid.
    """
    def __init__(self, grid_shape: Tuple[int, int, int], cfg: PhysicsCfg) -> None:
        self.grid_shape = tuple(int(x) for x in grid_shape)
        assert len(self.grid_shape) == 3 and all(s > 0 for s in self.grid_shape), "grid_shape must be (nx,ny,nz)>0"
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.energy = np.zeros(self.grid_shape, dtype=np.float32)
        # small random init for symmetry breaking
        self.energy += self.rng.normal(0.0, 0.01, self.grid_shape).astype(np.float32)

    @property
    def nx(self) -> int: return self.grid_shape[0]
    @property
    def ny(self) -> int: return self.grid_shape[1]
    @property
    def nz(self) -> int: return self.grid_shape[2]

    def neighbors6(self, x: int, y: int, z: int) -> Iterable[Tuple[int,int,int]]:
        if x+1 < self.nx: yield (x+1, y, z)
        if x-1 >= 0    : yield (x-1, y, z)
        if y+1 < self.ny: yield (x, y+1, z)
        if y-1 >= 0    : yield (x, y-1, z)
        if z+1 < self.nz: yield (x, y, z+1)
        if z-1 >= 0    : yield (x, y, z-1)

    def iter_coords(self) -> Iterable[Tuple[int,int,int]]:
        for x in range(self.nx):
            for y in range(self.ny):
                for z in range(self.nz):
                    yield (x,y,z)


def step_diffuse(world: World3D) -> Dict[str, Any]:
    """
    One update step of the background field:
      1) Add white noise (std = T)
      2) 6-neighbour isotropic diffusion (flux_limit = alpha)
      3) Boundary leak at the 6 faces
    Returns diagnostics for logging.
    """
    cfg = world.cfg
    E = world.energy

    # (1) noise
    if cfg.T > 0:
        E = E + world.rng.normal(0.0, cfg.T, world.grid_shape).astype(np.float32)

    # (2) diffusion with six neighbours (fast & in-bounds)
    alpha = float(np.clip(cfg.flux_limit, 0.0, 1.0))
    if alpha > 0:
        nx, ny, nz = world.nx, world.ny, world.nz
        mean = np.zeros_like(E, dtype=np.float32)
        count = np.zeros_like(E, dtype=np.int32)

        def add_shift(dx: int, dy: int, dz: int) -> None:
            xs = slice(max(0, dx), nx - max(0, -dx))
            ys = slice(max(0, dy), ny - max(0, -dy))
            zs = slice(max(0, dz), nz - max(0, -dz))
            xs_src = slice(max(0, -dx), nx - max(0, dx))
            ys_src = slice(max(0, -dy), ny - max(0, dy))
            zs_src = slice(max(0, -dz), nz - max(0, dz))
            mean[xs, ys, zs] += E[xs_src, ys_src, zs_src]
            count[xs, ys, zs] += 1

        for dx,dy,dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            add_shift(dx,dy,dz)

        mask = count > 0
        mean[mask] = mean[mask] / count[mask]
        E = (1.0 - alpha) * E
        E[mask] += alpha * mean[mask]

    # (3) boundary leak
    leak = float(np.clip(cfg.boundary_leak, 0.0, 1.0))
    if leak > 0:
        E[ 0, :, :] *= (1.0 - leak)
        E[-1, :, :] *= (1.0 - leak)
        E[:,  0, :] *= (1.0 - leak)
        E[:, -1, :] *= (1.0 - leak)
        E[:, :,  0] *= (1.0 - leak)
        E[:, :, -1] *= (1.0 - leak)

    world.energy = E

    return {
        "sum_energy": float(np.sum(E)),
        "mean_energy": float(np.mean(E)),
        "max_energy": float(np.max(E)),
        "std_energy": float(np.std(E)),
    }


def detect_local_maxima(world: World3D, sigma_thresh: float = 3.0) -> Iterable[Tuple[int,int,int,float]]:
    """
    A cell is an event if it's strictly greater than all 6-neighbours AND > mean + sigma_thresh*std.
    Checks interior cells only for speed.
    """
    E = world.energy
    mean = float(np.mean(E))
    std = float(np.std(E)) + 1e-12
    cut = mean + sigma_thresh * std

    nx, ny, nz = world.nx, world.ny, world.nz
    for x in range(1, nx-1):
        for y in range(1, ny-1):
            for z in range(1, nz-1):
                v = float(E[x,y,z])
                if v <= cut: continue
                if (v > E[x+1,y,z] and v > E[x-1,y,z] and
                    v > E[x,y+1,z] and v > E[x,y-1,z] and
                    v > E[x,y,z+1] and v > E[x,y,z-1]):
                    yield (x,y,z,v)


# ----------------------------
# Catalysts (3D)
# ----------------------------
@dataclass
class CatalystsCfg:
    enabled: bool = True
    max_count: int = 256
    spawn_prob_base: float = 0.08  # per step probability, scaled by (1 - n/max)
    strength: float = 0.75         # base injection strength
    radius: float = 3.0            # gaussian sigma (in grid cells)
    decay_per_step: float = 0.0025 # multiplicative decay of strength
    jitter_std: float = 0.6        # random-walk step std (cells)
    drift: Tuple[float,float,float] = (0.0,0.0,0.0)  # deterministic drift vector
    reflect_at_boundary: bool = True
    min_strength: float = 0.05     # drop when below this
    deposit_clip: float = 2.0      # clip added energy per deposit (stability)


class CatalystsSystem:
    """
    A population of point catalysts that wander (random walk + drift) and inject
    localized energy via a 3D Gaussian kernel each step. Strength decays, and
    new catalysts spawn stochastically up to max_count.
    """
    def __init__(self, world: World3D, cfg: CatalystsCfg) -> None:
        self.world = world
        self.cfg = cfg
        self.rng = world.rng  # share RNG with world
        self.pos: List[np.ndarray] = []  # float positions [x,y,z]
        self.strength: List[float] = []

    def _spawn_one(self) -> None:
        x = self.rng.uniform(0, self.world.nx-1)
        y = self.rng.uniform(0, self.world.ny-1)
        z = self.rng.uniform(0, self.world.nz-1)
        self.pos.append(np.array([x,y,z], dtype=np.float32))
        self.strength.append(float(self.cfg.strength))

    def _maybe_spawn(self) -> int:
        n = len(self.pos)
        if n >= self.cfg.max_count:
            return 0
        # scale probability down as we approach the cap
        p = max(0.0, float(self.cfg.spawn_prob_base) * (1.0 - n / max(1, self.cfg.max_count)))
        k = 0
        if self.rng.random() < p:
            self._spawn_one()
            k = 1
        return k

    def _reflect_or_clamp(self, p: np.ndarray) -> np.ndarray:
        if self.cfg.reflect_at_boundary:
            # reflect at faces: if p < 0 -> -p; if p > max -> 2*max - p
            if p[0] < 0: p[0] = -p[0]
            if p[1] < 0: p[1] = -p[1]
            if p[2] < 0: p[2] = -p[2]
            if p[0] > self.world.nx-1: p[0] = 2*(self.world.nx-1) - p[0]
            if p[1] > self.world.ny-1: p[1] = 2*(self.world.ny-1) - p[1]
            if p[2] > self.world.nz-1: p[2] = 2*(self.world.nz-1) - p[2]
        else:
            p[0] = np.clip(p[0], 0, self.world.nx-1)
            p[1] = np.clip(p[1], 0, self.world.ny-1)
            p[2] = np.clip(p[2], 0, self.world.nz-1)
        return p

    def _deposit_gaussian(self, center: np.ndarray, amp: float, sigma: float) -> float:
        """
        Add a truncated 3D Gaussian blob centered near 'center' into world.energy.
        Returns the total added energy (for diagnostics).
        """
        x0, y0, z0 = center
        nx, ny, nz = self.world.nx, self.world.ny, self.world.nz

        # truncate at ~3 sigma for efficiency
        r = max(1.0, sigma) * 3.0
        xmin = int(max(0, np.floor(x0 - r)))
        xmax = int(min(nx-1, np.ceil (x0 + r)))
        ymin = int(max(0, np.floor(y0 - r)))
        ymax = int(min(ny-1, np.ceil (y0 + r)))
        zmin = int(max(0, np.floor(z0 - r)))
        zmax = int(min(nz-1, np.ceil (z0 + r)))
        if xmin > xmax or ymin > ymax or zmin > zmax:
            return 0.0

        xs = np.arange(xmin, xmax+1, dtype=np.float32)
        ys = np.arange(ymin, ymax+1, dtype=np.float32)
        zs = np.arange(zmin, zmax+1, dtype=np.float32)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        # isotropic gaussian
        inv2s2 = 1.0 / (2.0 * (sigma**2 + 1e-12))
        G = np.exp(-((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2) * inv2s2).astype(np.float32)
        blob = (amp * G).astype(np.float32)

        if self.cfg.deposit_clip is not None:
            np.clip(blob, -float(self.cfg.deposit_clip), float(self.cfg.deposit_clip), out=blob)

        self.world.energy[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] += blob
        return float(np.sum(blob))

    def update(self) -> Dict[str, Any]:
        """
        One catalysts step:
          - Spawn (probabilistic)
          - Random walk + optional drift, with boundary handling
          - Deposit energy blobs
          - Decay strength and prune weak
        """
        spawned = self._maybe_spawn()

        if not self.pos:
            return {"spawned": spawned, "alive": 0, "total_deposit": 0.0}

        total_deposit = 0.0
        sigma = float(max(1e-6, self.cfg.radius))
        decay = float(np.clip(self.cfg.decay_per_step, 0.0, 1.0))
        jitter_std = float(max(0.0, self.cfg.jitter_std))
        drift = np.array(self.cfg.drift, dtype=np.float32)

        new_pos: List[np.ndarray] = []
        new_strength: List[float] = []

        for p, s in zip(self.pos, self.strength):
            # move
            dp = self.rng.normal(0.0, jitter_std, size=3).astype(np.float32) + drift
            p = p + dp
            p = self._reflect_or_clamp(p)

            # deposit
            added = self._deposit_gaussian(p, amp=s, sigma=sigma)
            total_deposit += added

            # decay
            s = s * (1.0 - decay)

            if s >= self.cfg.min_strength:
                new_pos.append(p)
                new_strength.append(s)

        self.pos = new_pos
        self.strength = new_strength

        # small chance to spawn >1 when population is very low
        if len(self.pos) == 0 and self.cfg.enabled and self.cfg.max_count > 0:
            if self.rng.random() < 0.5:
                self._spawn_one(); spawned += 1

        return {"spawned": spawned, "alive": len(self.pos), "total_deposit": float(total_deposit)}

# ---- additive helper for observer/flux math (non-breaking) ----
def get_alpha(cfg: PhysicsCfg) -> float:
    """Return the numeric diffusion blend in [0,1] as used by step_diffuse."""
    import numpy as np
    return float(np.clip(cfg.flux_limit, 0.0, 1.0))