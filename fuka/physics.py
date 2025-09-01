from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Any


@dataclass
class PhysicsCfg:
    T: float = 0.0015           # noise std dev
    flux_limit: float = 0.20    # smoothing / diffusion factor in [0,1]
    boundary_leak: float = 0.01 # fractional loss at boundary faces per step
    radius: int = 1             # neighbour radius (currently used for peak checks)
    seed: int | None = None     # RNG seed (None = nondeterministic)


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
        # small random init
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
    One update step:
      1) Add white noise (std = T).
      2) 6-neighbour isotropic diffusion (flux_limit = alpha).
      3) Boundary leak at the 6 faces.
    Returns cheap diagnostics for logging.
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
    Simple peak detector: a cell is an event if it's greater than all 6-neighbours AND > mean + sigma_thresh*std.
    """
    E = world.energy
    mean = float(np.mean(E))
    std = float(np.std(E)) + 1e-12
    cut = mean + sigma_thresh * std

    nx, ny, nz = world.nx, world.ny, world.nz
    # To keep it fast, only scan interior cells when checking 6-neighbour strict maxima.
    for x in range(1, nx-1):
        for y in range(1, ny-1):
            for z in range(1, nz-1):
                v = float(E[x,y,z])
                if v <= cut: continue
                if v > E[x+1,y,z] and v > E[x-1,y,z] and v > E[x,y+1,z] and v > E[x,y-1,z] and v > E[x,y,z+1] and v > E[x,y,z-1]:
                    yield (x,y,z,v)
