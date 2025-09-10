# fuka/physics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple, Any

import numpy as np


# ============================================================
# Configuration dataclasses
# ============================================================

@dataclass
class PhysicsCfg:
    """
    Base field dynamics.
      - T:          additive Gaussian noise std (thermodynamic "temperature")
      - flux_limit: per-step clamp on absolute Laplacian flux (stability guard)
      - boundary_leak: fraction leaked at domain boundary per step (0..1)
      - update_mode: "euler" (explicit) or "random" (random-site micro-steps, kept for compatibility)
      - alpha: diffusion strength (0..1], if None derived from grid via get_alpha()
    """
    T: float = 0.0015
    flux_limit: float = 0.20
    boundary_leak: float = 0.01
    update_mode: str = "euler"
    alpha: float | None = None


@dataclass
class CatalystsCfg:
    """
    Minimal, headless-friendly catalyst system.
      - spawn_rate: expected births per step (Poisson)
      - decay_p:    probability of removal per step
      - deposit:    energy deposit per visit (added to world.energy at position)
      - walk_sigma: std of Gaussian step in voxel units (clamped to 1-neighbour hops)
      - max_count:  hard cap on simultaneous catalysts
      - seed:       RNG seed for reproducibility
    """
    spawn_rate: float = 0.0
    decay_p: float = 0.01
    deposit: float = 0.0
    walk_sigma: float = 0.7
    max_count: int = 10_000
    seed: int | None = None


# ============================================================
# World
# ============================================================

class World3D:
    """
    Holds the primary scalar field (energy) and RNG.
    All arrays are float32 for memory + speed; indices are int64.
    """
    def __init__(self, grid_shape: Tuple[int, int, int], pcfg: PhysicsCfg) -> None:
        self.nx, self.ny, self.nz = [int(max(1, v)) for v in grid_shape]
        self.energy = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

        # Deterministic RNG derived from grid + physics signature
        seed = _stable_seed_from(("physics", self.nx, self.ny, self.nz, pcfg.T, pcfg.flux_limit, pcfg.boundary_leak, pcfg.update_mode, pcfg.alpha))
        self.rng = np.random.default_rng(seed)

    def clamp_coords(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.clip(x, 0, self.nx - 1)
        y = np.clip(y, 0, self.ny - 1)
        z = np.clip(z, 0, self.nz - 1)
        return x, y, z


# ============================================================
# Helpers
# ============================================================

def _stable_seed_from(items: Iterable[Any]) -> int:
    """
    Deterministic 64-bit seed from a tuple of hashables.
    """
    import hashlib, struct
    h = hashlib.blake2b(repr(tuple(items)).encode("utf-8"), digest_size=8).digest()
    return struct.unpack("<Q", h)[0] & ((1 << 63) - 1)


def get_alpha(pcfg: PhysicsCfg) -> float:
    """
    Diffusion coefficient used by step_diffuse().
    If pcfg.alpha is given, use it; otherwise return a conservative default.
    """
    if pcfg.alpha is not None:
        return float(pcfg.alpha)
    # conservative default suited for explicit 6-neighbour Laplacian
    return 0.18


# ============================================================
# Core physics
# ============================================================

def step_diffuse(world: World3D, pcfg: PhysicsCfg) -> Dict[str, float]:
    """
    One explicit Euler diffusion step with 6-neighbour Laplacian,
    boundary leakage, per-step flux limiting, and additive Gaussian noise.

    Returns step statistics for diagnostics.
    """
    E = world.energy
    nx, ny, nz = world.nx, world.ny, world.nz
    alpha = float(get_alpha(pcfg))

    # 6-neighbour Laplacian (Neumann in interior)
    lap = np.zeros_like(E, dtype=np.float32)
    # x
    lap[1:-1, :, :] += E[2:, :, :] - 2 * E[1:-1, :, :] + E[0:-2, :, :]
    # y
    lap[:, 1:-1, :] += E[:, 2:, :] - 2 * E[:, 1:-1, :] + E[:, 0:-2, :]
    # z
    lap[:, :, 1:-1] += E[:, :, 2:] - 2 * E[:, :, 1:-1] + E[:, :, 0:-2]

    # Flux limiting for stability on large gradients
    if pcfg.flux_limit is not None and pcfg.flux_limit > 0:
        lim = float(pcfg.flux_limit)
        np.clip(lap, -lim, +lim, out=lap)

    # Explicit update
    E += alpha * lap

    # Boundary leakage (simple multiplicative decay at faces)
    leak = float(max(0.0, min(1.0, pcfg.boundary_leak)))
    if leak > 0.0:
        E[0, :, :] *= (1.0 - leak)
        E[-1, :, :] *= (1.0 - leak)
        E[:, 0, :] *= (1.0 - leak)
        E[:, -1, :] *= (1.0 - leak)
        E[:, :, 0] *= (1.0 - leak)
        E[:, :, -1] *= (1.0 - leak)

    # Additive noise (temperature)
    T = float(max(0.0, pcfg.T))
    if T > 0.0:
        E += world.rng.normal(0.0, T, size=E.shape).astype(np.float32)

    # Return diagnostics
    return {
        "alpha": alpha,
        "leak": leak,
        "T": T,
        "mean": float(np.mean(E)),
        "std": float(np.std(E)),
        "min": float(np.min(E)),
        "max": float(np.max(E)),
    }


def detect_local_maxima(world: World3D, sigma_thresh: float = 3.0) -> Iterator[Tuple[int, int, int, float]]:
    """
    Yield strict 6-neighbour local maxima above (mean + sigma_thresh * std).

    This is vectorized and memory-safe. It returns an iterator of tuples:
      (x, y, z, value)
    """
    E = world.energy
    mu = float(np.mean(E))
    sd = float(np.std(E))
    thr = mu + float(sigma_thresh) * sd

    # Prepare neighbour shifts
    # Interior-only comparisons to avoid padding overhead
    core = E[1:-1, 1:-1, 1:-1]

    gt_xm = core > E[0:-2, 1:-1, 1:-1]
    gt_xp = core > E[2:,   1:-1, 1:-1]
    gt_ym = core > E[1:-1, 0:-2, 1:-1]
    gt_yp = core > E[1:-1, 2:,   1:-1]
    gt_zm = core > E[1:-1, 1:-1, 0:-2]
    gt_zp = core > E[1:-1, 1:-1, 2:]

    is_local_max = gt_xm & gt_xp & gt_ym & gt_yp & gt_zm & gt_zp & (core > thr)

    if not np.any(is_local_max):
        return iter(())  # empty iterator

    # Coordinates relative to the full grid
    xi, yi, zi = np.nonzero(is_local_max)
    xv = xi + 1
    yv = yi + 1
    zv = zi + 1
    vals = core[is_local_max]

    # Yield as simple tuples
    for x, y, z, v in zip(xv, yv, zv, vals):
        yield int(x), int(y), int(z), float(v)


# ============================================================
# Catalysts system (minimal, deterministic)
# ============================================================

class CatalystsSystem:
    """
    Lightweight particle system:
      - Poisson births
      - Bernoulli decays
      - Gaussian-walk clamped to 6-neighbour hops
      - Optional energy deposit at visited voxel
    """
    def __init__(self, world: World3D, cfg: CatalystsCfg) -> None:
        self.world = world
        self.cfg = cfg
        seed = _stable_seed_from(("catalysts", world.nx, world.ny, world.nz, cfg.spawn_rate, cfg.decay_p, cfg.deposit, cfg.walk_sigma, cfg.max_count, cfg.seed))
        self.rng = np.random.default_rng(seed if cfg.seed is None else int(cfg.seed))
        self.pos: np.ndarray = np.zeros((0, 3), dtype=np.int64)  # N x {x,y,z}

    def _spawn(self) -> int:
        if self.cfg.spawn_rate <= 0.0 or self.pos.shape[0] >= self.cfg.max_count:
            return 0
        lam = float(self.cfg.spawn_rate)
        births = int(self.rng.poisson(lam=lam))
        if births <= 0:
            return 0
        births = int(min(births, max(0, self.cfg.max_count - self.pos.shape[0])))
        if births <= 0:
            return 0
        xs = self.rng.integers(0, self.world.nx, size=births, endpoint=False)
        ys = self.rng.integers(0, self.world.ny, size=births, endpoint=False)
        zs = self.rng.integers(0, self.world.nz, size=births, endpoint=False)
        batch = np.stack([xs, ys, zs], axis=1).astype(np.int64)
        if self.pos.size == 0:
            self.pos = batch
        else:
            self.pos = np.concatenate([self.pos, batch], axis=0)
        return births

    def _decay(self) -> int:
        if self.pos.size == 0 or self.cfg.decay_p <= 0.0:
            return 0
        p = float(self.cfg.decay_p)
        keep = self.rng.random(self.pos.shape[0]) > p
        removed = int(self.pos.shape[0] - np.count_nonzero(keep))
        if removed > 0:
            self.pos = self.pos[keep]
        return removed

    def _walk_and_deposit(self) -> Tuple[int, float]:
        if self.pos.size == 0:
            return 0, 0.0

        # Gaussian proposal rounded to nearest axis step, clamped to {-1,0,1}
        # Prefer axis-aligned 6-neighbour moves
        s = max(1e-6, float(self.cfg.walk_sigma))
        d = self.rng.normal(0.0, s, size=self.pos.shape)
        d = np.round(d).astype(np.int64)
        d = np.clip(d, -1, 1)

        # break ties to keep 6-neighbour: zero out two axes if multiple non-zeros
        nonzero = (d != 0).astype(np.int64)
        too_many = np.sum(nonzero, axis=1) > 1
        if np.any(too_many):
            # keep the axis with largest |delta|, zero the others
            sel = np.argmax(np.abs(d[too_many]), axis=1)
            mask = np.zeros_like(d[too_many])
            mask[np.arange(mask.shape[0]), sel] = 1
            d[too_many] = d[too_many] * mask

        # apply move
        self.pos[:, 0] += d[:, 0]
        self.pos[:, 1] += d[:, 1]
        self.pos[:, 2] += d[:, 2]
        self.pos[:, 0], self.pos[:, 1], self.pos[:, 2] = self.world.clamp_coords(
            self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]
        )

        total_deposit = 0.0
        dep = float(self.cfg.deposit)
        if dep != 0.0:
            # vectorized in-place deposit
            x, y, z = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]
            self.world.energy[x, y, z] += dep
            total_deposit = float(dep) * float(self.pos.shape[0])

        return int(self.pos.shape[0]), total_deposit

    def step(self) -> Dict[str, float | int]:
        spawned = self._spawn()
        removed = self._decay()
        alive, total_deposit = self._walk_and_deposit()
        return {
            "spawned": int(spawned),
            "removed": int(removed),
            "alive": int(alive),
            "total_deposit": float(total_deposit),
        }