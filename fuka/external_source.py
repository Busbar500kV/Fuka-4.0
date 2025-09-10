# fuka/external_source.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List, Dict, Any
import numpy as np

from .physics import World3D, _stable_seed_from


SourceKind = Literal[
    "off",          # no-op
    "pulse",        # periodic delta injection at fixed coords
    "sin",          # sinusoidal drive at fixed coords
    "random",       # white-noise injection across the field
    "mask_pulse",   # pulse over a static binary mask
    "moving_pulse", # pulse whose center moves linearly
]


@dataclass
class ExternalSourceCfg:
    """
    Deterministic stimulus injector for the energy field.

    Common
    ------
    kind            : SourceKind
    seed            : Optional[int]      # RNG seed; if None, derived from cfg
    amplitude       : float              # base amplitude of injection
    start_step      : int                # first step (inclusive)
    stop_step       : Optional[int]      # last step (inclusive); None = forever
    every           : int                # apply every N steps (>=1)

    Pulse/Sin
    ---------
    coord           : Tuple[int,int,int] # voxel coordinate (x,y,z)
    omega_hz        : float              # for kind="sin" (cycles/sec equivalent)
    dt_seconds      : float              # timestep seconds (engine passes via cfg)

    Random
    ------
    random_frac     : float              # fraction of voxels to excite per call (0..1)

    Mask pulse
    ---------
    mask_seed       : Optional[int]      # to generate deterministic mask
    mask_frac       : float              # fraction of voxels set to 1 (0..1)

    Moving pulse
    ------------
    start_coord     : Tuple[int,int,int]
    end_coord       : Tuple[int,int,int]
    """
    kind: SourceKind = "off"
    seed: Optional[int] = None
    amplitude: float = 0.0
    start_step: int = 0
    stop_step: Optional[int] = None
    every: int = 1

    # pulse / sin
    coord: Tuple[int, int, int] = (0, 0, 0)
    omega_hz: float = 1.0
    dt_seconds: float = 0.001

    # random
    random_frac: float = 0.0

    # mask pulse
    mask_seed: Optional[int] = None
    mask_frac: float = 0.0

    # moving pulse
    start_coord: Tuple[int, int, int] = (0, 0, 0)
    end_coord: Tuple[int, int, int] = (0, 0, 0)


class ExternalSource:
    """
    Stateless interface from the Engine's POV; internally holds RNG and (optional) mask.
    All effects are **additive** to world.energy (float32).
    """
    def __init__(self, world: World3D, cfg: ExternalSourceCfg) -> None:
        self.world = world
        self.cfg = cfg

        # Deterministic RNG
        seed = cfg.seed
        if seed is None:
            seed = _stable_seed_from((
                "ext", cfg.kind, cfg.amplitude, cfg.start_step, cfg.stop_step, cfg.every,
                cfg.coord, cfg.omega_hz, cfg.dt_seconds, cfg.random_frac,
                cfg.mask_seed, cfg.mask_frac, cfg.start_coord, cfg.end_coord,
                world.nx, world.ny, world.nz
            ))
        self.rng = np.random.default_rng(int(seed))

        # Precompute mask if needed
        self._mask: Optional[np.ndarray] = None
        if cfg.kind == "mask_pulse":
            self._mask = self._make_mask(cfg)

    # ------------------ public API ------------------

    def step(self, step: int) -> None:
        c = self.cfg
        if c.kind == "off":
            return
        if step < c.start_step:
            return
        if c.stop_step is not None and step > c.stop_step:
            return
        if c.every <= 0 or (step % max(1, c.every)) != 0:
            return
        amp = float(c.amplitude)
        if amp == 0.0:
            return

        if c.kind == "pulse":
            self._inject_pulse(c.coord, amp)
        elif c.kind == "sin":
            self._inject_sin(c.coord, amp, step)
        elif c.kind == "random":
            self._inject_random(amp, frac=c.random_frac)
        elif c.kind == "mask_pulse":
            self._inject_mask_pulse(amp)
        elif c.kind == "moving_pulse":
            self._inject_moving_pulse(amp, step)
        else:
            # unknown kind -> no-op (forward compatible)
            return

    # ------------------ implementations ------------------

    def _inject_pulse(self, coord: Tuple[int,int,int], amp: float) -> None:
        x, y, z = self._clamp_coord(coord)
        self.world.energy[x, y, z] += amp

    def _inject_sin(self, coord: Tuple[int,int,int], amp: float, step: int) -> None:
        # sin(2π f t), t = step * dt
        t = float(step) * float(self.cfg.dt_seconds)
        val = float(np.sin(2.0 * np.pi * float(self.cfg.omega_hz) * t))
        x, y, z = self._clamp_coord(coord)
        self.world.energy[x, y, z] += amp * val

    def _inject_random(self, amp: float, frac: float) -> None:
        frac = float(np.clip(frac, 0.0, 1.0))
        if frac <= 0.0:
            return
        nx, ny, nz = self.world.nx, self.world.ny, self.world.nz
        N = nx * ny * nz
        k = int(np.round(frac * N))
        if k <= 0:
            return
        # sample k distinct flat indices (deterministic RNG)
        idx = self.rng.choice(N, size=k, replace=False)
        x = (idx // (ny * nz)).astype(np.int64)
        rem = idx % (ny * nz)
        y = (rem // nz).astype(np.int64)
        z = (rem % nz).astype(np.int64)
        self.world.energy[x, y, z] += amp

    def _inject_mask_pulse(self, amp: float) -> None:
        if self._mask is None:
            return
        self.world.energy[self._mask] += amp

    def _inject_moving_pulse(self, amp: float, step: int) -> None:
        # Linear interpolation in index space from start_coord -> end_coord over active window
        c = self.cfg
        t0 = int(c.start_step)
        t1 = int(c.stop_step) if c.stop_step is not None else t0
        if t1 <= t0:
            # degenerate or single step window
            coord = c.end_coord if step >= t1 else c.start_coord
            x, y, z = self._clamp_coord(coord)
            self.world.energy[x, y, z] += amp
            return
        # normalized phase
        u = np.clip((step - t0) / float(t1 - t0), 0.0, 1.0)
        p0 = np.array(c.start_coord, dtype=float)
        p1 = np.array(c.end_coord, dtype=float)
        p = (1.0 - u) * p0 + u * p1
        coord = tuple(int(round(v)) for v in p.tolist())
        x, y, z = self._clamp_coord(coord)
        self.world.energy[x, y, z] += amp

    # ------------------ helpers ------------------

    def _clamp_coord(self, coord: Tuple[int,int,int]) -> Tuple[int,int,int]:
        x, y, z = coord
        x = int(np.clip(x, 0, self.world.nx - 1))
        y = int(np.clip(y, 0, self.world.ny - 1))
        z = int(np.clip(z, 0, self.world.nz - 1))
        return x, y, z

    def _make_mask(self, cfg: ExternalSourceCfg) -> np.ndarray:
        nx, ny, nz = self.world.nx, self.world.ny, self.world.nz
        frac = float(np.clip(cfg.mask_frac, 0.0, 1.0))
        N = nx * ny * nz
        k = int(np.round(frac * N))
        mask = np.zeros((nx, ny, nz), dtype=bool)
        if k <= 0:
            return mask
        seed = cfg.mask_seed
        if seed is None:
            seed = _stable_seed_from(("ext_mask", cfg.mask_frac, nx, ny, nz))
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(N, size=k, replace=False)
        x = (idx // (ny * nz)).astype(np.int64)
        rem = idx % (ny * nz)
        y = (rem // nz).astype(np.int64)
        z = (rem % nz).astype(np.int64)
        mask[x, y, z] = True
        return mask