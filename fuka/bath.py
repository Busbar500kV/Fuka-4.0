# fuka/bath.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

from .physics import World3D


@dataclass
class BathCfg:
    """
    External bath that rescales the energy field every step.

    Parameters
    ----------
    target_std : float
        Desired standard deviation of energy field after scaling.
        If <=0, bath is inactive.
    rate : float
        Relaxation rate (0..1). 1.0 = instant rescale to target,
        0.0 = no effect.
    """
    target_std: float = 0.0
    rate: float = 1.0


def step_bath(world: World3D, cfg: BathCfg) -> float:
    """
    Rescale world.energy toward target_std with exponential relaxation.

    Returns
    -------
    rho : float
        The actual scaling factor applied (1.0 if bath inactive).
    """
    if cfg is None or cfg.target_std <= 0.0:
        return 1.0

    E = world.energy
    current = float(np.std(E))
    target = float(cfg.target_std)
    if current <= 1e-12:
        return 1.0

    # scale factor needed to reach target std
    raw_rho = target / current
    rate = float(np.clip(cfg.rate, 0.0, 1.0))
    rho = (1.0 - rate) * 1.0 + rate * raw_rho

    # apply
    E *= rho
    return rho