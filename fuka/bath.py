from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .physics import World3D

@dataclass
class BathCfg:
    enabled: bool = True
    mode: str = "energy"   # "energy" or "std"
    kappa: float = 0.02
    rho_max: float = 0.05

def step_bath(world: World3D, cfg: BathCfg) -> float:
    """Compute rho from the current field and apply uniform scaling."""
    if not cfg.enabled:
        return 0.0
    E = world.energy
    if cfg.mode == "std":
        metric = float(np.std(E))
    else:
        metric = float(np.sum(E))
    rho = float(min(cfg.rho_max, max(0.0, cfg.kappa * metric)))
    world.energy *= (1.0 - rho)
    return rho