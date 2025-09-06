from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np, json
from .physics import World3D, get_alpha

@dataclass
class ObserverCfg:
    enabled: bool = True
    lambda_edge: float = 0.98  # EMA factor

class Observer:
    def __init__(self, world: World3D, cfg: ObserverCfg, out_dir: Path):
        self.world = world
        self.cfg = cfg
        self.out_dir = Path(out_dir) / "observer"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        nx,ny,nz = world.nx, world.ny, world.nz
        self.Wx = np.zeros((nx-1,ny,nz), dtype=np.float32)
        self.Wy = np.zeros((nx,ny-1,nz), dtype=np.float32)
        self.Wz = np.zeros((nx,ny,nz-1), dtype=np.float32)

    def step(self) -> None:
        if not self.cfg.enabled: return
        E = self.world.energy
        lam = float(self.cfg.lambda_edge)
        # finite-diff flux magnitude proxy ~ alpha * |grad|
        fx = np.abs(E[1:,:,:] - E[:-1,:,:])
        fy = np.abs(E[:,1:,:] - E[:,:-1,:])
        fz = np.abs(E[:,:,1:] - E[:,:,:-1])
        self.Wx = lam*self.Wx + (1.0-lam)*fx
        self.Wy = lam*self.Wy + (1.0-lam)*fy
        self.Wz = lam*self.Wz + (1.0-lam)*fz

    def finalize(self, extra_metrics: dict | None = None) -> None:
        if not self.cfg.enabled: return
        np.savez_compressed(self.out_dir / "effective_edges.npz",
                            Wx=self.Wx, Wy=self.Wy, Wz=self.Wz)
        metrics = {"sum_energy": float(np.sum(self.world.energy))}
        if extra_metrics: metrics.update(extra_metrics)
        (self.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))