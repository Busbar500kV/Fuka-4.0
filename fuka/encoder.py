from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class EncoderCfg:
    enabled: bool = True
    eta: float = 0.02     # learning rate
    gamma: float = 0.02   # decay
    lam: float = 0.98     # EMA for auxiliary stats
    tau_encode: float = 0.05  # show edges with G above this (UI can override)
    save_every: int = 500     # dump to disk every N steps

class Encoder:
    """
    Maintains long-lived, per-edge 'conductance' G (encoded connections), plus
    small embeddings from temporal stats to color-code edges.
    Stores 3D arrays for axis-aligned edges: Gx (nx-1,ny,nz), Gy (nx,ny-1,nz), Gz (nx,ny,nz-1).
    """
    def __init__(self, world, cfg: EncoderCfg, out_dir: Path):
        self.w = world
        self.cfg = cfg
        self.out_dir = Path(out_dir) / "observer"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        nx,ny,nz = world.nx, world.ny, world.nz
        self.Gx = np.full((nx-1,ny,nz), 0.10, dtype=np.float32)
        self.Gy = np.full((nx,ny-1,nz), 0.10, dtype=np.float32)
        self.Gz = np.full((nx,ny,nz-1), 0.10, dtype=np.float32)
        # simple temporal features for embeddings
        self.Ex = np.zeros_like(self.Gx)  # EMA |phi|
        self.Ey = np.zeros_like(self.Gy)
        self.Ez = np.zeros_like(self.Gz)
        self.Sx = np.zeros_like(self.Gx)  # EMA signed phi
        self.Sy = np.zeros_like(self.Gy)
        self.Sz = np.zeros_like(self.Gz)
        self.Px = np.zeros_like(self.Gx)  # persistence counter proxy
        self.Py = np.zeros_like(self.Gy)
        self.Pz = np.zeros_like(self.Gz)
        self._step = 0
        self._S_prev = float(np.sum(self.w.energy))

    def _advantage(self) -> float:
        S = float(np.sum(self.w.energy))
        A = S - self._S_prev
        self._S_prev = S
        return A

    def _update_axis(self, G, E_pre_a, E_pre_b, Ex, Sx, Px, A):
        # phi = alpha*(b - a); use alpha from physics
        # NOTE: alpha already applied in your physics; here we don’t need absolute scaling, only relative
        phi = (E_pre_b - E_pre_a).astype(np.float32)
        abs_phi = np.abs(phi)
        lam = float(self.cfg.lam)
        # temporal EMAs (for embeddings)
        Ex[:] = lam*Ex + (1.0-lam)*abs_phi
        Sx[:] = lam*Sx + (1.0-lam)*phi
        # persistence proxy: increment where abs_phi significant
        Px[:] = lam*Px + (1.0-lam)*(abs_phi > (Ex.mean() + 1.5*Ex.std())).astype(np.float32)
        # conductance update
        G[:] = G + float(self.cfg.eta)*(abs_phi*A - float(self.cfg.gamma)*G)
        return G, Ex, Sx, Px

    def step(self, E_pre: np.ndarray) -> None:
        if not self.cfg.enabled: return
        nx,ny,nz = self.w.nx, self.w.ny, self.w.nz
        A = self._advantage()

        # X edges: (x,y,z) <-> (x+1,y,z)
        self.Gx, self.Ex, self.Sx, self.Px = self._update_axis(
            self.Gx, E_pre[:-1,:,:], E_pre[1:,:,:], self.Ex, self.Sx, self.Px, A
        )
        # Y edges
        self.Gy, self.Ey, self.Sy, self.Py = self._update_axis(
            self.Gy, E_pre[:,:-1,:], E_pre[:,1:,:], self.Ey, self.Sy, self.Py, A
        )
        # Z edges
        self.Gz, self.Ez, self.Sz, self.Pz = self._update_axis(
            self.Gz, E_pre[:,:,:-1], E_pre[:,:,1:], self.Ez, self.Sz, self.Pz, A
        )

        self._step += 1
        if (self._step % int(self.cfg.save_every)) == 0:
            self.save()

    def save(self) -> None:
        path = self.out_dir / "encoded_edges.npz"
        np.savez_compressed(
            path,
            Gx=self.Gx, Gy=self.Gy, Gz=self.Gz,
            Ex=self.Ex, Ey=self.Ey, Ez=self.Ez,
            Sx=self.Sx, Sy=self.Sy, Sz=self.Sz,
            Px=self.Px, Py=self.Py, Pz=self.Pz,
            tau_encode=np.float32(self.cfg.tau_encode),
            step=np.int32(self._step)
        )