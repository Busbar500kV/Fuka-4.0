from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np

from .physics import World3D, PhysicsCfg, step_diffuse, detect_local_maxima


class Engine:
    """
    3D Engine (backward compatible):
      - If cfg['world']['grid_shape'] is present -> use (nx,ny,nz)
      - Else if cfg['world']['grid_size'] present -> interpret as (N,1,1)
      - Else fallback to (128,1,1)

    Logs:
      - env:   physics params every 1000 steps (and step 0)
      - state: top-K energetic cells per step (x,y,z,value)
      - edges: gradient edges from top-K to max neighbour (optional-ish viz)
      - events: strict 6-neighbour local maxima above mean+sigma*std
      - spectra: energy histogram (bins) per step
      - ledger: rolled-up totals (sum/mean/std/max)
    """
    def __init__(self, recorder, steps: int, cfg: Dict[str, Any] | None = None) -> None:
        self.rec = recorder
        self.steps = int(steps)
        self.cfg = cfg or {}

        # world shape
        world_cfg = self.cfg.get("world", {})
        if "grid_shape" in world_cfg:
            nx,ny,nz = world_cfg["grid_shape"]
        else:
            n = int(world_cfg.get("grid_size", 256))
            nx,ny,nz = int(n), 1, 1  # backward-compat 1D -> embedded 3D

        # physics cfg
        phy = self.cfg.get("physics", {})
        self.pcfg = PhysicsCfg(
            T=float(phy.get("T", 0.0015)),
            flux_limit=float(phy.get("flux_limit", 0.20)),
            boundary_leak=float(phy.get("boundary_leak", 0.01)),
            radius=int(phy.get("radius", 1)),
            seed=phy.get("seed", None),
        )

        self.world = World3D((nx,ny,nz), self.pcfg)

        # IO knobs
        io = self.cfg.get("io", {})
        self.state_topk = int(io.get("state_topk", 256))
        self.edges_topk = int(io.get("edges_topk", 512))
        self.spectra_bins = int(io.get("spectra_bins", 64))
        self.event_sigma = float(io.get("event_threshold_sigma", 3.0))
        self.log_env_every = int(io.get("log_env_every", 1000))

    # ------------ helpers ------------
    def _topk_cells(self, k: int) -> List[Tuple[int,int,int,float]]:
        E = self.world.energy
        k = int(max(1, min(k, E.size)))
        flat_idx = np.argpartition(E.ravel(), -k)[-k:]
        flat_idx = flat_idx[np.argsort(E.ravel()[flat_idx])][::-1]  # sort descending
        coords_vals: List[Tuple[int,int,int,float]] = []
        nx, ny = self.world.nx, self.world.ny
        for idx in flat_idx:
            x = int(idx // (ny*self.world.nz))
            rem = int(idx % (ny*self.world.nz))
            y = int(rem // self.world.nz)
            z = int(rem % self.world.nz)
            coords_vals.append((x,y,z,float(E[x,y,z])))
        return coords_vals

    def _best_neighbour(self, x: int, y: int, z: int) -> Tuple[int,int,int,float] | None:
        """Return neighbour with highest energy (6-connectivity)."""
        best = None
        best_v = -1e30
        for xn, yn, zn in self.world.neighbors6(x,y,z):
            v = float(self.world.energy[xn,yn,zn])
            if v > best_v:
                best_v = v; best = (xn,yn,zn,v)
        return best

    # ------------ main loop ------------
    def run(self) -> None:
        # initial env log (step 0)
        self.rec.log_env(step=0, nx=self.world.nx, ny=self.world.ny, nz=self.world.nz,
                         T=self.pcfg.T, flux_limit=self.pcfg.flux_limit,
                         boundary_leak=self.pcfg.boundary_leak, radius=self.pcfg.radius)

        for step in range(1, self.steps + 1):
            # physics update
            stats = step_diffuse(self.world)

            # ledger (rollups)
            self.rec.log_ledger(step=step, **stats)

            # spectra (histogram)
            counts, edges = np.histogram(self.world.energy, bins=self.spectra_bins)
            self.rec.log_spectrum(
                step=step,
                bin_edges=edges.astype(float).tolist(),
                counts=counts.astype(int).tolist(),
            )

            # state (top-K energetic cells)
            for (x,y,z,v) in self._topk_cells(self.state_topk):
                self.rec.log_state(step=step, x=int(x), y=int(y), z=int(z), value=float(v))

            # edges (each top cell -> strongest neighbour)
            e_logged = 0
            for (x,y,z,v) in self._topk_cells(min(self.edges_topk, self.state_topk)):
                nb = self._best_neighbour(x,y,z)
                if nb is None: continue
                xn,yn,zn,vn = nb
                if (xn,yn,zn) == (x,y,z): continue
                self.rec.log_edge(step=step,
                                  x0=int(x), y0=int(y), z0=int(z), v0=float(v),
                                  x1=int(xn),y1=int(yn),z1=int(zn), v1=float(vn))
                e_logged += 1
                if e_logged >= self.edges_topk: break

            # events (strict local maxima by sigma)
            for (x,y,z,v) in detect_local_maxima(self.world, sigma_thresh=self.event_sigma):
                self.rec.log_event(step=step, x=int(x), y=int(y), z=int(z), value=float(v))

            # env every N steps
            if (step % self.log_env_every) == 0:
                self.rec.log_env(step=step, nx=self.world.nx, ny=self.world.ny, nz=self.world.nz,
                                 T=self.pcfg.T, flux_limit=self.pcfg.flux_limit,
                                 boundary_leak=self.pcfg.boundary_leak, radius=self.pcfg.radius)

        self.rec.finalize()