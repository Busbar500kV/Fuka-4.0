from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
from .external_source import ExternalSource, ExternalSourceCfg
from .bath import step_bath, BathCfg
from .guess_field import GuessField, GuessFieldCfg
from .observer import Observer, ObserverCfg
from pathlib import Path
from .encoder import Encoder, EncoderCfg
from pathlib import Path

from .physics import (
    World3D, PhysicsCfg, step_diffuse, detect_local_maxima,
    CatalystsCfg, CatalystsSystem
)


class Engine:
    """
    3D Engine (backward compatible):
      - If cfg['world']['grid_shape'] present -> use (nx,ny,nz)
      - Else if cfg['world']['grid_size'] present -> interpret as (N,1,1)
      - Else fallback to (128,1,1)

    Logs each step:
      - ledger: sum/mean/std/max of energy (+ catalyst totals)
      - spectra: histogram (counts + bin_edges)
      - state: top-K cells by energy (x,y,z,value)
      - edges: gradient edge to strongest 6-neighbour (subset of top-K)
      - events: strict 6-neighbour local maxima above mean + sigma*std
      - env: world + physics params (every N steps and at step 0)
      - catalysts: (cid,x,y,z,strength)
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

        # catalysts cfg
        cat_cfg = self.cfg.get("catalysts", {})
        self.ccfg = CatalystsCfg(
            enabled=bool(cat_cfg.get("enabled", True)),
            max_count=int(cat_cfg.get("max_count", 256)),
            spawn_prob_base=float(cat_cfg.get("spawn_prob_base", 0.08)),
            strength=float(cat_cfg.get("strength", 0.75)),
            radius=float(cat_cfg.get("radius", 3.0)),
            decay_per_step=float(cat_cfg.get("decay_per_step", 0.0025)),
            jitter_std=float(cat_cfg.get("jitter_std", 0.6)),
            drift=tuple(cat_cfg.get("drift", (0.0,0.0,0.0))),
            reflect_at_boundary=bool(cat_cfg.get("reflect_at_boundary", True)),
            min_strength=float(cat_cfg.get("min_strength", 0.05)),
            deposit_clip=float(cat_cfg.get("deposit_clip", 2.0)),
        )
        self.catalysts = CatalystsSystem(self.world, self.ccfg)

        # ---- NEW: external source / guess field / bath / observer ----
        time_cfg = self.cfg.get("time", {})
        dt_seconds = float(time_cfg.get("dt_seconds", 0.001))  # default 1 ms/step
        
        ext_cfg = self.cfg.get("external_source", {})
        self.ext = ExternalSource(
            self.world,
            ExternalSourceCfg(
                enabled=bool(ext_cfg.get("enabled", True)),
                pulses=ext_cfg.get("pulses", []),
                sinusoids=ext_cfg.get("sinusoids", []),
                clip=float(ext_cfg.get("clip", 2.0)),
                dt_seconds=dt_seconds,
            )
        )


        
       
        
        gf_cfg = self.cfg.get("guess_field", {})
        self.guess = GuessField(self.world, GuessFieldCfg(
            enabled=bool(gf_cfg.get("enabled", True)),
            eta=float(gf_cfg.get("eta", 0.2)),
            decay=float(gf_cfg.get("decay", 0.05)),
            diffuse=float(gf_cfg.get("diffuse", 0.02)),
            s0=float(gf_cfg.get("s0", 0.5)),
            sigma_c=float(gf_cfg.get("sigma_c", 3.0)),
            sigma_reward=float(gf_cfg.get("sigma_reward", 1.5)),
            beta_jitter=float(gf_cfg.get("beta_jitter", 2.0)),
        ), harvest_mask=None)
        
        bath_cfg = self.cfg.get("bath", {})
        self.bath = BathCfg(
            enabled=bool(bath_cfg.get("enabled", True)),
            mode=str(bath_cfg.get("mode", "energy")),
            kappa=float(bath_cfg.get("kappa", 0.02)),
            rho_max=float(bath_cfg.get("rho_max", 0.05)),
        )
        
        obs_cfg = self.cfg.get("observer", {})
        # recorder exposes run_dir path; we keep it as-is
        run_dir = Path(self.rec.run_dir) if hasattr(self.rec, "run_dir") else Path(".")
        self.observer = Observer(self.world, ObserverCfg(
            enabled=bool(obs_cfg.get("enabled", True)),
            lambda_edge=float(obs_cfg.get("lambda_edge", 0.98)),
        ), out_dir=run_dir)
        
        encfg = self.cfg.get("encoder", {})
        run_dir = Path(self.rec.run_dir) if hasattr(self.rec, "run_dir") else Path(".")
        self.encoder = Encoder(
            self.world,
            EncoderCfg(
                enabled=bool(encfg.get("enabled", True)),
                eta=float(encfg.get("eta", 0.02)),
                gamma=float(encfg.get("gamma", 0.02)),
                lam=float(encfg.get("lam", 0.98)),
                tau_encode=float(encfg.get("tau_encode", 0.05)),
                save_every=int(encfg.get("save_every", 500))
            ),
            out_dir=run_dir
        )


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
                         boundary_leak=self.pcfg.boundary_leak, radius=self.pcfg.radius,
                         catalysts_enabled=int(self.ccfg.enabled))

        for step in range(1, self.steps + 1):
            
            # 0) external pulses (time-varying source)
            try:
                self.ext.step(step)
            except Exception:
                pass  # never break a run
            
            # 1) top-down catalyst guess (DishBrain-style structured vs noisy)
            try:
                self.guess.step(step, k_fires=1)
            except Exception:
                pass
            
            # 2.1) capture a copy of the field for pre-mix flux
            E_pre = self.world.energy.copy()
            
            # 2.2) background field update (unchanged physics)
            stats = step_diffuse(self.world)
            
            # 3) global bath dissipation (existence coupling)
            try:
                rho = step_bath(self.world, self.bath)
                # let guess field see post-bath energy for reward/penalty update
                self.guess.post_bath_update()
            except Exception:
                rho = 0.0
            
            # 4) existing wandering catalysts (as before)
            cat_totals = {"spawned": 0, "alive": 0, "total_deposit": 0.0}
            if self.ccfg.enabled:
                cat_totals = self.catalysts.update()
            
            # 5) intrinsic observer accumulation
            try:
                # encoded connections update uses pre-mix field
                self.encoder.step(E_pre)
                # periodic checkpoint so encoded_edges.npz exists mid-run
                if self.encoder.cfg.enabled and (step % max(1, self.encoder.cfg.save_every)) == 0:
                    self.encoder.save()

            except Exception:
                pass
            
            try:
                self.observer.step()
            except Exception:
                pass

            # ledger (rollups)
            self.rec.log_ledger(step=step, **stats, cat_alive=int(cat_totals["alive"]),
                                cat_spawned=int(cat_totals["spawned"]),
                                cat_deposit=float(cat_totals["total_deposit"]))

            # spectra (histogram)
            counts, edges = np.histogram(self.world.energy, bins=self.spectra_bins)
            self.rec.log_spectrum(
                step=step,
                bin_edges=edges.astype(float).tolist(),
                counts=counts.astype(int).tolist(),
            )

            # catalysts (positions/strengths)
            if self.ccfg.enabled and self.catalysts.pos:
                for cid, (p, s) in enumerate(zip(self.catalysts.pos, self.catalysts.strength)):
                    self.rec.log_catalyst(step=step, cid=int(cid),
                                          x=float(p[0]), y=float(p[1]), z=float(p[2]),
                                          strength=float(s))

            # state (top-K energetic cells)
            top_state = self._topk_cells(self.state_topk)
            for (x,y,z,v) in top_state:
                self.rec.log_state(step=step, x=int(x), y=int(y), z=int(z), value=float(v))

            # edges (each top cell -> strongest neighbour)
            e_logged = 0
            for (x,y,z,v) in top_state[:min(self.edges_topk, len(top_state))]:
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

            # env periodic
            if (step % self.log_env_every) == 0:
                self.rec.log_env(step=step, nx=self.world.nx, ny=self.world.ny, nz=self.world.nz,
                                 T=self.pcfg.T, flux_limit=self.pcfg.flux_limit,
                                 boundary_leak=self.pcfg.boundary_leak, radius=self.pcfg.radius,
                                 catalysts_enabled=int(self.ccfg.enabled))

        try:
            self.observer.finalize({"rho_last": float(rho) if 'rho' in locals() else 0.0})
        except Exception:
            pass

        try:
            self.encoder.save()
        except Exception:
            pass
        
        self.rec.finalize()
