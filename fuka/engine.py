# fuka/engine.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .physics import (
    PhysicsCfg, World3D, step_diffuse, detect_local_maxima,
    CatalystsCfg, CatalystsSystem, get_alpha
)
from .bath import BathCfg, step_bath
from .external_source import ExternalSourceCfg, ExternalSource
from .guess_field import GuessFieldCfg, GuessField
from .observer import ObserverCfg, Observer
from .encoder import EncoderCfg, Encoder

# Recorder is provided by runner; we only assume it exposes:
#   add(table: str, rows: List[dict])  and  finalize()
try:
    from .recorder import ParquetRecorder  # type: ignore
except Exception:  # pragma: no cover
    ParquetRecorder = object  # type: ignore


def _cfg_subset(dc_cls, src: Dict[str, Any]) -> Any:
    """Filter a dict to dataclass fields."""
    fields = {f.name for f in dc_cls.__dataclass_fields__.values()}  # type: ignore
    return dc_cls(**{k: v for k, v in (src or {}).items() if k in fields})


def _topk_flat_indices(arr: np.ndarray, k: int) -> np.ndarray:
    k = int(max(0, k))
    n = arr.size
    if k <= 0 or n == 0:
        return np.zeros((0,), dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    # argpartition for speed
    idx = np.argpartition(arr.reshape(-1), -k)[-k:]
    # sort descending by value
    vals = arr.reshape(-1)[idx]
    order = np.argsort(-vals, kind="stable")
    return idx[order]


def _edge_pairs_topk_by_grad(E: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-axis neighbor differences and select global top-k by |delta|.
    Returns arrays: x0,y0,z0,x1,y1,z1,v0,v1  (all 1D of length <=k)
    """
    nx, ny, nz = E.shape
    # X edges
    dx = E[1:,:,:] - E[:-1,:,:]
    dy = E[:,1:,:] - E[:,:-1,:]
    dz = E[:,:,1:] - E[:,:,:-1]
    # Flatten magnitudes with axis tags
    mags = [
        (np.abs(dx).reshape(-1), 'x'),
        (np.abs(dy).reshape(-1), 'y'),
        (np.abs(dz).reshape(-1), 'z'),
    ]
    # total candidates
    total = sum(m.shape[0] for m,_ in mags)
    if k <= 0 or total == 0:
        return tuple(np.zeros((0,), dtype=np.float32) for _ in range(8))  # type: ignore

    k = min(k, total)
    # Concatenate magnitudes and track source axis and local index
    cat = np.concatenate([m for m,_ in mags])
    gidx = _topk_flat_indices(cat, k)
    # Prepare outputs
    x0=[];y0=[];z0=[];x1=[];y1=[];z1=[];v0=[];v1=[]
    # Helper to map a flat index in an axis block to coordinates
    sizes = [dx.size, dy.size, dz.size]
    offsets = np.cumsum([0] + sizes)
    for gi in gidx:
        gi = int(gi)
        if gi < offsets[1]:
            # x edge
            li = gi - offsets[0]
            ix = li // (ny*nz)
            rem = li % (ny*nz)
            iy = rem // nz
            iz = rem % nz
            x0.append(ix); y0.append(iy); z0.append(iz)
            x1.append(ix+1); y1.append(iy); z1.append(iz)
            v0.append(float(E[ix,iy,iz])); v1.append(float(E[ix+1,iy,iz]))
        elif gi < offsets[2]:
            li = gi - offsets[1]
            ix = li // ( (ny-1)*nz )
            rem = li % ( (ny-1)*nz )
            iy = rem // nz
            iz = rem % nz
            x0.append(ix); y0.append(iy); z0.append(iz)
            x1.append(ix); y1.append(iy+1); z1.append(iz)
            v0.append(float(E[ix,iy,iz])); v1.append(float(E[ix,iy+1,iz]))
        else:
            li = gi - offsets[2]
            ix = li // ( ny*(nz-1) )
            rem = li % ( ny*(nz-1) )
            iy = rem // (nz-1)
            iz = rem % (nz-1)
            x0.append(ix); y0.append(iy); z0.append(iz)
            x1.append(ix); y1.append(iy); z1.append(iz+1)
            v0.append(float(E[ix,iy,iz])); v1.append(float(E[ix,iy,iz+1]))
    def arr(a): return np.asarray(a, dtype=np.float32)
    return arr(x0),arr(y0),arr(z0),arr(x1),arr(y1),arr(z1),arr(v0),arr(v1)


@dataclass
class Engine:
    recorder: ParquetRecorder
    steps: int
    cfg: Dict[str, Any]

    def __post_init__(self) -> None:
        c = self.cfg

        # ---- world & physics ----
        wcfg = (c.get("world") or {})
        grid_shape = tuple(int(x) for x in wcfg.get("grid_shape", (64,64,64)))
        pcfg = _cfg_subset(PhysicsCfg, c.get("physics") or {})
        self.world = World3D(grid_shape, pcfg)
        self.pcfg = pcfg

        # ---- subsystems ----
        self.ecfg  = _cfg_subset(ExternalSourceCfg, c.get("external_source") or {})
        self.ext   = ExternalSource(self.world, self.ecfg)

        self.bcfg  = _cfg_subset(BathCfg, c.get("bath") or {})
        self.gfcfg = _cfg_subset(GuessFieldCfg, c.get("guess_field") or {})
        self.guess = GuessField(self.world, self.gfcfg)

        # observer & encoder output directory
        # try to derive from recorder.manifest_path: .../data/runs/<run_id>/manifest.json
        out_dir = None
        run_dir = None
        try:
            manifest_path = Path(getattr(self.recorder, "manifest_path"))
            run_dir = manifest_path.parent
            out_dir = run_dir
        except Exception:
            out_dir = Path("data") / "runs" / "UNKNOWN"

        self.ocfg = _cfg_subset(ObserverCfg, c.get("observer") or {})
        self.observer = Observer(self.world, self.ocfg, out_dir)

        self.encfg = _cfg_subset(EncoderCfg, c.get("encoder") or {})
        self.encoder = Encoder(self.world, self.encfg, out_dir)

        self.ccfg = _cfg_subset(CatalystsCfg, c.get("catalysts") or {})
        self.catalysts = CatalystsSystem(self.world, self.ccfg)

        # ---- IO policy ----
        iocfg = c.get("io") or {}
        self.flush_every = int(iocfg.get("flush_every", 200))
        self.state_topk  = int(iocfg.get("state_topk", 400))
        self.edges_topk  = int(iocfg.get("edges_topk", 0))
        self.event_sigma = float(iocfg.get("event_threshold_sigma", 3.0))
        self.log_env_every = int(iocfg.get("log_env_every", 200))

        # ---- time ----
        self.dt_seconds = float(((c.get("time") or {}).get("dt_seconds", 0.001)))

        # counters
        self._since_flush = 0

        # log initial env
        self._log_env(step=0, extra={"alpha": get_alpha(self.pcfg)})

    # ------------- helpers: recording -------------
    def _rec(self, table: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        # recorder.add API (list[dict])
        self.recorder.add(table, rows)  # type: ignore[attr-defined]
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            # most recorder impls flush on demand per-table; finalize() at end handles any residue
            self._since_flush = 0

    def _log_env(self, step: int, extra: Dict[str, Any] | None = None) -> None:
        rows = [dict(step=int(step),
                     dt_seconds=float(self.dt_seconds),
                     alpha=float(get_alpha(self.pcfg)),
                     nx=int(self.world.nx), ny=int(self.world.ny), nz=int(self.world.nz),
                     **(extra or {}))]
        self._rec("env", rows)

    # ------------- main loop -------------
    def run(self) -> None:
        steps = int(self.steps)
        rng = self.world.rng

        for step in range(steps):
            # 0) external source (deterministic given cfg + seed)
            try:
                self.ext.step(step)
            except Exception:
                pass

            # 1) top-down guess field (DishBrain-inspired)
            try:
                self.guess.step(step, k_fires=1)
            except Exception:
                pass

            # 2) keep a copy before mixing for encoder features
            E_pre = self.world.energy.copy()

            # 3) physics: diffuse + boundary leak + noise
            stats = step_diffuse(self.world, self.pcfg)

            # 4) bath scaling
            try:
                rho = step_bath(self.world, self.bcfg)
            except Exception:
                rho = 0.0

            # 5) catalysts (spawn/walk/deposit/decay)
            try:
                cat_stats = self.catalysts.step()
            except Exception:
                cat_stats = {"spawned": 0, "alive": 0, "total_deposit": 0.0}

            # 6) observer & encoder
            try:
                self.observer.step()
            except Exception:
                pass
            try:
                self.encoder.step(E_pre)
            except Exception:
                pass

            # 7) record sparse state top-k
            if self.state_topk > 0:
                E = self.world.energy
                idx = _topk_flat_indices(E, self.state_topk)
                if idx.size:
                    x = (idx // (self.world.ny * self.world.nz)).astype(np.int64)
                    rem = idx % (self.world.ny * self.world.nz)
                    y = (rem // self.world.nz).astype(np.int64)
                    z = (rem % self.world.nz).astype(np.int64)
                    rows = [dict(step=int(step),
                                 x=int(xi), y=int(yi), z=int(zi),
                                 value=float(E[xi, yi, zi]))
                            for xi, yi, zi in zip(x, y, z)]
                    self._rec("state", rows)

            # 8) record edges by gradient strength (optional)
            if self.edges_topk > 0:
                E = self.world.energy
                x0,y0,z0,x1,y1,z1,v0,v1 = _edge_pairs_topk_by_grad(E, self.edges_topk)
                if x0.size:
                    rows = [dict(step=int(step),
                                 x0=int(x0[i]), y0=int(y0[i]), z0=int(z0[i]),
                                 x1=int(x1[i]), y1=int(y1[i]), z1=int(z1[i]),
                                 v0=float(v0[i]), v1=float(v1[i]))
                            for i in range(x0.size)]
                    self._rec("edges", rows)

            # 9) events via strict local maxima
            try:
                events = []
                for (xi, yi, zi, v) in detect_local_maxima(self.world, sigma_thresh=self.event_sigma):
                    events.append(dict(step=int(step), x=int(xi), y=int(yi), z=int(zi), value=float(v)))
                if events:
                    self._rec("events", events)
            except Exception:
                pass

            # 10) periodic env log
            if self.log_env_every > 0 and (step % self.log_env_every) == 0 and step != 0:
                self._log_env(step, extra={"rho": float(rho)})

        # end loop

        # attempt to write closing env snapshot
        try:
            self._log_env(steps, extra={"done": 1})
        except Exception:
            pass

        # finalize observer/encoder artifacts if possible
        try:
            self.observer.finalize({"rho_last": float(rho) if 'rho' in locals() else 0.0})
        except Exception:
            pass
        try:
            self.encoder.save()
        except Exception:
            pass

        # flush recorder
        self.recorder.finalize()