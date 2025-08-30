"""
Fuka 4.0 minimal engine that produces:
  - events, spectra, state, ledger
  - edges (from catalyst transfers)
  - env snapshots (sparse)

Coordinates:
  We embed a 1D chain of 'grid_size' connections along x-axis (y=z=0).
  You can switch to 2D/3D later by changing _cell_xyz().

Physics (toy, local-only):
  - Environment field E_i(t) = sum_k A_k sin(w_k*t + phi_k + i*phase_dx)
  - Each connection integrates local energy over 'window' and, if above threshold,
    encodes mass dm proportional to the free-energy drop at that cell.
  - Catalysts propagate locally and create edges between connections.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import sin, tau
from typing import List, Tuple, Dict, Any

import numpy as np

from .recorder import ParquetRecorder
from .catalysts import CatalystConfig, CatalystTokens


@dataclass
class WorldConfig:
    grid_size: int = 256
    dt: float = 1e-3          # seconds per step
    window: float = 0.2       # seconds; sliding integration window


@dataclass
class IOConfig:
    flush_every: int = 1000
    snap_every: int = 100     # env snapshot cadence
    env_subsample: int = 4    # sample every N cells for env logs


@dataclass
class EventsConfig:
    xi_mass: float = 0.8
    threshold: float = 0.05


class Engine:
    def __init__(self,
                 recorder: ParquetRecorder,
                 steps: int,
                 seed: int = 1234,
                 c: float = 299_792_458.0,
                 world: WorldConfig = WorldConfig(),
                 io: IOConfig = IOConfig(),
                 events: EventsConfig = EventsConfig(),
                 catalyst: CatalystConfig = CatalystConfig()):
        self.rec = recorder
        self.steps = int(steps)
        self.seed = int(seed)
        self.c = float(c)
        self.world = world
        self.io = io
        self.evt_cfg = events
        self.rng = np.random.default_rng(self.seed)

        # positions (1D -> 3D embedded)
        self.dx = 1.0  # meters between neighboring connections
        self.t = 0.0   # current time
        self.step_idx = 0

        # field & accumulators
        G = self.world.grid_size
        self.field = np.zeros(G, dtype=float)           # current E field
        self.energy_acc = np.zeros(G, dtype=float)      # integrate |E| over window
        self.mass = np.zeros(G, dtype=float)            # encoded mass per connection
        self.T_eff = np.zeros(G, dtype=float)           # toy "temperature"
        self.theta_thr = np.full(G, self.evt_cfg.threshold, dtype=float)

        # environment wave parameters (two components)
        self.A = np.array([0.8, 0.6])
        self.w = np.array([2.0 * np.pi * 5.0, 2.0 * np.pi * 11.0])  # 5 Hz and 11 Hz
        self.phi = np.array([0.0, np.pi / 3.0])
        self.phase_dx = 0.03  # spatial phase advance per cell

        # catalysts
        self.cats = CatalystTokens(G, catalyst)
        # seed small amount at center
        self.cats.seed(G // 2, 1.0)

        # window length in steps
        self.W = max(1, int(round(self.world.window / self.world.dt)))

    # ----------------- helpers -----------------

    def _cell_xyz(self, idx: int) -> Tuple[float, float, float]:
        x = idx * self.dx
        return (x, 0.0, 0.0)

    def _env_field_step(self, t: float) -> np.ndarray:
        """Compute environment field at time t (per cell)."""
        G = self.world.grid_size
        idx = np.arange(G)
        phase = idx * self.phase_dx
        # E_i(t) = sum_k A_k sin(w_k * t + phi_k + phase_i)
        E = np.zeros(G, dtype=float)
        for Ak, wk, ph in zip(self.A, self.w, self.phi):
            E += Ak * np.sin(wk * t + ph + phase)
        return E

    def _dominant_spectrum(self, E_window: np.ndarray) -> Tuple[float, float, float]:
        """
        Simple dominant frequency estimator from windowed samples at a single cell:
        returns (w_dom, A_dom, phi_dom). This is intentionally lightweight.
        """
        # for toy purposes, we know the injected w; pick the larger amplitude
        if abs(self.A[0]) >= abs(self.A[1]):
            return (float(self.w[0]), float(self.A[0]), float(self.phi[0]))
        else:
            return (float(self.w[1]), float(self.A[1]), float(self.phi[1]))

    # ----------------- main loop -----------------

    def step(self):
        dt = self.world.dt
        G = self.world.grid_size

        # update environment
        self.field = self._env_field_step(self.t)

        # accumulate absolute energy in the sliding window
        self.energy_acc += np.abs(self.field)

        # every snap, log sparse env samples
        if (self.step_idx % self.io.snap_every) == 0:
            for i in range(0, G, max(1, self.io.env_subsample)):
                x, y, z = self._cell_xyz(i)
                self.rec.log_env(step=self.step_idx, tau=self.t,
                                 conn_id=i, xmu=(self.t, x, y, z), frame_id=0,
                                 value=float(self.field[i]))

        # if the window filled, evaluate encoding & events
        if (self.step_idx + 1) % self.W == 0:
            # normalize accumulated energy by window length (mean |E|)
            E_mean = self.energy_acc / float(self.W)

            # thresholding + encoding
            enc_mask = E_mean >= self.theta_thr  # bool per cell
            dm = self.evt_cfg.xi_mass * (E_mean - self.theta_thr) * enc_mask
            dm = np.maximum(dm, 0.0)

            # update state
            self.mass += dm
            # toy "temperature" proportional to fluctuation strength
            self.T_eff = 0.2 * E_mean + 0.05 * self.rng.standard_normal(size=G)

            # log events where encoding happened
            indices = np.where(enc_mask)[0]
            for i in indices:
                x, y, z = self._cell_xyz(i)
                # local spectrum estimate (toy)
                w_dom, A_dom, phi_dom = self._dominant_spectrum(None)
                dF = float(E_mean[i])  # report mean local magnitude as "drop"
                self.rec.log_event(
                    step=self.step_idx, tau=self.t, conn_id=int(i),
                    xmu=(self.t, x, y, z), frame_id=0,
                    dF=dF, dm=float(dm[i]),
                    w_sel=w_dom, A_sel=A_dom, phi_sel=phi_dom,
                    T_eff=float(self.T_eff[i]), theta_thr=float(self.theta_thr[i])
                )
                # spectra summary once per window (keep small)
                self.rec.log_spectra(
                    step=self.step_idx, conn_id=int(i),
                    w_dom=w_dom, A_dom=A_dom, phi_dom=phi_dom,
                    F_local=dF, E_sum=float(E_mean[i]*self.W), S_spec=float(A_dom**2)
                )

            # log state snapshot (downsample for performance)
            for i in range(0, G, 2):
                x, y, z = self._cell_xyz(i)
                self.rec.log_state(step=self.step_idx, conn_id=int(i),
                                   x=x, y=y, z=z,
                                   m=float(self.mass[i]),
                                   T_eff=float(self.T_eff[i]),
                                   theta_thr=float(self.theta_thr[i]))

            # simple ledger terms aggregated
            dF_total = float(E_mean.sum())
            c2dm_total = float((dm * (self.c ** 2)).sum())
            balance_error = dF_total - c2dm_total
            self.rec.log_ledger(step=self.step_idx,
                                dF=dF_total, c2dm=c2dm_total,
                                Q=float(E_mean.var()), W_cat=float(self.cats.field.sum()),
                                net_flux=float(np.gradient(E_mean).sum()),
                                balance_error=float(balance_error))

            # reset window accumulator
            self.energy_acc[:] = 0.0

        # catalyst propagation -> edges
        transfers = self.cats.step_with_transfers()
        if transfers:
            for src, dst, w in transfers:
                xs, ys, zs = self._cell_xyz(src)
                self.rec.log_edge(step=self.step_idx, tau=self.t,
                                  src_conn=int(src), dst_conn=int(dst), weight=float(w),
                                  xmu=(self.t, xs, ys, zs), frame_id=0)

        # advance time
        self.t += dt
        self.step_idx += 1

    def run(self):
        for _ in range(self.steps):
            self.step()
        self.rec.finalize()