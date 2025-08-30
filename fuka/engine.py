from __future__ import annotations
import numpy as np
from typing import Dict
from .physics import spectral_stats, subtract_mode, encode_mass
from .catalysts import CatalystTokens, apply_catalyst_effects

class Engine:
    def __init__(self, cfg: Dict, recorder):
        self.cfg = cfg
        G = cfg["world"]["grid_size"]
        self.grid_size = G
        self.dt = cfg["world"]["dt"]
        self.fs = 1.0 / self.dt
        self.window = cfg["world"]["window"]
        self.window_len = max(8, int(self.window * self.fs))

        rng = np.random.default_rng(cfg["seed"])
        # demo source: mixture of a few sinusoids + noise, spatially varying
        w1, w2 = 2*np.pi*3.0, 2*np.pi*12.0
        self.source_freqs = np.linspace(w1, w2, G)
        self.phase = rng.uniform(0, 2*np.pi, size=G)
        self.amp = np.linspace(0.6, 1.2, G)
        self.noise_std = 0.1

        # state
        self.t = 0.0
        self.step_idx = 0
        self.T_eff = np.full(G, 0.5)          # effective temperature
        self.theta_thr = np.full(G, cfg["events"]["threshold"])
        self.m = np.zeros(G)                  # encoded mass per cell
        self.c = cfg["c"]
        self.rec = recorder

        # catalysts
        cat_cfg = cfg["catalyst"]
        self.cats = CatalystTokens(G, lam=cat_cfg["lambda"],
                                      hop=cat_cfg["speed_cells_per_step"])
        self.beta_cat = cat_cfg["beta"]

        # circular buffers for local windows
        self.buff = np.zeros((G, self.window_len))
        self.buff_pos = 0

        self.xi_mass = cfg["events"]["xi_mass"]

    def _synthesize_step(self):
        # generate one sample per cell (local rest frame simplification)
        n = self.grid_size
        t = self.t
        vals = self.amp * np.cos(self.source_freqs * t + self.phase)
        vals += np.random.normal(0, self.noise_std, size=n)
        return vals

    def step(self):
        # 1) advance world by one dt
        vals = self._synthesize_step()
        self.buff[:, self.buff_pos] = vals
        self.buff_pos = (self.buff_pos + 1) % self.window_len
        self.t += self.dt
        self.step_idx += 1

        # 2) when buffer is "full", process each cell locally
        if self.step_idx < self.window_len: 
            return

        # roll out a contiguous window view per cell
        if self.buff_pos == 0:
            win = self.buff.copy()
        else:
            win = np.concatenate([self.buff[:, self.buff_pos:],
                                  self.buff[:, :self.buff_pos]], axis=1)

        # 3) local analysis & event test
        for idx in range(self.grid_size):
            signal = win[idx]
            Tloc = self.T_eff[idx]
            stats = spectral_stats(signal, fs=self.fs, T_eff=Tloc)
            F_old = stats["F_local"]

            # attractor = remove dominant mode
            residual = subtract_mode(signal, fs=self.fs,
                                     A=stats["A_dom"],
                                     w=stats["w_dom"],
                                     phi=stats["phi_dom"])
            stats2 = spectral_stats(residual, fs=self.fs, T_eff=Tloc)
            F_new = stats2["F_local"]

            dF, dm, Q = encode_mass(F_old, F_new, xi_mass=self.xi_mass, c=self.c)

            # catalyst effects at this location before threshold check
            cat_val = self.cats.field[idx]
            self.theta_thr[idx], self.T_eff[idx] = apply_catalyst_effects(
                self.theta_thr[idx], self.T_eff[idx], cat_val
            )

            if dF > self.theta_thr[idx]:
                # event fires: record, update mass, emit catalysts
                self.m[idx] += dm
                self.cats.emit(idx, dF, beta=self.beta_cat)

                # ledger: net_flux is 0 in this toy (no spatial flux yet)
                self.rec.log_event(
                    step=self.step_idx, tau=self.step_idx*self.dt,
                    conn_id=idx, xmu=(self.t, float(idx), 0.0, 0.0), frame_id=0,
                    dF=dF, dm=dm, Q=Q, W_cat=self.beta_cat*dF,
                    A_sel=stats["A_dom"], w_sel=stats["w_dom"], phi_sel=stats["phi_dom"],
                    theta_thr=float(self.theta_thr[idx]), T_eff=float(self.T_eff[idx]),
                    R_est=0.0
                )
                self.rec.log_ledger(
                    step=self.step_idx, tau=self.step_idx*self.dt,
                    conn_id=idx, xmu=(self.t, float(idx), 0.0, 0.0), frame_id=0,
                    dF=dF, c2dm=(self.c**2)*dm, Q=Q, W_cat=self.beta_cat*dF, net_flux=0.0,
                    balance_error=abs(dF - ((self.c**2)*dm + Q + self.beta_cat*dF))
                )

            # light spectra logging (downsample in real runs)
            if (self.step_idx % 50) == 0:
                self.rec.log_spectra(
                    step=self.step_idx, tau=self.step_idx*self.dt,
                    conn_id=idx, xmu=(self.t, float(idx), 0.0, 0.0), frame_id=0,
                    E_sum=stats["E_sum"], S_spec=stats["S_spec"], F_local=stats["F_local"],
                    A_dom=stats["A_dom"], w_dom=stats["w_dom"], phi_dom=stats["phi_dom"],
                    bandpower_low=0.0, bandpower_mid=0.0, bandpower_high=0.0
                )

            # state snapshots (very sparse)
            if (self.step_idx % 200) == 0:
                self.rec.log_state(
                    step=self.step_idx, tau=self.step_idx*self.dt,
                    conn_id=idx, xmu=(self.t, float(idx), 0.0, 0.0), frame_id=0,
                    m=float(self.m[idx]), C=float(self.cats.field[idx]),
                    T_eff=float(self.T_eff[idx]), theta_thr=float(self.theta_thr[idx])
                )

        # 4) propagate catalysts (finite speed, decay)
        self.cats.step()