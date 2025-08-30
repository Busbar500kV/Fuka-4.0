from __future__ import annotations
import numpy as np
from scipy.signal import get_window
from typing import Dict, Tuple

def spectral_stats(signal: np.ndarray, fs: float, T_eff: float) -> Dict:
    """Return E_sum, S_spec, F_local, dominant mode (A, ω, φ)."""
    n = len(signal)
    w = get_window("hann", n)
    xw = signal * w
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mags = np.abs(X)

    # mode energy proxy (∝ ω^2 A^2). Use |X|^2 scaled by ω^2
    e = (mags**2) * (2*np.pi*freqs)**2
    e[0] = 0.0
    E_sum = e.sum() + 1e-12

    p = e / E_sum
    S_spec = -np.sum(p * np.log(np.maximum(p, 1e-20)))
    F_local = E_sum - T_eff * S_spec

    # dominant bin → (A, ω, φ)
    k = int(np.argmax(e))
    A_dom = mags[k] * 2.0 / n
    omega_dom = 2*np.pi*freqs[k]
    phi_dom = np.angle(X[k])

    return dict(E_sum=E_sum, S_spec=S_spec, F_local=F_local,
                A_dom=A_dom, w_dom=omega_dom, phi_dom=phi_dom)ඉ


def encode_mass(F_before: float, F_after: float, xi_mass: float, c: float):
    dF = max(0.0, F_before - F_after)
    dm = xi_mass * dF / (c**2)
    Q = (1.0 - xi_mass) * dF
    return dF, dm, Q
    
def subtract_mode(signal: np.ndarray, fs: float, A: float, w: float, phi: float) -> np.ndarray:
    n = len(signal)
    t = np.arange(n) / fs
    xhat = A * np.cos(w*t + phi)
    return signal - xhat