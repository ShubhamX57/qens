"""
Diffusion models and spectral basis functions.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from .constants import hbar_mevps

_GAMMA_FLOOR = 1e-5

def ce(q, d, l):
    """
    Chudley-Elliott jump diffusion HWHM in meV.
    """
    if d <= 0:
        raise ValueError(f"d must be > 0, got {d}")
    l = abs(l)
    tau = l**2 / (6 * d)
    return (hbar_mevps / tau) * (1 - np.sinc(np.asarray(q) * l / np.pi))



def fickian(q, d):
    """
    Fickian diffusion HWHM: Γ = ħ D Q².
    """
    return hbar_mevps * d * np.asarray(q)**2



def ss_model(q, d, tau_s):
    """
    Singwi-Sjolander jump diffusion HWHM.
    """
    q = np.asarray(q)
    return hbar_mevps * d * q**2 / (1 + d * q**2 * tau_s)



def lorentz(w, gamma):
    """
    Normalised Lorentzian, area=1.
    """
    gamma = max(float(gamma), _GAMMA_FLOOR)
    w = np.asarray(w, dtype=float)
    return (1 / np.pi) * gamma / (w**2 + gamma**2)



def gnorm(w, sigma):
    """
    Normalised Gaussian, area=1.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    w = np.asarray(w, dtype=float)
    return np.exp(-0.5 * (w / sigma)**2) / (sigma * np.sqrt(2 * np.pi))



def make_basis(e_grid, q_val, d, l, sigma_res):
    """
    3-column basis for NNLS: [elastic, QENS, background].
    """
    dt = e_grid[1] - e_grid[0]
    gamma = max(float(ce(q_val, d, l)), _GAMMA_FLOOR)

    el = gnorm(e_grid, sigma_res)
    el /= el.max() if el.max() > 0 else 1.0

    ql = fftconvolve(lorentz(e_grid, gamma), gnorm(e_grid, sigma_res), mode="same") * dt
    ql /= ql.max() if ql.max() > 0 else 1.0

    return np.column_stack([el, ql, np.ones(len(e_grid))])
