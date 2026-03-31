"""
models.py

The actual physics: diffusion models and the spectral functions
we use to build model spectra.

All three diffusion models share the same convention for D:
it's the physical self-diffusion coefficient in Å²/ps.
The HWHM they return is in meV.

The key relationship is Γ = ħ·D·Q² in the Fickian (low-Q) limit,
where ħ = 0.6582 meV·ps. That ħ factor is what links the diffusion
coefficient (Å²/ps) to the energy scale (meV).

Quick overview of what's here
------------------------------
    ce(q, d, l)           — Chudley-Elliott jump diffusion
    fickian(q, d)         — simple continuous diffusion
    ss_model(q, d, tau_s) — Singwi-Sjolander (correlated jumps)
    lorentz(w, gamma)     — normalised Lorentzian lineshape
    gnorm(w, sigma)       — normalised Gaussian
    make_basis(...)       — builds the 3-column spectral basis matrix
"""

import numpy as np
from scipy.signal import fftconvolve

from .constants import hbar_mevps


# --- diffusion models --------------------------------------------------------

def ce(q, d, l):
    """
    Chudley-Elliott jump diffusion HWHM.

    At low Q (q·l << 1) this reduces to Fickian diffusion: Γ → ħ·D·Q².
    At high Q the HWHM saturates at ħ/τ, where τ = l²/(6D) is the residence
    time. The saturation is what tells you the jump length — without it
    you can't distinguish CE from Fickian.

    Parameters
    ----------
    q : array or float — momentum transfer in Å⁻¹
    d : float          — diffusion coefficient in Å²/ps
    l : float          — mean jump length in Å (sign doesn't matter, we use sinc)

    Returns
    -------
    HWHM Γ(Q) in meV

    Note on the sinc: numpy's sinc is sin(πx)/(πx), so we pass q*l/π
    to evaluate sin(q*l)/(q*l).
    """
    tau = l ** 2 / (6.0 * d)
    return (hbar_mevps / tau) * (1.0 - np.sinc(q * l / np.pi))


def fickian(q, d):
    """
    Continuous Brownian diffusion HWHM: Γ = ħ·D·Q²

    This is the low-Q limit of both CE and SS, so all three models
    agree at small momentum transfer. If your data looks linear in
    Γ vs Q² all the way to high Q, you're probably in the Fickian regime
    (or your Q range isn't large enough to see the saturation).

    Parameters
    ----------
    q : array or float — momentum transfer in Å⁻¹
    d : float          — diffusion coefficient in Å²/ps

    Returns
    -------
    HWHM Γ(Q) in meV
    """
    return hbar_mevps * d * np.asarray(q) ** 2


def ss_model(q, d, tau_s):
    """
    Singwi-Sjolander jump diffusion HWHM.

    Gives the same low-Q behaviour as CE (Γ → ħ·D·Q²) but saturates
    differently at high Q: Γ → ħ/τ_s. The physical picture is slightly
    different from CE — it models velocity correlations between jumps
    rather than a lattice geometry.

    Parameters
    ----------
    q     : array or float — momentum transfer in Å⁻¹
    d     : float          — diffusion coefficient in Å²/ps
    tau_s : float          — characteristic jump time in ps

    Returns
    -------
    HWHM Γ(Q) in meV
    """
    q = np.asarray(q)
    return hbar_mevps * d * q ** 2 / (1.0 + d * q ** 2 * tau_s)


# --- lineshape primitives ----------------------------------------------------

def lorentz(w, gamma):
    """
    Normalised Lorentzian, area = 1.

    This is the quasi-elastic lineshape for a diffusing particle
    (before convolution with the resolution). The HWHM is gamma.

    Parameters
    ----------
    w     : array — energy transfer grid in meV
    gamma : float — HWHM in meV

    Returns
    -------
    L(w, gamma) — evaluated on the grid
    """
    w = np.asarray(w)
    return (1.0 / np.pi) * gamma / (w ** 2 + gamma ** 2)


def gnorm(w, sigma):
    """
    Normalised Gaussian, area = 1.

    Used for the resolution function. The sigma here is the standard
    deviation in meV; FWHM = 2.355 * sigma.

    Parameters
    ----------
    w     : array — energy transfer grid in meV
    sigma : float — standard deviation in meV
    """
    w = np.asarray(w)
    return np.exp(-0.5 * (w / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# --- spectral basis ----------------------------------------------------------

def make_basis(e_grid, q_val, d, l, sigma_res):
    """
    Build the three-column basis matrix for a single Q value.

    The columns are:
        col 0 — elastic peak (Gaussian with the instrument resolution)
        col 1 — quasi-elastic component (Lorentzian convolved with resolution)
        col 2 — flat background (all ones)

    Both signal columns are peak-normalised to 1. That way the NNLS
    amplitudes you get back are directly interpretable as fractions of
    the peak height, and you can compute the EISF as amp[0]/(amp[0]+amp[1]).

    Parameters
    ----------
    e_grid    : 1D array — energy grid in meV
    q_val     : float    — Q value in Å⁻¹ (used to compute CE linewidth)
    d         : float    — diffusion coefficient in Å²/ps
    l         : float    — jump length in Å
    sigma_res : float    — instrument resolution sigma in meV

    Returns
    -------
    basis matrix, shape (len(e_grid), 3)
    """
    dt    = e_grid[1] - e_grid[0]
    gamma = ce(q_val, d, l)

    # elastic column — just the resolution function, normalised
    el    = gnorm(e_grid, sigma_res)
    el    = el / el.max()

    # quasi-elastic column — Lorentzian convolved with resolution
    ql    = fftconvolve(lorentz(e_grid, gamma), gnorm(e_grid, sigma_res), mode="same") * dt
    if ql.max() > 0:
        ql = ql / ql.max()

    return np.column_stack([el, ql, np.ones(len(e_grid))])
