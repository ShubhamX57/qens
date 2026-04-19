"""
models.py — Diffusion models and spectral basis construction
=============================================================

This module contains:

1. Physical model functions that predict the quasi-elastic linewidth Γ(Q):
   - ``ce``       : Chudley-Elliott jump diffusion
   - ``fickian``  : Simple Fickian (continuous) diffusion
   - ``ss_model`` : Singwi-Sjölander jump diffusion

2. Spectral line-shape functions:
   - ``lorentz``  : Normalised Lorentzian  (quasi-elastic component)
   - ``gnorm``    : Normalised Gaussian    (resolution / elastic component)

3. The NNLS basis matrix builder:
   - ``make_basis``: Constructs the [elastic | quasi-elastic | background]
                     matrix used in the Bayesian likelihood.

Physical context
----------------
The measured spectrum at each Q is modelled as:

    S_obs(Q, ω) = a_el · R(ω) + a_ql · [L(ω, Γ) ⊗ R(ω)] + bg

where
    R(ω)  = Gaussian resolution function with width sigma_res
    L(ω)  = Lorentzian with half-width Γ = ce(Q, D, l)
    ⊗      = convolution in energy
    a_el, a_ql, bg = non-negative amplitudes solved by NNLS

The convolution of L with R is essential — the instrument always smears the
true quasi-elastic signal.  This follows directly from the Van Hove formalism:
    S_obs = S_true ⊗ R(ω).

Model selection guide
---------------------
Use Fickian when Γ(Q) is linear in Q² across your entire Q-range.
Use CE (Chudley-Elliott) when you see saturation of Γ at high Q — this is
the fingerprint of jump diffusion.
Use SS (Singwi-Sjölander) as an alternative jump model; it has a different
crossover shape but the same low- and high-Q limits as CE.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .constants import hbar_mevps

# Minimum allowed Lorentzian HWHM (meV).  Used as a floor to prevent
# numerical issues when D or l approaches zero.
_GAMMA_FLOOR = 1e-5


# ── Diffusion model functions ─────────────────────────────────────────────────

def ce(q, d, l):
    """
    Chudley-Elliott jump diffusion HWHM.

    Predicts the half-width at half-maximum (HWHM) of the quasi-elastic
    Lorentzian for a molecule undergoing discrete jump diffusion: the molecule
    sits at a lattice site for mean time τ = l²/(6D) then jumps to a
    neighbouring site a distance l away.

    The formula is derived by Fourier-transforming the jump probability over
    an isotropic (spherical) distribution of jump directions:

        Γ(Q) = (ħ / τ) · [1 − sinc(Ql / π)]

    where sinc(x) = sin(πx) / (πx) (NumPy normalised convention).

    Limiting behaviour
    ------------------
    Low Q  (Ql → 0) : sinc → 1 − (Ql)²/6, so Γ(Q) ≈ ħ·D·Q²  (Fickian limit)
    High Q (Ql → ∞): sinc → 0,             so Γ(Q) → ħ/τ = constant

    The saturation of Γ at high Q is the experimental fingerprint of jump
    diffusion.  For benzene at 290 K, saturation starts around Q* ≈ π/l ≈ 1.5
    Å⁻¹, which falls within the MARI measurement window.

    Parameters
    ----------
    q : array-like
        Momentum transfer values (Å⁻¹).
    d : float
        Self-diffusion coefficient (Å²/ps).  Must be > 0.
    l : float
        Mean jump length (Å).  Absolute value is taken internally.

    Returns
    -------
    numpy.ndarray
        HWHM Γ(Q) in meV, same shape as q.

    Raises
    ------
    ValueError
        If d ≤ 0.

    Examples
    --------
    >>> import numpy as np
    >>> from qens.models import ce
    >>> q = np.linspace(0.3, 2.5, 10)
    >>> gamma = ce(q, D=0.32, l=2.1)  # meV
    """
    if d <= 0:
        raise ValueError(f"d must be > 0, got {d}")

    l = abs(l)  # l is always positive — take abs to guard against sampler sign flips

    # Residence time between jumps (ps)
    tau = l**2 / (6 * d)

    # NumPy sinc is the *normalised* sinc: sinc(x) = sin(πx)/(πx).
    # So sinc(q*l/π) = sin(q*l) / (q*l), which is the spherical average
    # of exp(iQ·r) over jump vectors of length l.
    return (hbar_mevps / tau) * (1 - np.sinc(np.asarray(q) * l / np.pi))


def fickian(q, d):
    """
    Fickian (continuous) diffusion HWHM.

    In the limit of infinitesimally small, continuous jumps the Chudley-Elliott
    model reduces to Fick's law:

        Γ(Q) = ħ · D · Q²

    This is a straight line through the origin when plotted against Q².  Any
    measured data that curves or saturates at high Q is evidence of jump
    diffusion (use CE or SS instead).

    Parameters
    ----------
    q : array-like
        Momentum transfer values (Å⁻¹).
    d : float
        Self-diffusion coefficient (Å²/ps).

    Returns
    -------
    numpy.ndarray
        HWHM Γ(Q) in meV, same shape as q.
    """
    return hbar_mevps * d * np.asarray(q)**2


def ss_model(q, d, tau_s):
    """
    Singwi-Sjölander jump diffusion HWHM.

    An alternative jump-diffusion model that assumes exponentially distributed
    waiting times between jumps (a continuous-time random walk), rather than
    a fixed lattice geometry as in CE:

        Γ(Q) = ħ · D · Q² / (1 + D · Q² · τ_s)

    Limiting behaviour
    ------------------
    Low Q  : Γ ≈ ħ·D·Q²          (same Fickian limit as CE)
    High Q : Γ → ħ / τ_s          (same saturation as CE, different crossover)

    For benzene at the Q-range accessible on MARI, CE and SS give
    statistically indistinguishable fits.  Prefer SS on physical grounds for
    liquids with no long-range lattice order.

    Parameters
    ----------
    q : array-like
        Momentum transfer values (Å⁻¹).
    d : float
        Self-diffusion coefficient (Å²/ps).
    tau_s : float
        Residence time between jumps (ps).

    Returns
    -------
    numpy.ndarray
        HWHM Γ(Q) in meV, same shape as q.
    """
    q = np.asarray(q)
    return hbar_mevps * d * q**2 / (1 + d * q**2 * tau_s)


# ── Spectral line-shape functions ─────────────────────────────────────────────

def lorentz(w, gamma):
    """
    Normalised Lorentzian line shape (area = 1).

    The quasi-elastic component of a QENS spectrum is a Lorentzian in energy
    transfer ω whose half-width Γ encodes the timescale of atomic motion.
    Before being added to the model, it is convolved with the resolution
    Gaussian in ``make_basis``.

    Formula:
        L(ω, Γ) = (1/π) · Γ / (ω² + Γ²)

    Parameters
    ----------
    w : array-like
        Energy transfer values (meV).
    gamma : float
        Half-width at half-maximum (meV).  Clamped to _GAMMA_FLOOR if below it.

    Returns
    -------
    numpy.ndarray
        Lorentzian values in meV⁻¹, same shape as w.  Integrates to 1 over ω.
    """
    # Guard against gamma = 0 (would cause division by zero and a delta function)
    gamma = max(float(gamma), _GAMMA_FLOOR)
    w = np.asarray(w, dtype=float)
    return (1 / np.pi) * gamma / (w**2 + gamma**2)


def gnorm(w, sigma):
    """
    Normalised Gaussian line shape (area = 1).

    Used as the instrument resolution function R(ω).  The Gaussian
    approximation is valid for most time-of-flight spectrometers near the
    elastic line.

    Formula:
        G(ω, σ) = exp(−ω²/(2σ²)) / (σ√(2π))

    Parameters
    ----------
    w : array-like
        Energy transfer values (meV).
    sigma : float
        Standard deviation of the Gaussian (meV).
        Relates to FWHM by: FWHM = 2.355 · sigma.

    Returns
    -------
    numpy.ndarray
        Gaussian values in meV⁻¹, same shape as w.  Integrates to 1 over ω.

    Raises
    ------
    ValueError
        If sigma ≤ 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    w = np.asarray(w, dtype=float)
    return np.exp(-0.5 * (w / sigma)**2) / (sigma * np.sqrt(2 * np.pi))


# ── NNLS basis matrix ─────────────────────────────────────────────────────────

def make_basis(e_grid, q_val, d, l, sigma_res):
    """
    Build the three-column basis matrix for NNLS spectral decomposition.

    At a given (D, l), the observed spectrum at momentum transfer q_val is
    modelled as a non-negative combination of three components:

        S_obs ≈ a_el · col0  +  a_ql · col1  +  bg · col2

    Column layout
    -------------
    col0 : Elastic component
        Gaussian at the resolution width sigma_res, normalised to peak = 1.
        Represents scattering that does not involve energy transfer — atoms
        that are stationary on the timescale of the measurement.

    col1 : Quasi-elastic component
        Lorentzian with HWHM = ce(q_val, d, l), convolved with the resolution
        Gaussian, normalised to peak = 1.
        The convolution is physically required: the instrument always smears
        the true quasi-elastic signal.

    col2 : Flat background
        A vector of ones, capturing any incoherent or multiple-scattering
        background that is flat in energy over the fitting window.

    Normalisation
    -------------
    Both col0 and col1 are divided by their peak value.  Without this, a very
    broad (flat) Lorentzian would have column values orders of magnitude
    smaller than the elastic column, and NNLS would effectively ignore it.
    Normalising to unity puts both components on the same scale so that the
    amplitudes (a_el, a_ql) directly represent physical fractions.

    Parameters
    ----------
    e_grid : numpy.ndarray
        Energy transfer axis of the data (meV), uniformly spaced.
    q_val : float
        Momentum transfer for this Q-bin (Å⁻¹).
    d : float
        Diffusion coefficient trial value (Å²/ps).
    l : float
        Jump length trial value (Å).
    sigma_res : float
        Instrument resolution sigma (meV).

    Returns
    -------
    numpy.ndarray, shape (len(e_grid), 3)
        Basis matrix [elastic | quasi-elastic | background].
        Pass directly to ``scipy.optimize.nnls``.

    Notes
    -----
    dt = e_grid[1] − e_grid[0] is applied after fftconvolve to preserve the
    physical normalisation of the convolution (converts the discrete sum to an
    integral approximation).
    """
    # Energy step — needed to normalise the convolution to unit integral
    dt = e_grid[1] - e_grid[0]

    # Quasi-elastic HWHM from the CE model at this Q value.
    # Clamped to _GAMMA_FLOOR to avoid numerical issues at very small D or l.
    gamma = max(float(ce(q_val, d, l)), _GAMMA_FLOOR)

    # ── Elastic column (col0) ─────────────────────────────────────────────────
    # Resolution Gaussian centred at ω = 0.  Normalised to peak = 1 so that
    # the amplitude a_el directly represents the elastic fraction.
    el = gnorm(e_grid, sigma_res)
    el /= el.max() if el.max() > 0 else 1.0

    # ── Quasi-elastic column (col1) ───────────────────────────────────────────
    # Lorentzian of width gamma, convolved with the resolution Gaussian.
    # fftconvolve returns a discrete convolution; multiplying by dt converts
    # it to a proper integral approximation.
    ql = fftconvolve(lorentz(e_grid, gamma), gnorm(e_grid, sigma_res), mode="same") * dt
    ql /= ql.max() if ql.max() > 0 else 1.0

    # ── Assemble and return ───────────────────────────────────────────────────
    # column_stack produces shape (N_energy, 3)
    return np.column_stack([el, ql, np.ones(len(e_grid))])
