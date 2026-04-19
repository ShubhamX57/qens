"""
constants.py — Physical constants for QENS analysis
=====================================================

All constants are in SI units unless the name explicitly states otherwise.
The module is intentionally small — import only what you need.

Unit conventions used across the library
-----------------------------------------
Energy      : meV (milli-electron-volts)
Momentum    : Å⁻¹ (inverse ångströms)
Length      : Å (ångströms)
Time        : ps (picoseconds)
Temperature : K (kelvin)

Key derived relationship
------------------------
hbar_mevps converts a timescale τ (ps) into an energy width Γ (meV):

    Γ (meV) = hbar_mevps / τ (ps)

This is used in every diffusion model in models.py.
"""

import numpy as np

# ── Fundamental constants (SI) ────────────────────────────────────────────────

mn = 1.67493e-27
"""float : Neutron rest mass (kg).
Used to compute the incident wavevector ki from the incident energy Ei:
    ki (Å⁻¹) = sqrt(2 * mn * Ei_J) / hbar * 1e-10
"""

hbar = 1.05457e-34
"""float : Reduced Planck constant (J·s).
Used alongside mn and mev_j to convert Ei (meV) to ki (Å⁻¹).
"""

mev_j = 1.60218e-22
"""float : Conversion factor from meV to Joules (J / meV).
1 meV = 1.60218e-22 J.
Used to convert Ei from the meV values stored in .nxspe files to SI for the
ki calculation:
    Ei_J = Ei_meV * mev_j
"""

# ── Derived constant in working units ─────────────────────────────────────────

hbar_mevps = 0.65821
"""float : Reduced Planck constant in meV·ps.
This is the most frequently used constant in the model functions.

Derivation
----------
    hbar = 1.05457e-34 J·s
         = 1.05457e-34 / 1.60218e-22 meV·s     (divide by mev_j)
         = 6.5821e-13 meV·s
         = 0.65821 meV·ps                        (1 s = 1e12 ps)

Physical meaning
----------------
Every diffusion HWHM formula has the form  Γ = hbar / τ  (in some variant).
Using hbar_mevps means τ can be given in ps and Γ comes out directly in meV —
the native units of the spectrometer.

For example, the Chudley-Elliott model:
    Γ(Q) = (hbar_mevps / τ) * [1 − sinc(Ql/π)]   (meV)
where τ = l²/(6D) is in ps when D is in Å²/ps and l is in Å.
"""
