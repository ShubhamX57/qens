"""
constants.py

All the physical constants and Pelican-specific numbers live here.
Nothing fancy — just a single place to look when you need to know
what value we're using for, say, hbar or the neutron mass.

If you're adding support for a different instrument you'd probably
want to subclass Config and override the detector geometry stuff,
but these physical constants obviously stay the same.
"""

import numpy as np


# --- physical constants ---------------------------------------------------
# all SI unless the name says otherwise

mn         = 1.6749e-27    # neutron rest mass, kg
hbar       = 1.0546e-34    # reduced planck constant, J·s
mev_j      = 1.6022e-22    # 1 meV expressed in joules
hbar_mevps = 0.6582        # hbar in meV·ps — crops up constantly in linewidth formulas


# --- Pelican detector geometry -------------------------------------------
# 249 detector groups spanning 3.5 to 140 degrees two-theta

n_det     = 249
n_bin     = 320   # energy bins per detector

# the actual two-theta angles for each detector group
two_theta = np.linspace(3.5, 140.0, n_det)   # degrees


# --- Pelican .nxspe binary file layout -----------------------------------
# these are byte offsets where the data arrays live inside the binary file.
# they were determined empirically — default values work for standard Pelican
# output. if you ever get a "cannot locate offsets" error on a weird file,
# this is where to start looking.

off_e_default   = 0x30C0    # energy bin edges
off_d_default   = 0x41B8    # S(Q,w) intensity array
off_err_default = 0x9FFB8   # uncertainties
