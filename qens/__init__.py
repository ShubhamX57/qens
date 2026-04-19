"""
qens — Quasi-Elastic Neutron Scattering analysis
=================================================

A modular Python library for the complete QENS analysis pipeline:

    Load → Preprocess → Model → Fit → Sample → Visualise

Typical usage
-------------
>>> from qens import Config, load_dataset, fit_elastic_peak, assign_resolution
>>> from qens import build_data_bins, find_map, run_mcmc, summarise
>>> from qens import plotting

>>> cfg     = Config()
>>> dataset = load_dataset(cfg.files_to_fit, data_dir="data/")
>>> for d in dataset.values():
...     fit_elastic_peak(d)
>>> assign_resolution(dataset)

>>> d_inc    = dataset[cfg.primary_file]
>>> bins     = build_data_bins(d_inc, cfg)
>>> d_map, l_map, tau_map = find_map(bins, d_inc["sigma_res"], cfg)
>>> samples  = run_mcmc(bins, d_inc["sigma_res"], d_map, l_map, cfg)
>>> summarise(samples[:, 0], "D (Å²/ps)")

Module overview
---------------
config       : Config dataclass — all analysis parameters in one place.
io           : Read ISIS Mantid .nxspe (NeXus HDF5) files.
preprocessing: Elastic peak alignment and instrument resolution assignment.
models       : Diffusion model functions (CE, Fickian, SS) and spectral basis.
fitting      : HWHM extraction, Bayesian log-posterior, MAP optimisation.
sampling     : MCMC via emcee (or Metropolis-Hastings fallback).
plotting     : All diagnostic and publication figures.
constants    : Physical constants in SI and meV/Å/ps unit systems.
"""

# ── Public API ────────────────────────────────────────────────────────────────
# Each import below pulls the most commonly used symbol from its module so that
# users can write `from qens import Config` instead of
# `from qens.config import Config`.

from .config        import Config

# Data loading
from .io            import read_nxspe, read_nxspe_hdf5, load_dataset

# Preprocessing
from .preprocessing import fit_elastic_peak, assign_resolution

# Physical models and spectral basis
from .models        import ce, fickian, ss_model, lorentz, gnorm, make_basis

# Fitting — HWHM extraction and Bayesian posterior
from .fitting       import (extract_hwhm, save_hwhm_csv, build_data_bins,
                            log_likelihood, log_prior, log_posterior, find_map)

# MCMC sampling and convergence diagnostics
from .sampling      import run_mcmc, summarise, gelman_rubin

# Plotting (imported as a namespace so callers do `qens.plotting.plot_hwhm(...)`)
from .              import plotting

# ── Package metadata ──────────────────────────────────────────────────────────
__version__ = "0.2.0"
__author__  = "QENS Analysis Contributors"

# Explicit public API — controls `from qens import *`
__all__ = [
    # Configuration
    "Config",

    # I/O
    "read_nxspe", "read_nxspe_hdf5", "load_dataset",

    # Preprocessing
    "fit_elastic_peak", "assign_resolution",

    # Models
    "ce", "fickian", "ss_model", "lorentz", "gnorm", "make_basis",

    # Fitting
    "extract_hwhm", "save_hwhm_csv", "build_data_bins",
    "log_likelihood", "log_prior", "log_posterior", "find_map",

    # Sampling
    "run_mcmc", "summarise", "gelman_rubin",

    # Plotting namespace
    "plotting",
]
