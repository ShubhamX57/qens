<p align="center">
  <img src="https://raw.githubusercontent.com/ShubhamX57/qens/main/assets/qens_logo.svg" width="990" alt="QENS Logo"/>
</p>


# qens — Quasi-Elastic Neutron Scattering Analysis

A Python library for analysing QENS data, which handles everything from reading the raw binary files through to Bayesian
inference of diffusion parameters and publication-quality figures.

The physical model at the heart of this is the **Chudley-Elliott** jump-diffusion
model, which gives you the diffusion coefficient D and the mean jump length ℓ.
Fickian and Singwi-Sjölander models are also available for comparison.

---

## Installation

```bash
# from the repository root
pip install .

# with the emcee ensemble sampler (strongly recommended for MCMC)
pip install ".[mcmc]"

# development install
pip install -e ".[dev,mcmc]"
```

Python 3.10 or newer is required.

---

## Quick start

```python
from qens.config        import Config
from qens.io            import load_dataset
from qens.preprocessing import fit_elastic_peak, assign_resolution
from qens.fitting       import extract_hwhm, build_data_bins, find_map
from qens.sampling      import run_mcmc, summarise
import qens.plotting as qplt

# set up parameters
cfg = Config(q_min=0.3, q_max=2.5, n_walkers=32)

# load and preprocess
ds = load_dataset(["my_290_360_inc.nxspe", "my_260_360_inc.nxspe"],
                  data_dir="/data/run42")
for d in ds.values():
    fit_elastic_peak(d)
assign_resolution(ds)

# primary dataset
d_inc = ds["my_290_360_inc.nxspe"]

# extract HWHM by least squares
q, g, g_err, eisf = extract_hwhm(d_inc, cfg)

# Bayesian inference
bins = build_data_bins(d_inc, cfg)
d_map, l_map, tau = find_map(bins, d_inc["sigma_res"], cfg)
samples = run_mcmc(bins, d_inc["sigma_res"], d_map, l_map, cfg)

# results
d_med, d_lo, d_hi = summarise(samples[:, 0], "D (Å²/ps)")
l_med, l_lo, l_hi = summarise(samples[:, 1], "ℓ (Å)")
```

Or run the whole pipeline from the command line:

```bash
python run_analysis.py --data-dir /data/run42 --save-dir results/
```

---

## Library structure

```
qens/
├── qens/                     # core library
│   ├── __init__.py
│   ├── config.py             # Config dataclass
│   ├── constants.py          # Physical constants
│   ├── io.py                 # .nxspe HDF5 reader
│   ├── preprocessing.py      # Elastic peak & resolution
│   ├── models.py             # Diffusion models & basis
│   ├── fitting.py            # HWHM extraction & Bayesian posterior
│   ├── sampling.py           # MCMC & Gelman‑Rubin
│   └── plotting.py           # Publication figures
├── qens_terminal.py          # CLI pipeline (standalone)
├── QENS_Interactive_App.ipynb   # Jupyter interactive UI
└── README.md

```

### Module by module

**config.py** — everything you'd want to change between runs: Q range, energy
window, bin counts, MCMC settings. Serialises to/from JSON so results are
always traceable to their settings.

**io.py** — reads the Pelican `.nxspe` binary format. Validates byte offsets
before reading to catch corrupt files early. Handles batch loading with
graceful fallback for missing non-critical files.

**preprocessing.py** — two steps before fitting: align the energy axis to
zero by fitting the elastic peak, and assign the instrument resolution sigma.
Resolution priority: 260 K frozen sample > COH file > raw INC width.

**models.py** — the physics. All three diffusion models return HWHM in meV,
with D in Å²/ps. The `make_basis` function builds the spectral matrix used
in both least-squares and Bayesian fitting.

**fitting.py** — two approaches: `extract_hwhm` does a fast least-squares fit
per Q-bin; the Bayesian functions (`log_likelihood`, `find_map`) handle the
full posterior. Spectral amplitudes are analytically marginalised via NNLS so
MCMC only needs to explore D and ℓ.

**sampling.py** — MCMC wrapper that prefers emcee but falls back to a
hand-written Metropolis-Hastings if emcee isn't installed. Includes the
Gelman-Rubin R-hat convergence diagnostic.

**plotting.py** — five figures: dataset overview, single-spectrum fit,
S(Q,ω) intensity maps, Γ(Q) vs Q² with posterior fan, and posterior histograms.

---

## Diffusion models

| Model | Γ(Q) | Parameters |
|---|---|---|
| `ce(q, d, l)` | $\frac{\hbar}{\tau}\left(1 - \frac{\sin Q\ell}{Q\ell}\right)$ | D (Å²/ps), ℓ (Å) |
| `fickian(q, d)` | $\hbar D Q^2$ | D (Å²/ps) |
| `ss_model(q, d, tau_s)` | $\frac{\hbar D Q^2}{1 + DQ^2\tau_s}$ | D (Å²/ps), τ_s (ps) |

All use the same D convention: physical self-diffusion coefficient in Å²/ps,
with ħ = 0.6582 meV·ps as the bridge to the energy scale.

---

## Configuration

```python
from qens.config import Config

cfg = Config(q_min=0.4,
             q_max=2.2,
             n_walkers=64,
             n_warmup=1000,
             n_keep=4000,
             random_seed=0,)

# save for reproducibility
cfg.to_json("run_settings.json")

# reload later
cfg2 = Config.from_json("run_settings.json")
```

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥ 1.24 | array maths |
| scipy | ≥ 1.10 | curve_fit, nnls, fftconvolve |
| matplotlib | ≥ 3.7 | figures |
| emcee | ≥ 3.1 | ensemble MCMC (optional but recommended) |

---

## License

MIT
