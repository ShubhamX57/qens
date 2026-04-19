"""
config.py — Analysis configuration dataclass
=============================================

All tuneable parameters live in a single ``Config`` object.  Every downstream
function accepts a ``cfg`` argument so that nothing is hardcoded in the
analysis modules.  Saving a Config to JSON alongside results makes an
experiment fully reproducible.

Example
-------
>>> cfg = Config()                      # default values
>>> cfg.to_json("results/config.json") # save for reproducibility
>>> cfg2 = Config.from_json("results/config.json")  # reload later
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Config:
    """
    All analysis parameters in one place.

    Q-range and energy windows
    --------------------------
    q_min / q_max : float
        Momentum transfer range (Å⁻¹) used in all fitting steps.
        Below q_min the quasi-elastic broadening Γ(Q) is below the instrument
        resolution and cannot be resolved.  Above q_max the signal is weak
        and multiple-scattering contamination grows.  Default: [0.30, 2.50].

    ewin_hwhm : float
        Half-width of the energy window (meV) used by ``extract_hwhm``.
        Data outside ±ewin_hwhm is ignored during the per-bin Lorentzian fit.

    ewin_mcmc : float
        Half-width of the energy window (meV) used by ``build_data_bins`` and
        the Bayesian likelihood.  Set equal to ewin_hwhm unless you have a
        specific reason to differ.

    Q-binning
    ---------
    n_bins : int
        Number of Q-bins for HWHM extraction (``extract_hwhm``).  More bins
        give finer Q-resolution on Γ(Q) but fewer counts per bin.

    n_bins_mc : int
        Number of Q-bins for the Bayesian fit (``build_data_bins``).  Kept
        smaller than n_bins because each bin requires a full NNLS solve at
        every MCMC step.

    MCMC hyperparameters
    --------------------
    n_walkers : int
        Number of emcee ensemble walkers.  Must be even and ≥ 4.  Rule of
        thumb: at least 2 × ndim + 2.  Default: 32 (for ndim=2).

    n_warmup : int
        Steps discarded as burn-in before collecting samples.  The chain has
        not converged during warmup.  Default: 500.

    n_keep : int
        Steps retained per walker after burn-in.  Total posterior samples
        = n_walkers × n_keep / thin.  Default: 2000.

    thin : int
        Keep every thin-th step to reduce autocorrelation in the chain.
        Should ideally be ≥ half the autocorrelation time.  Default: 5.

    Reproducibility
    ---------------
    random_seed : int
        Seeds numpy's default_rng in MAP optimisation and MCMC initialisation.
        Fix this to get identical results across runs.

    Output
    ------
    save_dir : str
        Directory where CSV tables and figures are written.
    """

    # ── File selection ────────────────────────────────────────────────────────
    files_to_fit: List[str] = field(
        default_factory=lambda: [
            "benzene_290_197_inc.nxspe",   # Ei = 1.97 meV — lower resolution
            "benzene_290_360_inc.nxspe",   # Ei = 3.60 meV — primary file
        ]
    )

    # The .nxspe file used for the main Bayesian fit and all detailed plots.
    primary_file: str = "benzene_290_360_inc.nxspe"

    # ── Q-range ───────────────────────────────────────────────────────────────
    q_min: float = 0.30   # Å⁻¹  — lower Q cut-off
    q_max: float = 2.50   # Å⁻¹  — upper Q cut-off

    # ── Energy windows ────────────────────────────────────────────────────────
    ewin_hwhm: float = 0.80   # meV — window for HWHM extraction
    ewin_mcmc: float = 0.80   # meV — window for Bayesian likelihood

    # ── Q-binning ─────────────────────────────────────────────────────────────
    n_bins:    int = 13   # bins for extract_hwhm  (finer Q grid)
    n_bins_mc: int = 10   # bins for build_data_bins / MCMC (cheaper per bin)

    # ── MCMC ─────────────────────────────────────────────────────────────────
    n_walkers: int = 32    # emcee ensemble size; must be even and ≥ 4
    n_warmup:  int = 500   # burn-in steps (discarded)
    n_keep:    int = 2000  # steps retained per walker
    thin:      int = 5     # thinning factor to reduce chain autocorrelation

    # ── Misc ──────────────────────────────────────────────────────────────────
    random_seed: int = 42        # NumPy RNG seed for reproducibility
    save_dir:    str = "results" # output directory for CSV and figures

    # ── Validation ───────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """
        Validate parameter combinations immediately after construction.

        Raises
        ------
        ValueError
            If any parameter is outside its physically or numerically valid
            range.
        """
        if self.q_min >= self.q_max:
            raise ValueError(
                f"q_min ({self.q_min}) must be < q_max ({self.q_max})"
            )
        if self.ewin_hwhm <= 0 or self.ewin_mcmc <= 0:
            raise ValueError("Energy windows must be > 0")
        if self.n_walkers < 4:
            raise ValueError("n_walkers must be ≥ 4")
        if self.n_walkers % 2 != 0:
            # emcee requires an even number of walkers for the stretch move
            raise ValueError("n_walkers must be even")
        if self.n_bins < 2 or self.n_bins_mc < 2:
            raise ValueError("n_bins and n_bins_mc must be ≥ 2")
        if self.thin < 1:
            raise ValueError("thin must be ≥ 1")

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Return all fields as a plain Python dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """
        Write the configuration to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. "results/config.json").
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  config saved → {path}")

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """
        Load a Config from a previously saved JSON file.

        Parameters
        ----------
        path : str
            Path to a JSON file written by ``to_json``.

        Returns
        -------
        Config
            A new Config instance populated from the file.
        """
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self) -> str:
        """Human-readable representation listing every parameter."""
        lines = ["Config("]
        for k, v in self.to_dict().items():
            lines.append(f"    {k} = {v!r},")
        lines.append(")")
        return "\n".join(lines)
