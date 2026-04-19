"""
io.py — Read ISIS Mantid .nxspe (NeXus HDF5) files
====================================================

.nxspe files are NeXus HDF5 archives produced by the ISIS Mantid data
reduction pipeline.  They contain the reduced S(Q, ω) data — counts, errors,
and detector geometry — for a single spectrometer run.

File naming convention (ISIS)
------------------------------
    <sample>_<temp>_<Ei×100>_<kind>.nxspe

Example: ``benzene_290_360_inc.nxspe``
    sample = benzene
    temp   = 290 K
    Ei     = 360 / 100 = 3.60 meV
    kind   = inc (incoherent scattering)

HDF5 tree (typical structure)
------------------------------
The structure varies between ISIS instruments, so ``_read_polar`` tries
several candidate paths for the detector angles.  Typical layout:

    entry1/
        NXSPE_info/
            efixed          (float scalar — incident energy Ei in meV)
        data/
            energy          (array, N+1 energy bin edges in meV)
            data            (2D array, shape [n_det, N_energy])
            error           (2D array, shape [n_det, N_energy])
            polar           (1D array, shape [n_det], scattering angles in deg)
"""

from __future__ import annotations

import os
import numpy as np

from .constants import mn, hbar, mev_j


# ── Diagnostic helper ─────────────────────────────────────────────────────────

def inspect_nxspe(path: str) -> None:
    """
    Print the full HDF5 tree of a .nxspe file to stdout.

    Useful for diagnosing files from instruments whose internal structure
    differs from the expected layout.

    Parameters
    ----------
    path : str
        Path to the .nxspe file.

    Raises
    ------
    ImportError
        If h5py is not installed.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py")

    print(f"HDF5 tree: {path}")
    print("-" * 60)

    def _visitor(name, obj):
        """h5py visitor callback — prints name, shape and dtype of each node."""
        indent = "  " * name.count("/")
        leaf   = name.split("/")[-1]
        if hasattr(obj, "shape"):
            # Dataset node — show shape and dtype
            print(f"{indent}{leaf}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            # Group node — just print the name
            print(f"{indent}{leaf}/")

    with h5py.File(path, "r") as hf:
        hf.visititems(_visitor)

    print("-" * 60)


# ── Single-file reader ────────────────────────────────────────────────────────

def read_nxspe(path: str) -> dict:
    """
    Load a single .nxspe file into a dataset dictionary.

    Performs the following steps in order:
    1. Parse metadata (temp, Ei, kind) from the filename.
    2. Open the HDF5 file and read energy edges, counts, errors, angles.
    3. Compute energy bin centres from edges.
    4. Compute Q per detector from Ei and scattering angle (elastic approx.).
    5. Identify good detectors (> 50% of channels are positive and finite).
    6. Return a dict containing all arrays needed downstream.

    Elastic approximation for Q
    ----------------------------
    Q = 2 · ki · sin(θ), where ki = √(2·mn·Ei) / ħ and θ = two_theta / 2.
    This assumes the neutron wavevector does not change on scattering —
    valid for QENS where |ω| ≪ Ei.

    Good detector selection
    -----------------------
    A detector is marked good if more than half its energy channels contain
    positive, finite counts.  Dead, masked, or mostly-zero detectors are
    excluded.  All downstream functions operate on data[good] rather than
    data, so dead detectors never contaminate averaged spectra.

    Parameters
    ----------
    path : str
        Absolute or relative path to the .nxspe file.

    Returns
    -------
    dict with keys:
        name     : str     — basename of the file
        temp     : int     — sample temperature (K)
        ei       : float   — incident energy (meV)
        kind     : str     — "inc" or "coh"
        e_raw    : ndarray — energy bin centres before elastic peak correction
        e        : ndarray — copy of e_raw (will be shifted by fit_elastic_peak)
        data     : ndarray — S(Q, ω), shape (n_det, N_energy)
        errs     : ndarray — statistical errors, shape (n_det, N_energy)
        good     : ndarray — 1D int array of usable detector indices
        q        : ndarray — momentum transfer per detector (Å⁻¹)
        format   : str     — "hdf5"

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ImportError
        If h5py is not installed.
    ValueError
        If the filename does not follow the ISIS naming convention, or if no
        usable detectors are found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py")

    # ── Parse filename metadata ───────────────────────────────────────────────
    name  = os.path.basename(path)
    parts = name.replace(".nxspe", "").split("_")

    if len(parts) < 4:
        raise ValueError(
            f"Filename '{name}' does not match expected pattern "
            f"<sample>_<temp>_<Ei×100>_<kind>"
        )

    try:
        temp = int(parts[1])   # temperature in Kelvin
    except ValueError:
        raise ValueError(f"Cannot parse temperature from '{name}'")

    kind = parts[3]  # "inc" (incoherent) or "coh" (coherent)

    # ── Read HDF5 content ─────────────────────────────────────────────────────
    with h5py.File(path, "r") as hf:
        # The top-level entry key varies by instrument (e.g. "entry1", "mantid_workspace_1")
        entry_key = next(iter(hf.keys()))
        entry     = hf[entry_key]

        # Incident energy — try HDF5 metadata first, fall back to filename
        ei = _read_ei(entry, parts)

        # Energy axis: file stores bin *edges*, not centres
        energy_edges = np.asarray(entry["data"]["energy"], dtype=float)
        # Midpoints give N-1 energy points from N edges
        e_raw = 0.5 * (energy_edges[:-1] + energy_edges[1:])

        # Spectral data and errors, shape (n_det, N_energy)
        data = np.asarray(entry["data"]["data"],  dtype=float)
        errs = np.asarray(entry["data"]["error"], dtype=float)

        # Scattering angles per detector (degrees)
        two_theta_det = _read_polar(entry, path)

    # ── Clean non-finite values ───────────────────────────────────────────────
    # Replace NaN/Inf in data with 0 (they appear in masked detectors)
    data = np.where(np.isfinite(data),              data, 0.0)
    # Replace non-positive or non-finite errors with 0 (handled downstream)
    errs = np.where(np.isfinite(errs) & (errs > 0), errs, 0.0)

    # ── Good detector mask ────────────────────────────────────────────────────
    # A detector is usable if more than half its energy channels have positive,
    # finite counts.  This excludes dead detectors and masked regions.
    good_mask = (
        np.sum((data > 0) & np.isfinite(data), axis=1) > data.shape[1] // 2
    )
    good = np.where(good_mask)[0]  # integer index array

    if good.size == 0:
        raise ValueError(f"No usable detectors found in '{name}'")

    # ── Q per detector (elastic approximation) ────────────────────────────────
    # ki in Å⁻¹: ki = sqrt(2 * mn * Ei_J) / hbar * 1e-10
    ki = np.sqrt(2 * mn * ei * mev_j) / hbar * 1e-10
    # Q = 2 ki sin(theta),  theta = two_theta / 2
    q  = 2 * ki * np.sin(np.radians(two_theta_det / 2))

    return dict(
        name   = name,
        temp   = temp,
        ei     = ei,
        kind   = kind,
        e_raw  = e_raw,
        e      = e_raw.copy(),  # will be shifted in fit_elastic_peak
        data   = data,
        errs   = errs,
        good   = good,
        q      = q,
        format = "hdf5",
    )


# Alias for backwards compatibility
read_nxspe_hdf5 = read_nxspe


# ── Private helpers ───────────────────────────────────────────────────────────

def _read_ei(entry, filename_parts: list[str]) -> float:
    """
    Extract the incident energy Ei (meV) from an open HDF5 entry.

    Tries two common key names in NXSPE_info first, then falls back to
    parsing Ei from the filename (third underscore-separated token × 100).

    Parameters
    ----------
    entry : h5py.Group
        The top-level entry group of the .nxspe file.
    filename_parts : list of str
        Filename split on underscores (without extension).

    Returns
    -------
    float
        Incident energy in meV.

    Raises
    ------
    ValueError
        If Ei cannot be determined from either the HDF5 metadata or filename.
    """
    # Try standard NeXus NXSPE_info keys
    for key in ("efixed", "fixed_energy"):
        try:
            return float(entry["NXSPE_info"][key][()])
        except (KeyError, TypeError):
            pass

    # Fall back to filename convention: <sample>_<temp>_<Ei×100>_<kind>
    try:
        return int(filename_parts[2]) / 100.0
    except (ValueError, IndexError):
        raise ValueError(
            "Cannot determine Ei from HDF5 metadata or filename.  "
            "Check NXSPE_info/efixed or the filename format."
        )


def _read_polar(entry, path: str) -> np.ndarray:
    """
    Extract the polar (scattering) angles (degrees) for each detector.

    Different ISIS instruments write this array to different HDF5 paths.
    The function tries a prioritised list of candidate locations and falls
    back to azimuthal angles as a last resort (with a warning).

    Parameters
    ----------
    entry : h5py.Group
        The top-level entry group of the .nxspe file.
    path : str
        Full file path — used only in the error message.

    Returns
    -------
    numpy.ndarray, shape (n_det,)
        Polar (two-theta) angles in degrees for each detector.

    Raises
    ------
    ValueError
        If no suitable angle array is found anywhere in the file.
    """
    # Ordered list of (group_path, dataset_name) pairs to try
    candidates = [
        ("data",                  "polar"),         # most common
        ("instrument/detector",   "polar"),          # some instruments
        ("instrument/detector_1", "polar_angle"),    # MAPS-style
    ]

    for grp_path, ds_name in candidates:
        try:
            grp = entry
            # Navigate the group path step by step
            for part in grp_path.split("/"):
                grp = grp[part]
            arr = np.asarray(grp[ds_name], dtype=float)
            # Sanity check: must be 1D and non-empty
            if arr.ndim == 1 and arr.size > 0:
                return arr
        except (KeyError, TypeError):
            continue  # path doesn't exist in this file — try the next one

    # Last resort: azimuthal angles are sometimes a proxy for polar on
    # instruments where the two-theta geometry is arranged azimuthally
    try:
        arr = np.asarray(entry["data"]["azimuthal"], dtype=float)
        if arr.ndim == 1 and arr.size > 0:
            import warnings
            warnings.warn(
                "Using azimuthal angles as a proxy for polar angles.  "
                "Q values may be approximate.",
                UserWarning,
                stacklevel=2,
            )
            return arr
    except (KeyError, TypeError):
        pass

    raise ValueError(
        f"Cannot locate polar detector angles in '{os.path.basename(path)}'.  "
        f"Tried: {[c[1] for c in candidates]} and data/azimuthal.  "
        f"Run inspect_nxspe() to see the full HDF5 tree."
    )


# ── Multi-file loader ─────────────────────────────────────────────────────────

def load_dataset(
    file_list: list[str],
    data_dir: str = ".",
    critical_files: list[str] | None = None,
) -> dict:
    """
    Load multiple .nxspe files into a single dataset dictionary.

    Iterates over file_list, calling ``read_nxspe`` on each file found in
    data_dir.  Files that are missing or fail to load are skipped unless they
    appear in critical_files, in which case an exception is raised.

    Parameters
    ----------
    file_list : list of str
        Filenames to load (basenames only; data_dir is prepended).
    data_dir : str
        Directory containing the .nxspe files.  Default: current directory.
    critical_files : list of str or None
        Subset of file_list that must load successfully.  A missing or
        unreadable critical file raises an exception rather than being skipped.
        If None, all failures are non-fatal.

    Returns
    -------
    dict
        Mapping of filename → dataset dict (as returned by ``read_nxspe``).
        Only successfully loaded files appear as keys.

    Raises
    ------
    FileNotFoundError
        If a critical file is missing from data_dir.
    RuntimeError
        If no files in file_list could be loaded at all.

    Notes
    -----
    The returned dict is used by downstream functions as:
        dataset[cfg.primary_file]  → main warm-sample dataset
    """
    if critical_files is None:
        critical_files = []

    dataset = {}

    for fname in file_list:
        full_path = os.path.join(data_dir, fname)

        # ── File existence check ──────────────────────────────────────────────
        if not os.path.exists(full_path):
            if fname in critical_files:
                raise FileNotFoundError(f"Critical file missing: {fname}")
            print(f"  skipping (not found): {fname}")
            continue

        # ── Attempt to load ───────────────────────────────────────────────────
        try:
            d = read_nxspe(full_path)
            dataset[fname] = d
            print(
                f"  loaded [hdf5]: {fname}  "
                f"Ei={d['ei']:.2f} meV  "
                f"T={d['temp']} K  "
                f"good={len(d['good'])} det"
            )
        except Exception as exc:
            if fname in critical_files:
                raise  # re-raise for critical files
            print(f"  failed to load {fname}: {exc}")

    if not dataset:
        raise RuntimeError(
            "No files loaded successfully.  "
            "Check data_dir and file_list in Config."
        )

    return dataset
