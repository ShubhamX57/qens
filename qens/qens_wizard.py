
#!/usr/bin/env python3
"""Interactive staged CLI for QENS analysis."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import pathlib
import sys
import textwrap
import traceback
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, nnls
from scipy.signal import fftconvolve

QENS_LIB_DIR = os.path.join(os.getcwd(), "qens_library")
if os.path.isdir(QENS_LIB_DIR) and QENS_LIB_DIR not in sys.path:
    sys.path.insert(0, QENS_LIB_DIR)

try:
    import qens
    from qens.config import Config
    from qens.constants import hbar_mevps
    from qens.io import load_dataset, read_nxspe_hdf5, inspect_nxspe
    from qens.preprocessing import fit_elastic_peak, assign_resolution
    from qens.models import ce, fickian, ss_model, lorentz, gnorm
    from qens.fitting import extract_hwhm, save_hwhm_csv, build_data_bins, find_map
    from qens.sampling import run_mcmc
    HAVE_QENS = True
except Exception as exc:
    HAVE_QENS = False
    QENS_IMPORT_ERROR = exc
    qens = None
    Config = object  # type: ignore
    hbar_mevps = 0.6582119514

    def _missing(*_: Any, **__: Any) -> None:
        raise RuntimeError(
            "qens_library could not be imported. Run this script from the project root "
            "or make sure qens_library/ is on PYTHONPATH."
        ) from QENS_IMPORT_ERROR

    load_dataset = read_nxspe_hdf5 = inspect_nxspe = _missing  # type: ignore
    fit_elastic_peak = assign_resolution = _missing  # type: ignore
    extract_hwhm = save_hwhm_csv = build_data_bins = find_map = run_mcmc = _missing  # type: ignore

    def ce(q: np.ndarray, D: float, l: float) -> np.ndarray:  # type: ignore
        q = np.asarray(q)
        x = q * l
        sinc = np.sinc(x / np.pi)
        return hbar_mevps * D * (1.0 - sinc) / max(l * l, 1e-12)

    def fickian(q: np.ndarray, D: float) -> np.ndarray:  # type: ignore
        q = np.asarray(q)
        return hbar_mevps * D * q * q

    def ss_model(q: np.ndarray, D: float, tau_s: float) -> np.ndarray:  # type: ignore
        q = np.asarray(q)
        return hbar_mevps * D * q * q / (1.0 + np.abs(tau_s) * D * q * q)

    def lorentz(w: np.ndarray, gamma: float) -> np.ndarray:  # type: ignore
        return (gamma / np.pi) / (w * w + gamma * gamma)

    def gnorm(w: np.ndarray, sigma: float) -> np.ndarray:  # type: ignore
        sigma = max(float(sigma), 1e-12)
        return np.exp(-(w * w) / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))

try:
    import emcee  # noqa: F401
    HAVE_EMCEE = True
except Exception:
    HAVE_EMCEE = False

MODEL_COLORS = {
    "ce": "#c0392b",
    "fickian": "#2471a3",
    "ss_model": "#1e8449",
    "lorentz": "#6c3483",
}

MODEL_LABELS = {
    "ce": "Chudley-Elliott",
    "fickian": "Fickian",
    "ss_model": "Singwi-Sjolander",
    "lorentz": "Lorentzian",
}

CMAP = LinearSegmentedColormap.from_list(
    "qens", ["#0a0e1a", "#0c2d6b", "#1565c0", "#42a5f5", "#e3f2fd", "#ff8f00", "#e65100"], N=512
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#555",
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#ccc",
    "figure.dpi": 100,
})


def builtin_registry() -> Dict[str, dict]:
    return {
        "ce": {
            "func": ce,
            "param_names": ["D", "l"],
            "p0": [0.30, 2.5],
            "bounds": ([0.0, 0.5], [3.0, 6.0]),
            "description": "Jump diffusion on a lattice.",
            "color": MODEL_COLORS["ce"],
            "label": MODEL_LABELS["ce"],
        },
        "fickian": {
            "func": fickian,
            "param_names": ["D"],
            "p0": [0.30],
            "bounds": ([0.0], [3.0]),
            "description": "Continuous Brownian diffusion.",
            "color": MODEL_COLORS["fickian"],
            "label": MODEL_LABELS["fickian"],
        },
        "ss_model": {
            "func": ss_model,
            "param_names": ["D", "tau_s"],
            "p0": [0.30, 1.0],
            "bounds": ([0.0, 0.01], [3.0, 20.0]),
            "description": "Singwi-Sjolander correlated jump diffusion.",
            "color": MODEL_COLORS["ss_model"],
            "label": MODEL_LABELS["ss_model"],
        },
        "lorentz": {
            "func": None,
            "param_names": ["mean_hwhm_mev"],
            "p0": [],
            "bounds": ([], []),
            "description": "Phenomenological constant Lorentzian width.",
            "color": MODEL_COLORS["lorentz"],
            "label": MODEL_LABELS["lorentz"],
        },
    }


def validate_model_entry(key: str, entry: dict) -> List[str]:
    errs: List[str] = []
    required = ["func", "param_names", "p0", "bounds"]
    for field_name in required:
        if field_name not in entry:
            errs.append(f"model '{key}' missing required field '{field_name}'")
    if errs:
        return errs
    if not callable(entry["func"]):
        errs.append(f"model '{key}' field 'func' must be callable")
    param_names = entry["param_names"]
    p0 = entry["p0"]
    bounds = entry["bounds"]
    if not isinstance(param_names, (list, tuple)) or not all(isinstance(p, str) for p in param_names):
        errs.append(f"model '{key}' field 'param_names' must be a list[str]")
    if not isinstance(p0, (list, tuple)):
        errs.append(f"model '{key}' field 'p0' must be a list/tuple")
    if len(param_names) != len(p0):
        errs.append(f"model '{key}' has {len(param_names)} param_names but {len(p0)} initial values")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        errs.append(f"model '{key}' field 'bounds' must be (lower, upper)")
    else:
        lo, hi = bounds
        if len(lo) != len(param_names) or len(hi) != len(param_names):
            errs.append(f"model '{key}' bounds lengths must match param_names")
    return errs


def load_custom_model_file(path: str) -> Dict[str, dict]:
    path_obj = pathlib.Path(path).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Custom model file not found: {path_obj}")
    spec = importlib.util.spec_from_file_location("qens_custom_models", str(path_obj))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model file: {path_obj}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "MODEL_INFO"):
        raise ValueError("Custom model file must expose MODEL_INFO = {...}")
    raw = getattr(module, "MODEL_INFO")
    if not isinstance(raw, dict):
        raise ValueError("MODEL_INFO must be a dictionary")
    errors: List[str] = []
    cleaned: Dict[str, dict] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            errors.append(f"model '{key}' entry must be a dict")
            continue
        errs = validate_model_entry(str(key), entry)
        if errs:
            errors.extend(errs)
            continue
        fixed = dict(entry)
        fixed.setdefault("label", str(key))
        fixed.setdefault("description", "Custom model")
        fixed.setdefault("color", "#78350f")
        cleaned[str(key)] = fixed
    if errors:
        raise ValueError("Custom model validation failed:\n- " + "\n- ".join(errors))
    return cleaned


def model_template_text() -> str:
    return textwrap.dedent(
        """
        import numpy as np

        def my_jump_model(q, D, tau):
            q = np.asarray(q)
            return 0.6582119514 * D * q**2 / (1 + tau * D * q**2)

        MODEL_INFO = {
            "my_jump_model": {
                "func": my_jump_model,
                "param_names": ["D", "tau"],
                "p0": [0.30, 1.0],
                "bounds": ([0.0, 0.01], [3.0, 20.0]),
                "label": "My Jump Model",
                "description": "Example custom Gamma(Q) model",
                "color": "#92400e",
            }
        }
        """
    ).strip()


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    if raw == "" and default is not None:
        return str(default)
    return raw


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_str}]: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def ask_int(prompt: str, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            print("Enter an integer.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}")
            continue
        return value


def ask_float(prompt: str, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    while True:
        raw = ask(prompt, f"{default}")
        try:
            value = float(raw)
        except ValueError:
            print("Enter a number.")
            continue
        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}")
            continue
        return value


def choose_one(prompt: str, options: Sequence[str], default_index: int = 0) -> str:
    for i, opt in enumerate(options, start=1):
        marker = "*" if i - 1 == default_index else " "
        print(f" {marker} {i}. {opt}")
    while True:
        raw = ask(prompt, str(default_index + 1))
        try:
            idx = int(raw) - 1
        except ValueError:
            print("Enter a valid number.")
            continue
        if 0 <= idx < len(options):
            return options[idx]
        print("Choice out of range.")


def choose_many(prompt: str, options: Sequence[str], default_indices: Optional[Sequence[int]] = None) -> List[str]:
    default_indices = list(default_indices or [])
    for i, opt in enumerate(options, start=1):
        marker = "*" if i - 1 in default_indices else " "
        print(f" {marker} {i}. {opt}")
    default_text = ",".join(str(i + 1) for i in default_indices) if default_indices else None
    while True:
        raw = ask(prompt, default_text or "")
        if raw.lower() == "all":
            return list(options)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        try:
            indices = sorted({int(p) - 1 for p in parts})
        except ValueError:
            print("Enter numbers separated by commas, or 'all'.")
            continue
        if not indices:
            print("Select at least one option.")
            continue
        if any(i < 0 or i >= len(options) for i in indices):
            print("One or more choices are out of range.")
            continue
        return [options[i] for i in indices]


@dataclass
class RunConfigData:
    files: List[str] = field(default_factory=list)
    primary_file: str = ""
    selected_models: List[str] = field(default_factory=lambda: ["ce", "fickian"])
    custom_model_file: Optional[str] = None
    fit_method: str = "ls"
    q_min: float = 0.30
    q_max: float = 2.50
    ewin: float = 0.80
    n_bins: int = 13
    q_target: float = 1.06
    n_walkers: int = 32
    n_warmup: int = 300
    n_keep: int = 1000
    thin: int = 5
    maxfev: int = 10000
    save_plots: bool = True
    save_posteriors: bool = True
    results_dir: str = "results"
    tag: str = ""
    save_spectra: bool = False
    save_chain: bool = False
    random_seed: int = 42

    def to_jsonable(self) -> dict:
        return asdict(self)


class RunLogger:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def log(self, message: str, kind: str = "info") -> None:
        ts = dt.datetime.now().strftime("%H:%M:%S")
        prefix = {"info": "[INFO]", "ok": "[ OK ]", "warn": "[WARN]", "err": "[ERR ]"}.get(kind, "[INFO]")
        line = f"[{ts}] {prefix} {message}"
        self.lines.append(line)
        print(line)

    def save(self, path: str) -> None:
        pathlib.Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def discover_nxspe_in_directory(directory: str) -> List[str]:
    path = pathlib.Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return sorted(str(p) for p in path.glob("*.nxspe"))


def read_file_list(path: str) -> List[str]:
    lines = pathlib.Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    return [str(pathlib.Path(line.strip()).expanduser().resolve()) for line in lines if line.strip()]


def normalise_files(files: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for f in files:
        p = str(pathlib.Path(f).expanduser().resolve())
        if p not in seen and pathlib.Path(p).exists() and p.endswith(".nxspe"):
            seen.add(p)
            out.append(p)
    return sorted(out)


def file_metadata(path: str) -> dict:
    meta = {"name": pathlib.Path(path).name, "path": path}
    try:
        d = read_nxspe_hdf5(path)
        good = d.get("good", [])
        n_good = len(good) if hasattr(good, "__len__") else None
        meta.update({"temp": d.get("temp"), "ei": d.get("ei"), "n_good": n_good})
    except Exception:
        meta.update({"temp": None, "ei": None, "n_good": None})
    return meta


def section_data_loading(cfg: RunConfigData) -> None:
    header("1. Data Loading")
    print("Possible options:")
    print("  1. Scan a directory for .nxspe files")
    print("  2. Enter file paths manually")
    print("  3. Load from a file list")
    print("  4. Keep current selection")
    choice = choose_one("Choose input mode", [
        "Scan a directory for .nxspe files",
        "Enter file paths manually",
        "Load from a text file list",
        "Keep current selection",
    ])

    files: List[str] = list(cfg.files)
    if choice.startswith("Scan"):
        directory = ask("Directory", os.getcwd())
        files = discover_nxspe_in_directory(directory)
    elif choice.startswith("Enter"):
        print("Enter one or more file paths separated by commas.")
        raw = ask("Paths")
        files = [p.strip() for p in raw.split(",") if p.strip()]
    elif choice.startswith("Load from"):
        list_path = ask("Path to file list")
        files = read_file_list(list_path)
    elif not files:
        print("No current selection exists yet.")
        return section_data_loading(cfg)

    files = normalise_files(files)
    if not files:
        print("No readable .nxspe files were found.")
        return section_data_loading(cfg)

    cfg.files = files
    print(f"\nFound {len(files)} file(s):")
    preview = [file_metadata(p) for p in files[: min(10, len(files))]]
    for idx, item in enumerate(preview):
        temp = f"T={item['temp']} K" if item.get("temp") is not None else "T=?"
        ei = f"Ei={item['ei']:.2f} meV" if item.get("ei") is not None else "Ei=?"
        print(f"  [{idx}] {item['name']}  |  {temp}  |  {ei}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

    primary_default = cfg.primary_file if cfg.primary_file in cfg.files else cfg.files[0]
    cfg.primary_file = ask("Primary file for S(Q,w) / residual plots", primary_default)
    if cfg.primary_file not in cfg.files and pathlib.Path(cfg.primary_file).exists():
        cfg.files.append(str(pathlib.Path(cfg.primary_file).resolve()))
    elif cfg.primary_file not in cfg.files:
        cfg.primary_file = cfg.files[0]


def section_model_selection(cfg: RunConfigData, registry: Dict[str, dict]) -> Dict[str, dict]:
    header("2. Model Selection")
    print("Built-in models:")
    keys = list(registry.keys())
    for i, key in enumerate(keys, start=1):
        info = registry[key]
        print(f"  {i}. {info.get('label', key)} — {info.get('description', '')}")
    print("\nCustom model schema:")
    print(textwrap.indent(model_template_text(), prefix="    "))

    use_custom = ask_yes_no("Load or reload a custom model file", default=cfg.custom_model_file is not None)
    local_registry = dict(registry)
    if use_custom:
        while True:
            path = ask("Custom model file path", cfg.custom_model_file or "custom_models.py")
            try:
                custom = load_custom_model_file(path)
            except Exception as exc:
                print(f"Custom model validation failed:\n{exc}")
                if not ask_yes_no("Try another file", True):
                    break
                continue
            cfg.custom_model_file = str(pathlib.Path(path).expanduser().resolve())
            local_registry.update(custom)
            print("Loaded custom models:")
            for name, entry in custom.items():
                print(f"  - {name}: {entry.get('description', '')}")
            break

    model_keys = list(local_registry.keys())
    labels = [f"{local_registry[k].get('label', k)} [{k}]" for k in model_keys]
    default_idx = [model_keys.index(k) for k in cfg.selected_models if k in model_keys]
    selected = choose_many("Select one or more models by number, or 'all'", labels, default_indices=default_idx)
    cfg.selected_models = [model_keys[labels.index(s)] for s in selected]
    return local_registry


def section_fit_method(cfg: RunConfigData) -> None:
    header("3. Fitting Method")
    choice = choose_one("Choose fitting mode", ["Least Squares", "Bayesian / MCMC"], default_index=0 if cfg.fit_method == "ls" else 1)
    cfg.fit_method = "ls" if choice.startswith("Least") else "bayes"
    if cfg.fit_method == "ls":
        print("\nLeast-squares settings:")
        cfg.maxfev = ask_int("Max function evaluations", cfg.maxfev, 100, 200000)
    else:
        print("\nBayesian / MCMC settings:")
        cfg.n_walkers = ask_int("Walkers", cfg.n_walkers, 8, 512)
        cfg.n_warmup = ask_int("Burn-in", cfg.n_warmup, 10, 100000)
        cfg.n_keep = ask_int("Samples to keep", cfg.n_keep, 100, 500000)
        cfg.thin = ask_int("Thin", cfg.thin, 1, 1000)


def section_analysis_settings(cfg: RunConfigData) -> None:
    header("4. Analysis Settings")
    print("Possible options:")
    print("  - Q range")
    print("  - energy window (+/- meV)")
    print("  - number of Q bins")
    print("  - Q target for detailed residual plot")
    cfg.q_min = ask_float("Q min (1/A)", cfg.q_min, 0.0)
    cfg.q_max = ask_float("Q max (1/A)", cfg.q_max, cfg.q_min)
    cfg.ewin = ask_float("Energy half-window +/-E (meV)", cfg.ewin, 0.05)
    cfg.n_bins = ask_int("Number of Q bins", cfg.n_bins, 2, 200)
    cfg.q_target = ask_float("Q target for single-spectrum fit", cfg.q_target, 0.0)
    cfg.random_seed = ask_int("Random seed", cfg.random_seed, 0)


def section_output_settings(cfg: RunConfigData) -> None:
    header("5. Output Settings")
    cfg.results_dir = ask("Base results directory", cfg.results_dir)
    cfg.tag = ask("Optional run tag", cfg.tag)
    cfg.save_plots = ask_yes_no("Save standard plots", cfg.save_plots)
    cfg.save_posteriors = ask_yes_no("Save posterior plots when available", cfg.save_posteriors)
    cfg.save_chain = ask_yes_no("Save raw MCMC chains when available", cfg.save_chain)
    cfg.save_spectra = ask_yes_no("Save extra spectrum tables", cfg.save_spectra)


def section_review(cfg: RunConfigData, registry: Dict[str, dict]) -> str:
    header("6. Review")
    print(f"Files selected      : {len(cfg.files)}")
    print(f"Primary file        : {pathlib.Path(cfg.primary_file).name if cfg.primary_file else '(none)'}")
    print(f"Models              : {', '.join(registry[m].get('label', m) for m in cfg.selected_models)}")
    print(f"Fit method          : {'Bayesian / MCMC' if cfg.fit_method == 'bayes' else 'Least Squares'}")
    print(f"Q range             : {cfg.q_min:.3f} to {cfg.q_max:.3f} 1/A")
    print(f"Energy window       : +/- {cfg.ewin:.3f} meV")
    print(f"Q bins              : {cfg.n_bins}")
    print(f"Q target            : {cfg.q_target:.3f} 1/A")
    print(f"Results directory   : {cfg.results_dir}")
    print(f"Tag                 : {cfg.tag or '(none)'}")
    if cfg.fit_method == "bayes":
        print(f"Walkers / burn / keep / thin : {cfg.n_walkers} / {cfg.n_warmup} / {cfg.n_keep} / {cfg.thin}")
    options = [
        "Run",
        "Edit data loading",
        "Edit model selection",
        "Edit fit method",
        "Edit analysis settings",
        "Edit output settings",
        "Save config and quit",
        "Quit without running",
    ]
    return choose_one("Choose action", options)


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fit_gamma_model(q_arr: np.ndarray, g_arr: np.ndarray, model_name: str, registry: Dict[str, dict], maxfev: int) -> dict:
    q_arr = np.asarray(q_arr, dtype=float)
    g_arr = np.asarray(g_arr, dtype=float)
    try:
        if model_name == "lorentz":
            return {"mean_hwhm_mev": float(np.mean(g_arr))}
        info = registry[model_name]
        func = info["func"]
        p0 = info["p0"]
        bounds = info["bounds"]
        p, cov = curve_fit(func, q_arr, g_arr, p0=p0, bounds=bounds, maxfev=maxfev)
        res = {name: float(val) for name, val in zip(info["param_names"], p)}
        res["cov"] = cov.tolist() if cov is not None else None
        if model_name == "ce":
            res["tau"] = float(res["l"] ** 2 / max(6 * res["D"], 1e-12))
        return res
    except Exception as exc:
        return {"error": str(exc)}


def fig_sqw_map(d_inc: dict, ewin: float = 1.0) -> plt.Figure:
    good = d_inc["good"]
    q = d_inc["q"]
    e = d_inc["e"]
    emask = (e >= -ewin) & (e <= ewin)
    qs = np.argsort(q[good])
    img = d_inc["data"][np.ix_(good, emask)]
    img = np.where(np.isfinite(img) & (img > 0), img, np.nan)
    ism = gaussian_filter(np.where(np.isfinite(img[qs]), img[qs], 0), sigma=[1.5, 0.8])
    ism[ism <= 0] = np.nan
    vmin, vmax = np.nanpercentile(ism, 2), np.nanpercentile(ism, 99)

    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    im = ax.pcolormesh(e[emask], q[good][qs], ism, cmap=CMAP, norm=LogNorm(vmin=max(vmin, 1e-6), vmax=vmax), shading="auto", rasterized=True)
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel(r"Energy transfer  $\hbar\omega$  (meV)")
    ax.set_ylabel(r"Momentum transfer  $Q$  ($\AA^{-1}$)")
    ax.set_title(r"$S(Q,\omega)$  measured intensity")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.038)
    cb.set_label(r"$S(Q,\omega)$  (a.u.)")
    fig.tight_layout()
    return fig


def fig_fit_residuals(d_inc: dict, d_map: float, l_map: float, q_target: float = 1.06) -> tuple[plt.Figure, float]:
    gp = d_inc["good"]
    qg = d_inc["q"][gp]
    sr = d_inc["sigma_res"]
    emask = (d_inc["e"] >= -0.8) & (d_inc["e"] <= 0.8)
    ew = d_inc["e"][emask]
    near = np.where(np.abs(qg - q_target) < 0.10)[0]
    if len(near) == 0:
        near = np.argsort(np.abs(qg - q_target))[:4]
    spec = np.nanmean([d_inc["data"][gp[j]][emask] for j in near], axis=0)
    errs = np.sqrt(np.nanmean([d_inc["errs"][gp[j]][emask] ** 2 for j in near], axis=0))
    spec = np.where(np.isfinite(spec), spec, 0)
    errs = np.where(errs > 0, errs, max(spec.max(), 1e-8) * 0.05)
    sn = spec / max(spec.max(), 1e-12)
    en = errs / max(spec.max(), 1e-12)

    wf = np.linspace(-0.8, 0.8, 1000)
    dtw = wf[1] - wf[0]
    gamma = ce(np.asarray([q_target]), d_map, l_map)[0]
    el = gnorm(wf, sr)
    el = el / max(el.max(), 1e-12)
    ql_raw = fftconvolve(lorentz(wf, gamma), gnorm(wf, sr), mode="same") * dtw
    ql = ql_raw / max(ql_raw.max(), 1e-12)
    amp, _ = nnls(np.column_stack([el, ql, np.ones(len(wf))]), np.interp(wf, ew, sn))
    fit = amp[0] * el + amp[1] * ql + amp[2]
    fit_d = np.interp(ew, wf, fit)
    resid = (sn - fit_d) / np.where(en > 0, en, 1.0)
    chi2r = float(np.sum(resid**2) / max(len(resid) - 4, 1))

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.1), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    axes[0].errorbar(ew, sn, yerr=en, fmt=".", color="#333", ms=3.5, elinewidth=0.7, alpha=0.8, label="Data", zorder=5)
    axes[0].fill_between(wf, amp[2], amp[0] * el + amp[2], alpha=0.22, color="#1565c0", label="Elastic")
    axes[0].fill_between(wf, amp[2], amp[1] * ql + amp[2], alpha=0.22, color="#e65100", label="Quasi-elastic")
    axes[0].plot(wf, fit, "-", color="#c0392b", lw=2.0, label=rf"CE-like spectrum fit  $\chi^2_r={chi2r:.2f}$")
    axes[0].set_ylabel(r"$S(Q,\omega)$  normalised")
    axes[0].set_title(rf"Single-spectrum fit  $Q={q_target:.2f}$ $\AA^{{-1}}$")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.18, lw=0.6)
    _despine(axes[0])

    axes[1].axhline(0, color="#555", lw=0.9)
    axes[1].axhline(2, color="#999", ls="--", lw=0.7)
    axes[1].axhline(-2, color="#999", ls="--", lw=0.7)
    axes[1].plot(ew, resid, ".", color="#c0392b", ms=3.5, alpha=0.9)
    axes[1].set_ylabel("resid / σ")
    axes[1].set_xlabel(r"Energy transfer  $\hbar\omega$  (meV)")
    axes[1].grid(True, alpha=0.18, lw=0.6)
    _despine(axes[1])
    fig.tight_layout()
    return fig, chi2r


def fig_hwhm_comparison(q_hwhm: np.ndarray, g_hwhm: np.ndarray, g_err: np.ndarray, model_results: dict, registry: Dict[str, dict], res_hwhm_uev: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    q2 = q_hwhm**2
    ax.errorbar(q2, g_hwhm * 1000, yerr=g_err * 1000, fmt="o", ms=5.5, color="#111", ecolor="#777", capsize=3, label="Extracted HWHM")
    q_dense = np.linspace(max(0.0, q_hwhm.min() * 0.95), q_hwhm.max() * 1.05, 400)
    q2_dense = q_dense**2

    for model, fit in model_results.items():
        if "error" in fit:
            continue
        info = registry[model]
        label = info.get("label", model)
        color = info.get("color", "#333")
        if model == "lorentz":
            pred = np.full_like(q_dense, fit["mean_hwhm_mev"])
        else:
            params = [fit[name] for name in info["param_names"]]
            pred = info["func"](q_dense, *params)
        ax.plot(q2_dense, pred * 1000, lw=2.0, label=label, color=color)

    ax.axhline(res_hwhm_uev, color="#888", lw=1.2, ls="--", label=f"Resolution HWHM = {res_hwhm_uev:.0f} µeV")
    ax.axhspan(0, res_hwhm_uev * 1.15, alpha=0.04, color="#888")
    ax.set_xlabel(r"$Q^2$  ($\AA^{-2}$)")
    ax.set_ylabel(r"$\Gamma(Q)$  ($\mu$eV)")
    ax.set_title(r"$\Gamma(Q)$ vs $Q^2$")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xlim(left=-0.04)
    ax.set_ylim(bottom=-8)
    ax.grid(True, alpha=0.18, lw=0.6)
    _despine(ax)
    fig.tight_layout()
    return fig


def fig_posteriors_multi(samples_map: Dict[str, np.ndarray], registry: Dict[str, dict]) -> Optional[plt.Figure]:
    entries = [(m, s) for m, s in samples_map.items() if s is not None and len(s) > 0]
    if not entries:
        return None
    n = len(entries)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.6 * n), squeeze=False)
    for row, (model, smp) in enumerate(entries):
        color = registry[model].get("color", "#333")
        label = registry[model].get("label", model)
        d_s = smp[:, 0]
        l_s = np.abs(smp[:, 1]) if smp.shape[1] > 1 else np.full_like(d_s, np.nan)
        tau_s = np.where(np.isfinite(l_s), l_s**2 / np.maximum(6 * d_s, 1e-12), np.nan)
        for ci, (arr, name, unit) in enumerate([
            (d_s, r"$D$", r"$\AA^2\,\mathrm{ps}^{-1}$"),
            (l_s, r"$\ell$", r"$\AA$"),
            (tau_s, r"$\tau$", r"$\mathrm{ps}$"),
        ]):
            ax = axes[row][ci]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center")
                ax.axis("off")
                continue
            med = np.median(arr)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            cnt, _, _ = ax.hist(arr, bins=60, density=True, color=color, alpha=0.75, edgecolor="white", lw=0.2)
            pk = max(cnt.max(), 1e-12)
            ax.axvspan(lo, hi, alpha=0.16, color=color)
            ax.axvline(med, color="#111", lw=1.8, label=rf"Median={med:.4f}")
            ax.set_xlabel(f"{name} ({unit})")
            ax.set_ylabel("Density")
            ax.set_ylim(0, pk * 1.28)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.15, lw=0.6)
            _despine(ax)
            if ci == 0:
                ax.set_title(label, fontsize=11, color=color, fontweight="bold", loc="left")
    fig.suptitle("Posterior distributions", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def compile_results_table(d_inc: dict, model_results: dict, samples_map: dict, method: str, chi2r: float, q_hwhm: np.ndarray) -> List[dict]:
    rows: List[dict] = []
    rows.append({"section": "dataset", "parameter": "File", "value": d_inc.get("name"), "unit": "", "notes": ""})
    rows.append({"section": "dataset", "parameter": "Temperature", "value": d_inc.get("temp"), "unit": "K", "notes": ""})
    rows.append({"section": "dataset", "parameter": "Ei", "value": d_inc.get("ei"), "unit": "meV", "notes": ""})
    rows.append({"section": "dataset", "parameter": "Resolution FWHM", "value": d_inc.get("fwhm_res", np.nan) * 1000, "unit": "µeV", "notes": d_inc.get("res_source", "")})
    rows.append({"section": "fit", "parameter": "Method", "value": method, "unit": "", "notes": ""})
    rows.append({"section": "fit", "parameter": "Spectrum chi2r", "value": chi2r, "unit": "", "notes": "ideal ~ 1"})
    rows.append({"section": "fit", "parameter": "Q bins", "value": len(q_hwhm), "unit": "", "notes": ""})
    for model, ls in model_results.items():
        if "error" in ls:
            rows.append({"section": model, "parameter": "status", "value": "failed", "unit": "", "notes": ls["error"]})
            continue
        for key, value in ls.items():
            if key == "cov":
                continue
            rows.append({"section": model, "parameter": key, "value": value, "unit": "", "notes": ""})
        if model in samples_map:
            smp = samples_map[model]
            if smp is not None and len(smp) > 0:
                for idx, name in enumerate(["D", "l"][: smp.shape[1]]):
                    arr = smp[:, idx]
                    lo, hi = np.percentile(arr, [2.5, 97.5])
                    rows.append({"section": f"{model}_bayes", "parameter": name + "_median", "value": float(np.median(arr)), "unit": "", "notes": f"95% CI [{lo:.5f}, {hi:.5f}]"})
    return rows


def save_csv_rows(rows: List[dict], path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["section", "parameter", "value", "unit", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def execute_run(cfg: RunConfigData, registry: Dict[str, dict]) -> str:
    if not HAVE_QENS:
        raise RuntimeError(str(QENS_IMPORT_ERROR))

    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{cfg.tag}" if cfg.tag else ""
    run_dir = pathlib.Path(cfg.results_dir).expanduser().resolve() / f"{run_ts}{tag}"
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger()
    logger.log(f"Output folder: {run_dir}")
    logger.log(f"Models: {[registry[m].get('label', m) for m in cfg.selected_models]}")
    logger.log(f"Fit method: {'Bayesian / MCMC' if cfg.fit_method == 'bayes' else 'Least Squares'}")
    logger.log(f"Q range: {cfg.q_min:.3f} to {cfg.q_max:.3f} 1/A")
    logger.log(f"Energy window: +/- {cfg.ewin:.3f} meV")

    np.random.seed(cfg.random_seed)
    grouped: Dict[str, List[str]] = {}
    for p in cfg.files:
        grouped.setdefault(str(pathlib.Path(p).parent), []).append(pathlib.Path(p).name)

    dataset: Dict[str, dict] = {}
    for directory, fnames in grouped.items():
        primary_name = pathlib.Path(cfg.primary_file).name
        critical = [primary_name] if str(pathlib.Path(cfg.primary_file).parent.resolve()) == str(pathlib.Path(directory).resolve()) else []
        dataset.update(load_dataset(fnames, data_dir=directory, critical_files=critical))
    if not dataset:
        raise RuntimeError("No files loaded.")

    for d in dataset.values():
        fit_elastic_peak(d)
    assign_resolution(dataset)

    primary_name = pathlib.Path(cfg.primary_file).name
    d_inc = dataset.get(primary_name, next(iter(dataset.values())))
    logger.log(f"Loaded {len(dataset)} file(s). Primary: {d_inc['name']}", "ok")

    run_config = Config(
        primary_file=d_inc["name"],
        q_min=cfg.q_min,
        q_max=cfg.q_max,
        n_bins=cfg.n_bins,
        n_bins_mc=max(4, cfg.n_bins - 3),
        ewin_hwhm=cfg.ewin,
        ewin_mcmc=cfg.ewin,
        n_walkers=cfg.n_walkers,
        n_warmup=cfg.n_warmup,
        n_keep=cfg.n_keep,
        thin=cfg.thin,
        random_seed=cfg.random_seed,
        save_dir=str(run_dir),
    )
    if hasattr(run_config, "to_json"):
        run_config.to_json(str(run_dir / "config_from_qens.json"))
    (run_dir / "wizard_config.json").write_text(json.dumps(cfg.to_jsonable(), indent=2), encoding="utf-8")
    (run_dir / "discovered_files.txt").write_text("\n".join(cfg.files) + "\n", encoding="utf-8")

    logger.log("Rendering S(Q,w) map")
    if cfg.save_plots:
        save_figure(fig_sqw_map(d_inc, ewin=min(cfg.ewin, 1.0)), str(fig_dir / "sqw_map.png"))

    logger.log("Extracting HWHM")
    q_hwhm, g_hwhm, g_err, eisf = extract_hwhm(d_inc, run_config)
    if len(q_hwhm) == 0:
        raise RuntimeError("No HWHM bins converged.")
    save_hwhm_csv(q_hwhm, g_hwhm, g_err, eisf, str(run_dir))
    logger.log(f"{len(q_hwhm)} HWHM bins extracted", "ok")

    model_results: Dict[str, dict] = {}
    samples_map: Dict[str, np.ndarray] = {}
    best_model: Optional[str] = None
    best_chi2r = math.inf
    best_d = 0.30
    best_l = 2.50
    bayes_dbins = None

    for model in cfg.selected_models:
        logger.log(f"Fitting {registry[model].get('label', model)}")
        fit = fit_gamma_model(q_hwhm, g_hwhm, model, registry, cfg.maxfev)
        model_results[model] = fit
        if "error" in fit:
            logger.log(f"Fit failed: {fit['error']}", "warn")
            continue
        logger.log(", ".join(f"{k}={v:.5g}" for k, v in fit.items() if isinstance(v, (float, int))), "ok")

        if model == "ce":
            pred = ce(q_hwhm, fit["D"], fit["l"])
            chi2r = float(np.sum((g_hwhm - pred) ** 2 / np.where(g_err > 0, g_err**2, 1e-30)) / max(len(q_hwhm) - 2, 1))
            fit["_chi2r"] = chi2r
            if chi2r < best_chi2r:
                best_chi2r = chi2r
                best_model = model
                best_d = fit["D"]
                best_l = fit["l"]
        elif model == "fickian":
            pred = fickian(q_hwhm, fit["D"])
            chi2r = float(np.sum((g_hwhm - pred) ** 2 / np.where(g_err > 0, g_err**2, 1e-30)) / max(len(q_hwhm) - 1, 1))
            fit["_chi2r"] = chi2r
            if chi2r < best_chi2r:
                best_chi2r = chi2r
                best_model = model
                best_d = fit["D"]
        elif model == "ss_model":
            pred = ss_model(q_hwhm, fit["D"], fit["tau_s"])
            chi2r = float(np.sum((g_hwhm - pred) ** 2 / np.where(g_err > 0, g_err**2, 1e-30)) / max(len(q_hwhm) - 2, 1))
            fit["_chi2r"] = chi2r
            if chi2r < best_chi2r:
                best_chi2r = chi2r
                best_model = model
                best_d = fit["D"]
        elif model in registry and registry[model].get("func") is not None:
            params = [fit[name] for name in registry[model]["param_names"]]
            pred = registry[model]["func"](q_hwhm, *params)
            dof = max(len(q_hwhm) - len(params), 1)
            chi2r = float(np.sum((g_hwhm - pred) ** 2 / np.where(g_err > 0, g_err**2, 1e-30)) / dof)
            fit["_chi2r"] = chi2r

        if cfg.fit_method == "bayes" and model == "ce":
            logger.log("Running Bayesian MCMC for CE")
            if not HAVE_EMCEE:
                logger.log("emcee is not available; skipping Bayesian sampling.", "warn")
            else:
                try:
                    if bayes_dbins is None:
                        bayes_dbins = build_data_bins(d_inc, run_config)
                    d_map, l_map, _ = find_map(bayes_dbins, d_inc["sigma_res"], run_config)
                    smp = run_mcmc(bayes_dbins, d_inc["sigma_res"], d_map, l_map, run_config)
                    samples_map[model] = smp
                    if cfg.save_chain:
                        np.save(run_dir / f"posterior_{model}.npy", smp)
                    logger.log(f"Saved {len(smp)} posterior samples", "ok")
                    if best_model == model:
                        best_d = d_map
                        best_l = l_map
                except Exception as exc:
                    logger.log(f"Bayesian sampling failed: {exc}", "warn")

    if best_model is None:
        best_model = cfg.selected_models[0]
        best = model_results.get(best_model, {})
        best_d = float(best.get("D", best_d))
        best_l = float(best.get("l", best_l))

    logger.log(f"Rendering residual plot using {registry[best_model].get('label', best_model)}")
    fit_fig, spectrum_chi2r = fig_fit_residuals(d_inc, best_d, best_l, q_target=cfg.q_target)
    if cfg.save_plots:
        save_figure(fit_fig, str(fig_dir / f"fit_residuals_q{cfg.q_target:.2f}.png"))
    else:
        plt.close(fit_fig)

    logger.log("Rendering Gamma vs Q^2 plot")
    hwhm_fig = fig_hwhm_comparison(q_hwhm, g_hwhm, g_err, model_results, registry, res_hwhm_uev=d_inc["fwhm_res"] / 2 * 1000)
    if cfg.save_plots:
        save_figure(hwhm_fig, str(fig_dir / "gamma_vs_q2.png"))
    else:
        plt.close(hwhm_fig)

    if cfg.save_posteriors and samples_map:
        post_fig = fig_posteriors_multi(samples_map, registry)
        if post_fig is not None:
            save_figure(post_fig, str(fig_dir / "posteriors.png"))
            logger.log("Posterior plot saved", "ok")

    rows = compile_results_table(d_inc, model_results, samples_map, cfg.fit_method, spectrum_chi2r, q_hwhm)
    save_csv_rows(rows, str(run_dir / "fit_summary.csv"))

    results_json = {
        "timestamp": dt.datetime.now().isoformat(),
        "dataset": d_inc["name"],
        "temperature_K": int(d_inc["temp"]),
        "Ei_meV": float(d_inc["ei"]),
        "method": cfg.fit_method,
        "res_fwhm_meV": float(d_inc["fwhm_res"]),
        "res_source": d_inc.get("res_source", ""),
        "spectrum_chi2r": float(spectrum_chi2r),
        "models": model_results,
    }
    (run_dir / "results.json").write_text(json.dumps(results_json, indent=2, default=float), encoding="utf-8")

    logger.save(str(run_dir / "run_log.txt"))
    logger.log(f"Done. Results saved to: {run_dir}", "ok")
    return str(run_dir)


def load_config_json(path: str) -> RunConfigData:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    cfg = RunConfigData()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def save_config_json(cfg: RunConfigData, path: str) -> None:
    pathlib.Path(path).write_text(json.dumps(cfg.to_jsonable(), indent=2), encoding="utf-8")


def interactive_main(initial_cfg: Optional[RunConfigData] = None) -> int:
    cfg = initial_cfg or RunConfigData()
    registry = builtin_registry()

    while True:
        if not cfg.files:
            section_data_loading(cfg)
        registry = section_model_selection(cfg, registry)
        section_fit_method(cfg)
        section_analysis_settings(cfg)
        section_output_settings(cfg)

        while True:
            action = section_review(cfg, registry)
            if action == "Run":
                try:
                    execute_run(cfg, registry)
                    return 0
                except Exception as exc:
                    print("\nRun failed:")
                    print(exc)
                    print(traceback.format_exc())
                    if ask_yes_no("Return to review screen", True):
                        continue
                    return 1
            if action == "Edit data loading":
                section_data_loading(cfg)
                break
            if action == "Edit model selection":
                registry = section_model_selection(cfg, builtin_registry())
                continue
            if action == "Edit fit method":
                section_fit_method(cfg)
                continue
            if action == "Edit analysis settings":
                section_analysis_settings(cfg)
                continue
            if action == "Edit output settings":
                section_output_settings(cfg)
                continue
            if action == "Save config and quit":
                path = ask("Config output path", "qens_run_config.json")
                save_config_json(cfg, path)
                print(f"Saved config to {path}")
                return 0
            if action == "Quit without running":
                return 0


def noninteractive_run(cfg: RunConfigData) -> int:
    registry = builtin_registry()
    if cfg.custom_model_file:
        registry.update(load_custom_model_file(cfg.custom_model_file))
    if not cfg.files:
        raise ValueError("No input files specified in config")
    if not cfg.selected_models:
        raise ValueError("No models selected in config")
    execute_run(cfg, registry)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive staged CLI for QENS analysis")
    p.add_argument("--config", help="Load settings from JSON config")
    p.add_argument("--run-config", action="store_true", help="Run non-interactively from --config")
    p.add_argument("--write-model-template", help="Write a custom model template to the given path and exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_model_template:
        pathlib.Path(args.write_model_template).write_text(model_template_text() + "\n", encoding="utf-8")
        print(f"Wrote model template to {args.write_model_template}")
        return 0

    cfg = load_config_json(args.config) if args.config else RunConfigData()
    if args.run_config:
        return noninteractive_run(cfg)

    print("QENS Analysis Wizard")
    print("This script provides staged sections for loading, model selection, fitting, and output setup.")
    if not HAVE_QENS:
        print("\nWarning: qens_library could not be imported.")
        print(QENS_IMPORT_ERROR)
        print("The wizard can still build configs and custom model files, but analysis will fail until qens_library is available.")
    return interactive_main(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
