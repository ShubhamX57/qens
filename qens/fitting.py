"""
fitting.py

Two main things here:

1. extract_hwhm — bins detectors by Q and fits each bin to get the
   linewidth Gamma(Q). This is the "deterministic" path: fast, gives
   you a Gamma value and uncertainty for each Q bin.

2. The Bayesian stuff — log_likelihood, log_prior, log_posterior, find_map.
   These work together with sampling.py to do proper uncertainty quantification.
   The key insight is that we analytically marginalise over the spectral
   amplitudes (elastic, QENS, background) using NNLS, so only D and l
   are sampled by MCMC.

Everything in here assumes that fit_elastic_peak and assign_resolution
have already been run on the dataset dict.
"""

import csv
import os

import numpy as np
from scipy.optimize import curve_fit, minimize, nnls
from scipy.signal import fftconvolve

from .models import ce, gnorm, lorentz, make_basis
from .config import Config


# --- HWHM extraction ---------------------------------------------------------

def extract_hwhm(d, cfg=None):
    """
    Bin detectors by Q, average spectra in each bin, and fit to get Gamma(Q).

    The fit model is:
        S(Q, w) = A_el * R(w) + A_ql * [L_gamma ⊗ R](w) + background

    where R is the instrument Gaussian and L_gamma is a Lorentzian with
    HWHM gamma. The four parameters (A_el, A_ql, gamma, bg) are fitted
    with scipy's curve_fit, weighted by experimental uncertainties.

    Parameters
    ----------
    d   : dict   — dataset dict (needs e, data, errs, good, q, sigma_res)
    cfg : Config — analysis configuration (uses defaults if None)

    Returns
    -------
    q_centres  : array — Q bin centres in Å⁻¹
    gamma      : array — fitted HWHM values in meV
    gamma_err  : array — 1-sigma uncertainties on gamma
    eisf       : array — elastic incoherent structure factor per bin
    """
    if cfg is None:
        cfg = Config()

    good   = d["good"]
    q_arr  = d["q"][good]
    e      = d["e"]
    sr     = d["sigma_res"]

    # restrict to the configured Q range
    q_mask = (q_arr >= cfg.q_min) & (q_arr <= cfg.q_max)
    good   = good[q_mask]
    q_arr  = q_arr[q_mask]

    if len(good) < 4:
        print(f"  only {len(good)} detectors in Q=[{cfg.q_min}, {cfg.q_max}] — results may be unreliable")

    # energy mask and Q bin edges (equal-population bins)
    emask   = (e >= -cfg.ewin_hwhm) & (e <= cfg.ewin_hwhm)
    ew      = e[emask]
    q_edges = np.percentile(q_arr, np.linspace(0, 100, cfg.n_bins + 1))

    def _model(x, a_el, a_ql, gamma, bg):
        # elastic component
        el = np.exp(-0.5 * (x / sr) ** 2)
        el /= el.max()
        dt  = x[1] - x[0]
        # QENS component: Lorentzian convolved with resolution
        lor = (1 / np.pi) * gamma / (x ** 2 + gamma ** 2)
        ql  = fftconvolve(lor, el / (el.sum() * dt), mode="same") * dt
        if ql.max() > 0:
            ql /= ql.max()
        return a_el * el + a_ql * ql + bg

    q_out, g_out, ge_out, eisf_out = [], [], [], []

    for k in range(cfg.n_bins):
        in_bin = np.where(
            (q_arr >= q_edges[k]) & (q_arr < q_edges[k + 1])
        )[0]

        if len(in_bin) < 2:
            continue

        # average spectra and errors across detectors in this bin
        specs = [d["data"][good[j]][emask] for j in in_bin]
        errs  = [d["errs"][good[j]][emask] for j in in_bin]
        spec  = np.nanmean(specs, axis=0)
        spec  = np.where(np.isfinite(spec), spec, 0)
        err   = np.sqrt(np.nanmean(np.array(errs) ** 2, axis=0))
        err   = np.where(err > 0, err, spec.max() * 0.05)
        q_mid = q_arr[in_bin].mean()

        try:
            p0 = [spec.max() * 0.5, spec.max() * 0.5, max(sr, 0.05), spec.min()]
            popt, pcov = curve_fit(
                _model, ew, spec,
                p0=p0,
                sigma=err,
                bounds=(
                    [0, 0, sr * 0.2, 0],
                    [np.inf, np.inf, cfg.ewin_hwhm * 0.8, np.inf],
                ),
                maxfev=8000,
            )
            gamma_val = abs(popt[2])
            gamma_err = (
                np.sqrt(pcov[2, 2]) if np.isfinite(pcov[2, 2])
                else gamma_val * 0.1
            )
            # EISF = elastic fraction of total signal
            eisf_val = (
                popt[0] / (popt[0] + popt[1])
                if (popt[0] + popt[1]) > 0 else 0.5
            )
            q_out.append(q_mid)
            g_out.append(gamma_val)
            ge_out.append(gamma_err)
            eisf_out.append(eisf_val)

        except Exception as exc:
            print(f"  Q-bin {k} fit failed: {exc}")

    return (np.array(q_out),
            np.array(g_out),
            np.array(ge_out),
            np.array(eisf_out),)


def save_hwhm_csv(q_hwhm, g_hwhm, g_err, eisf, save_dir):
    """Save the HWHM table to CSV. Returns the path."""
    path = os.path.join(save_dir, "hwhm_table.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["q_centre_ainv", "hwhm_mev", "hwhm_err_mev", "eisf"])
        for q, g, ge, ei in zip(q_hwhm, g_hwhm, g_err, eisf):
            writer.writerow([f"{q:.4f}", f"{g:.6f}", f"{ge:.6f}", f"{ei:.4f}"])
    return path


# --- Bayesian posterior ------------------------------------------------------

def build_data_bins(d_inc, cfg=None):
    """
    Prepare Q-binned spectra for the MCMC likelihood.

    This slices the data into cfg.n_bins_mc Q bins and packages each one
    as a (e_grid, spec, errs, q_mid) tuple. The MCMC sampler will call
    log_likelihood once per sample, and it loops over these bins.

    Having fewer, wider Q bins here than in extract_hwhm is intentional —
    you want enough counts in each bin to make the likelihood meaningful,
    and the MCMC is slower than curve_fit so you don't want too many bins.
    """
    if cfg is None:
        cfg = Config()

    good   = d_inc["good"]
    q_g    = d_inc["q"][good]
    e      = d_inc["e"]

    q_mask  = (q_g >= cfg.q_min) & (q_g <= cfg.q_max)
    good    = good[q_mask]
    q_g     = q_g[q_mask]

    emask   = (e >= -cfg.ewin_mcmc) & (e <= cfg.ewin_mcmc)
    ew      = e[emask]
    q_edges = np.percentile(q_g, np.linspace(0, 100, cfg.n_bins_mc + 1))

    bins = []
    for k in range(cfg.n_bins_mc):
        mask = (q_g >= q_edges[k]) & (q_g < q_edges[k + 1])
        if mask.sum() < 2:
            continue
        idxs = good[mask]
        spec = np.nanmean([d_inc["data"][i][emask] for i in idxs], axis=0)
        errs = np.sqrt(
            np.nanmean([d_inc["errs"][i][emask] ** 2 for i in idxs], axis=0)
        )
        spec = np.where(np.isfinite(spec), spec, 0)
        errs = np.where(errs > 0, errs, spec.max() * 0.05)
        bins.append((ew, spec, errs, float(q_g[mask].mean())))

    print(f"  prepared {len(bins)} Q bins for MCMC")
    return bins


def log_likelihood(d_val, l, data_bins, sr):
    """
    Log-likelihood of the CE model given the binned data.

    For each Q bin, we analytically marginalise over the spectral
    amplitudes (elastic, QENS, background) using NNLS — this is the
    part that makes Bayesian QENS tractable. Without this trick,
    you'd need to sample five parameters per Q bin.

    Parameters
    ----------
    d_val     : float — diffusion coefficient (Å²/ps)
    l         : float — jump length (Å)
    data_bins : list  — from build_data_bins
    sr        : float — resolution sigma (meV)

    Returns
    -------
    float — log-likelihood, or -inf if parameters are unphysical
    """
    if d_val <= 0 or l <= 0:
        return -np.inf

    logl = 0.0
    for e_grid, spec, errs, q_val in data_bins:
        basis = make_basis(e_grid, q_val, d_val, l, sr)
        try:
            amp, _ = nnls(basis / errs[:, None], spec / errs)
        except Exception:
            return -np.inf
        resid  = spec - basis @ amp
        logl  -= 0.5 * np.sum((resid / errs) ** 2)

    return logl


def log_prior(d_val, l):
    """
    Flat (uninformative) prior over physically reasonable parameter ranges.

    D in (0, 3) Å²/ps covers everything from slow glassy dynamics to
    fast molecular diffusion. l in (0.5, 6) Å covers bond lengths to
    ~2nd nearest neighbour. These bounds are wide enough to be essentially
    uninformative for a typical liquid sample.

    Returns 0.0 inside the support, -inf outside.
    """
    if 0 < d_val < 3 and 0.5 < abs(l) < 6:
        return 0.0
    return -np.inf


def log_posterior(d_val, l, data_bins, sr):
    """log-posterior = log-prior + log-likelihood (unnormalised)"""
    lp = log_prior(d_val, l)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(d_val, l, data_bins, sr)


def find_map(data_bins, sr, cfg=None):
    """
    Find the Maximum A Posteriori (MAP) estimate using multi-start
    Nelder-Mead optimisation.

    We run 20 random starts from within the prior support and keep the
    best result. This is fast (~seconds) and gives a good starting point
    for the MCMC walkers.

    Returns (d_map, l_map, tau_map)
    """
    if cfg is None:
        cfg = Config()

    rng = np.random.default_rng(cfg.random_seed)

    def neg_lp(params):
        return -log_posterior(params[0], abs(params[1]), data_bins, sr)

    best_val = np.inf
    best_p   = None

    print("  finding MAP (20 random starts)...")
    for _ in range(20):
        d0 = rng.uniform(0.2, 1.0)
        l0 = rng.uniform(1.5, 3.5)
        res = minimize(
            neg_lp, [d0, l0],
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_p   = res.x

    d_map   = float(best_p[0])
    l_map   = abs(float(best_p[1]))
    tau_map = l_map ** 2 / (6 * d_map)

    print(f"  MAP: D={d_map:.5f}  l={l_map:.5f}  tau={tau_map:.5f}")
    return d_map, l_map, tau_map
