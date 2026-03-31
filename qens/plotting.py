"""
plotting.py

All the figures in one place.

Each function returns (fig, ax) or (fig, axes) so you can further
customise before saving. They also accept a save_path argument
that writes a PDF directly if you provide one.

The colour choices here are deliberate:
    red  (#c0392b) — CE model
    blue (#2471a3) — Fickian
    green (#1e8449) — Singwi-Sjolander
    black — data points (always black so they read well against any model)

These aren't random — they match what's used in the analysis app,
so figures from the interactive tool and standalone scripts look consistent.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls
from scipy.signal import fftconvolve

from .models import ce, fickian, gnorm, lorentz


# a custom colormap for the S(Q,w) maps — dark blue through pale yellow to orange-red.
# looks good in print and is reasonably colourblind-friendly.
_sqw_cmap = LinearSegmentedColormap.from_list(
    "qens",
    ["#0a0e1a", "#0c2d6b", "#1565c0", "#42a5f5", "#e3f2fd", "#ff8f00", "#e65100"],
    N=512,
)


def _despine(ax):
    """remove the top and right spines — standard for publication figures"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_overview(dataset, save_path=None):
    """
    Panel overview of all loaded datasets.

    Useful as a sanity check after loading — shows you the elastic
    peak shape for each file, with colour coding:
        green  — 260 K frozen reference
        red    — INC scans at 290 K
        blue   — COH scans at 290 K

    Parameters
    ----------
    dataset   : dict — from load_dataset, after preprocessing
    save_path : str  — if given, saves a PDF here

    Returns
    -------
    fig, axes
    """
    plot_order = [
        ("benzene_290_124_inc.nxspe",  "INC 290 K  1.24 meV"),
        ("benzene_290_124_coh.nxspe",  "COH 290 K  1.24 meV"),
        ("benzene_290_360_inc.nxspe",  "INC 290 K  3.60 meV  [primary]"),
        ("benzene_290_360_coh.nxspe",  "COH 290 K  3.60 meV"),
        ("benzene_260_360_inc.nxspe",  "INC 260 K  3.60 meV  [resolution]"),
        ("benzene_290_861_inc.nxspe",  "INC 290 K  8.61 meV"),
        ("benzene_290_861_coh.nxspe",  "COH 290 K  8.61 meV"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, (fname, label) in zip(axes, plot_order):
        if fname not in dataset:
            ax.text(0.5, 0.5, "file missing", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
            ax.set_title(label, fontsize=8)
            continue

        d    = dataset[fname]
        e    = d["e"]
        good = d["good"]
        n_lo = max(2, len(good) // 5)

        avg  = np.nanmean([d["data"][good[j]] for j in range(n_lo)], axis=0)
        avg  = np.where(np.isfinite(avg), avg, 0)
        ewin = min(0.5 * d["ei"], 1.2)
        mask = (e >= -ewin) & (e <= ewin)

        y = avg[mask] / avg[mask].max() if avg[mask].max() > 0 else avg[mask]

        # colour by type
        if d["temp"] == 260:
            col = "#2ca02c"     # green = resolution reference
        elif d["kind"] == "inc":
            col = "#c0392b"     # red = INC
        else:
            col = "#2471a3"     # blue = COH

        ax.plot(e[mask], y, color=col, lw=1.8)
        ax.axvline(0, color="#aaa", lw=0.8, ls=":", alpha=0.6)
        ax.set_title(label, fontsize=8, color=col)
        ax.set_xlabel("ω (meV)", fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
        ax.text(
            0.03, 0.96,
            f"E0={d['e0']:+.3f}\nFWHM={d['fwhm_res']*1000:.0f} µeV [{d['res_source']}]",
            transform=ax.transAxes, va="top", fontsize=5.5,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
        _despine(ax)

    axes[-1].axis("off")
    fig.suptitle("All datasets  |  red=INC  blue=COH  green=260 K resolution", fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, axes


def plot_spectrum(d_inc, d_map, l_map, q_target=1.06, d_res=None, save_path=None):
    """
    Single-Q diagnostic: measured spectrum vs CE model, decomposed
    into elastic and quasi-elastic components.

    The CE model uses d_map and l_map for the linewidth. If d_res is
    provided (a 260 K or COH dataset) it's plotted as a dashed red line
    to show the resolution.

    Parameters
    ----------
    d_inc    : dict  — primary INC dataset
    d_map    : float — MAP diffusion coefficient (Å²/ps)
    l_map    : float — MAP jump length (Å)
    q_target : float — target Q value in Å⁻¹
    d_res    : dict  — resolution reference dataset (optional)
    save_path: str   — save PDF here if given

    Returns
    -------
    fig, ax
    """
    gp    = d_inc["good"]
    qg    = d_inc["q"][gp]
    sr    = d_inc["sigma_res"]
    emask = (d_inc["e"] >= -0.8) & (d_inc["e"] <= 0.8)
    ew    = d_inc["e"][emask]

    # find detectors near the target Q
    near = np.where(np.abs(qg - q_target) < 0.10)[0]
    if len(near) == 0:
        near = np.argsort(np.abs(qg - q_target))[:4]

    spec = np.nanmean([d_inc["data"][gp[j]][emask] for j in near], axis=0)
    errs = np.sqrt(np.nanmean(
        [d_inc["errs"][gp[j]][emask] ** 2 for j in near], axis=0
    ))
    spec = np.where(np.isfinite(spec), spec, 0)
    errs = np.where(errs > 0, errs, spec.max() * 0.05)

    # normalise for plotting
    sn = spec / spec.max()
    en = errs / spec.max()

    # build the CE model spectrum
    wf    = np.linspace(-0.8, 0.8, 1000)
    dt    = wf[1] - wf[0]
    gamma = ce(q_target, d_map, l_map)

    el  = gnorm(wf, sr);   el /= el.max()
    ql_ = fftconvolve(lorentz(wf, gamma), gnorm(wf, sr), mode="same") * dt
    ql  = ql_ / ql_.max() if ql_.max() > 0 else ql_

    amp, _ = nnls(
        np.column_stack([el, ql, np.ones(len(wf))]),
        np.interp(wf, ew, sn),
    )
    fit   = amp[0]*el + amp[1]*ql + amp[2]
    chi2r = np.sum(((sn - np.interp(ew, wf, fit)) / en) ** 2) / max(len(ew) - 4, 1)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # filled components
    ax.fill_between(wf, amp[2], amp[0]*el + amp[2],
                    alpha=0.22, color="#2471a3", label="elastic")
    ax.fill_between(wf, amp[2], amp[1]*ql + amp[2],
                    alpha=0.22, color="#e67e22", label="quasi-elastic")

    # data
    ax.errorbar(ew, sn, yerr=en, fmt=".", color="#333", ms=3.5,
                elinewidth=0.7, alpha=0.8, label=f"data  Q={q_target:.2f} Å⁻¹", zorder=5)

    # resolution reference
    if d_res is not None:
        gc   = d_res["good"]
        qc   = d_res["q"][gc]
        mk   = (d_res["e"] >= -0.8) & (d_res["e"] <= 0.8)
        ec   = d_res["e"][mk]
        nr   = np.where(np.abs(qc - q_target) < 0.10)[0]
        if len(nr) == 0:
            nr = np.argsort(np.abs(qc - q_target))[:4]
        rs   = np.nanmean([d_res["data"][gc[j]][mk] for j in nr], axis=0)
        rs   = np.where(np.isfinite(rs), rs, 0)
        if rs.max() > 0:
            rs /= rs.max()
        ax.plot(ec, rs, "--", color="#c0392b", lw=1.8,
                label=f"resolution ({d_res['name']})")

    # model
    ax.plot(wf, fit, "-", color="#c0392b", lw=2.2,
            label=rf"CE model  $\chi^2_r={chi2r:.2f}$")

    # HWHM annotation arrow
    ax.annotate(
        "", xy=(gamma, 0.50), xytext=(0, 0.50),
        arrowprops=dict(arrowstyle="<->", color="#e67e22", lw=1.8),
    )
    ax.text(gamma / 2, 0.56,
            f"HWHM = {gamma*1000:.0f} µeV",
            ha="center", color="#e67e22", fontsize=9.5, fontweight="bold")

    ax.axvline(0, color="#aaa", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel(r"energy transfer  $\hbar\omega$  (meV)", fontsize=12)
    ax.set_ylabel(r"$S(Q,\omega)$  normalised", fontsize=12)
    ax.set_title(rf"spectrum at $Q={q_target:.2f}$ Å$^{{-1}}$", fontsize=11)
    ax.legend(fontsize=9.5)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.05, 1.20)
    ax.grid(True, alpha=0.18)
    _despine(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_sqw_maps(d_inc, d_coh=None, save_path=None):
    """
    Side-by-side S(Q,w) false-colour maps: COH (or INC as resolution proxy)
    on the left, INC on the right.

    The progressive widening of the quasi-elastic signal with Q is the
    clearest visual signature of diffusive motion.

    Parameters
    ----------
    d_inc     : dict  — primary INC dataset
    d_coh     : dict  — COH dataset for left panel (optional)
    save_path : str   — save PDF here if given

    Returns
    -------
    fig, axes
    """
    left_data  = d_coh if d_coh is not None else d_inc
    left_label = "COH 290 K  (resolution)" if d_coh is not None else "INC 290 K (no COH)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#0d1117")

    for ax, (d, title) in zip(axes, [(left_data, left_label), (d_inc, "INC 290 K  QENS signal")]):
        g     = d["good"]
        qg    = d["q"][g]
        e     = d["e"]
        emask = (e >= -1.2) & (e <= 1.2)

        img = d["data"][np.ix_(g, emask)]
        img = np.where(np.isfinite(img) & (img > 0), img, np.nan)
        qs  = np.argsort(qg)
        ism = gaussian_filter(
            np.where(np.isfinite(img[qs]), img[qs], 0), sigma=[1.5, 0.8]
        )
        ism[ism <= 0] = np.nan

        vmin = np.nanpercentile(ism, 2)
        vmax = np.nanpercentile(ism, 99)

        im = ax.pcolormesh(
            e[emask], qg[qs], ism,
            cmap=_sqw_cmap,
            norm=LogNorm(vmin=max(vmin, 1e-6), vmax=vmax),
            shading="auto", rasterized=True,
        )

        ax.axvline(0, color="white", lw=1.0, ls="--", alpha=0.4)
        ax.set_xlabel("ω (meV)", color="white", fontsize=12)
        ax.set_title(title, color="white", fontsize=11, pad=10)
        ax.tick_params(colors="white")
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555")

        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
        cb.set_label("S(Q,ω)", color="white", fontsize=10)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    axes[0].set_ylabel("Q (Å⁻¹)", color="white", fontsize=12)
    fig.suptitle("S(Q,ω)  benzene", color="white", fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="#0d1117", dpi=150)

    return fig, axes


def plot_hwhm(q_hwhm, g_hwhm, g_err, samples, d_map, l_map, d_inc, save_path=None):
    """
    Γ(Q) vs Q² — the key diagnostic plot for diffusion mechanism.

    Shows:
        - measured HWHM points with ±2σ error bars
        - MAP CE model (red)
        - 95% posterior fan from MCMC samples (blue shading)
        - Fickian reference (dashed, for comparison)
        - resolution HWHM limit (dotted grey)

    The bending-over of Γ(Q²) at high Q is the signature of jump diffusion.
    If it were straight, you'd be seeing pure Fickian diffusion.

    Parameters
    ----------
    samples  : ndarray, shape (n, 2) — MCMC samples, columns [D, l]
    d_map    : float — MAP D
    l_map    : float — MAP l
    d_inc    : dict  — primary dataset (for resolution HWHM)
    """
    d_s = samples[:, 0]
    l_s = np.abs(samples[:, 1])

    q_fine  = np.linspace(0.35, 2.45, 400)
    q2_fine = q_fine ** 2

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # posterior fan — plot a random subset of samples to avoid overdrawing
    rng   = np.random.default_rng(0)
    idx_f = rng.choice(len(d_s), min(600, len(d_s)), replace=False)
    for i in idx_f:
        ax.plot(q2_fine, ce(q_fine, d_s[i], l_s[i]) * 1000,
                color="#2471a3", alpha=0.012, lw=0.8)

    # 95% CI band
    g_fan = np.array([ce(q_fine, d_s[i], l_s[i]) * 1000 for i in range(len(d_s))])
    ax.fill_between(
        q2_fine,
        np.percentile(g_fan, 2.5,  axis=0),
        np.percentile(g_fan, 97.5, axis=0),
        alpha=0.25, color="#2471a3",
        label=f"95% posterior  ({len(d_s)} samples)",
    )

    # MAP CE model
    ax.plot(
        q2_fine, ce(q_fine, d_map, l_map) * 1000,
        "-", color="#c0392b", lw=3.0, zorder=5,
        label=rf"CE  MAP  $D={d_map:.4f}$  $\ell={l_map:.4f}$ Å",
    )

    # Fickian reference at the MAP D value (straight line for comparison)
    ax.plot(
        q2_fine, fickian(q_fine, d_map) * 1000,
        "--", color="#555", lw=1.8, zorder=4,
        label=rf"Fickian  $D={d_map:.4f}$ (low-Q limit)",
    )

    # measured data points
    ax.errorbar(
        q_hwhm ** 2, g_hwhm * 1000, yerr=2 * g_err * 1000,
        fmt="o", color="#111", ms=7, capsize=4, elinewidth=1.8,
        label=r"data  $\pm2\sigma$", zorder=6,
    )

    # resolution limit
    res_hwhm_uev = d_inc["fwhm_res"] / 2 * 1000
    ax.axhline(
        res_hwhm_uev, color="#888", ls=":", lw=1.5,
        label=rf"resolution HWHM = {res_hwhm_uev:.0f} µeV",
    )
    ax.axhspan(0, res_hwhm_uev * 1.1, alpha=0.04, color="#888")

    ax.set_xlabel(r"$Q^2$  (Å$^{-2}$)", fontsize=13)
    ax.set_ylabel(r"$\Gamma(Q)$  (µeV)", fontsize=13)
    ax.set_title("peak width vs Q²  — bending over = jump diffusion", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.06, q2_fine[-1] + 0.10)
    ax.set_ylim(-15)
    ax.grid(True, alpha=0.20)
    _despine(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_posteriors(samples, d_map, l_map, d_inc, save_path=None):
    """
    Posterior histograms for D, l (jump length), and τ (residence time),
    with 95% CI annotations and MAP/literature reference lines.

    Parameters
    ----------
    samples  : ndarray (n, 2) — MCMC samples [D, l]
    d_map    : float — MAP D value
    l_map    : float — MAP l value
    d_inc    : dict  — for resolution info in the title
    save_path: str   — save PDF here if given

    Returns
    -------
    fig, axes
    """
    d_s   = samples[:, 0]
    l_s   = np.abs(samples[:, 1])
    tau_s = l_s ** 2 / (6 * d_s)

    tau_map = l_map ** 2 / (6 * d_map)
    d_lo,   d_hi   = np.percentile(d_s,   [2.5, 97.5])
    l_lo,   l_hi   = np.percentile(l_s,   [2.5, 97.5])
    tau_lo, tau_hi = np.percentile(tau_s, [2.5, 97.5])

    # reference values from NMR and MD — useful to sanity-check against
    refs = {
        "D":   [(0.22, "NMR 290 K", "#1565c0", "--"),
                (0.25, "MD sim",    "#e67e22",  ":")],
        "l":   [(2.42, "next-nearest C–C", "#1565c0", "--"),
                (1.40, "C–C bond",          "#9b59b6",  ":")],
        "tau": [(1.10, "MD sim",    "#1565c0", "--")],
    }

    params = [
        (d_s,   d_map,   d_lo,   d_hi,   "D  (Å²/ps)",   "#c0392b",  refs["D"]),
        (l_s,   l_map,   l_lo,   l_hi,   "ℓ  (Å)",        "#1e8449",  refs["l"]),
        (tau_s, tau_map, tau_lo, tau_hi, "τ  (ps)",        "#e67e22",  refs["tau"]),
    ]

    try:
        import emcee
        sampler_label = "emcee"
    except ImportError:
        sampler_label = "MH fallback"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        f"Bayesian posteriors  |  CE model  |  {len(d_s)} samples  "
        f"({sampler_label})\n"
        f"res FWHM = {d_inc['fwhm_res']*1000:.0f} µeV  [{d_inc['res_source']}]",
        fontsize=11,
    )

    for ax, (arr, map_val, lo, hi, xlabel, col, reflines) in zip(axes, params):
        med = np.median(arr)
        cnt, _, _ = ax.hist(
            arr, bins=80, density=True,
            color=col, alpha=0.80, edgecolor="white", lw=0.25,
        )
        pk = cnt.max()

        ax.axvspan(lo, hi, alpha=0.18, color=col)
        ax.axvline(med,     color="#111", lw=2.5, label=f"median = {med:.4f}")
        ax.axvline(map_val, color=col,   lw=1.8, ls="--", label=f"MAP = {map_val:.4f}")

        for val, label_str, lc, ls in reflines:
            ax.axvline(val, color=lc, lw=2.0, ls=ls, label=f"{label_str} = {val}")

        # CI bracket annotation
        ax.annotate(
            "", xy=(hi, pk * 1.10), xytext=(lo, pk * 1.10),
            arrowprops=dict(arrowstyle="<->", color=col, lw=2.0),
        )
        ax.text(
            (lo + hi) / 2, pk * 1.17,
            f"95% CI  [{lo:.4f}, {hi:.4f}]",
            ha="center", fontsize=8.5, color=col, fontweight="bold",
        )

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("density", fontsize=11)
        ax.set_title(xlabel, fontsize=12, color=col, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.92)
        ax.set_ylim(0, pk * 1.32)
        ax.grid(True, alpha=0.20)
        _despine(ax)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig, axes
