"""
preprocessing.py

Two things happen here before any real fitting:
1. We find the elastic peak centre and shift the energy axis to zero.
2. We figure out the instrument resolution.

These are separate from the physics fitting because they're the same
regardless of which diffusion model you use.

Quick note on resolution: the sigma you get from fitting the elastic
peak of an incoherent (INC) file at 290 K is NOT the instrument
resolution — it's already broadened by the quasielastic signal. Use
the 260 K frozen sample or the coherent (COH) file if you can.
assign_resolution() handles the priority logic for you.
"""

import numpy as np
from scipy.optimize import curve_fit


def fit_elastic_peak(d):
    """
    Fit a Gaussian to the elastic peak (averaged over the lowest-Q detectors)
    to find where zero energy transfer actually lands in the data.

    Updates the dict in place:
        d["e0"]      — elastic peak position (meV)
        d["sig_raw"] — raw Gaussian sigma (meV)
        d["e"]       — energy axis shifted so the elastic line is at 0

    Returns (e0, sig_raw) if you want them directly.

    The fallback values (pk position + 0.043 meV sigma) kick in if curve_fit
    fails, which occasionally happens with very noisy low-Q detectors. The
    0.043 sigma corresponds to ~100 µeV FWHM, which is typical for Pelican.
    """
    good  = d["good"]
    e     = d["e_raw"]

    # use the lowest-Q detectors where the elastic peak dominates
    n_low = max(3, len(good) // 7)
    avg   = np.nanmean([d["data"][i] for i in good[:n_low]], axis=0)
    avg   = np.where(np.isfinite(avg), avg, 0)

    def gauss(x, a, mu, sigma, bg):
        return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + bg

    pk = np.argmax(avg)

    try:
        popt, _ = curve_fit(
            gauss, e, avg,
            p0=[avg[pk], e[pk], 0.05, avg.min()],
            bounds=(
                [-np.inf, e[0],  1e-4, -np.inf],
                [ np.inf, e[-1], 2.0,   np.inf],
            ),
            maxfev=8000,
        )
        e0      = float(popt[1])
        sig_raw = abs(float(popt[2]))
    except Exception as exc:
        print(f"  elastic peak fit failed for {d['name']}: {exc}")
        e0      = float(e[pk])
        sig_raw = 0.043   # fallback: ~100 µeV FWHM

    d["e0"]      = e0
    d["sig_raw"] = sig_raw
    d["e"]       = e - e0

    return e0, sig_raw


def assign_resolution(dataset):
    """
    Go through every loaded file and assign an instrument resolution sigma.

    The priority is:
        1. 260 K frozen benzene INC — cleanest resolution measurement
        2. COH file at the same Ei — coherent elastic peak is narrow
        3. Raw INC width — last resort, will be too wide due to QENS broadening

    Adds these keys to each dataset entry:
        sigma_res  — resolution sigma in meV
        fwhm_res   — FWHM = 2.355 * sigma_res
        res_source — string describing which method was used

    Nothing is returned — modifies the dicts in place.
    """

    # check if we have the 260 K frozen reference
    res_260k = {}
    key_260  = "benzene_260_360_inc.nxspe"

    if key_260 in dataset:
        d260 = dataset[key_260]
        res_260k[d260["ei"]] = d260["sig_raw"]
        print(
            f"  found 260 K reference: Ei={d260['ei']:.2f} meV, "
            f"FWHM={d260['sig_raw'] * 2355:.1f} µeV"
        )
    else:
        print("  no 260 K reference — will fall back to COH or raw INC")

    # collect resolutions from COH files
    coh_sigma = {
        d["ei"]: d["sig_raw"]
        for d in dataset.values()
        if d["kind"] == "coh"
    }

    # now assign to each file
    for fname, d in dataset.items():
        ei = d["ei"]

        if ei in res_260k:
            d["sigma_res"] = res_260k[ei]
            d["res_source"] = "260 K frozen"

        elif ei in coh_sigma:
            d["sigma_res"] = coh_sigma[ei]
            d["res_source"] = "COH file"

        else:
            d["sigma_res"] = d["sig_raw"]
            d["res_source"] = "raw INC (may be inflated)"
            print(f"  WARNING: no better resolution for {fname} — using raw INC peak")

        d["fwhm_res"] = 2.355 * d["sigma_res"]

    # quick summary so you can see what happened
    print("\n  resolution summary:")
    for fname, d in dataset.items():
        print(
            f"    {fname:<44}  "
            f"{d['fwhm_res'] * 1000:6.1f} µeV  [{d['res_source']}]"
        )
