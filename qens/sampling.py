"""
sampling.py

MCMC posterior sampling for the CE model.

We prefer emcee's ensemble sampler when it's installed — it handles
correlated parameters well and the acceptance fraction is easy to
monitor. If emcee isn't available, there's a fallback Metropolis-Hastings
implementation that runs 4 independent chains.

The Gelman-Rubin R-hat is the main convergence diagnostic. Values below
1.01 mean the chains have effectively converged; above 1.05 means you
probably need to run longer.

Usage
-----
    from qens.sampling import run_mcmc, summarise

    samples = run_mcmc(data_bins, sigma_res, d_map, l_map, cfg)

    # samples[:,0] is D, samples[:,1] is l
    d_med, d_lo, d_hi = summarise(samples[:,0], "D (Å²/ps)")
"""

import numpy as np

from .fitting import log_posterior
from .config import Config


def gelman_rubin(chains):
    """
    Compute the Gelman-Rubin R-hat for a single parameter.

    Takes a list of 1D arrays (one per chain) and returns R-hat.
    Values close to 1.0 mean the chains have converged.

    This is the classic Gelman & Rubin (1992) formula — nothing fancy.
    """
    m = len(chains)
    n = len(chains[0])

    # within-chain variance
    w = np.mean([c.var(ddof=1) for c in chains])

    # between-chain variance
    b = n * np.array([c.mean() for c in chains]).var(ddof=1)

    if w == 0:
        return float("nan")

    var_hat = (1 - 1 / n) * w + b / n
    return float(np.sqrt(var_hat / w))


def _run_emcee(data_bins, sr, d_map, l_map, cfg):
    """Use emcee's ensemble sampler."""
    import emcee

    def log_prob(params):
        return log_posterior(params[0], abs(params[1]), data_bins, sr)

    ndim = 2
    rng  = np.random.default_rng(cfg.random_seed)

    # initialise walkers in a small ball around the MAP
    p0 = [
        np.array([d_map, l_map]) + rng.normal(0, 0.01, ndim)
        for _ in range(cfg.n_walkers)
    ]

    sampler = emcee.EnsembleSampler(cfg.n_walkers, ndim, log_prob)

    print(f"  running emcee: {cfg.n_walkers} walkers × {cfg.n_warmup + cfg.n_keep} steps")
    sampler.run_mcmc(p0, cfg.n_warmup + cfg.n_keep, progress=True)

    samples = sampler.get_chain(discard=cfg.n_warmup, thin=cfg.thin, flat=True)

    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"  acceptance fraction: {acc:.3f}  (target: 0.2–0.5)")

    # try to get autocorrelation estimate — this can fail if the chain
    # isn't long enough, which is fine
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"  autocorrelation time: D={tau[0]:.1f}  l={tau[1]:.1f} steps")
    except Exception:
        print("  autocorrelation estimate didn't converge — consider more steps")

    return samples


def _run_mh(data_bins, sr, d_map, l_map, cfg):
    """
    Metropolis-Hastings fallback — runs 4 independent chains.

    The step sizes are set to 5% of the MAP values, which usually gives
    decent acceptance rates for this problem. If you're getting very low
    acceptance, try narrowing the priors or starting closer to the MAP.
    """
    def _chain(start, n_steps, step, seed):
        rng_c  = np.random.default_rng(seed)
        d_cur, l_cur = start
        cur_lp = log_posterior(d_cur, l_cur, data_bins, sr)
        samples = [(d_cur, l_cur)]
        n_acc   = 0

        for _ in range(n_steps):
            d_new = d_cur + rng_c.normal(0, step[0])
            l_new = l_cur + rng_c.normal(0, step[1])
            new_lp = log_posterior(d_new, abs(l_new), data_bins, sr)

            if np.log(rng_c.random()) < new_lp - cur_lp:
                d_cur, l_cur = d_new, l_new
                cur_lp = new_lp
                n_acc += 1

            samples.append((d_cur, l_cur))

        return np.array(samples), n_acc / n_steps

    step    = np.array([d_map * 0.05, l_map * 0.05])
    n_total = cfg.n_warmup + cfg.n_keep
    rng     = np.random.default_rng(cfg.random_seed)

    chains = []
    print(f"  running 4 MH chains × {n_total} steps (thin={cfg.thin})")

    for cid in range(4):
        start = [
            d_map + rng.normal(0, step[0]),
            l_map + rng.normal(0, step[1]),
        ]
        chain, acc = _chain(start, n_total, step, seed=cfg.random_seed + cid)
        chains.append(chain[cfg.n_warmup :: cfg.thin])
        print(f"  chain {cid + 1}: acceptance={acc:.2f}  kept={len(chains[-1])}")

    # convergence check
    rhat_d = gelman_rubin([c[:, 0] for c in chains])
    rhat_l = gelman_rubin([c[:, 1] for c in chains])
    print(f"  R-hat: D={rhat_d:.4f}  l={rhat_l:.4f}  (< 1.01 is good)")

    return np.vstack(chains)


def run_mcmc(data_bins, sr, d_map, l_map, cfg=None):
    """
    Sample the CE posterior. Uses emcee if available, otherwise
    falls back to the hand-written Metropolis-Hastings sampler.

    Parameters
    ----------
    data_bins : list  — from build_data_bins
    sr        : float — instrument resolution sigma (meV)
    d_map     : float — MAP diffusion coefficient, used to initialise walkers
    l_map     : float — MAP jump length, used to initialise walkers
    cfg       : Config

    Returns
    -------
    samples : ndarray, shape (n_samples, 2)
        Column 0 is D (Å²/ps), column 1 is l (Å).
    """
    if cfg is None:
        cfg = Config()

    try:
        import emcee
        use_emcee = True
    except ImportError:
        use_emcee = False
        print("  emcee not found — using Metropolis-Hastings fallback")
        print("  (pip install emcee for better sampling)")

    if use_emcee:
        samples = _run_emcee(data_bins, sr, d_map, l_map, cfg)
    else:
        samples = _run_mh(data_bins, sr, d_map, l_map, cfg)

    print(f"  total samples: {len(samples)}")
    return samples


def summarise(arr, label):
    """
    Print a one-line summary and return (median, lower_95, upper_95).

    Example
    -------
        d_med, d_lo, d_hi = summarise(samples[:, 0], "D (Å²/ps)")
        # prints: D (Å²/ps)   median=0.29841   95% CI=[0.27033, 0.32819]
    """
    lo, hi = np.percentile(arr, [2.5, 97.5])
    med    = float(np.median(arr))
    print(f"  {label:<14}  median={med:.5f}   95% CI=[{lo:.5f}, {hi:.5f}]")
    return med, float(lo), float(hi)
