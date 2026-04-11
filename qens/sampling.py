"""
MCMC sampling for CE model.
"""

from __future__ import annotations
import numpy as np
from .fitting import log_posterior
from .config import Config


def gelman_rubin(chains):
    """
    Gelman-Rubin R-hat for convergence.
    """
    m = len(chains)
    n = min(len(c) for c in chains)
    chains = [np.asarray(c[:n]) for c in chains]
    w = np.mean([c.var(ddof=1) for c in chains])
    if w == 0.0:
        return float("nan")
    chain_means = np.array([c.mean() for c in chains])
    b = n * chain_means.var(ddof=1)
    var_hat = (1 - 1/n) * w + b / n
    return float(np.sqrt(var_hat / w))


def _run_emcee(data_bins, sr, d_map, l_map, cfg):
    import emcee

    def log_prob(params):
        return log_posterior(params[0], abs(params[1]), data_bins, sr)

    ndim = 2
    rng = np.random.default_rng(cfg.random_seed)
    p0 = [np.array([d_map, l_map]) * (1 + rng.normal(0, 0.05, ndim)) for _ in range(cfg.n_walkers)]
    p0 = [np.clip(p, [1e-3, 0.6], [2.9, 5.9]) for p in p0]


    sampler = emcee.EnsembleSampler(cfg.n_walkers, ndim, log_prob)
    total_steps = cfg.n_warmup + cfg.n_keep
    print(f"  running emcee: {cfg.n_walkers} walkers × {total_steps} steps (thin={cfg.thin})")
    sampler.run_mcmc(p0, total_steps, progress=True)

    samples = sampler.get_chain(discard=cfg.n_warmup, thin=cfg.thin, flat=True)
    acc = float(np.mean(sampler.acceptance_fraction))
    print(f"  acceptance fraction: {acc:.3f}")


    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"  autocorrelation time: D={tau[0]:.1f}  l={tau[1]:.1f} steps")
    except Exception:
        print("  autocorrelation estimate failed")
    return samples


def _run_mh(data_bins, sr, d_map, l_map, cfg):
    rng_global = np.random.default_rng(cfg.random_seed)

    def _chain(start, n_steps, step_d, step_log_l, seed):
        rng_c = np.random.default_rng(seed)
        d_cur, l_cur = float(start[0]), float(start[1])
        cur_lp = log_posterior(d_cur, l_cur, data_bins, sr)
        samples = [(d_cur, l_cur)]
        n_acc = 0
        for _ in range(n_steps):
            d_new = d_cur + rng_c.normal(0, step_d)
            log_l_new = np.log(l_cur) + rng_c.normal(0, step_log_l)
            l_new = np.exp(log_l_new)
            new_lp = log_posterior(d_new, l_new, data_bins, sr)
            log_accept = new_lp - cur_lp + (log_l_new - np.log(l_cur))
            if np.log(rng_c.random() + 1e-300) < log_accept:
                d_cur, l_cur = d_new, l_new
                cur_lp = new_lp
                n_acc += 1
            samples.append((d_cur, l_cur))
        return np.array(samples), n_acc / n_steps

    step_d = d_map * 0.1
    step_log_l = 0.1
    n_total = cfg.n_warmup + cfg.n_keep
    chains = []
    print(f"  running 4 MH chains × {n_total} steps (thin={cfg.thin})")
    for cid in range(4):
        start = np.array([d_map, l_map]) * (1 + rng_global.normal(0, 0.05, 2))
        start = np.clip(start, [1e-3, 0.6], [2.9, 5.9])
        chain, acc = _chain(start, n_total, step_d, step_log_l, seed=cfg.random_seed+cid)
        chains.append(chain[cfg.n_warmup::cfg.thin])
        print(f"  chain {cid+1}: acceptance={acc:.3f} kept={len(chains[-1])}")

    rhat_d = gelman_rubin([c[:,0] for c in chains])
    rhat_l = gelman_rubin([c[:,1] for c in chains])
    print(f"  R-hat: D={rhat_d:.4f}  l={rhat_l:.4f}")
    return np.vstack(chains)



def run_mcmc(data_bins, sr, d_map, l_map, cfg: Config | None = None):
    if cfg is None:
        cfg = Config()
    try:
        import emcee
        use_emcee = True
    except ImportError:
        use_emcee = False
        print("  emcee not found — using Metropolis-Hastings fallback")

    if use_emcee:
        samples = _run_emcee(data_bins, sr, d_map, l_map, cfg)
    else:
        samples = _run_mh(data_bins, sr, d_map, l_map, cfg)

    samples[:,1] = np.abs(samples[:,1])
    print(f"  total posterior samples: {len(samples)}")
    return samples



def summarise(arr, label):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    med = float(np.median(arr))
    print(f"  {label:<16}  median={med:.5f}   95% CI=[{lo:.5f}, {hi:.5f}]")
    return med, float(lo), float(hi)
