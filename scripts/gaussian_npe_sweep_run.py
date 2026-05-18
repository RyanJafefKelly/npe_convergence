#!/usr/bin/env python
"""Single parametrised Gaussian-NPE training run for Phase 2.4c.

Replicates `run_gnk_gaussian.py` with CLI-configurable hidden_dims and LR,
saving outputs to a sweep-specific subdir: res/gnk/sweep_gnpe/h{H}_lr{LR}_seed_{S}/.
Only n_obs=1000, n_sims=1_000_000 supported (Stage 1 scope).
"""
from __future__ import annotations

import argparse
import os
import pickle as pkl
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import expit, logit

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import (  # noqa: E402
    gnk, get_summaries_batches, ss_octile,
)
from npe_convergence.methods.gaussian_npe import (  # noqa: E402
    ConditionalGaussianNPE, TrainConfig, fit, sample,
)
from npe_convergence.metrics import (  # noqa: E402
    kullback_leibler, median_heuristic, unbiased_mmd,
)

numpyro.set_host_device_count(4)

TRUE_PARAMS = jnp.array([3.0, 1.0, 2.0, 0.5])
PARAM_NAMES = ["A", "B", "g", "k"]


def _nuts_cache_path(n_obs: int, seed: int) -> Path:
    return REPO_ROOT / "res" / "gnk" / f"nuts_cache_v2_n_obs_{n_obs}_seed_{seed}.pkl"


def load_nuts_cache(n_obs: int, seed: int) -> np.ndarray:
    p = _nuts_cache_path(n_obs, seed)
    if not p.exists():
        # try flow-suffixed cache
        p2 = REPO_ROOT / "res" / "gnk" / f"nuts_cache_v2_flow_n_obs_{n_obs}_seed_{seed}.pkl"
        if p2.exists():
            p = p2
        else:
            raise FileNotFoundError(f"No NUTS cache for n_obs={n_obs}, seed={seed}")
    with open(p, "rb") as f:
        return np.asarray(pkl.load(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--n-sims", type=int, default=1_000_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=2000)
    args = parser.parse_args()

    hidden_dims = (args.hidden_dim, args.hidden_dim)
    dirname = (
        REPO_ROOT / "res" / "gnk" / "sweep_gnpe"
        / f"h{args.hidden_dim}_lr{args.lr:.0e}_seed_{args.seed}"
    )
    dirname.mkdir(parents=True, exist_ok=True)
    print(f"[START] config: hidden={hidden_dims}, lr={args.lr}, seed={args.seed}, "
          f"dirname={dirname}")

    t0 = time.time()
    key = random.key(args.seed)

    # -- Observed data --
    key, sub = random.split(key)
    z = random.normal(sub, shape=(args.n_obs,))
    x_obs = jnp.atleast_2d(gnk(z, *TRUE_PARAMS))
    x_obs = jnp.squeeze(ss_octile(x_obs))
    print(f"[{time.time()-t0:.0f}s] x_obs computed.")

    # -- NUTS reference --
    true_post = jnp.asarray(load_nuts_cache(args.n_obs, args.seed))
    print(f"[{time.time()-t0:.0f}s] NUTS cache loaded: {true_post.shape}")

    # -- Simulate training data --
    tol = 1e-6
    key, sub = random.split(key)
    thetas_bounded = dist.Uniform(0 + tol, 10 - tol).sample(sub, (args.n_sims, 4))
    thetas_unbounded = logit(thetas_bounded / 10)
    A, B, g, k = thetas_bounded.T

    key, sub = random.split(key)
    x_sims = get_summaries_batches(sub, A, B, g, k, args.n_obs, args.n_sims, batch_size=1000)
    sim_summ_data = x_sims.T
    print(f"[{time.time()-t0:.0f}s] sim_summ_data shape: {sim_summ_data.shape}")

    # Standardise
    sim_mean = sim_summ_data.mean(axis=0)
    sim_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_mean) / sim_std
    x_obs_std = (x_obs - sim_mean) / sim_std

    thetas_mean = thetas_unbounded.mean(axis=0)
    thetas_std = thetas_unbounded.std(axis=0)
    thetas = (thetas_unbounded - thetas_mean) / thetas_std
    print(f"[{time.time()-t0:.0f}s] Standardisation done.")

    # -- Train --
    key, sub = random.split(key)
    model = ConditionalGaussianNPE(
        d_summary=7, d_theta=4, hidden_dims=hidden_dims, key=sub,
    )
    key, sub = random.split(key)
    config = TrainConfig(
        lr=args.lr, batch_size=256, max_epochs=args.max_epochs, patience=args.patience,
    )
    model, losses = fit(model, thetas, sim_summ_data, key=sub, config=config)
    print(f"[{time.time()-t0:.0f}s] Training done. epochs_run={len(losses['train'])}, "
          f"best_val={min(losses['val']):.4f}")

    plt.plot(losses["train"], label="train")
    plt.plot(losses["val"], label="val")
    plt.legend()
    plt.savefig(dirname / "losses.pdf")
    plt.close()

    # -- Sample posterior --
    key, sub = random.split(key)
    post_std = sample(model, x_obs_std, 10_000, key=sub)
    post_unbounded = post_std * thetas_std + thetas_mean
    posterior_samples = expit(post_unbounded) * 10
    print(f"[{time.time()-t0:.0f}s] Posterior sampling done.")

    # -- Metrics --
    n_metric = 2000
    key, sub = random.split(key)
    idx_npe = random.permutation(sub, posterior_samples.shape[0])[:n_metric]
    key, sub = random.split(key)
    idx_true = random.permutation(sub, true_post.shape[0])[:n_metric]
    ps_thin = posterior_samples[idx_npe]
    ts_thin = true_post[idx_true]

    kl = kullback_leibler(ts_thin, ps_thin)
    lengthscale = median_heuristic(jnp.vstack([ts_thin, ps_thin]))
    mmd = unbiased_mmd(ts_thin, ps_thin, lengthscale)
    print(f"[{time.time()-t0:.0f}s] KL={kl:.4f} MMD={mmd:.6f}")

    # Sigma-ratios in theta space
    sigma_ratios = np.asarray(posterior_samples).std(axis=0) / np.asarray(true_post).std(axis=0)

    with open(dirname / "posterior_samples.pkl", "wb") as f:
        pkl.dump(posterior_samples, f)
    with open(dirname / "kl.txt", "w") as f:
        f.write(str(float(kl)))
    with open(dirname / "mmd.txt", "w") as f:
        f.write(str(float(mmd)))
    with open(dirname / "summary.txt", "w") as f:
        f.write(f"hidden_dim={args.hidden_dim}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"kl={float(kl):.6f}\n")
        f.write(f"mmd={float(mmd):.6f}\n")
        f.write(f"sigma_ratio_A={sigma_ratios[0]:.4f}\n")
        f.write(f"sigma_ratio_B={sigma_ratios[1]:.4f}\n")
        f.write(f"sigma_ratio_g={sigma_ratios[2]:.4f}\n")
        f.write(f"sigma_ratio_k={sigma_ratios[3]:.4f}\n")
        f.write(f"epochs_run={len(losses['train'])}\n")
        f.write(f"best_val_loss={min(losses['val']):.6f}\n")
        f.write(f"wall_seconds={time.time()-t0:.1f}\n")
    print(f"[{time.time()-t0:.0f}s] DONE. Wrote {dirname}")


if __name__ == "__main__":
    main()
