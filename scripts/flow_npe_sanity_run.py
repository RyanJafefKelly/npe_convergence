#!/usr/bin/env python
"""Flow-NPE sanity check for Phase 2.4d.

Mirrors run_gnk.py at (n_obs=1000, n_sims=1e6, seed=1) with a CLI-configurable
learning rate (primary variable tested) to see whether flow training is
optimisation-limited.
"""
from __future__ import annotations

import argparse
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
from flowjax.distributions import Normal  # type: ignore
from flowjax.flows import coupling_flow  # type: ignore
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
from jax.scipy.special import expit, logit

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import (  # noqa: E402
    gnk, get_summaries_batches, ss_octile,
)
from npe_convergence.metrics import (  # noqa: E402
    kullback_leibler, median_heuristic, unbiased_mmd,
)

numpyro.set_host_device_count(4)
TRUE_PARAMS = jnp.array([3.0, 1.0, 2.0, 0.5])


def _nuts_cache(n_obs: int, seed: int) -> Path:
    p1 = REPO_ROOT / "res" / "gnk" / f"nuts_cache_v2_flow_n_obs_{n_obs}_seed_{seed}.pkl"
    if p1.exists():
        return p1
    return REPO_ROOT / "res" / "gnk" / f"nuts_cache_v2_n_obs_{n_obs}_seed_{seed}.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--n-sims", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--nn-depth", type=int, default=2)
    parser.add_argument("--patience", type=int, default=200)
    args = parser.parse_args()

    dirname = (
        REPO_ROOT / "res" / "gnk" / "sanity_flow"
        / f"lr{args.lr:.0e}_depth{args.nn_depth}_seed_{args.seed}"
    )
    dirname.mkdir(parents=True, exist_ok=True)
    print(f"[START] flow sanity lr={args.lr}, nn_depth={args.nn_depth}, seed={args.seed}")

    t0 = time.time()
    key = random.key(args.seed)
    key, sub = random.split(key)
    z = random.normal(sub, shape=(args.n_obs,))
    x_obs = jnp.atleast_2d(gnk(z, *TRUE_PARAMS))
    x_obs = jnp.squeeze(ss_octile(x_obs))

    p = _nuts_cache(args.n_obs, args.seed)
    with open(p, "rb") as f:
        true_post = jnp.asarray(pkl.load(f))
    print(f"[{time.time()-t0:.0f}s] NUTS loaded ({true_post.shape})")

    tol = 1e-6
    key, sub = random.split(key)
    thetas_bounded = dist.Uniform(0 + tol, 10 - tol).sample(sub, (args.n_sims, 4))
    thetas_unbounded = logit(thetas_bounded / 10)
    A, B, g, k = thetas_bounded.T

    key, sub = random.split(key)
    x_sims = get_summaries_batches(sub, A, B, g, k, args.n_obs, args.n_sims, batch_size=1000)
    sim_summ_data = x_sims.T
    sim_mean = sim_summ_data.mean(axis=0)
    sim_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_mean) / sim_std
    x_obs_std = (x_obs - sim_mean) / sim_std
    thetas_mean = thetas_unbounded.mean(axis=0)
    thetas_std = thetas_unbounded.std(axis=0)
    thetas = (thetas_unbounded - thetas_mean) / thetas_std
    print(f"[{time.time()-t0:.0f}s] Data prepared.")

    key, sub = random.split(key)
    flow = coupling_flow(
        key=sub, base_dist=Normal(jnp.zeros(4)),
        transformer=RationalQuadraticSpline(knots=10, interval=5),
        cond_dim=7, nn_depth=args.nn_depth,
    )
    key, sub = random.split(key)
    flow, losses = fit_to_data(
        key=sub, dist=flow, x=thetas, condition=sim_summ_data,
        learning_rate=args.lr, max_epochs=2000, max_patience=args.patience,
        batch_size=256,
    )
    print(f"[{time.time()-t0:.0f}s] Training done. best_val={min(losses['val']):.4f} "
          f"epochs={len(losses['train'])}")

    plt.plot(losses["train"], label="train")
    plt.plot(losses["val"], label="val")
    plt.legend()
    plt.savefig(dirname / "losses.pdf")
    plt.close()

    key, sub = random.split(key)
    post_std = flow.sample(sub, sample_shape=(10_000,), condition=x_obs_std)
    post_unbounded = post_std * thetas_std + thetas_mean
    posterior_samples = expit(post_unbounded) * 10

    n_metric = 2000
    key, sub = random.split(key)
    idx_npe = random.permutation(sub, posterior_samples.shape[0])[:n_metric]
    key, sub = random.split(key)
    idx_true = random.permutation(sub, true_post.shape[0])[:n_metric]
    ps_thin = posterior_samples[idx_npe]
    ts_thin = true_post[idx_true]
    kl = kullback_leibler(ts_thin, ps_thin)
    print(f"[{time.time()-t0:.0f}s] KL={kl:.4f}")

    sigma_ratios = np.asarray(posterior_samples).std(axis=0) / np.asarray(true_post).std(axis=0)

    with open(dirname / "posterior_samples.pkl", "wb") as f:
        pkl.dump(posterior_samples, f)
    with open(dirname / "kl.txt", "w") as f:
        f.write(str(float(kl)))
    with open(dirname / "summary.txt", "w") as f:
        f.write(f"lr={args.lr}\nnn_depth={args.nn_depth}\nseed={args.seed}\n")
        f.write(f"kl={float(kl):.6f}\n")
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
