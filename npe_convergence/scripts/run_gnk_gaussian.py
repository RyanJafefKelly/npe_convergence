"""Run g-and-k model with conditional Gaussian NPE.

Mirrors run_gnk.py but replaces the normalising flow with a conditional
Gaussian posterior approximation.  NUTS results are cached so that
repeated runs across different N values skip the expensive MCMC step.

Usage:
    python npe_convergence/scripts/run_gnk_gaussian.py --seed=1 --n_obs=1000 --n_sims=31623
"""

import argparse
import os
import pickle as pkl

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax.scipy.special import expit, logit
from numpyro.diagnostics import hpdi  # type: ignore

from npe_convergence.examples.gnk import gnk, get_summaries_batches, run_nuts, ss_octile
from npe_convergence.methods.gaussian_npe import (
    ConditionalGaussianNPE,
    TrainConfig,
    fit,
    sample,
)
from npe_convergence.metrics import kullback_leibler, median_heuristic, unbiased_mmd


# ---------------------------------------------------------------------------
# NUTS caching
# ---------------------------------------------------------------------------


def _nuts_cache_path(n_obs: int, seed: int) -> str:
    return f"res/gnk/nuts_cache_n_obs_{n_obs}_seed_{seed}.pkl"


def get_nuts_posterior(
    seed: int,
    x_obs: jnp.ndarray,
    n_obs: int,
    num_samples: int = 10_000,
    num_warmup: int = 10_000,
) -> jnp.ndarray:
    """Load cached NUTS samples or run + cache if not found.

    Returns array of shape (num_samples, 4) with columns [A, B, g, k].
    """
    cache_path = _nuts_cache_path(n_obs, seed)
    if os.path.exists(cache_path):
        print(f"Loading cached NUTS posterior from {cache_path}")
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    print("Running NUTS (will cache for future runs)...")
    mcmc = run_nuts(
        seed=1, obs=x_obs, n_obs=n_obs, num_samples=num_samples, num_warmup=num_warmup
    )
    mcmc.print_summary()
    samples_dict = mcmc.get_samples()

    # Stack into (num_samples, 4) array
    param_names = ["A", "B", "g", "k"]
    samples = jnp.column_stack([samples_dict[p] for p in param_names])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pkl.dump(samples, f)
    print(f"Cached NUTS posterior to {cache_path}")
    return samples


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_gnk_gaussian(*args, **kwargs):
    try:
        seed, n_obs, n_sims = args
    except ValueError:
        args = args[0]
        seed, n_obs, n_sims = args.seed, args.n_obs, args.n_sims

    dirname = f"res/gnk/gaussian_npe_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}/"
    os.makedirs(dirname, exist_ok=True)

    # -- Ground truth data --------------------------------------------------
    a, b, g, k = 3.0, 1.0, 2.0, 0.5
    true_params = jnp.array([a, b, g, k])
    key = random.key(seed)

    key, subkey = random.split(key)
    z = random.normal(subkey, shape=(n_obs,))
    x_obs = gnk(z, *true_params)
    x_obs = jnp.atleast_2d(x_obs)
    x_obs = jnp.squeeze(ss_octile(x_obs))
    print("x_obs:", x_obs)

    # -- Exact posterior (NUTS, cached) -------------------------------------
    true_posterior_samples = get_nuts_posterior(seed, x_obs, n_obs)
    param_names = ["A", "B", "g", "k"]
    print(">>> Plotting true posterior...")
    for ii, name in enumerate(param_names):
        plt.hist(true_posterior_samples[:, ii], bins=50, label=name)
        plt.axvline(true_params[ii], color="red")
        plt.legend()
        plt.savefig(f"{dirname}true_samples_{name}.pdf")
        plt.clf()

    # -- Prior simulation ---------------------------------------------------
    key, subkey = random.split(key)
    tol = 1e-6
    thetas_bounded = dist.Uniform(0 + tol, 10 - tol).sample(subkey, (n_sims, 4))
    thetas_unbounded = logit(thetas_bounded / 10)

    A_sim, B_sim, g_sim, k_sim = thetas_bounded.T
    key, subkey = random.split(key)
    batch_size = min(1000, n_sims)
    print(">>> Simulating prior predictive...")
    x_sims = get_summaries_batches(
        subkey, A_sim, B_sim, g_sim, k_sim, n_obs, n_sims, batch_size=batch_size
    )
    print(">>> Simulations done, standardising...")

    # -- Standardise --------------------------------------------------------
    thetas_mean = thetas_unbounded.mean(axis=0)
    thetas_std = thetas_unbounded.std(axis=0)
    thetas = (thetas_unbounded - thetas_mean) / thetas_std

    sim_summ_data = x_sims.T
    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std
    x_obs_std = (x_obs - sim_summ_data_mean) / sim_summ_data_std

    # -- Train conditional Gaussian -----------------------------------------
    theta_dims = 4
    summary_dims = 7

    key, subkey = random.split(key)
    model = ConditionalGaussianNPE(
        d_summary=summary_dims,
        d_theta=theta_dims,
        hidden_dims=(128, 128),
        key=subkey,
    )

    key, subkey = random.split(key)
    config = TrainConfig(
        lr=5e-4,
        batch_size=256,
        max_epochs=2000,
        patience=10,
    )
    model, losses = fit(
        model,
        thetas,
        sim_summ_data,
        key=subkey,
        config=config,
    )
    print(">>> Training done, sampling posterior...")

    plt.plot(losses["train"], label="train")
    plt.plot(losses["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.legend()
    plt.savefig(f"{dirname}losses.pdf")
    plt.clf()

    mu_hat, L_hat = model(x_obs_std)
    print("Learned mu (std space):", mu_hat)
    print("Learned L diagonal:", jnp.diag(L_hat))

    # -- Sample posterior + invert standardisation --------------------------
    num_posterior_samples = 10_000
    key, subkey = random.split(key)
    posterior_std = sample(model, x_obs_std, num_posterior_samples, key=subkey)
    posterior_unbounded = posterior_std * thetas_std + thetas_mean
    posterior_samples = expit(posterior_unbounded) * 10
    print(">>> Sampling done, computing metrics...")

    for ii, name in enumerate(param_names):
        _, bins, _ = plt.hist(
            posterior_samples[:, ii], bins=50, alpha=0.8, label="Gaussian NPE"
        )
        plt.hist(true_posterior_samples[:, ii], bins=bins, alpha=0.8, label="NUTS")
        plt.axvline(true_params[ii], color="black")
        plt.legend()
        plt.savefig(f"{dirname}posterior_samples_{name}.pdf")
        plt.clf()

    # -- Metrics (thin to avoid OOM on pairwise distances) ------------------
    n_metric = 400
    key, subkey = random.split(key)
    idx_npe = random.permutation(subkey, posterior_samples.shape[0])[:n_metric]
    key, subkey = random.split(key)
    idx_true = random.permutation(subkey, true_posterior_samples.shape[0])[:n_metric]

    ps_thin = posterior_samples[idx_npe]
    ts_thin = true_posterior_samples[idx_true]

    kl = kullback_leibler(ts_thin, ps_thin)
    lengthscale = median_heuristic(jnp.vstack([ts_thin, ps_thin]))
    mmd = unbiased_mmd(ts_thin, ps_thin, lengthscale)
    print(f"KL: {kl:.4f}, MMD: {mmd:.6f}")

    with open(f"{dirname}posterior_samples.pkl", "wb") as f:
        pkl.dump(posterior_samples, f)
    with open(f"{dirname}true_posterior_samples.pkl", "wb") as f:
        pkl.dump(true_posterior_samples, f)
    with open(f"{dirname}kl.txt", "w") as f:
        f.write(str(kl))
    with open(f"{dirname}mmd.txt", "w") as f:
        f.write(str(mmd))

    # -- Coverage analysis --------------------------------------------------
    num_coverage_samples = 100
    coverage_levels = [0.8, 0.9, 0.95]
    coverage_counts = np.zeros((theta_dims, len(coverage_levels)))
    all_biases = []

    for i in range(num_coverage_samples):
        key, subkey = random.split(key)
        z_cov = random.normal(subkey, shape=(n_obs,))
        x_obs_cov = get_summaries_batches(
            subkey,
            jnp.array([a]),
            jnp.array([b]),
            jnp.array([g]),
            jnp.array([k]),
            n_obs=n_obs,
            n_sims=1,
            batch_size=1,
        )
        x_obs_cov = jnp.squeeze(x_obs_cov)
        x_obs_cov = (x_obs_cov - sim_summ_data_mean) / sim_summ_data_std

        key, subkey = random.split(key)
        cov_samples_std = sample(model, x_obs_cov, num_posterior_samples, key=subkey)
        cov_samples = expit(cov_samples_std * thetas_std + thetas_mean) * 10

        bias = jnp.mean(cov_samples, axis=0) - true_params
        all_biases.append(bias)

        for j in range(theta_dims):
            for ci, cl in enumerate(coverage_levels):
                lo, hi = hpdi(cov_samples[:, j], cl)
                if lo < true_params[j] < hi:
                    coverage_counts[j, ci] += 1

    estimated_coverage = coverage_counts / num_coverage_samples
    biases = jnp.stack(all_biases).ravel()

    print("Estimated coverage (rows=params, cols=80/90/95%):")
    print(estimated_coverage)

    np.save(f"{dirname}estimated_coverage.npy", estimated_coverage)
    np.save(f"{dirname}biases.npy", biases)

    return kl, mmd


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    parser = argparse.ArgumentParser(
        description="Run g-and-k with conditional Gaussian NPE.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_obs", type=int, default=1_000)
    parser.add_argument("--n_sims", type=int, default=31_623)
    args = parser.parse_args()
    run_gnk_gaussian(args)
