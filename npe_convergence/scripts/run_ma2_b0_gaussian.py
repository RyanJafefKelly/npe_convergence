"""Run MA(2) b0 model with conditional Gaussian NPE.

Mirrors run_ma2_b0.py but replaces the normalising flow with a conditional
Gaussian posterior approximation. Uses NUTS as the reference posterior
for KL/MMD evaluation.

Usage:
    python npe_convergence/scripts/run_ma2_b0_gaussian.py --seed=1 --n_obs=1000 --n_sims=6907
"""

import argparse
import os
import pickle as pkl

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro  # type: ignore
from jax.scipy.special import expit, logit
from numpyro.diagnostics import hpdi  # type: ignore
from numpyro.infer import MCMC, NUTS  # type: ignore

from npe_convergence.examples.ma2 import (
    get_summaries_batches,
    numpyro_model_b0,
)
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
    return f"res/ma2_b0/nuts_cache_n_obs_{n_obs}_seed_{seed}.pkl"


def get_nuts_posterior(
    seed: int,
    x_obs: jnp.ndarray,
    n_obs: int,
    num_samples: int = 10_000,
    num_warmup: int = 2_000,
) -> jnp.ndarray:
    """Load cached NUTS samples or run + cache if not found.

    Returns array of shape (num_samples, 2) with columns [t1, t2].
    """
    cache_path = _nuts_cache_path(n_obs, seed)
    if os.path.exists(cache_path):
        print(f"Loading cached NUTS posterior from {cache_path}")
        with open(cache_path, "rb") as f:
            return pkl.load(f)

    print("Running NUTS (will cache for future runs)...")
    nuts_kernel = NUTS(numpyro_model_b0)
    thinning = 10
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples * thinning,
        thinning=thinning,
    )
    mcmc.run(
        random.key(1),
        obs=x_obs,
        init_params={"t1": 0.0, "t2": 0.0},
        n_obs=n_obs,
    )
    mcmc.print_summary()
    samples_dict = mcmc.get_samples()
    samples = jnp.column_stack([samples_dict["t1"], samples_dict["t2"]])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pkl.dump(samples, f)
    print(f"Cached NUTS posterior to {cache_path}")
    return samples


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_ma2_b0_gaussian(*args, **kwargs):
    try:
        seed, n_obs, n_sims = args
    except ValueError:
        args = args[0]
        seed, n_obs, n_sims = args.seed, args.n_obs, args.n_sims

    dirname = f"res/ma2_b0/gaussian_npe_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}/"
    os.makedirs(dirname, exist_ok=True)

    # -- Ground truth data --------------------------------------------------
    true_params = jnp.array([0.6, 0.2])
    key = random.key(seed)

    key, subkey = random.split(key)
    x_obs = get_summaries_batches(
        subkey,
        jnp.atleast_1d(true_params[0]),
        jnp.atleast_1d(true_params[1]),
        n_obs, 1, 1,
    )
    x_obs = jnp.squeeze(x_obs)
    print("x_obs:", x_obs)

    # -- Exact posterior (NUTS, cached) -------------------------------------
    true_posterior_samples = get_nuts_posterior(seed, x_obs, n_obs)
    param_names = ["t1", "t2"]
    for ii, name in enumerate(param_names):
        plt.hist(true_posterior_samples[:, ii], bins=50, label=name)
        plt.axvline(true_params[ii], color="red")
        plt.legend()
        plt.savefig(f"{dirname}true_samples_{name}.pdf")
        plt.clf()

    # -- Prior simulation ---------------------------------------------------
    key, subkey = random.split(key)
    u1 = random.uniform(subkey, shape=(n_sims,))
    t1_bounded = 2 * u1 - 1
    t1_unbounded = logit(u1)

    key, subkey = random.split(key)
    u2 = random.uniform(subkey, shape=(n_sims,))
    t2_bounded = 2 * u2 - 1
    t2_unbounded = logit(u2)

    key, subkey = random.split(key)
    batch_size = min(1000, n_sims)
    print(">>> Simulating prior predictive...")
    sim_summ_data = get_summaries_batches(
        subkey, t1_bounded, t2_bounded, n_obs, n_sims, batch_size
    )
    print(">>> Simulations done, standardising...")

    # -- Standardise --------------------------------------------------------
    thetas_unbounded = jnp.column_stack([t1_unbounded, t2_unbounded])
    thetas_mean = thetas_unbounded.mean(axis=0)
    thetas_std = thetas_unbounded.std(axis=0)
    thetas = (thetas_unbounded - thetas_mean) / thetas_std

    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std
    x_obs_std = (x_obs - sim_summ_data_mean) / sim_summ_data_std

    # -- Train conditional Gaussian -----------------------------------------
    theta_dims = 2
    summary_dims = 3

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
        patience=20,
    )
    model, losses = fit(
        model, thetas, sim_summ_data, key=subkey, config=config,
    )
    print(">>> Training done, sampling posterior...")

    plt.plot(losses["train"], label="train")
    plt.plot(losses["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.legend()
    plt.savefig(f"{dirname}losses.pdf")
    plt.clf()

    # -- Sample posterior + invert standardisation --------------------------
    num_posterior_samples = 10_000
    key, subkey = random.split(key)
    posterior_std = sample(model, x_obs_std, num_posterior_samples, key=subkey)
    posterior_unbounded = posterior_std * thetas_std + thetas_mean
    posterior_samples = jnp.column_stack([
        2 * expit(posterior_unbounded[:, 0]) - 1,
        2 * expit(posterior_unbounded[:, 1]) - 1,
    ])
    print(">>> Sampling done, computing metrics...")

    for ii, name in enumerate(param_names):
        _, bins, _ = plt.hist(
            posterior_samples[:, ii], bins=50, alpha=0.8, label="Gaussian NPE"
        )
        plt.hist(true_posterior_samples[:, ii], bins=bins, alpha=0.8, label="NUTS")
        plt.axvline(true_params[ii], color="black")
        plt.legend()
        plt.savefig(f"{dirname}posterior_{name}.pdf")
        plt.clf()

    # -- Metrics ------------------------------------------------------------
    n_metric = 2000
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
        x_obs_cov = get_summaries_batches(
            subkey,
            jnp.atleast_1d(true_params[0]),
            jnp.atleast_1d(true_params[1]),
            n_obs, 1, 1,
        )
        x_obs_cov = jnp.squeeze(x_obs_cov)
        x_obs_cov = (x_obs_cov - sim_summ_data_mean) / sim_summ_data_std

        key, subkey = random.split(key)
        cov_samples_std = sample(model, x_obs_cov, num_posterior_samples, key=subkey)
        cov_unbounded = cov_samples_std * thetas_std + thetas_mean
        cov_samples = jnp.column_stack([
            2 * expit(cov_unbounded[:, 0]) - 1,
            2 * expit(cov_unbounded[:, 1]) - 1,
        ])

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
        description="Run MA(2) b0 with conditional Gaussian NPE.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_obs", type=int, default=1_000)
    parser.add_argument("--n_sims", type=int, default=6_907)
    args = parser.parse_args()
    run_ma2_b0_gaussian(args)
