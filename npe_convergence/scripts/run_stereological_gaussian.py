"""Run stereological model with conditional Gaussian NPE.

Mirrors run_stereological.py but replaces the normalising flow with a
conditional Gaussian posterior approximation. Evaluation is coverage + bias
only (no KL/MMD — the stereological likelihood is intractable).

Usage:
    python npe_convergence/scripts/run_stereological_gaussian.py --seed=1 --n_obs=1000 --n_sims=31623
"""

import argparse
import os
import pickle as pkl

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import numpyro  # type: ignore
from numpyro.diagnostics import hpdi  # type: ignore

from npe_convergence.examples.stereological import (
    get_prior_samples,
    get_summaries,
    get_summaries_batches,
    stereological,
    transform_to_bounded,
    transform_to_unbounded,
)
from npe_convergence.methods.gaussian_npe import (
    ConditionalGaussianNPE,
    TrainConfig,
    fit,
    sample,
)


def run_stereological_gaussian(*args, **kwargs):
    try:
        seed, n_obs, n_sims = args
    except ValueError:
        args = args[0]
        seed, n_obs, n_sims = args.seed, args.n_obs, args.n_sims

    dirname = f"res/stereological/gaussian_npe_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}/"
    os.makedirs(dirname, exist_ok=True)

    # -- Ground truth data --------------------------------------------------
    true_params = jnp.array([100.0, 2.0, -0.1])
    key = random.key(seed)

    key, subkey = random.split(key)
    x_obs = stereological(subkey, *true_params, num_samples=1, n_obs=n_obs)
    x_obs = get_summaries(x_obs)
    x_obs_original = x_obs.copy()
    print("x_obs:", x_obs)

    # -- Prior simulation ---------------------------------------------------
    key, subkey = random.split(key)
    thetas_bounded = get_prior_samples(subkey, n_sims)
    thetas_unbounded = transform_to_unbounded(thetas_bounded)

    key, subkey = random.split(key)
    batch_size = min(50, n_sims)
    print(">>> Simulating prior predictive...")
    sim_summ_data = get_summaries_batches(
        subkey, thetas_bounded, n_obs, n_sims, batch_size
    )
    print(">>> Simulations done, standardising...")

    # -- Standardise --------------------------------------------------------
    thetas_mean = thetas_unbounded.mean(axis=0)
    thetas_std = thetas_unbounded.std(axis=0)
    thetas = (thetas_unbounded - thetas_mean) / thetas_std

    sim_summ_data_mean = jnp.nanmean(sim_summ_data, axis=0)
    sim_summ_data_std = jnp.nanstd(sim_summ_data, axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std
    x_obs_std = (x_obs - sim_summ_data_mean) / sim_summ_data_std

    # -- Train conditional Gaussian -----------------------------------------
    theta_dims = 3
    summary_dims = 4

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
        patience=200,
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
    posterior_samples = transform_to_bounded(posterior_unbounded)
    print(">>> Sampling done, saving results...")

    param_names = ["lambda", "sigma", "xi"]
    for ii, name in enumerate(param_names):
        plt.hist(posterior_samples[:, ii], bins=50, alpha=0.8, label="Gaussian NPE")
        plt.axvline(true_params[ii], color="black")
        plt.legend()
        plt.savefig(f"{dirname}posterior_{name}.pdf")
        plt.clf()

    with open(f"{dirname}posterior_samples.pkl", "wb") as f:
        pkl.dump(posterior_samples, f)

    # -- Coverage analysis --------------------------------------------------
    num_coverage_samples = 100
    coverage_levels = [0.8, 0.9, 0.95]
    coverage_counts = np.zeros((theta_dims, len(coverage_levels)))
    all_biases = []

    for i in range(num_coverage_samples):
        key, subkey = random.split(key)
        x_obs_cov = stereological(subkey, *true_params, num_samples=1, n_obs=n_obs)
        x_obs_cov = get_summaries(x_obs_cov)
        x_obs_cov = (x_obs_cov - sim_summ_data_mean) / sim_summ_data_std

        key, subkey = random.split(key)
        cov_samples_std = sample(model, x_obs_cov, num_posterior_samples, key=subkey)
        cov_samples = transform_to_bounded(
            cov_samples_std * thetas_std + thetas_mean
        )

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

    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run stereological model with conditional Gaussian NPE.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_obs", type=int, default=1_000)
    parser.add_argument("--n_sims", type=int, default=31_623)
    args = parser.parse_args()
    run_stereological_gaussian(args)
