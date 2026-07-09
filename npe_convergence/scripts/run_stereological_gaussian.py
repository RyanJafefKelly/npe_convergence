"""Run stereological model with conditional Gaussian NPE.

Mirrors run_stereological.py but replaces the normalising flow with a
conditional Gaussian posterior approximation. Evaluation is coverage + bias
only (no KL/MMD — the stereological likelihood is intractable).

Usage:
    python npe_convergence/scripts/run_stereological_gaussian.py --seed=1 --n_obs=1000 --n_sims=31623
"""

import argparse
import json
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


def _namespace_or_kwarg(args, kwargs, name, default=None):
    if name in kwargs:
        return kwargs[name]
    if len(args) == 1 and hasattr(args[0], name):
        return getattr(args[0], name)
    return default


def _resolve_run_args(args, kwargs):
    if len(args) == 3:
        seed, n_obs, n_sims = args
    elif len(args) == 1:
        namespace = args[0]
        seed, n_obs, n_sims = namespace.seed, namespace.n_obs, namespace.n_sims
    elif not args:
        seed = kwargs["seed"]
        n_obs = kwargs["n_obs"]
        n_sims = kwargs["n_sims"]
    else:
        raise ValueError(
            "run_stereological_gaussian expects a namespace, three positional values, "
            "or keyword values"
        )

    output_root = _namespace_or_kwarg(args, kwargs, "output_root", "res/stereological")
    output_dir = _namespace_or_kwarg(args, kwargs, "output_dir")
    return seed, n_obs, n_sims, output_root, output_dir


def _ensure_finite(name, value):
    arr = np.asarray(value)
    if not np.isfinite(arr).all():
        bad = int(np.size(arr) - np.isfinite(arr).sum())
        raise ValueError(f"{name} contains {bad} non-finite values")


def _safe_standardise(data, name):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    _ensure_finite(f"{name} mean", mean)
    std = jnp.where(jnp.isfinite(std) & (std > 0), std, 1.0)
    _ensure_finite(f"{name} std", std)
    return (data - mean) / std, mean, std


def _filter_finite_training_rows(thetas_unbounded, sim_summ_data):
    mask = np.asarray(
        jnp.all(jnp.isfinite(thetas_unbounded), axis=1)
        & jnp.all(jnp.isfinite(sim_summ_data), axis=1)
    )
    raw_count = int(mask.shape[0])
    kept_count = int(mask.sum())
    dropped_count = raw_count - kept_count
    if kept_count < 2:
        raise ValueError(
            "fewer than two finite simulation rows remain after filtering: "
            f"raw={raw_count}, kept={kept_count}, dropped={dropped_count}"
        )
    return thetas_unbounded[mask], sim_summ_data[mask], {
        "raw_simulation_rows": raw_count,
        "finite_simulation_rows": kept_count,
        "dropped_nonfinite_simulation_rows": dropped_count,
    }


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def run_stereological_gaussian(*args, **kwargs):
    seed, n_obs, n_sims, output_root, output_dir = _resolve_run_args(args, kwargs)
    num_posterior_samples = int(_namespace_or_kwarg(args, kwargs, "num_posterior_samples", 10_000))
    num_coverage_samples = int(_namespace_or_kwarg(args, kwargs, "num_coverage_samples", 100))
    max_epochs = int(_namespace_or_kwarg(args, kwargs, "max_epochs", 2000))
    patience = int(_namespace_or_kwarg(args, kwargs, "patience", 20))
    train_batch_size = int(_namespace_or_kwarg(args, kwargs, "train_batch_size", 256))
    learning_rate = float(_namespace_or_kwarg(args, kwargs, "learning_rate", 5e-4))

    if output_dir is None:
        dirname = os.path.join(
            str(output_root),
            f"gaussian_npe_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}",
        )
    else:
        dirname = str(output_dir)
    dirname = dirname.rstrip("/") + "/"
    os.makedirs(dirname, exist_ok=True)

    # -- Ground truth data --------------------------------------------------
    true_params = jnp.array([100.0, 2.0, -0.1])
    key = random.key(seed)

    key, subkey = random.split(key)
    x_obs = stereological(subkey, *true_params, num_samples=1, n_obs=n_obs)
    x_obs = jnp.squeeze(get_summaries(x_obs))
    _ensure_finite("observed summary", x_obs)
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
    thetas_unbounded, sim_summ_data, diagnostics = _filter_finite_training_rows(
        thetas_unbounded, sim_summ_data
    )
    print(
        "finite training rows: "
        f"{diagnostics['finite_simulation_rows']}/{diagnostics['raw_simulation_rows']} "
        f"(dropped {diagnostics['dropped_nonfinite_simulation_rows']})"
    )

    # -- Standardise --------------------------------------------------------
    thetas, thetas_mean, thetas_std = _safe_standardise(thetas_unbounded, "theta")
    sim_summ_data, sim_summ_data_mean, sim_summ_data_std = _safe_standardise(
        sim_summ_data, "summary"
    )
    x_obs_std = (x_obs - sim_summ_data_mean) / sim_summ_data_std
    diagnostics["theta_std_floored_count"] = int(np.sum(np.asarray(thetas_std) == 1.0))
    diagnostics["summary_std_floored_count"] = int(np.sum(np.asarray(sim_summ_data_std) == 1.0))
    _write_json(f"{dirname}run_diagnostics.json", diagnostics)

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
        lr=learning_rate,
        batch_size=train_batch_size,
        max_epochs=max_epochs,
        patience=patience,
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
    key, subkey = random.split(key)
    posterior_std = sample(model, x_obs_std, num_posterior_samples, key=subkey)
    posterior_unbounded = posterior_std * thetas_std + thetas_mean
    posterior_samples = transform_to_bounded(posterior_unbounded)
    _ensure_finite("posterior samples", posterior_samples)
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
    coverage_levels = [0.8, 0.9, 0.95]
    coverage_counts = np.zeros((theta_dims, len(coverage_levels)))
    all_biases = []

    for i in range(num_coverage_samples):
        key, subkey = random.split(key)
        x_obs_cov = stereological(subkey, *true_params, num_samples=1, n_obs=n_obs)
        x_obs_cov = jnp.squeeze(get_summaries(x_obs_cov))
        _ensure_finite("coverage observed summary", x_obs_cov)
        x_obs_cov = (x_obs_cov - sim_summ_data_mean) / sim_summ_data_std

        key, subkey = random.split(key)
        cov_samples_std = sample(model, x_obs_cov, num_posterior_samples, key=subkey)
        cov_samples = transform_to_bounded(
            cov_samples_std * thetas_std + thetas_mean
        )
        _ensure_finite("coverage posterior samples", cov_samples)

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
    parser.add_argument("--output-root", type=str, default="res/stereological")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--num-coverage-samples", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    args = parser.parse_args()
    run_stereological_gaussian(args)
