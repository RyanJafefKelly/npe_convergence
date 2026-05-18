"""Run MA(2)-b0 Gaussian-NPE compatibility refresh cell.

Each invocation trains one conditional Gaussian NPE for a single
``(seed, n_obs, n_sims)`` cell, then exports compatibility posterior samples at
fixed summaries ``[delta_0, 0, 0]``. The default output root is the fresh Task 3
namespace and the CLI is compatible with the reviewed PBS template.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import blackjax  # type: ignore
import blackjax.smc.resampling as resampling  # type: ignore
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax.scipy.special import expit, logit

from npe_convergence.examples.ma2 import autocov_exact, get_summaries_batches
from npe_convergence.methods.gaussian_npe import (
    ConditionalGaussianNPE,
    TrainConfig,
    fit,
    sample,
)
from npe_convergence.metrics import kullback_leibler, median_heuristic, unbiased_mmd


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "res" / "ma2_b0_compat_refresh"
DEFAULT_DELTA0_VALUES = (0.01, 0.99)
PARAM_NAMES = ("t1", "t2")
D_THETA = 2
D_SUMMARY = 3
REFERENCE_SMC_CONVENTION = (
    "legacy_run_ma2_b0_standardised_particle_logit_no_jacobian"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_git(args: list[str], default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)
    except Exception:
        return default
    return out.strip() or default


def git_dirty() -> bool:
    return bool(run_git(["status", "--porcelain"], default="dirty"))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_output_root(output_root: str | Path) -> Path:
    path = Path(output_root)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def delta_label(value: float) -> str:
    return f"{value:.12g}"


def parse_delta0_values(spec: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in spec.split(",") if part.strip())
    if not values:
        raise ValueError("--delta0-values must contain at least one value")
    return values


def parse_hidden_dims(spec: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in spec.split(",") if part.strip())
    if not dims:
        raise ValueError("--hidden-dims must contain at least one integer")
    return dims


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def write_validation_curve(path: Path, losses: dict[str, list[float]]) -> None:
    train = losses.get("train", [])
    val = losses.get("val", [])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train", "val"])
        writer.writeheader()
        for epoch in range(max(len(train), len(val))):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train": train[epoch] if epoch < len(train) else "",
                    "val": val[epoch] if epoch < len(val) else "",
                }
            )


def output_dir(output_root: Path, n_obs: int, n_sims: int, seed: int) -> Path:
    return (
        output_root
        / f"gaussian_npe_compat_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}"
    )


def delta_dir(out: Path, delta0: float) -> Path:
    return out / f"delta_0_{delta_label(delta0)}"


def expected_paths(
    output_root: Path,
    *,
    n_obs: int,
    n_sims: int,
    seed: int,
    delta0_values: tuple[float, ...],
    save_plots: bool,
) -> dict[str, str | dict[str, str]]:
    out = output_dir(output_root, n_obs, n_sims, seed)
    paths: dict[str, str | dict[str, str]] = {
        "config": rel(out / "config.json"),
        "validation_curve": rel(out / "validation_curve.csv"),
        "training_metadata": rel(out / "training_metadata.json"),
        "standardisation": rel(out / "standardisation.npz"),
    }
    for delta0 in delta0_values:
        ddir = delta_dir(out, delta0)
        delta_paths = {
            "gaussian_npe_standardised_posterior": rel(
                ddir / "gaussian_npe_standardised_posterior.npz"
            ),
            "reference_samples": rel(ddir / "reference_samples.npz"),
            "gaussian_npe_posterior_samples": rel(
                ddir / "gaussian_npe_posterior_samples.npz"
            ),
            "metrics": rel(ddir / "metrics.json"),
            "kl": rel(ddir / "kl.txt"),
            "mmd": rel(ddir / "mmd.txt"),
        }
        if save_plots:
            delta_paths.update(
                {
                    "t1_posterior_overlay": rel(ddir / "t1_posterior_overlay.pdf"),
                    "t2_posterior_overlay": rel(ddir / "t2_posterior_overlay.pdf"),
                }
            )
        paths[f"delta_0_{delta_label(delta0)}"] = delta_paths
    return paths


def fail_if_selected_output_exists(out: Path) -> None:
    if out.exists():
        raise FileExistsError(
            f"Refusing to run because output directory already exists: {out}"
        )


def bounded_from_unbounded(u: jnp.ndarray) -> jnp.ndarray:
    return 2.0 * expit(u) - 1.0


def draw_prior(
    key: Any,
    n_sims: int,
) -> tuple[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key, subkey = random.split(key)
    u1 = random.uniform(subkey, shape=(n_sims,))
    t1_bounded = 2.0 * u1 - 1.0
    t1_unbounded = logit(u1)

    key, subkey = random.split(key)
    u2 = random.uniform(subkey, shape=(n_sims,))
    t2_bounded = 2.0 * u2 - 1.0
    t2_unbounded = logit(u2)
    theta_unbounded = jnp.column_stack([t1_unbounded, t2_unbounded])
    return key, t1_bounded, t2_bounded, theta_unbounded, jnp.column_stack(
        [t1_bounded, t2_bounded]
    )


def standardise(values: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = jnp.where(std > 0, std, 1.0)
    return (values - mean) / std, mean, std


def compute_gamma_h(thetas: jnp.ndarray) -> jnp.ndarray:
    h_vals = jnp.arange(-4, 5)
    theta0 = 1.0

    def gamma_fn(h: jnp.ndarray) -> jnp.ndarray:
        abs_h = jnp.abs(h)
        return jnp.where(
            abs_h == 0,
            theta0 + thetas[0] ** 2 + thetas[1] ** 2,
            jnp.where(
                abs_h == 1,
                thetas[0] + thetas[0] * thetas[1],
                jnp.where(abs_h == 2, thetas[1], 0.0),
            ),
        )

    return jnp.array([gamma_fn(h) for h in h_vals])


def legacy_compat_covariance_matrix(
    thetas: jnp.ndarray, n_obs: int, max_lag: int = 2
) -> jnp.ndarray:
    """Covariance convention copied from the legacy b0 SMC block.

    This intentionally mirrors ``run_ma2_b0.py`` for compatibility refreshes.
    """
    gamma_h = compute_gamma_h(thetas)
    num_lags = max_lag + 1
    k_vals = jnp.arange(num_lags)
    h_vals = jnp.arange(-2, 3)
    i_vals = jnp.arange(-2, 3)

    k1, k2 = jnp.meshgrid(k_vals, k_vals, indexing="ij")
    delta_k = k1 - k2

    h_plus_delta = h_vals[:, None, None] + delta_k[None, :, :]
    valid_s1 = (h_plus_delta >= -2) & (h_plus_delta <= 2)
    h_indices = h_vals[:, None, None] + 4
    h_plus_delta_indices = h_plus_delta + 4
    gamma_products_s1 = gamma_h[h_indices] * gamma_h[h_plus_delta_indices] * valid_s1
    s1 = jnp.sum(gamma_products_s1, axis=0)

    k1_plus_i = k1[None, :, :] + i_vals[:, None, None]
    k2_minus_i = k2[None, :, :] - i_vals[:, None, None]
    valid_s2 = (
        (k1_plus_i >= -2)
        & (k1_plus_i <= 2)
        & (k2_minus_i >= -2)
        & (k2_minus_i <= 2)
    )
    k1_plus_i_indices = k1_plus_i + 4
    k2_minus_i_indices = k2_minus_i + 4
    gamma_products_s2 = gamma_h[k1_plus_i_indices] * gamma_h[k2_minus_i_indices] * valid_s2
    s2 = jnp.sum(gamma_products_s2, axis=0)
    return (s1 + s2) / n_obs


def legacy_log_prior_single(theta_particle: jnp.ndarray) -> jnp.ndarray:
    t1, t2 = bounded_from_unbounded(theta_particle)
    logp_t1 = dist.Uniform(-1, 1, validate_args=True).log_prob(t1)
    logp_t2 = dist.Uniform(-1, 1, validate_args=True).log_prob(t2)
    return logp_t1 + logp_t2


def legacy_log_prior(params: jnp.ndarray) -> jnp.ndarray:
    if params.ndim == 1:
        return legacy_log_prior_single(params)
    return jax.vmap(legacy_log_prior_single)(params)


def legacy_log_likelihood_single(
    theta_particle: jnp.ndarray,
    obs: jnp.ndarray,
    n_obs: int,
) -> jnp.ndarray:
    ma_order = 2
    thetas = bounded_from_unbounded(theta_particle)
    mean = jnp.array([autocov_exact(thetas, k, ma_order) for k in range(ma_order + 1)])
    cov_matrix = legacy_compat_covariance_matrix(thetas, n_obs, max_lag=ma_order)
    cov_matrix = cov_matrix + 1e-6 * jnp.eye(ma_order + 1)
    return dist.MultivariateNormal(mean, cov_matrix).log_prob(obs)


def legacy_log_likelihood(
    params: jnp.ndarray,
    *,
    obs: jnp.ndarray,
    n_obs: int,
) -> jnp.ndarray:
    if params.ndim == 1:
        return legacy_log_likelihood_single(params, obs=obs, n_obs=n_obs)
    return jax.vmap(legacy_log_likelihood_single, in_axes=(0, None, None))(
        params, obs, n_obs
    )


def resized_particles(particles: jnp.ndarray, n_particles: int) -> jnp.ndarray:
    if particles.shape[0] >= n_particles:
        return particles[:n_particles]
    return jnp.resize(particles, (n_particles, particles.shape[1]))


def smc_inference_loop(
    rng_key: Any,
    smc_kernel: Any,
    initial_state: Any,
    *,
    max_steps: int,
) -> tuple[int, Any]:
    def cond(carry: tuple[jnp.ndarray, Any, Any]) -> jnp.ndarray:
        i, state, _key = carry
        return (state.lmbda < 1) & (i < max_steps)

    def one_step(carry: tuple[jnp.ndarray, Any, Any]) -> tuple[jnp.ndarray, Any, Any]:
        i, state, key = carry
        key, subkey = random.split(key, 2)
        state, _ = smc_kernel(subkey, state)
        return i + 1, state, key

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (jnp.array(0), initial_state, rng_key)
    )
    return int(n_iter), final_state


def reference_samples_for_delta(
    *,
    key: Any,
    fixed_summary: jnp.ndarray,
    initial_particles: jnp.ndarray,
    n_obs: int,
    num_reference_samples: int,
    smc_step_size: float,
    smc_inverse_mass: float,
    smc_integration_steps: int,
    smc_num_mcmc_steps: int,
    smc_ess_threshold: float,
    smc_max_steps: int,
    smoke_reference_samples: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, Any], Any]:
    if smoke_reference_samples:
        key, subkey = random.split(key)
        noise = 0.05 * random.normal(subkey, shape=(num_reference_samples, D_THETA))
        theta = jnp.clip(jnp.array([0.0, 0.0]) + noise, -1.0 + 1e-6, 1.0 - 1e-6)
        return (
            theta,
            logit((theta + 1.0) / 2.0),
            {
                "reference_source": "synthetic_smoke_reference",
                "smc_iterations": 0,
                "final_lambda": None,
                "smoke_reference_samples": True,
            },
            key,
        )

    hmc_parameters = {
        "step_size": jnp.full(num_reference_samples, smc_step_size),
        "inverse_mass_matrix": smc_inverse_mass
        * jnp.ones((num_reference_samples, D_THETA)),
        "num_integration_steps": jnp.full(
            num_reference_samples, smc_integration_steps
        ),
    }
    particles0 = resized_particles(initial_particles, num_reference_samples)
    tempered_smc = blackjax.adaptive_tempered_smc(
        legacy_log_prior,
        lambda params: legacy_log_likelihood(params, obs=fixed_summary, n_obs=n_obs),
        blackjax.hmc.build_kernel(),
        blackjax.hmc.init,
        hmc_parameters,
        resampling.systematic,
        smc_ess_threshold,
        num_mcmc_steps=smc_num_mcmc_steps,
    )
    initial_state = tempered_smc.init(particles0)
    key, subkey = random.split(key)
    n_iter, smc_state = smc_inference_loop(
        subkey,
        tempered_smc.step,
        initial_state,
        max_steps=smc_max_steps,
    )
    if smc_state.lmbda < 1:
        raise RuntimeError(
            "Adaptive tempered SMC did not reach lambda=1 "
            f"within {smc_max_steps} steps for summary {np.asarray(fixed_summary)}"
        )
    particles = smc_state.particles
    theta = bounded_from_unbounded(particles)
    return (
        theta,
        particles,
        {
            "reference_source": "blackjax_adaptive_tempered_smc",
            "smc_iterations": n_iter,
            "final_lambda": float(smc_state.lmbda),
            "smoke_reference_samples": False,
        },
        key,
    )


def metric_values(
    *,
    key: Any,
    reference_samples: jnp.ndarray,
    posterior_samples: jnp.ndarray,
    num_metric_samples: int,
) -> tuple[float, float, int, Any]:
    metric_n = min(
        int(num_metric_samples),
        int(reference_samples.shape[0]),
        int(posterior_samples.shape[0]),
    )
    if metric_n < 2:
        return float("nan"), float("nan"), metric_n, key

    key, subkey = random.split(key)
    idx_npe = random.permutation(subkey, posterior_samples.shape[0])[:metric_n]
    key, subkey = random.split(key)
    idx_ref = random.permutation(subkey, reference_samples.shape[0])[:metric_n]
    ps_thin = posterior_samples[idx_npe]
    rs_thin = reference_samples[idx_ref]
    kl = float(kullback_leibler(rs_thin, ps_thin))
    lengthscale = median_heuristic(jnp.vstack([rs_thin, ps_thin]))
    mmd = float(unbiased_mmd(rs_thin, ps_thin, lengthscale))
    return kl, mmd, metric_n, key


def write_delta_plots(
    *,
    ddir: Path,
    reference_samples: jnp.ndarray,
    posterior_samples: jnp.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    for idx, name in enumerate(PARAM_NAMES):
        _, bins, _ = plt.hist(
            np.asarray(reference_samples[:, idx]),
            bins=50,
            alpha=0.6,
            label="Reference",
        )
        plt.hist(
            np.asarray(posterior_samples[:, idx]),
            bins=bins,
            alpha=0.6,
            label="Gaussian NPE",
        )
        plt.xlabel(name)
        plt.legend()
        plt.savefig(ddir / f"{name}_posterior_overlay.pdf")
        plt.clf()
    plt.close()


def build_config(
    *,
    output_root: Path,
    n_obs: int,
    n_sims: int,
    seed: int,
    delta0_values: tuple[float, ...],
    num_reference_samples: int,
    num_posterior_samples: int,
    num_metric_samples: int,
    train_config: TrainConfig,
    hidden_dims: tuple[int, ...],
    prior_batch_size: int,
    smc_settings: dict[str, Any],
    smoke_reference_samples: bool,
    save_plots: bool,
) -> dict[str, Any]:
    return {
        "task": "ma2-b0-gaussian-npe-compat-refresh",
        "created_at_utc": utc_now(),
        "git_commit": run_git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": git_dirty(),
        "branch": run_git(["branch", "--show-current"]),
        "method": "Gaussian-NPE compatibility export",
        "n_obs": n_obs,
        "n_sims": n_sims,
        "seed": seed,
        "delta0_values": [delta_label(value) for value in delta0_values],
        "fixed_summary_template": "[delta_0, 0, 0]",
        "d_summary": D_SUMMARY,
        "d_theta": D_THETA,
        "param_names": list(PARAM_NAMES),
        "output_root": rel(output_root),
        "output_dir": rel(output_dir(output_root, n_obs, n_sims, seed)),
        "num_reference_samples": num_reference_samples,
        "num_posterior_samples": num_posterior_samples,
        "num_metric_samples": num_metric_samples,
        "prior_batch_size": prior_batch_size,
        "train_config": {
            "lr": train_config.lr,
            "batch_size": train_config.batch_size,
            "max_epochs": train_config.max_epochs,
            "patience": train_config.patience,
            "val_frac": train_config.val_frac,
            "hidden_dims": list(hidden_dims),
        },
        "reference_smc_convention": REFERENCE_SMC_CONVENTION,
        "reference_smc_convention_note": (
            "The b0 reference path mirrors the legacy compatibility block in "
            "npe_convergence/scripts/run_ma2_b0.py: SMC particles live in the "
            "standardised training-theta coordinate and are mapped through "
            "2*expit(particle)-1 without an added log-Jacobian."
        ),
        "smc_settings": smc_settings,
        "smoke_reference_samples": smoke_reference_samples,
        "smoke_reference_samples_note": (
            "Synthetic reference samples are for non-scientific smoke tests only."
            if smoke_reference_samples
            else None
        ),
        "save_plots": save_plots,
        "expected_outputs": expected_paths(
            output_root,
            n_obs=n_obs,
            n_sims=n_sims,
            seed=seed,
            delta0_values=delta0_values,
            save_plots=save_plots,
        ),
    }


def print_dry_run(config: dict[str, Any]) -> None:
    print("MA2-b0 Gaussian-NPE compatibility dry-run")
    print(f"output_dir: {config['output_dir']}")
    print(f"n_obs: {config['n_obs']}")
    print(f"n_sims: {config['n_sims']}")
    print(f"seed: {config['seed']}")
    print(f"delta0_values: {', '.join(config['delta0_values'])}")
    print("expected_outputs:")
    for name, path in config["expected_outputs"].items():
        print(f"  {name}: {path}")


def run_ma2_b0_gaussian_compat_refresh(
    *,
    seed: int,
    n_obs: int,
    n_sims: int,
    delta0_values: tuple[float, ...] = DEFAULT_DELTA0_VALUES,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    fail_if_output_exists: bool = False,
    dry_run: bool = False,
    num_reference_samples: int = 10_000,
    num_posterior_samples: int = 10_000,
    num_metric_samples: int = 2000,
    hidden_dims: tuple[int, ...] = (128, 128),
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    max_epochs: int = 2000,
    patience: int = 20,
    val_frac: float = 0.1,
    prior_batch_size: int | None = None,
    smc_step_size: float = 5e-3,
    smc_inverse_mass: float = 0.1,
    smc_integration_steps: int = 100,
    smc_num_mcmc_steps: int = 5,
    smc_ess_threshold: float = 0.75,
    smc_max_steps: int = 10_000,
    smoke_reference_samples: bool = False,
    save_plots: bool = True,
) -> dict[str, Any]:
    if n_sims < 2:
        raise ValueError("--n-sims must be at least 2")
    if num_reference_samples < 2 or num_posterior_samples < 2:
        raise ValueError("sample counts must be at least 2")
    if prior_batch_size is None:
        prior_batch_size = min(1000, n_sims)

    output_root = resolve_output_root(output_root)
    out = output_dir(output_root, n_obs, n_sims, seed)
    train_config = TrainConfig(
        lr=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        val_frac=val_frac,
    )
    smc_settings = {
        "step_size": smc_step_size,
        "inverse_mass": smc_inverse_mass,
        "integration_steps": smc_integration_steps,
        "num_mcmc_steps": smc_num_mcmc_steps,
        "ess_threshold": smc_ess_threshold,
        "max_steps": smc_max_steps,
    }
    config = build_config(
        output_root=output_root,
        n_obs=n_obs,
        n_sims=n_sims,
        seed=seed,
        delta0_values=delta0_values,
        num_reference_samples=num_reference_samples,
        num_posterior_samples=num_posterior_samples,
        num_metric_samples=num_metric_samples,
        train_config=train_config,
        hidden_dims=hidden_dims,
        prior_batch_size=prior_batch_size,
        smc_settings=smc_settings,
        smoke_reference_samples=smoke_reference_samples,
        save_plots=save_plots,
    )
    if dry_run:
        print_dry_run(config)
        return config

    if fail_if_output_exists:
        fail_if_selected_output_exists(out)
    elif out.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {out}. "
            "Use a fresh output root."
        )

    if smoke_reference_samples and output_root.resolve().is_relative_to(
        (REPO_ROOT / "res").resolve()
    ):
        raise ValueError(
            "--smoke-reference-samples is only allowed outside the repo res/ tree"
        )

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)
    start = time.time()

    key = random.key(seed)
    key, t1_bounded, t2_bounded, theta_unbounded, _theta_bounded = draw_prior(
        key, n_sims
    )
    batch_size_sims = min(prior_batch_size, n_sims)
    print("Simulating MA2 prior predictive summaries...")
    key, subkey = random.split(key)
    sim_summ_data = get_summaries_batches(
        subkey,
        t1_bounded,
        t2_bounded,
        n_obs,
        n_sims,
        batch_size_sims,
    )

    thetas, thetas_mean, thetas_std = standardise(theta_unbounded)
    sim_summ_data, sim_summ_data_mean, sim_summ_data_std = standardise(sim_summ_data)
    initial_reference_particles = thetas

    np.savez(
        out / "standardisation.npz",
        theta_unbounded_mean=np.asarray(thetas_mean),
        theta_unbounded_std=np.asarray(thetas_std),
        summary_mean=np.asarray(sim_summ_data_mean),
        summary_std=np.asarray(sim_summ_data_std),
        param_names=np.asarray(PARAM_NAMES),
    )

    key, subkey = random.split(key)
    model = ConditionalGaussianNPE(
        d_summary=D_SUMMARY,
        d_theta=D_THETA,
        hidden_dims=hidden_dims,
        key=subkey,
    )
    print("Training conditional Gaussian NPE...")
    key, subkey = random.split(key)
    model, losses = fit(model, thetas, sim_summ_data, key=subkey, config=train_config)
    write_validation_curve(out / "validation_curve.csv", losses)

    delta_metrics: dict[str, Any] = {}
    for delta0 in delta0_values:
        label = delta_label(delta0)
        ddir = delta_dir(out, delta0)
        ddir.mkdir(parents=True, exist_ok=False)
        fixed_summary = jnp.array([delta0, 0.0, 0.0])
        fixed_summary_standardised = (
            fixed_summary - sim_summ_data_mean
        ) / sim_summ_data_std

        mu_standardised, L_standardised = model(fixed_summary_standardised)
        cov_standardised = L_standardised @ L_standardised.T

        key, subkey = random.split(key)
        posterior_standardised = sample(
            model,
            fixed_summary_standardised,
            num_posterior_samples,
            key=subkey,
        )
        posterior_unbounded = posterior_standardised * thetas_std + thetas_mean
        posterior_samples = bounded_from_unbounded(posterior_unbounded)

        print(f"Running reference export for delta_0={label}...")
        (
            reference_samples,
            reference_particles,
            reference_metadata,
            key,
        ) = reference_samples_for_delta(
            key=key,
            fixed_summary=fixed_summary,
            initial_particles=initial_reference_particles,
            n_obs=n_obs,
            num_reference_samples=num_reference_samples,
            smc_step_size=smc_step_size,
            smc_inverse_mass=smc_inverse_mass,
            smc_integration_steps=smc_integration_steps,
            smc_num_mcmc_steps=smc_num_mcmc_steps,
            smc_ess_threshold=smc_ess_threshold,
            smc_max_steps=smc_max_steps,
            smoke_reference_samples=smoke_reference_samples,
        )

        kl, mmd, metric_n, key = metric_values(
            key=key,
            reference_samples=reference_samples,
            posterior_samples=posterior_samples,
            num_metric_samples=num_metric_samples,
        )

        np.savez(
            ddir / "gaussian_npe_standardised_posterior.npz",
            mu_standardised=np.asarray(mu_standardised),
            L_standardised=np.asarray(L_standardised),
            cov_standardised=np.asarray(cov_standardised),
            fixed_summary=np.asarray(fixed_summary),
            fixed_summary_standardised=np.asarray(fixed_summary_standardised),
            theta_unbounded_mean=np.asarray(thetas_mean),
            theta_unbounded_std=np.asarray(thetas_std),
            summary_mean=np.asarray(sim_summ_data_mean),
            summary_std=np.asarray(sim_summ_data_std),
            param_names=np.asarray(PARAM_NAMES),
            n_obs=n_obs,
            n_sims=n_sims,
            seed=seed,
            delta0=delta0,
        )
        np.savez(
            ddir / "reference_samples.npz",
            samples=np.asarray(reference_samples),
            particles_standardised=np.asarray(reference_particles),
            fixed_summary=np.asarray(fixed_summary),
            param_names=np.asarray(PARAM_NAMES),
            n_obs=n_obs,
            n_sims=n_sims,
            seed=seed,
            delta0=delta0,
            sample_count=int(reference_samples.shape[0]),
            reference_smc_convention=REFERENCE_SMC_CONVENTION,
        )
        np.savez(
            ddir / "gaussian_npe_posterior_samples.npz",
            samples=np.asarray(posterior_samples),
            theta_unbounded=np.asarray(posterior_unbounded),
            theta_standardised=np.asarray(posterior_standardised),
            fixed_summary=np.asarray(fixed_summary),
            fixed_summary_standardised=np.asarray(fixed_summary_standardised),
            param_names=np.asarray(PARAM_NAMES),
            n_obs=n_obs,
            n_sims=n_sims,
            seed=seed,
            delta0=delta0,
            sample_count=int(posterior_samples.shape[0]),
        )

        metrics = {
            "delta0": label,
            "fixed_summary": np.asarray(fixed_summary).tolist(),
            "kl_reference_to_gaussian_npe": kl,
            "mmd_reference_vs_gaussian_npe": mmd,
            "metric_sample_count": metric_n,
            "reference_sample_count": int(reference_samples.shape[0]),
            "gaussian_npe_sample_count": int(posterior_samples.shape[0]),
            "reference_samples_shape": list(np.asarray(reference_samples).shape),
            "gaussian_npe_samples_shape": list(np.asarray(posterior_samples).shape),
            "reference_smc_convention": REFERENCE_SMC_CONVENTION,
            **reference_metadata,
            "output_dir": rel(ddir),
            "files": {
                "gaussian_npe_standardised_posterior": rel(
                    ddir / "gaussian_npe_standardised_posterior.npz"
                ),
                "reference_samples": rel(ddir / "reference_samples.npz"),
                "gaussian_npe_posterior_samples": rel(
                    ddir / "gaussian_npe_posterior_samples.npz"
                ),
                "metrics": rel(ddir / "metrics.json"),
                "kl": rel(ddir / "kl.txt"),
                "mmd": rel(ddir / "mmd.txt"),
            },
        }
        write_json(ddir / "metrics.json", metrics)
        with (ddir / "kl.txt").open("w") as f:
            f.write(str(kl))
        with (ddir / "mmd.txt").open("w") as f:
            f.write(str(mmd))
        if save_plots:
            write_delta_plots(
                ddir=ddir,
                reference_samples=reference_samples,
                posterior_samples=posterior_samples,
            )
        delta_metrics[label] = metrics

    training_metadata = {
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - start,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jnp.ones(1).devices()],
        "train_epochs_completed": len(losses.get("train", [])),
        "best_validation_loss": (
            min(losses.get("val", [])) if losses.get("val", []) else None
        ),
        "final_train_loss": (
            losses.get("train", [None])[-1] if losses.get("train", []) else None
        ),
        "final_validation_loss": (
            losses.get("val", [None])[-1] if losses.get("val", []) else None
        ),
        "n_obs": n_obs,
        "n_sims": n_sims,
        "seed": seed,
        "delta0_values": [delta_label(value) for value in delta0_values],
        "num_reference_samples": num_reference_samples,
        "num_posterior_samples": num_posterior_samples,
        "num_metric_samples": num_metric_samples,
        "train_config": config["train_config"],
        "smc_settings": smc_settings,
        "reference_smc_convention": REFERENCE_SMC_CONVENTION,
        "output_root": rel(output_root),
        "output_dir": rel(out),
        "delta_metrics": delta_metrics,
    }
    write_json(out / "training_metadata.json", training_metadata)
    print(f"completed {rel(out)}")
    return training_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-obs", "--n_obs", dest="n_obs", type=int, default=1000)
    parser.add_argument("--n-sims", "--n_sims", dest="n_sims", type=int, required=True)
    parser.add_argument(
        "--delta0-values",
        default=",".join(str(value) for value in DEFAULT_DELTA0_VALUES),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fail-if-output-exists", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-reference-samples", type=int, default=10_000)
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--num-metric-samples", type=int, default=2000)
    parser.add_argument("--hidden-dims", default="128,128")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--prior-batch-size", type=int, default=None)
    parser.add_argument("--smc-step-size", type=float, default=5e-3)
    parser.add_argument("--smc-inverse-mass", type=float, default=0.1)
    parser.add_argument("--smc-integration-steps", type=int, default=100)
    parser.add_argument("--smc-num-mcmc-steps", type=int, default=5)
    parser.add_argument("--smc-ess-threshold", type=float, default=0.75)
    parser.add_argument("--smc-max-steps", type=int, default=10_000)
    parser.add_argument(
        "--smoke-reference-samples",
        action="store_true",
        help="Use synthetic reference samples. Only allowed outside repo res/.",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-delta t1/t2 overlay PDFs.",
    )
    return parser


def main() -> None:
    numpyro.set_host_device_count(4)
    args = build_parser().parse_args()
    run_ma2_b0_gaussian_compat_refresh(
        seed=args.seed,
        n_obs=args.n_obs,
        n_sims=args.n_sims,
        delta0_values=parse_delta0_values(args.delta0_values),
        output_root=args.output_root,
        fail_if_output_exists=args.fail_if_output_exists,
        dry_run=args.dry_run,
        num_reference_samples=args.num_reference_samples,
        num_posterior_samples=args.num_posterior_samples,
        num_metric_samples=args.num_metric_samples,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_frac=args.val_frac,
        prior_batch_size=args.prior_batch_size,
        smc_step_size=args.smc_step_size,
        smc_inverse_mass=args.smc_inverse_mass,
        smc_integration_steps=args.smc_integration_steps,
        smc_num_mcmc_steps=args.smc_num_mcmc_steps,
        smc_ess_threshold=args.smc_ess_threshold,
        smc_max_steps=args.smc_max_steps,
        smoke_reference_samples=args.smoke_reference_samples,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()
