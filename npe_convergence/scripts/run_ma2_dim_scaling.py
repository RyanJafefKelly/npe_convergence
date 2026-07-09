"""Run MA(2) dimension-scaling cells for the Section 3.3 diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.environ.get('MA2_DIM_SCALING_CPU_DEVICES', '4')}",
)

import blackjax  # type: ignore
import blackjax.smc.resampling as resampling  # type: ignore
import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib
import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax.scipy.special import expit, logit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.ma2 import (  # noqa: E402
    autocov_exact,
    compute_covariance_matrix,
    get_summaries_batches,
)
from npe_convergence.metrics import kullback_leibler, median_heuristic, unbiased_mmd  # noqa: E402


PARAM_NAMES = ("t1", "t2")
TRUE_PARAMS = jnp.array([0.6, 0.2])
D_THETA = 2
D_S_GRID = (3, 5, 7, 11, 15)
METHODS = ("flow_npe", "gaussian_npe")
DEFAULT_N_OBS = 1000
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "res" / "overnight_20260601" / "dim_scaling" / "ma2_n1000"
REFERENCE_RE = re.compile(
    r"reference_cache_v1_d_s_(?P<d_s>\d+)_n_obs_(?P<n_obs>\d+)_seed_(?P<seed>\d+)\.npz$"
)
CELL_RE = re.compile(
    r"ma2_(?P<method>flow_npe|gaussian_npe)_d_s_(?P<d_s>\d+)_"
    r"n_obs_(?P<n_obs>\d+)_n_sims_(?P<n_sims>\d+)_seed_(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class CellSpec:
    d_s: int
    method: str
    seed: int
    n_obs: int
    n_sims: int

    @property
    def d(self) -> int:
        return self.d_s + D_THETA

    @property
    def scaled_budget(self) -> float:
        return self.n_sims / (self.d * self.d * self.n_obs)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_timestamp(value: str | None = None) -> str:
    if value is None:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        .astimezone(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )


def resolve_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def git_value(args: list[str], default: str | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except Exception:
        return default


def git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
        )
    except Exception:
        return None


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "numpyro_version": getattr(numpyro, "__version__", None),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX"),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["branch", "--show-current"]),
        "git_dirty": git_dirty(),
        "runner": str(Path(__file__).resolve()),
    }


def clean_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_for_json(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return clean_for_json(value.item())
    if isinstance(value, jax.Array):
        return clean_for_json(np.asarray(value))
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(clean_for_json(payload), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def rng_for(seed: int, *parts: object) -> jax.Array:
    key = random.key(seed)
    for part in parts:
        key = random.fold_in(key, stable_int(part))
    return key


def validate_d_s(d_s: int) -> int:
    if d_s not in D_S_GRID:
        raise ValueError(f"unsupported d_s={d_s}; expected one of {D_S_GRID}")
    return d_s


def max_lag_for_d_s(d_s: int) -> int:
    return validate_d_s(d_s) - 1


def d_total(d_s: int) -> int:
    return validate_d_s(d_s) + D_THETA


def scaled_n_sims(d_s: int, n_obs: int, scale: int) -> int:
    d = d_total(d_s)
    return int(scale * d * d * n_obs)


def bounded_from_unbounded(value: jnp.ndarray) -> jnp.ndarray:
    return 2.0 * expit(value) - 1.0


def observed_summary(seed: int, n_obs: int, d_s: int) -> np.ndarray:
    key = random.key(seed)
    key, subkey = random.split(key)
    summaries = get_summaries_batches(
        subkey,
        jnp.atleast_1d(TRUE_PARAMS[0]),
        jnp.atleast_1d(TRUE_PARAMS[1]),
        n_obs=n_obs,
        n_sims=1,
        batch_size=1,
        max_lag=max_lag_for_d_s(d_s),
    )
    return np.asarray(jnp.squeeze(summaries))


def hash_array(arr: np.ndarray) -> str:
    stable = np.asarray(arr, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(stable.shape).encode())
    digest.update(stable.tobytes())
    return digest.hexdigest()


def mean_summary(theta: jnp.ndarray, d_s: int) -> jnp.ndarray:
    return jnp.array([autocov_exact(theta, lag, ma_order=2) for lag in range(d_s)])


def covariance_summary(theta: jnp.ndarray, n_obs: int, d_s: int) -> jnp.ndarray:
    cov = compute_covariance_matrix(theta, n_obs, max_lag=max_lag_for_d_s(d_s))
    return cov + 1e-6 * jnp.eye(d_s)


def reference_cache_path(output_root: Path | str, d_s: int, n_obs: int, seed: int) -> Path:
    root = resolve_path(output_root)
    return root / f"reference_cache_v1_d_s_{d_s}_n_obs_{n_obs}_seed_{seed}.npz"


def reference_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.stem + ".json")


def cell_dir(output_root: Path | str, spec: CellSpec) -> Path:
    root = resolve_path(output_root)
    return root / (
        f"ma2_{spec.method}_d_s_{spec.d_s}_n_obs_{spec.n_obs}_"
        f"n_sims_{spec.n_sims}_seed_{spec.seed}"
    )


def log_prior_unbounded_single(theta_unbounded: jnp.ndarray) -> jnp.ndarray:
    bounded = bounded_from_unbounded(theta_unbounded)
    s = expit(theta_unbounded)
    log_uniform = dist.Uniform(-1, 1, validate_args=True).log_prob(bounded)
    log_jac = jnp.log(2.0) + jnp.log(s) + jnp.log1p(-s)
    return jnp.sum(log_uniform + log_jac)


def log_prior_unbounded(theta_unbounded: jnp.ndarray) -> jnp.ndarray:
    if theta_unbounded.ndim == 1:
        return log_prior_unbounded_single(theta_unbounded)
    return jax.vmap(log_prior_unbounded_single)(theta_unbounded)


def log_likelihood_unbounded_single(
    theta_unbounded: jnp.ndarray,
    obs: jnp.ndarray,
    n_obs: int,
    d_s: int,
) -> jnp.ndarray:
    theta = bounded_from_unbounded(theta_unbounded)
    return dist.MultivariateNormal(
        mean_summary(theta, d_s),
        covariance_summary(theta, n_obs, d_s),
    ).log_prob(obs)


def log_likelihood_unbounded(
    theta_unbounded: jnp.ndarray,
    *,
    obs: jnp.ndarray,
    n_obs: int,
    d_s: int,
) -> jnp.ndarray:
    if theta_unbounded.ndim == 1:
        return log_likelihood_unbounded_single(theta_unbounded, obs, n_obs, d_s)
    return jax.vmap(log_likelihood_unbounded_single, in_axes=(0, None, None, None))(
        theta_unbounded,
        obs,
        n_obs,
        d_s,
    )


def smc_loop(rng_key: jax.Array, smc_kernel: Any, initial_state: Any, max_steps: int) -> tuple[int, Any]:
    def cond(carry: tuple[jnp.ndarray, Any, Any]) -> jnp.ndarray:
        i, state, _key = carry
        return (state.lmbda < 1) & (i < max_steps)

    def step(carry: tuple[jnp.ndarray, Any, Any]) -> tuple[jnp.ndarray, Any, Any]:
        i, state, key = carry
        key, subkey = random.split(key)
        state, _ = smc_kernel(subkey, state)
        return i + 1, state, key

    n_iter, final_state, _ = jax.lax.while_loop(cond, step, (jnp.array(0), initial_state, rng_key))
    return int(n_iter), final_state


def precompute_reference(
    *,
    output_root: Path | str,
    d_s: int,
    seed: int,
    n_obs: int,
    num_reference_samples: int,
    smc_step_size: float,
    smc_inverse_mass: float,
    smc_integration_steps: int,
    smc_num_mcmc_steps: int,
    smc_ess_threshold: float,
    smc_max_steps: int,
    force: bool,
) -> Path:
    validate_d_s(d_s)
    root = resolve_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    cache = reference_cache_path(root, d_s, n_obs, seed)
    metadata_path = reference_metadata_path(cache)
    if cache.exists() and metadata_path.exists() and not force:
        metadata = load_json(metadata_path)
        if metadata.get("status") != "complete":
            raise RuntimeError(f"existing reference cache is not complete: {metadata_path}")
        print(f"Skipping existing MA2 reference cache: {cache}")
        return cache
    if force:
        for path in (cache, metadata_path):
            if path.exists():
                path.unlink()

    start = time.perf_counter()
    obs = observed_summary(seed, n_obs, d_s)
    obs_hash = hash_array(obs)
    metadata: dict[str, Any] = {
        "status": "started",
        "created_at_utc": utc_now(),
        "d_s": d_s,
        "d_theta": D_THETA,
        "d": d_total(d_s),
        "max_lag": max_lag_for_d_s(d_s),
        "n_obs": n_obs,
        "seed": seed,
        "true_params": np.asarray(TRUE_PARAMS).tolist(),
        "obs_hash": obs_hash,
        "num_reference_samples": num_reference_samples,
        "reference_method": "blackjax_adaptive_tempered_smc",
        "environment": environment_record(),
    }
    try:
        key = rng_for(seed, d_s, n_obs, num_reference_samples, "ma2-reference")
        key, subkey = random.split(key)
        u = random.uniform(subkey, shape=(num_reference_samples, D_THETA), minval=1e-6, maxval=1 - 1e-6)
        initial_particles = logit(u)
        hmc_parameters = {
            "step_size": jnp.full(num_reference_samples, smc_step_size),
            "inverse_mass_matrix": smc_inverse_mass * jnp.ones((num_reference_samples, D_THETA)),
            "num_integration_steps": jnp.full(num_reference_samples, smc_integration_steps),
        }
        tempered_smc = blackjax.adaptive_tempered_smc(
            log_prior_unbounded,
            lambda params: log_likelihood_unbounded(params, obs=jnp.asarray(obs), n_obs=n_obs, d_s=d_s),
            blackjax.hmc.build_kernel(),
            blackjax.hmc.init,
            hmc_parameters,
            resampling.systematic,
            smc_ess_threshold,
            num_mcmc_steps=smc_num_mcmc_steps,
        )
        initial_state = tempered_smc.init(initial_particles)
        key, subkey = random.split(key)
        n_iter, smc_state = smc_loop(subkey, tempered_smc.step, initial_state, smc_max_steps)
        if float(smc_state.lmbda) < 1.0:
            raise RuntimeError(f"SMC did not reach lambda=1 within {smc_max_steps} steps")
        particles = smc_state.particles
        samples = bounded_from_unbounded(particles)
        if not bool(jnp.isfinite(samples).all()):
            raise RuntimeError("reference samples contain non-finite values")
        tmp = cache.with_suffix(cache.suffix + ".tmp")
        np.savez_compressed(
            tmp,
            samples=np.asarray(samples),
            particles_unbounded=np.asarray(particles),
            observed_summary=np.asarray(obs),
            param_names=np.asarray(PARAM_NAMES),
        )
        if tmp.exists():
            tmp.replace(cache)
        else:
            tmp.with_suffix(tmp.suffix + ".npz").replace(cache)
        metadata.update(
            {
                "status": "complete",
                "elapsed_seconds": time.perf_counter() - start,
                "smc_iterations": n_iter,
                "final_lambda": float(smc_state.lmbda),
                "smc_settings": {
                    "step_size": smc_step_size,
                    "inverse_mass": smc_inverse_mass,
                    "integration_steps": smc_integration_steps,
                    "num_mcmc_steps": smc_num_mcmc_steps,
                    "ess_threshold": smc_ess_threshold,
                    "max_steps": smc_max_steps,
                },
            }
        )
        write_json(metadata_path, metadata)
        return cache
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "failure_reason": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        write_json(metadata_path, metadata)
        raise


def load_reference(output_root: Path | str, d_s: int, n_obs: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    cache = reference_cache_path(output_root, d_s, n_obs, seed)
    metadata_path = reference_metadata_path(cache)
    if not cache.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"missing MA2 reference cache or metadata: {cache}")
    metadata = load_json(metadata_path)
    if metadata.get("status") != "complete":
        raise RuntimeError(f"MA2 reference cache is not complete: {metadata_path}")
    obs = observed_summary(seed, n_obs, d_s)
    if metadata.get("obs_hash") != hash_array(obs):
        raise RuntimeError(f"MA2 reference obs_hash mismatch: {metadata_path}")
    payload = np.load(cache)
    samples = np.asarray(payload["samples"])
    if samples.ndim != 2 or samples.shape[1] != D_THETA:
        raise RuntimeError(f"invalid MA2 reference sample shape in {cache}: {samples.shape}")
    if not np.isfinite(samples).all():
        raise RuntimeError(f"MA2 reference cache contains non-finite samples: {cache}")
    return samples, metadata


def safe_standardise(arr: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    floored = (~np.isfinite(std)) | (std < eps)
    std = std.copy()
    std[floored] = 1.0
    return (arr - mean) / std, mean, std, floored


def simulate_training_data(
    *,
    seed: int,
    d_s: int,
    n_obs: int,
    n_sims: int,
    sim_batch_size: int,
) -> dict[str, np.ndarray]:
    theta_key = rng_for(seed, d_s, n_obs, n_sims, "theta-prior")
    sim_key = rng_for(seed, d_s, n_obs, n_sims, "summary-simulation")
    u = random.uniform(theta_key, shape=(n_sims, D_THETA), minval=1e-6, maxval=1 - 1e-6)
    theta_unbounded = logit(u)
    theta_bounded = bounded_from_unbounded(theta_unbounded)
    summaries = get_summaries_batches(
        sim_key,
        theta_bounded[:, 0],
        theta_bounded[:, 1],
        n_obs=n_obs,
        n_sims=n_sims,
        batch_size=min(sim_batch_size, n_sims),
        max_lag=max_lag_for_d_s(d_s),
    )
    return {
        "theta_bounded": np.asarray(theta_bounded),
        "theta_unbounded": np.asarray(theta_unbounded),
        "summaries": np.asarray(summaries),
    }


def fit_gaussian_npe(
    *,
    seed: int,
    d_s: int,
    n_obs: int,
    n_sims: int,
    theta_train: np.ndarray,
    summary_train: np.ndarray,
    learning_rate: float,
    train_batch_size: int,
    max_epochs: int,
    patience: int,
    val_frac: float,
) -> tuple[Any, dict[str, list[float]]]:
    from npe_convergence.methods.gaussian_npe import ConditionalGaussianNPE, TrainConfig, fit

    key = rng_for(seed, d_s, n_obs, n_sims, "gaussian-model")
    model = ConditionalGaussianNPE(
        d_summary=summary_train.shape[1],
        d_theta=theta_train.shape[1],
        hidden_dims=(128, 128),
        key=key,
    )
    fit_key = rng_for(seed, d_s, n_obs, n_sims, "gaussian-fit")
    model, losses = fit(
        model,
        jnp.asarray(theta_train),
        jnp.asarray(summary_train),
        key=fit_key,
        config=TrainConfig(
            lr=learning_rate,
            batch_size=train_batch_size,
            max_epochs=max_epochs,
            patience=patience,
            val_frac=val_frac,
        ),
    )
    return model, losses


def fit_flow_npe(
    *,
    seed: int,
    d_s: int,
    n_obs: int,
    n_sims: int,
    theta_train: np.ndarray,
    summary_train: np.ndarray,
    learning_rate: float,
    train_batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[Any, dict[str, list[float]]]:
    from flowjax.bijections import RationalQuadraticSpline  # type: ignore
    from flowjax.distributions import Normal  # type: ignore
    from flowjax.flows import coupling_flow  # type: ignore
    from flowjax.train.data_fit import fit_to_data  # type: ignore

    key = rng_for(seed, d_s, n_obs, n_sims, "flow-model")
    flow = coupling_flow(
        key=key,
        base_dist=Normal(jnp.zeros(theta_train.shape[1])),
        transformer=RationalQuadraticSpline(knots=10, interval=5),
        cond_dim=summary_train.shape[1],
        nn_depth=2,
    )
    fit_key = rng_for(seed, d_s, n_obs, n_sims, "flow-fit")
    flow, losses = fit_to_data(
        key=fit_key,
        dist=flow,
        x=jnp.asarray(theta_train),
        condition=jnp.asarray(summary_train),
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=patience,
        batch_size=train_batch_size,
    )
    return flow, losses


def sample_npe_posterior(
    *,
    method: str,
    model: Any,
    x_obs_std: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    seed: int,
    d_s: int,
    n_obs: int,
    n_sims: int,
    num_posterior_samples: int,
) -> np.ndarray:
    key = rng_for(seed, d_s, n_obs, n_sims, method, "posterior-samples")
    if method == "gaussian_npe":
        from npe_convergence.methods.gaussian_npe import sample

        posterior_standardised = sample(model, jnp.asarray(x_obs_std), num_posterior_samples, key=key)
    elif method == "flow_npe":
        posterior_standardised = model.sample(
            key,
            sample_shape=(num_posterior_samples,),
            condition=jnp.asarray(x_obs_std),
        )
    else:
        raise ValueError(f"unsupported method={method}")
    posterior_unbounded = posterior_standardised * jnp.asarray(theta_std) + jnp.asarray(theta_mean)
    posterior_bounded = bounded_from_unbounded(posterior_unbounded)
    return np.asarray(posterior_bounded)


def log_prob_standardised_theta(
    *,
    method: str,
    model: Any,
    theta_standardised: np.ndarray,
    x_obs_std: np.ndarray,
) -> np.ndarray:
    theta = jnp.asarray(theta_standardised)
    condition = jnp.asarray(x_obs_std)
    if method == "gaussian_npe":
        from npe_convergence.methods.gaussian_npe import gaussian_nll

        log_prob = -jax.vmap(lambda value: gaussian_nll(model, value, condition))(theta)
    elif method == "flow_npe":
        log_prob = jax.vmap(lambda value: model.log_prob(value, condition=condition))(theta)
    else:
        raise ValueError(f"unsupported method={method}")
    return np.asarray(log_prob)


def log_prob_bounded_theta(
    *,
    method: str,
    model: Any,
    theta_bounded: np.ndarray,
    x_obs_std: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    clipped = np.clip(np.asarray(theta_bounded, dtype=np.float64), -1.0 + eps, 1.0 - eps)
    y = (clipped + 1.0) / 2.0
    theta_unbounded = np.asarray(logit(jnp.asarray(y)))
    theta_standardised = (theta_unbounded - theta_mean) / theta_std
    log_prob_std = log_prob_standardised_theta(
        method=method,
        model=model,
        theta_standardised=theta_standardised,
        x_obs_std=x_obs_std,
    )
    log_standardise_jac = -float(np.sum(np.log(theta_std)))
    log_unbounded_jac = np.sum(-np.log(2.0) - np.log(y) - np.log1p(-y), axis=1)
    return np.asarray(log_prob_std, dtype=np.float64) + log_standardise_jac + log_unbounded_jac


def compute_cross_entropy(
    *,
    seed: int,
    d_s: int,
    n_obs: int,
    n_sims: int,
    method: str,
    model: Any,
    true_samples: np.ndarray,
    x_obs_std: np.ndarray,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    metric_samples: int,
) -> dict[str, float | int | bool | None | str]:
    metric_n = min(metric_samples, true_samples.shape[0])
    if metric_n < 2:
        return {
            "cross_entropy": None,
            "cross_entropy_samples": metric_n,
            "cross_entropy_finite": False,
            "cross_entropy_status": "too_few_samples",
        }
    key = rng_for(seed, d_s, n_obs, n_sims, method, "cross-entropy-reference")
    idx = np.asarray(random.permutation(key, true_samples.shape[0])[:metric_n])
    try:
        log_q = log_prob_bounded_theta(
            method=method,
            model=model,
            theta_bounded=true_samples[idx],
            x_obs_std=x_obs_std,
            theta_mean=theta_mean,
            theta_std=theta_std,
        )
    except Exception as exc:
        return {
            "cross_entropy": None,
            "cross_entropy_samples": metric_n,
            "cross_entropy_finite": False,
            "cross_entropy_status": f"log_prob_failed: {exc!r}",
        }
    finite = np.isfinite(log_q)
    if not bool(finite.all()):
        return {
            "cross_entropy": None,
            "cross_entropy_samples": metric_n,
            "cross_entropy_finite": False,
            "cross_entropy_nonfinite_count": int((~finite).sum()),
            "cross_entropy_status": "nonfinite_log_prob",
        }
    return {
        "cross_entropy": float(-np.mean(log_q)),
        "cross_entropy_samples": metric_n,
        "cross_entropy_finite": True,
        "cross_entropy_status": "complete",
        "log_density_space": "bounded_theta",
    }


def compute_metrics(
    *,
    seed: int,
    d_s: int,
    n_obs: int,
    n_sims: int,
    method: str,
    true_samples: np.ndarray,
    posterior_samples: np.ndarray,
    metric_samples: int,
) -> dict[str, float | int]:
    metric_n = min(metric_samples, true_samples.shape[0], posterior_samples.shape[0])
    if metric_n < 2:
        raise ValueError(f"need at least 2 samples per side for metrics, got {metric_n}")
    key_true = rng_for(seed, d_s, n_obs, n_sims, method, "metrics-true")
    key_post = rng_for(seed, d_s, n_obs, n_sims, method, "metrics-posterior")
    idx_true = np.asarray(random.permutation(key_true, true_samples.shape[0])[:metric_n])
    idx_post = np.asarray(random.permutation(key_post, posterior_samples.shape[0])[:metric_n])
    ts_thin = jnp.asarray(true_samples[idx_true])
    ps_thin = jnp.asarray(posterior_samples[idx_post])
    kl = float(kullback_leibler(ts_thin, ps_thin))
    lengthscale = median_heuristic(jnp.vstack([ts_thin, ps_thin]))
    mmd = float(unbiased_mmd(ts_thin, ps_thin, lengthscale))
    return {
        "metric_samples": metric_n,
        "kl_theta_knn_2000": kl,
        "mmd_theta_2000": mmd,
        "mmd_lengthscale": float(lengthscale),
    }


def write_losses(losses: dict[str, list[float]], output_dir: Path) -> None:
    write_json(output_dir / "losses.json", losses)
    if "train" not in losses or "val" not in losses:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses["train"], label="train")
    ax.plot(losses["val"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "losses.pdf")
    plt.close(fig)


def run_cell(
    *,
    output_root: Path | str,
    d_s: int,
    method: str,
    seed: int,
    n_obs: int,
    n_sims: int,
    sim_batch_size: int,
    num_posterior_samples: int,
    metric_samples: int,
    learning_rate: float,
    train_batch_size: int,
    max_epochs: int,
    patience: int,
    val_frac: float,
    force: bool,
) -> Path:
    validate_d_s(d_s)
    if method not in METHODS:
        raise ValueError(f"unsupported method={method}")
    spec = CellSpec(d_s=d_s, method=method, seed=seed, n_obs=n_obs, n_sims=n_sims)
    out = cell_dir(output_root, spec)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite existing cell directory: {out}")
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    diagnostics: dict[str, Any] = {
        "status": "started",
        "created_at_utc": utc_now(),
        "d_s": d_s,
        "d_theta": D_THETA,
        "d": spec.d,
        "max_lag": max_lag_for_d_s(d_s),
        "n_obs": n_obs,
        "n_sims": n_sims,
        "scaled_budget": spec.scaled_budget,
        "method": method,
        "seed": seed,
        "true_params": np.asarray(TRUE_PARAMS).tolist(),
        "environment": environment_record(),
    }
    start = time.perf_counter()
    try:
        true_samples, reference_metadata = load_reference(output_root, d_s, n_obs, seed)
        x_obs_raw = observed_summary(seed, n_obs, d_s)
        diagnostics["obs_hash"] = hash_array(x_obs_raw)
        diagnostics["reference_cache"] = str(reference_cache_path(output_root, d_s, n_obs, seed))
        diagnostics["reference_metadata"] = reference_metadata
        np.save(out / "x_obs.npy", x_obs_raw)

        sim_start = time.perf_counter()
        training_data = simulate_training_data(
            seed=seed,
            d_s=d_s,
            n_obs=n_obs,
            n_sims=n_sims,
            sim_batch_size=sim_batch_size,
        )
        diagnostics["simulation_time_seconds"] = time.perf_counter() - sim_start
        theta_unbounded = training_data["theta_unbounded"]
        summaries = training_data["summaries"]
        row_mask = np.isfinite(theta_unbounded).all(axis=1) & np.isfinite(summaries).all(axis=1)
        theta_unbounded = theta_unbounded[row_mask]
        summaries = summaries[row_mask]
        if theta_unbounded.shape[0] < 2:
            raise RuntimeError("fewer than two finite simulation rows")

        theta_train, theta_mean, theta_std, theta_floored = safe_standardise(theta_unbounded)
        summary_train, summary_mean, summary_std, summary_floored = safe_standardise(summaries)
        x_obs_std = (x_obs_raw - summary_mean) / summary_std
        np.savez_compressed(
            out / "training_standardisation.npz",
            theta_mean=theta_mean,
            theta_std=theta_std,
            summary_mean=summary_mean,
            summary_std=summary_std,
            x_obs_std=x_obs_std,
            finite_row_mask=row_mask,
        )
        diagnostics["training_rows_raw"] = int(row_mask.shape[0])
        diagnostics["training_rows_finite"] = int(row_mask.sum())
        diagnostics["dropped_nonfinite_rows"] = int(row_mask.shape[0] - row_mask.sum())
        diagnostics["theta_std_floored_count"] = int(theta_floored.sum())
        diagnostics["summary_std_floored_count"] = int(summary_floored.sum())

        train_start = time.perf_counter()
        if method == "gaussian_npe":
            model, losses = fit_gaussian_npe(
                seed=seed,
                d_s=d_s,
                n_obs=n_obs,
                n_sims=n_sims,
                theta_train=theta_train,
                summary_train=summary_train,
                learning_rate=learning_rate,
                train_batch_size=train_batch_size,
                max_epochs=max_epochs,
                patience=patience,
                val_frac=val_frac,
            )
        else:
            model, losses = fit_flow_npe(
                seed=seed,
                d_s=d_s,
                n_obs=n_obs,
                n_sims=n_sims,
                theta_train=theta_train,
                summary_train=summary_train,
                learning_rate=learning_rate,
                train_batch_size=train_batch_size,
                max_epochs=max_epochs,
                patience=patience,
            )
        diagnostics["training_time_seconds"] = time.perf_counter() - train_start
        diagnostics["epochs"] = len(losses.get("train", []))
        write_losses(losses, out)

        sample_start = time.perf_counter()
        posterior_samples = sample_npe_posterior(
            method=method,
            model=model,
            x_obs_std=x_obs_std,
            theta_mean=theta_mean,
            theta_std=theta_std,
            seed=seed,
            d_s=d_s,
            n_obs=n_obs,
            n_sims=n_sims,
            num_posterior_samples=num_posterior_samples,
        )
        diagnostics["posterior_sampling_time_seconds"] = time.perf_counter() - sample_start
        np.savez_compressed(out / "posterior_samples.npz", theta=posterior_samples)

        metrics = compute_metrics(
            seed=seed,
            d_s=d_s,
            n_obs=n_obs,
            n_sims=n_sims,
            method=method,
            true_samples=true_samples,
            posterior_samples=posterior_samples,
            metric_samples=metric_samples,
        )
        metrics.update(
            compute_cross_entropy(
                seed=seed,
                d_s=d_s,
                n_obs=n_obs,
                n_sims=n_sims,
                method=method,
                model=model,
                true_samples=true_samples,
                x_obs_std=x_obs_std,
                theta_mean=theta_mean,
                theta_std=theta_std,
                metric_samples=metric_samples,
            )
        )
        kl = float(metrics["kl_theta_knn_2000"])
        mmd = float(metrics["mmd_theta_2000"])
        (out / "kl.txt").write_text(f"{kl}\n")
        (out / "mmd.txt").write_text(f"{mmd}\n")
        write_json(out / "metrics.json", metrics)
        diagnostics["metrics"] = metrics
        diagnostics["total_wall_time_seconds"] = time.perf_counter() - start
        diagnostics["status"] = "complete" if math.isfinite(kl) and math.isfinite(mmd) else "nonfinite_metric"
        write_json(out / "diagnostics.json", diagnostics)
        if diagnostics["status"] != "complete":
            raise RuntimeError(f"non-finite metric for {out}")
        return out
    except Exception as exc:
        diagnostics["status"] = "failed"
        diagnostics["failure_reason"] = repr(exc)
        diagnostics["traceback"] = traceback.format_exc()
        diagnostics["total_wall_time_seconds"] = time.perf_counter() - start
        write_json(out / "diagnostics.json", diagnostics)
        raise


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def read_float_file(path: Path) -> float | None:
    try:
        return finite_float(path.read_text().strip())
    except Exception:
        return None


def cell_row_from_dir(path: Path) -> dict[str, Any] | None:
    match = CELL_RE.match(path.name)
    if not match:
        return None
    row: dict[str, Any] = match.groupdict()
    row["path"] = str(path)
    row["d_s"] = int(row["d_s"])
    row["d"] = d_total(row["d_s"])
    row["n_obs"] = int(row["n_obs"])
    row["n_sims"] = int(row["n_sims"])
    row["seed"] = int(row["seed"])
    row["scaled_budget"] = row["n_sims"] / (row["d"] * row["d"] * row["n_obs"])
    diagnostics_path = path / "diagnostics.json"
    diagnostics = load_json(diagnostics_path) if diagnostics_path.exists() else {}
    metrics_path = path / "metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    row["status"] = diagnostics.get("status", "missing_diagnostics")
    row["failure_reason"] = diagnostics.get("failure_reason")
    row["obs_hash"] = diagnostics.get("obs_hash")
    row["wall_time_seconds"] = diagnostics.get("total_wall_time_seconds")
    row["training_time_seconds"] = diagnostics.get("training_time_seconds")
    row["simulation_time_seconds"] = diagnostics.get("simulation_time_seconds")
    row["theta_kl"] = finite_float(metrics.get("kl_theta_knn_2000", read_float_file(path / "kl.txt")))
    row["mmd"] = finite_float(metrics.get("mmd_theta_2000", read_float_file(path / "mmd.txt")))
    row["cross_entropy"] = finite_float(metrics.get("cross_entropy"))
    row["finite_metric"] = row["theta_kl"] is not None and row["mmd"] is not None
    if row["status"] == "complete" and not row["finite_metric"]:
        row["status"] = "nonfinite_metric"
    return row


def information_matrix_diagnostics(n_obs: int) -> list[dict[str, Any]]:
    rows = []
    theta = TRUE_PARAMS
    jac = jnp.array(
        [
            [2.0 * theta[0], 2.0 * theta[1]],
            [1.0 + theta[1], theta[0]],
            [0.0, 1.0],
        ]
    )
    for d_s in D_S_GRID:
        if d_s > 3:
            jac_full = jnp.vstack([jac, jnp.zeros((d_s - 3, D_THETA))])
        else:
            jac_full = jac[:d_s]
        cov = covariance_summary(theta, n_obs, d_s)
        info = jac_full.T @ jnp.linalg.solve(cov, jac_full)
        eig = np.asarray(jnp.linalg.eigvalsh(info))
        rows.append(
            {
                "d_s": d_s,
                "d": d_total(d_s),
                "max_lag": max_lag_for_d_s(d_s),
                "min_eigenvalue": float(np.min(eig)),
                "max_eigenvalue": float(np.max(eig)),
                "condition_number": float(np.max(eig) / np.min(eig)),
                "warning": bool(np.min(eig) <= 1e-8 or np.max(eig) / np.min(eig) >= 1e8),
            }
        )
    return rows


def manifest_rows(
    *,
    num_seeds: int,
    n_obs: int,
    scales: tuple[int, ...],
    max_n_sims: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    reference_rows = []
    cell_rows = []
    skipped_rows = []
    for seed in range(num_seeds):
        for d_s in D_S_GRID:
            reference_rows.append(
                {
                    "array_index": len(reference_rows),
                    "kind": "reference",
                    "d_s": d_s,
                    "d": d_total(d_s),
                    "max_lag": max_lag_for_d_s(d_s),
                    "n_obs": n_obs,
                    "n_sims": 0,
                    "seed": seed,
                    "method": "",
                    "scaled_budget": "",
                }
            )
    for seed in range(num_seeds):
        for d_s in D_S_GRID:
            d = d_total(d_s)
            for scale in scales:
                n_sims = int(scale * d * d * n_obs)
                for method in METHODS:
                    row = {
                        "array_index": len(cell_rows),
                        "kind": "cell",
                        "d_s": d_s,
                        "d": d,
                        "max_lag": max_lag_for_d_s(d_s),
                        "n_obs": n_obs,
                        "n_sims": n_sims,
                        "seed": seed,
                        "method": method,
                        "scaled_budget": scale,
                    }
                    if n_sims > max_n_sims:
                        skipped = dict(row)
                        skipped["array_index"] = ""
                        skipped["reason"] = f"n_sims {n_sims} exceeds cap {max_n_sims}"
                        skipped_rows.append(skipped)
                    else:
                        row["array_index"] = len(cell_rows)
                        cell_rows.append(row)
    return reference_rows, cell_rows, skipped_rows


def prepare_manifests(
    *,
    output_root: Path | str,
    n_obs: int,
    scales: tuple[int, ...],
    num_seeds: int,
    max_n_sims: int,
    timestamp: str | None,
) -> dict[str, Path]:
    root = resolve_path(output_root)
    stamp = compact_timestamp(timestamp)
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    reference_rows, cell_rows, skipped_rows = manifest_rows(
        num_seeds=num_seeds,
        n_obs=n_obs,
        scales=scales,
        max_n_sims=max_n_sims,
    )
    fields = [
        "array_index",
        "kind",
        "d_s",
        "d",
        "max_lag",
        "n_obs",
        "n_sims",
        "seed",
        "method",
        "scaled_budget",
    ]
    skipped_fields = [*fields, "reason"]
    ref_csv = manifest_dir / f"ma2_dim_scaling_reference_manifest_{stamp}.csv"
    cells_csv = manifest_dir / f"ma2_dim_scaling_cells_manifest_{stamp}.csv"
    skipped_csv = manifest_dir / "skipped_cells.csv"
    info_csv = manifest_dir / "information_matrix_diagnostics.csv"
    write_csv(ref_csv, reference_rows, fields)
    write_csv(cells_csv, cell_rows, fields)
    write_csv(skipped_csv, skipped_rows, skipped_fields)
    info_rows = information_matrix_diagnostics(n_obs)
    write_csv(
        info_csv,
        info_rows,
        ["d_s", "d", "max_lag", "min_eigenvalue", "max_eigenvalue", "condition_number", "warning"],
    )
    summary_json = manifest_dir / f"ma2_dim_scaling_manifest_summary_{stamp}.json"
    write_json(
        summary_json,
        {
            "created_at_utc": utc_now(),
            "n_obs": n_obs,
            "scales": list(scales),
            "num_seeds": num_seeds,
            "max_n_sims": max_n_sims,
            "d_s_grid": list(D_S_GRID),
            "methods": list(METHODS),
            "reference_rows": len(reference_rows),
            "cell_rows": len(cell_rows),
            "skipped_rows": len(skipped_rows),
            "reference_manifest": str(ref_csv),
            "cells_manifest": str(cells_csv),
            "skipped_cells": str(skipped_csv),
            "information_matrix_diagnostics": info_rows,
            "environment": environment_record(),
        },
    )
    return {
        "reference_manifest": ref_csv,
        "cells_manifest": cells_csv,
        "skipped_cells": skipped_csv,
        "information_matrix_diagnostics": info_csv,
        "summary": summary_json,
    }


def aggregate_overnight_csv(
    *,
    output_root: Path | str,
    output_csv: Path | str,
    staged_csv: Path | str | None,
) -> Path:
    root = resolve_path(output_root)
    rows = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if not path.is_dir():
            continue
        row = cell_row_from_dir(path)
        if row is None:
            continue
        rows.append(
            {
                "method": row["method"],
                "seed": row["seed"],
                "n_obs": row["n_obs"],
                "d_s": row["d_s"],
                "d": row["d"],
                "N_sims": row["n_sims"],
                "N_over_d2n": row["scaled_budget"],
                "theta_kl": row["theta_kl"],
                "cross_entropy": row["cross_entropy"],
                "status": row["status"],
            }
        )
    fields = [
        "method",
        "seed",
        "n_obs",
        "d_s",
        "d",
        "N_sims",
        "N_over_d2n",
        "theta_kl",
        "cross_entropy",
        "status",
    ]
    out = resolve_path(output_csv)
    write_csv(out, rows, fields)
    if staged_csv is not None:
        staged = resolve_path(staged_csv)
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, staged)
    return out


def load_manifest_row(manifest: Path | str, array_index: int, kind: str | None) -> dict[str, str]:
    path = resolve_path(manifest)
    with path.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if int(row["array_index"]) == array_index]
    if len(rows) != 1:
        raise ValueError(f"array_index={array_index} matched {len(rows)} rows in {path}")
    row = rows[0]
    if kind is not None and row.get("kind") != kind:
        raise ValueError(f"manifest row kind={row.get('kind')} does not match requested kind={kind}")
    return row


def parse_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-s", "--d_s", dest="d_s", type=int, choices=D_S_GRID, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-obs", "--n_obs", dest="n_obs", type=int, default=DEFAULT_N_OBS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def add_reference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-reference-samples", type=int, default=10_000)
    parser.add_argument("--smc-step-size", type=float, default=5e-3)
    parser.add_argument("--smc-inverse-mass", type=float, default=0.1)
    parser.add_argument("--smc-integration-steps", type=int, default=100)
    parser.add_argument("--smc-num-mcmc-steps", type=int, default=5)
    parser.add_argument("--smc-ess-threshold", type=float, default=0.75)
    parser.add_argument("--smc-max-steps", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")


def add_training_args(parser: argparse.ArgumentParser, *, include_force: bool = True) -> None:
    parser.add_argument("--n-sims", "--n_sims", dest="n_sims", type=int)
    parser.add_argument("--sim-batch-size", type=int, default=1000)
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--metric-samples", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.1)
    if include_force:
        parser.add_argument("--force", action="store_true")


def cmd_precompute_reference(args: argparse.Namespace) -> None:
    precompute_reference(
        output_root=args.output_root,
        d_s=args.d_s,
        seed=args.seed,
        n_obs=args.n_obs,
        num_reference_samples=args.num_reference_samples,
        smc_step_size=args.smc_step_size,
        smc_inverse_mass=args.smc_inverse_mass,
        smc_integration_steps=args.smc_integration_steps,
        smc_num_mcmc_steps=args.smc_num_mcmc_steps,
        smc_ess_threshold=args.smc_ess_threshold,
        smc_max_steps=args.smc_max_steps,
        force=args.force,
    )


def cmd_run_cell(args: argparse.Namespace) -> None:
    n_sims = args.n_sims if args.n_sims is not None else scaled_n_sims(args.d_s, args.n_obs, 5)
    run_cell(
        output_root=args.output_root,
        d_s=args.d_s,
        method=args.method,
        seed=args.seed,
        n_obs=args.n_obs,
        n_sims=n_sims,
        sim_batch_size=args.sim_batch_size,
        num_posterior_samples=args.num_posterior_samples,
        metric_samples=args.metric_samples,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_frac=args.val_frac,
        force=args.force,
    )


def cmd_prepare_manifests(args: argparse.Namespace) -> None:
    outputs = prepare_manifests(
        output_root=args.output_root,
        n_obs=args.n_obs,
        scales=tuple(args.scales),
        num_seeds=args.num_seeds,
        max_n_sims=args.max_n_sims,
        timestamp=args.timestamp,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def cmd_run_manifest_row(args: argparse.Namespace) -> None:
    row = load_manifest_row(args.manifest, args.array_index, args.kind)
    d_s = int(row["d_s"])
    seed = int(row["seed"])
    n_obs = int(row["n_obs"])
    if row["kind"] == "reference":
        precompute_reference(
            output_root=args.output_root,
            d_s=d_s,
            seed=seed,
            n_obs=n_obs,
            num_reference_samples=args.num_reference_samples,
            smc_step_size=args.smc_step_size,
            smc_inverse_mass=args.smc_inverse_mass,
            smc_integration_steps=args.smc_integration_steps,
            smc_num_mcmc_steps=args.smc_num_mcmc_steps,
            smc_ess_threshold=args.smc_ess_threshold,
            smc_max_steps=args.smc_max_steps,
            force=args.force,
        )
    elif row["kind"] == "cell":
        run_cell(
            output_root=args.output_root,
            d_s=d_s,
            method=row["method"],
            seed=seed,
            n_obs=n_obs,
            n_sims=int(row["n_sims"]),
            sim_batch_size=args.sim_batch_size,
            num_posterior_samples=args.num_posterior_samples,
            metric_samples=args.metric_samples,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            val_frac=args.val_frac,
            force=args.force,
        )
    else:
        raise ValueError(f"unsupported manifest row kind={row['kind']}")


def cmd_aggregate_overnight(args: argparse.Namespace) -> None:
    path = aggregate_overnight_csv(
        output_root=args.output_root,
        output_csv=args.output_csv,
        staged_csv=args.staged_csv,
    )
    print(f"aggregated_csv: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("precompute-reference", help="Build one MA2 SMC reference cache.")
    parse_common_args(p)
    add_reference_args(p)
    p.set_defaults(func=cmd_precompute_reference)

    p = sub.add_parser("run-cell", help="Train and evaluate one MA2 dimension-scaling cell.")
    parse_common_args(p)
    p.add_argument("--method", choices=METHODS, required=True)
    add_training_args(p)
    p.set_defaults(func=cmd_run_cell)

    p = sub.add_parser("prepare-manifests", help="Write MA2 c-sweep PBS array manifests.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--n-obs", "--n_obs", dest="n_obs", type=int, default=DEFAULT_N_OBS)
    p.add_argument("--scales", nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--num-seeds", type=int, default=8)
    p.add_argument("--max-n-sims", type=int, default=6_000_000)
    p.add_argument("--timestamp")
    p.set_defaults(func=cmd_prepare_manifests)

    p = sub.add_parser("run-manifest-row", help="Run one row from a prepared MA2 manifest.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--kind", choices=("reference", "cell"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    add_reference_args(p)
    add_training_args(p, include_force=False)
    p.set_defaults(func=cmd_run_manifest_row)

    p = sub.add_parser("aggregate-overnight", help="Write the Section 3.3 MA2 aggregate CSV.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--staged-csv", type=Path)
    p.set_defaults(func=cmd_aggregate_overnight)

    return parser


def main(argv: list[str] | None = None) -> None:
    numpyro.set_host_device_count(4)
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "array_index", None) is None and os.environ.get("PBS_ARRAY_INDEX") is not None:
        args.array_index = int(os.environ["PBS_ARRAY_INDEX"])
    args.func(args)


if __name__ == "__main__":
    main()
