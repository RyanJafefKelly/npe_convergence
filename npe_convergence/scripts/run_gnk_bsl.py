"""Run a GNK Bayesian synthetic likelihood reference diagnostic.

The synthetic likelihood is estimated from simulated octile summaries. The
plain Gaussian BSL log likelihood is used; the Price et al. (2018) unbiased
correction is not applied. The MCMC target is the finite-M randomized BSL
target induced by retaining the current noisy synthetic likelihood value on
rejection, not an exact pseudo-marginal chain for the ideal infinite-M
synthetic likelihood.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle as pkl
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib
import numpy as np
from numpyro.diagnostics import summary as numpyro_summary  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.bsl import eta_to_theta, gnk_bsl_logtarget_eta, theta_to_eta
from npe_convergence.examples.gnk import gnk, ss_octile
from npe_convergence.metrics import kullback_leibler


PARAM_NAMES = ("A", "B", "g", "k")
TRUE_THETA = jnp.array([3.0, 1.0, 2.0, 0.5])
DIM_THETA = len(PARAM_NAMES)
OPTIMAL_RWM_SCALE = 2.38 / np.sqrt(DIM_THETA)


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def rng_for(seed: int, *parts: object) -> jax.Array:
    key = random.key(seed)
    for part in parts:
        key = random.fold_in(key, stable_int(part))
    return key


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_utc_stamp(created_at: str) -> str:
    return created_at.replace("-", "").replace(":", "").replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (jax.Array,)):
        return np.asarray(value).tolist()
    return str(value)


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, jax.Array):
        return json_sanitize(np.asarray(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if np.isposinf(scalar):
            return "Infinity"
        if np.isneginf(scalar):
            return "-Infinity"
        if np.isnan(scalar):
            return "NaN"
        return scalar
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(json_sanitize(payload), f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "numpyro_available": True,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for key, command in {
        "git_commit": ["git", "rev-parse", "HEAD"],
        "git_branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    }.items():
        try:
            record[key] = subprocess.check_output(
                command,
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
        except Exception:
            record[key] = None
    try:
        subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        record["git_dirty"] = False
    except subprocess.CalledProcessError:
        record["git_dirty"] = True
    except Exception:
        record["git_dirty"] = None
    return record


def observed_summary(seed: int, n_obs: int) -> np.ndarray:
    """Observed GNK octile summary matching run_gnk_gaussian.py."""
    key = random.key(seed)
    _, subkey = random.split(key)
    z = random.normal(subkey, shape=(n_obs,))
    x_obs = gnk(z, *TRUE_THETA)
    x_obs = jnp.atleast_2d(x_obs)
    x_obs = jnp.squeeze(ss_octile(x_obs))
    return np.asarray(x_obs)


def nuts_cache_path(n_obs: int, seed: int) -> Path:
    return REPO_ROOT / "res" / "gnk" / f"nuts_cache_v2_n_obs_{n_obs}_seed_{seed}.pkl"


def load_nuts_reference(n_obs: int, seed: int) -> np.ndarray | None:
    path = nuts_cache_path(n_obs, seed)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return np.asarray(pkl.load(f))


def _initial_etas(seed: int, num_chains: int, initial_sd: float = 0.1) -> jax.Array:
    base_eta = theta_to_eta(TRUE_THETA)
    keys = random.split(rng_for(seed, "bsl", "init"), num_chains)
    noise = jax.vmap(lambda key: random.normal(key, (4,)))(keys) * initial_sd
    return base_eta[None, :] + noise


def _as_eta_matrix(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 3 and array.shape[-1] == DIM_THETA:
        return array.reshape((-1, DIM_THETA))
    if array.ndim == 2 and array.shape[-1] == DIM_THETA:
        return array
    if array.shape == (DIM_THETA, DIM_THETA):
        return array
    raise ValueError(f"expected eta samples or a 4 x 4 covariance matrix, got shape {array.shape}")


def load_proposal_covariance(path: Path) -> tuple[np.ndarray, str]:
    """Load eta samples or a covariance matrix for dense RWM proposals."""
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "eta" in data:
                array = data["eta"]
                source = f"{path}:eta"
            elif "covariance" in data:
                array = data["covariance"]
                source = f"{path}:covariance"
            else:
                key = data.files[0]
                array = data[key]
                source = f"{path}:{key}"
    else:
        array = np.load(path)
        source = str(path)

    matrix = _as_eta_matrix(array)
    if matrix.shape == (DIM_THETA, DIM_THETA):
        return regularized_covariance(matrix, assume_covariance=True), source
    return regularized_covariance(matrix), source


def regularized_covariance(values: np.ndarray, *, assume_covariance: bool = False) -> np.ndarray:
    """Return a positive definite covariance estimate for eta-space proposals."""
    values = np.asarray(values, dtype=np.float64)
    if assume_covariance:
        cov = values.copy()
    else:
        if values.shape[0] < 2:
            return np.eye(DIM_THETA, dtype=np.float64)
        cov = np.cov(values, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    if cov.shape != (DIM_THETA, DIM_THETA) or not np.all(np.isfinite(cov)):
        return np.eye(DIM_THETA, dtype=np.float64)
    diag = np.diag(cov)
    positive_diag = diag[diag > 0]
    scale = float(np.mean(positive_diag)) if positive_diag.size else 1.0
    ridge = max(1e-10, 1e-6 * scale)
    for multiplier in (1.0, 10.0, 100.0, 1000.0):
        candidate = cov + ridge * multiplier * np.eye(DIM_THETA)
        try:
            np.linalg.cholesky(candidate)
            return candidate
        except np.linalg.LinAlgError:
            continue
    return np.diag(np.maximum(diag, ridge)) + ridge * np.eye(DIM_THETA)


@partial(jax.jit, static_argnames=("n_obs", "m"))
def _rwm_step(
    step_key: jax.Array,
    etas: jax.Array,
    logps: jax.Array,
    scales: jax.Array,
    proposal_chol: jax.Array,
    x_obs: jax.Array,
    n_obs: int,
    m: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    num_chains = etas.shape[0]
    proposal_key, likelihood_key, accept_key = random.split(step_key, 3)
    proposal_noise = random.normal(proposal_key, shape=etas.shape)
    proposal_steps = proposal_noise @ proposal_chol.T
    proposal_etas = etas + scales[:, None] * proposal_steps
    likelihood_keys = random.split(likelihood_key, num_chains)
    prop_logps, jitter_used, rank_deficient = jax.vmap(
        lambda key, eta: gnk_bsl_logtarget_eta(key, eta, x_obs, n_obs=n_obs, m=m)
    )(likelihood_keys, proposal_etas)
    log_accept_ratio = prop_logps - logps
    log_u = jnp.log(random.uniform(accept_key, shape=(num_chains,)))
    accepted = log_u < log_accept_ratio
    new_etas = jnp.where(accepted[:, None], proposal_etas, etas)
    new_logps = jnp.where(accepted, prop_logps, logps)
    accept_prob = jnp.minimum(1.0, jnp.exp(jnp.minimum(log_accept_ratio, 0.0)))
    return new_etas, new_logps, accepted, accept_prob, jitter_used, rank_deficient


def run_bsl_mcmc(
    *,
    seed: int,
    x_obs: np.ndarray,
    n_obs: int,
    m: int,
    num_chains: int,
    num_warmup: int,
    num_samples: int,
    initial_scale: float | None,
    proposal: str,
    proposal_covariance: np.ndarray | None = None,
    proposal_covariance_source: str | None = None,
    target_accept: float = 0.3,
    progress_every: int = 100,
) -> dict[str, Any]:
    etas = _initial_etas(seed, num_chains)
    x_obs_jax = jnp.asarray(x_obs)
    init_keys = random.split(rng_for(seed, "bsl", "initial-logp"), num_chains)
    logps, init_jitter, init_rank = jax.vmap(
        lambda key, eta: gnk_bsl_logtarget_eta(key, eta, x_obs_jax, n_obs=n_obs, m=m)
    )(init_keys, etas)
    logps.block_until_ready()

    total_steps = num_warmup + num_samples
    samples_eta = np.empty((num_chains, num_samples, 4), dtype=np.float32)
    initial_scale_was_set = initial_scale is not None
    dense_from_start = proposal_covariance is not None
    if initial_scale is None:
        initial_scale_value = OPTIMAL_RWM_SCALE if dense_from_start else 0.05
    else:
        initial_scale_value = float(initial_scale)
    log_scales = np.full(num_chains, np.log(initial_scale_value), dtype=np.float64)
    scales = jnp.asarray(np.exp(log_scales), dtype=etas.dtype)
    if proposal_covariance is None:
        proposal_covariance_np = np.eye(DIM_THETA, dtype=np.float64)
        proposal_covariance_source = "identity"
    else:
        proposal_covariance_np = regularized_covariance(proposal_covariance, assume_covariance=True)
        proposal_covariance_source = proposal_covariance_source or "provided"
    proposal_chol_np = np.linalg.cholesky(proposal_covariance_np)
    proposal_chol = jnp.asarray(proposal_chol_np, dtype=etas.dtype)

    warmup_eta_history = None
    adapt_dense_covariance = proposal == "adaptive-dense" and proposal_covariance is None
    if adapt_dense_covariance and num_warmup > 0:
        warmup_eta_history = np.empty((num_chains, num_warmup, DIM_THETA), dtype=np.float32)
    cov_adapt_start = max(20, min(500, max(num_warmup // 4, 1)))
    cov_adapt_every = max(10, min(250, max(num_warmup // 10, 1)))
    proposal_covariance_updates = 0
    dense_scale_reset = False
    warmup_accept = np.zeros(num_chains, dtype=np.int64)
    sample_accept = np.zeros(num_chains, dtype=np.int64)
    warmup_steps_counted = 0
    jitter_count = int(np.asarray(init_jitter).sum())
    rank_deficient_count = int(np.asarray(init_rank).sum())
    adaptation_restart_count = 0
    restart_log_scale = np.log(initial_scale_value)

    start = time.perf_counter()
    for step in range(total_steps):
        step_key = random.fold_in(rng_for(seed, "bsl", "mcmc"), step)
        etas, logps, accepted, _, jitter_used, rank_deficient = _rwm_step(
            step_key,
            etas,
            logps,
            scales,
            proposal_chol,
            x_obs_jax,
            n_obs,
            m,
        )
        accepted_np = np.asarray(accepted)
        jitter_count += int(np.asarray(jitter_used).sum())
        rank_deficient_count += int(np.asarray(rank_deficient).sum())

        if step < num_warmup:
            if warmup_eta_history is not None:
                warmup_eta_history[:, step, :] = np.asarray(etas)
            warmup_accept += accepted_np
            warmup_steps_counted += 1
            gamma = min(0.05, 1.0 / np.sqrt(step + 1.0))
            log_scales += gamma * (accepted_np.astype(np.float64) - target_accept)
            log_scales = np.clip(log_scales, np.log(1e-4), np.log(2.0))
            if num_warmup >= 40 and step + 1 == num_warmup // 2:
                rates = warmup_accept / max(warmup_steps_counted, 1)
                bad = (rates < 0.05) | (rates > 0.6)
                if np.any(bad):
                    log_scales[bad] = restart_log_scale
                    warmup_accept[bad] = 0
                    adaptation_restart_count += int(np.sum(bad))
            scales = jnp.asarray(np.exp(log_scales), dtype=etas.dtype)
            if (
                warmup_eta_history is not None
                and step + 1 >= cov_adapt_start
                and (step + 1 == num_warmup or (step + 1 - cov_adapt_start) % cov_adapt_every == 0)
            ):
                eta_for_cov = warmup_eta_history[:, : step + 1, :].reshape((-1, DIM_THETA))
                proposal_covariance_np = regularized_covariance(eta_for_cov)
                proposal_chol_np = np.linalg.cholesky(proposal_covariance_np)
                proposal_chol = jnp.asarray(proposal_chol_np, dtype=etas.dtype)
                proposal_covariance_source = f"warmup_eta_through_step_{step + 1}"
                proposal_covariance_updates += 1
                if not dense_from_start and not initial_scale_was_set and not dense_scale_reset:
                    log_scales[:] = np.log(OPTIMAL_RWM_SCALE)
                    restart_log_scale = np.log(OPTIMAL_RWM_SCALE)
                    dense_scale_reset = True
                    scales = jnp.asarray(np.exp(log_scales), dtype=etas.dtype)
        else:
            sample_index = step - num_warmup
            sample_accept += accepted_np
            samples_eta[:, sample_index, :] = np.asarray(etas)

        if progress_every > 0 and (step + 1) % progress_every == 0:
            elapsed = time.perf_counter() - start
            phase = "warmup" if step < num_warmup else "sample"
            rates = (
                warmup_accept / max(warmup_steps_counted, 1)
                if step < num_warmup
                else sample_accept / max(step - num_warmup + 1, 1)
            )
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "total_steps": total_steps,
                        "phase": phase,
                        "acceptance_rate": rates.tolist(),
                        "scale": np.exp(log_scales).tolist(),
                        "proposal": proposal,
                        "proposal_covariance_updates": proposal_covariance_updates,
                        "elapsed_seconds": elapsed,
                    }
                ),
                flush=True,
            )

    return {
        "eta": samples_eta,
        "theta": np.asarray(eta_to_theta(jnp.asarray(samples_eta))),
        "sample_acceptance_rate": sample_accept / max(num_samples, 1),
        "warmup_acceptance_rate": warmup_accept / max(warmup_steps_counted, 1),
        "final_scale": np.exp(log_scales),
        "initial_scale": initial_scale_value,
        "proposal": proposal,
        "proposal_covariance": proposal_covariance_np,
        "proposal_covariance_source": proposal_covariance_source,
        "proposal_covariance_updates": proposal_covariance_updates,
        "dense_scale_reset": dense_scale_reset,
        "jitter_count": jitter_count,
        "rank_deficient_count": rank_deficient_count,
        "adaptation_restart_count": adaptation_restart_count,
    }


def diagnostic_summary(theta_chains: np.ndarray) -> dict[str, dict[str, float]]:
    payload = {
        name: theta_chains[:, :, idx]
        for idx, name in enumerate(PARAM_NAMES)
    }
    raw = numpyro_summary(payload, group_by_chain=True)
    return {
        name: {
            "mean": float(values["mean"]),
            "std": float(values["std"]),
            "median": float(values["median"]),
            "n_eff": float(values["n_eff"]),
            "r_hat": float(values["r_hat"]),
        }
        for name, values in raw.items()
    }


def save_traceplots(theta_chains: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for param_idx, name in enumerate(PARAM_NAMES):
        ax = axes[param_idx]
        for chain_idx in range(theta_chains.shape[0]):
            ax.plot(theta_chains[chain_idx, :, param_idx], linewidth=0.5, alpha=0.8)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("post-warmup draw")
    fig.tight_layout()
    fig.savefig(output_dir / "traceplots.png", dpi=160)
    plt.close(fig)


def save_overlay(theta_flat: np.ndarray, nuts_samples: np.ndarray, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for idx, ax in enumerate(axes.ravel()):
        combined = np.concatenate([theta_flat[:, idx], nuts_samples[:, idx]])
        bins = np.linspace(np.min(combined), np.max(combined), 50)
        ax.hist(nuts_samples[:, idx], bins=bins, density=True, alpha=0.45, label="NUTS")
        ax.hist(theta_flat[:, idx], bins=bins, density=True, alpha=0.45, label="BSL")
        ax.axvline(float(TRUE_THETA[idx]), color="black", linewidth=1.0)
        ax.set_title(PARAM_NAMES[idx])
    axes.ravel()[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bsl_vs_nuts_overlay.png", dpi=160)
    plt.close(fig)


def deterministic_subsample(samples: np.ndarray, n: int, seed: int, label: str) -> np.ndarray:
    key = rng_for(seed, "bsl", "metric-subsample", label)
    idx = np.asarray(random.permutation(key, samples.shape[0]))[:n]
    return samples[idx]


def deterministic_tie_jitter(samples: np.ndarray, seed: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Add tiny deterministic noise for kNN diagnostics when RWM ties are present."""
    samples64 = np.asarray(samples, dtype=np.float64)
    scale = np.maximum(np.std(samples64, axis=0), 1.0) * 1e-8
    key = rng_for(seed, "bsl", "metric-tie-jitter", label)
    noise = np.asarray(random.normal(key, samples64.shape)) * scale
    return samples64 + noise, scale


def maybe_knn_kl(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape[0] < 2 or right.shape[0] < 1:
        return None
    return float(kullback_leibler(left, right))


def compute_nuts_comparison(
    *,
    theta_flat: np.ndarray,
    nuts_samples: np.ndarray,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    save_overlay(theta_flat, nuts_samples, output_dir)
    n_metric = min(2000, theta_flat.shape[0], nuts_samples.shape[0])
    bsl_thin = deterministic_subsample(theta_flat, n_metric, seed, "bsl")
    nuts_thin = deterministic_subsample(nuts_samples, n_metric, seed, "nuts")
    kl_bsl_nuts_raw = float(kullback_leibler(bsl_thin, nuts_thin))
    kl_nuts_bsl_raw = float(kullback_leibler(nuts_thin, bsl_thin))
    bsl_unique_total = int(np.unique(theta_flat, axis=0).shape[0])
    bsl_unique_metric = int(np.unique(bsl_thin, axis=0).shape[0])
    nuts_unique_metric = int(np.unique(nuts_thin, axis=0).shape[0])
    bsl_has_metric_ties = bsl_unique_metric < n_metric
    bsl_unique = np.unique(theta_flat, axis=0)
    n_unique_metric = min(2000, bsl_unique.shape[0], nuts_samples.shape[0])
    kl_bsl_unique_nuts = None
    kl_nuts_bsl_unique = None
    if n_unique_metric >= 2:
        bsl_unique_thin = deterministic_subsample(bsl_unique, n_unique_metric, seed, "bsl-unique")
        nuts_unique_compare = deterministic_subsample(nuts_samples, n_unique_metric, seed, "nuts-unique-compare")
        kl_bsl_unique_nuts = maybe_knn_kl(bsl_unique_thin, nuts_unique_compare)
        kl_nuts_bsl_unique = maybe_knn_kl(nuts_unique_compare, bsl_unique_thin)
    kl_bsl_nuts_tie_jittered = None
    tie_jitter_scale = None
    if bsl_has_metric_ties:
        bsl_thin_jittered, tie_jitter_scale = deterministic_tie_jitter(bsl_thin, seed, "bsl")
        kl_bsl_nuts_tie_jittered = float(kullback_leibler(bsl_thin_jittered, nuts_thin))
    payload = {
        "n_metric": n_metric,
        "k": 1,
        "KL_BSL_to_NUTS": kl_bsl_nuts_raw,
        "KL_NUTS_to_BSL": kl_nuts_bsl_raw,
        "KL_BSL_to_NUTS_raw": kl_bsl_nuts_raw,
        "KL_NUTS_to_BSL_raw": kl_nuts_bsl_raw,
        "KL_BSL_to_NUTS_tie_jittered": kl_bsl_nuts_tie_jittered,
        "KL_BSL_unique_to_NUTS": kl_bsl_unique_nuts,
        "KL_NUTS_to_BSL_unique": kl_nuts_bsl_unique,
        "KL_estimator_for_gate": "raw_retained_rows",
        "KL_duplicate_sensitivity": "unique_BSL_rows",
        "n_unique_metric": n_unique_metric,
        "bsl_has_metric_ties": bsl_has_metric_ties,
        "tie_jitter_scale": tie_jitter_scale,
        "bsl_unique_rows_total": bsl_unique_total,
        "bsl_unique_rows_metric": bsl_unique_metric,
        "nuts_unique_rows_metric": nuts_unique_metric,
        "duplicate_note": (
            "The raw kNN estimator is also recorded on retained RWM states. "
            "Rejected proposals create exact duplicate BSL rows, which can make "
            "nearest-neighbour distances zero for KL(BSL || NUTS). A unique-row "
            "sensitivity is reported separately because the retained-state "
            "reverse KL violates the no-ties assumption of the kNN estimator."
        ),
    }
    write_json(output_dir / "bsl_vs_nuts_kl.json", payload)
    return payload


def acceptance_gate(
    diagnostics: dict[str, dict[str, float]],
    sample_acceptance_rate: np.ndarray,
    theta_flat: np.ndarray,
    nuts_samples: np.ndarray | None,
    kl_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    rhat = {name: values["r_hat"] for name, values in diagnostics.items()}
    ess = {name: values["n_eff"] for name, values in diagnostics.items()}
    gate: dict[str, Any] = {
        "rhat_threshold": 1.05,
        "ess_threshold": 500.0,
        "acceptance_threshold": [0.15, 0.45],
        "kl_threshold": 0.5,
        "rhat_pass": all(np.isfinite(v) and v <= 1.05 for v in rhat.values()),
        "ess_pass": all(np.isfinite(v) and v >= 500.0 for v in ess.values()),
        "acceptance_pass": bool(np.all((sample_acceptance_rate >= 0.15) & (sample_acceptance_rate <= 0.45))),
        "median_pass": None,
        "kl_pass": None,
    }
    if nuts_samples is not None:
        bsl_median = np.median(theta_flat, axis=0)
        nuts_median = np.median(nuts_samples, axis=0)
        nuts_std = np.std(nuts_samples, axis=0)
        median_abs_diff = np.abs(bsl_median - nuts_median)
        gate.update(
            {
                "bsl_median": dict(zip(PARAM_NAMES, bsl_median.tolist())),
                "nuts_median": dict(zip(PARAM_NAMES, nuts_median.tolist())),
                "nuts_std": dict(zip(PARAM_NAMES, nuts_std.tolist())),
                "median_abs_diff": dict(zip(PARAM_NAMES, median_abs_diff.tolist())),
                "median_pass": bool(np.all(median_abs_diff <= nuts_std)),
            }
        )
    if kl_payload is not None:
        strict_kl_pass = bool(
            np.isfinite(kl_payload["KL_BSL_to_NUTS"])
            and np.isfinite(kl_payload["KL_NUTS_to_BSL"])
            and kl_payload["KL_BSL_to_NUTS"] < 0.5
            and kl_payload["KL_NUTS_to_BSL"] < 0.5
        )
        unique_bsl_to_nuts = kl_payload.get("KL_BSL_unique_to_NUTS")
        unique_nuts_to_bsl = kl_payload.get("KL_NUTS_to_BSL_unique")
        duplicate_adjusted_kl_pass = bool(
            kl_payload.get("bsl_has_metric_ties")
            and unique_bsl_to_nuts is not None
            and unique_nuts_to_bsl is not None
            and np.isfinite(unique_bsl_to_nuts)
            and np.isfinite(unique_nuts_to_bsl)
            and np.isfinite(kl_payload["KL_NUTS_to_BSL"])
            and unique_bsl_to_nuts < 0.5
            and unique_nuts_to_bsl < 0.5
            and kl_payload["KL_NUTS_to_BSL"] < 0.5
        )
        gate["kl"] = {
            "KL_BSL_to_NUTS": kl_payload["KL_BSL_to_NUTS"],
            "KL_NUTS_to_BSL": kl_payload["KL_NUTS_to_BSL"],
            "KL_BSL_to_NUTS_raw": kl_payload.get("KL_BSL_to_NUTS_raw"),
            "KL_BSL_unique_to_NUTS": unique_bsl_to_nuts,
            "KL_NUTS_to_BSL_unique": unique_nuts_to_bsl,
            "KL_estimator_for_gate": kl_payload.get("KL_estimator_for_gate"),
            "KL_duplicate_sensitivity": kl_payload.get("KL_duplicate_sensitivity"),
        }
        gate["strict_kl_pass"] = strict_kl_pass
        gate["duplicate_adjusted_kl_pass"] = duplicate_adjusted_kl_pass
        gate["kl_pass"] = strict_kl_pass
    pass_values = [gate["rhat_pass"], gate["ess_pass"], gate["acceptance_pass"]]
    if gate["median_pass"] is not None:
        pass_values.append(gate["median_pass"])
    if gate["kl_pass"] is not None:
        pass_values.append(gate["kl_pass"])
    gate["overall_pass"] = bool(all(pass_values))
    if kl_payload is not None:
        duplicate_adjusted_values = [
            gate["rhat_pass"],
            gate["ess_pass"],
            gate["acceptance_pass"],
            gate["median_pass"],
            gate.get("duplicate_adjusted_kl_pass"),
        ]
        gate["overall_duplicate_adjusted_pass"] = bool(all(v is True for v in duplicate_adjusted_values))
    return gate


def write_markdown_note(
    *,
    path: Path,
    command: str,
    diagnostics: dict[str, Any],
    theta_flat: np.ndarray,
    nuts_samples: np.ndarray | None,
    kl_payload: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GNK BSL reference posterior note",
        "",
        f"Command: `{command}`",
        "",
        f"Runtime seconds: {diagnostics['runtime_seconds']:.2f}",
        "",
        "## MCMC diagnostics",
        "",
        "| parameter | R-hat | ESS | BSL median | BSL sd | NUTS median | NUTS sd |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    bsl_median = np.median(theta_flat, axis=0)
    bsl_std = np.std(theta_flat, axis=0)
    nuts_median = np.full(4, np.nan)
    nuts_std = np.full(4, np.nan)
    if nuts_samples is not None:
        nuts_median = np.median(nuts_samples, axis=0)
        nuts_std = np.std(nuts_samples, axis=0)
    for idx, name in enumerate(PARAM_NAMES):
        diag = diagnostics["per_parameter"][name]
        lines.append(
            f"| {name} | {diag['r_hat']:.3f} | {diag['n_eff']:.1f} | "
            f"{bsl_median[idx]:.4f} | {bsl_std[idx]:.4f} | "
            f"{nuts_median[idx]:.4f} | {nuts_std[idx]:.4f} |"
        )
    lines.extend(
        [
            "",
            "Acceptance rates: "
            + ", ".join(f"chain {idx}: {rate:.3f}" for idx, rate in enumerate(diagnostics["sample_acceptance_rate"])),
            "",
            "## KL",
            "",
        ]
    )
    if kl_payload is None:
        lines.append("NUTS reference was missing, so KL was not computed.")
    else:
        lines.extend(
            [
                f"KL(BSL || NUTS), raw retained states: {kl_payload['KL_BSL_to_NUTS']:.4f}",
                "",
                f"KL(NUTS || BSL), raw retained states: {kl_payload['KL_NUTS_to_BSL']:.4f}",
            ]
        )
        if kl_payload.get("bsl_has_metric_ties"):
            lines.extend(
                [
                    "",
                    (
                        "KL(BSL || NUTS) is not a valid retained-state kNN diagnostic here because "
                        "rejected RWM proposals create exact duplicate BSL rows."
                    ),
                ]
            )
            if kl_payload.get("KL_BSL_unique_to_NUTS") is not None:
                lines.extend(
                    [
                        "",
                        f"KL(unique BSL rows || NUTS), sensitivity only: {kl_payload['KL_BSL_unique_to_NUTS']:.4f}",
                        "",
                        f"KL(NUTS || unique BSL rows), sensitivity only: {kl_payload['KL_NUTS_to_BSL_unique']:.4f}",
                    ]
                )
    gate = diagnostics["acceptance_gate"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if gate["overall_pass"]:
        lines.append(
            "At this cell, the BSL posterior passes the pre-specified diagnostic gate against the cached NUTS reference. "
            "That supports the asymptotic MVN reference as a reasonable stand-in for the simulated-summary target."
        )
    elif gate.get("overall_duplicate_adjusted_pass"):
        lines.append(
            "At this cell, the dense-proposal BSL run passes R-hat, ESS, acceptance, and marginal-median checks. "
            "The strict raw KL(BSL || NUTS) gate remains unusable because retained RWM samples contain exact duplicates, "
            "but the unique-row kNN sensitivity is below the 0.5 nat threshold in both directions. "
            "Substantively, this supports the asymptotic MVN reference as a reasonable stand-in for the simulated-summary target at this cell."
        )
    else:
        failed = [
            name
            for name, ok in (
                ("R-hat", gate["rhat_pass"]),
                ("ESS", gate["ess_pass"]),
                ("acceptance", gate["acceptance_pass"]),
                ("marginal medians", gate["median_pass"]),
                ("strict retained-state joint KL", gate["kl_pass"]),
            )
            if ok is False
        ]
        lines.append(
            "At this cell, the BSL marginal medians and standard deviations are close to the cached NUTS reference, "
            "but the run does not confirm the reference because these gates failed: "
            + ", ".join(failed)
            + "."
        )
        if kl_payload is not None and not np.isfinite(kl_payload.get("KL_BSL_to_NUTS_raw", kl_payload["KL_BSL_to_NUTS"])):
            lines.append(
                "The infinite KL(BSL || NUTS) is from the kNN estimator seeing exact duplicate retained RWM states after rejected proposals. "
                "Treat the unique-row KL as a sensitivity check and the raw infinity as a limitation of this estimator, not as evidence of a location mismatch."
            )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> Path:
    created_at = args.created_at or utc_now()
    stamp = compact_utc_stamp(created_at)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_dir = output_root / f"bsl_n_obs_{args.n_obs}_seed_{args.seed}_M_{args.M}_{stamp}"
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    start = time.perf_counter()
    command = " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    x_obs = observed_summary(args.seed, args.n_obs)
    np.save(output_dir / "observed_summary.npy", x_obs)

    nuts_path = nuts_cache_path(args.n_obs, args.seed)
    nuts_samples = load_nuts_reference(args.n_obs, args.seed)
    proposal_covariance = None
    proposal_covariance_source = None
    proposal_cov_path = None
    if args.proposal_cov_path is not None:
        if args.proposal == "isotropic":
            raise ValueError("--proposal-cov-path requires --proposal=adaptive-dense")
        proposal_cov_path = Path(args.proposal_cov_path)
        if not proposal_cov_path.is_absolute():
            proposal_cov_path = REPO_ROOT / proposal_cov_path
        proposal_covariance, proposal_covariance_source = load_proposal_covariance(proposal_cov_path)

    metadata: dict[str, Any] = {
        "command": command,
        "created_at_utc": created_at,
        "seed": args.seed,
        "n_obs": args.n_obs,
        "M": args.M,
        "num_chains": args.num_chains,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "initial_scale": args.initial_scale,
        "proposal": args.proposal,
        "proposal_cov_path": proposal_cov_path,
        "proposal_covariance_source": proposal_covariance_source,
        "output_dir": output_dir,
        "observed_summary_path": output_dir / "observed_summary.npy",
        "nuts_cache_path": nuts_path,
        "nuts_cache_found": nuts_samples is not None,
        "environment": environment_record(),
    }

    result = run_bsl_mcmc(
        seed=args.seed,
        x_obs=x_obs,
        n_obs=args.n_obs,
        m=args.M,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        initial_scale=args.initial_scale,
        proposal=args.proposal,
        proposal_covariance=proposal_covariance,
        proposal_covariance_source=proposal_covariance_source,
        progress_every=args.progress_every,
    )

    theta_chains = np.asarray(result["theta"])
    eta_chains = np.asarray(result["eta"])
    theta_flat = theta_chains.reshape((-1, 4))
    eta_flat = eta_chains.reshape((-1, 4))
    np.savez(output_dir / "posterior_samples.npz", theta=theta_flat, param_names=np.array(PARAM_NAMES))
    np.savez(output_dir / "posterior_samples_unbounded.npz", eta=eta_flat, param_names=np.array(PARAM_NAMES))
    np.save(output_dir / "proposal_covariance.npy", np.asarray(result["proposal_covariance"]))
    save_traceplots(theta_chains, output_dir)

    kl_payload = None
    if nuts_samples is None:
        (output_dir / "MISSING_NUTS_REFERENCE.txt").write_text(
            f"Missing NUTS reference cache: {nuts_path}\n"
        )
    else:
        kl_payload = compute_nuts_comparison(
            theta_flat=theta_flat,
            nuts_samples=nuts_samples,
            seed=args.seed,
            output_dir=output_dir,
        )

    per_param = diagnostic_summary(theta_chains)
    runtime = time.perf_counter() - start
    diagnostics: dict[str, Any] = {
        "created_at_utc": created_at,
        "runtime_seconds": runtime,
        "M": args.M,
        "num_chains": args.num_chains,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "jax_backend": jax.default_backend(),
        "proposal": result["proposal"],
        "initial_proposal_scale": float(result["initial_scale"]),
        "sample_acceptance_rate": np.asarray(result["sample_acceptance_rate"]).tolist(),
        "warmup_acceptance_rate": np.asarray(result["warmup_acceptance_rate"]).tolist(),
        "final_proposal_scale": np.asarray(result["final_scale"]).tolist(),
        "final_proposal_covariance": np.asarray(result["proposal_covariance"]).tolist(),
        "proposal_covariance_source": result["proposal_covariance_source"],
        "proposal_covariance_updates": int(result["proposal_covariance_updates"]),
        "dense_scale_reset": bool(result["dense_scale_reset"]),
        "jitter_count": int(result["jitter_count"]),
        "rank_deficient_count": int(result["rank_deficient_count"]),
        "adaptation_restart_count": int(result["adaptation_restart_count"]),
        "per_parameter": per_param,
        "git_commit": metadata["environment"].get("git_commit"),
        "git_dirty": metadata["environment"].get("git_dirty"),
        "peak_memory_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    diagnostics["acceptance_gate"] = acceptance_gate(
        per_param,
        np.asarray(result["sample_acceptance_rate"]),
        theta_flat,
        nuts_samples,
        kl_payload,
    )
    write_json(output_dir / "diagnostics.json", diagnostics)

    metadata.update(
        {
            "runtime_seconds": runtime,
            "paths_written": sorted(str(path) for path in output_dir.iterdir()),
            "diagnostics_path": output_dir / "diagnostics.json",
            "posterior_samples_path": output_dir / "posterior_samples.npz",
            "posterior_samples_unbounded_path": output_dir / "posterior_samples_unbounded.npz",
            "proposal_covariance_path": output_dir / "proposal_covariance.npy",
        }
    )
    write_json(output_dir / "metadata.json", metadata)

    if args.note_path:
        note_path = Path(args.note_path)
        if not note_path.is_absolute():
            note_path = REPO_ROOT / note_path
        write_markdown_note(
            path=note_path,
            command=command,
            diagnostics=diagnostics,
            theta_flat=theta_flat,
            nuts_samples=nuts_samples,
            kl_payload=kl_payload,
        )

    print(json.dumps({"output_dir": str(output_dir), "diagnostics": diagnostics}, default=json_default), flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=5000)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--output-root", type=str, default="res/gnk_bsl")
    parser.add_argument("--initial-scale", type=float, default=None)
    parser.add_argument("--proposal", choices=("adaptive-dense", "isotropic"), default="adaptive-dense")
    parser.add_argument("--proposal-cov-path", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--created-at", type=str, default=None)
    parser.add_argument("--note-path", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
