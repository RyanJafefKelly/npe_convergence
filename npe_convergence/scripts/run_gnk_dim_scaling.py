"""Run the GNK dimension-scaling pilot.

The pilot keeps all artifacts in ``res/gnk_dim_scaling`` and uses a phase split:

* ``precompute-nuts`` writes one reference cache per ``(d_s, seed)``.
* ``run-cell`` trains/evaluates one ``(d_s, method, seed, n_obs, n_sims)`` cell.
* ``aggregate`` builds CSV summaries and diagnostic figures.
* ``prepare-manifests`` writes explicit PBS array manifests.
* ``run-manifest-row`` executes one row from a manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle as pkl
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
from typing import Any, Callable

os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.environ.get('GNK_DIM_SCALING_CPU_DEVICES', '4')}",
)

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib
import numpy as np
import numpyro  # type: ignore
from jax.scipy.special import expit, logit
from numpyro.infer import MCMC, NUTS  # type: ignore


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import (  # noqa: E402
    get_summaries_batches,
    gnk,
    gnk_model,
    ss_duodecile,
    ss_hexadeciles,
    ss_octile,
    ss_sextile,
    ss_vigintile,
)
from npe_convergence.metrics import kullback_leibler, median_heuristic, unbiased_mmd  # noqa: E402


PARAM_NAMES = ("A", "B", "g", "k")
TRUE_PARAMS = jnp.array([3.0, 1.0, 2.0, 0.5])
D_S_GRID = (5, 7, 11, 15, 19)
METHODS = ("flow_npe", "gaussian_npe")
DEFAULT_N_OBS = 500
DEFAULT_SCALE = 5
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "res" / "gnk_dim_scaling"
DEFAULT_PLOT_DIR = REPO_ROOT / "notebooks" / "plots"
NUTS_CACHE_RE = re.compile(
    r"nuts_cache_v2_d_s_(?P<d_s>\d+)_n_obs_(?P<n_obs>\d+)_seed_(?P<seed>\d+)\.pkl$"
)
CELL_RE = re.compile(
    r"gnk_(?P<method>flow_npe|gaussian_npe)_d_s_(?P<d_s>\d+)_"
    r"n_obs_(?P<n_obs>\d+)_n_sims_(?P<n_sims>\d+)_seed_(?P<seed>\d+)$"
)


SUMMARY_FNS: dict[int, Callable[[jnp.ndarray], jnp.ndarray]] = {
    5: ss_sextile,
    7: ss_octile,
    11: ss_duodecile,
    15: ss_hexadeciles,
    19: ss_vigintile,
}


@dataclass(frozen=True)
class CellSpec:
    d_s: int
    method: str
    seed: int
    n_obs: int
    n_sims: int

    @property
    def d(self) -> int:
        return self.d_s + len(PARAM_NAMES)

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
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (jax.Array,)):
        return np.asarray(value).tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pkl.dump(payload, f)
    tmp.replace(path)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pkl.load(f)


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


def d_total(d_s: int) -> int:
    return validate_d_s(d_s) + len(PARAM_NAMES)


def scaled_n_sims(d_s: int, n_obs: int = DEFAULT_N_OBS, scale: int = DEFAULT_SCALE) -> int:
    d = d_total(d_s)
    return int(scale * d * d * n_obs)


def quantile_probabilities(d_s: int) -> np.ndarray:
    validate_d_s(d_s)
    return np.linspace(1.0 / (d_s + 1), d_s / (d_s + 1), d_s)


def summary_fn(d_s: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    validate_d_s(d_s)
    return SUMMARY_FNS[d_s]


def observed_summary(seed: int, n_obs: int, d_s: int) -> np.ndarray:
    """Observed GNK summary using the legacy flow runner convention."""
    key = random.key(seed)
    z = random.normal(key, shape=(n_obs,))
    x_obs = gnk(z, *TRUE_PARAMS)
    summaries = summary_fn(d_s)(jnp.atleast_2d(x_obs))
    return np.asarray(jnp.squeeze(summaries))


def hash_array(arr: np.ndarray) -> str:
    stable = np.asarray(arr, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(stable.shape).encode())
    digest.update(stable.tobytes())
    return digest.hexdigest()


def nuts_cache_path(output_root: Path | str, d_s: int, n_obs: int, seed: int) -> Path:
    root = resolve_path(output_root)
    return root / f"nuts_cache_v2_d_s_{d_s}_n_obs_{n_obs}_seed_{seed}.pkl"


def nuts_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.stem + ".json")


def nuts_observed_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.stem + "_observed_summary.npy")


def cell_dir(output_root: Path | str, spec: CellSpec) -> Path:
    root = resolve_path(output_root)
    return root / (
        f"gnk_{spec.method}_d_s_{spec.d_s}_n_obs_{spec.n_obs}_"
        f"n_sims_{spec.n_sims}_seed_{spec.seed}"
    )


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def diagnostics_pass(
    diagnostics: dict[str, dict[str, float | None]],
    *,
    max_rhat: float,
    min_ess: float,
) -> bool:
    for name in PARAM_NAMES:
        values = diagnostics.get(name, {})
        r_hat = finite_float(values.get("r_hat"))
        ess_bulk = finite_float(values.get("ess_bulk"))
        ess_tail = finite_float(values.get("ess_tail"))
        if r_hat is None or ess_bulk is None or ess_tail is None:
            return False
        if r_hat > max_rhat or min(ess_bulk, ess_tail) < min_ess:
            return False
    return True


def stack_chain_samples(samples_by_chain: dict[str, jax.Array]) -> np.ndarray:
    chains = [np.asarray(samples_by_chain[name]) for name in PARAM_NAMES]
    return np.stack(chains, axis=-1)


def nuts_diagnostics_from_mcmc(mcmc: MCMC) -> dict[str, dict[str, float | None]]:
    import arviz as az

    inference_data = az.from_numpyro(mcmc)
    summary = az.summary(inference_data, var_names=list(PARAM_NAMES), kind="all")
    diagnostics: dict[str, dict[str, float | None]] = {}
    for name in PARAM_NAMES:
        if name in summary.index:
            row = summary.loc[name]
        else:
            matches = [idx for idx in summary.index if str(idx).split("[", 1)[0] == name]
            if not matches:
                diagnostics[name] = {
                    "mean": None,
                    "sd": None,
                    "ess_bulk": None,
                    "ess_tail": None,
                    "r_hat": None,
                }
                continue
            row = summary.loc[matches[0]]
        diagnostics[name] = {
            "mean": finite_float(row.get("mean")),
            "sd": finite_float(row.get("sd")),
            "ess_bulk": finite_float(row.get("ess_bulk")),
            "ess_tail": finite_float(row.get("ess_tail")),
            "r_hat": finite_float(row.get("r_hat")),
        }
    return diagnostics


def run_nuts_reference(
    *,
    seed: int,
    d_s: int,
    n_obs: int,
    obs: np.ndarray,
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    thinning: int,
    progress_bar: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float | None]]]:
    numpyro.set_host_device_count(num_chains)
    draws_per_chain = max(1, math.ceil(num_samples / num_chains))
    mcmc_num_samples = draws_per_chain * max(1, thinning)
    kernel = NUTS(gnk_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=mcmc_num_samples,
        thinning=thinning,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    def init_param_to_unbounded(value: float, subkey: jax.Array) -> jax.Array:
        base = jnp.repeat(logit(jnp.array([value]) / 10), num_chains)
        noise = random.normal(subkey, (num_chains,)) * 0.05
        return base + noise

    rng_key = rng_for(seed, d_s, n_obs, "nuts")
    rng_key, *subkeys = random.split(rng_key, 5)
    init_params = {
        "A": init_param_to_unbounded(3.0, subkeys[0]),
        "B": init_param_to_unbounded(1.0, subkeys[1]),
        "g": init_param_to_unbounded(2.0, subkeys[2]),
        "k": init_param_to_unbounded(0.5, subkeys[3]),
    }
    mcmc.run(rng_key=rng_key, init_params=init_params, obs=jnp.asarray(obs), n_obs=n_obs)
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    chain_samples = stack_chain_samples(samples_by_chain)
    flat_samples = chain_samples.reshape(-1, len(PARAM_NAMES))[:num_samples]
    diagnostics = nuts_diagnostics_from_mcmc(mcmc)
    return flat_samples, chain_samples, diagnostics


def precompute_nuts(
    *,
    output_root: Path | str,
    d_s: int,
    seed: int,
    n_obs: int,
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    thinning: int,
    max_rhat: float,
    min_ess: float,
    skip_diagnostic_gate: bool,
    force: bool,
    progress_bar: bool,
) -> Path:
    validate_d_s(d_s)
    root = resolve_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    cache = nuts_cache_path(root, d_s, n_obs, seed)
    metadata_path = nuts_metadata_path(cache)
    if cache.exists() and metadata_path.exists() and not force:
        metadata = load_json(metadata_path)
        if not skip_diagnostic_gate and not diagnostics_pass(
            metadata.get("diagnostics", {}), max_rhat=max_rhat, min_ess=min_ess
        ):
            raise RuntimeError(f"existing NUTS cache fails diagnostics: {metadata_path}")
        print(f"Skipping existing NUTS cache: {cache}")
        return cache

    start = time.perf_counter()
    numpyro.set_host_device_count(num_chains)
    x_obs = observed_summary(seed, n_obs, d_s)
    obs_hash = hash_array(x_obs)
    flat_samples, chain_samples, diagnostics = run_nuts_reference(
        seed=seed,
        d_s=d_s,
        n_obs=n_obs,
        obs=x_obs,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        thinning=thinning,
        progress_bar=progress_bar,
    )
    passed = diagnostics_pass(diagnostics, max_rhat=max_rhat, min_ess=min_ess)
    metadata = {
        "status": "complete" if passed else "diagnostic_failed",
        "created_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - start,
        "d_s": d_s,
        "d_theta": len(PARAM_NAMES),
        "d": d_total(d_s),
        "n_obs": n_obs,
        "seed": seed,
        "observed_data_convention": "run_gnk.py: random.key(seed) then random.normal(key)",
        "obs_hash": obs_hash,
        "quantile_probabilities": quantile_probabilities(d_s).tolist(),
        "param_names": list(PARAM_NAMES),
        "num_samples_requested": num_samples,
        "num_samples_saved": int(flat_samples.shape[0]),
        "num_warmup": num_warmup,
        "num_chains": num_chains,
        "thinning": thinning,
        "nuts_seed_rule": "rng_for(seed, d_s, n_obs, 'nuts')",
        "diagnostics": diagnostics,
        "diagnostic_thresholds": {"max_rhat": max_rhat, "min_ess": min_ess},
        "diagnostic_pass": passed,
        "environment": environment_record(),
    }
    payload = {
        "samples": np.asarray(flat_samples),
        "chains": np.asarray(chain_samples),
        "param_names": list(PARAM_NAMES),
    }
    if force:
        for path in (cache, metadata_path, nuts_observed_path(cache)):
            if path.exists():
                path.unlink()
    save_pickle(cache, payload)
    np.save(nuts_observed_path(cache), x_obs)
    write_json(metadata_path, metadata)
    if not passed and not skip_diagnostic_gate:
        raise RuntimeError(f"NUTS diagnostics failed for d_s={d_s}, seed={seed}")
    return cache


def load_nuts_reference(
    *,
    output_root: Path | str,
    d_s: int,
    seed: int,
    n_obs: int,
    max_rhat: float,
    min_ess: float,
    skip_diagnostic_gate: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache = nuts_cache_path(output_root, d_s, n_obs, seed)
    metadata_path = nuts_metadata_path(cache)
    if not cache.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"missing NUTS cache or metadata: {cache}")
    metadata = load_json(metadata_path)
    x_obs = observed_summary(seed, n_obs, d_s)
    expected_hash = hash_array(x_obs)
    expected_quantiles = quantile_probabilities(d_s)
    if metadata.get("d_s") != d_s or metadata.get("n_obs") != n_obs or metadata.get("seed") != seed:
        raise RuntimeError(f"NUTS cache metadata does not match requested cell: {metadata_path}")
    if metadata.get("obs_hash") != expected_hash:
        raise RuntimeError(f"NUTS cache obs_hash mismatch: {metadata_path}")
    if not np.allclose(np.asarray(metadata.get("quantile_probabilities")), expected_quantiles):
        raise RuntimeError(f"NUTS cache quantiles mismatch: {metadata_path}")
    if not skip_diagnostic_gate and not diagnostics_pass(
        metadata.get("diagnostics", {}), max_rhat=max_rhat, min_ess=min_ess
    ):
        raise RuntimeError(f"NUTS cache fails diagnostics: {metadata_path}")
    payload = load_pickle(cache)
    samples = np.asarray(payload["samples"])
    if samples.ndim != 2 or samples.shape[1] != len(PARAM_NAMES):
        raise RuntimeError(f"invalid NUTS sample shape in {cache}: {samples.shape}")
    if not np.isfinite(samples).all():
        raise RuntimeError(f"NUTS cache contains non-finite samples: {cache}")
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
    tol = 1e-6
    theta_bounded = random.uniform(theta_key, (n_sims, len(PARAM_NAMES)), minval=tol, maxval=10.0 - tol)
    theta_unbounded = logit(theta_bounded / 10.0)
    A, B, g, k = theta_bounded.T
    summaries = get_summaries_batches(
        sim_key,
        A,
        B,
        g,
        k,
        n_obs=n_obs,
        n_sims=n_sims,
        batch_size=min(sim_batch_size, n_sims),
        sum_fn=summary_fn(d_s),
    ).T
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
    posterior_bounded = expit(posterior_unbounded) * 10.0
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
    clipped = np.clip(np.asarray(theta_bounded, dtype=np.float64), eps, 10.0 - eps)
    y = clipped / 10.0
    theta_unbounded = np.asarray(logit(jnp.asarray(y)))
    theta_standardised = (theta_unbounded - theta_mean) / theta_std
    log_prob_std = log_prob_standardised_theta(
        method=method,
        model=model,
        theta_standardised=theta_standardised,
        x_obs_std=x_obs_std,
    )
    log_standardise_jac = -float(np.sum(np.log(theta_std)))
    log_unbounded_jac = np.sum(-np.log(10.0) - np.log(y) - np.log1p(-y), axis=1)
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
    max_rhat: float,
    min_ess: float,
    skip_nuts_diagnostic_gate: bool,
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
        "d_theta": len(PARAM_NAMES),
        "d": spec.d,
        "n_obs": n_obs,
        "n_sims": n_sims,
        "scaled_budget": spec.scaled_budget,
        "method": method,
        "seed": seed,
        "observed_data_convention": "run_gnk.py: random.key(seed) then random.normal(key)",
        "environment": environment_record(),
    }
    start = time.perf_counter()
    try:
        true_samples, nuts_metadata = load_nuts_reference(
            output_root=output_root,
            d_s=d_s,
            seed=seed,
            n_obs=n_obs,
            max_rhat=max_rhat,
            min_ess=min_ess,
            skip_diagnostic_gate=skip_nuts_diagnostic_gate,
        )
        x_obs_raw = observed_summary(seed, n_obs, d_s)
        diagnostics["obs_hash"] = hash_array(x_obs_raw)
        diagnostics["nuts_cache"] = str(nuts_cache_path(output_root, d_s, n_obs, seed))
        diagnostics["nuts_metadata"] = nuts_metadata
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
    row["status"] = diagnostics.get("status", "missing_diagnostics")
    row["failure_reason"] = diagnostics.get("failure_reason")
    row["obs_hash"] = diagnostics.get("obs_hash")
    row["wall_time_seconds"] = diagnostics.get("total_wall_time_seconds")
    row["training_time_seconds"] = diagnostics.get("training_time_seconds")
    row["simulation_time_seconds"] = diagnostics.get("simulation_time_seconds")
    row["kl"] = read_float_file(path / "kl.txt")
    row["mmd"] = read_float_file(path / "mmd.txt")
    metrics_path = path / "metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    row["theta_kl"] = finite_float(metrics.get("kl_theta_knn_2000", row["kl"]))
    row["cross_entropy"] = finite_float(metrics.get("cross_entropy"))
    row["finite_metric"] = row["kl"] is not None and row["mmd"] is not None
    if row["status"] == "complete" and not row["finite_metric"]:
        row["status"] = "nonfinite_metric"
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["d_s"], row["d"], row["method"], row["n_obs"], row["n_sims"], row["scaled_budget"])
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for key, group in sorted(groups.items()):
        d_s, d, method, n_obs, n_sims, scaled_budget = key
        finite = [r for r in group if r.get("finite_metric")]
        kl = np.asarray([r["kl"] for r in finite], dtype=float)
        mmd = np.asarray([r["mmd"] for r in finite], dtype=float)
        summary_rows.append(
            {
                "d_s": d_s,
                "d": d,
                "method": method,
                "n_obs": n_obs,
                "n_sims": n_sims,
                "scaled_budget": scaled_budget,
                "n_cells": len(group),
                "n_complete": sum(1 for r in group if r.get("status") == "complete"),
                "n_finite": len(finite),
                "n_failed": sum(1 for r in group if r.get("status") != "complete"),
                "kl_median": float(np.median(kl)) if kl.size else None,
                "kl_q25": float(np.quantile(kl, 0.25)) if kl.size else None,
                "kl_q75": float(np.quantile(kl, 0.75)) if kl.size else None,
                "mmd_median": float(np.median(mmd)) if mmd.size else None,
                "mmd_q25": float(np.quantile(mmd, 0.25)) if mmd.size else None,
                "mmd_q75": float(np.quantile(mmd, 0.75)) if mmd.size else None,
            }
        )
    return summary_rows


def nonnull_float(value: Any) -> float | None:
    out = finite_float(value)
    return out


def plot_summary(summary_rows: list[dict[str, Any]], plot_dir: Path, timestamp: str) -> list[Path]:
    figures: list[Path] = []
    finite_rows = [row for row in summary_rows if nonnull_float(row.get("kl_median")) is not None]
    methods = [method for method in METHODS if any(row["method"] == method for row in finite_rows)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        rows = sorted([row for row in finite_rows if row["method"] == method], key=lambda r: r["d"])
        ax.plot([r["d"] for r in rows], [r["kl_median"] for r in rows], marker="o", label=method)
    ax.set_xlabel("Total dimension d")
    ax.set_ylabel("Median theta KL")
    ax.set_yscale("log")
    ax.set_title("GNK dimension scaling at matched scaled budget")
    ax.legend()
    fig.tight_layout()
    path = plot_dir / f"gnk_dim_scaling_pilot_{timestamp}_kl_vs_d.pdf"
    fig.savefig(path)
    plt.close(fig)
    figures.append(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        rows = sorted([row for row in finite_rows if row["method"] == method], key=lambda r: r["d"])
        ax.plot(
            [r["d"] * r["d"] for r in rows],
            [r["kl_median"] * r["n_sims"] / r["n_obs"] for r in rows],
            marker="o",
            label=method,
        )
    ax.set_xlabel("d^2")
    ax.set_ylabel("Median KL * N / n")
    ax.set_title("Scaled KL versus dimension squared")
    ax.legend()
    fig.tight_layout()
    path = plot_dir / f"gnk_dim_scaling_pilot_{timestamp}_scaled_kl_vs_d2.pdf"
    fig.savefig(path)
    plt.close(fig)
    figures.append(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    d_values = sorted({row["d"] for row in finite_rows})
    width = 0.35 if len(methods) > 1 else 0.55
    offsets = np.linspace(-width / 2, width / 2, max(1, len(methods)))
    for offset, method in zip(offsets, methods):
        values = []
        for d in d_values:
            match = next((row for row in finite_rows if row["method"] == method and row["d"] == d), None)
            values.append(match["kl_median"] if match else np.nan)
        ax.bar(np.asarray(d_values) + offset, values, width=width / max(1, len(methods)), label=method)
    ax.set_xlabel("Total dimension d")
    ax.set_ylabel("Median theta KL")
    ax.set_yscale("log")
    ax.set_title("Flow and Gaussian NPE at matched scaled budget")
    ax.legend()
    fig.tight_layout()
    path = plot_dir / f"gnk_dim_scaling_pilot_{timestamp}_method_comparison.pdf"
    fig.savefig(path)
    plt.close(fig)
    figures.append(path)
    return figures


def aggregate_results(*, output_root: Path | str, plot_dir: Path | str, timestamp: str | None) -> dict[str, Path]:
    root = resolve_path(output_root)
    plot_root = resolve_path(plot_dir)
    stamp = compact_timestamp(timestamp)
    rows = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if path.is_dir():
            row = cell_row_from_dir(path)
            if row is not None:
                rows.append(row)
    summary_rows = summarise_rows(rows)
    per_seed_fields = [
        "d_s",
        "d",
        "method",
        "n_obs",
        "n_sims",
        "scaled_budget",
        "seed",
        "status",
        "finite_metric",
        "kl",
        "mmd",
        "obs_hash",
        "wall_time_seconds",
        "simulation_time_seconds",
        "training_time_seconds",
        "failure_reason",
        "path",
    ]
    summary_fields = [
        "d_s",
        "d",
        "method",
        "n_obs",
        "n_sims",
        "scaled_budget",
        "n_cells",
        "n_complete",
        "n_finite",
        "n_failed",
        "kl_median",
        "kl_q25",
        "kl_q75",
        "mmd_median",
        "mmd_q25",
        "mmd_q75",
    ]
    per_seed_path = plot_root / f"gnk_dim_scaling_pilot_{stamp}_per_seed.csv"
    summary_path = plot_root / f"gnk_dim_scaling_pilot_{stamp}_summary.csv"
    write_csv(per_seed_path, rows, per_seed_fields)
    write_csv(summary_path, summary_rows, summary_fields)
    figures = plot_summary(summary_rows, plot_root, stamp)
    metadata_path = plot_root / f"gnk_dim_scaling_pilot_{stamp}_plot_metadata.json"
    write_json(
        metadata_path,
        {
            "created_at_utc": utc_now(),
            "timestamp": stamp,
            "output_root": str(root),
            "per_seed_csv": str(per_seed_path),
            "summary_csv": str(summary_path),
            "figures": [str(path) for path in figures],
            "rows": len(rows),
            "summary_rows": len(summary_rows),
            "environment": environment_record(),
        },
    )
    return {
        "per_seed_csv": per_seed_path,
        "summary_csv": summary_path,
        "plot_metadata": metadata_path,
    }


def manifest_rows(num_seeds: int, n_obs: int, scale: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nuts_rows = []
    cell_rows = []
    for seed in range(num_seeds):
        for d_s in D_S_GRID:
            n_sims = scaled_n_sims(d_s, n_obs=n_obs, scale=scale)
            nuts_rows.append(
                {
                    "array_index": len(nuts_rows),
                    "kind": "nuts",
                    "d_s": d_s,
                    "d": d_total(d_s),
                    "n_obs": n_obs,
                    "n_sims": n_sims,
                    "seed": seed,
                    "method": "",
                    "scaled_budget": scale,
                }
            )
            for method in METHODS:
                cell_rows.append(
                    {
                        "array_index": len(cell_rows),
                        "kind": "cell",
                        "d_s": d_s,
                        "d": d_total(d_s),
                        "n_obs": n_obs,
                        "n_sims": n_sims,
                        "seed": seed,
                        "method": method,
                        "scaled_budget": scale,
                    }
                )
    return nuts_rows, cell_rows


def manifest_rows_csweep(
    *,
    num_seeds: int,
    n_obs: int,
    scales: tuple[int, ...],
    max_n_sims: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    nuts_rows = []
    cell_rows = []
    skipped_rows = []
    for seed in range(num_seeds):
        for d_s in D_S_GRID:
            nuts_rows.append(
                {
                    "array_index": len(nuts_rows),
                    "kind": "nuts",
                    "d_s": d_s,
                    "d": d_total(d_s),
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
    return nuts_rows, cell_rows, skipped_rows


def prepare_manifests(
    *,
    output_root: Path | str,
    n_obs: int,
    scale: int,
    num_seeds: int,
    timestamp: str | None,
) -> dict[str, Path]:
    root = resolve_path(output_root)
    stamp = compact_timestamp(timestamp)
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    nuts_rows, cell_rows = manifest_rows(num_seeds, n_obs, scale)
    fields = ["array_index", "kind", "d_s", "d", "n_obs", "n_sims", "seed", "method", "scaled_budget"]
    nuts_csv = manifest_dir / f"gnk_dim_scaling_nuts_manifest_{stamp}.csv"
    cells_csv = manifest_dir / f"gnk_dim_scaling_cells_manifest_{stamp}.csv"
    write_csv(nuts_csv, nuts_rows, fields)
    write_csv(cells_csv, cell_rows, fields)
    summary_json = manifest_dir / f"gnk_dim_scaling_manifest_summary_{stamp}.json"
    write_json(
        summary_json,
        {
            "created_at_utc": utc_now(),
            "n_obs": n_obs,
            "scale": scale,
            "num_seeds": num_seeds,
            "d_s_grid": list(D_S_GRID),
            "methods": list(METHODS),
            "nuts_rows": len(nuts_rows),
            "cell_rows": len(cell_rows),
            "nuts_manifest": str(nuts_csv),
            "cells_manifest": str(cells_csv),
            "environment": environment_record(),
        },
    )
    return {"nuts_manifest": nuts_csv, "cells_manifest": cells_csv, "summary": summary_json}


def prepare_csweep_manifests(
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
    nuts_rows, cell_rows, skipped_rows = manifest_rows_csweep(
        num_seeds=num_seeds,
        n_obs=n_obs,
        scales=scales,
        max_n_sims=max_n_sims,
    )
    fields = ["array_index", "kind", "d_s", "d", "n_obs", "n_sims", "seed", "method", "scaled_budget"]
    skipped_fields = [*fields, "reason"]
    nuts_csv = manifest_dir / f"gnk_dim_scaling_nuts_manifest_{stamp}.csv"
    cells_csv = manifest_dir / f"gnk_dim_scaling_cells_manifest_{stamp}.csv"
    skipped_csv = manifest_dir / "skipped_cells.csv"
    write_csv(nuts_csv, nuts_rows, fields)
    write_csv(cells_csv, cell_rows, fields)
    write_csv(skipped_csv, skipped_rows, skipped_fields)
    summary_json = manifest_dir / f"gnk_dim_scaling_manifest_summary_{stamp}.json"
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
            "nuts_rows": len(nuts_rows),
            "cell_rows": len(cell_rows),
            "skipped_rows": len(skipped_rows),
            "nuts_manifest": str(nuts_csv),
            "cells_manifest": str(cells_csv),
            "skipped_cells": str(skipped_csv),
            "environment": environment_record(),
        },
    )
    return {
        "nuts_manifest": nuts_csv,
        "cells_manifest": cells_csv,
        "skipped_cells": skipped_csv,
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


def parse_common_cell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-s", "--d_s", dest="d_s", type=int, choices=D_S_GRID, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-obs", "--n_obs", dest="n_obs", type=int, default=DEFAULT_N_OBS)
    parser.add_argument("--n-sims", "--n_sims", dest="n_sims", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def add_nuts_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nuts-num-samples", type=int, default=10_000)
    parser.add_argument("--nuts-num-warmup", type=int, default=10_000)
    parser.add_argument("--nuts-num-chains", type=int, default=4)
    parser.add_argument("--nuts-thinning", type=int, default=10)
    parser.add_argument("--max-rhat", type=float, default=1.05)
    parser.add_argument("--min-ess", type=float, default=1000.0)
    parser.add_argument("--skip-nuts-diagnostic-gate", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--force", action="store_true")


def add_training_args(parser: argparse.ArgumentParser, *, include_gate_args: bool = True) -> None:
    parser.add_argument("--sim-batch-size", type=int, default=1000)
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--metric-samples", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.1)
    if include_gate_args:
        parser.add_argument("--max-rhat", type=float, default=1.05)
        parser.add_argument("--min-ess", type=float, default=1000.0)
        parser.add_argument("--skip-nuts-diagnostic-gate", action="store_true")
        parser.add_argument("--force", action="store_true")


def cmd_precompute_nuts(args: argparse.Namespace) -> None:
    precompute_nuts(
        output_root=args.output_root,
        d_s=args.d_s,
        seed=args.seed,
        n_obs=args.n_obs,
        num_samples=args.nuts_num_samples,
        num_warmup=args.nuts_num_warmup,
        num_chains=args.nuts_num_chains,
        thinning=args.nuts_thinning,
        max_rhat=args.max_rhat,
        min_ess=args.min_ess,
        skip_diagnostic_gate=args.skip_nuts_diagnostic_gate,
        force=args.force,
        progress_bar=args.progress_bar,
    )


def cmd_run_cell(args: argparse.Namespace) -> None:
    n_sims = args.n_sims if args.n_sims is not None else scaled_n_sims(args.d_s, args.n_obs)
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
        max_rhat=args.max_rhat,
        min_ess=args.min_ess,
        skip_nuts_diagnostic_gate=args.skip_nuts_diagnostic_gate,
        force=args.force,
    )


def cmd_aggregate(args: argparse.Namespace) -> None:
    outputs = aggregate_results(output_root=args.output_root, plot_dir=args.plot_dir, timestamp=args.timestamp)
    for name, path in outputs.items():
        print(f"{name}: {path}")


def cmd_prepare_manifests(args: argparse.Namespace) -> None:
    outputs = prepare_manifests(
        output_root=args.output_root,
        n_obs=args.n_obs,
        scale=args.scale,
        num_seeds=args.num_seeds,
        timestamp=args.timestamp,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def cmd_prepare_csweep_manifests(args: argparse.Namespace) -> None:
    outputs = prepare_csweep_manifests(
        output_root=args.output_root,
        n_obs=args.n_obs,
        scales=tuple(args.scales),
        num_seeds=args.num_seeds,
        max_n_sims=args.max_n_sims,
        timestamp=args.timestamp,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def cmd_aggregate_overnight(args: argparse.Namespace) -> None:
    path = aggregate_overnight_csv(
        output_root=args.output_root,
        output_csv=args.output_csv,
        staged_csv=args.staged_csv,
    )
    print(f"aggregated_csv: {path}")


def cmd_run_manifest_row(args: argparse.Namespace) -> None:
    row = load_manifest_row(args.manifest, args.array_index, args.kind)
    d_s = int(row["d_s"])
    seed = int(row["seed"])
    n_obs = int(row["n_obs"])
    n_sims = int(row["n_sims"])
    if row["kind"] == "nuts":
        precompute_nuts(
            output_root=args.output_root,
            d_s=d_s,
            seed=seed,
            n_obs=n_obs,
            num_samples=args.nuts_num_samples,
            num_warmup=args.nuts_num_warmup,
            num_chains=args.nuts_num_chains,
            thinning=args.nuts_thinning,
            max_rhat=args.max_rhat,
            min_ess=args.min_ess,
            skip_diagnostic_gate=args.skip_nuts_diagnostic_gate,
            force=args.force,
            progress_bar=args.progress_bar,
        )
    elif row["kind"] == "cell":
        run_cell(
            output_root=args.output_root,
            d_s=d_s,
            method=row["method"],
            seed=seed,
            n_obs=n_obs,
            n_sims=n_sims,
            sim_batch_size=args.sim_batch_size,
            num_posterior_samples=args.num_posterior_samples,
            metric_samples=args.metric_samples,
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            val_frac=args.val_frac,
            max_rhat=args.max_rhat,
            min_ess=args.min_ess,
            skip_nuts_diagnostic_gate=args.skip_nuts_diagnostic_gate,
            force=args.force,
        )
    else:
        raise ValueError(f"unsupported manifest row kind={row['kind']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("precompute-nuts", help="Build one NUTS reference cache.")
    parse_common_cell_args(p)
    add_nuts_args(p)
    p.set_defaults(func=cmd_precompute_nuts)

    p = sub.add_parser("run-cell", help="Train and evaluate one dimension-scaling cell.")
    parse_common_cell_args(p)
    p.add_argument("--method", choices=METHODS, required=True)
    add_training_args(p)
    p.set_defaults(func=cmd_run_cell)

    p = sub.add_parser("aggregate", help="Aggregate completed cells into CSVs and plots.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    p.add_argument("--timestamp")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("prepare-manifests", help="Write explicit PBS array manifests.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--n-obs", "--n_obs", dest="n_obs", type=int, default=DEFAULT_N_OBS)
    p.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    p.add_argument("--num-seeds", type=int, default=30)
    p.add_argument("--timestamp")
    p.set_defaults(func=cmd_prepare_manifests)

    p = sub.add_parser("prepare-csweep-manifests", help="Write multi-scale PBS array manifests with cap logging.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--n-obs", "--n_obs", dest="n_obs", type=int, default=DEFAULT_N_OBS)
    p.add_argument("--scales", nargs="+", type=int, default=[DEFAULT_SCALE])
    p.add_argument("--num-seeds", type=int, default=30)
    p.add_argument("--max-n-sims", type=int, default=6_000_000)
    p.add_argument("--timestamp")
    p.set_defaults(func=cmd_prepare_csweep_manifests)

    p = sub.add_parser("run-manifest-row", help="Run one row from a prepared manifest.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--kind", choices=("nuts", "cell"))
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    add_nuts_args(p)
    add_training_args(p, include_gate_args=False)
    p.set_defaults(func=cmd_run_manifest_row)

    p = sub.add_parser("aggregate-overnight", help="Write the Section 3.3 overnight aggregate CSV.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--staged-csv", type=Path)
    p.set_defaults(func=cmd_aggregate_overnight)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "array_index", None) is None and os.environ.get("PBS_ARRAY_INDEX") is not None:
        args.array_index = int(os.environ["PBS_ARRAY_INDEX"])
    args.func(args)


if __name__ == "__main__":
    main()
