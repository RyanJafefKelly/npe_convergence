"""Staged high-budget runners for GNK and stereological experiments.

This module intentionally leaves the legacy monolithic runners untouched.  It
adds cacheable simulation shards, resumable training, coverage shards, and
phase diagnostics so high-budget cells can be profiled and resumed under strict
PBS walltime limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle as pkl
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.scipy.special import expit, logit
from numpyro.diagnostics import hpdi  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import gnk, ss_octile, get_summaries_batches as gnk_summaries_batches
from npe_convergence.examples.stereological import (
    get_prior_samples as stereo_prior_samples,
    get_summaries as stereo_summaries,
    get_summaries_batches as stereo_summaries_batches,
    stereological,
    transform_to_bounded as stereo_to_bounded,
    transform_to_unbounded as stereo_to_unbounded,
)


DEFAULT_STAGING_ROOT = REPO_ROOT / "res" / "staged_high_budget"
DEFAULT_WALLTIME_BUFFER_SECONDS = 15 * 60
COVERAGE_LEVELS = (0.8, 0.9, 0.95)


@dataclass(frozen=True)
class ModelSpec:
    model: str
    theta_dims: int
    summary_dims: int
    true_params: tuple[float, ...]


MODEL_SPECS = {
    "gnk": ModelSpec("gnk", theta_dims=4, summary_dims=7, true_params=(3.0, 1.0, 2.0, 0.5)),
    "stereological": ModelSpec(
        "stereological",
        theta_dims=3,
        summary_dims=4,
        true_params=(100.0, 2.0, -0.1),
    ),
}


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def rng_for(seed: int, *parts: object):
    key = random.key(seed)
    for part in parts:
        key = random.fold_in(key, stable_int(part))
    return key


def resolve_root(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def run_dir(root: Path, model: str, method: str, seed: int, n_obs: int, n_sims: int) -> Path:
    return root / f"{model}_{method}_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}"


def diagnostics_path(base: Path) -> Path:
    return base / "diagnostics.jsonl"


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


def log_event(base: Path, phase: str, event: str, **payload: Any) -> None:
    base.mkdir(parents=True, exist_ok=True)
    record = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": phase,
        "event": event,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "host": platform.node(),
        **payload,
    }
    line = json.dumps(record, sort_keys=True, default=json_default)
    with diagnostics_path(base).open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def environment_record() -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runner_path": str(runner_path),
        "runner_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
    }
    git_commands = {
        "git_commit": ["git", "rev-parse", "HEAD"],
        "git_branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    }
    for key, command in git_commands.items():
        try:
            record[key] = subprocess.check_output(
                command,
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
        except Exception:  # pragma: no cover - depends on checkout metadata
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
    except Exception:  # pragma: no cover - depends on checkout metadata
        record["git_dirty"] = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
        record["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else None
        record["nvidia_smi_error"] = result.stderr.strip() if result.returncode != 0 else None
    except Exception as exc:  # pragma: no cover - depends on host tooling
        record["nvidia_smi"] = None
        record["nvidia_smi_error"] = repr(exc)
    return record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


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


def save_equinox_tree(path: Path, payload: Any) -> None:
    import equinox as eqx

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    eqx.tree_serialise_leaves(str(tmp), payload)
    tmp.replace(path)


def load_equinox_tree(path: Path, template: Any) -> Any:
    import equinox as eqx

    return eqx.tree_deserialise_leaves(str(path), template)


def safe_standardise(arr: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    floored = (~np.isfinite(std)) | (std < eps)
    std = std.copy()
    std[floored] = 1.0
    return (arr - mean) / std, mean, std, floored


def finite_rows(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr).all(axis=1)
    return mask


def gnk_to_unbounded(theta_bounded: jax.Array) -> jax.Array:
    return logit(theta_bounded / 10.0)


def gnk_to_bounded(theta_unbounded: jax.Array) -> jax.Array:
    return expit(theta_unbounded) * 10.0


def observed_summary(model: str, seed: int, n_obs: int) -> np.ndarray:
    if model == "gnk":
        true_params = jnp.array(MODEL_SPECS[model].true_params)
        z = random.normal(random.key(seed), shape=(n_obs,))
        return np.asarray(jnp.squeeze(ss_octile(jnp.atleast_2d(gnk(z, *true_params)))))
    if model == "stereological":
        true_params = jnp.array(MODEL_SPECS[model].true_params)
        key, subkey = random.split(random.key(seed))
        x_obs = stereological(subkey, *true_params, num_samples=1, n_obs=n_obs)
        return np.asarray(jnp.squeeze(stereo_summaries(x_obs)))
    raise ValueError(f"unsupported model: {model}")


def coverage_observed_summary(model: str, seed: int, n_obs: int, replicate_index: int) -> np.ndarray:
    key = rng_for(seed, model, "coverage-observed", replicate_index)
    true_params = jnp.array(MODEL_SPECS[model].true_params)
    if model == "gnk":
        a, b, g, k = [jnp.array([value]) for value in true_params]
        x_obs = gnk_summaries_batches(key, a, b, g, k, n_obs=n_obs, n_sims=1, batch_size=1)
        return np.asarray(jnp.squeeze(x_obs))
    if model == "stereological":
        x_obs = stereological(key, *true_params, num_samples=1, n_obs=n_obs)
        return np.asarray(jnp.squeeze(stereo_summaries(x_obs)))
    raise ValueError(f"unsupported model: {model}")


def transform_to_bounded(model: str, samples: jax.Array) -> jax.Array:
    if model == "gnk":
        return gnk_to_bounded(samples)
    if model == "stereological":
        return stereo_to_bounded(samples)
    raise ValueError(f"unsupported model: {model}")


def transform_standardised_to_bounded(
    model: str,
    samples_standardised: jax.Array,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
) -> jax.Array:
    theta_unbounded = jnp.asarray(samples_standardised) * jnp.asarray(theta_std) + jnp.asarray(theta_mean)
    return transform_to_bounded(model, theta_unbounded)


def simulate_one_shard(
    *,
    model: str,
    seed: int,
    n_obs: int,
    n_sims: int,
    shard_index: int,
    shard_size: int,
    sim_batch_size: int,
) -> dict[str, np.ndarray]:
    shard_start = shard_index * shard_size
    if shard_start >= n_sims:
        raise ValueError(f"shard_index={shard_index} starts beyond n_sims={n_sims}")
    shard_n = min(shard_size, n_sims - shard_start)
    theta_key, sim_key = random.split(rng_for(seed, model, "simulate-shard", shard_index))

    if model == "gnk":
        tol = 1e-6
        theta_bounded = jax.random.uniform(theta_key, (shard_n, 4), minval=tol, maxval=10.0 - tol)
        theta_unbounded = gnk_to_unbounded(theta_bounded)
        A, B, g, k = theta_bounded.T
        summaries = gnk_summaries_batches(
            sim_key,
            A,
            B,
            g,
            k,
            n_obs=n_obs,
            n_sims=shard_n,
            batch_size=min(sim_batch_size, shard_n),
        ).T
    elif model == "stereological":
        theta_bounded = stereo_prior_samples(theta_key, shard_n)
        theta_unbounded = stereo_to_unbounded(theta_bounded)
        summaries = stereo_summaries_batches(
            sim_key,
            theta_bounded,
            n_obs=n_obs,
            n_sims=shard_n,
            batch_size=min(sim_batch_size, shard_n),
        )
    else:
        raise ValueError(f"unsupported model: {model}")

    return {
        "theta_bounded": np.asarray(theta_bounded),
        "theta_unbounded": np.asarray(theta_unbounded),
        "summaries": np.asarray(summaries),
        "shard_start": np.array(shard_start, dtype=np.int64),
        "shard_n": np.array(shard_n, dtype=np.int64),
    }


def cmd_probe_env(args: argparse.Namespace) -> None:
    root = resolve_root(args.staging_root)
    base = root / "environment_probe"
    log_event(base, "probe-env", "environment", **environment_record())
    if args.require_gpu and jax.default_backend() != "gpu":
        raise SystemExit("GPU was required, but JAX default backend is not gpu")


def cmd_simulate_shard(args: argparse.Namespace) -> None:
    root = resolve_root(args.staging_root)
    base = run_dir(root, args.model, args.method, args.seed, args.n_obs, args.n_sims)
    shard_dir = base / "sim_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    out = shard_dir / f"sim_shard_{args.shard_index:05d}.npz"
    if out.exists() and not args.force:
        log_event(base, "simulate-shard", "skip-existing", path=out)
        return

    start = time.perf_counter()
    log_event(
        base,
        "simulate-shard",
        "start",
        model=args.model,
        method=args.method,
        seed=args.seed,
        n_obs=args.n_obs,
        n_sims=args.n_sims,
        shard_index=args.shard_index,
        shard_size=args.shard_size,
        sim_batch_size=args.sim_batch_size,
        environment=environment_record(),
    )
    shard = simulate_one_shard(
        model=args.model,
        seed=args.seed,
        n_obs=args.n_obs,
        n_sims=args.n_sims,
        shard_index=args.shard_index,
        shard_size=args.shard_size,
        sim_batch_size=args.sim_batch_size,
    )
    elapsed = time.perf_counter() - start
    metadata = {
        "model": args.model,
        "method": args.method,
        "seed": args.seed,
        "n_obs": args.n_obs,
        "n_sims": args.n_sims,
        "shard_index": args.shard_index,
        "shard_size": args.shard_size,
        "sim_batch_size": args.sim_batch_size,
        "elapsed_seconds": elapsed,
        "samples_per_second": float(shard["shard_n"]) / elapsed if elapsed > 0 else None,
        "environment": environment_record(),
    }
    np.savez_compressed(
        out,
        theta_bounded=shard["theta_bounded"],
        theta_unbounded=shard["theta_unbounded"],
        summaries=shard["summaries"],
        shard_start=shard["shard_start"],
        shard_n=shard["shard_n"],
        metadata=json.dumps(metadata, sort_keys=True),
    )
    write_json(out.with_suffix(".json"), metadata)
    log_event(
        base,
        "simulate-shard",
        "finish",
        path=out,
        elapsed_seconds=elapsed,
        shard_n=int(shard["shard_n"]),
        samples_per_second=metadata["samples_per_second"],
    )


def cmd_aggregate_sims(args: argparse.Namespace) -> None:
    root = resolve_root(args.staging_root)
    base = run_dir(root, args.model, args.method, args.seed, args.n_obs, args.n_sims)
    shard_files = sorted((base / "sim_shards").glob("sim_shard_*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"no simulation shards found under {base / 'sim_shards'}")
    if args.expected_shards is not None and len(shard_files) != args.expected_shards:
        raise ValueError(f"expected {args.expected_shards} shards, found {len(shard_files)}")

    start = time.perf_counter()
    log_event(base, "aggregate-sims", "start", shard_count=len(shard_files))
    theta_bounded_parts = []
    theta_unbounded_parts = []
    summary_parts = []
    shard_meta = []
    for path in shard_files:
        data = np.load(path, allow_pickle=False)
        theta_bounded_parts.append(data["theta_bounded"])
        theta_unbounded_parts.append(data["theta_unbounded"])
        summary_parts.append(data["summaries"])
        shard_meta.append(json.loads(str(data["metadata"])))

    theta_bounded = np.concatenate(theta_bounded_parts, axis=0)
    theta_unbounded = np.concatenate(theta_unbounded_parts, axis=0)
    summaries = np.concatenate(summary_parts, axis=0)
    if theta_bounded.shape[0] > args.n_sims:
        theta_bounded = theta_bounded[: args.n_sims]
        theta_unbounded = theta_unbounded[: args.n_sims]
        summaries = summaries[: args.n_sims]

    row_mask = finite_rows(theta_unbounded, summaries)
    theta_bounded = theta_bounded[row_mask]
    theta_unbounded = theta_unbounded[row_mask]
    summaries = summaries[row_mask]
    theta_train, theta_mean, theta_std, theta_floored = safe_standardise(theta_unbounded)
    summary_train, summary_mean, summary_std, summary_floored = safe_standardise(summaries)
    x_obs_raw = observed_summary(args.model, args.seed, args.n_obs)
    x_obs_std = (x_obs_raw - summary_mean) / summary_std

    out = base / "training_data.npz"
    np.savez_compressed(
        out,
        theta_bounded=theta_bounded,
        theta_unbounded=theta_unbounded,
        summaries=summaries,
        theta_train=theta_train,
        summary_train=summary_train,
        theta_mean=theta_mean,
        theta_std=theta_std,
        summary_mean=summary_mean,
        summary_std=summary_std,
        x_obs_raw=x_obs_raw,
        x_obs_std=x_obs_std,
        true_params=np.asarray(MODEL_SPECS[args.model].true_params),
        finite_row_mask=row_mask,
    )
    metadata = {
        "model": args.model,
        "method": args.method,
        "seed": args.seed,
        "n_obs": args.n_obs,
        "n_sims": args.n_sims,
        "shard_count": len(shard_files),
        "raw_rows": int(row_mask.shape[0]),
        "finite_rows": int(row_mask.sum()),
        "dropped_nonfinite_rows": int(row_mask.shape[0] - row_mask.sum()),
        "theta_std_floored_count": int(theta_floored.sum()),
        "summary_std_floored_count": int(summary_floored.sum()),
        "elapsed_seconds": time.perf_counter() - start,
        "shards": shard_meta,
        "environment": environment_record(),
    }
    write_json(base / "training_data.json", metadata)
    log_event(base, "aggregate-sims", "finish", path=out, **metadata)


def _load_training_data(base: Path) -> dict[str, np.ndarray]:
    path = base / "training_data.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing aggregated training data: {path}")
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _split_train_val(n: int, seed: int, model: str, method: str, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    key = rng_for(seed, model, method, "train-val-split")
    perm = np.asarray(random.permutation(key, n))
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]


def _init_model(method: str, theta_dims: int, summary_dims: int, key: jax.Array):
    if method == "gaussian_npe":
        return init_gaussian_params(summary_dims, theta_dims, (128, 128), key)
    if method == "flow_npe":
        from flowjax.bijections import RationalQuadraticSpline  # type: ignore
        from flowjax.distributions import Normal  # type: ignore
        from flowjax.flows import coupling_flow  # type: ignore

        return coupling_flow(
            key=key,
            base_dist=Normal(jnp.zeros(theta_dims)),
            transformer=RationalQuadraticSpline(knots=10, interval=5),
            cond_dim=summary_dims,
            nn_depth=2,
        )
    raise ValueError(f"unsupported method: {method}")


def init_gaussian_params(
    summary_dims: int,
    theta_dims: int,
    hidden_dims: tuple[int, ...],
    key: jax.Array,
) -> dict[str, Any]:
    dims = (summary_dims, *hidden_dims)
    keys = random.split(key, len(hidden_dims) + 2)
    hidden = []
    for i, width in enumerate(hidden_dims):
        fan_in = dims[i]
        scale = math.sqrt(2.0 / fan_in)
        hidden.append(
            {
                "w": random.normal(keys[i], (fan_in, width)) * scale,
                "b": jnp.zeros((width,)),
            }
        )
    last_dim = hidden_dims[-1] if hidden_dims else summary_dims
    n_chol = theta_dims * (theta_dims + 1) // 2
    return {
        "hidden": hidden,
        "mu": {
            "w": random.normal(keys[-2], (last_dim, theta_dims)) * math.sqrt(2.0 / last_dim),
            "b": jnp.zeros((theta_dims,)),
        },
        "chol": {
            "w": random.normal(keys[-1], (last_dim, n_chol)) * math.sqrt(2.0 / last_dim),
            "b": jnp.zeros((n_chol,)),
        },
    }


def gaussian_forward(params: dict[str, Any], summary: jax.Array) -> tuple[jax.Array, jax.Array]:
    h = summary
    for layer in params["hidden"]:
        h = jax.nn.relu(h @ layer["w"] + layer["b"])
    mu = h @ params["mu"]["w"] + params["mu"]["b"]
    chol_entries = h @ params["chol"]["w"] + params["chol"]["b"]
    theta_dims = params["mu"]["b"].shape[0]
    L = jnp.zeros((theta_dims, theta_dims)).at[jnp.tril_indices(theta_dims)].set(chol_entries)
    diag = jnp.diag_indices(theta_dims)
    L = L.at[diag].set(jax.nn.softplus(L[diag]) + 1e-6)
    return mu, L


def gaussian_nll(params: dict[str, Any], theta: jax.Array, summary: jax.Array) -> jax.Array:
    mu, L = gaussian_forward(params, summary)
    diff = theta - mu
    z = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    log_det = jnp.sum(jnp.log(jnp.diag(L)))
    d = theta.shape[0]
    return 0.5 * d * jnp.log(2.0 * jnp.pi) + log_det + 0.5 * jnp.dot(z, z)


def gaussian_batch_loss(params: dict[str, Any], theta: jax.Array, summary: jax.Array) -> jax.Array:
    return jax.vmap(lambda t, s: gaussian_nll(params, t, s))(theta, summary).mean()


def gaussian_sample(params: dict[str, Any], summary: jax.Array, n_samples: int, key: jax.Array) -> jax.Array:
    mu, L = gaussian_forward(params, summary)
    z = random.normal(key, (n_samples, mu.shape[0]))
    return mu + z @ L.T


def _make_train_fns(method: str, learning_rate: float):
    import optax

    if method == "gaussian_npe":

        optimizer = optax.adam(learning_rate)

        @jax.jit
        def step(model, opt_state, theta_batch, summary_batch):
            loss, grads = jax.value_and_grad(gaussian_batch_loss)(model, theta_batch, summary_batch)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = optax.apply_updates(model, updates)
            return model, opt_state, loss

        @jax.jit
        def eval_loss(model, theta, summary):
            return gaussian_batch_loss(model, theta, summary)

        def init_opt_state(model):
            return optimizer.init(model)

        return init_opt_state, step, eval_loss

    elif method == "flow_npe":
        import equinox as eqx

        optimizer = optax.adam(learning_rate)

        def batch_loss(model, theta, summary):
            return -jnp.mean(model.log_prob(theta, condition=summary))

        @eqx.filter_jit
        def step(model, opt_state, theta_batch, summary_batch):
            loss, grads = eqx.filter_value_and_grad(batch_loss)(model, theta_batch, summary_batch)
            updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        @eqx.filter_jit
        def eval_loss(model, theta, summary):
            return batch_loss(model, theta, summary)

        def init_opt_state(model):
            return optimizer.init(eqx.filter(model, eqx.is_array))

        return init_opt_state, step, eval_loss

    else:
        raise ValueError(f"unsupported method: {method}")


def _training_state_path(base: Path, method: str) -> Path:
    return base / "training" / f"{method}_state.pkl"


def _training_tree_path(base: Path, method: str, name: str) -> Path:
    return base / "training" / f"{method}_{name}.eqx"


def _training_complete_path(base: Path, method: str) -> Path:
    return base / "training" / f"{method}_complete.json"


def _save_training_state(base: Path, method: str, state: dict[str, Any]) -> None:
    payload = dict(state)
    if method == "flow_npe":
        save_equinox_tree(_training_tree_path(base, method, "model"), state["model"])
        save_equinox_tree(_training_tree_path(base, method, "best_model"), state["best_model"])
        save_equinox_tree(_training_tree_path(base, method, "opt_state"), state["opt_state"])
        payload.pop("model", None)
        payload.pop("best_model", None)
        payload.pop("opt_state", None)
        payload["tree_checkpoints"] = {
            "model": _training_tree_path(base, method, "model").name,
            "best_model": _training_tree_path(base, method, "best_model").name,
            "opt_state": _training_tree_path(base, method, "opt_state").name,
        }
    save_pickle(_training_state_path(base, method), payload)
    write_json(
        base / "training" / f"{method}_losses.json",
        {
            "epoch": state["epoch"],
            "best_epoch": state.get("best_epoch"),
            "best_val_loss": state.get("best_val_loss"),
            "wait": state.get("wait"),
            "losses": state["losses"],
        },
    )


def _flow_template_from_state(state: dict[str, Any]) -> Any:
    model_init = state["model_init"]
    key = rng_for(model_init["seed"], model_init["model"], model_init["method"], "model-init")
    return _init_model(
        model_init["method"],
        int(model_init["theta_dims"]),
        int(model_init["summary_dims"]),
        key,
    )


def _load_training_state(args: argparse.Namespace, base: Path, data: dict[str, np.ndarray]) -> dict[str, Any]:
    state = load_pickle(_training_state_path(base, args.method))
    if args.method != "flow_npe":
        return state

    if "model_init" not in state:
        state["model_init"] = {
            "seed": args.seed,
            "model": args.model,
            "method": args.method,
            "theta_dims": int(data["theta_train"].shape[1]),
            "summary_dims": int(data["summary_train"].shape[1]),
        }

    model_template = _flow_template_from_state(state)
    best_model_template = _flow_template_from_state(state)
    state["model"] = load_equinox_tree(_training_tree_path(base, args.method, "model"), model_template)
    state["best_model"] = load_equinox_tree(
        _training_tree_path(base, args.method, "best_model"),
        best_model_template,
    )
    init_opt_state, _, _ = _make_train_fns(args.method, float(state["config"]["learning_rate"]))
    opt_template = init_opt_state(state["model"])
    state["opt_state"] = load_equinox_tree(_training_tree_path(base, args.method, "opt_state"), opt_template)
    return state


def _load_or_init_training_state(args: argparse.Namespace, base: Path, data: dict[str, np.ndarray]) -> dict[str, Any]:
    state_path = _training_state_path(base, args.method)
    if state_path.exists() and not args.restart_training:
        return _load_training_state(args, base, data)

    key = rng_for(args.seed, args.model, args.method, "model-init")
    model = _init_model(args.method, data["theta_train"].shape[1], data["summary_train"].shape[1], key)
    init_opt_state, _, _ = _make_train_fns(args.method, args.learning_rate)
    opt_state = init_opt_state(model)
    train_idx, val_idx = _split_train_val(
        data["theta_train"].shape[0],
        args.seed,
        args.model,
        args.method,
        args.val_frac,
    )
    return {
        "model": model,
        "best_model": model,
        "opt_state": opt_state,
        "epoch": 0,
        "best_epoch": None,
        "best_val_loss": math.inf,
        "wait": 0,
        "losses": {"train": [], "val": [], "epoch_seconds": []},
        "train_idx": train_idx,
        "val_idx": val_idx,
        "rng_key": rng_for(args.seed, args.model, args.method, "training-loop"),
        "model_init": {
            "seed": args.seed,
            "model": args.model,
            "method": args.method,
            "theta_dims": int(data["theta_train"].shape[1]),
            "summary_dims": int(data["summary_train"].shape[1]),
        },
        "config": {
            "learning_rate": args.learning_rate,
            "batch_size": args.train_batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "val_frac": args.val_frac,
        },
    }


def cmd_train(args: argparse.Namespace) -> None:
    root = resolve_root(args.staging_root)
    base = run_dir(root, args.model, args.method, args.seed, args.n_obs, args.n_sims)
    complete_path = _training_complete_path(base, args.method)
    if complete_path.exists() and not args.restart_training:
        complete = load_json(complete_path)
        if complete.get("complete"):
            log_event(
                base,
                "train",
                "skip-complete",
                method=args.method,
                complete_path=complete_path,
                stop_reason=complete.get("stop_reason"),
                epoch=complete.get("epoch"),
            )
            return
    data = _load_training_data(base)
    train_theta = jnp.asarray(data["theta_train"])
    train_summary = jnp.asarray(data["summary_train"])
    state = _load_or_init_training_state(args, base, data)
    _, step, eval_loss = _make_train_fns(args.method, args.learning_rate)

    train_idx = np.asarray(state["train_idx"])
    val_idx = np.asarray(state["val_idx"])
    n_batches = max(1, math.ceil(train_idx.shape[0] / args.train_batch_size))
    run_start = time.perf_counter()
    stop_reason = None
    max_epoch_this_run = min(args.max_epochs, state["epoch"] + args.epochs_this_run)
    log_event(
        base,
        "train",
        "start",
        method=args.method,
        start_epoch=state["epoch"],
        max_epoch_this_run=max_epoch_this_run,
        train_rows=int(train_idx.shape[0]),
        val_rows=int(val_idx.shape[0]),
        batch_size=args.train_batch_size,
        environment=environment_record(),
    )

    while state["epoch"] < max_epoch_this_run:
        if args.max_runtime_seconds is not None:
            elapsed = time.perf_counter() - run_start
            if elapsed > max(0, args.max_runtime_seconds - args.walltime_buffer_seconds):
                stop_reason = "walltime-buffer"
                break

        epoch_start = time.perf_counter()
        key, subkey = random.split(state["rng_key"])
        state["rng_key"] = key
        perm = np.asarray(random.permutation(subkey, train_idx.shape[0]))
        epoch_loss = 0.0
        for batch in range(n_batches):
            idx = perm[batch * args.train_batch_size : (batch + 1) * args.train_batch_size]
            row_idx = train_idx[idx]
            state["model"], state["opt_state"], loss = step(
                state["model"],
                state["opt_state"],
                train_theta[row_idx],
                train_summary[row_idx],
            )
            epoch_loss += float(loss)

        train_loss = epoch_loss / n_batches
        val_loss = float(eval_loss(state["model"], train_theta[val_idx], train_summary[val_idx]))
        epoch_seconds = time.perf_counter() - epoch_start
        epoch = state["epoch"]
        state["losses"]["train"].append(train_loss)
        state["losses"]["val"].append(val_loss)
        state["losses"]["epoch_seconds"].append(epoch_seconds)

        if val_loss < state["best_val_loss"]:
            state["best_val_loss"] = val_loss
            state["best_epoch"] = epoch
            state["best_model"] = state["model"]
            state["wait"] = 0
        else:
            state["wait"] += 1

        state["epoch"] += 1
        if epoch % args.log_every_epochs == 0:
            log_event(
                base,
                "train",
                "epoch",
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=state["best_val_loss"],
                wait=state["wait"],
                epoch_seconds=epoch_seconds,
            )
        if state["epoch"] % args.checkpoint_every_epochs == 0:
            _save_training_state(base, args.method, state)
        if state["wait"] >= args.patience:
            stop_reason = "patience"
            break

    if stop_reason is None:
        stop_reason = "max-epochs" if state["epoch"] >= args.max_epochs else "epochs-this-run"
    _save_training_state(base, args.method, state)
    complete = stop_reason in {"patience", "max-epochs"}
    write_json(
        _training_complete_path(base, args.method),
        {
            "complete": complete,
            "stop_reason": stop_reason,
            "epoch": state["epoch"],
            "best_epoch": state["best_epoch"],
            "best_val_loss": state["best_val_loss"],
            "elapsed_seconds_this_run": time.perf_counter() - run_start,
            "environment": environment_record(),
        },
    )
    log_event(
        base,
        "train",
        "finish",
        complete=complete,
        stop_reason=stop_reason,
        epoch=state["epoch"],
        best_epoch=state["best_epoch"],
        best_val_loss=state["best_val_loss"],
    )


def _load_best_model(base: Path, method: str):
    state_path = _training_state_path(base, method)
    if not state_path.exists():
        raise FileNotFoundError(f"missing training checkpoint: {state_path}")
    state = load_pickle(state_path)
    if method == "flow_npe":
        if "model_init" not in state:
            raise ValueError(f"flow checkpoint is missing model_init metadata: {state_path}")
        template = _flow_template_from_state(state)
        return load_equinox_tree(_training_tree_path(base, method, "best_model"), template)
    return state["best_model"]


def sample_posterior(method: str, model_obj: Any, condition: np.ndarray, n_samples: int, key: jax.Array) -> jax.Array:
    condition_jax = jnp.asarray(condition)
    if method == "gaussian_npe":
        return gaussian_sample(model_obj, condition_jax, n_samples, key)
    if method == "flow_npe":
        return model_obj.sample(key, sample_shape=(n_samples,), condition=condition_jax)
    raise ValueError(f"unsupported method: {method}")


def cmd_evaluate_shard(args: argparse.Namespace) -> None:
    root = resolve_root(args.staging_root)
    base = run_dir(root, args.model, args.method, args.seed, args.n_obs, args.n_sims)
    complete_path = _training_complete_path(base, args.method)
    if not complete_path.exists():
        raise FileNotFoundError(f"missing training completion file: {complete_path}")
    complete = load_json(complete_path)
    if not complete.get("complete"):
        raise RuntimeError(
            "training is not marked complete; refusing to evaluate "
            f"epoch={complete.get('epoch')} stop_reason={complete.get('stop_reason')}"
        )
    data = _load_training_data(base)
    model_obj = _load_best_model(base, args.method)
    eval_dir = base / "eval_shards"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out = eval_dir / f"eval_shard_{args.shard_index:05d}.npz"
    if out.exists() and not args.force:
        log_event(base, "evaluate-shard", "skip-existing", path=out)
        return

    spec = MODEL_SPECS[args.model]
    counts = np.zeros((spec.theta_dims, len(COVERAGE_LEVELS)), dtype=np.int64)
    biases: list[np.ndarray] = []
    posterior_samples = None
    start = time.perf_counter()
    log_event(
        base,
        "evaluate-shard",
        "start",
        shard_index=args.shard_index,
        coverage_reps=args.coverage_reps,
        posterior_samples=args.num_posterior_samples,
        include_posterior=args.include_posterior,
        environment=environment_record(),
    )

    if args.include_posterior:
        key = rng_for(args.seed, args.model, args.method, "posterior", args.shard_index)
        posterior_std = sample_posterior(args.method, model_obj, data["x_obs_std"], args.num_posterior_samples, key)
        posterior_samples = np.asarray(
            transform_standardised_to_bounded(args.model, posterior_std, data["theta_mean"], data["theta_std"])
        )

    for offset in range(args.coverage_reps):
        replicate = args.shard_index * args.coverage_reps + offset
        x_obs_raw = coverage_observed_summary(args.model, args.seed, args.n_obs, replicate)
        x_obs_std = (x_obs_raw - data["summary_mean"]) / data["summary_std"]
        key = rng_for(args.seed, args.model, args.method, "coverage-posterior", replicate)
        posterior_std = sample_posterior(args.method, model_obj, x_obs_std, args.num_posterior_samples, key)
        posterior = np.asarray(
            transform_standardised_to_bounded(args.model, posterior_std, data["theta_mean"], data["theta_std"])
        )
        true_params = np.asarray(spec.true_params)
        biases.append(np.mean(posterior, axis=0) - true_params)
        for dim in range(spec.theta_dims):
            samples_i = jnp.asarray(posterior[:, dim])
            for level_index, level in enumerate(COVERAGE_LEVELS):
                lower, upper = hpdi(samples_i, level)
                if float(lower) < true_params[dim] < float(upper):
                    counts[dim, level_index] += 1

    elapsed = time.perf_counter() - start
    payload = {
        "coverage_counts": counts,
        "biases": np.asarray(biases).reshape(-1),
        "coverage_reps": np.array(args.coverage_reps, dtype=np.int64),
        "metadata": json.dumps(
            {
                "model": args.model,
                "method": args.method,
                "seed": args.seed,
                "n_obs": args.n_obs,
                "n_sims": args.n_sims,
                "shard_index": args.shard_index,
                "coverage_reps": args.coverage_reps,
                "num_posterior_samples": args.num_posterior_samples,
                "posterior_coordinate": "training-standardised theta",
                "posterior_transform": "theta_unbounded = theta_standardised * theta_std + theta_mean; then bounded transform",
                "elapsed_seconds": elapsed,
                "replicates_per_second": args.coverage_reps / elapsed if elapsed > 0 else None,
                "environment": environment_record(),
            },
            sort_keys=True,
            default=json_default,
        ),
    }
    if posterior_samples is not None:
        payload["posterior_samples"] = posterior_samples
    np.savez_compressed(out, **payload)
    write_json(out.with_suffix(".json"), json.loads(str(payload["metadata"])))
    log_event(
        base,
        "evaluate-shard",
        "finish",
        path=out,
        elapsed_seconds=elapsed,
        coverage_reps=args.coverage_reps,
        replicates_per_second=args.coverage_reps / elapsed if elapsed > 0 else None,
    )


def default_gnk_nuts_cache(n_obs: int, seed: int) -> Path:
    return REPO_ROOT / "res" / "gnk" / f"nuts_cache_v2_flow_n_obs_{n_obs}_seed_{seed}.pkl"


def cmd_aggregate_results(args: argparse.Namespace) -> None:
    root = resolve_root(args.staging_root)
    base = run_dir(root, args.model, args.method, args.seed, args.n_obs, args.n_sims)
    eval_files = sorted((base / "eval_shards").glob("eval_shard_*.npz"))
    if not eval_files:
        raise FileNotFoundError(f"no evaluation shards found under {base / 'eval_shards'}")

    spec = MODEL_SPECS[args.model]
    final = base / "final"
    final.mkdir(parents=True, exist_ok=True)
    counts = np.zeros((spec.theta_dims, len(COVERAGE_LEVELS)), dtype=np.int64)
    total_reps = 0
    biases = []
    posterior_samples = None
    metadata = []
    start = time.perf_counter()
    log_event(base, "aggregate-results", "start", shard_count=len(eval_files))
    for path in eval_files:
        data = np.load(path, allow_pickle=False)
        counts += data["coverage_counts"]
        total_reps += int(data["coverage_reps"])
        biases.append(data["biases"])
        if posterior_samples is None and "posterior_samples" in data.files:
            posterior_samples = data["posterior_samples"]
        metadata.append(json.loads(str(data["metadata"])))

    estimated_coverage = counts / max(1, total_reps)
    np.save(final / "estimated_coverage.npy", estimated_coverage)
    np.save(final / "biases.npy", np.concatenate(biases))
    if posterior_samples is not None:
        with (final / "posterior_samples.pkl").open("wb") as f:
            pkl.dump(posterior_samples, f)

    metric_status: dict[str, Any] = {}
    if args.model == "gnk":
        cache_path = resolve_root(args.gnk_nuts_cache) if args.gnk_nuts_cache else default_gnk_nuts_cache(args.n_obs, args.seed)
        if cache_path.exists() and posterior_samples is not None:
            from npe_convergence.metrics import kullback_leibler, median_heuristic, unbiased_mmd

            true_posterior = np.asarray(load_pickle(cache_path))
            with (final / "true_posterior_samples.pkl").open("wb") as f:
                pkl.dump(true_posterior, f)
            metric_n = min(args.metric_samples, true_posterior.shape[0], posterior_samples.shape[0])
            key = rng_for(args.seed, args.model, args.method, "metrics")
            key_a, key_b = random.split(key)
            idx_post = np.asarray(random.permutation(key_a, posterior_samples.shape[0])[:metric_n])
            idx_true = np.asarray(random.permutation(key_b, true_posterior.shape[0])[:metric_n])
            ps_thin = jnp.asarray(posterior_samples[idx_post])
            ts_thin = jnp.asarray(true_posterior[idx_true])
            kl = kullback_leibler(ts_thin, ps_thin)
            lengthscale = median_heuristic(jnp.vstack([ts_thin, ps_thin]))
            mmd = unbiased_mmd(ts_thin, ps_thin, lengthscale)
            (final / "kl.txt").write_text(str(float(kl)))
            (final / "mmd.txt").write_text(str(float(mmd)))
            metric_status = {
                "gnk_reference": str(cache_path),
                "metric_samples": metric_n,
                "kl": float(kl),
                "mmd": float(mmd),
            }
        else:
            metric_status = {
                "gnk_reference": str(cache_path),
                "metric_samples": None,
                "missing_reason": "missing reference cache or posterior_samples",
            }
            write_json(final / "metric_status.json", metric_status)

    result_metadata = {
        "model": args.model,
        "method": args.method,
        "seed": args.seed,
        "n_obs": args.n_obs,
        "n_sims": args.n_sims,
        "eval_shard_count": len(eval_files),
        "coverage_reps": total_reps,
        "elapsed_seconds": time.perf_counter() - start,
        "coverage_counts": counts.tolist(),
        "metric_status": metric_status,
        "eval_shards": metadata,
        "environment": environment_record(),
    }
    write_json(final / "result_metadata.json", result_metadata)
    log_event(base, "aggregate-results", "finish", final_dir=final, **result_metadata)


def _load_pickle_array(path: Path) -> np.ndarray:
    return np.asarray(load_pickle(path))


def cmd_verify(args: argparse.Namespace) -> None:
    root = resolve_root(args.staging_root)
    base = run_dir(root, args.model, args.method, args.seed, args.n_obs, args.n_sims)
    final = base / "final"
    spec = MODEL_SPECS[args.model]
    failures: list[str] = []
    coverage = final / "estimated_coverage.npy"
    biases = final / "biases.npy"
    if not coverage.exists():
        failures.append("missing estimated_coverage.npy")
    elif np.load(coverage).shape != (spec.theta_dims, len(COVERAGE_LEVELS)):
        failures.append("invalid estimated_coverage.npy shape")
    if not biases.exists():
        failures.append("missing biases.npy")
    else:
        expected_bias = spec.theta_dims * args.expected_coverage_samples
        if np.load(biases).shape != (expected_bias,):
            failures.append(f"invalid biases.npy shape, expected {(expected_bias,)}")

    posterior = final / "posterior_samples.pkl"
    needs_posterior = args.require_posterior or args.method == "flow_npe" or args.model == "gnk"
    if needs_posterior:
        if not posterior.exists():
            failures.append("missing posterior_samples.pkl")
        else:
            arr = _load_pickle_array(posterior)
            if arr.shape != (args.expected_posterior_samples, spec.theta_dims):
                failures.append(f"invalid posterior_samples.pkl shape: {arr.shape}")

    if args.model == "gnk":
        true = final / "true_posterior_samples.pkl"
        for name in ("kl.txt", "mmd.txt"):
            if not (final / name).exists():
                failures.append(f"missing {name}")
        if not true.exists():
            failures.append("missing true_posterior_samples.pkl")
        elif _load_pickle_array(true).shape[1] != spec.theta_dims:
            failures.append("invalid true_posterior_samples.pkl shape")

    if failures:
        log_event(base, "verify", "failed", failures=failures)
        raise SystemExit("; ".join(failures))
    log_event(base, "verify", "passed", final_dir=final)


def cmd_profile(args: argparse.Namespace) -> None:
    root = resolve_root(args.staging_root)
    base = run_dir(root, args.model, args.method, args.seed, args.n_obs, args.n_sims) / "profile"
    base.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    log_event(base, "profile", "start", environment=environment_record())
    shard = simulate_one_shard(
        model=args.model,
        seed=args.seed,
        n_obs=args.n_obs,
        n_sims=args.profile_sims,
        shard_index=0,
        shard_size=args.profile_sims,
        sim_batch_size=args.sim_batch_size,
    )
    elapsed = time.perf_counter() - start
    profile = {
        "model": args.model,
        "method": args.method,
        "seed": args.seed,
        "n_obs": args.n_obs,
        "profile_sims": args.profile_sims,
        "sim_batch_size": args.sim_batch_size,
        "elapsed_seconds": elapsed,
        "samples_per_second": int(shard["shard_n"]) / elapsed if elapsed > 0 else None,
        "projected_seconds_for_n_sims": args.n_sims / (int(shard["shard_n"]) / elapsed) if elapsed > 0 else None,
        "environment": environment_record(),
    }
    np.savez_compressed(
        base / "profile_sim_shard.npz",
        theta_bounded=shard["theta_bounded"],
        theta_unbounded=shard["theta_unbounded"],
        summaries=shard["summaries"],
    )
    write_json(base / "profile.json", profile)
    log_event(base, "profile", "finish", **profile)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Staged high-budget NPE pipeline.")
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_cell(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
        p.add_argument("--method", choices=("flow_npe", "gaussian_npe"), required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--n-obs", type=int, required=True)
        p.add_argument("--n-sims", type=int, required=True)

    p = sub.add_parser("probe-env", help="Record current JAX/device environment.")
    p.add_argument("--require-gpu", action="store_true")
    p.set_defaults(func=cmd_probe_env)

    p = sub.add_parser("profile", help="Profile one simulation shard.")
    add_cell(p)
    p.add_argument("--profile-sims", type=int, default=1000)
    p.add_argument("--sim-batch-size", type=int, default=1000)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("simulate-shard", help="Generate one simulation shard.")
    add_cell(p)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--shard-size", type=int, required=True)
    p.add_argument("--sim-batch-size", type=int, default=1000)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_simulate_shard)

    p = sub.add_parser("aggregate-sims", help="Aggregate simulation shards into training data.")
    add_cell(p)
    p.add_argument("--expected-shards", type=int)
    p.set_defaults(func=cmd_aggregate_sims)

    p = sub.add_parser("train", help="Run or resume checkpointed training.")
    add_cell(p)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=2000)
    p.add_argument("--epochs-this-run", type=int, default=20)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--checkpoint-every-epochs", type=int, default=1)
    p.add_argument("--log-every-epochs", type=int, default=1)
    p.add_argument("--max-runtime-seconds", type=float)
    p.add_argument("--walltime-buffer-seconds", type=float, default=DEFAULT_WALLTIME_BUFFER_SECONDS)
    p.add_argument("--restart-training", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate-shard", help="Run one coverage/evaluation shard.")
    add_cell(p)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--coverage-reps", type=int, default=10)
    p.add_argument("--num-posterior-samples", type=int, default=10_000)
    p.add_argument("--include-posterior", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_evaluate_shard)

    p = sub.add_parser("aggregate-results", help="Aggregate evaluation shards into final artifacts.")
    add_cell(p)
    p.add_argument("--metric-samples", type=int, default=2000)
    p.add_argument("--gnk-nuts-cache", type=Path)
    p.set_defaults(func=cmd_aggregate_results)

    p = sub.add_parser("verify", help="Verify final artifact shapes.")
    add_cell(p)
    p.add_argument("--expected-posterior-samples", type=int, default=10_000)
    p.add_argument("--expected-coverage-samples", type=int, default=100)
    p.add_argument("--require-posterior", action="store_true")
    p.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
