#!/usr/bin/env python
"""Run GNK Gaussian-NPE with robust summary standardisation.

This mirrors ``run_gnk_gaussian.py`` for the broad Uniform(0, 10)^4 prior,
but replaces per-coordinate summary z-scores with a training-set robust
transform:

    z_j = (T_j(s_j) - median_j) / (IQR_j / 1.349)

where ``T_j`` is either ``asinh(s_j / c_j)`` or identity. The transform
parameters are computed from simulated training summaries only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle as pkl
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if os.environ.get("JAX_ENABLE_X64") != "1":
    raise RuntimeError(
        "Set JAX_ENABLE_X64=1 at the process boundary before running this script."
    )

import jax

jax.config.update("jax_enable_x64", True)
assert jax.config.read("jax_enable_x64")

import jax.numpy as jnp
import jax.random as random
import matplotlib
import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
import equinox as eqx
import optax
from jax.scipy.special import expit, logit
from scipy.spatial.distance import pdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import gnk, get_summaries_batches, ss_octile
from npe_convergence.methods.gaussian_npe import (
    ConditionalGaussianNPE,
    TrainConfig,
    _batch_loss,
    sample,
)
from npe_convergence.metrics import kullback_leibler, unbiased_mmd


PARAM_NAMES = ("A", "B", "g", "k")
SUMMARY_NAMES = tuple(f"octile_{i}" for i in range(1, 8))
TRUE_THETA = jnp.asarray([3.0, 1.0, 2.0, 0.5], dtype=jnp.float64)
TRUE_THETA_FLOAT32 = jnp.asarray([3.0, 1.0, 2.0, 0.5], dtype=jnp.float32)
N_METRIC = 2000


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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def write_pickle_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pkl.dump(payload, f)
    tmp.replace(path)


def run_git(cmd: list[str], default: str | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *cmd],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except Exception:
        return default


def git_dirty() -> bool | None:
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
        return False
    except subprocess.CalledProcessError:
        return True
    except Exception:
        return None


def environment_record() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "numpyro_version": numpyro.__version__,
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": git_dirty(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def reconstruct_x_obs_float32(seed: int, n_obs: int, convention: str) -> jnp.ndarray:
    if convention == "gaussian":
        key = random.key(seed)
        _, z_key = random.split(key)
    else:
        raise ValueError(f"unknown convention: {convention}")
    z = random.normal(z_key, shape=(n_obs,), dtype=jnp.float32)
    x_raw = gnk(z, *TRUE_THETA_FLOAT32)
    return jnp.squeeze(ss_octile(jnp.atleast_2d(x_raw))).astype(jnp.float32)


def x_obs_sha256(x_obs_float32: jnp.ndarray) -> str:
    return hashlib.sha256(np.asarray(x_obs_float32).tobytes()).hexdigest()


def load_canonical_reference(
    v3_root: Path, n_obs: int, seed: int, convention: str
) -> tuple[Path, dict[str, Any], np.ndarray]:
    path = v3_root / f"nuts_n_obs_{n_obs}_seed_{seed}_conv_{convention}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"canonical reference missing: {path}")
    with path.open("rb") as f:
        fingerprint = pkl.load(f)
    grouped = np.asarray(fingerprint["samples"], dtype=np.float64)
    flat = grouped.reshape(-1, grouped.shape[-1])
    return path, fingerprint, flat


def deduplicate(samples: np.ndarray) -> tuple[np.ndarray, int]:
    unique = np.unique(np.asarray(samples, dtype=np.float64), axis=0)
    return unique, samples.shape[0] - unique.shape[0]


def deterministic_subsample(samples: np.ndarray, n: int, rng_seed: int) -> np.ndarray:
    if samples.shape[0] <= n:
        return samples
    key = random.key(rng_seed)
    idx = np.asarray(random.permutation(key, samples.shape[0])[:n])
    return samples[idx]


def median_heuristic_fast(x: np.ndarray) -> float:
    dists = pdist(np.asarray(x, dtype=np.float64), metric="euclidean")
    return float(np.sqrt(np.median(dists) / 2.0))


def column_stats(array: np.ndarray, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    payload: dict[str, Any] = {}
    for i, name in enumerate(SUMMARY_NAMES):
        rec = {
            "mean": float(np.mean(values[:, i])),
            "median": float(np.median(values[:, i])),
            "sd": float(np.std(values[:, i])),
            "iqr": float(q75[i] - q25[i]),
            "min": float(np.min(values[:, i])),
            "max": float(np.max(values[:, i])),
        }
        if extra is not None:
            for key, item in extra.items():
                rec[key] = float(np.asarray(item)[i])
        payload[name] = rec
    return payload


def compute_inner_scale(
    summaries: np.ndarray,
    mode: str,
    fixed_inner_scale: float | None,
) -> np.ndarray:
    if mode == "mad":
        raw_median = np.median(summaries, axis=0)
        mad = np.median(np.abs(summaries - raw_median), axis=0)
        return np.clip(np.maximum(1.0, mad), 1.0, 100.0).astype(np.float64)
    if mode == "fixed":
        if fixed_inner_scale is None:
            raise ValueError("--fixed-inner-scale is required when mode is fixed")
        value = float(np.clip(max(1.0, fixed_inner_scale), 1.0, 100.0))
        return np.full(summaries.shape[1], value, dtype=np.float64)
    raise ValueError(f"unknown inner scale mode: {mode}")


def transform_summaries(
    summaries: np.ndarray, transform: str, inner_scale: np.ndarray
) -> np.ndarray:
    values = np.asarray(summaries, dtype=np.float64)
    if transform == "asinh":
        return np.arcsinh(values / inner_scale)
    if transform == "identity":
        return values
    raise ValueError(f"unknown transform: {transform}")


def robust_standardise(
    summaries: np.ndarray,
    x_obs: np.ndarray,
    transform: str,
    inner_scale_mode: str,
    fixed_inner_scale: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    summaries_np = np.asarray(summaries, dtype=np.float64)
    x_obs_np = np.asarray(x_obs, dtype=np.float64)
    inner_scale = compute_inner_scale(summaries_np, inner_scale_mode, fixed_inner_scale)

    transformed = transform_summaries(summaries_np, transform, inner_scale)
    transformed_obs = transform_summaries(x_obs_np[None, :], transform, inner_scale)[0]

    center = np.median(transformed, axis=0)
    q25, q75 = np.percentile(transformed, [25.0, 75.0], axis=0)
    iqr = q75 - q25
    robust_scale = iqr / 1.349
    if np.any(~np.isfinite(robust_scale)) or np.any(robust_scale <= 0.0):
        raise ValueError(f"non-positive robust scale: {robust_scale}")

    z_summaries = (transformed - center) / robust_scale
    z_obs = (transformed_obs - center) / robust_scale

    legacy_mean = summaries_np.mean(axis=0)
    legacy_sd = summaries_np.std(axis=0)
    legacy_z = (summaries_np - legacy_mean) / legacy_sd

    stats = {
        "summary_names": SUMMARY_NAMES,
        "transform": transform,
        "inner_scale_mode": inner_scale_mode,
        "inner_scale_per_coord": inner_scale,
        "robust_center_per_coord": center,
        "robust_scale_per_coord": robust_scale,
        "raw_train": column_stats(summaries_np, {"chosen_c": inner_scale}),
        "transformed_train": column_stats(transformed, {"chosen_c": inner_scale}),
        "robust_z_train": column_stats(z_summaries, {"chosen_c": inner_scale}),
        "legacy_zscore_train": column_stats(legacy_z, {"chosen_c": inner_scale}),
        "x_obs_raw": x_obs_np,
        "x_obs_transformed": transformed_obs,
        "x_obs_robust_z": z_obs,
    }
    return z_summaries, z_obs, stats


def plot_loss(losses: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses["train"], label="train")
    ax.plot(losses["val"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fit_with_checkpoints(
    model: ConditionalGaussianNPE,
    thetas: jax.Array,
    summaries: jax.Array,
    *,
    key: jax.Array,
    config: TrainConfig,
    output_dir: Path,
    checkpoint_every_epochs: int,
    checkpoint_every_seconds: float,
    resume: bool,
) -> tuple[ConditionalGaussianNPE, dict[str, list[float]], dict[str, Any]]:
    n = thetas.shape[0]
    n_val = max(1, int(n * config.val_frac))
    key, subkey = random.split(key)
    perm = random.permutation(subkey, n)
    t_train, s_train = thetas[perm[n_val:]], summaries[perm[n_val:]]
    t_val, s_val = thetas[perm[:n_val]], summaries[perm[:n_val]]

    opt = optax.adam(config.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, t_batch, s_batch):
        loss, grads = eqx.filter_value_and_grad(_batch_loss)(model, t_batch, s_batch)
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def eval_loss(model, t, s):
        return _batch_loss(model, t, s)

    checkpoint_path = output_dir / "training_checkpoint.pkl"
    losses: dict[str, list[float]] = {"train": [], "val": [], "epoch_seconds": []}
    best_val_loss = float("inf")
    best_model = model
    best_epoch = -1
    wait = 0
    start_epoch = 0

    if resume and checkpoint_path.exists():
        with checkpoint_path.open("rb") as f:
            checkpoint = pkl.load(f)
        model = checkpoint["model"]
        opt_state = checkpoint["opt_state"]
        best_model = checkpoint["best_model"]
        losses = checkpoint["losses"]
        best_val_loss = float(checkpoint["best_val_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        wait = int(checkpoint["wait"])
        key = checkpoint["rng_key"]
        start_epoch = int(checkpoint["completed_epochs"])
        print(f">>> Resuming training from epoch {start_epoch}", flush=True)

    def save_checkpoint(reason: str, completed_epochs: int) -> None:
        state = {
            "completed_epochs": completed_epochs,
            "model": model,
            "opt_state": opt_state,
            "best_model": best_model,
            "losses": losses,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "wait": wait,
            "rng_key": key,
            "config": config._asdict(),
            "reason": reason,
            "updated_at_utc": utc_now(),
        }
        write_pickle_atomic(checkpoint_path, state)
        write_json(
            output_dir / "losses_partial.json",
            {
                "losses": losses,
                "completed_epochs": completed_epochs,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "wait": wait,
                "checkpoint_reason": reason,
                "updated_at_utc": state["updated_at_utc"],
            },
        )

    n_train = t_train.shape[0]
    n_batches = max(1, -(-n_train // config.batch_size))
    run_start = time.perf_counter()
    last_checkpoint = run_start
    stop_reason = "max_epochs"

    for epoch in range(start_epoch, config.max_epochs):
        epoch_start = time.perf_counter()
        key, subkey = random.split(key)
        idx_perm = random.permutation(subkey, n_train)
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * config.batch_size
            end = min(start + config.batch_size, n_train)
            idx = idx_perm[start:end]
            model, opt_state, loss = step(model, opt_state, t_train[idx], s_train[idx])
            epoch_loss += float(loss)

        train_loss = epoch_loss / n_batches
        val_loss = float(eval_loss(model, t_val, s_val))
        epoch_seconds = time.perf_counter() - epoch_start
        losses["train"].append(train_loss)
        losses["val"].append(val_loss)
        losses["epoch_seconds"].append(epoch_seconds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        completed_epochs = epoch + 1
        elapsed_since_checkpoint = time.perf_counter() - last_checkpoint
        due_epoch = checkpoint_every_epochs > 0 and completed_epochs % checkpoint_every_epochs == 0
        due_time = checkpoint_every_seconds > 0 and elapsed_since_checkpoint >= checkpoint_every_seconds
        if due_epoch or due_time:
            save_checkpoint("periodic", completed_epochs)
            last_checkpoint = time.perf_counter()

        if wait >= config.patience:
            stop_reason = "patience"
            break
    else:
        completed_epochs = config.max_epochs

    save_checkpoint("final", len(losses["train"]))
    info = {
        "stop_reason": stop_reason,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "wait": wait,
        "start_epoch": start_epoch,
        "completed_epochs": len(losses["train"]),
        "mean_epoch_seconds": float(np.mean(losses["epoch_seconds"]))
        if losses["epoch_seconds"]
        else None,
        "median_epoch_seconds": float(np.median(losses["epoch_seconds"]))
        if losses["epoch_seconds"]
        else None,
        "checkpoint_path": str(checkpoint_path),
    }
    return best_model, losses, info


def plot_parameter_overlays(
    posterior_theta: np.ndarray,
    reference_theta: np.ndarray,
    output_dir: Path,
) -> None:
    for i, name in enumerate(PARAM_NAMES):
        fig, ax = plt.subplots(figsize=(5, 4))
        _, bins, _ = ax.hist(
            posterior_theta[:, i],
            bins=50,
            density=True,
            alpha=0.55,
            label="robust Gaussian-NPE",
            color="#7b3294",
        )
        ax.hist(
            reference_theta[:, i],
            bins=bins,
            density=True,
            alpha=0.35,
            label="NUTS",
            color="black",
        )
        ax.axvline(float(TRUE_THETA[i]), color="black", linestyle=":")
        ax.set_title(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"posterior_samples_{name}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, name in enumerate(PARAM_NAMES):
        axes[i].hist(
            reference_theta[:, i],
            bins=50,
            density=True,
            histtype="step",
            color="black",
            linewidth=1.5,
            label="NUTS",
        )
        axes[i].hist(
            posterior_theta[:, i],
            bins=50,
            density=True,
            histtype="step",
            color="#7b3294",
            linewidth=1.5,
            label="robust Gaussian-NPE",
        )
        axes[i].axvline(float(TRUE_THETA[i]), color="black", linestyle=":")
        axes[i].set_title(name)
        axes[i].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_samples_overlay.pdf")
    plt.close(fig)


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_root / (
        f"gaussian_npe_n_obs_{args.n_obs}_n_sims_{args.n_sims}_seed_{args.seed}_"
        f"transform_{args.transform}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_obs", "--n-obs", dest="n_obs", type=int, required=True)
    parser.add_argument("--n_sims", "--n-sims", dest="n_sims", type=int, required=True)
    parser.add_argument("--transform", choices=("asinh", "identity"), default="asinh")
    parser.add_argument(
        "--inner-scale-mode", choices=("mad", "fixed"), default="mad"
    )
    parser.add_argument(
        "--fixed-inner-scale",
        type=float,
        default=None,
        help="Scalar c for all coordinates when --inner-scale-mode fixed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "res" / "gnk_robust_scale",
    )
    parser.add_argument(
        "--v3-root", type=Path, default=REPO_ROOT / "res" / "gnk_v3_refs"
    )
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--sim-batch-size", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=30)
    parser.add_argument("--checkpoint-every-seconds", type=float, default=1800.0)
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    run_start = time.perf_counter()
    args = build_parser().parse_args()
    convention = "gaussian"
    created_at = utc_now()
    env = environment_record()

    output_dir = output_dir_for(args)
    if output_dir.exists() and not args.force and not args.resume_training:
        raise FileExistsError(f"Refusing to overwrite existing output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "GNK robust-scale Gaussian-NPE: "
        f"n_obs={args.n_obs}, seed={args.seed}, n_sims={args.n_sims}, "
        f"transform={args.transform}, inner_scale_mode={args.inner_scale_mode}",
        flush=True,
    )

    ref_path, ref_fingerprint, ref_samples = load_canonical_reference(
        args.v3_root, args.n_obs, args.seed, convention
    )
    x_obs_f32 = reconstruct_x_obs_float32(args.seed, args.n_obs, convention)
    obs_sha = x_obs_sha256(x_obs_f32)
    ref_sha = ref_fingerprint.get("x_obs_summary_unstandardised_sha256")
    if obs_sha != ref_sha:
        raise RuntimeError(
            f"x_obs hash mismatch: reconstructed {obs_sha}, reference {ref_sha}"
        )
    x_obs = jnp.asarray(x_obs_f32, dtype=jnp.float64)
    print(f"  x_obs sha256: {obs_sha}", flush=True)
    print(f"  x_obs: {[float(v) for v in x_obs]}", flush=True)

    tol = 1e-6
    theta_key = rng_for(args.seed, "robust_scale", args.n_obs, args.n_sims, "theta")
    thetas_bounded = dist.Uniform(0.0 + tol, 10.0 - tol).sample(
        theta_key, (args.n_sims, 4)
    )
    thetas_unbounded = logit(thetas_bounded / 10.0)

    print(">>> Simulating prior predictive summaries...", flush=True)
    sim_key = rng_for(args.seed, "robust_scale", args.n_obs, args.n_sims, "summaries")
    A_sim, B_sim, g_sim, k_sim = thetas_bounded.T
    sim_start = time.perf_counter()
    x_sims = get_summaries_batches(
        sim_key,
        A_sim,
        B_sim,
        g_sim,
        k_sim,
        args.n_obs,
        args.n_sims,
        batch_size=min(args.sim_batch_size, args.n_sims),
    )
    simulation_seconds = time.perf_counter() - sim_start
    print(f">>> Simulations done in {simulation_seconds:.1f}s", flush=True)

    print(">>> Robust-standardising summaries...", flush=True)
    standardisation_start = time.perf_counter()
    thetas_mean = thetas_unbounded.mean(axis=0)
    thetas_std = thetas_unbounded.std(axis=0)
    thetas = (thetas_unbounded - thetas_mean) / thetas_std

    sim_summaries = np.asarray(x_sims.T, dtype=np.float64)
    z_summaries, z_obs, standardised_stats = robust_standardise(
        sim_summaries,
        np.asarray(x_obs, dtype=np.float64),
        args.transform,
        args.inner_scale_mode,
        args.fixed_inner_scale,
    )
    standardisation_seconds = time.perf_counter() - standardisation_start
    write_json(output_dir / "standardised_train_stats.json", standardised_stats)

    print(">>> Training conditional Gaussian-NPE...", flush=True)
    model_key = rng_for(args.seed, "robust_scale", args.n_obs, args.n_sims, "model")
    model = ConditionalGaussianNPE(
        d_summary=7,
        d_theta=4,
        hidden_dims=(128, 128),
        key=model_key,
    )
    train_cfg = TrainConfig(
        lr=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    fit_key = rng_for(args.seed, "robust_scale", args.n_obs, args.n_sims, "fit")
    train_start = time.perf_counter()
    model, losses, training_info = fit_with_checkpoints(
        model,
        thetas,
        jnp.asarray(z_summaries),
        key=fit_key,
        config=train_cfg,
        output_dir=output_dir,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        checkpoint_every_seconds=args.checkpoint_every_seconds,
        resume=args.resume_training,
    )
    training_seconds = time.perf_counter() - train_start
    print(
        f">>> Training done in {training_seconds:.1f}s "
        f"after {len(losses['train'])} epochs",
        flush=True,
    )
    plot_loss(losses, output_dir / "losses.pdf")
    write_json(output_dir / "losses.json", losses)

    print(">>> Sampling posterior...", flush=True)
    sampling_start = time.perf_counter()
    posterior_key = rng_for(
        args.seed, "robust_scale", args.n_obs, args.n_sims, "posterior"
    )
    posterior_std = sample(
        model,
        jnp.asarray(z_obs),
        args.num_posterior_samples,
        key=posterior_key,
    )
    posterior_unbounded = posterior_std * thetas_std + thetas_mean
    posterior_theta = np.asarray(expit(posterior_unbounded) * 10.0, dtype=np.float64)
    sampling_seconds = time.perf_counter() - sampling_start

    print(">>> Computing canonical-reference metrics...", flush=True)
    ref_unique, ref_n_dup = deduplicate(ref_samples)
    ps_unique, ps_n_dup = deduplicate(posterior_theta)
    ps_thin = deterministic_subsample(
        ps_unique,
        N_METRIC,
        stable_int("robust_scale", args.n_obs, args.seed, args.n_sims, "metric_npe"),
    )
    ref_thin = deterministic_subsample(
        ref_unique,
        N_METRIC,
        stable_int("robust_scale", args.n_obs, args.seed, args.n_sims, "metric_ref"),
    )
    metric_start = time.perf_counter()
    kl_value = float(kullback_leibler(jnp.asarray(ref_thin), jnp.asarray(ps_thin)))
    lengthscale = median_heuristic_fast(np.vstack([ref_thin, ps_thin]))
    mmd_value = float(
        unbiased_mmd(jnp.asarray(ref_thin), jnp.asarray(ps_thin), lengthscale)
    )
    metric_seconds = time.perf_counter() - metric_start
    print(f"  KL: {kl_value:.4f}, MMD: {mmd_value:.6f}", flush=True)

    with (output_dir / "posterior_samples.pkl").open("wb") as f:
        pkl.dump(posterior_theta, f)
    np.save(output_dir / "x_obs.npy", np.asarray(x_obs, dtype=np.float64))
    plotting_start = time.perf_counter()
    plot_parameter_overlays(posterior_theta, ref_samples, output_dir)
    plotting_seconds = time.perf_counter() - plotting_start

    fingerprint = {
        "created_at_utc": created_at,
        "environment": env,
        "transform": args.transform,
        "inner_scale_mode": args.inner_scale_mode,
        "fixed_inner_scale": args.fixed_inner_scale,
        "inner_scale_per_coord": standardised_stats["inner_scale_per_coord"],
        "robust_center_per_coord": standardised_stats["robust_center_per_coord"],
        "robust_scale_per_coord": standardised_stats["robust_scale_per_coord"],
        "jax_enable_x64_env": os.environ.get("JAX_ENABLE_X64"),
    }
    metrics = {
        "method": "gaussian_npe_robust_scale",
        "n_obs": args.n_obs,
        "n_sims": args.n_sims,
        "seed": args.seed,
        "convention": convention,
        "ref_path": str(ref_path),
        "ref_x_obs_sha256": ref_sha,
        "ref_density_version": ref_fingerprint.get("density_version"),
        "kl_value": kl_value,
        "mmd_value": mmd_value,
        "mmd_lengthscale": lengthscale,
        "n_metric": N_METRIC,
        "n_posterior_samples": int(posterior_theta.shape[0]),
        "n_posterior_duplicates_removed": int(ps_n_dup),
        "n_reference_samples": int(ref_samples.shape[0]),
        "n_reference_duplicates_removed": int(ref_n_dup),
        "simulation_seconds": simulation_seconds,
        "standardisation_seconds": standardisation_seconds,
        "training_seconds": training_seconds,
        "sampling_seconds": sampling_seconds,
        "metric_seconds": metric_seconds,
        "plotting_seconds": plotting_seconds,
        "runtime_seconds": time.perf_counter() - run_start,
        "train_epochs": len(losses["train"]),
        "train_config": train_cfg._asdict(),
        "training_info": training_info,
        "fingerprint": fingerprint,
        "args": {
            **vars(args),
            "output_root": str(args.output_root),
            "v3_root": str(args.v3_root),
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    print(f"  wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
