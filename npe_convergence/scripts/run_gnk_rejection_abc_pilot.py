#!/usr/bin/env python
"""Run the GNK rejection-ABC concentration pilot for Gaussian-NPE.

This is a single-cell diagnostic for
``n_obs=1000, seed=0, convention=gaussian`` by default. It simulates a broad
prior-predictive pool, keeps the closest summaries to the observed octiles under
a pool-median/IQR distance, then trains the standard z-score Gaussian-NPE on the
kept pairs only.
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

import equinox as eqx
import jax.numpy as jnp
import jax.random as random
import matplotlib
import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
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


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, jax.Array):
        return sanitize_for_json(np.asarray(value).tolist())
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(
            sanitize_for_json(payload),
            f,
            indent=2,
            sort_keys=True,
            default=json_default,
            allow_nan=False,
        )
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
    if convention != "gaussian":
        raise ValueError(f"unknown convention: {convention}")
    key = random.key(seed)
    _, z_key = random.split(key)
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


def safe_zscore(
    values: np.ndarray, obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    sd = np.std(values, axis=0)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0.0):
        raise ValueError(f"non-positive z-score scale: {sd}")
    return (values - mean) / sd, (obs - mean) / sd, mean, sd


def summarise_octile_concentration(
    pool_summaries: np.ndarray,
    kept_summaries: np.ndarray,
    distances: np.ndarray,
    kept_distances: np.ndarray,
    s_obs: np.ndarray,
    pool_center: np.ndarray,
    pool_iqr: np.ndarray,
    distance_threshold: float,
) -> dict[str, Any]:
    pool_q05, pool_q50, pool_q95 = np.percentile(
        pool_summaries, [5.0, 50.0, 95.0], axis=0
    )
    kept_q05, kept_q50, kept_q95 = np.percentile(
        kept_summaries, [5.0, 50.0, 95.0], axis=0
    )
    per_octile = {}
    for i, name in enumerate(SUMMARY_NAMES):
        per_octile[name] = {
            "s_obs": float(s_obs[i]),
            "pool_median": float(pool_q50[i]),
            "pool_q05": float(pool_q05[i]),
            "pool_q95": float(pool_q95[i]),
            "kept_median": float(kept_q50[i]),
            "kept_q05": float(kept_q05[i]),
            "kept_q95": float(kept_q95[i]),
            "kept_median_minus_s_obs": float(kept_q50[i] - s_obs[i]),
            "kept_median_abs_distance_from_s_obs": float(abs(kept_q50[i] - s_obs[i])),
        }
    return {
        "summary_names": SUMMARY_NAMES,
        "pool_center_median": pool_center,
        "pool_iqr": pool_iqr,
        "distance": {
            "metric": "euclidean_on_pool_median_iqr_standardised_octiles",
            "threshold": float(distance_threshold),
            "pool_min": float(np.min(distances)),
            "pool_median": float(np.median(distances)),
            "pool_q95": float(np.percentile(distances, 95.0)),
            "kept_min": float(np.min(kept_distances)),
            "kept_median": float(np.median(kept_distances)),
            "kept_q95": float(np.percentile(kept_distances, 95.0)),
            "kept_max": float(np.max(kept_distances)),
        },
        "per_octile": per_octile,
    }


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
        due_epoch = (
            checkpoint_every_epochs > 0
            and completed_epochs % checkpoint_every_epochs == 0
        )
        due_time = (
            checkpoint_every_seconds > 0
            and elapsed_since_checkpoint >= checkpoint_every_seconds
        )
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


def plot_posterior_overlay(
    posterior_theta: np.ndarray,
    reference_theta: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    axes_flat = axes.ravel()
    for i, name in enumerate(PARAM_NAMES):
        ax = axes_flat[i]
        _, bins, _ = ax.hist(
            reference_theta[:, i],
            bins=55,
            density=True,
            histtype="step",
            linewidth=1.6,
            color="black",
            label="v3 NUTS",
        )
        ax.hist(
            posterior_theta[:, i],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color="#2a9d8f",
            label="rejABC Gaussian-NPE",
        )
        ax.axvline(float(TRUE_THETA[i]), color="black", linestyle=":", linewidth=1.1)
        ax.set_title(name)
        ax.legend(fontsize=8)

    ax = axes_flat[4]
    rng = np.random.default_rng(0)
    n_ref = min(3500, reference_theta.shape[0])
    n_post = min(3500, posterior_theta.shape[0])
    ref_idx = rng.choice(reference_theta.shape[0], size=n_ref, replace=False)
    post_idx = rng.choice(posterior_theta.shape[0], size=n_post, replace=False)
    ax.scatter(
        reference_theta[ref_idx, 2],
        reference_theta[ref_idx, 3],
        s=5,
        alpha=0.18,
        color="black",
        label="v3 NUTS",
        rasterized=True,
    )
    ax.scatter(
        posterior_theta[post_idx, 2],
        posterior_theta[post_idx, 3],
        s=5,
        alpha=0.18,
        color="#2a9d8f",
        label="rejABC Gaussian-NPE",
        rasterized=True,
    )
    ax.scatter([float(TRUE_THETA[2])], [float(TRUE_THETA[3])], color="black", s=25)
    ax.set_xlabel("g")
    ax.set_ylabel("k")
    ax.set_title("(g, k)")
    ax.legend(fontsize=8)
    axes_flat[5].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_root / (
        f"gaussian_npe_n_obs_{args.n_obs}_n_pool_{args.n_pool}_seed_{args.seed}_"
        f"acc_{args.acceptance:g}"
    )


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def comparator_values(args: argparse.Namespace) -> dict[str, Any]:
    vanilla_metrics = load_json_if_exists(
        REPO_ROOT
        / "res"
        / "gnk"
        / "gaussian_npe_n_obs_1000_n_sims_1000000_seed_0"
        / "metrics_v3.json"
    )
    robust_metrics = load_json_if_exists(
        REPO_ROOT
        / "res"
        / "gnk_robust_scale"
        / "gaussian_npe_n_obs_1000_n_sims_1000000_seed_0_transform_asinh"
        / "metrics.json"
    )
    bsl_kl = load_json_if_exists(args.bsl_comparator / "bsl_vs_nuts_kl.json")
    bsl_diag = load_json_if_exists(args.bsl_comparator / "diagnostics.json")
    return {
        "vanilla_metrics_path": str(
            REPO_ROOT
            / "res"
            / "gnk"
            / "gaussian_npe_n_obs_1000_n_sims_1000000_seed_0"
            / "metrics_v3.json"
        ),
        "vanilla_canonical_kl": None
        if vanilla_metrics is None
        else vanilla_metrics.get("kl_value"),
        "robust_metrics_path": str(
            REPO_ROOT
            / "res"
            / "gnk_robust_scale"
            / "gaussian_npe_n_obs_1000_n_sims_1000000_seed_0_transform_asinh"
            / "metrics.json"
        ),
        "robust_canonical_kl": None
        if robust_metrics is None
        else robust_metrics.get("kl_value"),
        "bsl_comparator_path": str(args.bsl_comparator),
        "bsl_kl": bsl_kl,
        "bsl_diagnostics": bsl_diag,
    }


def write_outcome_doc(
    path: Path,
    *,
    metrics: dict[str, Any],
    config: dict[str, Any],
    concentration: dict[str, Any],
    posterior_theta: np.ndarray,
    ref_samples: np.ndarray,
    comparators: dict[str, Any],
) -> None:
    ref_median = np.median(ref_samples, axis=0)
    ref_sd = np.std(ref_samples, axis=0)
    post_median = np.median(posterior_theta, axis=0)
    shifts = (post_median - ref_median) / ref_sd

    vanilla_kl = comparators.get("vanilla_canonical_kl")
    robust_kl = comparators.get("robust_canonical_kl")
    bsl_kl_json = comparators.get("bsl_kl") or {}
    bsl_diag = comparators.get("bsl_diagnostics") or {}
    bsl_nuts_to_method = bsl_kl_json.get("KL_NUTS_to_BSL")
    bsl_method_to_nuts = bsl_kl_json.get("KL_BSL_to_NUTS")
    bsl_unique_note = "raw retained-state BSL->NUTS is not finite when duplicates are kept"
    bsl_shift = None
    gate = bsl_diag.get("acceptance_gate", {})
    if gate.get("bsl_median") and gate.get("nuts_median") and gate.get("nuts_std"):
        bsl_shift = max(
            abs(
                (
                    float(gate["bsl_median"][name])
                    - float(gate["nuts_median"][name])
                )
                / float(gate["nuts_std"][name])
            )
            for name in PARAM_NAMES
        )

    pilot_kl = metrics["kl_reference_to_npe_value"]
    pilot_reverse = metrics["kl_npe_to_reference_value"]
    vanilla_change = (
        None if vanilla_kl is None else (float(vanilla_kl) - pilot_kl) / float(vanilla_kl)
    )
    robust_change = (
        None if robust_kl is None else (float(robust_kl) - pilot_kl) / float(robust_kl)
    )
    pool_budget_ratio = config["n_pool"] / 1_000_000
    training_set_ratio = config["n_keep"] / 1_000_000

    lines = [
        "# T22 rejection-ABC pilot: GNK n_obs=1000 seed=0",
        "",
        "## 1. Setup recap",
        "",
        "| field | value |",
        "|---|---:|",
        f"| model | GNK g-and-k |",
        f"| n_obs | {config['n_obs']} |",
        f"| seed | {config['seed']} |",
        f"| convention | {config['convention']} |",
        f"| pool size | {config['n_pool']:,} |",
        f"| acceptance | {config['acceptance']:.4f} |",
        f"| kept pairs | {config['n_keep']:,} |",
        f"| reference | `{config['reference_path']}` |",
        f"| observation source | v3 reference hash-checked observed octiles |",
        "",
        "The training stage used the standard Gaussian-NPE z-score input "
        "standardisation on the kept set. It did not use the robust asinh "
        "median/IQR transform.",
        "",
        "The canonical Gaussian-NPE comparator at this cell used "
        f"`n_sims=1,000,000`. This pilot uses {pool_budget_ratio:.1f}x the "
        "simulator budget and trains on "
        f"{config['n_keep']:,} accepted pairs, i.e. "
        f"{100.0 * training_set_ratio:.1f}% as large a training set.",
        "",
        "## 2. KL table",
        "",
        "The comparable paper-style KL direction is `KL(v3 NUTS || method)`, "
        "matching the existing canonical evaluator. The pilot also records "
        "`KL(method || v3 NUTS)` because the T22 spec requested that direction.",
        "",
        "| method | KL(v3 NUTS || method) | KL(method || v3 NUTS) | max median shift (NUTS sd) | notes |",
        "|---|---:|---:|---:|---|",
        f"| Vanilla Gaussian-NPE, z-score | {float(vanilla_kl):.3f} | n/a | 4.890 | `N=1,000,000` existing result |"
        if vanilla_kl is not None
        else "| Vanilla Gaussian-NPE, z-score | n/a | n/a | n/a | metrics missing |",
        f"| Robust-scaling Gaussian-NPE | {float(robust_kl):.3f} | n/a | 0.461 | existing result |"
        if robust_kl is not None
        else "| Robust-scaling Gaussian-NPE | n/a | n/a | n/a | metrics missing |",
        f"| Rejection-ABC Gaussian-NPE | {pilot_kl:.3f} | {pilot_reverse:.3f} | {float(np.max(np.abs(shifts))):.3f} | this run |",
        f"| BSL comparator | {float(bsl_nuts_to_method):.3f} | {bsl_method_to_nuts} | {bsl_shift:.3f} | specified path; {bsl_unique_note} |"
        if bsl_nuts_to_method is not None and bsl_shift is not None
        else "| BSL comparator | n/a | n/a | n/a | comparator metrics missing |",
        "",
        "## 3. Marginal-shift table",
        "",
        "| parameter | NUTS median | rejABC median | NUTS sd | shift |",
        "|---|---:|---:|---:|---:|",
    ]
    for i, name in enumerate(PARAM_NAMES):
        lines.append(
            f"| {name} | {ref_median[i]:.6f} | {post_median[i]:.6f} | "
            f"{ref_sd[i]:.6f} | {shifts[i]:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Pool vs kept-set octile concentration",
            "",
            "| octile | s_obs | pool median | pool 5% | pool 95% | kept median | kept 5% | kept 95% | kept median - s_obs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in SUMMARY_NAMES:
        row = concentration["per_octile"][name]
        lines.append(
            f"| {name} | {row['s_obs']:.6f} | {row['pool_median']:.6f} | "
            f"{row['pool_q05']:.6f} | {row['pool_q95']:.6f} | "
            f"{row['kept_median']:.6f} | {row['kept_q05']:.6f} | "
            f"{row['kept_q95']:.6f} | {row['kept_median_minus_s_obs']:.6f} |"
        )

    dist = concentration["distance"]
    lines.extend(
        [
            "",
            f"Kept distance threshold: `{dist['threshold']:.6f}`. "
            f"Kept distance median/q95/max: `{dist['kept_median']:.6f}`, "
            f"`{dist['kept_q95']:.6f}`, `{dist['kept_max']:.6f}`.",
            "",
            "## 5. Interpretation",
            "",
        ]
    )
    interp = []
    if vanilla_change is not None:
        if vanilla_change >= 0:
            interp.append(
                f"Against the canonical z-score baseline, rejection filtering "
                f"changed the comparable KL from {float(vanilla_kl):.3f} to "
                f"{pilot_kl:.3f}, a {100.0 * vanilla_change:.1f}% reduction."
            )
        else:
            interp.append(
                f"Against the canonical z-score baseline, rejection filtering "
                f"changed the comparable KL from {float(vanilla_kl):.3f} to "
                f"{pilot_kl:.3f}, a {100.0 * abs(vanilla_change):.1f}% increase."
            )
    if robust_change is not None:
        if robust_change >= 0:
            interp.append(
                f"Relative to robust scaling, the rejection pilot is lower by "
                f"{100.0 * robust_change:.1f}% on the same KL direction."
            )
        else:
            interp.append(
                f"Relative to robust scaling, the rejection pilot is higher by "
                f"{100.0 * abs(robust_change):.1f}% on the same KL direction."
            )
    if (
        vanilla_kl is not None
        and robust_kl is not None
        and bsl_nuts_to_method is not None
        and float(vanilla_kl) > float(bsl_nuts_to_method)
    ):
        vanilla_bsl_gap = float(vanilla_kl) - float(bsl_nuts_to_method)
        robust_gap_closed = (float(vanilla_kl) - float(robust_kl)) / vanilla_bsl_gap
        pilot_gap_closed = (float(vanilla_kl) - pilot_kl) / vanilla_bsl_gap
        interp.append(
            f"Using the specified BSL comparator's finite "
            f"`KL(v3 NUTS || BSL)={float(bsl_nuts_to_method):.3f}` as the "
            f"local reference floor, robust scaling closes "
            f"{100.0 * robust_gap_closed:.1f}% of the vanilla-to-BSL KL gap, "
            f"and rejection filtering closes {100.0 * pilot_gap_closed:.1f}%."
        )
    interp.append(
        "The reverse-direction KL requested in the spec is reported separately "
        f"as {pilot_reverse:.3f}. This is a diagnostic only; no paper claim is "
        "implied by this single seed."
    )
    lines.extend(interp)

    lines.extend(
        [
            "",
            "## 6. Wallclock and resource use",
            "",
            "| stage | seconds |",
            "|---|---:|",
            f"| pool simulation | {metrics['simulation_seconds']:.1f} |",
            f"| rejection selection and standardisation | {metrics['selection_seconds'] + metrics['standardisation_seconds']:.1f} |",
            f"| training | {metrics['training_seconds']:.1f} |",
            f"| sampling | {metrics['sampling_seconds']:.1f} |",
            f"| metrics | {metrics['metric_seconds']:.1f} |",
            f"| plotting | {metrics['plotting_seconds']:.1f} |",
            f"| total | {metrics['runtime_seconds']:.1f} |",
            "",
            f"Training stopped after `{metrics['train_epochs']}` epochs "
            f"with reason `{metrics['training_info']['stop_reason']}`.",
            "",
            "Artifacts:",
            "",
            f"- Results directory: `{config['output_dir']}`",
            f"- Posterior overlay: `{config['output_dir']}/posterior_overlay.pdf`",
            f"- Concentration JSON: `{config['output_dir']}/kept_summary_concentration.json`",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", "--n-obs", dest="n_obs", type=int, default=1000)
    parser.add_argument(
        "--n-pool",
        "--n_pool",
        dest="n_pool",
        type=int,
        default=10_000_000,
    )
    parser.add_argument("--acceptance", type=float, default=0.01)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "res" / "gnk_rejection_abc",
    )
    parser.add_argument(
        "--v3-root", type=Path, default=REPO_ROOT / "res" / "gnk_v3_refs"
    )
    parser.add_argument(
        "--bsl-comparator",
        type=Path,
        default=REPO_ROOT
        / "res"
        / "gnk_bsl"
        / "bsl_n_obs_1000_seed_0_M_500_20260525T230504Z",
    )
    parser.add_argument(
        "--outcome-doc",
        type=Path,
        default=REPO_ROOT
        / "docs"
        / "meeting_2026_05_18"
        / "codex_outcome_T22_rejection_abc_pilot_gnk_n1000.md",
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

    if not (0.0 < args.acceptance < 1.0):
        raise ValueError("--acceptance must be in (0, 1)")
    n_keep = int(round(args.n_pool * args.acceptance))
    if n_keep <= 0:
        raise ValueError("accepted set is empty")

    output_dir = output_dir_for(args)
    if output_dir.exists() and not args.force and not args.resume_training:
        raise FileExistsError(f"Refusing to overwrite existing output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "GNK rejection-ABC Gaussian-NPE pilot: "
        f"n_obs={args.n_obs}, seed={args.seed}, n_pool={args.n_pool}, "
        f"acceptance={args.acceptance}, n_keep={n_keep}",
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
    x_obs = np.asarray(x_obs_f32, dtype=np.float64)
    print(f"  x_obs sha256: {obs_sha}", flush=True)
    print(f"  x_obs: {[float(v) for v in x_obs]}", flush=True)

    tol = 1e-6
    pool_key = random.key(args.seed)
    theta_key, sim_key = random.split(pool_key)
    thetas_bounded = dist.Uniform(0.0 + tol, 10.0 - tol).sample(
        theta_key, (args.n_pool, 4)
    )
    A_sim, B_sim, g_sim, k_sim = thetas_bounded.T

    print(">>> Simulating prior-predictive pool...", flush=True)
    sim_start = time.perf_counter()
    x_sims = get_summaries_batches(
        sim_key,
        A_sim,
        B_sim,
        g_sim,
        k_sim,
        args.n_obs,
        args.n_pool,
        batch_size=min(args.sim_batch_size, args.n_pool),
    )
    simulation_seconds = time.perf_counter() - sim_start
    print(f">>> Pool simulations done in {simulation_seconds:.1f}s", flush=True)

    print(">>> Computing rejection distances and selecting kept pairs...", flush=True)
    selection_start = time.perf_counter()
    pool_summaries = np.asarray(x_sims.T, dtype=np.float64)
    if not np.all(np.isfinite(pool_summaries)):
        raise RuntimeError("Non-finite values found in simulated summaries; stopping.")
    pool_center = np.median(pool_summaries, axis=0)
    q25, q75 = np.percentile(pool_summaries, [25.0, 75.0], axis=0)
    pool_iqr = q75 - q25
    if np.any(~np.isfinite(pool_iqr)) or np.any(pool_iqr <= 0.0):
        raise RuntimeError(f"Non-positive pool IQR: {pool_iqr}")
    z_pool = (pool_summaries - pool_center) / pool_iqr
    z_obs_for_distance = (x_obs - pool_center) / pool_iqr
    distances = np.linalg.norm(z_pool - z_obs_for_distance[None, :], axis=1)
    if not np.all(np.isfinite(distances)):
        raise RuntimeError("Non-finite values found in rejection distances; stopping.")
    keep_idx = np.argpartition(distances, n_keep - 1)[:n_keep]
    keep_order = np.argsort(distances[keep_idx])
    keep_idx = keep_idx[keep_order]
    kept_distances = distances[keep_idx]
    distance_threshold = float(kept_distances[-1])
    kept_thetas_bounded = np.asarray(thetas_bounded[keep_idx], dtype=np.float64)
    kept_summaries = pool_summaries[keep_idx]
    concentration = summarise_octile_concentration(
        pool_summaries,
        kept_summaries,
        distances,
        kept_distances,
        x_obs,
        pool_center,
        pool_iqr,
        distance_threshold,
    )
    write_json(output_dir / "kept_summary_concentration.json", concentration)
    selection_seconds = time.perf_counter() - selection_start
    print(
        f">>> Kept {kept_summaries.shape[0]} pairs with threshold "
        f"{distance_threshold:.6f} in {selection_seconds:.1f}s",
        flush=True,
    )

    del x_sims, pool_summaries, z_pool, distances
    del thetas_bounded, A_sim, B_sim, g_sim, k_sim

    print(">>> Applying standard z-score training standardisation...", flush=True)
    standardisation_start = time.perf_counter()
    kept_thetas_unbounded = logit(jnp.asarray(kept_thetas_bounded) / 10.0)
    theta_z, _, theta_mean, theta_sd = safe_zscore(
        np.asarray(kept_thetas_unbounded, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
    )
    summary_z, z_obs, summary_mean, summary_sd = safe_zscore(kept_summaries, x_obs)
    standardisation = {
        "theta_unbounded_mean": theta_mean,
        "theta_unbounded_sd": theta_sd,
        "summary_mean": summary_mean,
        "summary_sd": summary_sd,
        "x_obs_raw": x_obs,
        "x_obs_zscore": z_obs,
        "input_transform": "standard_zscore_on_kept_set",
    }
    write_json(output_dir / "standardisation.json", standardisation)
    standardisation_seconds = time.perf_counter() - standardisation_start

    print(">>> Training conditional Gaussian-NPE on kept set...", flush=True)
    model_key = rng_for(args.seed, "rejection_abc", args.n_obs, args.n_pool, "model")
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
    fit_key = rng_for(args.seed, "rejection_abc", args.n_obs, args.n_pool, "fit")
    train_start = time.perf_counter()
    model, losses, training_info = fit_with_checkpoints(
        model,
        jnp.asarray(theta_z),
        jnp.asarray(summary_z),
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
        args.seed, "rejection_abc", args.n_obs, args.n_pool, "posterior"
    )
    posterior_z = sample(
        model,
        jnp.asarray(z_obs),
        args.num_posterior_samples,
        key=posterior_key,
    )
    posterior_unbounded = posterior_z * theta_sd + theta_mean
    posterior_theta = np.asarray(expit(posterior_unbounded) * 10.0, dtype=np.float64)
    if not np.all(np.isfinite(posterior_theta)):
        raise RuntimeError("Non-finite posterior samples; stopping.")
    sampling_seconds = time.perf_counter() - sampling_start

    print(">>> Computing canonical-reference metrics...", flush=True)
    ref_unique, ref_n_dup = deduplicate(ref_samples)
    ps_unique, ps_n_dup = deduplicate(posterior_theta)
    ps_thin = deterministic_subsample(
        ps_unique,
        N_METRIC,
        stable_int("rejection_abc", args.n_obs, args.seed, args.n_pool, "metric_npe"),
    )
    ref_thin = deterministic_subsample(
        ref_unique,
        N_METRIC,
        stable_int("rejection_abc", args.n_obs, args.seed, args.n_pool, "metric_ref"),
    )
    metric_start = time.perf_counter()
    kl_reference_to_npe = float(
        kullback_leibler(jnp.asarray(ref_thin), jnp.asarray(ps_thin))
    )
    kl_npe_to_reference = float(
        kullback_leibler(jnp.asarray(ps_thin), jnp.asarray(ref_thin))
    )
    lengthscale = median_heuristic_fast(np.vstack([ref_thin, ps_thin]))
    mmd_value = float(
        unbiased_mmd(jnp.asarray(ref_thin), jnp.asarray(ps_thin), lengthscale)
    )
    metric_seconds = time.perf_counter() - metric_start
    print(
        f"  KL(ref||NPE): {kl_reference_to_npe:.4f}, "
        f"KL(NPE||ref): {kl_npe_to_reference:.4f}, MMD: {mmd_value:.6f}",
        flush=True,
    )

    with (output_dir / "posterior_samples.pkl").open("wb") as f:
        pkl.dump(posterior_theta, f)
    np.save(output_dir / "x_obs.npy", x_obs)

    plotting_start = time.perf_counter()
    plot_posterior_overlay(posterior_theta, ref_samples, output_dir / "posterior_overlay.pdf")
    plotting_seconds = time.perf_counter() - plotting_start

    ref_median = np.median(ref_samples, axis=0)
    ref_sd = np.std(ref_samples, axis=0)
    post_median = np.median(posterior_theta, axis=0)
    marginal_shifts = {
        name: {
            "nuts_median": float(ref_median[i]),
            "rejabc_median": float(post_median[i]),
            "nuts_sd": float(ref_sd[i]),
            "shift": float((post_median[i] - ref_median[i]) / ref_sd[i]),
        }
        for i, name in enumerate(PARAM_NAMES)
    }

    config_payload = {
        "method": "gaussian_npe_rejection_abc_pilot",
        "n_obs": args.n_obs,
        "n_pool": args.n_pool,
        "acceptance": args.acceptance,
        "n_keep": n_keep,
        "seed": args.seed,
        "convention": convention,
        "reference_path": str(ref_path),
        "bsl_comparator_path": str(args.bsl_comparator),
        "observation": {
            "source": "hash-checked reconstruction matching v3 reference",
            "x_obs_sha256": obs_sha,
            "x_obs": x_obs,
        },
        "output_dir": str(output_dir),
        "theta_prior": "Uniform(1e-6, 10 - 1e-6)^4; tolerance avoids logit infinities",
        "distance_standardisation": "pool marginal median/IQR, no 1.349 factor",
        "training_standardisation": "kept-set z-score for summaries and unbounded theta",
        "train_config": train_cfg._asdict(),
        "sim_batch_size": args.sim_batch_size,
        "num_posterior_samples": args.num_posterior_samples,
    }
    write_json(output_dir / "config.json", config_payload)

    comparators = comparator_values(args)
    fingerprint = {
        "created_at_utc": created_at,
        "environment": env,
        "jax_enable_x64_env": os.environ.get("JAX_ENABLE_X64"),
    }
    metrics = {
        "method": "gaussian_npe_rejection_abc_pilot",
        "n_obs": args.n_obs,
        "n_pool": args.n_pool,
        "acceptance": args.acceptance,
        "n_keep": n_keep,
        "seed": args.seed,
        "convention": convention,
        "ref_path": str(ref_path),
        "ref_x_obs_sha256": ref_sha,
        "ref_density_version": ref_fingerprint.get("density_version"),
        "kl_value": kl_reference_to_npe,
        "kl_reference_to_npe_value": kl_reference_to_npe,
        "kl_npe_to_reference_value": kl_npe_to_reference,
        "mmd_value": mmd_value,
        "mmd_lengthscale": lengthscale,
        "n_metric": N_METRIC,
        "n_posterior_samples": int(posterior_theta.shape[0]),
        "n_posterior_duplicates_removed": int(ps_n_dup),
        "n_reference_samples": int(ref_samples.shape[0]),
        "n_reference_duplicates_removed": int(ref_n_dup),
        "simulation_seconds": simulation_seconds,
        "selection_seconds": selection_seconds,
        "standardisation_seconds": standardisation_seconds,
        "training_seconds": training_seconds,
        "sampling_seconds": sampling_seconds,
        "metric_seconds": metric_seconds,
        "plotting_seconds": plotting_seconds,
        "runtime_seconds": time.perf_counter() - run_start,
        "train_epochs": len(losses["train"]),
        "train_config": train_cfg._asdict(),
        "training_info": training_info,
        "marginal_shifts": marginal_shifts,
        "comparators": comparators,
        "fingerprint": fingerprint,
        "args": {
            **vars(args),
            "output_root": str(args.output_root),
            "v3_root": str(args.v3_root),
            "bsl_comparator": str(args.bsl_comparator),
            "outcome_doc": str(args.outcome_doc),
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_outcome_doc(
        args.outcome_doc,
        metrics=metrics,
        config=config_payload,
        concentration=concentration,
        posterior_theta=posterior_theta,
        ref_samples=ref_samples,
        comparators=comparators,
    )
    print(f"  wrote {output_dir}", flush=True)
    print(f"  wrote {args.outcome_doc}", flush=True)


if __name__ == "__main__":
    main()
