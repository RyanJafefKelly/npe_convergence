"""Prepare, run, or evaluate the GNK asymptotic-MVN simulator-control pilot.

This is a guarded single-run driver for the GNK simulator-control diagnostic.
It keeps outputs separate from empirical GNK caches and does not change the
default empirical simulator path.
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
import resource
import socket
import subprocess
import sys
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)
assert jax.config.read("jax_enable_x64")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "res" / "gnk_model_control"
NUTS_CACHE_TEMPLATE = "res/gnk/nuts_cache_v2_n_obs_{n}_seed_{seed}.pkl"
PBS_SCRIPT = "npe_convergence/scripts/pbs_jobs/gnk_model_control_n500_x50_seed88.sh"

CANDIDATE_X = (25, 50)
SELECTED_X_DEFAULT = 50
N_OBS = 500
D_S = 7
D_THETA = 4
D_TOTAL = D_S + D_THETA
OBSERVED_SEED = 88
METHOD = "Gaussian-NPE"
SIMULATOR = "gnk_asymptotic_mvn"
EMPIRICAL_SIMULATOR_NAME = "gnk_empirical_quantile"
RESOURCE_REQUEST = {
    "scheduler": "PBS",
    "job_type": "single_non_array",
    "walltime": "47:00:00",
    "mem": "64GB",
    "ncpus": 4,
    "ngpus": 0,
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_git(args: list[str], default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)
    except Exception:
        return default
    return out.strip() or default


def git_dirty() -> bool:
    return bool(run_git(["status", "--porcelain"], default="dirty"))


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def rng_for(seed: int, *parts: object):
    import jax.random as random

    key = random.key(seed)
    for part in parts:
        key = random.fold_in(key, stable_int(part))
    return key


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_utc_stamp(created_at: str) -> str:
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%dT%H%M%SZ")


def candidate_paths(run_id: str) -> dict[str, str]:
    out = OUTPUT_ROOT / run_id
    return {
        "output_dir": rel(out),
        "config": rel(out / "config.yaml"),
        "validation_curve": rel(out / "validation_curve.csv"),
        "validation_curve_plot": rel(out / "validation_curve.pdf"),
        "predicted_u_mean_cov": rel(out / "gaussian_npe_u_posterior.npz"),
        "samples_10k": rel(out / "posterior_samples_10k.npz"),
        "posterior_samples_pkl": rel(out / "posterior_samples.pkl"),
        "metrics": rel(out / "metrics.json"),
        "timing_metadata": rel(out / "timing_metadata.json"),
        "simulator_diagnostics": rel(out / "simulator_diagnostics.json"),
        "evaluation": rel(out / "u_space_decomposition.json"),
        "stdout_log": rel(out / "logs/stdout.log"),
        "stderr_log": rel(out / "logs/stderr.log"),
    }


def build_rows(*, selected_x: int, created_at: str, run_id: str | None) -> list[dict[str, Any]]:
    rows = []
    stamp = compact_utc_stamp(created_at)
    for x in CANDIDATE_X:
        n_sims = x * D_TOTAL * D_TOTAL * N_OBS
        rid = run_id if x == selected_x and run_id else (
            f"gnk_asymptotic_mvn_gaussian_npe_n500_x{x}_seed88_{stamp}"
        )
        paths = candidate_paths(rid)
        rows.append(
            {
                "n": N_OBS,
                "x": x,
                "N": n_sims,
                "seed": OBSERVED_SEED,
                "method": METHOD,
                "simulator": SIMULATOR,
                "selected": x == selected_x,
                "run_id": rid,
                "paths": paths,
                "exists": {
                    name: (REPO_ROOT / path).exists()
                    for name, path in paths.items()
                },
            }
        )
    return rows


def build_config(
    *,
    selected_x: int,
    created_at: str,
    run_id: str | None,
    model_control_code_commit: str | None,
) -> dict[str, Any]:
    rows = build_rows(selected_x=selected_x, created_at=created_at, run_id=run_id)
    selected = next(row for row in rows if row["selected"])
    output_dir = selected["paths"]["output_dir"]
    config_path = selected["paths"]["config"]
    run_command = (
        f"python npe_convergence/scripts/run_gnk_model_control_pilot.py "
        f"--config {config_path}"
    )
    manual_hpc_command = (
        "mkdir -p "
        f"{output_dir}/logs && "
        f"{run_command} > {selected['paths']['stdout_log']} "
        f"2> {selected['paths']['stderr_log']}"
    )
    qsub_command = f"qsub {PBS_SCRIPT}"
    selected_collisions = collision_paths_for_paths(selected["paths"], allow_config=True)
    return {
        "task": "gnk-model-control-pilot",
        "run_id": selected["run_id"],
        "created_at_utc": created_at,
        "git_commit": run_git(["rev-parse", "--short", "HEAD"]),
        "model_control_code_commit": model_control_code_commit or run_git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": git_dirty(),
        "branch": run_git(["branch", "--show-current"]),
        "method": METHOD,
        "simulator": SIMULATOR,
        "empirical_simulator_name": EMPIRICAL_SIMULATOR_NAME,
        "n": N_OBS,
        "d_s": D_S,
        "d_theta": D_THETA,
        "d": D_TOTAL,
        "x": selected_x,
        "N": selected["N"],
        "observed_seed": OBSERVED_SEED,
        "simulation_seed": OBSERVED_SEED,
        "training_seed": OBSERVED_SEED,
        "posterior_sampling_seed": OBSERVED_SEED,
        "prior": {
            "implementation": "numpyro.distributions.Uniform(1e-6, 10-1e-6)",
            "scientific_prior": "independent Uniform(0, 10) for A, B, g, k",
            "theta_names": ["A", "B", "g", "k"],
        },
        "observed_summary": {
            "source": (
                "regenerated from true theta=(3,1,2,0.5), observed seed 88, "
                "and npe_convergence.examples.gnk.ss_octile, matching the NUTS cache convention"
            ),
            "dimension": D_S,
            "coordinate_convention": "octiles at probabilities 1/(d_s+1), ..., d_s/(d_s+1)",
        },
        "nuts_reference_cache_path": NUTS_CACHE_TEMPLATE.format(n=N_OBS, seed=OBSERVED_SEED),
        "training_pair_generator": {
            "theta_sampler": "prior",
            "summary_simulator": SIMULATOR,
            "mvn_mean_function": "npe_convergence.examples.gnk.gnk at normal quantiles",
            "mvn_covariance_function": "npe_convergence.examples.gnk.compute_covariance_matrix",
            "base_jitter": 1e-6,
            "covariance_handling": (
                "Use gnk_model's 1e-6 diagonal jitter. Resample rare prior "
                "draws whose asymptotic covariance is still non-SPD or non-finite; "
                "record invalid_covariance_* counts in simulator_diagnostics.json."
            ),
        },
        "output_dir": output_dir,
        "scheduler_resource_request": RESOURCE_REQUEST,
        "train_config": {
            "lr": 5e-4,
            "batch_size": 256,
            "max_epochs": 2000,
            "patience": 200,
            "val_frac": 0.1,
            "hidden_dims": [128, 128],
        },
        "expected_outputs": dict(selected["paths"]),
        "dry_run_rows": rows,
        "post_run_evaluation_command": (
            f"python npe_convergence/scripts/run_gnk_model_control_pilot.py "
            f"--evaluate --config {config_path}"
        ),
        "qsub_command": qsub_command,
        "pbs_script": PBS_SCRIPT,
        "manual_hpc_command": manual_hpc_command,
        "manual_hpc_command_note": (
            "Direct python command must be run inside an allocated compute job; "
            "do not run this 3M-simulation training job on an HPC login node."
        ),
        "run_command": run_command,
        "full_array_submitted": False,
        "selected_output_collisions": selected_collisions,
        "selected_output_collisions_at_prepare_time": bool(selected_collisions),
        "selection_rationale": (
            "Selected x=50 because it matches the completed empirical-GNK "
            "calibration at n=500, N=3,025,000, seed=88. x=25 is retained only "
            "as a fallback candidate."
        ),
    }


def refresh_exists(config: dict[str, Any]) -> None:
    for row in config["dry_run_rows"]:
        row["exists"] = {
            name: (REPO_ROOT / path).exists()
            for name, path in row["paths"].items()
        }


def collision_paths(config: dict[str, Any], *, allow_config: bool = False) -> list[str]:
    return collision_paths_for_paths(config["expected_outputs"], allow_config=allow_config)


def collision_paths_for_paths(paths: dict[str, str], *, allow_config: bool = False) -> list[str]:
    skip = {"output_dir", "stdout_log", "stderr_log"}
    if allow_config:
        skip.add("config")
    collisions = []
    for key, value in paths.items():
        if key in skip:
            continue
        if (REPO_ROOT / value).exists():
            collisions.append(value)
    return collisions


def print_dry_run(config: dict[str, Any]) -> None:
    refresh_exists(config)
    print("GNK asymptotic-MVN simulator-control dry-run")
    print(f"created_at_utc: {config['created_at_utc']}")
    print(f"git_commit: {config['git_commit']} dirty={config['git_dirty']}")
    print(f"branch: {config['branch']}")
    print(f"observed_seed: {config['observed_seed']}")
    print(f"simulator: {config['simulator']}")
    print(f"method: {config['method']}")
    print(f"qsub_command: {config.get('qsub_command', 'not_recorded')}")
    print("")
    header = (
        "selected  n    x    N        seed  method        simulator"
        "                 output_dir"
    )
    print(header)
    print("-" * len(header))
    for row in config["dry_run_rows"]:
        mark = "yes" if row["selected"] else "no "
        print(
            f"{mark:8s} {int(row['n']):<4d} {str(row['x']):<8s} {int(row['N']):<8d} "
            f"{row['seed']:<5d} {row['method']:<13s} "
            f"{row['simulator']:<25s} {row['paths']['output_dir']}"
        )
        for label in (
            "config",
            "validation_curve",
            "predicted_u_mean_cov",
            "samples_10k",
            "timing_metadata",
            "simulator_diagnostics",
            "evaluation",
            "stdout_log",
            "stderr_log",
        ):
            print(
                f"    {label}: {row['paths'][label]} "
                f"exists={str(row['exists'][label]).lower()}"
            )
    selected_collisions = collision_paths(config, allow_config=True)
    config["selected_output_collisions"] = selected_collisions
    config["selected_output_collisions_at_prepare_time"] = bool(selected_collisions)
    if selected_collisions:
        print("")
        print("WARNING: SELECTED-RUN OUTPUT PATH COLLISIONS DETECTED")
        for path in selected_collisions:
            print(f"  {path}")
    print("")
    print("DRY_RUN_JSON_BEGIN")
    print(json.dumps(config, indent=2, sort_keys=True))
    print("DRY_RUN_JSON_END")


def write_config(config: dict[str, Any], *, allow_existing_config: bool = False) -> None:
    config_path = REPO_ROOT / config["expected_outputs"]["config"]
    output_dir = REPO_ROOT / config["output_dir"]
    collisions = collision_paths(config, allow_config=True)
    if collisions:
        raise SystemExit(
            "Selected run has existing output paths; refusing to write config:\n"
            + "\n".join(collisions)
        )
    if config_path.exists() and not allow_existing_config:
        raise SystemExit(f"Config already exists, refusing to overwrite: {rel(config_path)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote resolved config: {rel(config_path)}")


def load_config(path: str) -> dict[str, Any]:
    config_path = REPO_ROOT / path
    with config_path.open() as f:
        config = yaml.safe_load(f)
    if config["simulator"] != SIMULATOR:
        raise ValueError(f"unexpected simulator: {config['simulator']}")
    x_value = config.get("x")
    if isinstance(x_value, int) or (isinstance(x_value, str) and x_value.isdigit()):
        if int(x_value) not in CANDIDATE_X:
            raise ValueError(f"unexpected x: {config['x']}")
    elif x_value != "n_squared":
        raise ValueError(f"unexpected x: {config['x']}")
    config.setdefault("git_dirty", git_dirty())
    config.setdefault("branch", run_git(["branch", "--show-current"]))
    config.setdefault("dry_run_rows", [resolved_config_row(config)])
    return config


def resolved_config_row(config: dict[str, Any]) -> dict[str, Any]:
    paths = dict(config["expected_outputs"])
    return {
        "n": int(config["n"]),
        "x": config["x"],
        "N": int(config["N"]),
        "seed": int(config["observed_seed"]),
        "method": config["method"],
        "simulator": config["simulator"],
        "selected": True,
        "run_id": config["run_id"],
        "paths": paths,
        "exists": {
            name: (REPO_ROOT / path).exists()
            for name, path in paths.items()
        },
    }


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "processor": platform.processor()}
    try:
        if Path("/proc/cpuinfo").exists():
            model = None
            count = 0
            with Path("/proc/cpuinfo").open() as f:
                for line in f:
                    if line.startswith("model name") and model is None:
                        model = line.split(":", 1)[1].strip()
                    if line.startswith("processor"):
                        count += 1
            info["model_name"] = model
            info["logical_cpu_count"] = count or os.cpu_count()
        else:
            info["logical_cpu_count"] = os.cpu_count()
    except Exception as exc:
        info["unavailable_reason"] = str(exc)
    return info


def gpu_info() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return {"nvidia_smi": [line.strip() for line in out.splitlines() if line.strip()]}


def save_losses(losses: dict[str, list[float]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    curve_path = output_dir / "validation_curve.csv"
    with curve_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_nll", "validation_nll"])
        for i, (train, val) in enumerate(zip(losses["train"], losses["val"])):
            writer.writerow([i, train, val])
    plt.plot(losses["train"], label="train")
    plt.plot(losses["val"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_curve.pdf")
    plt.clf()


def _observed_summary(seed: int, n_obs: int, convention: str = "gaussian"):
    import jax.numpy as jnp
    import jax.random as random

    from npe_convergence.examples.gnk import gnk, ss_octile

    key = random.key(seed)
    if convention == "flow":
        subkey = key
    elif convention == "gaussian":
        _, subkey = random.split(key)
    else:
        raise ValueError(f"unknown convention: {convention}")
    true_params = jnp.array([3.0, 1.0, 2.0, 0.5])
    z = random.normal(subkey, shape=(n_obs,))
    return jnp.squeeze(ss_octile(jnp.atleast_2d(gnk(z, *true_params))))


def _reference_convention(config: dict[str, Any]) -> str:
    path = str(config.get("nuts_reference_cache_path", ""))
    if "_conv_flow" in path:
        return "flow"
    if "_conv_gaussian" in path:
        return "gaussian"
    return "gaussian"


def load_reference_samples(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        payload = pkl.load(f)
    if isinstance(payload, dict) and "samples" in payload:
        samples = np.asarray(payload["samples"], dtype=np.float64)
        if samples.ndim == 3:
            return samples.reshape(-1, samples.shape[-1])
        return samples
    return np.asarray(payload, dtype=np.float64)


def deduplicate(samples: np.ndarray) -> tuple[np.ndarray, int]:
    unique = np.unique(np.asarray(samples, dtype=np.float64), axis=0)
    return unique, int(samples.shape[0] - unique.shape[0])


def write_config_snapshot(config: dict[str, Any], output_dir: Path) -> None:
    path = output_dir / "config.yaml"
    text = yaml.safe_dump(config, sort_keys=False)
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(
                f"Config snapshot already exists with different content: {rel(path)}"
            )
        return
    path.write_text(text)


def _sample_asymptotic_mvn_batch(key, theta_batch, n_obs: int):
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from jax.scipy.stats import norm

    from npe_convergence.examples.gnk import compute_covariance_matrix, gnk

    quantiles = jnp.linspace(1.0 / (D_S + 1), 1.0 - 1.0 / (D_S + 1), D_S)
    z = norm.ppf(quantiles)
    A, B, g, k = theta_batch.T
    means = gnk(z[None, :], A[:, None], B[:, None], g[:, None], k[:, None])
    covs = jax.vmap(
        lambda a, b, gg, kk: compute_covariance_matrix(a, b, gg, kk, quantiles, n_obs)
    )(A, B, g, k)

    eye = jnp.eye(D_S)
    base_jitter = 1e-6

    def choose_cov(cov):
        cov = 0.5 * (cov + cov.T)
        finite_cov = jnp.all(jnp.isfinite(cov))
        cov_for_eig = jnp.where(finite_cov, cov, jnp.eye(D_S))
        raw_min_eig = jnp.linalg.eigvalsh(cov_for_eig)[0]
        candidate = cov_for_eig + base_jitter * eye
        chol = jnp.linalg.cholesky(candidate)
        regularized_min_eig = jnp.linalg.eigvalsh(candidate)[0]
        ok = jnp.logical_and(finite_cov, jnp.all(jnp.isfinite(chol)))
        ok = jnp.logical_and(ok, regularized_min_eig > 0)
        return chol, raw_min_eig, regularized_min_eig, ok, finite_cov

    chols, raw_min_eig, regularized_min_eig, ok, finite_cov = (
        jax.vmap(choose_cov)(covs)
    )
    eps = random.normal(key, means.shape)
    draws = means + jnp.einsum("bij,bj->bi", chols, eps)
    diagnostics = {
        "batch_size": int(theta_batch.shape[0]),
        "summary_shape": [int(theta_batch.shape[0]), D_S],
        "finite_summary_count": int(jnp.isfinite(draws).sum()),
        "all_summaries_finite": bool(jnp.all(jnp.isfinite(draws))),
        "nonfinite_covariance_count": int(jnp.count_nonzero(~finite_cov)),
        "invalid_covariance_count": int(jnp.count_nonzero(~ok)),
        "max_jitter_used": base_jitter,
        "min_raw_cov_eig": float(jnp.min(raw_min_eig)),
        "min_cov_eig_after_jitter": float(jnp.min(regularized_min_eig)),
        "mvn_mean_function": "npe_convergence.examples.gnk.gnk",
        "mvn_covariance_function": "npe_convergence.examples.gnk.compute_covariance_matrix",
        "covariance_regularisation_rule": (
            "symmetrise covariance and add the same 1e-6 diagonal jitter used "
            "by gnk_model; invalid prior draws are resampled and counted"
        ),
    }
    return draws, diagnostics, ok


def sample_asymptotic_mvn_summaries(key, theta_bounded, n_obs: int, batch_size: int):
    import jax.numpy as jnp
    import jax.random as random
    import numpy as np

    n_sims = int(theta_bounded.shape[0])
    batches = []
    theta_batches = []
    aggregate = {
        "batches": 0,
        "simulations": n_sims,
        "summary_dimension": D_S,
        "nonfinite_covariance_count": 0,
        "invalid_covariance_initial_count": 0,
        "invalid_covariance_final_count": 0,
        "invalid_covariance_resample_count": 0,
        "max_resample_passes": 0,
        "max_jitter_used": 0.0,
        "min_raw_cov_eig": math.inf,
        "min_cov_eig_after_jitter": math.inf,
        "all_summaries_finite": True,
        "mvn_mean_function": "npe_convergence.examples.gnk.gnk",
        "mvn_covariance_function": "npe_convergence.examples.gnk.compute_covariance_matrix",
        "base_jitter": 1e-6,
        "covariance_regularisation_rule": (
            "symmetrise covariance and add the same 1e-6 diagonal jitter used "
            "by gnk_model; invalid prior draws are resampled and counted"
        ),
    }
    for start in range(0, n_sims, batch_size):
        end = min(start + batch_size, n_sims)
        theta_batch = theta_bounded[start:end]
        key, subkey = random.split(key)
        draws, diag, ok = _sample_asymptotic_mvn_batch(subkey, theta_batch, n_obs)
        initial_invalid = int(np.count_nonzero(~np.asarray(ok)))
        resample_passes = 0
        while initial_invalid and np.any(~np.asarray(ok)):
            bad = np.asarray(~ok)
            n_bad = int(np.count_nonzero(bad))
            if resample_passes >= 20:
                break
            key, subkey_theta, subkey_summary = random.split(key, 3)
            replacements = random.uniform(
                subkey_theta,
                shape=(n_bad, D_THETA),
                minval=1e-6,
                maxval=10 - 1e-6,
            )
            theta_batch = theta_batch.at[bad].set(replacements)
            draws = draws.at[bad].set(0.0)
            resample_passes += 1
            draws, diag, ok = _sample_asymptotic_mvn_batch(
                subkey_summary, theta_batch, n_obs
            )
            aggregate["invalid_covariance_resample_count"] += n_bad

        final_invalid = int(np.count_nonzero(~np.asarray(ok)))
        batches.append(draws)
        theta_batches.append(theta_batch)
        aggregate["batches"] += 1
        aggregate["nonfinite_covariance_count"] += diag["nonfinite_covariance_count"]
        aggregate["invalid_covariance_initial_count"] += initial_invalid
        aggregate["invalid_covariance_final_count"] += final_invalid
        aggregate["max_resample_passes"] = max(
            aggregate["max_resample_passes"], resample_passes
        )
        aggregate["max_jitter_used"] = max(aggregate["max_jitter_used"], diag["max_jitter_used"])
        aggregate["min_raw_cov_eig"] = min(
            aggregate["min_raw_cov_eig"], diag["min_raw_cov_eig"]
        )
        aggregate["min_cov_eig_after_jitter"] = min(
            aggregate["min_cov_eig_after_jitter"], diag["min_cov_eig_after_jitter"]
        )
        aggregate["all_summaries_finite"] = (
            aggregate["all_summaries_finite"] and diag["all_summaries_finite"]
        )
    return jnp.concatenate(batches, axis=0), aggregate, jnp.concatenate(theta_batches, axis=0)


def smoke_test(batch_size: int = 8, n_obs: int = N_OBS, seed: int = OBSERVED_SEED) -> dict[str, Any]:
    import jax.numpy as jnp
    import jax.random as random
    import numpyro.distributions as dist  # type: ignore

    key = random.key(seed)
    key, subkey = random.split(key)
    theta = dist.Uniform(1e-6, 10 - 1e-6).sample(subkey, (batch_size, D_THETA))
    key, subkey = random.split(key)
    summaries, diagnostics, _ = sample_asymptotic_mvn_summaries(
        subkey, theta, n_obs=n_obs, batch_size=batch_size
    )
    diagnostics["requested_batch_size"] = batch_size
    diagnostics["observed_shape_ok"] = tuple(summaries.shape) == (batch_size, D_S)
    diagnostics["finite_values_ok"] = bool(jnp.all(jnp.isfinite(summaries)))
    if (
        not diagnostics["observed_shape_ok"]
        or not diagnostics["finite_values_ok"]
        or diagnostics["invalid_covariance_final_count"] > 0
    ):
        raise SystemExit(f"Smoke check failed: {diagnostics}")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return diagnostics


def run_pilot(config: dict[str, Any]) -> int:
    import jax.numpy as jnp
    import jax.random as random
    import numpy as np
    import numpyro  # type: ignore
    import numpyro.distributions as dist  # type: ignore
    from jax.scipy.special import expit, logit

    from npe_convergence.methods.gaussian_npe import (
        ConditionalGaussianNPE,
        TrainConfig,
        fit,
        sample,
    )
    from npe_convergence.metrics import kullback_leibler, median_heuristic, unbiased_mmd

    output_dir = REPO_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    collisions = collision_paths(config, allow_config=True)
    if collisions:
        raise SystemExit(
            "Selected run has existing output paths; refusing to run:\n"
            + "\n".join(collisions)
        )

    metadata: dict[str, Any] = {
        "scheduler_job_id": os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "start_timestamp_utc": utc_now(),
        "end_timestamp_utc": None,
        "total_wall_time_seconds": None,
        "simulation_time_seconds": None,
        "training_time_seconds": None,
        "posterior_sampling_time_seconds": None,
        "simulations_per_second": None,
        "training_examples_per_second": None,
        "epochs_per_second": None,
        "cpu_info": cpu_info(),
        "gpu_info": gpu_info(),
        "environment": {
            "python": sys.version.split()[0],
            "jax_version": jax.__version__,
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "numpyro_version": numpyro.__version__,
            "git_commit": run_git(["rev-parse", "HEAD"]),
            "git_dirty": git_dirty(),
        },
        "peak_memory_kb": None,
        "peak_memory_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "exit_status": None,
        "failure_reason": None,
        "output_integrity": {},
    }

    start = time.perf_counter()
    try:
        seed = int(config["observed_seed"])
        n_obs = int(config["n"])
        n_sims = int(config["N"])
        simulation_seed = int(config.get("simulation_seed", seed))
        training_seed = int(config.get("training_seed", simulation_seed))
        posterior_seed = int(config.get("posterior_sampling_seed", seed))
        convention = _reference_convention(config)
        x_obs = _observed_summary(seed, n_obs, convention=convention)
        write_config_snapshot(config, output_dir)

        nuts_cache = REPO_ROOT / config["nuts_reference_cache_path"]
        if not nuts_cache.exists():
            raise FileNotFoundError(f"Required read-only NUTS cache is missing: {rel(nuts_cache)}")
        true_posterior_samples = load_reference_samples(nuts_cache)

        theta_key = rng_for(
            training_seed, "model_control", n_obs, seed, "theta_prior", config["simulator"]
        )
        theta_bounded = dist.Uniform(1e-6, 10 - 1e-6).sample(
            theta_key, (n_sims, D_THETA)
        )

        sim_start = time.perf_counter()
        subkey = rng_for(
            simulation_seed, "model_control", n_obs, seed, training_seed, "summaries"
        )
        summaries, simulator_diagnostics, theta_bounded = sample_asymptotic_mvn_summaries(
            subkey,
            theta_bounded,
            n_obs=n_obs,
            batch_size=min(1000, n_sims),
        )
        theta_eta = logit(theta_bounded / 10)
        metadata["simulation_time_seconds"] = time.perf_counter() - sim_start
        metadata["simulations_per_second"] = (
            n_sims / metadata["simulation_time_seconds"]
            if metadata["simulation_time_seconds"]
            else None
        )
        (output_dir / "simulator_diagnostics.json").write_text(
            json.dumps(simulator_diagnostics, indent=2, sort_keys=True)
        )
        if (
            simulator_diagnostics["invalid_covariance_final_count"] > 0
            or not simulator_diagnostics["all_summaries_finite"]
        ):
            raise RuntimeError(
                "asymptotic-MVN simulator generated non-finite summaries "
                "or unresolved covariance failures after resampling; "
                "see simulator_diagnostics.json"
            )

        theta_mean = theta_eta.mean(axis=0)
        theta_std = theta_eta.std(axis=0)
        theta_u = (theta_eta - theta_mean) / theta_std
        summary_mean = summaries.mean(axis=0)
        summary_std = summaries.std(axis=0)
        summary_std = jnp.where(summary_std == 0, 1.0, summary_std)
        summaries_std = (summaries - summary_mean) / summary_std
        x_obs_std = (x_obs - summary_mean) / summary_std

        train_cfg = config["train_config"]
        subkey = rng_for(training_seed, "model_control", n_obs, seed, "model_init")
        model = ConditionalGaussianNPE(
            d_summary=D_S,
            d_theta=D_THETA,
            hidden_dims=tuple(train_cfg["hidden_dims"]),
            key=subkey,
        )
        fit_config = TrainConfig(
            lr=float(train_cfg["lr"]),
            batch_size=int(train_cfg["batch_size"]),
            max_epochs=int(train_cfg["max_epochs"]),
            patience=int(train_cfg["patience"]),
            val_frac=float(train_cfg["val_frac"]),
        )
        train_start = time.perf_counter()
        subkey = rng_for(training_seed, "model_control", n_obs, seed, "fit")
        model, losses = fit(model, theta_u, summaries_std, key=subkey, config=fit_config)
        metadata["training_time_seconds"] = time.perf_counter() - train_start
        epochs = len(losses["train"])
        metadata["training_examples_per_second"] = (
            n_sims * epochs / metadata["training_time_seconds"]
            if metadata["training_time_seconds"]
            else None
        )
        metadata["epochs_per_second"] = (
            epochs / metadata["training_time_seconds"]
            if metadata["training_time_seconds"]
            else None
        )
        save_losses(losses, output_dir)

        mu_u, L_u = model(x_obs_std)
        cov_u = L_u @ L_u.T
        np.savez(
            output_dir / "gaussian_npe_u_posterior.npz",
            mu_u=np.asarray(mu_u),
            cov_u=np.asarray(cov_u),
            cholesky_u=np.asarray(L_u),
            theta_unbounded_mean=np.asarray(theta_mean),
            theta_unbounded_std=np.asarray(theta_std),
            summary_mean=np.asarray(summary_mean),
            summary_std=np.asarray(summary_std),
            x_obs=np.asarray(x_obs),
            simulator=SIMULATOR,
        )

        sample_start = time.perf_counter()
        subkey = rng_for(
            posterior_seed, "model_control", n_obs, seed, training_seed, "posterior"
        )
        posterior_u = sample(model, x_obs_std, 10_000, key=subkey)
        posterior_eta = posterior_u * theta_std + theta_mean
        posterior_theta = expit(posterior_eta) * 10
        metadata["posterior_sampling_time_seconds"] = time.perf_counter() - sample_start
        np.savez(
            output_dir / "posterior_samples_10k.npz",
            theta=np.asarray(posterior_theta),
            eta=np.asarray(posterior_eta),
            u=np.asarray(posterior_u),
        )
        with (output_dir / "posterior_samples.pkl").open("wb") as f:
            pkl.dump(np.asarray(posterior_theta), f)

        n_metric = 2000
        posterior_unique, ps_duplicates = deduplicate(np.asarray(posterior_theta))
        true_unique, ref_duplicates = deduplicate(true_posterior_samples)
        if posterior_unique.shape[0] < n_metric or true_unique.shape[0] < n_metric:
            raise RuntimeError(
                "not enough unique posterior or reference samples for 2000-sample KL"
            )
        subkey = rng_for(
            posterior_seed, "model_control", n_obs, seed, training_seed, "metric_npe"
        )
        idx_npe = np.asarray(random.permutation(subkey, posterior_unique.shape[0])[:n_metric])
        subkey = rng_for(
            posterior_seed, "model_control", n_obs, seed, training_seed, "metric_ref"
        )
        idx_true = np.asarray(random.permutation(subkey, true_unique.shape[0])[:n_metric])
        ps_thin = jnp.asarray(posterior_unique[idx_npe])
        ts_thin = jnp.asarray(true_unique[idx_true])
        kl = kullback_leibler(ts_thin, ps_thin)
        lengthscale = median_heuristic(jnp.vstack([ts_thin, ps_thin]))
        mmd = unbiased_mmd(ts_thin, ps_thin, lengthscale)
        metrics = {
            "kl_theta_knn_2000": float(kl),
            "mmd_theta_2000": float(mmd),
            "n_metric": n_metric,
            "reference_duplicates_removed": int(ref_duplicates),
            "posterior_duplicates_removed": int(ps_duplicates),
            "reference_path": rel(nuts_cache),
            "reference_convention": convention,
            "observed_seed": seed,
            "training_seed": training_seed,
            "simulation_seed": simulation_seed,
            "posterior_sampling_seed": posterior_seed,
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

        metadata["exit_status"] = 0
    except Exception as exc:
        metadata["exit_status"] = 1
        metadata["failure_reason"] = repr(exc)
        raise
    finally:
        metadata["end_timestamp_utc"] = utc_now()
        metadata["total_wall_time_seconds"] = time.perf_counter() - start
        metadata["peak_memory_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for key_name, path in config["expected_outputs"].items():
            if key_name == "output_dir":
                continue
            metadata["output_integrity"][key_name] = (REPO_ROOT / path).exists()
        with (output_dir / "timing_metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
    return int(metadata["exit_status"] or 0)


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    decomp_path = REPO_ROOT / "scripts" / "compute_gnk_u_space_kl_decomp.py"
    if not decomp_path.exists():
        raise FileNotFoundError(
            "Missing reviewed decomposition helper: "
            f"{rel(decomp_path)}. Pull the branch containing the u-space "
            "decomposition script before running --evaluate."
        )
    spec = importlib.util.spec_from_file_location("gnk_u_space_decomp", decomp_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load decomposition helper from {rel(decomp_path)}")
    decomp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = decomp
    spec.loader.exec_module(decomp)

    DECOMP_D_THETA = decomp.D_THETA
    DECOMP_D_TOTAL = decomp.D_TOTAL
    compute_oracle_moments = decomp.compute_oracle_moments
    gaussian_kl_decomp_from_moments = decomp.gaussian_kl_decomp_from_moments
    self_consistency_kl_u = decomp.self_consistency_kl_u
    theta_to_u_affine_invariant = decomp.theta_to_u_affine_invariant

    if DECOMP_D_THETA != D_THETA or DECOMP_D_TOTAL != D_TOTAL:
        raise RuntimeError("Decomposition constants do not match GNK octile control config.")
    output_dir = REPO_ROOT / config["output_dir"]
    gaussian_path = output_dir / "gaussian_npe_u_posterior.npz"
    sample_path = output_dir / "posterior_samples_10k.npz"
    posterior_pkl = output_dir / "posterior_samples.pkl"

    def require_shape(name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")

    if not gaussian_path.exists():
        raise FileNotFoundError(f"Missing saved Gaussian posterior: {rel(gaussian_path)}")
    with np.load(gaussian_path) as posterior:
        mu_u = np.asarray(posterior["mu_u"], dtype=np.float64)
        cov_u = np.asarray(posterior["cov_u"], dtype=np.float64)
        theta_unbounded_mean = np.asarray(
            posterior["theta_unbounded_mean"], dtype=np.float64
        )
        theta_unbounded_std = np.asarray(
            posterior["theta_unbounded_std"], dtype=np.float64
        )
    require_shape("mu_u", mu_u, (D_THETA,))
    require_shape("cov_u", cov_u, (D_THETA, D_THETA))
    require_shape("theta_unbounded_mean", theta_unbounded_mean, (D_THETA,))
    require_shape("theta_unbounded_std", theta_unbounded_std, (D_THETA,))
    if np.any(theta_unbounded_std <= 0):
        raise ValueError("theta_unbounded_std must be positive")

    if sample_path.exists():
        with np.load(sample_path) as data:
            q_theta = np.asarray(data["theta"], dtype=np.float64)
            q_u_saved = np.asarray(data["u"], dtype=np.float64)
            q_eta = np.asarray(data["eta"], dtype=np.float64)
        require_shape("posterior_samples.theta", q_theta, (10_000, D_THETA))
        require_shape("posterior_samples.u", q_u_saved, (10_000, D_THETA))
        require_shape("posterior_samples.eta", q_eta, (10_000, D_THETA))
    elif posterior_pkl.exists():
        with posterior_pkl.open("rb") as f:
            q_theta = np.asarray(pkl.load(f), dtype=np.float64)
        q_eta, _ = theta_to_u_affine_invariant(q_theta)
        q_u_saved = None
    else:
        raise FileNotFoundError(f"Missing posterior samples: {rel(sample_path)}")

    eta_from_theta, theta_clip_count = theta_to_u_affine_invariant(q_theta)
    eta_theta_max_abs_diff = float(np.max(np.abs(q_eta - eta_from_theta)))
    eta_u_max_abs_diff = None
    if q_u_saved is not None:
        eta_from_u = q_u_saved * theta_unbounded_std + theta_unbounded_mean
        eta_u_max_abs_diff = float(np.max(np.abs(q_eta - eta_from_u)))

    scale = np.diag(theta_unbounded_std)
    direct_eta_mean = mu_u * theta_unbounded_std + theta_unbounded_mean
    direct_eta_cov = scale @ cov_u @ scale

    oracle = compute_oracle_moments(
        REPO_ROOT / "res" / "gnk",
        int(config["n"]),
        int(config["observed_seed"]),
    )
    if oracle is None:
        raise FileNotFoundError("Missing NUTS cache for evaluation.")

    q_mean = q_eta.mean(axis=0)
    q_cov = np.cov(q_eta, rowvar=False)
    delta_total, delta_mean, delta_cov, oracle_min_eig, q_min_eig = (
        gaussian_kl_decomp_from_moments(oracle.u_mean, oracle.u_cov, q_mean, q_cov)
    )
    direct_total, direct_mean, direct_cov, _, direct_q_min_eig = (
        gaussian_kl_decomp_from_moments(
            oracle.u_mean,
            oracle.u_cov,
            direct_eta_mean,
            direct_eta_cov,
        )
    )
    self_kl = self_consistency_kl_u(
        q_eta,
        q_mean,
        q_cov,
        seed=int(config["observed_seed"]) + int(config["N"]) + 30_000,
        n_metric=2000,
    )
    self_direct_kl = self_consistency_kl_u(
        q_eta,
        direct_eta_mean,
        direct_eta_cov,
        seed=int(config["observed_seed"]) + int(config["N"]) + 31_000,
        n_metric=2000,
    )
    _, _, sample_cov_min_eig = decomp.stable_cov_matrix(q_cov)
    _, _, direct_cov_eta_min_eig = decomp.stable_cov_matrix(direct_eta_cov)

    result = {
        "task": "gnk-model-control-pilot-u-space-decomposition",
        "created_at_utc": utc_now(),
        "git_commit": run_git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": git_dirty(),
        "branch": run_git(["branch", "--show-current"]),
        "method": config["method"],
        "simulator": config["simulator"],
        "n": int(config["n"]),
        "N": int(config["N"]),
        "x": config["x"],
        "seed": int(config["observed_seed"]),
        "K_theta_star": oracle.K_theta_oracle,
        "K_u_star": oracle.K_u_oracle,
        "coord_offset": oracle.coord_offset,
        "Delta_N_u": delta_total,
        "Delta_N_u_mean_component": delta_mean,
        "Delta_N_u_covariance_component": delta_cov,
        "Delta_theta_total": delta_total + oracle.coord_offset,
        "self_consistency_kl_reconstructed_Qhat_N_u": self_kl,
        "direct_saved_gaussian_reconstruction": {
            "Delta_N_u": direct_total,
            "Delta_N_u_mean_component": direct_mean,
            "Delta_N_u_covariance_component": direct_cov,
            "Delta_theta_total": direct_total + oracle.coord_offset,
            "self_consistency_kl_saved_Gaussian_against_saved_eta_samples": self_direct_kl,
            "qhat_min_eig_eta": direct_q_min_eig,
            "cov_eta_min_eig": direct_cov_eta_min_eig,
        },
        "coordinate": "eta = logit(theta / 10), affine-equivalent to standardized u",
        "qhat_source": "posterior_samples_10k.npz eta sample moments",
        "oracle_min_eig_eta": oracle_min_eig,
        "qhat_min_eig_eta": q_min_eig,
        "sample_eta_cov_min_eig": sample_cov_min_eig,
        "schema_checks": {
            "posterior_theta_shape": list(q_theta.shape),
            "posterior_eta_shape": list(q_eta.shape),
            "posterior_u_shape": list(q_u_saved.shape) if q_u_saved is not None else None,
            "mu_u_shape": list(mu_u.shape),
            "cov_u_shape": list(cov_u.shape),
            "theta_clip_count_when_recomputing_eta": int(theta_clip_count),
            "eta_from_theta_max_abs_diff": eta_theta_max_abs_diff,
            "eta_from_saved_u_max_abs_diff": eta_u_max_abs_diff,
            "finite_decomposition_components": bool(
                np.isfinite(
                    [
                        oracle.K_theta_oracle,
                        oracle.K_u_oracle,
                        oracle.coord_offset,
                        delta_total,
                        delta_mean,
                        delta_cov,
                        direct_total,
                        direct_mean,
                        direct_cov,
                        self_kl,
                        self_direct_kl,
                    ]
                ).all()
            ),
            "covariance_component_nonnegative_with_tolerance": bool(
                delta_cov >= -1e-8 and direct_cov >= -1e-8
            ),
        },
        "nuts_cache_path": rel(oracle.nuts_path),
        "posterior_sample_path": rel(sample_path if sample_path.exists() else posterior_pkl),
        "gaussian_npe_u_posterior_path": rel(gaussian_path),
        "note": (
            "Delta_N,u is native u-space Gaussian-NPE error. This output uses "
            "the reviewed coordinate-aware eta convention, which is affine-equivalent "
            "to standardized u for exact Gaussian-Gaussian terms. Do not compare this "
            "to smoke-test metrics.json KL/MMD."
        ),
    }
    output_path = output_dir / "u_space_decomposition.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--allow-existing-config", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--selected-x", type=int, default=SELECTED_X_DEFAULT, choices=CANDIDATE_X)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--created-at", type=str, default=None)
    parser.add_argument("--model-control-code-commit", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-batch-size", type=int, default=8)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke_test:
        smoke_test(batch_size=args.smoke_batch_size)
        return 0

    if args.config:
        config = load_config(args.config)
    else:
        created_at = args.created_at or utc_now()
        config = build_config(
            selected_x=args.selected_x,
            created_at=created_at,
            run_id=args.run_id,
            model_control_code_commit=args.model_control_code_commit,
        )

    if args.dry_run:
        print_dry_run(config)
        if args.write_config:
            write_config(config, allow_existing_config=args.allow_existing_config)
        return 0

    if args.evaluate:
        if not args.config:
            parser.error("--evaluate requires --config")
        evaluate(config)
        return 0

    if not args.config:
        parser.error("run mode requires --config")

    import numpyro  # type: ignore

    numpyro.set_host_device_count(4)
    return run_pilot(config)


if __name__ == "__main__":
    raise SystemExit(main())
