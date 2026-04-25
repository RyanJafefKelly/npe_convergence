"""Prepare or run one guarded GNK Gaussian-NPE HPC calibration job.

This script intentionally supports one selected calibration run at a time.  The
dry-run mode prints both a table and JSON for x in {25, 50}; the run mode reads
the resolved config and executes only that single row.
"""

from __future__ import annotations

import argparse
import csv
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DECOMP_CSV = REPO_ROOT / "notebooks/plots/gnk_u_space_kl_decomp_20260425_per_seed.csv"
OUTPUT_ROOT = REPO_ROOT / "res/gnk_hpc_calibration"
CANDIDATE_X = (25, 50)
N_OBS = 500
D_S = 7
D_THETA = 4
D_TOTAL = D_S + D_THETA
METHOD = "Gaussian-NPE"
SIMULATOR = "empirical_gnk_prior_predictive_octile_summaries"
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


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_utc_stamp(created_at: str) -> str:
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%dT%H%M%SZ")


def select_seed(csv_path: Path = DECOMP_CSV) -> dict[str, Any]:
    try:
        rows: list[dict[str, Any]] = []
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if int(row["n"]) != 500 or int(row["N"]) != 250000:
                    continue
                delta = float(row["Delta_u_total"])
                if not math.isfinite(delta):
                    continue
                rows.append({"seed": int(row["seed"]), "Delta_u_total": delta})
        if not rows:
            raise ValueError("no complete finite rows found")
        deltas = sorted(r["Delta_u_total"] for r in rows)
        median = deltas[len(deltas) // 2]
        selected = min(rows, key=lambda r: abs(r["Delta_u_total"] - median))
        return {
            "seed": selected["seed"],
            "rule": (
                "closest finite Delta_u_total to the median among n=500, "
                "N=250000 complete-seed rows"
            ),
            "source_csv": rel(csv_path),
            "source_rows": len(rows),
            "median_Delta_u_total": median,
            "selected_seed_Delta_u_total": selected["Delta_u_total"],
            "fallback_used": False,
        }
    except Exception as exc:
        return {
            "seed": 0,
            "rule": (
                "fallback to seed 0 because the preferred seed-selection CSV "
                "was unavailable or malformed"
            ),
            "source_csv": rel(csv_path),
            "source_rows": 0,
            "median_Delta_u_total": None,
            "selected_seed_Delta_u_total": None,
            "fallback_used": True,
            "fallback_reason": str(exc),
        }


def candidate_paths(run_id: str) -> dict[str, str]:
    out = OUTPUT_ROOT / run_id
    return {
        "output_dir": rel(out),
        "config": rel(out / "config.yaml"),
        "validation_curve": rel(out / "validation_curve.csv"),
        "validation_curve_plot": rel(out / "validation_curve.pdf"),
        "predicted_u_mean_cov": rel(out / "gaussian_npe_u_posterior.npz"),
        "samples_10k": rel(out / "posterior_samples_10k.npz"),
        "timing_metadata": rel(out / "timing_metadata.json"),
        "stdout_log": rel(out / "logs/stdout.log"),
        "stderr_log": rel(out / "logs/stderr.log"),
    }


def build_rows(
    *,
    selected_x: int,
    seed: int,
    created_at: str,
    run_id: str | None,
) -> list[dict[str, Any]]:
    rows = []
    stamp = compact_utc_stamp(created_at)
    for x in CANDIDATE_X:
        n_sims = x * D_TOTAL * D_TOTAL * N_OBS
        rid = run_id if x == selected_x and run_id else (
            f"gnk_gaussian_npe_n500_x{x}_seed{seed}_{stamp}"
        )
        paths = candidate_paths(rid)
        exists = {name: (REPO_ROOT / path).exists() for name, path in paths.items()}
        rows.append(
            {
                "n": N_OBS,
                "x": x,
                "N": n_sims,
                "seed": seed,
                "method": METHOD,
                "simulator": SIMULATOR,
                "selected": x == selected_x,
                "run_id": rid,
                "paths": paths,
                "exists": exists,
            }
        )
    return rows


def build_config(
    *,
    selected_x: int,
    created_at: str,
    run_id: str | None,
) -> dict[str, Any]:
    seed_info = select_seed()
    rows = build_rows(
        selected_x=selected_x,
        seed=seed_info["seed"],
        created_at=created_at,
        run_id=run_id,
    )
    selected = next(row for row in rows if row["selected"])
    seed = seed_info["seed"]
    return {
        "task": "gnk-hpc-calibration",
        "run_id": selected["run_id"],
        "created_at_utc": created_at,
        "git_commit": run_git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": git_dirty(),
        "branch": run_git(["branch", "--show-current"]),
        "method": METHOD,
        "simulator": SIMULATOR,
        "n": N_OBS,
        "d_s": D_S,
        "d_theta": D_THETA,
        "d": D_TOTAL,
        "x": selected_x,
        "N": selected["N"],
        "observed_seed": seed,
        "simulation_seed": seed,
        "training_seed": seed,
        "posterior_sampling_seed": seed,
        "output_dir": selected["paths"]["output_dir"],
        "gpu_requested": False,
        "scheduler_resource_request": RESOURCE_REQUEST,
        "seed_selection": seed_info,
        "selection_rationale": (
            "Selected x=50 because it is the preferred calibration point, "
            "the resolved N=3,025,000 is below the existing cached n=5000, "
            "N=25,000,000 scale, and the repository already has PBS patterns "
            "using 47-48h and 64GB. The job remains a single non-array run."
        ),
        "train_config": {
            "lr": 5e-4,
            "batch_size": 256,
            "max_epochs": 2000,
            "patience": 200,
            "val_frac": 0.1,
            "hidden_dims": [128, 128],
        },
        "expected_outputs": selected["paths"],
        "dry_run_rows": rows,
        "full_array_submitted": False,
    }


def collision_paths(config: dict[str, Any], *, allow_config: bool = False) -> list[str]:
    skip = {"output_dir", "stdout_log", "stderr_log"}
    if allow_config:
        skip.add("config")
    collisions = []
    for key, value in config["expected_outputs"].items():
        if key in skip:
            continue
        if (REPO_ROOT / value).exists():
            collisions.append(value)
    return collisions


def print_dry_run(config: dict[str, Any]) -> None:
    refresh_exists(config)
    print("GNK Gaussian-NPE HPC calibration dry-run")
    print(f"created_at_utc: {config['created_at_utc']}")
    print(f"git_commit: {config['git_commit']} dirty={config['git_dirty']}")
    print(f"selected_seed: {config['observed_seed']}")
    print("")
    header = (
        "selected  n    x    N        seed  method        simulator"
        "                                output_dir"
    )
    print(header)
    print("-" * len(header))
    for row in config["dry_run_rows"]:
        mark = "yes" if row["selected"] else "no "
        print(
            f"{mark:8s} {row['n']:<4d} {row['x']:<4d} {row['N']:<8d} "
            f"{row['seed']:<5d} {row['method']:<13s} "
            f"{row['simulator']:<40s} {row['paths']['output_dir']}"
        )
        for label in (
            "config",
            "validation_curve",
            "predicted_u_mean_cov",
            "samples_10k",
            "timing_metadata",
            "stdout_log",
            "stderr_log",
        ):
            print(
                f"    {label}: {row['paths'][label]} "
                f"exists={str(row['exists'][label]).lower()}"
            )
    selected_collisions = collision_paths(config, allow_config=True)
    if selected_collisions:
        print("")
        print("WARNING: SELECTED-RUN OUTPUT PATH COLLISIONS DETECTED")
        for path in selected_collisions:
            print(f"  {path}")
    print("")
    print("DRY_RUN_JSON_BEGIN")
    print(json.dumps(config, indent=2, sort_keys=True))
    print("DRY_RUN_JSON_END")


def refresh_exists(config: dict[str, Any]) -> None:
    for row in config["dry_run_rows"]:
        row["exists"] = {
            name: (REPO_ROOT / path).exists()
            for name, path in row["paths"].items()
        }


def write_config(config: dict[str, Any], *, allow_existing_config: bool = False) -> None:
    config_path = REPO_ROOT / config["expected_outputs"]["config"]
    output_dir = REPO_ROOT / config["output_dir"]
    log_dir = output_dir / "logs"
    collisions = collision_paths(config, allow_config=True)
    if collisions:
        raise SystemExit(
            "Selected run has existing output paths; refusing to write config:\n"
            + "\n".join(collisions)
        )
    if config_path.exists() and not allow_existing_config:
        raise SystemExit(f"Config already exists, refusing to overwrite: {rel(config_path)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote resolved config: {rel(config_path)}")


def load_config(path: str) -> dict[str, Any]:
    config_path = REPO_ROOT / path
    with config_path.open() as f:
        config = yaml.safe_load(f)
    if config["x"] not in CANDIDATE_X:
        raise ValueError(f"unexpected calibration x: {config['x']}")
    return config


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


def run_calibration(config: dict[str, Any]) -> int:
    import jax.numpy as jnp
    import jax.random as random
    import numpy as np
    import numpyro.distributions as dist  # type: ignore
    from jax.scipy.special import expit, logit

    from npe_convergence.examples.gnk import gnk, get_summaries_batches, ss_octile
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
        key = random.key(seed)
        true_params = jnp.array([3.0, 1.0, 2.0, 0.5])

        key, subkey = random.split(key)
        z = random.normal(subkey, shape=(n_obs,))
        x_obs = jnp.squeeze(ss_octile(jnp.atleast_2d(gnk(z, *true_params))))

        nuts_cache = REPO_ROOT / f"res/gnk/nuts_cache_v2_n_obs_{n_obs}_seed_{seed}.pkl"
        if not nuts_cache.exists():
            raise FileNotFoundError(
                f"Required read-only NUTS cache is missing: {rel(nuts_cache)}"
            )
        with nuts_cache.open("rb") as f:
            true_posterior_samples = pkl.load(f)

        key, subkey = random.split(key)
        tol = 1e-6
        thetas_bounded = dist.Uniform(0 + tol, 10 - tol).sample(subkey, (n_sims, 4))
        thetas_unbounded = logit(thetas_bounded / 10)
        A_sim, B_sim, g_sim, k_sim = thetas_bounded.T

        key, subkey = random.split(key)
        sim_start = time.perf_counter()
        x_sims = get_summaries_batches(
            subkey,
            A_sim,
            B_sim,
            g_sim,
            k_sim,
            n_obs,
            n_sims,
            batch_size=min(1000, n_sims),
        )
        metadata["simulation_time_seconds"] = time.perf_counter() - sim_start
        metadata["simulations_per_second"] = (
            n_sims / metadata["simulation_time_seconds"]
            if metadata["simulation_time_seconds"]
            else None
        )

        thetas_mean = thetas_unbounded.mean(axis=0)
        thetas_std = thetas_unbounded.std(axis=0)
        thetas = (thetas_unbounded - thetas_mean) / thetas_std
        sim_summ_data = x_sims.T
        sim_summ_data_mean = sim_summ_data.mean(axis=0)
        sim_summ_data_std = sim_summ_data.std(axis=0)
        sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std
        x_obs_std = (x_obs - sim_summ_data_mean) / sim_summ_data_std

        train_cfg = config["train_config"]
        key, subkey = random.split(key)
        model = ConditionalGaussianNPE(
            d_summary=D_S,
            d_theta=D_THETA,
            hidden_dims=tuple(train_cfg["hidden_dims"]),
            key=subkey,
        )
        key, subkey = random.split(key)
        fit_config = TrainConfig(
            lr=float(train_cfg["lr"]),
            batch_size=int(train_cfg["batch_size"]),
            max_epochs=int(train_cfg["max_epochs"]),
            patience=int(train_cfg["patience"]),
            val_frac=float(train_cfg["val_frac"]),
        )
        train_start = time.perf_counter()
        model, losses = fit(model, thetas, sim_summ_data, key=subkey, config=fit_config)
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
            theta_unbounded_mean=np.asarray(thetas_mean),
            theta_unbounded_std=np.asarray(thetas_std),
            summary_mean=np.asarray(sim_summ_data_mean),
            summary_std=np.asarray(sim_summ_data_std),
            x_obs=np.asarray(x_obs),
        )

        sample_start = time.perf_counter()
        key, subkey = random.split(key)
        posterior_u = sample(model, x_obs_std, 10_000, key=subkey)
        posterior_eta = posterior_u * thetas_std + thetas_mean
        posterior_theta = expit(posterior_eta) * 10
        metadata["posterior_sampling_time_seconds"] = time.perf_counter() - sample_start
        np.savez(
            output_dir / "posterior_samples_10k.npz",
            theta=np.asarray(posterior_theta),
            u=np.asarray(posterior_u),
            eta=np.asarray(posterior_eta),
        )
        with (output_dir / "posterior_samples.pkl").open("wb") as f:
            pkl.dump(posterior_theta, f)

        n_metric = 2000
        key, subkey = random.split(key)
        idx_npe = random.permutation(subkey, posterior_theta.shape[0])[:n_metric]
        key, subkey = random.split(key)
        idx_true = random.permutation(subkey, true_posterior_samples.shape[0])[:n_metric]
        ps_thin = posterior_theta[idx_npe]
        ts_thin = true_posterior_samples[idx_true]
        kl = kullback_leibler(ts_thin, ps_thin)
        lengthscale = median_heuristic(jnp.vstack([ts_thin, ps_thin]))
        mmd = unbiased_mmd(ts_thin, ps_thin, lengthscale)
        metrics = {"kl_theta_knn_2000": float(kl), "mmd_theta_2000": float(mmd)}
        with (output_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

        metadata["exit_status"] = 0
    except Exception as exc:
        metadata["exit_status"] = 1
        metadata["failure_reason"] = repr(exc)
        raise
    finally:
        metadata["end_timestamp_utc"] = utc_now()
        metadata["total_wall_time_seconds"] = time.perf_counter() - start
        metadata["peak_memory_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for key, path in config["expected_outputs"].items():
            if key == "output_dir":
                continue
            metadata["output_integrity"][key] = (REPO_ROOT / path).exists()
        with (output_dir / "timing_metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    return int(metadata["exit_status"] or 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--allow-existing-config", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--selected-x", type=int, default=50, choices=CANDIDATE_X)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--created-at", type=str, default=None)
    args = parser.parse_args(argv)

    if args.config:
        config = load_config(args.config)
    else:
        created_at = args.created_at or utc_now()
        config = build_config(
            selected_x=args.selected_x,
            created_at=created_at,
            run_id=args.run_id,
        )

    if args.dry_run:
        print_dry_run(config)
        if args.write_config:
            write_config(config, allow_existing_config=args.allow_existing_config)
        return 0

    if not args.config:
        parser.error("run mode requires --config")
    import numpyro  # type: ignore

    numpyro.set_host_device_count(4)
    return run_calibration(config)


if __name__ == "__main__":
    raise SystemExit(main())
