"""Run g-and-k hexadecile summaries with conditional Gaussian NPE.

This is a cache-safe counterpart to ``run_gnk_gaussian.py`` for the
octile-vs-hexadecile paper figure. It writes to a fresh namespace by default
and refuses to run if the selected per-cell output directory already exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle as pkl
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax.scipy.special import expit, logit
from numpyro.diagnostics import hpdi  # type: ignore

from npe_convergence.examples.gnk import (
    get_summaries_batches,
    gnk,
    run_nuts,
    ss_hexadeciles,
)
from npe_convergence.methods.gaussian_npe import (
    ConditionalGaussianNPE,
    TrainConfig,
    fit,
    sample,
)
from npe_convergence.metrics import kullback_leibler, median_heuristic, unbiased_mmd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "res" / "gnk_hexadeciles_gaussian"
PARAM_NAMES = ("A", "B", "g", "k")
TRUE_PARAMS = jnp.array([3.0, 1.0, 2.0, 0.5])
SUMMARY_NAME = "hexadeciles"
D_SUMMARY = 15
D_THETA = 4
D_TOTAL = D_SUMMARY + D_THETA


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


def output_dir(output_root: Path, n_obs: int, n_sims: int, seed: int) -> Path:
    return output_root / f"gaussian_npe_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}"


def nuts_cache_path(output_root: Path, n_obs: int, seed: int) -> Path:
    return output_root / f"nuts_cache_v1_n_obs_{n_obs}_seed_{seed}.pkl"


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


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


def write_losses_csv(path: Path, losses: dict[str, list[float]]) -> None:
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


def expected_paths(out: Path) -> dict[str, Path]:
    return {
        "config": out / "config.json",
        "metadata": out / "metadata.json",
        "x_obs": out / "x_obs.npy",
        "standardization": out / "standardization.npz",
        "gaussian_npe_native_u_posterior": out / "gaussian_npe_native_u_posterior.npz",
        "posterior_samples": out / "posterior_samples.pkl",
        "true_posterior_samples": out / "true_posterior_samples.pkl",
        "kl": out / "kl.txt",
        "mmd": out / "mmd.txt",
        "estimated_coverage": out / "estimated_coverage.npy",
        "biases": out / "biases.npy",
        "losses": out / "losses.csv",
    }


def build_config(
    *,
    output_root: Path,
    n_obs: int,
    n_sims: int,
    seed: int,
    nuts_seed: int,
    num_posterior_samples: int,
    num_nuts_samples: int,
    num_nuts_warmup: int,
    num_coverage_samples: int,
    num_metric_samples: int,
    train_config: TrainConfig,
    hidden_dims: tuple[int, ...],
    batch_size: int,
    prior_batch_size: int,
    smoke_reference_samples: bool,
    save_plots: bool,
) -> dict[str, Any]:
    out = output_dir(output_root, n_obs, n_sims, seed)
    return {
        "task": "gnk-hexadecile-gaussian-npe",
        "created_at_utc": utc_now(),
        "git_commit": run_git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": git_dirty(),
        "branch": run_git(["branch", "--show-current"]),
        "method": "Gaussian-NPE",
        "summary_name": SUMMARY_NAME,
        "d_summary": D_SUMMARY,
        "d_theta": D_THETA,
        "d_total": D_TOTAL,
        "n_obs": n_obs,
        "n_sims": n_sims,
        "seed": seed,
        "nuts_seed": nuts_seed,
        "true_params": np.asarray(TRUE_PARAMS).tolist(),
        "output_root": rel(output_root),
        "output_dir": rel(out),
        "nuts_cache": rel(nuts_cache_path(output_root, n_obs, seed)),
        "num_posterior_samples": num_posterior_samples,
        "num_nuts_samples": num_nuts_samples,
        "num_nuts_warmup": num_nuts_warmup,
        "num_coverage_samples": num_coverage_samples,
        "num_metric_samples": num_metric_samples,
        "train_config": {
            "lr": train_config.lr,
            "batch_size": batch_size,
            "max_epochs": train_config.max_epochs,
            "patience": train_config.patience,
            "val_frac": train_config.val_frac,
            "hidden_dims": list(hidden_dims),
        },
        "prior_batch_size": prior_batch_size,
        "smoke_reference_samples": smoke_reference_samples,
        "smoke_reference_samples_note": (
            "Synthetic reference samples are for non-scientific smoke tests only."
            if smoke_reference_samples
            else None
        ),
        "save_plots": save_plots,
        "expected_outputs": {k: rel(v) for k, v in expected_paths(out).items()},
    }


def print_dry_run(config: dict[str, Any]) -> None:
    print("GNK hexadecile Gaussian-NPE dry-run")
    print(f"output_dir: {config['output_dir']}")
    print(f"n_obs: {config['n_obs']}")
    print(f"n_sims: {config['n_sims']}")
    print(f"seed: {config['seed']}")
    print(f"d_summary: {config['d_summary']}")
    print(f"d_theta: {config['d_theta']}")
    print(f"d_total: {config['d_total']}")
    print("expected_outputs:")
    for name, path in config["expected_outputs"].items():
        print(f"  {name}: {path}")


def fail_if_collision(out: Path) -> None:
    if out.exists():
        raise FileExistsError(
            f"Refusing to run because output directory already exists: {out}"
        )


def observed_hexadeciles(seed: int, n_obs: int) -> tuple[jnp.ndarray, Any]:
    key = random.key(seed)
    key, subkey = random.split(key)
    z = random.normal(subkey, shape=(n_obs,))
    x_obs = gnk(z, *TRUE_PARAMS)
    x_obs = jnp.squeeze(ss_hexadeciles(jnp.atleast_2d(x_obs)))
    return x_obs, key


def stack_nuts_samples(samples_dict: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.column_stack([samples_dict[name] for name in PARAM_NAMES])


def get_reference_posterior(
    *,
    output_root: Path,
    seed: int,
    nuts_seed: int,
    x_obs: jnp.ndarray,
    n_obs: int,
    num_samples: int,
    num_warmup: int,
    smoke_reference_samples: bool,
    key: Any,
) -> tuple[jnp.ndarray, str]:
    if smoke_reference_samples:
        if is_under(output_root, REPO_ROOT / "res"):
            raise ValueError(
                "--smoke-reference-samples is only allowed outside the repo res/ tree"
            )
        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=(num_samples, D_THETA)) * 0.05
        samples = jnp.clip(TRUE_PARAMS + noise, 1e-6, 10.0 - 1e-6)
        return samples, "synthetic_smoke_reference"

    cache_path = nuts_cache_path(output_root, n_obs, seed)
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pkl.load(f), "cache"

    print(f"Running NUTS reference posterior; cache will be {cache_path}")
    mcmc = run_nuts(
        seed=nuts_seed,
        obs=x_obs,
        n_obs=n_obs,
        num_samples=num_samples,
        num_warmup=num_warmup,
    )
    mcmc.print_summary()
    samples = stack_nuts_samples(mcmc.get_samples())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pkl.dump(samples, f)
    return samples, "nuts"


def metric_values(
    *,
    key: Any,
    true_posterior_samples: jnp.ndarray,
    posterior_samples: jnp.ndarray,
    num_metric_samples: int,
) -> tuple[float, float, Any]:
    metric_n = min(
        int(num_metric_samples),
        int(true_posterior_samples.shape[0]),
        int(posterior_samples.shape[0]),
    )
    if metric_n < 2:
        return float("nan"), float("nan"), key

    key, subkey = random.split(key)
    idx_npe = random.permutation(subkey, posterior_samples.shape[0])[:metric_n]
    key, subkey = random.split(key)
    idx_true = random.permutation(subkey, true_posterior_samples.shape[0])[:metric_n]
    ps_thin = posterior_samples[idx_npe]
    ts_thin = true_posterior_samples[idx_true]
    kl = float(kullback_leibler(ts_thin, ps_thin))
    lengthscale = median_heuristic(jnp.vstack([ts_thin, ps_thin]))
    mmd = float(unbiased_mmd(ts_thin, ps_thin, lengthscale))
    return kl, mmd, key


def maybe_write_plots(
    *,
    out: Path,
    true_posterior_samples: jnp.ndarray,
    posterior_samples: jnp.ndarray,
    losses: dict[str, list[float]],
) -> None:
    import matplotlib.pyplot as plt

    plt.plot(losses["train"], label="train")
    plt.plot(losses["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.legend()
    plt.savefig(out / "losses.pdf")
    plt.clf()

    for idx, name in enumerate(PARAM_NAMES):
        plt.hist(true_posterior_samples[:, idx], bins=50, label=name)
        plt.axvline(TRUE_PARAMS[idx], color="red")
        plt.legend()
        plt.savefig(out / f"true_samples_{name}.pdf")
        plt.clf()

        _, bins, _ = plt.hist(
            posterior_samples[:, idx], bins=50, alpha=0.8, label="Gaussian NPE"
        )
        plt.hist(true_posterior_samples[:, idx], bins=bins, alpha=0.8, label="NUTS")
        plt.axvline(TRUE_PARAMS[idx], color="black")
        plt.legend()
        plt.savefig(out / f"posterior_samples_{name}.pdf")
        plt.clf()
    plt.close()


def run_gnk_hexadeciles_gaussian(
    *,
    seed: int,
    n_obs: int,
    n_sims: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    fail_on_collision: bool = True,
    dry_run: bool = False,
    nuts_seed: int = 1,
    num_posterior_samples: int = 10_000,
    num_nuts_samples: int = 10_000,
    num_nuts_warmup: int = 10_000,
    num_coverage_samples: int = 100,
    num_metric_samples: int = 2000,
    hidden_dims: tuple[int, ...] = (128, 128),
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    max_epochs: int = 2000,
    patience: int = 200,
    val_frac: float = 0.1,
    prior_batch_size: int | None = None,
    smoke_reference_samples: bool = False,
    save_plots: bool = False,
) -> tuple[float, float] | dict[str, Any]:
    output_root = resolve_output_root(output_root)
    out = output_dir(output_root, n_obs, n_sims, seed)
    if prior_batch_size is None:
        prior_batch_size = min(1000, n_sims)

    train_config = TrainConfig(
        lr=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        val_frac=val_frac,
    )
    config = build_config(
        output_root=output_root,
        n_obs=n_obs,
        n_sims=n_sims,
        seed=seed,
        nuts_seed=nuts_seed,
        num_posterior_samples=num_posterior_samples,
        num_nuts_samples=num_nuts_samples,
        num_nuts_warmup=num_nuts_warmup,
        num_coverage_samples=num_coverage_samples,
        num_metric_samples=num_metric_samples,
        train_config=train_config,
        hidden_dims=hidden_dims,
        batch_size=batch_size,
        prior_batch_size=prior_batch_size,
        smoke_reference_samples=smoke_reference_samples,
        save_plots=save_plots,
    )
    if dry_run:
        print_dry_run(config)
        return config

    if not fail_on_collision:
        raise ValueError("fail_on_collision=False is unsupported for this cache-safe runner")
    if fail_on_collision:
        fail_if_collision(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)

    start = time.time()
    x_obs, key = observed_hexadeciles(seed, n_obs)
    true_posterior_samples, reference_source = get_reference_posterior(
        output_root=output_root,
        seed=seed,
        nuts_seed=nuts_seed,
        x_obs=x_obs,
        n_obs=n_obs,
        num_samples=num_nuts_samples,
        num_warmup=num_nuts_warmup,
        smoke_reference_samples=smoke_reference_samples,
        key=key,
    )

    key, subkey = random.split(key)
    tol = 1e-6
    thetas_bounded = dist.Uniform(tol, 10.0 - tol).sample(subkey, (n_sims, D_THETA))
    thetas_unbounded = logit(thetas_bounded / 10.0)
    A_sim, B_sim, g_sim, k_sim = thetas_bounded.T

    key, subkey = random.split(key)
    x_sims = get_summaries_batches(
        subkey,
        A_sim,
        B_sim,
        g_sim,
        k_sim,
        n_obs,
        n_sims,
        batch_size=prior_batch_size,
        sum_fn=ss_hexadeciles,
    )

    thetas_mean = thetas_unbounded.mean(axis=0)
    thetas_std = thetas_unbounded.std(axis=0)
    thetas = (thetas_unbounded - thetas_mean) / thetas_std

    sim_summ_data = x_sims.T
    sim_summ_data_mean = sim_summ_data.mean(axis=0)
    sim_summ_data_std = sim_summ_data.std(axis=0)
    sim_summ_data = (sim_summ_data - sim_summ_data_mean) / sim_summ_data_std
    x_obs_std = (x_obs - sim_summ_data_mean) / sim_summ_data_std

    key, subkey = random.split(key)
    model = ConditionalGaussianNPE(
        d_summary=D_SUMMARY,
        d_theta=D_THETA,
        hidden_dims=hidden_dims,
        key=subkey,
    )
    key, subkey = random.split(key)
    model, losses = fit(model, thetas, sim_summ_data, key=subkey, config=train_config)
    write_losses_csv(out / "losses.csv", losses)

    mu_hat, L_hat = model(x_obs_std)
    cov_hat = L_hat @ L_hat.T

    key, subkey = random.split(key)
    posterior_std = sample(model, x_obs_std, num_posterior_samples, key=subkey)
    posterior_unbounded = posterior_std * thetas_std + thetas_mean
    posterior_samples = expit(posterior_unbounded) * 10.0

    kl, mmd, key = metric_values(
        key=key,
        true_posterior_samples=true_posterior_samples,
        posterior_samples=posterior_samples,
        num_metric_samples=num_metric_samples,
    )

    coverage_levels = [0.8, 0.9, 0.95]
    coverage_counts = np.zeros((D_THETA, len(coverage_levels)))
    all_biases = []
    for _ in range(num_coverage_samples):
        key, subkey = random.split(key)
        x_obs_cov = get_summaries_batches(
            subkey,
            jnp.array([TRUE_PARAMS[0]]),
            jnp.array([TRUE_PARAMS[1]]),
            jnp.array([TRUE_PARAMS[2]]),
            jnp.array([TRUE_PARAMS[3]]),
            n_obs=n_obs,
            n_sims=1,
            batch_size=1,
            sum_fn=ss_hexadeciles,
        )
        x_obs_cov = jnp.squeeze(x_obs_cov)
        x_obs_cov = (x_obs_cov - sim_summ_data_mean) / sim_summ_data_std

        key, subkey = random.split(key)
        cov_samples_std = sample(model, x_obs_cov, num_posterior_samples, key=subkey)
        cov_samples = expit(cov_samples_std * thetas_std + thetas_mean) * 10.0
        all_biases.append(jnp.mean(cov_samples, axis=0) - TRUE_PARAMS)

        for j in range(D_THETA):
            for ci, cl in enumerate(coverage_levels):
                lo, hi = hpdi(cov_samples[:, j], cl)
                if lo < TRUE_PARAMS[j] < hi:
                    coverage_counts[j, ci] += 1

    estimated_coverage = coverage_counts / max(1, num_coverage_samples)
    biases = (
        jnp.stack(all_biases).ravel()
        if all_biases
        else jnp.array([], dtype=posterior_samples.dtype)
    )

    np.save(out / "x_obs.npy", np.asarray(x_obs))
    np.savez(
        out / "standardization.npz",
        theta_mean=np.asarray(thetas_mean),
        theta_std=np.asarray(thetas_std),
        summary_mean=np.asarray(sim_summ_data_mean),
        summary_std=np.asarray(sim_summ_data_std),
        x_obs_std=np.asarray(x_obs_std),
    )
    np.savez(
        out / "gaussian_npe_native_u_posterior.npz",
        mu_u=np.asarray(mu_hat),
        L_u=np.asarray(L_hat),
        cov_u=np.asarray(cov_hat),
    )
    with (out / "posterior_samples.pkl").open("wb") as f:
        pkl.dump(posterior_samples, f)
    with (out / "true_posterior_samples.pkl").open("wb") as f:
        pkl.dump(true_posterior_samples, f)
    with (out / "kl.txt").open("w") as f:
        f.write(str(kl))
    with (out / "mmd.txt").open("w") as f:
        f.write(str(mmd))
    np.save(out / "estimated_coverage.npy", estimated_coverage)
    np.save(out / "biases.npy", np.asarray(biases))

    if save_plots:
        maybe_write_plots(
            out=out,
            true_posterior_samples=true_posterior_samples,
            posterior_samples=posterior_samples,
            losses=losses,
        )

    metadata = {
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - start,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "jax_devices": [str(device) for device in jnp.ones(1).devices()],
        "reference_source": reference_source,
        "posterior_samples_shape": list(np.asarray(posterior_samples).shape),
        "true_posterior_samples_shape": list(np.asarray(true_posterior_samples).shape),
        "kl": kl,
        "mmd": mmd,
        "output_dir": rel(out),
    }
    write_json(out / "metadata.json", metadata)
    print(f"completed {rel(out)}")
    print(f"KL: {kl}, MMD: {mmd}")
    return kl, mmd


def parse_hidden_dims(spec: str) -> tuple[int, ...]:
    return tuple(int(part) for part in spec.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run g-and-k hexadeciles with conditional Gaussian NPE.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_obs", "--n-obs", dest="n_obs", type=int, default=1000)
    parser.add_argument("--n_sims", "--n-sims", dest="n_sims", type=int, default=31622)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-collision", action="store_true", default=True)
    parser.add_argument("--nuts-seed", type=int, default=1)
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--num-nuts-samples", type=int, default=10_000)
    parser.add_argument("--num-nuts-warmup", type=int, default=10_000)
    parser.add_argument("--num-coverage-samples", type=int, default=100)
    parser.add_argument("--num-metric-samples", type=int, default=2000)
    parser.add_argument("--hidden-dims", type=str, default="128,128")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--prior-batch-size", type=int, default=None)
    parser.add_argument(
        "--smoke-reference-samples",
        action="store_true",
        help="Use synthetic reference samples. Only allowed outside repo res/.",
    )
    parser.add_argument("--save-plots", action="store_true")
    return parser


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    args = build_parser().parse_args()
    run_gnk_hexadeciles_gaussian(
        seed=args.seed,
        n_obs=args.n_obs,
        n_sims=args.n_sims,
        output_root=args.output_root,
        fail_on_collision=args.fail_on_collision,
        dry_run=args.dry_run,
        nuts_seed=args.nuts_seed,
        num_posterior_samples=args.num_posterior_samples,
        num_nuts_samples=args.num_nuts_samples,
        num_nuts_warmup=args.num_nuts_warmup,
        num_coverage_samples=args.num_coverage_samples,
        num_metric_samples=args.num_metric_samples,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_frac=args.val_frac,
        prior_batch_size=args.prior_batch_size,
        smoke_reference_samples=args.smoke_reference_samples,
        save_plots=args.save_plots,
    )
