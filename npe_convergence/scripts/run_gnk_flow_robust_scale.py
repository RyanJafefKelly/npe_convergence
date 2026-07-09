#!/usr/bin/env python
"""Run GNK flow-NPE with robust summary standardisation.

This mirrors ``run_gnk.py`` for the broad Uniform(0, 10)^4 prior, but uses
the same robust per-coordinate summary transform as
``run_gnk_gaussian_robust_scale.py`` and scores against the canonical v3 NUTS
reference instead of regenerating a reference posterior.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle as pkl
import sys
import time
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
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import Normal  # type: ignore
from flowjax.flows import coupling_flow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore
from jax.scipy.special import expit, logit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import gnk, get_summaries_batches, ss_octile
from npe_convergence.metrics import kullback_leibler, unbiased_mmd
from npe_convergence.scripts.run_gnk_gaussian_robust_scale import (
    N_METRIC,
    PARAM_NAMES,
    TRUE_THETA,
    TRUE_THETA_FLOAT32,
    deduplicate,
    deterministic_subsample,
    environment_record,
    load_canonical_reference,
    median_heuristic_fast,
    plot_loss,
    robust_standardise,
    rng_for,
    stable_int,
    utc_now,
    write_json,
)


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


def reconstruct_x_obs_float32(seed: int, n_obs: int) -> jnp.ndarray:
    key = random.key(seed)
    z = random.normal(key, shape=(n_obs,), dtype=jnp.float32)
    x_raw = gnk(z, *TRUE_THETA_FLOAT32)
    return jnp.squeeze(ss_octile(jnp.atleast_2d(x_raw))).astype(jnp.float32)


def x_obs_sha256(x_obs_float32: jnp.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(x_obs_float32).tobytes()).hexdigest()


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
            label="robust flow-NPE",
            color="#008837",
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
            color="#008837",
            linewidth=1.5,
            label="robust flow-NPE",
        )
        axes[i].axvline(float(TRUE_THETA[i]), color="black", linestyle=":")
        axes[i].set_title(name)
        axes[i].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_samples_overlay.pdf")
    plt.close(fig)


def output_dir_for(args: argparse.Namespace) -> Path:
    return args.output_root / (
        f"flow_npe_n_obs_{args.n_obs}_n_sims_{args.n_sims}_seed_{args.seed}_"
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
        default=REPO_ROOT / "res" / "gnk_flow_robust_scale",
    )
    parser.add_argument(
        "--v3-root", type=Path, default=REPO_ROOT / "res" / "gnk_v3_refs"
    )
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--sim-batch-size", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--max-patience", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    run_start = time.perf_counter()
    args = build_parser().parse_args()
    convention = "flow"
    created_at = utc_now()
    env = environment_record()

    output_dir = output_dir_for(args)
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "GNK robust-scale flow-NPE: "
        f"n_obs={args.n_obs}, seed={args.seed}, n_sims={args.n_sims}, "
        f"transform={args.transform}, inner_scale_mode={args.inner_scale_mode}",
        flush=True,
    )

    ref_path, ref_fingerprint, ref_samples = load_canonical_reference(
        args.v3_root, args.n_obs, args.seed, convention
    )
    x_obs_f32 = reconstruct_x_obs_float32(args.seed, args.n_obs)
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
    theta_key = rng_for(args.seed, "flow_robust_scale", args.n_obs, args.n_sims, "theta")
    thetas_bounded = dist.Uniform(0.0 + tol, 10.0 - tol).sample(
        theta_key, (args.n_sims, 4)
    )
    thetas_unbounded = logit(thetas_bounded / 10.0)

    print(">>> Simulating prior predictive summaries...", flush=True)
    sim_key = rng_for(args.seed, "flow_robust_scale", args.n_obs, args.n_sims, "summaries")
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

    print(">>> Training conditional flow-NPE...", flush=True)
    theta_dims = 4
    summary_dims = 7
    flow_key = rng_for(args.seed, "flow_robust_scale", args.n_obs, args.n_sims, "flow")
    flow = coupling_flow(
        key=flow_key,
        base_dist=Normal(jnp.zeros(theta_dims)),
        transformer=RationalQuadraticSpline(knots=10, interval=5),
        cond_dim=summary_dims,
        nn_depth=2,
    )
    fit_key = rng_for(args.seed, "flow_robust_scale", args.n_obs, args.n_sims, "fit")
    train_start = time.perf_counter()
    flow, losses = fit_to_data(
        key=fit_key,
        dist=flow,
        x=jnp.asarray(thetas),
        condition=jnp.asarray(z_summaries),
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        max_patience=args.max_patience,
        batch_size=args.batch_size,
    )
    training_seconds = time.perf_counter() - train_start
    print(f">>> Training done in {training_seconds:.1f}s", flush=True)
    plot_loss(losses, output_dir / "losses.pdf")
    write_json(output_dir / "losses.json", losses)

    print(">>> Sampling posterior...", flush=True)
    sampling_start = time.perf_counter()
    posterior_key = rng_for(
        args.seed, "flow_robust_scale", args.n_obs, args.n_sims, "posterior"
    )
    posterior_std = flow.sample(
        posterior_key,
        sample_shape=(args.num_posterior_samples,),
        condition=jnp.asarray(z_obs),
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
        stable_int("flow_robust_scale", args.n_obs, args.seed, args.n_sims, "metric_npe"),
    )
    ref_thin = deterministic_subsample(
        ref_unique,
        N_METRIC,
        stable_int("flow_robust_scale", args.n_obs, args.seed, args.n_sims, "metric_ref"),
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
        "method": "flow_npe_robust_scale",
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
        "train_config": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "max_patience": args.max_patience,
        },
        "fingerprint": fingerprint,
        "args": {
            **vars(args),
            "output_root": str(args.output_root),
            "v3_root": str(args.v3_root),
        },
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")
    print(f"  wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
