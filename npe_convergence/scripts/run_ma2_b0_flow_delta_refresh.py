"""Run MA(2)-b0 flow-NPE delta0 refresh cell.

Each invocation trains one normalising-flow NPE for a single
``(seed, n_obs, n_sims)`` cell, then exports posterior samples and metrics at
fixed summaries ``[delta_0, 0, 0]``. This is the flow analogue of
``run_ma2_b0_gaussian_compat_refresh.py`` and is intended for exact
``delta0=1.0`` follow-up runs in a fresh output namespace.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro  # type: ignore
from flowjax.bijections import RationalQuadraticSpline  # type: ignore
from flowjax.distributions import Normal  # type: ignore
from flowjax.flows import coupling_flow  # type: ignore
from flowjax.train.data_fit import fit_to_data  # type: ignore

from npe_convergence.examples.ma2 import get_summaries_batches
from npe_convergence.scripts.run_ma2_b0_gaussian_compat_refresh import (
    D_SUMMARY,
    D_THETA,
    PARAM_NAMES,
    REFERENCE_SMC_CONVENTION,
    bounded_from_unbounded,
    delta_label,
    draw_prior,
    metric_values,
    parse_delta0_values,
    reference_samples_for_delta,
    rel,
    resolve_output_root,
    standardise,
    write_json,
    write_validation_curve,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "res" / "ma2_b0_delta1_refresh"
DEFAULT_DELTA0_VALUES = (1.0,)


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


def output_dir(output_root: Path, n_obs: int, n_sims: int, seed: int) -> Path:
    return (
        output_root
        / f"flow_npe_compat_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}"
    )


def delta_dir(out: Path, delta0: float) -> Path:
    return out / f"delta_0_{delta_label(delta0)}"


def expected_paths(
    output_root: Path,
    *,
    n_obs: int,
    n_sims: int,
    seed: int,
    delta0_values: tuple[float, ...],
    save_plots: bool,
) -> dict[str, str | dict[str, str]]:
    out = output_dir(output_root, n_obs, n_sims, seed)
    paths: dict[str, str | dict[str, str]] = {
        "config": rel(out / "config.json"),
        "validation_curve": rel(out / "validation_curve.csv"),
        "training_metadata": rel(out / "training_metadata.json"),
        "standardisation": rel(out / "standardisation.npz"),
    }
    for delta0 in delta0_values:
        ddir = delta_dir(out, delta0)
        delta_paths = {
            "reference_samples": rel(ddir / "reference_samples.npz"),
            "flow_npe_posterior_samples": rel(
                ddir / "flow_npe_posterior_samples.npz"
            ),
            "metrics": rel(ddir / "metrics.json"),
            "kl": rel(ddir / "kl.txt"),
            "mmd": rel(ddir / "mmd.txt"),
        }
        if save_plots:
            delta_paths.update(
                {
                    "t1_posterior_overlay": rel(ddir / "t1_posterior_overlay.pdf"),
                    "t2_posterior_overlay": rel(ddir / "t2_posterior_overlay.pdf"),
                }
            )
        paths[f"delta_0_{delta_label(delta0)}"] = delta_paths
    return paths


def fail_if_selected_output_exists(out: Path) -> None:
    if out.exists():
        raise FileExistsError(
            f"Refusing to run because output directory already exists: {out}"
        )


def write_delta_plots(
    *,
    ddir: Path,
    reference_samples: jnp.ndarray,
    posterior_samples: jnp.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    for idx, name in enumerate(PARAM_NAMES):
        _, bins, _ = plt.hist(
            np.asarray(reference_samples[:, idx]),
            bins=50,
            alpha=0.6,
            label="Reference",
        )
        plt.hist(
            np.asarray(posterior_samples[:, idx]),
            bins=bins,
            alpha=0.6,
            label="Flow NPE",
        )
        plt.xlabel(name)
        plt.legend()
        plt.savefig(ddir / f"{name}_posterior_overlay.pdf")
        plt.clf()
    plt.close()


def build_config(
    *,
    output_root: Path,
    n_obs: int,
    n_sims: int,
    seed: int,
    delta0_values: tuple[float, ...],
    num_reference_samples: int,
    num_posterior_samples: int,
    num_metric_samples: int,
    prior_batch_size: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    max_patience: int,
    nn_depth: int,
    spline_knots: int,
    spline_interval: float,
    smc_settings: dict[str, Any],
    smoke_reference_samples: bool,
    save_plots: bool,
) -> dict[str, Any]:
    return {
        "task": "ma2-b0-flow-npe-delta-refresh",
        "created_at_utc": utc_now(),
        "git_commit": run_git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": git_dirty(),
        "branch": run_git(["branch", "--show-current"]),
        "method": "Flow-NPE delta0 export",
        "n_obs": n_obs,
        "n_sims": n_sims,
        "seed": seed,
        "delta0_values": [delta_label(value) for value in delta0_values],
        "fixed_summary_template": "[delta_0, 0, 0]",
        "d_summary": D_SUMMARY,
        "d_theta": D_THETA,
        "param_names": list(PARAM_NAMES),
        "output_root": rel(output_root),
        "output_dir": rel(output_dir(output_root, n_obs, n_sims, seed)),
        "num_reference_samples": num_reference_samples,
        "num_posterior_samples": num_posterior_samples,
        "num_metric_samples": num_metric_samples,
        "prior_batch_size": prior_batch_size,
        "train_config": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "max_patience": max_patience,
            "nn_depth": nn_depth,
            "spline_knots": spline_knots,
            "spline_interval": spline_interval,
        },
        "reference_smc_convention": REFERENCE_SMC_CONVENTION,
        "smc_settings": smc_settings,
        "smoke_reference_samples": smoke_reference_samples,
        "smoke_reference_samples_note": (
            "Synthetic reference samples are for non-scientific smoke tests only."
            if smoke_reference_samples
            else None
        ),
        "save_plots": save_plots,
        "expected_outputs": expected_paths(
            output_root,
            n_obs=n_obs,
            n_sims=n_sims,
            seed=seed,
            delta0_values=delta0_values,
            save_plots=save_plots,
        ),
    }


def print_dry_run(config: dict[str, Any]) -> None:
    print("MA2-b0 flow-NPE delta refresh dry-run")
    print(f"output_dir: {config['output_dir']}")
    print(f"n_obs: {config['n_obs']}")
    print(f"n_sims: {config['n_sims']}")
    print(f"seed: {config['seed']}")
    print(f"delta0_values: {', '.join(config['delta0_values'])}")
    print("expected_outputs:")
    for name, path in config["expected_outputs"].items():
        print(f"  {name}: {path}")


def run_ma2_b0_flow_delta_refresh(
    *,
    seed: int,
    n_obs: int,
    n_sims: int,
    delta0_values: tuple[float, ...] = DEFAULT_DELTA0_VALUES,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    fail_if_output_exists: bool = False,
    dry_run: bool = False,
    num_reference_samples: int = 10_000,
    num_posterior_samples: int = 10_000,
    num_metric_samples: int = 2000,
    learning_rate: float = 5e-4,
    batch_size: int = 256,
    max_epochs: int = 2000,
    max_patience: int = 20,
    nn_depth: int = 2,
    spline_knots: int = 10,
    spline_interval: float = 5.0,
    prior_batch_size: int | None = None,
    smc_step_size: float = 5e-3,
    smc_inverse_mass: float = 0.1,
    smc_integration_steps: int = 100,
    smc_num_mcmc_steps: int = 5,
    smc_ess_threshold: float = 0.75,
    smc_max_steps: int = 10_000,
    smoke_reference_samples: bool = False,
    save_plots: bool = True,
) -> dict[str, Any]:
    if n_sims < 2:
        raise ValueError("--n-sims must be at least 2")
    if num_reference_samples < 2 or num_posterior_samples < 2:
        raise ValueError("sample counts must be at least 2")
    if prior_batch_size is None:
        prior_batch_size = min(1000, n_sims)

    output_root = resolve_output_root(output_root)
    out = output_dir(output_root, n_obs, n_sims, seed)
    smc_settings = {
        "step_size": smc_step_size,
        "inverse_mass": smc_inverse_mass,
        "integration_steps": smc_integration_steps,
        "num_mcmc_steps": smc_num_mcmc_steps,
        "ess_threshold": smc_ess_threshold,
        "max_steps": smc_max_steps,
    }
    config = build_config(
        output_root=output_root,
        n_obs=n_obs,
        n_sims=n_sims,
        seed=seed,
        delta0_values=delta0_values,
        num_reference_samples=num_reference_samples,
        num_posterior_samples=num_posterior_samples,
        num_metric_samples=num_metric_samples,
        prior_batch_size=prior_batch_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        max_patience=max_patience,
        nn_depth=nn_depth,
        spline_knots=spline_knots,
        spline_interval=spline_interval,
        smc_settings=smc_settings,
        smoke_reference_samples=smoke_reference_samples,
        save_plots=save_plots,
    )
    if dry_run:
        print_dry_run(config)
        return config

    if fail_if_output_exists:
        fail_if_selected_output_exists(out)
    elif out.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {out}. "
            "Use a fresh output root."
        )

    if smoke_reference_samples and output_root.resolve().is_relative_to(
        (REPO_ROOT / "res").resolve()
    ):
        raise ValueError(
            "--smoke-reference-samples is only allowed outside repo res/ tree"
        )

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)
    start = time.time()

    key = random.key(seed)
    key, t1_bounded, t2_bounded, theta_unbounded, _theta_bounded = draw_prior(
        key, n_sims
    )
    batch_size_sims = min(prior_batch_size, n_sims)
    print("Simulating MA2 prior predictive summaries...")
    key, subkey = random.split(key)
    sim_summ_data = get_summaries_batches(
        subkey,
        t1_bounded,
        t2_bounded,
        n_obs,
        n_sims,
        batch_size_sims,
    )

    thetas, thetas_mean, thetas_std = standardise(theta_unbounded)
    sim_summ_data, sim_summ_data_mean, sim_summ_data_std = standardise(sim_summ_data)
    initial_reference_particles = thetas

    np.savez(
        out / "standardisation.npz",
        theta_unbounded_mean=np.asarray(thetas_mean),
        theta_unbounded_std=np.asarray(thetas_std),
        summary_mean=np.asarray(sim_summ_data_mean),
        summary_std=np.asarray(sim_summ_data_std),
        param_names=np.asarray(PARAM_NAMES),
    )

    key, subkey = random.split(key)
    flow = coupling_flow(
        key=subkey,
        base_dist=Normal(jnp.zeros(D_THETA)),
        transformer=RationalQuadraticSpline(
            knots=spline_knots,
            interval=spline_interval,
        ),
        cond_dim=D_SUMMARY,
        nn_depth=nn_depth,
    )
    print("Training flow NPE...")
    key, subkey = random.split(key)
    flow, losses = fit_to_data(
        key=subkey,
        dist=flow,
        x=thetas,
        condition=sim_summ_data,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        max_patience=max_patience,
        batch_size=batch_size,
    )
    write_validation_curve(out / "validation_curve.csv", losses)

    delta_metrics: dict[str, Any] = {}
    for delta0 in delta0_values:
        label = delta_label(delta0)
        ddir = delta_dir(out, delta0)
        ddir.mkdir(parents=True, exist_ok=False)
        fixed_summary = jnp.array([delta0, 0.0, 0.0])
        fixed_summary_standardised = (
            fixed_summary - sim_summ_data_mean
        ) / sim_summ_data_std

        key, subkey = random.split(key)
        posterior_standardised = flow.sample(
            subkey,
            sample_shape=(num_posterior_samples,),
            condition=fixed_summary_standardised,
        )
        posterior_unbounded = posterior_standardised * thetas_std + thetas_mean
        posterior_samples = bounded_from_unbounded(posterior_unbounded)
        posterior_samples = jnp.squeeze(posterior_samples)

        print(f"Running reference export for delta_0={label}...")
        (
            reference_samples,
            reference_particles,
            reference_metadata,
            key,
        ) = reference_samples_for_delta(
            key=key,
            fixed_summary=fixed_summary,
            initial_particles=initial_reference_particles,
            n_obs=n_obs,
            num_reference_samples=num_reference_samples,
            smc_step_size=smc_step_size,
            smc_inverse_mass=smc_inverse_mass,
            smc_integration_steps=smc_integration_steps,
            smc_num_mcmc_steps=smc_num_mcmc_steps,
            smc_ess_threshold=smc_ess_threshold,
            smc_max_steps=smc_max_steps,
            smoke_reference_samples=smoke_reference_samples,
        )

        kl, mmd, metric_n, key = metric_values(
            key=key,
            reference_samples=reference_samples,
            posterior_samples=posterior_samples,
            num_metric_samples=num_metric_samples,
        )

        np.savez(
            ddir / "reference_samples.npz",
            samples=np.asarray(reference_samples),
            particles_standardised=np.asarray(reference_particles),
            fixed_summary=np.asarray(fixed_summary),
            param_names=np.asarray(PARAM_NAMES),
            n_obs=n_obs,
            n_sims=n_sims,
            seed=seed,
            delta0=delta0,
            sample_count=int(reference_samples.shape[0]),
            reference_smc_convention=REFERENCE_SMC_CONVENTION,
        )
        np.savez(
            ddir / "flow_npe_posterior_samples.npz",
            samples=np.asarray(posterior_samples),
            theta_unbounded=np.asarray(posterior_unbounded),
            theta_standardised=np.asarray(posterior_standardised),
            fixed_summary=np.asarray(fixed_summary),
            fixed_summary_standardised=np.asarray(fixed_summary_standardised),
            param_names=np.asarray(PARAM_NAMES),
            n_obs=n_obs,
            n_sims=n_sims,
            seed=seed,
            delta0=delta0,
            sample_count=int(posterior_samples.shape[0]),
        )

        metrics = {
            "delta0": label,
            "fixed_summary": np.asarray(fixed_summary).tolist(),
            "kl_reference_to_flow_npe": kl,
            "mmd_reference_vs_flow_npe": mmd,
            "metric_sample_count": metric_n,
            "reference_sample_count": int(reference_samples.shape[0]),
            "flow_npe_sample_count": int(posterior_samples.shape[0]),
            "reference_samples_shape": list(np.asarray(reference_samples).shape),
            "flow_npe_samples_shape": list(np.asarray(posterior_samples).shape),
            "reference_smc_convention": REFERENCE_SMC_CONVENTION,
            **reference_metadata,
            "output_dir": rel(ddir),
            "files": {
                "reference_samples": rel(ddir / "reference_samples.npz"),
                "flow_npe_posterior_samples": rel(
                    ddir / "flow_npe_posterior_samples.npz"
                ),
                "metrics": rel(ddir / "metrics.json"),
                "kl": rel(ddir / "kl.txt"),
                "mmd": rel(ddir / "mmd.txt"),
            },
        }
        write_json(ddir / "metrics.json", metrics)
        with (ddir / "kl.txt").open("w") as f:
            f.write(str(kl))
        with (ddir / "mmd.txt").open("w") as f:
            f.write(str(mmd))
        if save_plots:
            write_delta_plots(
                ddir=ddir,
                reference_samples=reference_samples,
                posterior_samples=posterior_samples,
            )
        delta_metrics[label] = metrics

    training_metadata = {
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.time() - start,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jnp.ones(1).devices()],
        "train_epochs_completed": len(losses.get("train", [])),
        "best_validation_loss": (
            min(losses.get("val", [])) if losses.get("val", []) else None
        ),
        "final_train_loss": (
            losses.get("train", [None])[-1] if losses.get("train", []) else None
        ),
        "final_validation_loss": (
            losses.get("val", [None])[-1] if losses.get("val", []) else None
        ),
        "n_obs": n_obs,
        "n_sims": n_sims,
        "seed": seed,
        "delta0_values": [delta_label(value) for value in delta0_values],
        "num_reference_samples": num_reference_samples,
        "num_posterior_samples": num_posterior_samples,
        "num_metric_samples": num_metric_samples,
        "train_config": config["train_config"],
        "smc_settings": smc_settings,
        "reference_smc_convention": REFERENCE_SMC_CONVENTION,
        "output_root": rel(output_root),
        "output_dir": rel(out),
        "delta_metrics": delta_metrics,
    }
    write_json(out / "training_metadata.json", training_metadata)
    print(f"completed {rel(out)}")
    return training_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-obs", "--n_obs", dest="n_obs", type=int, required=True)
    parser.add_argument("--n-sims", "--n_sims", dest="n_sims", type=int, required=True)
    parser.add_argument(
        "--delta0-values",
        default=",".join(str(value) for value in DEFAULT_DELTA0_VALUES),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fail-if-output-exists", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-reference-samples", type=int, default=10_000)
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--num-metric-samples", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--max-patience", type=int, default=20)
    parser.add_argument("--nn-depth", type=int, default=2)
    parser.add_argument("--spline-knots", type=int, default=10)
    parser.add_argument("--spline-interval", type=float, default=5.0)
    parser.add_argument("--prior-batch-size", type=int, default=None)
    parser.add_argument("--smc-step-size", type=float, default=5e-3)
    parser.add_argument("--smc-inverse-mass", type=float, default=0.1)
    parser.add_argument("--smc-integration-steps", type=int, default=100)
    parser.add_argument("--smc-num-mcmc-steps", type=int, default=5)
    parser.add_argument("--smc-ess-threshold", type=float, default=0.75)
    parser.add_argument("--smc-max-steps", type=int, default=10_000)
    parser.add_argument(
        "--smoke-reference-samples",
        action="store_true",
        help="Use synthetic reference samples. Only allowed outside repo res/.",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-delta t1/t2 overlay PDFs.",
    )
    return parser


def main() -> None:
    numpyro.set_host_device_count(4)
    args = build_parser().parse_args()
    run_ma2_b0_flow_delta_refresh(
        seed=args.seed,
        n_obs=args.n_obs,
        n_sims=args.n_sims,
        delta0_values=parse_delta0_values(args.delta0_values),
        output_root=args.output_root,
        fail_if_output_exists=args.fail_if_output_exists,
        dry_run=args.dry_run,
        num_reference_samples=args.num_reference_samples,
        num_posterior_samples=args.num_posterior_samples,
        num_metric_samples=args.num_metric_samples,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        max_patience=args.max_patience,
        nn_depth=args.nn_depth,
        spline_knots=args.spline_knots,
        spline_interval=args.spline_interval,
        prior_batch_size=args.prior_batch_size,
        smc_step_size=args.smc_step_size,
        smc_inverse_mass=args.smc_inverse_mass,
        smc_integration_steps=args.smc_integration_steps,
        smc_num_mcmc_steps=args.smc_num_mcmc_steps,
        smc_ess_threshold=args.smc_ess_threshold,
        smc_max_steps=args.smc_max_steps,
        smoke_reference_samples=args.smoke_reference_samples,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()
