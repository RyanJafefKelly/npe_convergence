"""Canonical x64 NUTS reference run for one GNK (n_obs, seed, convention) cell.

Each invocation writes ONE fingerprinted dict to
``res/gnk_v3_refs/nuts_n_obs_{n_obs}_seed_{seed}_conv_{convention}.pkl``.

x64 must be enabled before any JAX import. The recommended invocation is via
``JAX_ENABLE_X64=1`` in the environment. The script also calls
``jax.config.update("jax_enable_x64", True)`` and asserts that x64 is on.

NUTS settings: dense mass matrix, target_accept_prob=0.9, 5 chains,
10,000 warmup, 2,000 retained per chain (no thinning), parallel chains.
Sampler seed derived from (n_obs, seed) via stable_int rather than hardcoded
to 1.

Convention controls the reconstruction of x_obs:
    flow:     key = random.key(seed); z = random.normal(key, ...)
    gaussian: key = random.key(seed); _, sk = random.split(key);
              z = random.normal(sk, ...)
The reconstructed octile summary is kept in float32 (matching the NPE
training-time data convention) and cast to float64 for the NUTS likelihood.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle as pkl
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# x64 must be enabled before any JAX-dependent import. Host device count must
# also be set before NumPyro touches devices, otherwise parallel chains fall
# back to sequential execution.
os.environ.setdefault("JAX_ENABLE_X64", "1")
_DEFAULT_NUM_CHAINS = 5
os.environ.setdefault(
    "XLA_FLAGS", f"--xla_force_host_platform_device_count={_DEFAULT_NUM_CHAINS}"
)

import jax

jax.config.update("jax_enable_x64", True)
assert jax.config.jax_enable_x64, "x64 must be on for canonical references"

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax.scipy.special import logit
from jax.scipy.stats import norm
from numpyro.diagnostics import summary as numpyro_summary  # type: ignore
from numpyro.infer import MCMC, NUTS  # type: ignore

numpyro.set_host_device_count(_DEFAULT_NUM_CHAINS)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import (
    compute_covariance_matrix,
    gnk,
    ss_octile,
)

PARAM_NAMES = ("A", "B", "g", "k")
TRUE_THETA_FLOAT32 = jnp.asarray([3.0, 1.0, 2.0, 0.5], dtype=jnp.float32)
DENSITY_VERSION = "phi_over_Qprime_v1_postfix_af438f5"


# ---------------------------------------------------------------------------
# Reproducible RNG helpers (match the BSL script convention).
# ---------------------------------------------------------------------------


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# x_obs reconstruction.
# ---------------------------------------------------------------------------


def reconstruct_x_obs_float32(seed: int, n_obs: int, convention: str) -> jnp.ndarray:
    """Match the NPE training convention used by the existing paper-grid runs.

    flow:     key = random.key(seed); z = random.normal(key, (n_obs,), float32)
    gaussian: key = random.key(seed); _, sk = random.split(key);
              z = random.normal(sk, (n_obs,), float32)
    Returns the 7-vector octile summary in float32.
    """
    if convention == "flow":
        key = random.key(seed)
        z_key = key
    elif convention == "gaussian":
        key = random.key(seed)
        _, z_key = random.split(key)
    else:
        raise ValueError(f"unknown convention: {convention}")
    z = random.normal(z_key, shape=(n_obs,), dtype=jnp.float32)
    x_raw = gnk(z, *TRUE_THETA_FLOAT32)
    summary = jnp.squeeze(ss_octile(jnp.atleast_2d(x_raw)))
    return summary.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


def gnk_model_x64(obs: jnp.ndarray, n_obs: int) -> None:
    """GNK MVN model, x64 numerics. Mirrors gnk_model in npe_convergence.examples.gnk."""
    A = numpyro.sample("A", dist.Uniform(0.0, 10.0))
    B = numpyro.sample("B", dist.Uniform(0.0, 10.0))
    g = numpyro.sample("g", dist.Uniform(0.0, 10.0))
    k = numpyro.sample("k", dist.Uniform(0.0, 10.0))

    quantile_length = 1.0 / (len(obs) + 1)
    quantiles = jnp.linspace(quantile_length, 1.0 - quantile_length, len(obs))
    z = norm.ppf(quantiles)
    expected = gnk(z, A, B, g, k)

    cov = compute_covariance_matrix(A, B, g, k, quantiles, n_obs)
    cov = cov + 1e-6 * jnp.eye(len(obs))

    numpyro.sample(
        "obs",
        dist.MultivariateNormal(expected, cov),
        obs=obs,
    )


# ---------------------------------------------------------------------------
# NUTS run + diagnostics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NUTSResult:
    samples_grouped: np.ndarray  # (num_chains, samples_per_chain, 4)
    extra_fields: dict[str, np.ndarray]
    runtime_seconds: float


def _init_params_truth_centred(
    sampler_seed: int, num_chains: int, init_sd: float = 0.05
) -> dict[str, jnp.ndarray]:
    rng = random.key(sampler_seed)
    rng, *subkeys = random.split(rng, 5)

    def init_for(value: float, sk: jax.Array) -> jnp.ndarray:
        base = jnp.repeat(logit(jnp.array([value]) / 10.0), num_chains)
        noise = random.normal(sk, (num_chains,)) * init_sd
        return base + noise

    return {
        "A": init_for(3.0, subkeys[0]),
        "B": init_for(1.0, subkeys[1]),
        "g": init_for(2.0, subkeys[2]),
        "k": init_for(0.5, subkeys[3]),
    }


def run_nuts_canonical(
    *,
    x_obs_f64: jnp.ndarray,
    n_obs: int,
    sampler_seed: int,
    num_chains: int = 5,
    num_warmup: int = 10_000,
    num_samples: int = 2_000,
    target_accept_prob: float = 0.9,
) -> NUTSResult:
    """Run NUTS with dense mass, no thinning, chain-grouped output."""
    kernel = NUTS(
        gnk_model_x64,
        dense_mass=True,
        target_accept_prob=target_accept_prob,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel",
        thinning=1,
        progress_bar=False,
    )

    init_params = _init_params_truth_centred(sampler_seed, num_chains)
    extra_field_names = ("diverging", "accept_prob", "num_steps", "energy")

    sampler_key = random.key(sampler_seed)
    start = time.perf_counter()
    mcmc.run(
        rng_key=sampler_key,
        init_params=init_params,
        obs=x_obs_f64,
        n_obs=n_obs,
        extra_fields=extra_field_names,
    )
    runtime = time.perf_counter() - start

    samples_dict = mcmc.get_samples(group_by_chain=True)
    samples_grouped = np.stack(
        [np.asarray(samples_dict[name]) for name in PARAM_NAMES],
        axis=-1,
    )  # (num_chains, num_samples, 4)

    extras: dict[str, np.ndarray] = {}
    raw_extras = mcmc.get_extra_fields(group_by_chain=True)
    for name in extra_field_names:
        if name in raw_extras:
            extras[name] = np.asarray(raw_extras[name])

    return NUTSResult(
        samples_grouped=samples_grouped,
        extra_fields=extras,
        runtime_seconds=runtime,
    )


def diagnostics_from_samples(
    samples_grouped: np.ndarray, extras: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Per-parameter R-hat, ESS, MCSE, plus chain-level extras."""
    samples_dict = {
        name: jnp.asarray(samples_grouped[:, :, i]) for i, name in enumerate(PARAM_NAMES)
    }
    summary = numpyro_summary(samples_dict, group_by_chain=True)
    per_param: dict[str, dict[str, float]] = {}
    for name in PARAM_NAMES:
        s = summary[name]
        # numpyro_summary fields: mean, std, median, 5.0%, 95.0%, n_eff, r_hat.
        # Treat n_eff as the bulk ESS approximation. Compute MCSE per sd.
        std = float(np.asarray(s["std"]))
        n_eff = float(np.asarray(s["n_eff"]))
        mcse_mean = std / np.sqrt(n_eff) if n_eff > 0 else float("inf")
        mcse_median = 1.253 * mcse_mean
        per_param[name] = {
            "mean": float(np.asarray(s["mean"])),
            "std": std,
            "median": float(np.asarray(s["median"])),
            "q05": float(np.asarray(s["5.0%"])),
            "q95": float(np.asarray(s["95.0%"])),
            "n_eff": n_eff,
            "r_hat": float(np.asarray(s["r_hat"])),
            "mcse_mean_per_sd": (mcse_mean / std) if std > 0 else float("inf"),
            "mcse_median_per_sd": (mcse_median / std) if std > 0 else float("inf"),
        }

    out: dict[str, Any] = {"per_parameter": per_param}
    if "diverging" in extras:
        out["divergence_count"] = int(extras["diverging"].sum())
        out["divergences_per_chain"] = extras["diverging"].sum(axis=1).astype(int).tolist()
    if "accept_prob" in extras:
        out["mean_accept_prob_per_chain"] = extras["accept_prob"].mean(axis=1).tolist()
    if "num_steps" in extras:
        out["mean_num_steps_per_chain"] = extras["num_steps"].mean(axis=1).tolist()
        out["max_num_steps_per_chain"] = extras["num_steps"].max(axis=1).astype(int).tolist()
    return out


# ---------------------------------------------------------------------------
# Environment + fingerprint.
# ---------------------------------------------------------------------------


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
    try:
        numpyro_version = numpyro.__version__  # type: ignore[attr-defined]
    except AttributeError:
        numpyro_version = "unknown"
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_x64_enabled": bool(jax.config.jax_enable_x64),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "numpyro_version": numpyro_version,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": git_dirty(),
    }


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--convention",
        type=str,
        required=True,
        choices=("flow", "gaussian"),
        help=(
            "Data-generation convention for x_obs. 'flow' matches run_gnk.py "
            "(no split); 'gaussian' matches run_gnk_gaussian.py and BSL "
            "(one split)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("res/gnk_v3_refs"),
    )
    parser.add_argument("--num-chains", type=int, default=5)
    parser.add_argument("--num-warmup", type=int, default=10_000)
    parser.add_argument("--num-samples", type=int, default=2_000)
    parser.add_argument("--target-accept-prob", type=float, default=0.9)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file if present.",
    )
    return parser


def output_path_for(
    output_root: Path, n_obs: int, seed: int, convention: str
) -> Path:
    return output_root / f"nuts_n_obs_{n_obs}_seed_{seed}_conv_{convention}.pkl"


def main() -> None:
    args = build_parser().parse_args()

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_path_for(output_root, args.n_obs, args.seed, args.convention)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing output: {out_path}")

    print(
        f"GNK NUTS refresh (x64): n_obs={args.n_obs}, seed={args.seed}, "
        f"convention={args.convention}",
        flush=True,
    )

    # Reconstruct float32 octile summary then cast to float64 for NUTS.
    x_obs_f32 = reconstruct_x_obs_float32(args.seed, args.n_obs, args.convention)
    x_obs_f64 = jnp.asarray(x_obs_f32, dtype=jnp.float64)

    x_f32_bytes = np.asarray(x_obs_f32).tobytes()
    sha = hashlib.sha256(x_f32_bytes).hexdigest()
    print(f"  x_obs sha256 (float32 bytes): {sha}", flush=True)

    sampler_seed = stable_int("nuts_v3", args.n_obs, args.seed, args.convention)
    print(f"  sampler_seed: {sampler_seed}", flush=True)

    result = run_nuts_canonical(
        x_obs_f64=x_obs_f64,
        n_obs=args.n_obs,
        sampler_seed=sampler_seed,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        target_accept_prob=args.target_accept_prob,
    )

    diagnostics = diagnostics_from_samples(result.samples_grouped, result.extra_fields)

    fingerprint: dict[str, Any] = {
        "samples": np.asarray(result.samples_grouped, dtype=np.float64),
        "param_order": list(PARAM_NAMES),
        "x_obs_summary_unstandardised_float32": np.asarray(x_obs_f32),
        "x_obs_summary_unstandardised_float64": np.asarray(x_obs_f64),
        "x_obs_summary_unstandardised_sha256": sha,
        "n_obs": args.n_obs,
        "data_seed": args.seed,
        "convention": args.convention,
        "sampler_seed": sampler_seed,
        "density_version": DENSITY_VERSION,
        "data_epoch": "post_random_key_2024_11",
        "jitter": 1e-6,
        "c": 0.8,
        "quantile_method": "type-7",
        "num_chains": args.num_chains,
        "num_warmup": args.num_warmup,
        "num_samples_per_chain": args.num_samples,
        "thinning": 1,
        "target_accept_prob": args.target_accept_prob,
        "mass_matrix": "dense",
        "runtime_seconds": result.runtime_seconds,
        "utc_timestamp": utc_now(),
        "environment": environment_record(),
        "diagnostics": diagnostics,
    }
    for name, arr in result.extra_fields.items():
        fingerprint[f"extra_{name}"] = np.asarray(arr)

    with out_path.open("wb") as f:
        pkl.dump(fingerprint, f)
    print(f"  wrote {out_path}", flush=True)
    print(
        "  R-hat: "
        + ", ".join(
            f"{name}={diagnostics['per_parameter'][name]['r_hat']:.4f}"
            for name in PARAM_NAMES
        ),
        flush=True,
    )
    print(
        "  n_eff: "
        + ", ".join(
            f"{name}={diagnostics['per_parameter'][name]['n_eff']:.0f}"
            for name in PARAM_NAMES
        ),
        flush=True,
    )
    if "divergence_count" in diagnostics:
        print(f"  divergences: {diagnostics['divergence_count']}", flush=True)
    print(f"  runtime: {result.runtime_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
