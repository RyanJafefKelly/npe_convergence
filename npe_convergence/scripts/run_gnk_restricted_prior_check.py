"""Restricted-prior Gaussian-NPE sanity check for GNK.

Pro's section 8 / 18 in the GPT-5.5 brief recommends a restricted /
local-proposal NPE sanity check at the headline cell (n=5000, N=n^2) to
distinguish broad-prior amortisation error from NPE implementation
limits. This script trains Gaussian-NPE under a per-parameter local box
that contains the truth, using a per-parameter affine-logit transform.

Usage:
    JAX_ENABLE_X64=1 python npe_convergence/scripts/run_gnk_restricted_prior_check.py \\
        --n-obs 5000 --seed 50 --convention gaussian --n-sims 25000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle as pkl
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
assert jax.config.read("jax_enable_x64")

import jax.numpy as jnp
import jax.random as random
import matplotlib
import numpy as np
import numpyro  # type: ignore
import numpyro.distributions as dist  # type: ignore
from jax.scipy.special import expit, logit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import gnk, get_summaries_batches, ss_octile
from npe_convergence.methods.gaussian_npe import (
    ConditionalGaussianNPE,
    TrainConfig,
    fit,
    sample,
)
from npe_convergence.metrics import kullback_leibler, median_heuristic, unbiased_mmd


PARAM_NAMES = ("A", "B", "g", "k")
TRUE_THETA = jnp.asarray([3.0, 1.0, 2.0, 0.5])
DEFAULT_BOX = {
    "A": (2.5, 3.5),
    "B": (0.6, 1.4),
    "g": (1.4, 2.6),
    "k": (0.2, 0.8),
}
PRIOR_BOX_HARD = (0.0, 10.0)
MASS_THRESHOLD_PASS = 0.995  # canonical posterior mass inside local box
MASS_THRESHOLD_TARGET = 0.999


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_utc_stamp(created_at: str) -> str:
    return created_at.replace("-", "").replace(":", "").replace("+00:00", "Z")


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def rng_for(seed: int, *parts: object):
    key = random.key(seed)
    for part in parts:
        key = random.fold_in(key, stable_int(part))
    return key


# ---------------------------------------------------------------------------
# Convention-aware x_obs reconstruction. Matches run_gnk_nuts_refresh.py.
# ---------------------------------------------------------------------------


def reconstruct_x_obs(n_obs: int, seed: int, convention: str) -> jnp.ndarray:
    if convention == "flow":
        key = random.key(seed)
        z_key = key
    elif convention == "gaussian":
        key = random.key(seed)
        _, z_key = random.split(key)
    else:
        raise ValueError(f"unknown convention: {convention}")
    z = random.normal(z_key, shape=(n_obs,))
    x_raw = gnk(z, *TRUE_THETA)
    summary = jnp.squeeze(ss_octile(jnp.atleast_2d(x_raw)))
    return summary


# ---------------------------------------------------------------------------
# Per-parameter affine logit transform.
# ---------------------------------------------------------------------------


def affine_logit(theta: jnp.ndarray, lows: jnp.ndarray, highs: jnp.ndarray) -> jnp.ndarray:
    u = (theta - lows) / (highs - lows)
    u = jnp.clip(u, 1e-6, 1.0 - 1e-6)
    return logit(u)


def affine_logit_inverse(
    eta: jnp.ndarray, lows: jnp.ndarray, highs: jnp.ndarray
) -> jnp.ndarray:
    u = expit(eta)
    return lows + u * (highs - lows)


# ---------------------------------------------------------------------------
# Canonical reference loading + local-box truncation.
# ---------------------------------------------------------------------------


def load_canonical_reference(
    n_obs: int, seed: int, convention: str, v3_root: Path
) -> np.ndarray:
    path = v3_root / f"nuts_n_obs_{n_obs}_seed_{seed}_conv_{convention}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"canonical reference missing: {path}")
    with path.open("rb") as f:
        fingerprint = pkl.load(f)
    grouped = np.asarray(fingerprint["samples"])
    return grouped.reshape(-1, grouped.shape[-1])


def truncate_to_box(
    samples: np.ndarray, lows: np.ndarray, highs: np.ndarray
) -> tuple[np.ndarray, float]:
    mask = np.all((samples >= lows) & (samples <= highs), axis=1)
    inside = samples[mask]
    return inside, float(mask.mean())


def deduplicate(samples: np.ndarray) -> tuple[np.ndarray, int]:
    unique = np.unique(np.asarray(samples, dtype=np.float64), axis=0)
    return unique, int(samples.shape[0] - unique.shape[0])


# ---------------------------------------------------------------------------
# Main pipeline.
# ---------------------------------------------------------------------------


def parse_box(value: str) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for spec in value.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        name, range_str = spec.split(":", 1)
        lo, hi = (float(s) for s in range_str.split(","))
        out[name.strip()] = (lo, hi)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Overrides model-init, fit, and posterior-sampling RNG only.",
    )
    parser.add_argument(
        "--convention", choices=("flow", "gaussian"), default="gaussian"
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        required=True,
        help="Simulation budget for NPE training (e.g. n^2).",
    )
    parser.add_argument(
        "--box",
        type=str,
        default="A:2.5,3.5;B:0.6,1.4;g:1.4,2.6;k:0.2,0.8",
        help="Per-parameter local box, semicolon-separated 'name:lo,hi'.",
    )
    parser.add_argument(
        "--v3-root", type=Path, default=REPO_ROOT / "res" / "gnk_v3_refs"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "res" / "gnk_restricted_prior",
    )
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Simulator batch size."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=2000, help="Training epochs."
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument(
        "--mass-threshold",
        type=float,
        default=MASS_THRESHOLD_PASS,
        help="Required canonical posterior mass inside the box to proceed.",
    )
    parser.add_argument(
        "--force-low-mass",
        action="store_true",
        help="Continue even if canonical mass inside box is below threshold.",
    )
    parser.add_argument(
        "--save-training-summaries",
        action="store_true",
        help="Persist simulated summaries and standardisation arrays for local density checks.",
    )
    parser.add_argument(
        "--created-at",
        type=str,
        default=None,
        help="UTC timestamp for reproducible output directory names.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def run_git(cmd: list[str], default: str | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *cmd], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return default


def main() -> None:
    args = build_parser().parse_args()
    created_at = args.created_at or utc_now()
    training_seed = args.seed if args.training_seed is None else args.training_seed

    box = parse_box(args.box)
    if set(box.keys()) != set(PARAM_NAMES):
        raise ValueError(
            f"box must specify all of {PARAM_NAMES}, got {sorted(box.keys())}"
        )
    lows = jnp.asarray([box[n][0] for n in PARAM_NAMES])
    highs = jnp.asarray([box[n][1] for n in PARAM_NAMES])
    # Sanity check that truth is inside the box.
    if not jnp.all((TRUE_THETA >= lows) & (TRUE_THETA <= highs)):
        raise ValueError(
            f"truth {TRUE_THETA.tolist()} not contained in box {box}; refusing"
        )

    print(f"Restricted-prior sanity check: n_obs={args.n_obs}, seed={args.seed}, "
          f"training_seed={training_seed}, convention={args.convention}, "
          f"n_sims={args.n_sims}")
    print(f"  local box: " + ", ".join(
        f"{name}=[{box[name][0]:.3f}, {box[name][1]:.3f}]" for name in PARAM_NAMES
    ))

    # -- Load canonical v3 reference and check mass inside box --------------
    ref_samples = load_canonical_reference(
        args.n_obs, args.seed, args.convention, args.v3_root
    )
    ref_inside, mass_inside = truncate_to_box(ref_samples, np.asarray(lows), np.asarray(highs))
    print(
        f"  canonical reference: {ref_samples.shape[0]} samples, "
        f"{mass_inside:.4f} inside box (target {args.mass_threshold:.4f})"
    )
    if mass_inside < args.mass_threshold and not args.force_low_mass:
        raise RuntimeError(
            f"canonical mass inside box {mass_inside:.4f} below threshold "
            f"{args.mass_threshold:.4f}; pass --force-low-mass to override"
        )

    # -- Reconstruct x_obs --------------------------------------------------
    x_obs = reconstruct_x_obs(args.n_obs, args.seed, args.convention)
    print(f"  x_obs (unstandardised): {[float(v) for v in x_obs]}")

    # -- Simulator on restricted prior --------------------------------------
    data_rng = random.key(args.seed)
    data_rng, sim_key = random.split(data_rng)
    n_sims = args.n_sims
    # Sample thetas from local box uniformly.
    u_samples = random.uniform(
        sim_key, shape=(n_sims, 4), minval=jnp.zeros(4), maxval=jnp.ones(4)
    )
    thetas_bounded = lows + u_samples * (highs - lows)
    thetas_unbounded = affine_logit(thetas_bounded, lows, highs)

    print(">>> Simulating from restricted prior...")
    A_s, B_s, g_s, k_s = thetas_bounded.T
    data_rng, sub = random.split(data_rng)
    x_sims = get_summaries_batches(
        sub,
        A_s,
        B_s,
        g_s,
        k_s,
        n_obs=args.n_obs,
        n_sims=n_sims,
        batch_size=min(args.batch_size, n_sims),
    )
    print(">>> Simulations done, standardising and training Gaussian-NPE...")

    # Standardise.
    thetas_mean = thetas_unbounded.mean(axis=0)
    thetas_std = thetas_unbounded.std(axis=0)
    thetas_std_arr = (thetas_unbounded - thetas_mean) / thetas_std

    sim_summ = x_sims.T
    sim_summ_mean = sim_summ.mean(axis=0)
    sim_summ_sd = sim_summ.std(axis=0)
    sim_summ_std = (sim_summ - sim_summ_mean) / sim_summ_sd
    x_obs_std = (x_obs - sim_summ_mean) / sim_summ_sd

    # Train Gaussian-NPE.
    sub = rng_for(training_seed, "restricted_prior", args.n_obs, args.seed, args.n_sims, "model_init")
    model = ConditionalGaussianNPE(
        d_summary=7, d_theta=4, hidden_dims=(128, 128), key=sub
    )
    sub = rng_for(training_seed, "restricted_prior", args.n_obs, args.seed, args.n_sims, "fit")
    train_cfg = TrainConfig(
        lr=args.lr,
        batch_size=256,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    t0 = time.perf_counter()
    model, losses = fit(model, thetas_std_arr, sim_summ_std, key=sub, config=train_cfg)
    training_seconds = time.perf_counter() - t0
    print(f">>> Training done in {training_seconds:.1f}s")

    # Sample posterior, invert transform back to theta-space.
    sub = rng_for(
        training_seed, "restricted_prior", args.n_obs, args.seed, args.n_sims, "posterior"
    )
    posterior_std = sample(model, x_obs_std, args.num_posterior_samples, key=sub)
    posterior_unbounded = posterior_std * thetas_std + thetas_mean
    posterior_theta = affine_logit_inverse(posterior_unbounded, lows, highs)
    posterior_theta = np.asarray(posterior_theta)

    # -- Compare to canonical v3 reference, truncated to local box ----------
    n_metric = 2000
    posterior_unique, posterior_duplicates = deduplicate(posterior_theta)
    ref_unique, ref_duplicates = deduplicate(ref_inside)
    if posterior_unique.shape[0] < n_metric or ref_unique.shape[0] < n_metric:
        raise RuntimeError("not enough unique samples for 2000-sample KL")
    sub = rng_for(
        training_seed, "restricted_prior", args.n_obs, args.seed, args.n_sims, "metric_npe"
    )
    idx_npe = np.asarray(
        random.permutation(sub, posterior_unique.shape[0])[:n_metric]
    )
    sub = rng_for(
        training_seed, "restricted_prior", args.n_obs, args.seed, args.n_sims, "metric_ref"
    )
    idx_ref = np.asarray(
        random.permutation(sub, ref_unique.shape[0])[:n_metric]
    )
    ps_thin = posterior_unique[idx_npe]
    ref_thin = ref_unique[idx_ref]

    kl_value = float(kullback_leibler(jnp.asarray(ref_thin), jnp.asarray(ps_thin)))
    lengthscale = float(median_heuristic(jnp.vstack([jnp.asarray(ref_thin), jnp.asarray(ps_thin)])))
    mmd_value = float(unbiased_mmd(jnp.asarray(ref_thin), jnp.asarray(ps_thin), lengthscale))

    # Per-parameter agreement.
    ref_med = np.median(ref_inside, axis=0)
    ref_std = ref_inside.std(axis=0)
    npe_med = np.median(posterior_theta, axis=0)
    npe_std = posterior_theta.std(axis=0)
    median_shifts = {
        name: float(abs(npe_med[i] - ref_med[i]) / ref_std[i])
        for i, name in enumerate(PARAM_NAMES)
    }
    sd_ratios = {
        name: float(npe_std[i] / ref_std[i]) for i, name in enumerate(PARAM_NAMES)
    }

    pass_kl = kl_value < 0.2
    pass_median = max(median_shifts.values()) < 0.1
    pass_sd = all(0.9 <= r <= 1.1 for r in sd_ratios.values())
    overall_pass = pass_kl and pass_median and pass_sd

    # -- Persist ------------------------------------------------------------
    output_dir = args.output_root / (
        f"restricted_n_obs_{args.n_obs}_seed_{args.seed}_conv_{args.convention}_"
        f"n_sims_{args.n_sims}_train_{training_seed}_{compact_utc_stamp(created_at)}"
    )
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "posterior_samples.npz",
        theta=posterior_theta,
        unbounded=np.asarray(posterior_unbounded),
    )
    if args.save_training_summaries:
        np.savez_compressed(
            output_dir / "training_summaries.npz",
            summaries=np.asarray(sim_summ),
            summary_mean=np.asarray(sim_summ_mean),
            summary_sd=np.asarray(sim_summ_sd),
            x_obs=np.asarray(x_obs),
            n_obs=args.n_obs,
            seed=args.seed,
            training_seed=training_seed,
            n_sims=args.n_sims,
            convention=args.convention,
        )

    # Plot overlay.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, name in enumerate(PARAM_NAMES):
        axes[i].hist(ref_inside[:, i], bins=50, alpha=0.6, label="NUTS (box)", density=True)
        axes[i].hist(posterior_theta[:, i], bins=50, alpha=0.6, label="restricted NPE", density=True)
        axes[i].axvline(float(TRUE_THETA[i]), color="black", linestyle=":")
        axes[i].set_xlim(box[name])
        axes[i].set_title(name)
        axes[i].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "overlay.png", dpi=120)
    plt.close(fig)

    summary = {
        "n_obs": args.n_obs,
        "seed": args.seed,
        "training_seed": training_seed,
        "convention": args.convention,
        "n_sims": args.n_sims,
        "box": {name: list(box[name]) for name in PARAM_NAMES},
        "canonical_mass_inside_box": mass_inside,
        "training_seconds": training_seconds,
        "kl_npe_vs_truncated_nuts": kl_value,
        "mmd_npe_vs_truncated_nuts": mmd_value,
        "n_metric": n_metric,
        "posterior_duplicates_removed": posterior_duplicates,
        "reference_duplicates_removed": ref_duplicates,
        "median_shifts_in_nuts_sds": median_shifts,
        "sd_ratios_npe_over_nuts": sd_ratios,
        "pass_kl_lt_0p2": pass_kl,
        "pass_median_shifts_lt_0p1_sd": pass_median,
        "pass_sd_ratio_in_0p9_1p1": pass_sd,
        "overall_pass": overall_pass,
        "created_at_utc": created_at,
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_dirty": bool(run_git(["status", "--porcelain"], default="dirty")),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "numpyro_version": numpyro.__version__,
        "training_summaries_path": (
            str(output_dir / "training_summaries.npz")
            if args.save_training_summaries
            else None
        ),
        "args": vars(args) | {
            "v3_root": str(args.v3_root),
            "output_root": str(args.output_root),
            "training_seed_resolved": training_seed,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"  wrote {output_dir}")
    print(f"  KL = {kl_value:.4f} (pass < 0.2: {pass_kl})")
    print(f"  median shifts (sd units): {median_shifts}")
    print(f"  sd ratios: {sd_ratios}")
    print(f"  OVERALL PASS: {overall_pass}")


if __name__ == "__main__":
    main()
