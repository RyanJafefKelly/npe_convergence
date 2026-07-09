#!/usr/bin/env python
"""Inspect the standardised simulated summaries for the GNK training pipeline.

Checks whether extreme prior-predictive draws under Uniform(0, 10)^4 cause
z-score standardisation to "squish" the typical (informative) training
summaries into a tiny window, leaving the NPE with poor effective
resolution where it matters.

For both full-prior and restricted-prior designs:
  - simulate N draws,
  - compute octiles per draw,
  - report tail behaviour and standardisation impact,
  - show where x_obs sits after standardisation.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random
import matplotlib
import numpy as np
import numpyro.distributions as dist  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import gnk, get_summaries_batches, ss_octile

TRUE_THETA = jnp.asarray([3.0, 1.0, 2.0, 0.5], dtype=jnp.float32)
DEFAULT_BOX = {
    "A": (2.5, 3.5),
    "B": (0.6, 1.4),
    "g": (1.4, 2.6),
    "k": (0.2, 0.8),
}


def reconstruct_x_obs(n_obs: int, seed: int, convention: str) -> np.ndarray:
    if convention == "flow":
        key = random.key(seed)
        z_key = key
    else:
        key = random.key(seed)
        _, z_key = random.split(key)
    z = random.normal(z_key, shape=(n_obs,), dtype=jnp.float32)
    x = gnk(z, *TRUE_THETA)
    summary = jnp.squeeze(ss_octile(jnp.atleast_2d(x)))
    return np.asarray(summary, dtype=np.float64)


def simulate_full_prior(n_obs: int, n_sims: int, key_seed: int) -> np.ndarray:
    """Draw (theta, octiles) under Uniform(0,10)^4."""
    key = random.key(key_seed)
    key, sub = random.split(key)
    tol = 1e-6
    thetas = dist.Uniform(0 + tol, 10 - tol).sample(sub, (n_sims, 4))
    A, B, g, k = thetas.T
    key, sub = random.split(key)
    summaries = get_summaries_batches(
        sub, A, B, g, k, n_obs=n_obs, n_sims=n_sims, batch_size=min(1000, n_sims)
    )
    return np.asarray(summaries).T  # (n_sims, 7)


def simulate_restricted(n_obs: int, n_sims: int, key_seed: int) -> np.ndarray:
    key = random.key(key_seed)
    key, sub = random.split(key)
    box = DEFAULT_BOX
    lows = jnp.asarray([box["A"][0], box["B"][0], box["g"][0], box["k"][0]])
    highs = jnp.asarray([box["A"][1], box["B"][1], box["g"][1], box["k"][1]])
    u = random.uniform(sub, shape=(n_sims, 4))
    thetas = lows + u * (highs - lows)
    A, B, g, k = thetas.T
    key, sub = random.split(key)
    summaries = get_summaries_batches(
        sub, A, B, g, k, n_obs=n_obs, n_sims=n_sims, batch_size=min(1000, n_sims)
    )
    return np.asarray(summaries).T  # (n_sims, 7)


def report_summary_stats(summaries: np.ndarray, label: str) -> dict:
    print(f"\n=== {label}: N={summaries.shape[0]} draws ===")
    mu = summaries.mean(axis=0)
    sd = summaries.std(axis=0)
    print(f"{'octile':<9}{'mean':>15}{'sd':>15}{'min':>15}{'q01':>15}{'median':>15}{'q99':>15}{'max':>15}")
    rec = {}
    for i in range(7):
        q01, med, q99 = np.quantile(summaries[:, i], [0.01, 0.5, 0.99])
        print(
            f"{i+1:<9d}{mu[i]:>15.3f}{sd[i]:>15.3f}{summaries[:, i].min():>15.3f}"
            f"{q01:>15.3f}{med:>15.3f}{q99:>15.3f}{summaries[:, i].max():>15.3f}"
        )
        rec[f"oct{i+1}"] = {
            "mean": float(mu[i]),
            "sd": float(sd[i]),
            "min": float(summaries[:, i].min()),
            "max": float(summaries[:, i].max()),
            "q01": float(q01),
            "q99": float(q99),
        }
    return {"mean": mu, "sd": sd, "per_octile": rec}


def report_standardisation(
    summaries: np.ndarray, x_obs: np.ndarray, label: str
) -> dict:
    print(f"\n=== {label}: standardisation impact ===")
    mu = summaries.mean(axis=0)
    sd = summaries.std(axis=0)
    z_summaries = (summaries - mu) / sd
    z_obs = (x_obs - mu) / sd
    # The "typical" range of z values; where x_obs falls.
    print(f"{'octile':<9}{'sd_train':>12}{'z_obs':>12}{'|z_obs|':>12}"
          f"{'z_q05':>12}{'z_q95':>12}{'z_iqr_width':>14}")
    rec = {}
    for i in range(7):
        q05, q95 = np.quantile(z_summaries[:, i], [0.05, 0.95])
        q25, q75 = np.quantile(z_summaries[:, i], [0.25, 0.75])
        iqr_width = q75 - q25
        print(
            f"{i+1:<9d}{sd[i]:>12.3f}{z_obs[i]:>12.4f}{abs(z_obs[i]):>12.4f}"
            f"{q05:>12.3f}{q95:>12.3f}{iqr_width:>14.3f}"
        )
        rec[f"oct{i+1}"] = {
            "sd_train": float(sd[i]),
            "z_obs": float(z_obs[i]),
            "z_q05": float(q05),
            "z_q95": float(q95),
            "z_iqr_width": float(iqr_width),
        }
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-sims",
        type=int,
        default=100_000,
        help="Subsample of n_sims for diagnostic (full 25M not needed).",
    )
    parser.add_argument("--key-seed", type=int, default=0)
    parser.add_argument(
        "--convention",
        choices=("flow", "gaussian"),
        default="gaussian",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "meeting_2026_05_18" / "gnk_standardisation_diag",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GNK standardisation diagnostic")
    print(f"  n_obs={args.n_obs}, seed={args.seed}, convention={args.convention}")
    print(f"  diagnostic N_sims={args.n_sims}")

    # Observed summary in the convention used by the cell.
    x_obs = reconstruct_x_obs(args.n_obs, args.seed, args.convention)
    print(f"  x_obs (unstandardised octiles): {x_obs.tolist()}")

    # Full-prior training summaries.
    full = simulate_full_prior(args.n_obs, args.n_sims, args.key_seed)
    full_stats = report_summary_stats(full, "FULL PRIOR Uniform(0,10)^4")
    full_std = report_standardisation(full, x_obs, "FULL PRIOR after z-score")

    # Restricted-prior summaries.
    rest = simulate_restricted(args.n_obs, args.n_sims, args.key_seed + 99)
    rest_stats = report_summary_stats(rest, "RESTRICTED LOCAL BOX")
    rest_std = report_standardisation(rest, x_obs, "RESTRICTED PRIOR after z-score")

    # Plot. Histogram per octile of standardised summaries, x_obs marked.
    fig, axes = plt.subplots(2, 7, figsize=(28, 8), sharey="row")
    for j in range(7):
        mu_f, sd_f = full[:, j].mean(), full[:, j].std()
        z_full = (full[:, j] - mu_f) / sd_f
        z_obs_full = (x_obs[j] - mu_f) / sd_f

        mu_r, sd_r = rest[:, j].mean(), rest[:, j].std()
        z_rest = (rest[:, j] - mu_r) / sd_r
        z_obs_rest = (x_obs[j] - mu_r) / sd_r

        axes[0, j].hist(z_full, bins=200, range=(-10, 10), color="C0", alpha=0.6)
        axes[0, j].axvline(z_obs_full, color="red", linewidth=2)
        axes[0, j].set_title(f"oct {j+1} (full prior)\nx_obs at z={z_obs_full:.2f}, sd_train={sd_f:.2f}")
        axes[0, j].set_xlim(-10, 10)

        axes[1, j].hist(z_rest, bins=100, range=(-5, 5), color="C2", alpha=0.6)
        axes[1, j].axvline(z_obs_rest, color="red", linewidth=2)
        axes[1, j].set_title(f"oct {j+1} (restricted)\nx_obs at z={z_obs_rest:.2f}, sd_train={sd_r:.2f}")
        axes[1, j].set_xlim(-5, 5)

    fig.suptitle(
        f"Standardised octile distributions (N_sims={args.n_sims} per row)\n"
        "Full prior 'squishing': sd_train dominated by extreme tails, "
        "x_obs ends up in a tiny window near zero.",
        fontsize=12,
    )
    fig.tight_layout()
    fig_path = args.output_dir / "standardised_summary_histograms.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\nWrote {fig_path}")

    # Save a JSON summary for later citation.
    import json

    payload = {
        "n_obs": args.n_obs,
        "seed": args.seed,
        "convention": args.convention,
        "n_sims_diagnostic": args.n_sims,
        "x_obs": x_obs.tolist(),
        "full_prior": {
            "summary_stats": full_stats["per_octile"],
            "standardisation": full_std,
            "raw_sd": full_stats["sd"].tolist(),
        },
        "restricted_prior": {
            "summary_stats": rest_stats["per_octile"],
            "standardisation": rest_std,
            "raw_sd": rest_stats["sd"].tolist(),
        },
    }
    json_path = args.output_dir / "standardised_summary_stats.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
