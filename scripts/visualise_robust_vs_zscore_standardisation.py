#!/usr/bin/env python
"""Visual confirmation that asinh + median/IQR fixes the GNK squashing.

Produces a multi-panel figure showing per-octile:
  (row 1) raw simulated octile distribution under Uniform(0, 10)^4
          (symlog x-axis so extreme tails are visible without
          obliterating the central mass).
  (row 2) z-score standardised distribution. x_obs marked. Central
          mass is squashed near zero, extreme outliers spread out.
  (row 3) asinh(s/c) + median/IQR robust standardisation, same
          training set, same c_j as the codex run. x_obs marked.
          Central mass is spread out, extreme outliers compressed.
  (row 4) the transform curve s -> z robust for each octile, to
          verify monotonicity and smoothness.

This is a sanity-check figure for the audit; it is not a paper
figure. Outputs to
docs/meeting_2026_05_18/gnk_standardisation_diag/robust_vs_zscore_panel.png.
"""
from __future__ import annotations

import argparse
import json
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


def reconstruct_x_obs(n_obs: int, seed: int) -> np.ndarray:
    key = random.key(seed)
    _, z_key = random.split(key)
    z = random.normal(z_key, shape=(n_obs,), dtype=jnp.float32)
    x = gnk(z, *TRUE_THETA)
    summary = jnp.squeeze(ss_octile(jnp.atleast_2d(x)))
    return np.asarray(summary, dtype=np.float64)


def simulate_full_prior(n_obs: int, n_sims: int, key_seed: int) -> np.ndarray:
    key = random.key(key_seed)
    key, sub = random.split(key)
    tol = 1e-6
    thetas = dist.Uniform(0 + tol, 10 - tol).sample(sub, (n_sims, 4))
    A, B, g, k = thetas.T
    key, sub = random.split(key)
    summaries = get_summaries_batches(
        sub, A, B, g, k, n_obs=n_obs, n_sims=n_sims, batch_size=min(1000, n_sims)
    )
    return np.asarray(summaries).T


def asinh_transform(s: np.ndarray, c: float) -> np.ndarray:
    return np.arcsinh(s / c)


def compute_c_per_coord(summaries: np.ndarray, c_min: float = 1.0, c_max: float = 100.0) -> np.ndarray:
    """Median absolute deviation per coord, clipped to [c_min, c_max]."""
    median = np.median(summaries, axis=0)
    mad = np.median(np.abs(summaries - median), axis=0)
    return np.clip(mad, c_min, c_max)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-sims", type=int, default=100_000)
    parser.add_argument("--key-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "meeting_2026_05_18"
            / "gnk_standardisation_diag"
            / "robust_vs_zscore_panel.png"
        ),
    )
    parser.add_argument(
        "--c-cap",
        type=float,
        default=100.0,
        help="Cap for MAD-based inner scale c_j.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Simulating {args.n_sims} prior-predictive octile draws at n_obs={args.n_obs}...")
    summaries = simulate_full_prior(args.n_obs, args.n_sims, args.key_seed)
    x_obs = reconstruct_x_obs(args.n_obs, args.seed)
    print(f"  summaries shape: {summaries.shape}")
    print(f"  x_obs: {x_obs.tolist()}")

    # Per-coordinate stats for z-score and robust.
    mu = summaries.mean(axis=0)
    sd = summaries.std(axis=0)
    c_per_coord = compute_c_per_coord(summaries, c_max=args.c_cap)

    z_summaries = (summaries - mu) / sd
    z_obs = (x_obs - mu) / sd

    asinh_summaries = np.zeros_like(summaries)
    asinh_x_obs = np.zeros_like(x_obs)
    for j in range(7):
        asinh_summaries[:, j] = asinh_transform(summaries[:, j], c_per_coord[j])
        asinh_x_obs[j] = asinh_transform(x_obs[j], c_per_coord[j])
    robust_median = np.median(asinh_summaries, axis=0)
    robust_iqr = (
        np.percentile(asinh_summaries, 75, axis=0)
        - np.percentile(asinh_summaries, 25, axis=0)
    )
    robust_scale = robust_iqr / 1.349
    z_robust_summaries = (asinh_summaries - robust_median) / robust_scale
    z_robust_obs = (asinh_x_obs - robust_median) / robust_scale

    print()
    print("Per-octile diagnostic numbers:")
    print(
        f"{'oct':<5}{'c_j':>12}{'raw_sd':>14}{'zscore_iqr':>14}"
        f"{'robust_iqr':>14}{'zscore z_obs':>16}{'robust z_obs':>16}"
    )
    for j in range(7):
        zscore_iqr = (
            np.percentile(z_summaries[:, j], 75) - np.percentile(z_summaries[:, j], 25)
        )
        rob_iqr = (
            np.percentile(z_robust_summaries[:, j], 75)
            - np.percentile(z_robust_summaries[:, j], 25)
        )
        print(
            f"{j+1:<5d}{c_per_coord[j]:>12.3f}{sd[j]:>14.3f}"
            f"{zscore_iqr:>14.3f}{rob_iqr:>14.3f}"
            f"{z_obs[j]:>16.3f}{z_robust_obs[j]:>16.3f}"
        )

    fig, axes = plt.subplots(4, 7, figsize=(28, 16))

    for j in range(7):
        # Row 1: raw with symlog. The range can span 10 orders of magnitude.
        ax = axes[0, j]
        s_j = summaries[:, j]
        finite_mask = np.isfinite(s_j)
        s_j = s_j[finite_mask]
        # symlog-friendly bins. Use linspace on symlog-transformed values.
        s_signed_log = np.sign(s_j) * np.log10(1 + np.abs(s_j))
        ax.hist(s_signed_log, bins=200, color="C0", alpha=0.7)
        x_obs_signed_log = np.sign(x_obs[j]) * np.log10(1 + np.abs(x_obs[j]))
        ax.axvline(x_obs_signed_log, color="red", linewidth=2, label="x_obs")
        ax.set_title(f"octile {j+1}, raw (sign x log10(1+|s|))\n"
                     f"min={s_j.min():.2g}, max={s_j.max():.2g}, sd={sd[j]:.2g}")
        ax.set_yscale("log")
        if j == 0:
            ax.set_ylabel("raw\nlog hist count")

        # Row 2: z-score.
        ax = axes[1, j]
        # Use a wide range so we can see how isolated x_obs is from the tails.
        clip = 20
        z_clip = np.clip(z_summaries[:, j], -clip, clip)
        ax.hist(z_clip, bins=200, range=(-clip, clip), color="C1", alpha=0.7)
        ax.axvline(z_obs[j], color="red", linewidth=2, label="x_obs")
        z25, z75 = np.percentile(z_summaries[:, j], [25, 75])
        ax.axvspan(z25, z75, color="gray", alpha=0.25, label="IQR")
        ax.set_xlim(-clip, clip)
        ax.set_title(f"z-score (clipped at +/-{clip})\n"
                     f"z_obs={z_obs[j]:.3f}, iqr={(z75-z25):.3f}")
        if j == 0:
            ax.set_ylabel("z-score\nhist count")

        # Row 3: robust asinh + median/IQR.
        ax = axes[2, j]
        clip_r = 8
        zr = z_robust_summaries[:, j]
        ax.hist(np.clip(zr, -clip_r, clip_r), bins=200, range=(-clip_r, clip_r),
                color="C2", alpha=0.7)
        ax.axvline(z_robust_obs[j], color="red", linewidth=2, label="x_obs")
        r25, r75 = np.percentile(zr, [25, 75])
        ax.axvspan(r25, r75, color="gray", alpha=0.25, label="IQR")
        ax.set_xlim(-clip_r, clip_r)
        ax.set_title(f"asinh + median/IQR\n"
                     f"z_obs={z_robust_obs[j]:.3f}, iqr={(r75-r25):.3f}, c={c_per_coord[j]:.2g}")
        if j == 0:
            ax.set_ylabel("robust\nhist count")

        # Row 4: transform curve s -> z_robust.
        ax = axes[3, j]
        s_grid = np.sort(s_j)
        z_grid = (asinh_transform(s_grid, c_per_coord[j]) - robust_median[j]) / robust_scale[j]
        # Plot only finite range; use signed-log x to fit huge tails.
        s_signed_log_grid = np.sign(s_grid) * np.log10(1 + np.abs(s_grid))
        ax.plot(s_signed_log_grid, z_grid, color="C3", linewidth=1.2)
        ax.axhline(z_robust_obs[j], color="red", linestyle="--", alpha=0.6)
        ax.axvline(np.sign(x_obs[j]) * np.log10(1 + abs(x_obs[j])), color="red",
                   linestyle="--", alpha=0.6)
        ax.scatter([np.sign(x_obs[j]) * np.log10(1 + abs(x_obs[j]))],
                   [z_robust_obs[j]], color="red", zorder=5, s=40,
                   label=f"x_obs=({x_obs[j]:.2f}, z={z_robust_obs[j]:.2f})")
        ax.set_title("transform s -> z_robust (monotone?)")
        ax.set_xlabel("sign x log10(1+|s|)")
        if j == 0:
            ax.set_ylabel("transform\nz_robust")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Robust (asinh + median/IQR) vs z-score standardisation, "
        f"n_obs={args.n_obs}, x_obs at seed={args.seed}, N_sims_diag={args.n_sims}\n"
        f"Row 1 raw (symlog), row 2 z-score, row 3 robust, row 4 transform curve. "
        f"Red = x_obs.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=110)
    plt.close(fig)
    print(f"\nWrote {args.output}")

    # Also save numerical snapshot for posterity.
    json_path = args.output.with_suffix(".json")
    record = {
        "n_obs": args.n_obs,
        "seed": args.seed,
        "n_sims_diagnostic": args.n_sims,
        "x_obs": x_obs.tolist(),
        "c_per_coord": c_per_coord.tolist(),
        "robust_median_per_coord": robust_median.tolist(),
        "robust_scale_per_coord": robust_scale.tolist(),
        "z_obs_zscore": z_obs.tolist(),
        "z_obs_robust": z_robust_obs.tolist(),
        "raw_sd_per_coord": sd.tolist(),
        "raw_max_per_coord": summaries.max(axis=0).tolist(),
        "raw_min_per_coord": summaries.min(axis=0).tolist(),
    }
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
