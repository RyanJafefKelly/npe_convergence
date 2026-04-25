#!/usr/bin/env python
"""Small robustness check for eta vs standardised-u oracle KL estimates.

Exact KLs are invariant under the affine map
u = (eta - mu_eta) / sigma_eta, eta = logit(theta / 10).  The finite-sample
Perez-Cruz/kNN estimator uses Euclidean distances, so it is not guaranteed to be
exactly invariant to anisotropic diagonal scaling.  This script regenerates the
Gaussian-NPE training standardisation constants for a small selected subset and
compares K_u^* estimated in eta coordinates with K_u^* estimated in actual
standardised u coordinates.
"""
from __future__ import annotations

import argparse
import json
import pickle as pkl
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from jax.scipy.special import logit

from npe_convergence.metrics import kullback_leibler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "res" / "gnk"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "plots"
FIT_SIZE = 5000
N_METRIC = 2000
THETA_EPS = 1e-6
JITTER = 1e-8


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_pickle_array(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        return np.asarray(pkl.load(f), dtype=np.float64)


def nuts_path(cache_dir: Path, n: int, seed: int) -> Path:
    for prefix in ("nuts_cache_v2_n_obs", "nuts_cache_v2_flow_n_obs"):
        path = cache_dir / f"{prefix}_{n}_seed_{seed}.pkl"
        if path.exists():
            return path
    raise FileNotFoundError(f"No NUTS cache for n={n}, seed={seed}")


def theta_to_eta(theta: np.ndarray) -> tuple[np.ndarray, int]:
    clipped = np.clip(theta, THETA_EPS, 10.0 - THETA_EPS)
    clip_count = int(np.count_nonzero(clipped != theta))
    x = clipped / 10.0
    return np.log(x) - np.log1p(-x), clip_count


def training_transform_constants(n_sims: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate run_gnk_gaussian.py's training theta standardisation constants."""
    key = random.key(seed)
    key, _ = random.split(key)  # observed-data z key
    key, subkey = random.split(key)  # prior theta key
    tol = 1e-6
    thetas_bounded = dist.Uniform(0 + tol, 10 - tol).sample(subkey, (n_sims, 4))
    thetas_unbounded = logit(thetas_bounded / 10)
    return (
        np.asarray(thetas_unbounded.mean(axis=0), dtype=np.float64),
        np.asarray(thetas_unbounded.std(axis=0), dtype=np.float64),
    )


def stable_cov(samples: np.ndarray) -> np.ndarray:
    cov = np.cov(samples, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    eye = np.eye(cov.shape[0])
    for scale in (0.0, JITTER, 1e-7, 1e-6, 1e-5):
        candidate = cov + scale * eye
        try:
            if np.linalg.slogdet(candidate)[0] > 0:
                np.linalg.cholesky(candidate)
                return candidate
        except np.linalg.LinAlgError:
            pass
    raise np.linalg.LinAlgError("Could not regularise covariance")


def finite_sample_kl(true_samples: np.ndarray, sim_samples: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(kullback_leibler(true_samples, sim_samples))
    if np.isfinite(value):
        return value, 0.0
    pooled = np.vstack([true_samples, sim_samples])
    scale = np.maximum(pooled.std(axis=0, ddof=1), 1.0)
    for eps in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        true_j = true_samples + rng.normal(0.0, eps * scale, size=true_samples.shape)
        sim_j = sim_samples + rng.normal(0.0, eps * scale, size=sim_samples.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = float(kullback_leibler(true_j, sim_j))
        if np.isfinite(value):
            return value, eps
    return value, float("nan")


def oracle_kl(samples: np.ndarray, seed: int) -> tuple[float, float]:
    fit = samples[:FIT_SIZE]
    held_out = samples[FIT_SIZE : FIT_SIZE + N_METRIC]
    mean = fit.mean(axis=0)
    cov = stable_cov(fit)
    rng = np.random.default_rng(seed)
    oracle = rng.multivariate_normal(mean, cov, size=len(held_out), check_valid="raise")
    return finite_sample_kl(held_out, oracle, rng)


def parse_specs(specs: list[str]) -> list[tuple[int, int, int]]:
    triples = []
    for spec in specs:
        n, n_sims, seed = spec.split(":")
        triples.append((int(n), int(n_sims), int(seed)))
    return triples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default="gnk_eta_vs_u_oracle_check_20260425")
    parser.add_argument(
        "--spec",
        action="append",
        default=None,
        help="Triples n:N:seed. Repeat for multiple checks.",
    )
    args = parser.parse_args()

    rows = []
    specs = args.spec if args.spec is not None else ["100:1000:0", "500:3107:0", "5000:42585:0"]
    for n, n_sims, seed in parse_specs(specs):
        path = nuts_path(args.cache_dir, n, seed)
        nuts = load_pickle_array(path)
        eta, clip_count = theta_to_eta(nuts)
        mu_eta, sigma_eta = training_transform_constants(n_sims, seed)
        u = (eta - mu_eta) / sigma_eta
        eta_kl, eta_jitter = oracle_kl(eta, seed + 10_000)
        u_kl, u_jitter = oracle_kl(u, seed + 10_000)
        rows.append(
            {
                "n": n,
                "N": n_sims,
                "seed": seed,
                "nuts_cache_path": rel(path),
                "clip_count_nuts": clip_count,
                "K_u_eta_knn": eta_kl,
                "K_u_standardised_knn": u_kl,
                "difference_standardised_minus_eta": u_kl - eta_kl,
                "eta_kl_jitter": eta_jitter,
                "standardised_u_kl_jitter": u_jitter,
                "mu_eta": ",".join(f"{x:.10g}" for x in mu_eta),
                "sigma_eta": ",".join(f"{x:.10g}" for x in sigma_eta),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = args.output_dir / f"{args.output_prefix}.csv"
    df.to_csv(csv_path, index=False)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "script": rel(Path(__file__)),
        "output_csv": rel(csv_path),
        "rows": rows,
        "max_abs_difference": float(df["difference_standardised_minus_eta"].abs().max()),
        "notes": [
            "Exact KL is affine invariant, but finite-sample kNN estimates can differ under diagonal standardisation.",
            "This check regenerates training mu_eta/sigma_eta for selected n:N:seed triples.",
        ],
    }
    json_path = args.output_dir / f"{args.output_prefix}.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(df.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
