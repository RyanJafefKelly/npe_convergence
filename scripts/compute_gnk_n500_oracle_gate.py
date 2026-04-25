#!/usr/bin/env python
"""Compute the GNK n=500 theta-space Gaussian oracle KL gate.

This reuses the oracle convention from ``scripts/kl_vs_n_theory_plot.py``:
fit a moment-matched Gaussian to the first 5,000 NUTS samples, compare 2,000
held-out NUTS samples against 2,000 Gaussian samples with the Perez-Cruz KL
estimator, and aggregate across cached seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle as pkl
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "res" / "gnk"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "plots"

sys.path.insert(0, str(REPO_ROOT))
from npe_convergence.metrics import kullback_leibler  # noqa: E402

FIT_SIZE = 5000
N_METRIC = 2000
CACHE_PREFIXES = ("nuts_cache_v2_n_obs", "nuts_cache_v2_flow_n_obs")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def parse_seed_range(value: str) -> range:
    start, stop = map(int, value.split(",", 1))
    return range(start, stop + 1)


def git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        )
    except Exception:
        return "unknown"
    return out.strip()


def load_nuts(cache_dir: Path, n_obs: int, seed: int) -> tuple[np.ndarray, Path] | None:
    for prefix in CACHE_PREFIXES:
        path = cache_dir / f"{prefix}_{n_obs}_seed_{seed}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return np.asarray(pkl.load(f)), path
    return None


def compute_kl_oracle(nuts: np.ndarray, seed: int) -> float:
    mu = nuts[:FIT_SIZE].mean(axis=0)
    sigma = np.cov(nuts[:FIT_SIZE], rowvar=False)
    rng = np.random.default_rng(seed)
    if len(nuts) >= FIT_SIZE + N_METRIC:
        held_out = nuts[FIT_SIZE:FIT_SIZE + N_METRIC]
    else:
        idx = rng.permutation(len(nuts))[:N_METRIC]
        held_out = nuts[idx]
    oracle_samples = rng.multivariate_normal(mu, sigma, size=N_METRIC)
    return float(kullback_leibler(held_out, oracle_samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=500)
    parser.add_argument("--seeds", type=str, default="0,100")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default="gnk_n500_oracle_gate_20260425")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    seed_range = parse_seed_range(args.seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}_per_seed.csv"
    json_path = output_dir / f"{args.output_prefix}_summary.json"
    if not args.overwrite:
        for path in (csv_path, json_path):
            if path.exists():
                raise SystemExit(f"Refusing to overwrite existing output: {path}")

    rows = []
    missing_seeds = []
    for seed in seed_range:
        loaded = load_nuts(cache_dir, args.n_obs, seed)
        if loaded is None:
            missing_seeds.append(seed)
            continue
        nuts, cache_path = loaded
        kl = compute_kl_oracle(nuts, seed)
        rows.append(
            {
                "n_obs": args.n_obs,
                "seed": seed,
                "cache_path": display_path(cache_path),
                "posterior_samples": int(len(nuts)),
                "fit_size": FIT_SIZE,
                "metric_samples": N_METRIC,
                "k_theta_star": kl,
            }
        )

    if not rows:
        raise SystemExit(
            f"No cached NUTS posterior samples found for n_obs={args.n_obs} in {args.cache_dir}"
        )

    values = np.array([row["k_theta_star"] for row in rows], dtype=float)
    sample_counts = sorted({row["posterior_samples"] for row in rows})
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    if median <= 0.1:
        recommendation = "use n=500 for high-budget curve"
        decision_bucket = "<=0.1 nats: acceptable"
    elif median < 0.3:
        recommendation = "treat n=500 as diagnostic only; be cautious"
        decision_bucket = "0.1-0.3 nats: diagnostic"
    else:
        recommendation = "use n=1000 for main high-budget BvM-rate evidence"
        decision_bucket = ">=0.3 nats: use n=1000"

    summary = {
        "task": "gnk-n500-oracle-gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_short_hash(),
        "command": " ".join(sys.argv),
        "n_obs": args.n_obs,
        "cache_dir": display_path(cache_dir),
        "cache_prefixes": list(CACHE_PREFIXES),
        "seeds_requested": [seed_range.start, seed_range.stop - 1],
        "seeds_found": [int(row["seed"]) for row in rows],
        "missing_seeds": missing_seeds,
        "n_seeds": len(rows),
        "posterior_samples_per_seed": sample_counts,
        "fit_size": FIT_SIZE,
        "metric_samples": N_METRIC,
        "k_theta_star_median": float(median),
        "k_theta_star_q25": float(q25),
        "k_theta_star_q75": float(q75),
        "k_theta_star_iqr": [float(q25), float(q75)],
        "decision_bucket": decision_bucket,
        "recommendation": recommendation,
        "per_seed_csv": display_path(csv_path),
        "summary_json": display_path(json_path),
    }

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Read {len(rows)} cached NUTS posterior files from {cache_dir}")
    print(f"Missing seeds: {missing_seeds if missing_seeds else 'none'}")
    print(f"Posterior samples per seed: {sample_counts}")
    print(
        "K_theta^*(n={}) median={:.6f}, IQR=[{:.6f}, {:.6f}]".format(
            args.n_obs, median, q25, q75
        )
    )
    print(f"Decision bucket: {decision_bucket}")
    print(f"Recommendation: {recommendation}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
