#!/usr/bin/env python
"""Aggregate Phase 2.4c Stage 1 sweep results.

Reads `summary.txt` from each `res/gnk/sweep_gnpe/h*_lr*_seed_*/`, computes
Delta_N = kl - kl_oracle (where kl_oracle is computed from the NUTS cache for
that seed), and reports a sorted table + baseline comparison.

Decision rule (per plan + ChatGPT):
- Baseline (h=128, lr=5e-4) matches the paper's current config.
- Stage 2 trigger: any config with >= 25% reduction in Delta_N vs baseline.
- Stage 3 trigger: >= 50% reduction.
"""
from __future__ import annotations

import argparse
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO_ROOT / "res" / "gnk" / "sweep_gnpe"

sys.path.insert(0, str(REPO_ROOT))
from npe_convergence.metrics import kullback_leibler  # noqa: E402

FIT_SIZE = 5000
N_METRIC = 2000


def load_nuts(n_obs: int, seed: int) -> np.ndarray | None:
    for prefix in ("nuts_cache_v2_n_obs", "nuts_cache_v2_flow_n_obs"):
        p = REPO_ROOT / "res" / "gnk" / f"{prefix}_{n_obs}_seed_{seed}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                return np.asarray(pkl.load(f))
    return None


def compute_oracle_kl(nuts: np.ndarray, seed: int) -> float:
    mu = nuts[:FIT_SIZE].mean(axis=0)
    Sigma = np.cov(nuts[:FIT_SIZE], rowvar=False)
    rng = np.random.default_rng(seed)
    held_out = nuts[FIT_SIZE:FIT_SIZE + N_METRIC] if len(nuts) >= FIT_SIZE + N_METRIC else nuts[:N_METRIC]
    oracle_samples = rng.multivariate_normal(mu, Sigma, size=N_METRIC)
    return float(kullback_leibler(held_out, oracle_samples))


def parse_summary(path: Path) -> dict:
    d = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        try:
            d[k.strip()] = float(v.strip())
        except ValueError:
            d[k.strip()] = v.strip()
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not SWEEP_DIR.exists():
        print(f"No sweep directory at {SWEEP_DIR}; nothing to aggregate.")
        return

    nuts = load_nuts(args.n_obs, args.seed)
    if nuts is None:
        raise SystemExit(f"No NUTS cache for n_obs={args.n_obs}, seed={args.seed}")
    kl_oracle = compute_oracle_kl(nuts, args.seed)
    print(f"Oracle KL (moment-matched, seed={args.seed}): {kl_oracle:.4f}")

    rows = []
    for cfg_dir in sorted(SWEEP_DIR.iterdir()):
        if not cfg_dir.is_dir() or not cfg_dir.name.endswith(f"_seed_{args.seed}"):
            continue
        summary_path = cfg_dir / "summary.txt"
        if not summary_path.exists():
            print(f"[pending] {cfg_dir.name}")
            continue
        d = parse_summary(summary_path)
        delta_N = float(d["kl"]) - kl_oracle
        rows.append({
            "config": cfg_dir.name.replace(f"_seed_{args.seed}", ""),
            "hidden": int(d.get("hidden_dim", 0)),
            "lr": float(d.get("lr", 0)),
            "kl": float(d["kl"]),
            "delta_N": delta_N,
            "sig_A": float(d["sigma_ratio_A"]),
            "sig_B": float(d["sigma_ratio_B"]),
            "sig_g": float(d["sigma_ratio_g"]),
            "sig_k": float(d["sigma_ratio_k"]),
            "epochs": int(d.get("epochs_run", 0)),
            "wall_s": float(d.get("wall_seconds", 0)),
        })

    if not rows:
        print("No completed sweep runs yet.")
        return

    df = pd.DataFrame(rows).sort_values("delta_N").reset_index(drop=True)
    print("\n=== Sweep results (seed={}, n_obs={}), sorted by Delta_N ===".format(
        args.seed, args.n_obs))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Find baseline (hidden=128, lr=5e-4)
    base_mask = (df.hidden == 128) & (np.isclose(df.lr, 5e-4))
    if base_mask.any():
        base = df[base_mask].iloc[0]
        best = df.iloc[0]
        reduction = (base["delta_N"] - best["delta_N"]) / base["delta_N"] * 100
        print(f"\nBaseline (h=128, lr=5e-4): Delta_N = {base['delta_N']:.3f}")
        print(f"Best config:               Delta_N = {best['delta_N']:.3f}  "
              f"({best['config']})")
        print(f"Best-vs-baseline reduction: {reduction:.1f}%")
        if reduction >= 50:
            print("==> TRIGGER: >=50% reduction. Rerun Phase 2.1 with tuned config.")
        elif reduction >= 25:
            print("==> TRIGGER: 25-50% reduction. Replicate best config across seeds 2-5.")
        else:
            print("==> NO IMPROVEMENT. Commit to Branch B (report finite-N residual).")
    else:
        print("\nBaseline (h=128, lr=5e-4) not yet completed.")


if __name__ == "__main__":
    main()
