#!/usr/bin/env python
"""Aggregate GNK NPE diagnostics across seeds at a given (n_obs, n_sims).

For every seed with cached artifacts, computes per-parameter:
- sigma_ratio_*   std(NPE) / std(NUTS). 1.0 = matched width.
- centre_dev_*    (mean(NPE) - mean(NUTS)) / std(NUTS). disagreement with NUTS.
- freq_err_*      (mean(NPE) - theta_0) / std(NUTS). frequentist centre error.
- pit_*           empirical CDF of NPE at theta_0. uniform across seeds if calibrated.

Also computes NUTS self-diagnostics (freq_err_nuts, nuts_covers95) to test
whether the NUTS reference itself is well-calibrated with respect to theta_0.

Outputs:
- notebooks/plots/gnk_per_seed_<n_obs>_<n_sims>.csv
- notebooks/plots/gnk_aggregate_<n_obs>_<n_sims>.csv
- notebooks/plots/gnk_distributions_<n_obs>_<n_sims>.pdf
"""
from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RES_DIR = REPO_ROOT / "res" / "gnk"
PLOTS_DIR = REPO_ROOT / "notebooks" / "plots"

TRUE_PARAMS = np.array([3.0, 1.0, 2.0, 0.5])
PARAM_NAMES = ["A", "B", "g", "k"]


def _load_pkl(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return np.asarray(pkl.load(f))


def load_nuts(n_obs: int, seed: int) -> np.ndarray | None:
    for prefix in ("nuts_cache_v2_n_obs", "nuts_cache_v2_flow_n_obs"):
        arr = _load_pkl(RES_DIR / f"{prefix}_{n_obs}_seed_{seed}.pkl")
        if arr is not None:
            return arr
    return None


def load_npe(flavor: str, n_obs: int, n_sims: int, seed: int) -> np.ndarray | None:
    subdir = "npe" if flavor == "flow" else "gaussian_npe"
    return _load_pkl(
        RES_DIR / f"{subdir}_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}" / "posterior_samples.pkl"
    )


def per_seed_row(nuts: np.ndarray, flow: np.ndarray | None, gnpe: np.ndarray | None) -> dict:
    nuts_mu = nuts.mean(axis=0)
    nuts_sigma = nuts.std(axis=0)
    nuts_q025 = np.quantile(nuts, 0.025, axis=0)
    nuts_q975 = np.quantile(nuts, 0.975, axis=0)

    row: dict[str, float] = {}
    for j, p in enumerate(PARAM_NAMES):
        row[f"freq_err_nuts_{p}"] = (nuts_mu[j] - TRUE_PARAMS[j]) / nuts_sigma[j]
        row[f"nuts_covers95_{p}"] = int(nuts_q025[j] <= TRUE_PARAMS[j] <= nuts_q975[j])
        row[f"sigma_nuts_{p}"] = float(nuts_sigma[j])

    for method, samples in (("flow", flow), ("gnpe", gnpe)):
        if samples is None:
            for p in PARAM_NAMES:
                row[f"sigma_ratio_{p}_{method}"] = np.nan
                row[f"centre_dev_{p}_{method}"] = np.nan
                row[f"freq_err_{p}_{method}"] = np.nan
                row[f"pit_{p}_{method}"] = np.nan
            continue
        mu = samples.mean(axis=0)
        sigma = samples.std(axis=0)
        for j, p in enumerate(PARAM_NAMES):
            row[f"sigma_ratio_{p}_{method}"] = sigma[j] / nuts_sigma[j]
            row[f"centre_dev_{p}_{method}"] = (mu[j] - nuts_mu[j]) / nuts_sigma[j]
            row[f"freq_err_{p}_{method}"] = (mu[j] - TRUE_PARAMS[j]) / nuts_sigma[j]
            row[f"pit_{p}_{method}"] = float((samples[:, j] < TRUE_PARAMS[j]).mean())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--n-sims", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=str, default="0,100",
                        help="inclusive range 'start,end' (default: 0,100)")
    args = parser.parse_args()

    s0, s1 = map(int, args.seeds.split(","))
    rows = []
    skipped = 0
    for seed in range(s0, s1 + 1):
        nuts = load_nuts(args.n_obs, seed)
        if nuts is None:
            skipped += 1
            continue
        flow = load_npe("flow", args.n_obs, args.n_sims, seed)
        gnpe = load_npe("gaussian", args.n_obs, args.n_sims, seed)
        if flow is None and gnpe is None:
            skipped += 1
            continue
        row = per_seed_row(nuts, flow, gnpe)
        row["seed"] = seed
        rows.append(row)

    print(f"Aggregated {len(rows)} seeds (skipped {skipped}).")
    if not rows:
        raise SystemExit("No seeds with cached artifacts found.")

    df = pd.DataFrame(rows)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_csv = PLOTS_DIR / f"gnk_per_seed_n_obs_{args.n_obs}_n_sims_{args.n_sims}.csv"
    df.to_csv(per_seed_csv, index=False)
    print(f"Wrote {per_seed_csv}")

    # Headline summary
    print(f"\n=== HEADLINE: median across seeds | n_obs={args.n_obs}, n_sims={args.n_sims}, N_seeds={len(df)} ===")
    header = f"{'param':<3}{'σflow/σnuts':>12}{'σgnpe/σnuts':>13}{'PITf med':>10}{'PITg med':>10}{'NUTS cov95':>12}{'|freq_err_N|med':>17}"
    print(header)
    print("-" * len(header))
    for p in PARAM_NAMES:
        rf = df[f"sigma_ratio_{p}_flow"].median()
        rg = df[f"sigma_ratio_{p}_gnpe"].median()
        pf = df[f"pit_{p}_flow"].median()
        pg = df[f"pit_{p}_gnpe"].median()
        nc = df[f"nuts_covers95_{p}"].mean()
        fe = df[f"freq_err_nuts_{p}"].abs().median()
        print(f"{p:<3}{rf:>12.2f}{rg:>13.2f}{pf:>10.2f}{pg:>10.2f}{nc:>12.2f}{fe:>17.2f}")

    # Aggregate CSV
    agg_cols = [c for c in df.columns if c != "seed"]
    summary = df[agg_cols].describe(percentiles=[0.1, 0.5, 0.9]).T
    summary_csv = PLOTS_DIR / f"gnk_aggregate_n_obs_{args.n_obs}_n_sims_{args.n_sims}.csv"
    summary.to_csv(summary_csv)
    print(f"\nWrote {summary_csv}")

    # Distribution plot
    fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    for j, p in enumerate(PARAM_NAMES):
        ax = axes[j, 0]
        ax.hist(df[f"sigma_ratio_{p}_flow"].dropna(), bins=25, alpha=0.5, label="flow", color="tab:blue")
        ax.hist(df[f"sigma_ratio_{p}_gnpe"].dropna(), bins=25, alpha=0.5, label="Gaussian-NPE", color="tab:orange")
        ax.axvline(1.0, color="black", linestyle="--", lw=1)
        ax.set_title(f"{p}: $\\sigma_{{method}} / \\sigma_{{NUTS}}$")
        ax.set_xlabel("ratio")
        if j == 0:
            ax.legend(fontsize=8)

        ax = axes[j, 1]
        ax.hist(df[f"pit_{p}_flow"].dropna(), bins=20, alpha=0.5, label="flow", color="tab:blue")
        ax.hist(df[f"pit_{p}_gnpe"].dropna(), bins=20, alpha=0.5, label="Gaussian-NPE", color="tab:orange")
        ax.axvline(0.5, color="black", linestyle=":", lw=1)
        ax.axhline(len(df) / 20, color="grey", linestyle="--", lw=1)
        ax.set_title(f"{p}: PIT of $\\theta_0$")
        ax.set_xlabel("empirical CDF of NPE at $\\theta_0$")
        if j == 0:
            ax.legend(fontsize=8)

        ax = axes[j, 2]
        ax.hist(df[f"freq_err_nuts_{p}"].dropna(), bins=25, color="#555555")
        ax.axvline(0.0, color="red", lw=1)
        ax.set_title(f"{p}: NUTS freq-err $(\\mu_{{NUTS}} - \\theta_0)/\\sigma_{{NUTS}}$")
        ax.set_xlabel("std-units")

    fig.suptitle(
        f"GNK seed aggregation: $n={args.n_obs}$, $N={args.n_sims}$, {len(df)} seeds",
        fontsize=12,
    )
    fig.tight_layout()
    dist_pdf = PLOTS_DIR / f"gnk_distributions_n_obs_{args.n_obs}_n_sims_{args.n_sims}.pdf"
    fig.savefig(dist_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {dist_pdf}")


if __name__ == "__main__":
    main()
