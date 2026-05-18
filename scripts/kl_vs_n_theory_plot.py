#!/usr/bin/env python
"""Central GNK figure: KL(Pi_NUTS || Q) vs N, aggregated across seeds.

For each n_obs in {1000, 5000}, aggregates cached kl.txt across available
seeds for flow-NPE and Gaussian-NPE, and computes per-seed oracle KL
(moment-matched Gaussian to NUTS samples). Plots median with IQR band.

Directly illustrates the theory-aligned decomposition:
  (i)   N-dependent NPE estimation error decays with N.
  (ii)  BvM-regime family sufficiency: flow and Gaussian-NPE track each other.
  (iii) Finite-N amortisation residual: both curves stay above the
        oracle Gaussian BvM reference line at feasible N.

Deliberately does NOT overclaim. The oracle line is a BvM reference, not a
lower bound on attainable KL.
"""
from __future__ import annotations

import argparse
import math
import pickle as pkl
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RES_DIR = REPO_ROOT / "res" / "gnk"
PLOTS_DIR = REPO_ROOT / "notebooks" / "plots"

sys.path.insert(0, str(REPO_ROOT))
from npe_convergence.metrics import kullback_leibler  # noqa: E402

N_METRIC = 2000
FIT_SIZE = 5000
BUDGET_LABELS = ["n", "n log n", "n^1.5", "n^2"]
COLORS = {"flow": "tab:blue", "gaussian": "tab:orange"}
LABELS = {"flow": "flow-NPE", "gaussian": "Gaussian-NPE"}


def n_sims_grid(n_obs: int) -> list[int]:
    return [
        n_obs,
        int(n_obs * math.log(n_obs)),
        int(n_obs ** 1.5),
        n_obs ** 2,
    ]


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


def load_kl(flavor: str, n_obs: int, n_sims: int, seed: int) -> float | None:
    subdir = "npe" if flavor == "flow" else "gaussian_npe"
    path = (
        RES_DIR
        / f"{subdir}_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}"
        / "kl.txt"
    )
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except ValueError:
        return None


def compute_kl_oracle(nuts: np.ndarray, seed: int) -> float:
    mu = nuts[:FIT_SIZE].mean(axis=0)
    Sigma = np.cov(nuts[:FIT_SIZE], rowvar=False)
    rng = np.random.default_rng(seed)
    if len(nuts) >= FIT_SIZE + N_METRIC:
        held_out = nuts[FIT_SIZE:FIT_SIZE + N_METRIC]
    else:
        idx = rng.permutation(len(nuts))[:N_METRIC]
        held_out = nuts[idx]
    oracle_samples = rng.multivariate_normal(mu, Sigma, size=N_METRIC)
    return float(kullback_leibler(held_out, oracle_samples))


def aggregate_block(n_obs: int, seed_range: range):
    """Return long-format dataframe of kl values + oracle per seed."""
    rows = []
    oracle_by_seed: dict[int, float] = {}
    for seed in seed_range:
        nuts = load_nuts(n_obs, seed)
        if nuts is None:
            continue
        oracle_by_seed[seed] = compute_kl_oracle(nuts, seed)
        for n_sims, label in zip(n_sims_grid(n_obs), BUDGET_LABELS):
            for flavor in ("flow", "gaussian"):
                kl = load_kl(flavor, n_obs, n_sims, seed)
                if kl is None or not np.isfinite(kl):
                    continue
                rows.append({
                    "n_obs": n_obs,
                    "n_sims": n_sims,
                    "budget_label": label,
                    "flavor": flavor,
                    "seed": seed,
                    "kl": kl,
                })
    return pd.DataFrame(rows), oracle_by_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,100")
    parser.add_argument("--n-obs-values", type=str, default="1000,5000")
    args = parser.parse_args()

    s0, s1 = map(int, args.seeds.split(","))
    seed_range = range(s0, s1 + 1)
    n_obs_values = [int(x) for x in args.n_obs_values.split(",")]

    fig, axes = plt.subplots(
        1, len(n_obs_values), figsize=(6.5 * len(n_obs_values), 5), sharey=False,
    )
    if len(n_obs_values) == 1:
        axes = [axes]

    summary_rows = []

    for ax, n_obs in zip(axes, n_obs_values):
        print(f"\n--- n_obs = {n_obs} ---")
        df, oracle_by_seed = aggregate_block(n_obs, seed_range)
        if df.empty:
            ax.set_title(f"n = {n_obs}: no data")
            continue
        oracle_arr = np.array(list(oracle_by_seed.values()))
        oracle_med = float(np.median(oracle_arr))
        oracle_q25 = float(np.percentile(oracle_arr, 25))
        oracle_q75 = float(np.percentile(oracle_arr, 75))
        print(f"  Oracle KL across {len(oracle_arr)} seeds: median={oracle_med:.4f}, IQR=[{oracle_q25:.4f}, {oracle_q75:.4f}]")

        for flavor in ("flow", "gaussian"):
            sub = df[df.flavor == flavor]
            if sub.empty:
                continue
            grouped = sub.groupby("n_sims")
            meds = grouped.kl.median().sort_index()
            q25 = grouped.kl.quantile(0.25).sort_index()
            q75 = grouped.kl.quantile(0.75).sort_index()
            counts = grouped.kl.count().sort_index()

            ns = meds.index.values
            ax.plot(ns, meds.values, marker="o", color=COLORS[flavor],
                    label=f"{LABELS[flavor]} (median over {counts.median():.0f} seeds)")
            ax.fill_between(ns, q25.values, q75.values, color=COLORS[flavor], alpha=0.18)

            for n in ns:
                summary_rows.append({
                    "n_obs": n_obs,
                    "n_sims": int(n),
                    "flavor": flavor,
                    "kl_median": float(meds[n]),
                    "kl_q25": float(q25[n]),
                    "kl_q75": float(q75[n]),
                    "n_seeds": int(counts[n]),
                    "oracle_median": oracle_med,
                })
                print(f"    N={n:>10d}  {flavor:>9s}: KL med={meds[n]:6.3f}  IQR=[{q25[n]:6.3f}, {q75[n]:6.3f}]  (n_seeds={counts[n]})")

        ax.axhline(oracle_med, color="black", linestyle="--", linewidth=1.3,
                   label=f"oracle Gaussian (BvM reference, med={oracle_med:.3f})")
        x_min, x_max = df.n_sims.min(), df.n_sims.max()
        ax.fill_between([x_min, x_max], [oracle_q25] * 2, [oracle_q75] * 2,
                        color="black", alpha=0.08)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$N$ (simulation budget)")
        ax.set_ylabel(r"$\widehat{\mathrm{KL}}(\Pi_{\mathrm{NUTS}}\,\|\,Q)$")
        ax.set_title(f"$n = {n_obs}$")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.2)

    fig.suptitle(
        "GNK: NPE-to-target KL vs simulation budget (aggregated across seeds)",
        y=1.02,
    )
    fig.tight_layout()
    out_pdf = PLOTS_DIR / "gnk_kl_vs_n_theory.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_pdf}")

    summary_csv = PLOTS_DIR / "gnk_kl_vs_n_theory.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
