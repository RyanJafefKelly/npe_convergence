#!/usr/bin/env python
"""Scaled-budget diagnostic: Delta_N vs N/(d^2 n).

Loads the aggregated CSV produced by kl_vs_n_theory_plot.py and plots:
  (a) Delta_N = kl_median - oracle_median vs raw N, per n (existing story).
  (b) Delta_N vs scaled budget N/(d^2 n), all n overlaid.

If (b) shows curves collapsing onto a common trajectory, the empirical decay
is consistent with Corollary BvM's N/(d^2 n) rate. If curves do not collapse,
there are finite-n constants that the asymptotic rate does not capture.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "notebooks" / "plots"

D_S = 7
D_THETA = 4
D_TOTAL = D_S + D_THETA  # 11
D2 = D_TOTAL ** 2        # 121


def main():
    csv_path = PLOTS_DIR / "gnk_kl_vs_n_theory.csv"
    df = pd.read_csv(csv_path)
    df["delta_N"] = df["kl_median"] - df["oracle_median"]
    df["scaled_budget"] = df["n_sims"] / (D2 * df["n_obs"])

    # Display the consolidated table
    print("=== All cached (n, N) cells, sorted by N/(d^2 n) ===")
    display_df = df[["n_obs", "n_sims", "flavor", "kl_median", "oracle_median",
                     "delta_N", "scaled_budget", "n_seeds"]].sort_values("scaled_budget")
    print(display_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel A: Delta_N vs raw N, per n ---
    ax = axes[0]
    n_values = sorted(df.n_obs.unique())
    cmap = plt.colormaps["viridis"]
    colors = {n: cmap(i / max(1, len(n_values) - 1)) for i, n in enumerate(n_values)}

    for n in n_values:
        for flavor, marker, linestyle in (("flow", "o", "-"), ("gaussian", "s", "--")):
            sub = df[(df.n_obs == n) & (df.flavor == flavor)].sort_values("n_sims")
            if sub.empty:
                continue
            label = f"n={n}, {flavor}"
            ax.plot(sub.n_sims, sub.delta_N, marker=marker, linestyle=linestyle,
                    color=colors[n], label=label, markersize=6, lw=1.3, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (simulation budget)")
    ax.set_ylabel(r"$\Delta_N = \widehat{\mathrm{KL}}(\Pi\,\|\,\widehat Q_N) - \widehat{\mathrm{KL}}(\Pi\,\|\,G^*)$  (nats)")
    ax.set_title("Panel A — Δ_N vs raw N  (per n)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(True, which="both", alpha=0.2)

    # --- Panel B: Delta_N vs N/(d^2 n), all n overlaid ---
    ax = axes[1]
    for n in n_values:
        for flavor, marker, linestyle in (("flow", "o", "-"), ("gaussian", "s", "--")):
            sub = df[(df.n_obs == n) & (df.flavor == flavor)].sort_values("scaled_budget")
            if sub.empty:
                continue
            label = f"n={n}, {flavor}"
            ax.plot(sub.scaled_budget, sub.delta_N, marker=marker, linestyle=linestyle,
                    color=colors[n], label=label, markersize=6, lw=1.3, alpha=0.85)
    # Corollary BvM threshold: N/(d^2 n) = 1 means N = d^2 n (the stated sufficient scale).
    ax.axvline(1.0, color="red", linestyle=":", lw=1.2, alpha=0.8,
               label=r"$N = d^2 n$ threshold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N / (d^2 n)$  (scaled budget; $d = d_s + d_\theta = 11$)")
    ax.set_ylabel(r"$\Delta_N$ (nats)")
    ax.set_title("Panel B — Δ_N vs scaled budget  (all n overlaid)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(True, which="both", alpha=0.2)

    fig.suptitle("GNK: NPE-to-BvM-oracle gap vs simulation budget (median across seeds)",
                 y=1.02)
    fig.tight_layout()
    out_pdf = PLOTS_DIR / "gnk_delta_N_vs_scaled_budget.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_pdf}")


if __name__ == "__main__":
    main()
