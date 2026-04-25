#!/usr/bin/env python
"""Plot the GNK coordinate-aware Gaussian-NPE KL decomposition."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "notebooks" / "plots" / "gnk_u_space_kl_decomp_20260425_per_seed.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "plots"


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


def median_q25_q75(grouped: pd.core.groupby.generic.DataFrameGroupBy, column: str) -> pd.DataFrame:
    return grouped[column].agg(
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    )


def plot_delta_components(
    df: pd.DataFrame,
    out_path: Path,
    min_N_over_n: float,
    exclude_N_equals_n: bool,
    min_seeds: int,
) -> pd.DataFrame:
    if exclude_N_equals_n:
        plot_df = df[df["N"] > df["n"]].copy()
        filter_text = r"; plotted rows have $N > n$"
    else:
        plot_df = df[df["N"] / df["n"] >= min_N_over_n].copy()
        filter_text = "" if min_N_over_n <= 1.0 else rf"; plotted rows have $N/n \geq {min_N_over_n:g}$"
    if plot_df.empty:
        raise SystemExit("No rows remain after applying the Delta component plot filter")
    grouped = (
        plot_df.groupby(["n", "N", "scaled_budget"], as_index=False)
        .agg(
            Delta_u_mean=("Delta_u_mean", "median"),
            Delta_u_cov=("Delta_u_cov", "median"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["scaled_budget", "n", "N"])
        .reset_index(drop=True)
    )
    if min_seeds > 1:
        grouped = grouped[grouped["n_seeds"] >= min_seeds].reset_index(drop=True)
        plot_df = plot_df.merge(grouped[["n", "N"]], on=["n", "N"], how="inner")
    if grouped.empty:
        raise SystemExit("No grouped rows remain after applying the seed-count filter")
    labels = [
        f"{row.scaled_budget:.3g}\n(n={int(row.n)})"
        for row in grouped.itertuples(index=False)
    ]
    x = np.arange(len(grouped))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(9, 0.52 * len(grouped)), 5.3))
    ax.bar(
        x - width / 2,
        grouped["Delta_u_mean"],
        width=width,
        color="#4C78A8",
        label=r"$\Delta_{\mathrm{mean},u}$",
    )
    ax.bar(
        x + width / 2,
        grouped["Delta_u_cov"],
        width=width,
        color="#F58518",
        label=r"$\Delta_{\mathrm{cov},u}$",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    seed_text = "" if min_seeds <= 1 else rf"; groups have at least {min_seeds} seeds"
    ax.set_xlabel(r"scaled budget $N/(d_{\mathrm{total}}^2 n)$; u-space decomposition groups" + filter_text + seed_text)
    ax.set_ylabel("u-space KL component (nats)")
    ax.set_title("GNK Gaussian-NPE native u-space KL decomposition")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def plot_coord_offset(df: pd.DataFrame, out_path: Path) -> None:
    oracle = df.drop_duplicates(["n", "seed"])
    grouped = median_q25_q75(oracle.groupby("n"), "coord_offset").reset_index()

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = grouped["n"].to_numpy(dtype=float)
    med = grouped["median"].to_numpy(dtype=float)
    q25 = grouped["q25"].to_numpy(dtype=float)
    q75 = grouped["q75"].to_numpy(dtype=float)
    yerr = np.vstack([med - q25, q75 - med])
    ax.errorbar(
        x,
        med,
        yerr=yerr,
        marker="o",
        color="#222222",
        ecolor="#777777",
        elinewidth=1.2,
        capsize=3,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$n$ (observations; theta-space posterior projected to u-space)")
    ax.set_ylabel(r"coordinate offset $K_u^* - K_\theta^*$ (u-space minus theta-space, nats)")
    ax.set_title("GNK coordinate-projection offset by n")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default="gnk_u_space_kl_decomp_20260425")
    parser.add_argument(
        "--min-N-over-n",
        type=float,
        default=1.0,
        help="Filter Delta component plot to rows with N/n at least this value unless --exclude-N-equals-n is set; coord_offset plot is unaffected.",
    )
    parser.add_argument(
        "--exclude-N-equals-n",
        action="store_true",
        help="Filter Delta component plot to rows with N > n; raw input CSV and coord_offset plot are unaffected.",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=1,
        help="Minimum seed count per (n,N) group for the Delta component plot; coord_offset plot is unaffected.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {
        "n",
        "N",
        "seed",
        "scaled_budget",
        "coord_offset",
        "Delta_u_mean",
        "Delta_u_cov",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input CSV missing required columns: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    delta_path = args.output_dir / f"{args.output_prefix}_delta_u_mean_cov.pdf"
    coord_path = args.output_dir / f"{args.output_prefix}_coord_offset_vs_n.pdf"
    plotted_df = plot_delta_components(
        df,
        delta_path,
        min_N_over_n=args.min_N_over_n,
        exclude_N_equals_n=args.exclude_N_equals_n,
        min_seeds=args.min_seeds,
    )
    plot_coord_offset(df, coord_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "script": rel(Path(__file__)),
        "input_csv": rel(args.input_csv),
        "outputs": [rel(delta_path), rel(coord_path)],
        "delta_plot_min_N_over_n": args.min_N_over_n,
        "delta_plot_exclude_N_equals_n": bool(args.exclude_N_equals_n),
        "delta_plot_min_seeds": int(args.min_seeds),
        "delta_plot_groups": int(plotted_df[["n", "N"]].drop_duplicates().shape[0]),
        "delta_plot_rows": int(len(plotted_df)),
        "input_rows": int(len(df)),
        "notes": [
            "Delta components are native u-space Gaussian-NPE approximation error.",
            "Coordinate offset is K_u^* - K_theta^*, explicitly u-space minus theta-space.",
            "Axis labels include theta-space/u-space where relevant.",
        ],
    }
    meta_path = args.output_dir / f"{args.output_prefix}_plot_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Wrote {delta_path}")
    print(f"Wrote {coord_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
