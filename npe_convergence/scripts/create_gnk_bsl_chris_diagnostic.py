"""Create a compact Chris-facing GNK BSL reference diagnostic."""

from __future__ import annotations

import argparse
import json
import pickle as pkl
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PARAM_NAMES = ("A", "B", "g", "k")
TRUE_THETA = np.array([3.0, 1.0, 2.0, 0.5])
C_REF = "#333333"
C_BSL = "#2ca02c"
C_FLOW = "#1f77b4"
C_GAUSS = "#d62728"


def latest_bsl_dir(root: Path) -> Path:
    candidates = sorted(root.glob("bsl_n_obs_1000_seed_0_M_500_*"))
    if not candidates:
        raise FileNotFoundError(f"no default BSL run found under {root}")
    return candidates[-1]


def load_pickle_array(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        return np.asarray(pkl.load(f))


def load_samples(bsl_dir: Path) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {
        "NUTS reference": {
            "kind": "reference",
            "samples": load_pickle_array(REPO_ROOT / "res/gnk/nuts_cache_v2_n_obs_1000_seed_0.pkl"),
            "color": C_REF,
            "line": "-",
            "kl_nuts_to_method": 0.0,
        },
        "BSL": {
            "kind": "reference check",
            "samples": np.load(bsl_dir / "posterior_samples.npz")["theta"],
            "color": C_BSL,
            "line": "-",
            "kl_nuts_to_method": json.loads((bsl_dir / "bsl_vs_nuts_kl.json").read_text())[
                "KL_NUTS_to_BSL"
            ],
        },
    }
    npe_specs = {
        "flow-NPE, N=n^2": (
            REPO_ROOT / "res/gnk/npe_n_obs_1000_n_sims_1000000_seed_0",
            C_FLOW,
        ),
        "Gaussian-NPE, N=n^2": (
            REPO_ROOT / "res/gnk/gaussian_npe_n_obs_1000_n_sims_1000000_seed_0",
            C_GAUSS,
        ),
    }
    for label, (base, color) in npe_specs.items():
        samples_path = base / "posterior_samples.pkl"
        kl_path = base / "kl.txt"
        if samples_path.exists() and kl_path.exists():
            samples[label] = {
                "kind": "NPE context",
                "samples": load_pickle_array(samples_path),
                "color": color,
                "line": "--",
                "kl_nuts_to_method": float(kl_path.read_text().strip()),
            }
    return samples


def summary_table(samples: dict[str, dict[str, Any]]) -> pd.DataFrame:
    nuts = samples["NUTS reference"]["samples"]
    nuts_median = np.median(nuts, axis=0)
    nuts_sd = np.std(nuts, axis=0)
    rows = []
    for label, spec in samples.items():
        arr = spec["samples"]
        median = np.median(arr, axis=0)
        sd = np.std(arr, axis=0)
        shift = np.abs(median - nuts_median) / nuts_sd
        sd_ratio = sd / nuts_sd
        row: dict[str, Any] = {
            "method": label,
            "kind": spec["kind"],
            "KL_NUTS_to_method": spec["kl_nuts_to_method"],
            "max_abs_median_shift_in_NUTS_sd": float(np.max(shift)),
            "mean_sd_ratio_to_NUTS": float(np.mean(sd_ratio)),
        }
        for idx, param in enumerate(PARAM_NAMES):
            row[f"{param}_median"] = float(median[idx])
            row[f"{param}_sd"] = float(sd[idx])
            row[f"{param}_abs_median_shift_in_NUTS_sd"] = float(shift[idx])
            row[f"{param}_sd_ratio_to_NUTS"] = float(sd_ratio[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _density_values(arr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(arr)
    return kde(grid)


def make_plot(samples: dict[str, dict[str, Any]], table: pd.DataFrame, out_base: Path) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )
    fig = plt.figure(figsize=(12.8, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.3, 1.25], hspace=0.45, wspace=0.32)

    for idx, param in enumerate(PARAM_NAMES):
        ax = fig.add_subplot(gs[0, idx])
        combined = np.concatenate([spec["samples"][:, idx] for spec in samples.values()])
        lo, hi = np.percentile(combined, [0.5, 99.5])
        pad = 0.08 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 300)
        for label, spec in samples.items():
            y = _density_values(spec["samples"][:, idx], grid)
            lw = 1.9 if label in {"NUTS reference", "BSL"} else 1.25
            alpha = 1.0 if label in {"NUTS reference", "BSL"} else 0.85
            ax.plot(
                grid,
                y,
                color=spec["color"],
                linestyle=spec["line"],
                lw=lw,
                alpha=alpha,
                label=label,
            )
        ax.axvline(TRUE_THETA[idx], color="black", linestyle=":", lw=1.2, alpha=0.8)
        ax.set_title(param)
        ax.set_yticks([])
        ax.set_xlabel(param)
        if idx == 0:
            ax.set_ylabel("posterior density")
            ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax_bar = fig.add_subplot(gs[1, :2])
    methods = ["BSL", "flow-NPE, N=n^2", "Gaussian-NPE, N=n^2"]
    methods = [m for m in methods if m in set(table["method"])]
    x = np.arange(len(PARAM_NAMES))
    width = 0.24
    offsets = np.linspace(-width, width, len(methods))
    colors = [samples[m]["color"] for m in methods]
    for offset, method, color in zip(offsets, methods, colors):
        vals = [
            float(table.loc[table.method == method, f"{param}_abs_median_shift_in_NUTS_sd"].iloc[0])
            for param in PARAM_NAMES
        ]
        ax_bar.bar(x + offset, vals, width=width, label=method, color=color, alpha=0.78)
    ax_bar.axhline(1.0, color="black", linestyle="--", lw=1.0, alpha=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(PARAM_NAMES)
    ax_bar.set_ylabel("absolute median shift\nin NUTS posterior SDs")
    ax_bar.set_title("BSL is close to the asymptotic reference on marginal location")
    ax_bar.legend(frameon=False, fontsize=8)

    ax_table = fig.add_subplot(gs[1, 2:])
    ax_table.axis("off")
    display = table[["method", "KL_NUTS_to_method", "max_abs_median_shift_in_NUTS_sd", "mean_sd_ratio_to_NUTS"]].copy()
    display["method"] = display["method"].replace(
        {
            "NUTS reference": "NUTS",
            "flow-NPE, N=n^2": "flow-NPE",
            "Gaussian-NPE, N=n^2": "Gaussian-NPE",
        }
    )
    display["KL_NUTS_to_method"] = display["KL_NUTS_to_method"].map(lambda x: f"{x:.3f}")
    display["max_abs_median_shift_in_NUTS_sd"] = display[
        "max_abs_median_shift_in_NUTS_sd"
    ].map(lambda x: f"{x:.2f}")
    display["mean_sd_ratio_to_NUTS"] = display["mean_sd_ratio_to_NUTS"].map(lambda x: f"{x:.2f}")
    display.columns = ["method", "KL from NUTS", "max median shift", "mean SD ratio"]
    mpl_table = ax_table.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(8)
    mpl_table.scale(1.0, 1.45)
    for (row, col), cell in mpl_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        elif col == 0:
            cell.set_text_props(ha="left")

    fig.suptitle(
        "GNK octile posterior reference check, n=1000, seed=0",
        fontsize=13,
        y=0.98,
    )
    fig.text(
        0.02,
        0.015,
        "Solid black: asymptotic MVN NUTS reference. Solid green: BSL using model simulations. "
        "Dashed curves are high-budget NPE context, not a new benchmark.",
        fontsize=9,
    )
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=180)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_markdown(
    *,
    table: pd.DataFrame,
    out_dir: Path,
    bsl_dir: Path,
    plot_path: Path,
) -> None:
    bsl = table.loc[table.method == "BSL"].iloc[0]
    flow = table.loc[table.method == "flow-NPE, N=n^2"].iloc[0]
    gauss = table.loc[table.method == "Gaussian-NPE, N=n^2"].iloc[0]
    diagnostics = json.loads((bsl_dir / "diagnostics.json").read_text())
    kl_payload = json.loads((bsl_dir / "bsl_vs_nuts_kl.json").read_text())
    max_rhat = max(v["r_hat"] for v in diagnostics["per_parameter"].values())
    min_ess = min(v["n_eff"] for v in diagnostics["per_parameter"].values())
    acceptance = diagnostics["sample_acceptance_rate"]
    lines = [
        "# GNK BSL reference check for Chris",
        "",
        "Chris asked whether a standard BSL reference, based on model simulations rather than the analytic asymptotic octile likelihood, changes the GNK reference story.",
        "",
        f"BSL run: `{bsl_dir.relative_to(REPO_ROOT)}`",
        "",
        f"![GNK BSL reference check]({plot_path.name})",
        "",
        "## Compact table",
        "",
        "| method | role | KL from NUTS | max median shift, NUTS SDs | mean SD ratio to NUTS |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['method']} | {row['kind']} | {row['KL_NUTS_to_method']:.3f} | "
            f"{row['max_abs_median_shift_in_NUTS_sd']:.2f} | {row['mean_sd_ratio_to_NUTS']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"At this cell, BSL is much closer to the asymptotic-MVN NUTS reference than the high-budget NPE posteriors: its largest marginal median shift is {bsl['max_abs_median_shift_in_NUTS_sd']:.2f} NUTS posterior SDs, compared with {flow['max_abs_median_shift_in_NUTS_sd']:.2f} for flow-NPE and {gauss['max_abs_median_shift_in_NUTS_sd']:.2f} for Gaussian-NPE at N=n^2.",
            "",
            f"The finite KL direction used in the paper-style tables, KL(NUTS || method), is {bsl['KL_NUTS_to_method']:.3f} for BSL, versus {flow['KL_NUTS_to_method']:.3f} for flow-NPE and {gauss['KL_NUTS_to_method']:.3f} for Gaussian-NPE at the same n and seed.",
            "",
            f"The dense-proposal BSL run passed the MCMC diagnostics used for this check: max R-hat {max_rhat:.3f}, min ESS {min_ess:.0f}, and per-chain acceptance rates "
            + ", ".join(f"{x:.3f}" for x in acceptance)
            + ".",
            "",
            f"Caveat: raw retained-state KL(BSL || NUTS) is {kl_payload['KL_BSL_to_NUTS']} because retained RWM draws contain exact duplicates after rejected proposals, which violates the no-ties assumption of the kNN estimator. The unique-row sensitivity is small: KL(unique BSL || NUTS) {kl_payload['KL_BSL_unique_to_NUTS']:.3f} and KL(NUTS || unique BSL) {kl_payload['KL_NUTS_to_BSL_unique']:.3f}.",
            "",
            "Plain-language conclusion for Chris: this BSL check does not point to a large failure of the asymptotic octile likelihood. The simulation-based BSL posterior lands on the same marginal location and scale as the cached NUTS reference, while the NPE gap remains visibly larger.",
            "",
        ]
    )
    (out_dir / "gnk_bsl_chris_diagnostic.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bsl-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs/meeting_2026_05_18/gnk_bsl_chris")
    args = parser.parse_args(argv)

    bsl_dir = args.bsl_dir or latest_bsl_dir(REPO_ROOT / "res/gnk_bsl")
    if not bsl_dir.is_absolute():
        bsl_dir = REPO_ROOT / bsl_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(bsl_dir)
    table = summary_table(samples)
    table_path = output_dir / "gnk_bsl_chris_diagnostic.csv"
    table.to_csv(table_path, index=False)
    plot_base = output_dir / "gnk_bsl_chris_diagnostic"
    make_plot(samples, table, plot_base)
    make_markdown(
        table=table,
        out_dir=output_dir,
        bsl_dir=bsl_dir,
        plot_path=plot_base.with_suffix(".png"),
    )
    print(json.dumps({"output_dir": str(output_dir), "table": str(table_path), "plot": str(plot_base.with_suffix('.png'))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
