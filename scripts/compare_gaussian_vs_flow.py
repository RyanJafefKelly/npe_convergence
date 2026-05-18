#!/usr/bin/env python
"""Diagnostic: Gaussian-NPE vs flow-NPE vs NUTS for the g-and-k model.

Produces two artifacts in notebooks/plots/:

- gnk_gaussian_vs_flow_diagnostic.pdf
    3-panel figure. Panel A: NUTS vs oracle moment-matched Gaussian in (g, k).
    Panel B: NUTS + flow-NPE + Gaussian-NPE + oracle in (g, k). Panel C: 1D k
    marginal for all four distributions.

- gnk_gaussian_vs_flow_diagnostics.csv
    Per (n_obs, n_sims) row: kl_flow, kl_gaussian_npe, kl_oracle,
    delta_family = kl_oracle - kl_flow, delta_train = kl_gaussian_npe -
    kl_oracle, plus off-support mass fractions for each method.

All three KL values use the same Perez-Cruz estimator (metrics.kullback_leibler)
with 2000 samples per side, matching the convention in run_gnk.py:230.
"""
from __future__ import annotations

import argparse
import math
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, gaussian_kde, multivariate_normal, norm

from npe_convergence.metrics import kullback_leibler

REPO_ROOT = Path(__file__).resolve().parent.parent
RES_DIR = REPO_ROOT / "res" / "gnk"
PLOTS_DIR = REPO_ROOT / "notebooks" / "plots"

TRUE_PARAMS = {"A": 3.0, "B": 1.0, "g": 2.0, "k": 0.5}
PARAM_NAMES = ["A", "B", "g", "k"]
N_METRIC = 2000
FIT_SIZE = 5000

BUDGET_LABELS = ["n", "n log n", "n^1.5", "n^2"]
COLORS = {"NUTS": "#555555", "flow-NPE": "tab:blue", "Gaussian-NPE": "tab:orange"}


def n_sims_grid(n_obs: int) -> list[int]:
    return [
        n_obs,
        int(n_obs * math.log(n_obs)),
        int(n_obs ** 1.5),
        n_obs ** 2,
    ]


def load_nuts(n_obs: int, seed: int) -> np.ndarray:
    for prefix in ("nuts_cache_v2_n_obs", "nuts_cache_v2_flow_n_obs"):
        path = RES_DIR / f"{prefix}_{n_obs}_seed_{seed}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return np.asarray(pkl.load(f))
    raise FileNotFoundError(f"No NUTS cache for (n_obs={n_obs}, seed={seed})")


def load_npe(flavor: str, n_obs: int, n_sims: int, seed: int) -> np.ndarray | None:
    subdir = "npe" if flavor == "flow" else "gaussian_npe"
    path = (
        RES_DIR
        / f"{subdir}_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}"
        / "posterior_samples.pkl"
    )
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return np.asarray(pkl.load(f))


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


def fit_oracle(nuts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fit = nuts[:FIT_SIZE]
    return fit.mean(axis=0), np.cov(fit, rowvar=False)


def compute_kl_oracle(nuts: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if len(nuts) >= FIT_SIZE + N_METRIC:
        held_out = nuts[FIT_SIZE:FIT_SIZE + N_METRIC]
    else:
        idx = rng.permutation(len(nuts))[:N_METRIC]
        held_out = nuts[idx]
    oracle_samples = rng.multivariate_normal(mu, Sigma, size=N_METRIC)
    return float(kullback_leibler(held_out, oracle_samples))


def oos_fraction(samples: np.ndarray | None, lo: float = 0.0, hi: float = 10.0) -> float | None:
    if samples is None:
        return None
    mask = np.any((samples < lo) | (samples > hi), axis=1)
    return float(mask.mean())


def _hdr_sample_levels(kde: gaussian_kde, samples: np.ndarray, masses: list[float]) -> list[float]:
    """Sample-based HDR density thresholds: level at rank (m * M) of sorted densities."""
    d = kde(samples.T)
    d_sorted = np.sort(d)[::-1]
    M = len(d_sorted)
    levels = []
    for m in masses:
        idx = min(max(int(m * M) - 1, 0), M - 1)
        levels.append(float(d_sorted[idx]))
    return sorted(levels)


def _mvn_hdr_levels(Sigma_gk: np.ndarray, masses: list[float]) -> list[float]:
    """Analytic HDR density thresholds for a 2D MVN with covariance Sigma_gk."""
    det_S = float(np.linalg.det(Sigma_gk))
    norm_const = 1.0 / (2.0 * np.pi * math.sqrt(det_S))
    levels = [norm_const * math.exp(-0.5 * chi2(2).ppf(m)) for m in masses]
    return sorted(levels)


def _gk_axis_limits(samples_dict: dict, mu_gk: np.ndarray, Sigma_gk: np.ndarray) -> tuple:
    g_all = np.concatenate([s[:, 2] for s in samples_dict.values()])
    k_all = np.concatenate([s[:, 3] for s in samples_dict.values()])
    g_min, g_max = np.percentile(g_all, [0.5, 99.5])
    k_min, k_max = np.percentile(k_all, [0.5, 99.5])
    sig_g = math.sqrt(Sigma_gk[0, 0])
    sig_k = math.sqrt(Sigma_gk[1, 1])
    g_min = min(g_min, mu_gk[0] - 3 * sig_g)
    g_max = max(g_max, mu_gk[0] + 3 * sig_g)
    k_min = min(k_min, mu_gk[1] - 3 * sig_k)
    k_max = max(k_max, mu_gk[1] + 3 * sig_k)
    return g_min, g_max, k_min, k_max


def plot_panel_2d(
    ax,
    samples_dict: dict,
    mu_gk: np.ndarray,
    Sigma_gk: np.ndarray,
    title: str,
    show_nuts_scatter: bool = False,
    xlim=None,
    ylim=None,
):
    if xlim is None or ylim is None:
        g_min, g_max, k_min, k_max = _gk_axis_limits(samples_dict, mu_gk, Sigma_gk)
    else:
        g_min, g_max = xlim
        k_min, k_max = ylim
    grid_g, grid_k = np.meshgrid(
        np.linspace(g_min, g_max, 150),
        np.linspace(k_min, k_max, 150),
    )
    grid_pts = np.vstack([grid_g.ravel(), grid_k.ravel()])

    for label, samples in samples_dict.items():
        gk = samples[:, [2, 3]]
        if label == "NUTS" and show_nuts_scatter:
            ax.scatter(gk[:, 0], gk[:, 1], s=2, alpha=0.08, color="grey", zorder=1)
        kde = gaussian_kde(gk.T)
        density = kde(grid_pts).reshape(grid_g.shape)
        levels = _hdr_sample_levels(kde, gk, [0.9, 0.5])
        ax.contour(
            grid_g, grid_k, density,
            levels=levels, colors=[COLORS[label]], linewidths=1.5, zorder=3,
        )

    oracle_grid = np.dstack([grid_g, grid_k])
    oracle_density = multivariate_normal(mu_gk, Sigma_gk).pdf(oracle_grid)
    oracle_levels = _mvn_hdr_levels(Sigma_gk, [0.9, 0.5])
    ax.contour(
        grid_g, grid_k, oracle_density,
        levels=oracle_levels, colors="black",
        linestyles="--", linewidths=1.2, zorder=4,
    )

    ax.plot(TRUE_PARAMS["g"], TRUE_PARAMS["k"], "x", color="red", markersize=10, mew=2, zorder=6)
    ax.set_xlim(g_min, g_max)
    ax.set_ylim(k_min, k_max)
    ax.set_xlabel("g")
    ax.set_ylabel("k")
    ax.set_title(title)


def plot_panel_1d_k(ax, samples_dict: dict, mu_k: float, sigma_k: float):
    k_all = np.concatenate([s[:, 3] for s in samples_dict.values()])
    k_min = min(float(k_all.min()), mu_k - 3 * sigma_k, 0.0) - 0.1
    k_max = max(float(k_all.max()), mu_k + 3 * sigma_k) + 0.1
    grid = np.linspace(k_min, k_max, 500)
    for label, samples in samples_dict.items():
        kde = gaussian_kde(samples[:, 3])
        ax.plot(grid, kde(grid), color=COLORS[label], label=label, lw=1.5)
    ax.plot(
        grid, norm(mu_k, sigma_k).pdf(grid),
        color="black", ls="--", lw=1.2, label="oracle Gaussian",
    )
    ax.axvline(TRUE_PARAMS["k"], color="red", lw=0.8, alpha=0.6)
    ax.axvline(0, color="grey", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel("k")
    ax.set_ylabel("density")
    ax.set_title("Panel C — $k$ marginal (dotted: prior boundary $k=0$)")
    ax.legend(loc="best", fontsize=8)


def make_figure(n_obs: int, n_sims: int, seed: int, out_path: Path):
    nuts = load_nuts(n_obs, seed)
    flow = load_npe("flow", n_obs, n_sims, seed)
    gaussian = load_npe("gaussian", n_obs, n_sims, seed)
    if flow is None or gaussian is None:
        raise SystemExit(
            f"Missing NPE samples for n_obs={n_obs}, n_sims={n_sims}, seed={seed}. "
            f"flow={flow is not None}, gaussian={gaussian is not None}."
        )

    mu, Sigma = fit_oracle(nuts)
    mu_gk = mu[[2, 3]]
    Sigma_gk = Sigma[np.ix_([2, 3], [2, 3])]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))

    all_samples = {"NUTS": nuts, "flow-NPE": flow, "Gaussian-NPE": gaussian}
    g_min, g_max, k_min, k_max = _gk_axis_limits(all_samples, mu_gk, Sigma_gk)

    plot_panel_2d(
        axes[0], {"NUTS": nuts}, mu_gk, Sigma_gk,
        title="Panel A — target vs oracle Gaussian",
        show_nuts_scatter=True,
        xlim=(g_min, g_max), ylim=(k_min, k_max),
    )
    plot_panel_2d(
        axes[1], all_samples, mu_gk, Sigma_gk,
        title=f"Panel B — methods ($n={n_obs}$, $N={n_sims}$, seed={seed})",
        show_nuts_scatter=False,
        xlim=(g_min, g_max), ylim=(k_min, k_max),
    )
    plot_panel_1d_k(axes[2], all_samples, float(mu[3]), float(math.sqrt(Sigma[3, 3])))

    legend_handles = [
        plt.Line2D([], [], color=COLORS["NUTS"], lw=1.5, label="NUTS (truth)"),
        plt.Line2D([], [], color=COLORS["flow-NPE"], lw=1.5, label="flow-NPE"),
        plt.Line2D([], [], color=COLORS["Gaussian-NPE"], lw=1.5, label="Gaussian-NPE"),
        plt.Line2D([], [], color="black", ls="--", lw=1.2, label="oracle Gaussian (moment-matched to NUTS)"),
        plt.Line2D([], [], color="red", marker="x", lw=0, markersize=8, mew=2, label="true $(g, k)$"),
    ]
    axes[1].legend(handles=legend_handles, loc="best", fontsize=7)

    fig.suptitle(
        "GNK partial posterior: NUTS vs flow-NPE vs Gaussian-NPE",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _std_ratio(samples: np.ndarray | None, col: int, nuts_std: float) -> float | None:
    if samples is None:
        return None
    return float(samples[:, col].std() / nuts_std)


def build_table(seed: int, out_path: Path, n_obs_values=(1000, 5000)) -> pd.DataFrame:
    rows = []
    for n_obs in n_obs_values:
        try:
            nuts = load_nuts(n_obs, seed)
        except FileNotFoundError as e:
            print(f"Skipping n_obs={n_obs}: {e}")
            continue
        mu, Sigma = fit_oracle(nuts)
        kl_oracle = compute_kl_oracle(nuts, mu, Sigma, seed=seed)
        nuts_std_g = float(nuts[:, 2].std())
        nuts_std_k = float(nuts[:, 3].std())

        for n_sims, label in zip(n_sims_grid(n_obs), BUDGET_LABELS):
            kl_flow = load_kl("flow", n_obs, n_sims, seed)
            kl_gn = load_kl("gaussian", n_obs, n_sims, seed)
            flow_s = load_npe("flow", n_obs, n_sims, seed)
            gn_s = load_npe("gaussian", n_obs, n_sims, seed)

            delta_family = (
                kl_oracle - kl_flow if (kl_oracle is not None and kl_flow is not None) else None
            )
            delta_train = (
                kl_gn - kl_oracle if (kl_gn is not None and kl_oracle is not None) else None
            )

            rows.append({
                "n_obs": n_obs,
                "n_sims": n_sims,
                "budget_label": label,
                "kl_flow": kl_flow,
                "kl_gaussian_npe": kl_gn,
                "kl_oracle": kl_oracle,
                "delta_family": delta_family,
                "delta_train": delta_train,
                "sigma_g_flow_over_nuts": _std_ratio(flow_s, 2, nuts_std_g),
                "sigma_g_gnpe_over_nuts": _std_ratio(gn_s, 2, nuts_std_g),
                "sigma_k_flow_over_nuts": _std_ratio(flow_s, 3, nuts_std_k),
                "sigma_k_gnpe_over_nuts": _std_ratio(gn_s, 3, nuts_std_k),
                "oos_gnpe": oos_fraction(gn_s),
                "oos_flow": oos_fraction(flow_s),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--n-obs", type=int, default=1000,
                        help="n for the flagship figure (default: 1000)")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed for both figure and table (default: 1)")
    parser.add_argument("--n-sims", type=int, default=1_000_000,
                        help="N for the flagship figure (default: 1_000_000 = n^2 at n=1000)")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig_path = PLOTS_DIR / "gnk_gaussian_vs_flow_diagnostic.pdf"
    make_figure(args.n_obs, args.n_sims, args.seed, fig_path)

    csv_path = PLOTS_DIR / "gnk_gaussian_vs_flow_diagnostics.csv"
    df = build_table(args.seed, csv_path)
    print()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))


if __name__ == "__main__":
    main()
