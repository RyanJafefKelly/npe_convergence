#!/usr/bin/env python
"""Two supporting diagnostics for the GNK amortisation-residual story.

A. Summary-likelihood leakage (D^2). For each sample theta from NUTS,
   oracle Gaussian, flow-NPE, and Gaussian-NPE, compute
     D^2(theta) = (S_n - b(theta))^T Sigma_S(theta)^-1 (S_n - b(theta)),
   where b(theta) is the theoretical octile vector and Sigma_S(theta) is the
   asymptotic octile covariance at theta. If NPE assigns mass to low-summary-
   likelihood theta, its D^2 distribution is shifted right of NUTS's.

B. Transform-chain audit in training coordinates. Project all sample sets
   through u = (logit(theta / 10) - mu_eta) / sigma_eta and recompute
   sigma-ratios. If NPE overdispersion persists in u-space, the inverse
   transform is not the cause.

Uses seed=1 by default. Aggregates across seeds with --seeds-range if desired
for (A) only (transform audit is same story seed-by-seed).
"""
from __future__ import annotations

import argparse
import math
import pickle as pkl
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax.scipy.special import logit
from jax.scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parent.parent
RES_DIR = REPO_ROOT / "res" / "gnk"
PLOTS_DIR = REPO_ROOT / "notebooks" / "plots"

sys.path.insert(0, str(REPO_ROOT))
from npe_convergence.examples.gnk import (  # noqa: E402
    compute_covariance_matrix,
    gnk,
    ss_octile,
)

TRUE_PARAMS = np.array([3.0, 1.0, 2.0, 0.5])
PARAM_NAMES = ["A", "B", "g", "k"]
QUANTILES = np.linspace(0.125, 0.875, 7)
N_METRIC = 2000
FIT_SIZE = 5000


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


def fit_oracle_samples(nuts: np.ndarray, seed: int, size: int = 10_000) -> np.ndarray:
    mu = nuts[:FIT_SIZE].mean(axis=0)
    Sigma = np.cov(nuts[:FIT_SIZE], rowvar=False)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, Sigma, size=size)


def generate_x_obs(n_obs: int, seed: int) -> np.ndarray:
    """Reproduce the x_obs used in training for the given seed."""
    key = random.key(seed)
    z = random.normal(key, shape=(n_obs,))
    x = gnk(z, *TRUE_PARAMS)
    x = jnp.atleast_2d(x)
    x_obs = ss_octile(x)
    return np.asarray(jnp.squeeze(x_obs))


# ---------------- D^2 leakage ----------------

def _d2_single(theta: jnp.ndarray, S_n: jnp.ndarray, n_obs: int) -> jnp.ndarray:
    A, B, g, k = theta
    z = norm.ppf(jnp.asarray(QUANTILES))
    b_theta = gnk(z, A, B, g, k)
    Sigma = compute_covariance_matrix(A, B, g, k, jnp.asarray(QUANTILES), n_obs) + 1e-8 * jnp.eye(7)
    diff = S_n - b_theta
    return diff @ jnp.linalg.solve(Sigma, diff)


def compute_d2(samples: np.ndarray, S_n: np.ndarray, n_obs: int, max_samples: int = 2000) -> np.ndarray:
    if len(samples) > max_samples:
        idx = np.random.default_rng(0).choice(len(samples), size=max_samples, replace=False)
        samples = samples[idx]
    # Clip to prior support to avoid numerical problems in compute_covariance_matrix
    samples = np.clip(samples, 1e-4, 10 - 1e-4)
    S_n_j = jnp.asarray(S_n)
    batched = jax.vmap(lambda t: _d2_single(t, S_n_j, n_obs))
    d2_vals = np.asarray(batched(jnp.asarray(samples)))
    d2_vals = d2_vals[np.isfinite(d2_vals)]
    return d2_vals


# ---------------- Transform audit ----------------

def theta_to_u(theta_samples: np.ndarray, mu_eta: np.ndarray, sigma_eta: np.ndarray) -> np.ndarray:
    eps = 1e-6
    theta = np.clip(theta_samples, eps, 10 - eps)
    eta = np.log((theta / 10) / (1 - theta / 10))
    return (eta - mu_eta) / sigma_eta


def training_transform_constants(n_sims: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate thetas_bounded with same seed pattern as run_gnk.py, return (mu_eta, sigma_eta)."""
    import numpyro.distributions as dist  # local import; numpyro present per requirements
    # run_gnk.py does: key = random.key(seed); then random.split(key, ...) several times before
    # sampling thetas. Reproducing exactly requires matching the RNG consumption. For the purposes
    # of this audit (accuracy to ~0.1% at N=1e6), we approximate by taking a single seed-derived
    # subkey and noting that for large N the sample mean/std ~ population values.
    key = random.key(seed)
    tol = 1e-6
    # Match: several splits happen before thetas are drawn in run_gnk.py — we skip the exact
    # bookkeeping and just sample from the same distribution with a fixed RNG.
    thetas_bounded = dist.Uniform(0 + tol, 10 - tol).sample(key, (n_sims, 4))
    thetas_unbounded = logit(thetas_bounded / 10)
    mu_eta = np.asarray(thetas_unbounded.mean(axis=0))
    sigma_eta = np.asarray(thetas_unbounded.std(axis=0))
    return mu_eta, sigma_eta


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--n-sims", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--d2-samples", type=int, default=2000)
    args = parser.parse_args()

    nuts = load_nuts(args.n_obs, args.seed)
    flow = load_npe("flow", args.n_obs, args.n_sims, args.seed)
    gnpe = load_npe("gaussian", args.n_obs, args.n_sims, args.seed)
    if nuts is None or flow is None or gnpe is None:
        raise SystemExit("Missing cached artifacts")

    oracle = fit_oracle_samples(nuts, seed=args.seed)
    S_n = generate_x_obs(args.n_obs, args.seed)

    # ---- Part A: D^2 leakage ----
    print(f"\n=== A. Summary-likelihood leakage D^2 at seed={args.seed} (max {args.d2_samples} samples/method) ===")
    print(f"S_n = {S_n}")

    d2 = {}
    for name, samples in (("NUTS", nuts), ("oracle", oracle), ("flow-NPE", flow), ("Gaussian-NPE", gnpe)):
        d2[name] = compute_d2(samples, S_n, args.n_obs, max_samples=args.d2_samples)

    print(f"\n{'method':<15}{'median':>10}{'q25':>10}{'q75':>10}{'frac>14':>10}{'frac>20':>10}")
    print("-" * 65)
    from scipy.stats import chi2
    print(f"(chi^2_7 benchmark: median={chi2(7).median():.2f}, 95% = {chi2(7).ppf(0.95):.2f})")
    for name, arr in d2.items():
        med = float(np.median(arr))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        f14 = float(np.mean(arr > 14.07))  # chi^2_7 95% quantile
        f20 = float(np.mean(arr > 20))
        print(f"{name:<15}{med:>10.2f}{q25:>10.2f}{q75:>10.2f}{f14:>10.3f}{f20:>10.3f}")

    # D^2 histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, min(np.percentile(np.concatenate(list(d2.values())), 99), 60), 60)
    colors = {"NUTS": "#555555", "oracle": "black", "flow-NPE": "tab:blue", "Gaussian-NPE": "tab:orange"}
    for name, arr in d2.items():
        style = "--" if name == "oracle" else "-"
        hist, edges = np.histogram(arr, bins=bins, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(centres, hist, linestyle=style, color=colors[name], lw=1.5, label=name)
    # chi^2_7 reference
    x = np.linspace(0, bins.max(), 200)
    ax.plot(x, chi2(7).pdf(x), ":", color="red", lw=1, label=r"$\chi^2_7$ reference")
    ax.axvline(chi2(7).ppf(0.95), color="red", ls=":", alpha=0.5)
    ax.set_xlabel(r"$D^2(\theta) = (S_n - b(\theta))^\top \Sigma_S(\theta)^{-1} (S_n - b(\theta))$")
    ax.set_ylabel("density")
    ax.set_title(
        f"Summary-likelihood leakage at $S_n$ (n={args.n_obs}, N={args.n_sims}, seed={args.seed})"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    d2_pdf = PLOTS_DIR / f"gnk_d2_leakage_n_obs_{args.n_obs}_n_sims_{args.n_sims}_seed_{args.seed}.pdf"
    fig.savefig(d2_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {d2_pdf}")

    # ---- Part B: Transform audit ----
    print(f"\n=== B. Transform audit: sigma-ratios in training u-space vs theta-space ===")
    mu_eta, sigma_eta = training_transform_constants(args.n_sims, seed=args.seed)
    print(f"mu_eta = {mu_eta}")
    print(f"sigma_eta = {sigma_eta}")

    u_nuts = theta_to_u(nuts, mu_eta, sigma_eta)
    u_flow = theta_to_u(flow, mu_eta, sigma_eta)
    u_gnpe = theta_to_u(gnpe, mu_eta, sigma_eta)

    nuts_sigma_theta = nuts.std(axis=0)
    nuts_sigma_u = u_nuts.std(axis=0)
    print(f"\n{'param':<5}"
          f"{'theta sigma_flow/nuts':>25}"
          f"{'u sigma_flow/nuts':>22}"
          f"{'theta sigma_gnpe/nuts':>25}"
          f"{'u sigma_gnpe/nuts':>22}")
    print("-" * 100)
    for j, p in enumerate(PARAM_NAMES):
        rt_f = float(flow[:, j].std() / nuts_sigma_theta[j])
        ru_f = float(u_flow[:, j].std() / nuts_sigma_u[j])
        rt_g = float(gnpe[:, j].std() / nuts_sigma_theta[j])
        ru_g = float(u_gnpe[:, j].std() / nuts_sigma_u[j])
        print(f"{p:<5}{rt_f:>25.3f}{ru_f:>22.3f}{rt_g:>25.3f}{ru_g:>22.3f}")

    print(
        "\nInterpretation: if overdispersion is caused by the logit/expit transform, "
        "u-space ratios should be ~1 even when theta-space ratios are not. "
        "If u-space and theta-space ratios are both >> 1, the transform is not the cause."
    )


if __name__ == "__main__":
    main()
