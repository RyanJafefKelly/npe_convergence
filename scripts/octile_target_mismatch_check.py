#!/usr/bin/env python
"""Direct target-mismatch check for GNK.

NUTS (gnk_model) uses an asymptotic MVN likelihood with octile covariance
given by compute_covariance_matrix(theta, quantiles, n_obs). NPE trains on
empirical sample octiles of simulated datasets. These two "likelihoods"
agree as n -> infty but can differ at finite n.

This script quantifies the gap at theta_0 = (3, 1, 2, 0.5):

1. Simulates M datasets of size n_obs, computes sample octiles, and estimates
   the empirical covariance V_emp of the 7-vector of octiles.
2. Computes V_asy = compute_covariance_matrix(theta_0, quantiles, n_obs).
3. Propagates both through a local-linearization approximation to derive
   the implied parameter-space posterior covariance at theta_0:
       Sigma_theta = (grad_b(theta_0)^T V^{-1} grad_b(theta_0))^{-1}
   and reports per-parameter sigma_emp / sigma_asy.

If sigma_emp / sigma_asy matches the empirical sigma_NPE / sigma_NUTS from
the seed aggregation, target mismatch is the dominant explanation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.scipy.stats import norm

from npe_convergence.examples.gnk import (
    compute_covariance_matrix,
    get_summaries_batches,
    gnk,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "notebooks" / "plots"

TRUE_PARAMS = np.array([3.0, 1.0, 2.0, 0.5])
PARAM_NAMES = ["A", "B", "g", "k"]
QUANTILES = np.linspace(0.125, 0.875, 7)


def simulate_octiles(n_obs: int, M: int, seed: int = 0) -> np.ndarray:
    """Return M x 7 array of sample octiles at theta_0."""
    key = random.key(seed)
    A_arr = jnp.full((M,), TRUE_PARAMS[0])
    B_arr = jnp.full((M,), TRUE_PARAMS[1])
    g_arr = jnp.full((M,), TRUE_PARAMS[2])
    k_arr = jnp.full((M,), TRUE_PARAMS[3])
    summaries = get_summaries_batches(
        key, A_arr, B_arr, g_arr, k_arr,
        n_obs=n_obs, n_sims=M, batch_size=1000,
    )
    return np.asarray(summaries).T  # (M, 7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--M", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Simulating M={args.M} datasets of size n_obs={args.n_obs} at theta_0={TRUE_PARAMS}...")
    octiles = simulate_octiles(args.n_obs, args.M, seed=args.seed)
    print(f"Got octile samples of shape {octiles.shape}")

    V_emp = np.cov(octiles, rowvar=False)
    V_asy = np.asarray(
        compute_covariance_matrix(
            TRUE_PARAMS[0], TRUE_PARAMS[1], TRUE_PARAMS[2], TRUE_PARAMS[3],
            jnp.asarray(QUANTILES), n_obs=args.n_obs,
        )
    )

    print("\n=== Per-octile variance: empirical vs asymptotic ===")
    diag_e = np.diag(V_emp)
    diag_a = np.diag(V_asy)
    q_labels = [f"{q*100:.1f}%" for q in QUANTILES]
    print(f"{'q':<8}{'var_emp':>14}{'var_asy':>14}{'emp/asy':>10}")
    for q, ve, va in zip(q_labels, diag_e, diag_a):
        print(f"{q:<8}{ve:>14.4e}{va:>14.4e}{ve/va:>10.3f}")
    print(f"\nTrace ratio (tr V_emp / tr V_asy): {np.trace(V_emp)/np.trace(V_asy):.3f}")
    print(f"Det ratio   (det V_emp / det V_asy): "
          f"{np.linalg.det(V_emp)/np.linalg.det(V_asy):.3f}")

    # ---- Propagate to parameter-space posterior covariance ----
    # Binding function b(theta) = E[octiles | theta] under the asymptotic model
    # is just the theoretical quantile Q_gk(z_q, theta) where z_q = norm.ppf(q).
    z = norm.ppf(jnp.asarray(QUANTILES))
    def b(theta):
        A_, B_, g_, k_ = theta
        return gnk(z, A_, B_, g_, k_)

    grad_b = np.asarray(jax.jacfwd(b)(jnp.asarray(TRUE_PARAMS)))  # (7, 4)

    V_emp_inv = np.linalg.inv(V_emp)
    V_asy_inv = np.linalg.inv(V_asy)
    Sigma_emp = np.linalg.inv(grad_b.T @ V_emp_inv @ grad_b)
    Sigma_asy = np.linalg.inv(grad_b.T @ V_asy_inv @ grad_b)

    print("\n=== Implied parameter-space posterior std-dev (local-Laplace approximation) ===")
    print(f"{'param':<5}{'sigma_emp':>14}{'sigma_asy':>14}{'emp/asy':>10}")
    for j, p in enumerate(PARAM_NAMES):
        se = float(np.sqrt(Sigma_emp[j, j]))
        sa = float(np.sqrt(Sigma_asy[j, j]))
        print(f"{p:<5}{se:>14.5f}{sa:>14.5f}{se/sa:>10.3f}")

    # Also compute what the empirical sample OCTILES show for skew/kurtosis.
    print("\n=== Marginal skewness & excess kurtosis of sample octiles ===")
    from scipy.stats import skew, kurtosis
    sk = skew(octiles, axis=0)
    ku = kurtosis(octiles, axis=0)
    print(f"{'q':<8}{'skewness':>12}{'excess kurt':>14}")
    for q, s, k_ in zip(q_labels, sk, ku):
        print(f"{q:<8}{s:>12.3f}{k_:>14.3f}")


if __name__ == "__main__":
    main()
