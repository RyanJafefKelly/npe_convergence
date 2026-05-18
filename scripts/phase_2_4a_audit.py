#!/usr/bin/env python
"""Phase 2.4a — zero-cost audit of the Gaussian-NPE pipeline.

Checks:
  #1  Best-vs-last checkpoint. (Already verified by code inspection:
      gaussian_npe.fit saves best_model at val improvement; flowjax
      fit_to_data defaults to return_best=True. Audit just documents.)
  #2  Exact conditional cross-entropy. For Gaussian-NPE, reconstruct the
      predicted MVN at S_n from posterior samples via sample moments in
      u-space (where the model is exactly MVN), then evaluate analytic
      log-density at NUTS samples. Compare to sample-based Perez-Cruz KL.
  #3  Analytic MVN log-density check: evaluate -E_NUTS[log q(theta)] via
      the u-space MVN + change-of-variables. Expose any sampling-transform
      bug that would make the analytic CE disagree with the sample-based KL.
  #4  Cholesky/jitter sanity. Already visually inspected in the build helper
      and in the inverse-transform chain; confirm numerically that the
      reconstructed u-space covariance is symmetric positive-definite and
      matches expected packing.
  #5  u-space residual (already done this morning) - summarize here.

Output: scripts/plots/... + a concise markdown-style audit log to stdout.
"""
from __future__ import annotations

import argparse
import math
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal as mvn

REPO_ROOT = Path(__file__).resolve().parent.parent
RES_DIR = REPO_ROOT / "res" / "gnk"
PLOTS_DIR = REPO_ROOT / "notebooks" / "plots"

sys.path.insert(0, str(REPO_ROOT))
from npe_convergence.metrics import kullback_leibler  # noqa: E402

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


def training_transform_constants(n_sims: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce training-time (mu_eta, sigma_eta) used by run_gnk_gaussian.py."""
    import jax.random as jr
    from jax.scipy.special import logit as j_logit
    import numpyro.distributions as dist
    key = jr.key(seed)
    tol = 1e-6
    thetas_bounded = dist.Uniform(0 + tol, 10 - tol).sample(key, (n_sims, 4))
    thetas_unbounded = j_logit(thetas_bounded / 10)
    return np.asarray(thetas_unbounded.mean(axis=0)), np.asarray(thetas_unbounded.std(axis=0))


def theta_to_u(theta: np.ndarray, mu_eta: np.ndarray, sigma_eta: np.ndarray) -> np.ndarray:
    eps = 1e-6
    t = np.clip(theta, eps, 10 - eps)
    eta = np.log((t / 10) / (1 - t / 10))
    return (eta - mu_eta) / sigma_eta


def log_jacobian_u_given_theta(theta: np.ndarray, sigma_eta: np.ndarray) -> np.ndarray:
    """Log |du/dtheta| summed over dims, per sample (shape: (M,))."""
    eps = 1e-6
    t = np.clip(theta, eps, 10 - eps)
    # u_j = (logit(theta_j/10) - mu_eta_j) / sigma_eta_j
    # du_j/dtheta_j = (1/sigma_eta_j) * (10/(theta_j * (10-theta_j)))
    per_dim = -np.log(sigma_eta) + np.log(10) - np.log(t) - np.log(10 - t)
    return per_dim.sum(axis=-1)


def audit(n_obs: int, n_sims: int, seed: int) -> dict:
    log_lines = []
    def log(msg: str):
        log_lines.append(msg)

    log(f"\n=== AUDIT: (n_obs={n_obs}, n_sims={n_sims}, seed={seed}) ===")

    nuts = load_nuts(n_obs, seed)
    gnpe = load_npe("gaussian", n_obs, n_sims, seed)
    if nuts is None or gnpe is None:
        log("MISSING artifacts, skipping.")
        return {"log": log_lines, "status": "skipped"}

    # ---- Item #5 prep: u-space transforms ----
    mu_eta, sigma_eta = training_transform_constants(n_sims, seed)
    u_nuts = theta_to_u(nuts, mu_eta, sigma_eta)
    u_gnpe = theta_to_u(gnpe, mu_eta, sigma_eta)
    log(f"mu_eta = {mu_eta.round(4)}")
    log(f"sigma_eta = {sigma_eta.round(4)}")

    # ---- Item #2/#3: reconstruct Gaussian-NPE's predicted MVN in u-space ----
    # With 10k samples drawn exactly from N(mu, LL^T), sample mean/cov recover
    # (mu, LL^T) to within ~1/sqrt(10k) ~ 1% relative error.
    mu_u = u_gnpe.mean(axis=0)
    Sigma_u = np.cov(u_gnpe, rowvar=False)

    # Sanity checks on Sigma_u
    eigs = np.linalg.eigvalsh(Sigma_u)
    log(f"\n--- Item #4: Cholesky/covariance sanity ---")
    log(f"reconstructed u-space Sigma eigenvalues: {eigs.round(5)}")
    log(f"symmetric? max|Sigma - Sigma^T| = {np.max(np.abs(Sigma_u - Sigma_u.T)):.2e}")
    log(f"min eigenvalue (should be >0): {eigs.min():.5e}")

    # ---- Item #5: u-space sigma-ratio residual ----
    nuts_sigma_u = u_nuts.std(axis=0)
    gnpe_sigma_u = u_gnpe.std(axis=0)
    log(f"\n--- Item #5: u-space sigma-ratios (gnpe/nuts) ---")
    for j, p in enumerate(PARAM_NAMES):
        log(f"  {p}: {gnpe_sigma_u[j]/nuts_sigma_u[j]:.3f}  "
            f"(theta-space ratio: {gnpe[:, j].std()/nuts[:, j].std():.3f})")

    # ---- Item #2/#3: analytic CE on NUTS samples via u-space MVN ----
    # log q_theta(theta) = log N(u(theta); mu_u, Sigma_u) + sum_j log|du_j/dtheta_j|
    # CE = -E_NUTS[log q_theta(theta)]
    mvn_u = mvn(mean=mu_u, cov=Sigma_u, allow_singular=False)
    log_q_u_at_nuts = mvn_u.logpdf(u_nuts)
    log_jac = log_jacobian_u_given_theta(nuts, sigma_eta)
    log_q_theta_at_nuts = log_q_u_at_nuts + log_jac
    CE_analytic = float(-log_q_theta_at_nuts.mean())

    # Sample-based Perez-Cruz KL on 2000/2000 (same convention as kl.txt)
    rng = np.random.default_rng(seed)
    idx_npe = rng.permutation(len(gnpe))[:2000]
    idx_nuts = rng.permutation(len(nuts))[:2000]
    kl_pc = float(kullback_leibler(nuts[idx_nuts], gnpe[idx_npe]))

    # Entropy of NUTS: estimate via Perez-Cruz KL(NUTS || NUTS/2)
    # (Simpler: we don't need H(NUTS) exactly because CE_analytic and KL_pc have
    # the same relation: KL = CE - H(NUTS). So CE - KL = H(NUTS).)
    implied_H_nuts = CE_analytic - kl_pc
    log(f"\n--- Items #2 and #3: Analytic MVN CE vs sample-based KL ---")
    log(f"CE_analytic (Gaussian-NPE MVN eval'd on NUTS):  {CE_analytic:.4f} nats/sample")
    log(f"KL_PC (Perez-Cruz sample-based, NUTS vs GNPE):  {kl_pc:.4f} nats")
    log(f"Implied H(NUTS) = CE_analytic - KL_PC:          {implied_H_nuts:.4f} nats")
    log(f"(Sanity check: differential entropy of 4D Gaussian with NUTS cov "
        f"would be ~{0.5*np.log(np.linalg.det(2*np.pi*np.e*np.cov(nuts, rowvar=False))):.4f})")

    # Check if Perez-Cruz KL agrees with CE - H across a couple seeds
    # (If CE_analytic >> implied_H_nuts_4D_gaussian, sample-based KL may be biased).

    # ---- Gaussian-NPE sample self-consistency: do samples match their own MVN? ----
    # Draw M' fresh samples from the reconstructed MVN and compare to the saved samples
    # via Perez-Cruz. If they're close, the saved posterior_samples are self-consistent
    # with an MVN prediction. If far, there's a sampling-pipeline bug.
    fresh_u = rng.multivariate_normal(mu_u, Sigma_u, size=2000)
    gnpe_2k = gnpe[idx_npe]
    fresh_theta = None
    # Convert fresh_u back to theta-space via inverse transform
    # theta = 10 * expit(u * sigma_eta + mu_eta)
    from scipy.special import expit
    fresh_eta = fresh_u * sigma_eta + mu_eta
    fresh_theta = expit(fresh_eta) * 10
    kl_self = float(kullback_leibler(gnpe_2k, fresh_theta))
    log(f"\n--- Self-consistency: saved GNPE samples vs resampled from reconstructed MVN ---")
    log(f"KL(saved GNPE || fresh MVN-resample): {kl_self:.4f} nats")
    log(f"(Should be near 0 if saved samples are drawn from the same MVN. If far, sampling bug.)")

    return {
        "log": log_lines,
        "status": "ok",
        "CE_analytic": CE_analytic,
        "kl_pc": kl_pc,
        "kl_self": kl_self,
        "sigma_ratio_u": (gnpe_sigma_u / nuts_sigma_u).tolist(),
        "sigma_ratio_theta": (gnpe.std(axis=0) / nuts.std(axis=0)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--n-sims", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = []

    print("# Phase 2.4a audit report")
    print(f"# Config: n_obs={args.n_obs}, n_sims={args.n_sims}, seeds={seeds}")
    print()
    print("## Item #1 (best-vs-last checkpoint): by code inspection")
    print("   - flowjax `fit_to_data` default `return_best=True` (flowjax/train/loops.py:91)")
    print("   - `gaussian_npe.fit` saves `best_model` on val improvement "
          "(npe_convergence/methods/gaussian_npe.py:195-197)")
    print("   Result: saved `posterior_samples.pkl` is from best-val checkpoint. CLEAN.")
    print()

    for seed in seeds:
        result = audit(args.n_obs, args.n_sims, seed)
        print("\n".join(result["log"]))
        all_results.append(result)

    # Consolidated summary
    print("\n=== CONSOLIDATED (medians across seeds) ===")
    ok_results = [r for r in all_results if r.get("status") == "ok"]
    if not ok_results:
        print("No successful audits.")
        return
    ce_med = np.median([r["CE_analytic"] for r in ok_results])
    kl_med = np.median([r["kl_pc"] for r in ok_results])
    self_med = np.median([r["kl_self"] for r in ok_results])
    print(f"median CE_analytic: {ce_med:.3f}")
    print(f"median KL_PC (sample-based):  {kl_med:.3f}")
    print(f"median self-consistency KL:   {self_med:.3f}  "
          "(near 0 = saved samples match reconstructed MVN)")
    print()
    print("Interpretation:")
    print("  - If median kl_self is near 0 (<0.1 nats): sampling pipeline is self-consistent,")
    print("    so the large kl_PC vs NUTS is a real property of the trained Gaussian-NPE,")
    print("    not a sampling/transform bug.")
    print("  - If kl_self is non-trivial (>0.5 nats): saved samples do NOT match what the")
    print("    MVN they were drawn from would say, indicating a sampling or transform issue.")


if __name__ == "__main__":
    main()
