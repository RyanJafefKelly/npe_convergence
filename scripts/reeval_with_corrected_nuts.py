"""Re-evaluate Gaussian NPE and Flow NPE against corrected NUTS reference.

This script:
1. Reconstructs the exact x_obs each method used (they differ due to RNG chains)
2. Runs NUTS with the corrected gnk_density_from_z (phi(z)/Q'(z), not phi(z)^2/Q'(z))
3. Loads saved NPE posterior samples
4. Computes KL and MMD with MATCHED sample counts (both use same n)
5. Reports old vs new metrics side by side
"""

import os
import pickle as pkl
import sys

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
from jax.scipy.special import expit

from npe_convergence.examples.gnk import gnk, run_nuts, ss_octile
from npe_convergence.metrics import kullback_leibler, median_heuristic, unbiased_mmd

numpyro.set_host_device_count(4)

SEED = 1
N_OBS = 1000
TRUE_PARAMS = jnp.array([3.0, 1.0, 2.0, 0.5])
N_METRIC = 2000  # matched sample count for both methods


def reconstruct_x_obs_gaussian(seed, n_obs):
    """Gaussian NPE RNG chain: key -> split -> use subkey for z."""
    key = random.key(seed)
    key, subkey = random.split(key)
    z = random.normal(subkey, shape=(n_obs,))
    x_obs = gnk(z, *TRUE_PARAMS)
    x_obs = jnp.atleast_2d(x_obs)
    return jnp.squeeze(ss_octile(x_obs))


def reconstruct_x_obs_flow(seed, n_obs):
    """Flow NPE RNG chain: key used directly for z (no split before)."""
    key = random.key(seed)
    z = random.normal(key, shape=(n_obs,))
    x_obs = gnk(z, *TRUE_PARAMS)
    x_obs = jnp.atleast_2d(x_obs)
    return jnp.squeeze(ss_octile(x_obs))


def run_corrected_nuts(x_obs, n_obs, num_samples=10_000, num_warmup=10_000):
    """Run NUTS with the corrected covariance (gnk_density_from_z is now fixed)."""
    mcmc = run_nuts(seed=1, obs=x_obs, n_obs=n_obs,
                    num_samples=num_samples, num_warmup=num_warmup)
    samples_dict = mcmc.get_samples()
    return jnp.column_stack([samples_dict[p] for p in ["A", "B", "g", "k"]])


def compute_metrics(true_samples, npe_samples, n_metric, key):
    """Compute KL and MMD using matched sample counts."""
    key, sk1 = random.split(key)
    key, sk2 = random.split(key)
    idx_npe = random.permutation(sk1, npe_samples.shape[0])[:n_metric]
    idx_true = random.permutation(sk2, true_samples.shape[0])[:n_metric]
    ps = npe_samples[idx_npe]
    ts = true_samples[idx_true]
    kl = kullback_leibler(ts, ps)
    lengthscale = median_heuristic(jnp.vstack([ts, ps]))
    mmd = unbiased_mmd(ts, ps, lengthscale)
    return float(kl), float(mmd)


# ===== GAUSSIAN NPE: n_obs=1000, n_sims=31623, seed=1 =====
print("=" * 70)
print("GAUSSIAN NPE: n_obs=1000, n_sims=31623, seed=1")
print("=" * 70)

gauss_dir = "res/gnk/gaussian_npe_n_obs_1000_n_sims_31623_seed_1"

# Load saved samples
with open(f"{gauss_dir}/posterior_samples.pkl", "rb") as f:
    gauss_npe_samples = pkl.load(f)
with open(f"{gauss_dir}/true_posterior_samples.pkl", "rb") as f:
    gauss_old_nuts = pkl.load(f)

# Read old metrics
with open(f"{gauss_dir}/kl.txt") as f:
    gauss_old_kl = float(f.read().strip())
with open(f"{gauss_dir}/mmd.txt") as f:
    gauss_old_mmd = float(f.read().strip())

print(f"Old metrics (buggy NUTS reference, 400 samples): KL={gauss_old_kl:.4f}, MMD={gauss_old_mmd:.4f}")

# Recompute old metrics with matched sample count for fairness
key = random.key(999)
gauss_old_kl_matched, gauss_old_mmd_matched = compute_metrics(
    gauss_old_nuts, gauss_npe_samples, N_METRIC, key
)
print(f"Old metrics (buggy NUTS, {N_METRIC} samples): KL={gauss_old_kl_matched:.4f}, MMD={gauss_old_mmd_matched:.4f}")

# Reconstruct x_obs and run corrected NUTS
print("\nReconstructing x_obs for Gaussian NPE...")
x_obs_gauss = reconstruct_x_obs_gaussian(SEED, N_OBS)
print(f"x_obs: {x_obs_gauss}")

print("Running NUTS with corrected covariance...")
gauss_new_nuts = run_corrected_nuts(x_obs_gauss, N_OBS)
print(f"Corrected NUTS samples shape: {gauss_new_nuts.shape}")
print(f"Corrected NUTS mean: {jnp.mean(gauss_new_nuts, axis=0)}")
print(f"Old (buggy) NUTS mean: {jnp.mean(gauss_old_nuts, axis=0)}")

# New metrics
key = random.key(999)
gauss_new_kl, gauss_new_mmd = compute_metrics(
    gauss_new_nuts, gauss_npe_samples, N_METRIC, key
)
print(f"\nNew metrics (corrected NUTS, {N_METRIC} samples): KL={gauss_new_kl:.4f}, MMD={gauss_new_mmd:.4f}")

# ===== FLOW NPE: n_obs=1000, n_sims=1000, seed=1 =====
print("\n" + "=" * 70)
print("FLOW NPE: n_obs=1000, n_sims=1000, seed=1")
print("=" * 70)

flow_dir = "res/gnk/npe_n_obs_1000_n_sims_1000_seed_1"

with open(f"{flow_dir}/posterior_samples.pkl", "rb") as f:
    flow_npe_samples = pkl.load(f)
with open(f"{flow_dir}/true_posterior_samples.pkl", "rb") as f:
    flow_old_nuts = pkl.load(f)
with open(f"{flow_dir}/kl.txt") as f:
    flow_old_kl = float(f.read().strip())
with open(f"{flow_dir}/mmd.txt") as f:
    flow_old_mmd = float(f.read().strip())

print(f"Old metrics (buggy NUTS, 10000 samples): KL={flow_old_kl:.4f}, MMD={flow_old_mmd:.4f}")

# Recompute old metrics with matched sample count
key = random.key(999)
flow_old_kl_matched, flow_old_mmd_matched = compute_metrics(
    flow_old_nuts, flow_npe_samples, N_METRIC, key
)
print(f"Old metrics (buggy NUTS, {N_METRIC} samples): KL={flow_old_kl_matched:.4f}, MMD={flow_old_mmd_matched:.4f}")

# Reconstruct x_obs and run corrected NUTS
print("\nReconstructing x_obs for Flow NPE...")
x_obs_flow = reconstruct_x_obs_flow(SEED, N_OBS)
print(f"x_obs: {x_obs_flow}")

print("Running NUTS with corrected covariance...")
flow_new_nuts = run_corrected_nuts(x_obs_flow, N_OBS)
print(f"Corrected NUTS samples shape: {flow_new_nuts.shape}")
print(f"Corrected NUTS mean: {jnp.mean(flow_new_nuts, axis=0)}")
print(f"Old (buggy) NUTS mean: {jnp.mean(flow_old_nuts, axis=0)}")

# New metrics
key = random.key(999)
flow_new_kl, flow_new_mmd = compute_metrics(
    flow_new_nuts, flow_npe_samples, N_METRIC, key
)
print(f"\nNew metrics (corrected NUTS, {N_METRIC} samples): KL={flow_new_kl:.4f}, MMD={flow_new_mmd:.4f}")

# ===== SUMMARY TABLE =====
print("\n" + "=" * 70)
print("SUMMARY: Old vs New Metrics (all at n_obs=1000, seed=1)")
print("=" * 70)
print(f"{'Method':<35s} {'Old KL':>8s} {'New KL':>8s} {'Old MMD':>10s} {'New MMD':>10s}")
print("-" * 75)
print(f"{'Gaussian NPE (N=31623, matched)':<35s} {gauss_old_kl_matched:8.4f} {gauss_new_kl:8.4f} {gauss_old_mmd_matched:10.4f} {gauss_new_mmd:10.4f}")
print(f"{'Gaussian NPE (N=31623, original)':<35s} {gauss_old_kl:8.4f} {'':>8s} {gauss_old_mmd:10.4f} {'':>10s}")
print(f"{'Flow NPE (N=1000, matched)':<35s} {flow_old_kl_matched:8.4f} {flow_new_kl:8.4f} {flow_old_mmd_matched:10.4f} {flow_new_mmd:10.4f}")
print(f"{'Flow NPE (N=1000, original)':<35s} {flow_old_kl:8.4f} {'':>8s} {flow_old_mmd:10.4f} {'':>10s}")

print("\n\nNote: 'Old' = against buggy NUTS (covariance inflated 6-24x)")
print("      'New' = against corrected NUTS (correct covariance)")
print(f"      All 'matched' metrics use {N_METRIC} samples from each distribution")
print("      Different x_obs between methods (RNG chain divergence at seed=1)")

# Save corrected NUTS for inspection
os.makedirs(f"{gauss_dir}", exist_ok=True)
with open(f"{gauss_dir}/corrected_nuts_samples.pkl", "wb") as f:
    pkl.dump(gauss_new_nuts, f)
with open(f"{flow_dir}/corrected_nuts_samples.pkl", "wb") as f:
    pkl.dump(flow_new_nuts, f)
print("\nSaved corrected NUTS samples to both directories.")
