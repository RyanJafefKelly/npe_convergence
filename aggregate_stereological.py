"""Aggregate stereological Gaussian NPE and Flow NPE results across seeds.

Coverage and bias comparison (no KL/MMD — likelihood is intractable).
"""
import os
import math
import numpy as np

BASE_DIRS = ["res/stereological", "res/stereological/stereological"]

n_obs_list = [100, 500, 1000, 5000]
sim_fns = [
    lambda n: n,
    lambda n: int(n * math.log(n)),
    lambda n: int(n ** (3 / 2)),
    lambda n: n ** 2,
]
labels = ["N=n", "N=nlog(n)", "N=n^3/2", "N=n^2"]
params = ["lambda", "sigma", "xi"]


def find_file(prefix, n_obs, n_sims, seed, filename):
    """Search across base dirs for a result file."""
    for base in BASE_DIRS:
        p = f"{base}/{prefix}_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}/{filename}"
        if os.path.isfile(p):
            return p
    return None


def print_coverage_table(prefix, method_name):
    """Print coverage table for one method."""
    print(f"\nCOVERAGE: {method_name} (mean 80%/90%/95% across seeds)")
    print("=" * 80)
    for n_obs in n_obs_list:
        for label, fn in zip(labels, sim_fns):
            n_sims = fn(n_obs)
            covs = []
            for s in range(101):
                p = find_file(prefix, n_obs, n_sims, s, "estimated_coverage.npy")
                if p is not None:
                    covs.append(np.load(p))
            if covs:
                mc = np.mean(covs, axis=0)
                parts = [
                    f"{params[j]}={mc[j,0]:.2f}/{mc[j,1]:.2f}/{mc[j,2]:.2f}"
                    for j in range(3)
                ]
                print(
                    f"n={n_obs:>5d}, {label:>10s} ({len(covs):>3d} seeds): "
                    + ", ".join(parts)
                )
        print()


def print_bias_table(prefix, method_name):
    """Print mean absolute bias across seeds."""
    print(f"\nBIAS: {method_name} (mean abs bias across seeds)")
    print("=" * 80)
    header = f"{'n_obs':>6s} {'N':>10s} {'N_val':>10s}"
    for p in params:
        header += f"  {p:>10s}"
    header += "  {'#':>3s}"
    print(header)
    print("-" * 70)

    for n_obs in n_obs_list:
        for label, fn in zip(labels, sim_fns):
            n_sims = fn(n_obs)
            all_biases = []
            for s in range(101):
                p = find_file(prefix, n_obs, n_sims, s, "biases.npy")
                if p is not None:
                    b = np.load(p)
                    # biases are stored as flattened (num_coverage_samples * 3,)
                    b = b.reshape(-1, 3)
                    all_biases.append(np.mean(np.abs(b), axis=0))
            if all_biases:
                mb = np.mean(all_biases, axis=0)
                parts = "  ".join(f"{mb[j]:10.3f}" for j in range(3))
                print(
                    f"{n_obs:>6d} {label:>10s} {n_sims:>10d}  {parts}  {len(all_biases):>3d}"
                )
        print()


# --- Main output ---
print("=" * 80)
print("STEREOLOGICAL MODEL: Gaussian NPE vs Flow NPE")
print("=" * 80)

print_coverage_table("gaussian_npe", "Gaussian NPE")
print_coverage_table("npe", "Flow NPE")

print_bias_table("gaussian_npe", "Gaussian NPE")
print_bias_table("npe", "Flow NPE")
