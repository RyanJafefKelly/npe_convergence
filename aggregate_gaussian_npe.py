"""Aggregate Gaussian NPE and Flow NPE results across seeds.

Searches for results in res/gnk/ (handles nested res/gnk/gnk/ from SCP).
"""
import os
import math
import numpy as np

BASE_DIRS = ["res/gnk", "res/gnk/gnk"]

n_obs_list = [100, 500, 1000, 5000, 10000, 20000]
sim_fns = [
    lambda n: n,
    lambda n: int(n * math.log(n)),
    lambda n: int(n ** (3 / 2)),
    lambda n: n ** 2,
]
labels = ["N=n", "N=nlog(n)", "N=n^3/2", "N=n^2"]


def find_file(prefix, n_obs, n_sims, seed, filename):
    """Search across base dirs for a result file."""
    for base in BASE_DIRS:
        p = f"{base}/{prefix}_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}/{filename}"
        if os.path.isfile(p):
            return p
    return None


def collect_metric(prefix, n_obs, n_sims, metric_file):
    """Collect finite metric values across seeds 0-100."""
    vals = []
    for s in range(101):
        p = find_file(prefix, n_obs, n_sims, s, metric_file)
        if p is not None:
            v = float(open(p).read().strip())
            if np.isfinite(v):
                vals.append(v)
    return vals


def print_metric_table(metric_file, fmt=".3f"):
    """Print side-by-side Gaussian vs Flow for one metric."""
    header = f"{'n_obs':>6s} {'N':>8s} {'N_val':>10s}"
    header += f"  {'Gauss':>18s} {'#':>3s}"
    header += f"  {'Flow':>18s} {'#':>3s}"
    print(header)
    print("-" * 80)

    for n_obs in n_obs_list:
        for label, fn in zip(labels, sim_fns):
            n_sims = fn(n_obs)
            g_vals = collect_metric("gaussian_npe", n_obs, n_sims, metric_file)
            f_vals = collect_metric("npe", n_obs, n_sims, metric_file)

            g_str = f"{np.mean(g_vals):{fmt}} ({np.std(g_vals):{fmt}})" if g_vals else "---"
            f_str = f"{np.mean(f_vals):{fmt}} ({np.std(f_vals):{fmt}})" if f_vals else "---"
            print(f"{n_obs:>6d} {label:>8s} {n_sims:>10d}  {g_str:>18s} {len(g_vals):>3d}  {f_str:>18s} {len(f_vals):>3d}")
        print()


# --- Main output ---
print("=" * 80)
print("KL DIVERGENCE: Gaussian NPE vs Flow NPE  (mean (std), excluding inf)")
print("=" * 80)
print_metric_table("kl.txt", fmt=".3f")

print("=" * 80)
print("MMD: Gaussian NPE vs Flow NPE  (mean (std))")
print("=" * 80)
print_metric_table("mmd.txt", fmt=".4f")

print("=" * 80)
print("COVERAGE: Gaussian NPE (mean 80%/90%/95% across seeds)")
print("=" * 80)
params = ["A", "B", "g", "k"]
for n_obs in n_obs_list:
    for label, fn in zip(labels, sim_fns):
        n_sims = fn(n_obs)
        covs = []
        for s in range(101):
            p = find_file("gaussian_npe", n_obs, n_sims, s, "estimated_coverage.npy")
            if p is not None:
                covs.append(np.load(p))
        if covs:
            mc = np.mean(covs, axis=0)
            parts = [
                f"{params[j]}={mc[j,0]:.2f}/{mc[j,1]:.2f}/{mc[j,2]:.2f}"
                for j in range(4)
            ]
            print(f"n={n_obs:>5d}, {label:>10s} ({len(covs):>3d} seeds): " + ", ".join(parts))
