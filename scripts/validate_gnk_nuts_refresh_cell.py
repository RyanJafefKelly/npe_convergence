#!/usr/bin/env python
"""Validate a canonical GNK NUTS reference cell against intrinsic criteria.

This is the two-cell smoke-test gate from
docs/meeting_2026_05_18/gnk_reference_gpt55_response.md (Pro's section 2).
The new x64 reference is the source of truth; the legacy cache is reported
for context only, not used as a pass/fail gate.

Pass criteria:
  1. x_obs SHA-256 deterministic across two reconstructions.
  2. In-chain self-split kNN KL well under 0.1.
  3. Marginal medians and sds agree across two disjoint chain halves
     within Monte Carlo noise.

Reported but not gated:
  4. Comparison to the legacy central cache.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle as pkl
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import gnk, ss_octile
from npe_convergence.metrics import kullback_leibler

PARAM_NAMES = ("A", "B", "g", "k")
TRUE_THETA = jnp.asarray([3.0, 1.0, 2.0, 0.5], dtype=jnp.float32)


def reconstruct_x_obs(n_obs: int, seed: int, convention: str) -> np.ndarray:
    if convention == "flow":
        key = random.key(seed)
        z_key = key
    elif convention == "gaussian":
        key = random.key(seed)
        _, z_key = random.split(key)
    else:
        raise ValueError(f"unknown convention: {convention}")
    z = random.normal(z_key, shape=(n_obs,), dtype=jnp.float32)
    x = gnk(z, *TRUE_THETA)
    summary = jnp.squeeze(ss_octile(jnp.atleast_2d(x)))
    return np.asarray(summary, dtype=np.float32)


def legacy_central_cache(n_obs: int, seed: int, convention: str) -> Path:
    flow_suffix = "_flow" if convention == "flow" else ""
    return REPO_ROOT / "res" / "gnk" / f"nuts_cache_v2{flow_suffix}_n_obs_{n_obs}_seed_{seed}.pkl"


def load_v3(path: Path) -> dict:
    with path.open("rb") as f:
        return pkl.load(f)


def split_self_kl(samples_flat: np.ndarray, seed: int) -> tuple[float, int]:
    """kNN KL between two disjoint random halves of samples.

    NUTS produces a small number of exact duplicate rows (zero-energy
    transitions). The kNN estimator returns nan when a query sample has
    zero distance to its nearest neighbour in either set, so we dedupe
    both halves before estimating. Returns (kl_estimate, n_duplicates).
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(samples_flat, axis=0)
    n_duplicates = samples_flat.shape[0] - unique.shape[0]
    idx = rng.permutation(unique.shape[0])
    half = unique.shape[0] // 2
    a = unique[idx[:half]]
    b = unique[idx[half:half + half]]
    target = min(2000, half)
    a_sub = a[rng.choice(a.shape[0], target, replace=False)]
    b_sub = b[rng.choice(b.shape[0], target, replace=False)]
    kl = float(kullback_leibler(jnp.asarray(a_sub), jnp.asarray(b_sub)))
    return kl, n_duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--convention", choices=("flow", "gaussian"), required=True)
    parser.add_argument(
        "--v3-root", type=Path, default=REPO_ROOT / "res" / "gnk_v3_refs",
    )
    args = parser.parse_args()

    v3_path = (
        args.v3_root
        / f"nuts_n_obs_{args.n_obs}_seed_{args.seed}_conv_{args.convention}.pkl"
    )
    print(f"Validating: {v3_path.name}")
    fingerprint = load_v3(v3_path)

    samples_grouped = np.asarray(fingerprint["samples"])
    n_chains, n_per_chain, _ = samples_grouped.shape
    samples_flat = samples_grouped.reshape(-1, samples_grouped.shape[-1])
    print(
        f"  samples shape: {samples_grouped.shape} "
        f"(total={samples_flat.shape[0]})"
    )

    # 1. x_obs SHA determinism.
    x_obs_f32 = reconstruct_x_obs(args.n_obs, args.seed, args.convention)
    sha_recompute = hashlib.sha256(x_obs_f32.tobytes()).hexdigest()
    sha_stored = fingerprint["x_obs_summary_unstandardised_sha256"]
    sha_match = sha_recompute == sha_stored
    print(f"  x_obs SHA-256 match: {sha_match}")
    if not sha_match:
        print(f"    stored:    {sha_stored}")
        print(f"    recomputed: {sha_recompute}")

    # 2. In-chain self-split KL.
    kl_self, n_duplicates = split_self_kl(samples_flat, seed=args.seed * 7919 + 11)
    print(
        f"  in-chain self-split kNN KL (2000-vs-2000 of deduped samples): "
        f"{kl_self:.4f} (deduped {n_duplicates} exact duplicates)"
    )

    # 3. Marginal stats across two disjoint chain halves.
    half = n_chains // 2
    chain_a = samples_grouped[:half].reshape(-1, 4)
    chain_b = samples_grouped[half:].reshape(-1, 4)
    median_a = np.median(chain_a, axis=0)
    median_b = np.median(chain_b, axis=0)
    std_a = chain_a.std(axis=0)
    std_b = chain_b.std(axis=0)
    overall_std = samples_flat.std(axis=0)

    print(
        f"  median shift across chain halves (in posterior sd units):"
    )
    for i, name in enumerate(PARAM_NAMES):
        shift = abs(median_a[i] - median_b[i]) / overall_std[i]
        print(f"    {name}: {shift:.4f}")
    print(f"  sd ratio chain_a / chain_b:")
    for i, name in enumerate(PARAM_NAMES):
        ratio = std_a[i] / std_b[i]
        print(f"    {name}: {ratio:.4f}")

    # 4. Comparison to legacy central cache (report only).
    legacy_path = legacy_central_cache(args.n_obs, args.seed, args.convention)
    if legacy_path.exists():
        with legacy_path.open("rb") as f:
            legacy = np.asarray(pkl.load(f))
        legacy_median = np.median(legacy, axis=0)
        legacy_std = legacy.std(axis=0)
        v3_median = np.median(samples_flat, axis=0)
        v3_std = samples_flat.std(axis=0)
        print(f"  legacy comparison ({legacy_path.name}):")
        for i, name in enumerate(PARAM_NAMES):
            shift = abs(legacy_median[i] - v3_median[i]) / v3_std[i]
            ratio = legacy_std[i] / v3_std[i]
            print(
                f"    {name}: median shift {shift:.3f} sds, sd ratio "
                f"{ratio:.3f}"
            )
    else:
        print(f"  no legacy cache to compare against ({legacy_path.name})")

    # Diagnostics from fingerprint.
    diag = fingerprint["diagnostics"]
    print(f"  R-hat: " + ", ".join(
        f"{n}={diag['per_parameter'][n]['r_hat']:.4f}" for n in PARAM_NAMES
    ))
    print(f"  n_eff: " + ", ".join(
        f"{n}={diag['per_parameter'][n]['n_eff']:.0f}" for n in PARAM_NAMES
    ))
    if "divergence_count" in diag:
        print(f"  divergences: {diag['divergence_count']}")

    # Pass / fail summary.
    pass_sha = sha_match
    pass_self_kl = kl_self < 0.1
    median_shifts = [
        abs(median_a[i] - median_b[i]) / overall_std[i]
        for i in range(4)
    ]
    sd_ratios = [std_a[i] / std_b[i] for i in range(4)]
    pass_chain = (
        max(median_shifts) < 0.10 and all(0.95 < r < 1.05 for r in sd_ratios)
    )
    print()
    print(f"  PASS x_obs SHA   : {pass_sha}")
    print(f"  PASS self-split KL: {pass_self_kl}  (KL={kl_self:.4f}, threshold 0.1)")
    print(f"  PASS chain agreement: {pass_chain}")
    print(f"  OVERALL: {pass_sha and pass_self_kl and pass_chain}")


if __name__ == "__main__":
    main()
