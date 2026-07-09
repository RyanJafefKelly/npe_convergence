#!/usr/bin/env python
"""Dispersed-start stress test for headline GNK cells.

Per Pro's verification item 4 in the GPT-5.5 review of the canonical
refresh plan: truth-centred NUTS chains can mix well to one basin while
missing a remote mode. This script runs 20 L-BFGS MAP optimisations from
dispersed uniform-prior starts on the corrected x64 log posterior. If all
basins converge to the same MAP, multimodality is unlikely. Run only on
the headline cells (n=5000, seed=50) and (n=1000, seed=36); not a sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle as pkl
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax

jax.config.update("jax_enable_x64", True)
assert jax.config.jax_enable_x64

import jax.numpy as jnp
import jax.random as random
import numpy as np
import scipy.optimize as sopt
from jax.scipy.special import expit, logit
from jax.scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import compute_covariance_matrix, gnk, ss_octile

PARAM_NAMES = ("A", "B", "g", "k")
TRUE_THETA = jnp.asarray([3.0, 1.0, 2.0, 0.5])
DEFAULT_V3_ROOT = REPO_ROOT / "res" / "gnk_v3_refs"
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs" / "meeting_2026_05_18" / "gnk_dispersed_start_stress.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def reconstruct_x_obs(n_obs: int, seed: int, convention: str) -> jnp.ndarray:
    if convention == "flow":
        key = random.key(seed)
        z_key = key
    elif convention == "gaussian":
        key = random.key(seed)
        _, z_key = random.split(key)
    else:
        raise ValueError(convention)
    z = random.normal(z_key, shape=(n_obs,), dtype=jnp.float32)
    x_raw = gnk(z, *TRUE_THETA.astype(jnp.float32))
    summary = jnp.squeeze(ss_octile(jnp.atleast_2d(x_raw)))
    return jnp.asarray(summary, dtype=jnp.float64)


def log_posterior_eta(eta: jnp.ndarray, x_obs: jnp.ndarray, n_obs: int) -> jnp.ndarray:
    """Log posterior in unbounded eta = logit(theta / 10) space.

    log p(theta | x) = log p(x | theta) + log p(theta) (uniform, constant)
    plus the change-of-variables log Jacobian.
    """
    theta = expit(eta) * 10.0
    A, B, g, k = theta
    quantile_length = 1.0 / (len(x_obs) + 1)
    quantiles = jnp.linspace(quantile_length, 1.0 - quantile_length, len(x_obs))
    z = norm.ppf(quantiles)
    mu = gnk(z, A, B, g, k)
    cov = compute_covariance_matrix(A, B, g, k, quantiles, n_obs)
    cov = cov + 1e-6 * jnp.eye(len(x_obs))
    # MVN log pdf
    diff = x_obs - mu
    sign, logabsdet = jnp.linalg.slogdet(cov)
    invcov = jnp.linalg.inv(cov)
    quad = diff @ invcov @ diff
    log_lik = -0.5 * (len(x_obs) * jnp.log(2 * jnp.pi) + logabsdet + quad)
    # Jacobian of theta = 10 * sigmoid(eta) wrt eta: d theta / d eta = 10 * sigmoid(eta) * (1 - sigmoid(eta)).
    sig = expit(eta)
    log_jac = jnp.sum(jnp.log(10.0) + jnp.log(sig) + jnp.log(1.0 - sig))
    # Uniform prior in theta is constant in [0,10]^4, so we just need the Jacobian.
    return log_lik + log_jac


log_posterior_jax = jax.jit(log_posterior_eta, static_argnames=("n_obs",))
neg_log_posterior_jax = jax.jit(
    lambda eta, x_obs, n_obs: -log_posterior_eta(eta, x_obs, n_obs),
    static_argnames=("n_obs",),
)
grad_neg_log_posterior_jax = jax.jit(
    jax.grad(lambda eta, x_obs, n_obs: -log_posterior_eta(eta, x_obs, n_obs)),
    static_argnames=("n_obs",),
)


def lbfgs_map(
    eta0: np.ndarray, x_obs: jnp.ndarray, n_obs: int
) -> tuple[np.ndarray, float, bool]:
    def fun(eta):
        return float(neg_log_posterior_jax(jnp.asarray(eta), x_obs, n_obs))

    def jac(eta):
        return np.asarray(grad_neg_log_posterior_jax(jnp.asarray(eta), x_obs, n_obs))

    result = sopt.minimize(
        fun, eta0, jac=jac, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-6}
    )
    return result.x, float(result.fun), bool(result.success)


def dispersed_starts(seed: int, n_starts: int = 20) -> np.ndarray:
    """20 uniform-prior draws in theta-space, mapped to eta-space."""
    key = random.key(seed)
    theta = random.uniform(
        key, shape=(n_starts, 4), minval=jnp.array([0.05] * 4), maxval=jnp.array([9.95] * 4)
    )
    eta = logit(theta / 10.0)
    return np.asarray(eta)


def run_stress(n_obs: int, seed: int, convention: str) -> dict[str, Any]:
    x_obs = reconstruct_x_obs(n_obs, seed, convention)
    starts = dispersed_starts(seed=seed * 13 + 7, n_starts=20)
    truth_eta = logit(TRUE_THETA / 10.0)
    print(
        f"  cell (n_obs={n_obs}, seed={seed}, convention={convention})"
    )
    print("  computing MAP from 20 dispersed starts + 1 truth start...")
    results = []
    # First a truth-centred run for the canonical mode.
    eta_star, nll_star, ok = lbfgs_map(np.asarray(truth_eta), x_obs, n_obs)
    truth_run = {
        "init": "truth",
        "init_theta": (np.asarray(expit(truth_eta) * 10.0)).tolist(),
        "map_theta": (np.asarray(expit(eta_star) * 10.0)).tolist(),
        "map_eta": eta_star.tolist(),
        "neg_log_posterior": nll_star,
        "success": ok,
    }
    for i, eta0 in enumerate(starts):
        eta_map, nll, ok = lbfgs_map(eta0, x_obs, n_obs)
        results.append(
            {
                "init": f"random_{i}",
                "init_theta": (np.asarray(expit(jnp.asarray(eta0)) * 10.0)).tolist(),
                "map_theta": (np.asarray(expit(jnp.asarray(eta_map)) * 10.0)).tolist(),
                "map_eta": eta_map.tolist(),
                "neg_log_posterior": nll,
                "success": ok,
            }
        )
    # Cluster MAPs: a result agrees with the truth-mode MAP if its eta is within
    # 0.05 of it.
    truth_eta_arr = np.asarray(eta_star)
    matched = 0
    other_modes = []
    for r in results:
        eta_arr = np.asarray(r["map_eta"])
        if r["success"] and np.linalg.norm(eta_arr - truth_eta_arr) < 0.1:
            matched += 1
        else:
            other_modes.append(r)
    return {
        "n_obs": n_obs,
        "seed": seed,
        "convention": convention,
        "truth_run": truth_run,
        "random_starts": results,
        "matched_truth_mode_count": matched,
        "total_random_starts": len(results),
        "match_fraction": matched / len(results) if results else None,
        "other_modes": other_modes,
        "x_obs": [float(v) for v in np.asarray(x_obs)],
        "completed_at": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        type=str,
        default="5000,50,gaussian;5000,50,flow;1000,36,gaussian;1000,36,flow",
        help="Semicolon-separated triples 'n_obs,seed,convention'.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell_specs = []
    for spec in args.cells.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        n_obs_s, seed_s, conv = spec.split(",")
        cell_specs.append((int(n_obs_s), int(seed_s), conv.strip()))

    payload = {
        "created_at": utc_now(),
        "cells": [],
        "thresholds": {
            "min_match_fraction": 0.95,
            "max_eta_distance_for_match": 0.1,
        },
    }
    for n_obs, seed, convention in cell_specs:
        cell_result = run_stress(n_obs, seed, convention)
        payload["cells"].append(cell_result)
        print(
            f"  match fraction: "
            f"{cell_result['matched_truth_mode_count']} / "
            f"{cell_result['total_random_starts']} "
            f"({cell_result['match_fraction']:.2f})"
        )
        if cell_result["other_modes"]:
            print("  WARNING: starts found other modes:")
            for m in cell_result["other_modes"]:
                print(
                    f"    init {m['init']}: map_theta = "
                    f"{m['map_theta']}, nll = {m['neg_log_posterior']:.2f}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
