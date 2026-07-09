#!/usr/bin/env python
"""Read-only audit of cached GNK NUTS reference artefacts.

For every (n_obs, seed) in STANDARD_N x range(101), this script walks the
on-disk artefacts that claim to be the NUTS reference for that cell:

  - Central caches:
      res/gnk/nuts_cache_v2_n_obs_X_seed_Y.pkl       (gaussian convention)
      res/gnk/nuts_cache_v2_flow_n_obs_X_seed_Y.pkl  (flow convention)
  - Per-cell dumps:
      res/gnk/npe_n_obs_X_n_sims_Z_seed_Y/true_posterior_samples.pkl
      res/gnk/gaussian_npe_n_obs_X_n_sims_Z_seed_Y/true_posterior_samples.pkl

For each artefact we classify on three axes via local Laplace
approximations at the truth:

  - Likelihood epoch: corrected phi/Q' vs buggy phi^2/Q'.
  - Data convention: flow (no split) vs gaussian (post-split subkey).
  - sd-ratio fit: artefact std vector vs predicted Laplace std vector.

Output: one CSV row per artefact with the classification details.
This is a fast audit; no NUTS is rerun and no NPE is retrained.
"""
from __future__ import annotations

import argparse
import csv
import pickle as pkl
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import (
    compute_covariance_matrix,
    gnk,
    gnk_deriv,
    ss_octile,
)

PARAM_NAMES = ("A", "B", "g", "k")
TRUE_THETA = np.array([3.0, 1.0, 2.0, 0.5])
TRUE_THETA_F32 = jnp.asarray(TRUE_THETA, dtype=jnp.float32)
STANDARD_N = (100, 500, 1000, 5000)
DEFAULT_SEEDS = tuple(range(101))
QUANTILES = jnp.linspace(0.125, 0.875, 7)
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "meeting_2026_05_18" / "gnk_reference_identity_audit.csv"


# ---------------------------------------------------------------------------
# x_obs reconstruction.
# ---------------------------------------------------------------------------


def reconstruct_x_obs(n_obs: int, seed: int, convention: str) -> np.ndarray:
    """Match the NPE training conventions used by the existing paper-grid runs."""
    if convention == "flow":
        key = random.key(seed)
        z_key = key
    elif convention == "gaussian":
        key = random.key(seed)
        _, z_key = random.split(key)
    else:
        raise ValueError(f"unknown convention: {convention}")
    z = random.normal(z_key, shape=(n_obs,), dtype=jnp.float32)
    x = gnk(z, *TRUE_THETA_F32)
    summary = jnp.squeeze(ss_octile(jnp.atleast_2d(x)))
    return np.asarray(summary, dtype=np.float64)


# ---------------------------------------------------------------------------
# Laplace approximations under corrected and buggy densities.
# ---------------------------------------------------------------------------


def _summary_jacobian(theta_np: np.ndarray) -> np.ndarray:
    """Jacobian d mu_i / d theta_j at theta_np, where mu_i = Q(z_i, theta)."""
    theta = jnp.asarray(theta_np)
    z = norm.ppf(QUANTILES)

    def expected_summary(theta_vec):
        A, B, g, k = theta_vec
        return gnk(z, A, B, g, k)

    return np.asarray(jax.jacfwd(expected_summary)(theta))


def _covariance_corrected(n_obs: int, theta_np: np.ndarray) -> np.ndarray:
    """Asymptotic summary covariance under the corrected density phi/Q'."""
    cov = compute_covariance_matrix(
        theta_np[0], theta_np[1], theta_np[2], theta_np[3], QUANTILES, n_obs=n_obs
    )
    return np.asarray(cov)


def _covariance_buggy(n_obs: int, theta_np: np.ndarray) -> np.ndarray:
    """Asymptotic summary covariance under the buggy density phi^2/Q'.

    Under the bug, f(x) was phi(z)^2 / Q'(z) instead of phi(z) / Q'(z), so
    diag_i = p(1-p) / (n_obs * f_i^2) gains a factor 1 / phi(z_i)^2 relative
    to the corrected version, and the off-diagonals gain 1 / (phi_i * phi_j).
    Build this directly from the corrected covariance for speed.
    """
    z = np.asarray(norm.ppf(QUANTILES))
    phi = np.asarray(norm.pdf(z))
    inflation = np.outer(1.0 / phi, 1.0 / phi)  # (7, 7)
    return _covariance_corrected(n_obs, theta_np) * inflation


def _laplace_at_truth(
    n_obs: int, x_obs_arr: np.ndarray, cov_summary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Local Laplace approximation: linearise mu(theta) about the truth.

    Posterior mean shift: dtheta = (J^T V^-1 J)^-1 J^T V^-1 (x_obs - mu(truth))
    Posterior cov:        Sigma  = (J^T V^-1 J)^-1
    """
    mu_truth = np.asarray(gnk(norm.ppf(QUANTILES), *TRUE_THETA_F32))
    jac = _summary_jacobian(TRUE_THETA)
    cov_inv = np.linalg.inv(cov_summary)
    info = jac.T @ cov_inv @ jac
    sigma_theta = np.linalg.inv(info)
    residual = x_obs_arr - mu_truth
    shift = sigma_theta @ jac.T @ cov_inv @ residual
    mean_theta = TRUE_THETA + shift
    return mean_theta, sigma_theta


def precompute_laplace_grid(n_obs_values: tuple[int, ...]) -> dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]]:
    """Cache Laplace approximations for each (n_obs, convention, likelihood) combo."""
    out: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for n_obs in n_obs_values:
        cov_correct = _covariance_corrected(n_obs, TRUE_THETA)
        cov_buggy = _covariance_buggy(n_obs, TRUE_THETA)
        for convention in ("flow", "gaussian"):
            x_obs = reconstruct_x_obs(n_obs, seed=0, convention=convention)
            # We need per-seed x_obs in the audit loop. This caches the cov only.
            out[(n_obs, convention, "corrected")] = (cov_correct, cov_buggy)
            del x_obs
        out[(n_obs, "_cov_correct", "_")] = (cov_correct, cov_correct)
        out[(n_obs, "_cov_buggy", "_")] = (cov_buggy, cov_buggy)
    return out


def laplace_candidates(n_obs: int, seed: int) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Return Laplace (mean, cov) candidates keyed by (likelihood, convention)."""
    cov_correct = _covariance_corrected(n_obs, TRUE_THETA)
    cov_buggy = _covariance_buggy(n_obs, TRUE_THETA)
    candidates = {}
    for convention in ("flow", "gaussian"):
        x_obs = reconstruct_x_obs(n_obs, seed, convention=convention)
        for likelihood, cov in (("corrected", cov_correct), ("buggy", cov_buggy)):
            mean, sigma = _laplace_at_truth(n_obs, x_obs, cov)
            candidates[(likelihood, convention)] = (mean, sigma)
    return candidates


# ---------------------------------------------------------------------------
# Artefact discovery.
# ---------------------------------------------------------------------------


PER_CELL_RE = re.compile(
    r"^(?P<prefix>npe|gaussian_npe)_n_obs_(?P<n_obs>\d+)_n_sims_(?P<n_sims>\d+)_seed_(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class Artefact:
    path: Path
    n_obs: int
    seed: int
    source_kind: str  # central_gaussian, central_flow, per_cell_flow, per_cell_gaussian
    method: str       # gaussian, flow, na
    n_sims: int | None


def discover_artefacts(
    n_obs_values: tuple[int, ...],
    seeds: tuple[int, ...],
    res_root: Path,
    include_per_cell: bool,
) -> list[Artefact]:
    artefacts: list[Artefact] = []

    # Central caches.
    for n_obs in n_obs_values:
        for seed in seeds:
            gpath = res_root / "gnk" / f"nuts_cache_v2_n_obs_{n_obs}_seed_{seed}.pkl"
            if gpath.is_file():
                artefacts.append(
                    Artefact(
                        path=gpath,
                        n_obs=n_obs,
                        seed=seed,
                        source_kind="central_gaussian",
                        method="gaussian",
                        n_sims=None,
                    )
                )
            fpath = res_root / "gnk" / f"nuts_cache_v2_flow_n_obs_{n_obs}_seed_{seed}.pkl"
            if fpath.is_file():
                artefacts.append(
                    Artefact(
                        path=fpath,
                        n_obs=n_obs,
                        seed=seed,
                        source_kind="central_flow",
                        method="flow",
                        n_sims=None,
                    )
                )

    if not include_per_cell:
        return artefacts

    # Per-cell true_posterior_samples.pkl from npe_* and gaussian_npe_* dirs.
    n_obs_set = set(n_obs_values)
    seed_set = set(seeds)
    for d in (res_root / "gnk").iterdir():
        if not d.is_dir():
            continue
        m = PER_CELL_RE.match(d.name)
        if m is None:
            continue
        n_obs = int(m.group("n_obs"))
        seed = int(m.group("seed"))
        if n_obs not in n_obs_set or seed not in seed_set:
            continue
        prefix = m.group("prefix")
        sample_path = d / "true_posterior_samples.pkl"
        if not sample_path.is_file():
            continue
        method = "flow" if prefix == "npe" else "gaussian"
        artefacts.append(
            Artefact(
                path=sample_path,
                n_obs=n_obs,
                seed=seed,
                source_kind=f"per_cell_{method}",
                method=method,
                n_sims=int(m.group("n_sims")),
            )
        )
    return artefacts


# ---------------------------------------------------------------------------
# Classification.
# ---------------------------------------------------------------------------


def load_samples(path: Path) -> np.ndarray | None:
    try:
        with path.open("rb") as f:
            obj = pkl.load(f)
    except Exception:
        return None
    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        return None
    return arr


def classify_artefact(
    samples: np.ndarray,
    candidates: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    """Classify by which Laplace candidate best matches the artefact.

    Score combines mean L2 distance (in posterior-sd units) and std-ratio
    deviation from 1.
    """
    sample_mean = samples.mean(axis=0)
    sample_std = samples.std(axis=0)

    best_score = float("inf")
    best_key: tuple[str, str] | None = None
    scores: dict[tuple[str, str], float] = {}
    for key, (mean_pred, sigma_pred) in candidates.items():
        std_pred = np.sqrt(np.diag(sigma_pred))
        mean_dev = np.abs(sample_mean - mean_pred) / std_pred  # in posterior-sd units
        std_ratio = sample_std / std_pred
        std_dev = np.abs(np.log(std_ratio))
        score = mean_dev.max() + std_dev.max()
        scores[key] = score
        if score < best_score:
            best_score = score
            best_key = key

    likelihood, convention = (None, None) if best_key is None else best_key
    result: dict[str, object] = {
        "classified_likelihood": likelihood,
        "classified_convention": convention,
        "classification_score": best_score,
    }
    for (lik, conv), (mean_pred, sigma_pred) in candidates.items():
        std_pred = np.sqrt(np.diag(sigma_pred))
        for i, name in enumerate(PARAM_NAMES):
            result[f"meandev_{lik}_{conv}_{name}"] = float(
                abs(sample_mean[i] - mean_pred[i]) / std_pred[i]
            )
            result[f"sdratio_{lik}_{conv}_{name}"] = float(sample_std[i] / std_pred[i])
    return result


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-obs",
        type=str,
        default=",".join(str(v) for v in STANDARD_N),
        help="Comma-separated n_obs values to audit.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0-100",
        help="Seed range (e.g. '0-100' or '0,50,99').",
    )
    parser.add_argument(
        "--res-root", type=Path, default=REPO_ROOT / "res",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--include-per-cell",
        action="store_true",
        default=True,
        help="Include per-cell true_posterior_samples.pkl artefacts (default on).",
    )
    parser.add_argument(
        "--no-per-cell",
        action="store_false",
        dest="include_per_cell",
        help="Skip per-cell artefacts (central caches only).",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> tuple[int, ...]:
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return tuple(sorted(out))


def main() -> None:
    args = parse_args()
    n_obs_values = parse_int_list(args.n_obs)
    seeds = parse_int_list(args.seeds)

    artefacts = discover_artefacts(
        n_obs_values=n_obs_values,
        seeds=seeds,
        res_root=args.res_root,
        include_per_cell=args.include_per_cell,
    )
    print(f"Discovered {len(artefacts)} artefacts to audit.")

    # Pre-compute Laplace candidates per (n_obs, seed).
    candidate_cache: dict[tuple[int, int], dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]] = {}

    rows: list[dict[str, object]] = []
    for i, art in enumerate(artefacts, 1):
        if i % 100 == 0:
            print(f"  processed {i} / {len(artefacts)} artefacts...")
        key = (art.n_obs, art.seed)
        if key not in candidate_cache:
            candidate_cache[key] = laplace_candidates(art.n_obs, art.seed)
        candidates = candidate_cache[key]

        samples = load_samples(art.path)
        if samples is None:
            rows.append(
                {
                    **asdict(art),
                    "path": str(art.path),
                    "error": "load_failure",
                }
            )
            continue

        sample_median = np.median(samples, axis=0)
        sample_mean = samples.mean(axis=0)
        sample_std = samples.std(axis=0)
        try:
            mtime = art.path.stat().st_mtime
        except OSError:
            mtime = None

        row: dict[str, object] = {
            "path": str(art.path.relative_to(REPO_ROOT) if art.path.is_relative_to(REPO_ROOT) else art.path),
            "source_kind": art.source_kind,
            "method": art.method,
            "n_obs": art.n_obs,
            "seed": art.seed,
            "n_sims": art.n_sims,
            "mtime": mtime,
            "n_samples": int(samples.shape[0]),
        }
        for i_p, name in enumerate(PARAM_NAMES):
            row[f"median_{name}"] = float(sample_median[i_p])
            row[f"mean_{name}"] = float(sample_mean[i_p])
            row[f"std_{name}"] = float(sample_std[i_p])
        row.update(classify_artefact(samples, candidates))
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No artefacts found; not writing CSV.")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    # Quick summary.
    likelihood_counts: dict[str, int] = {}
    convention_counts: dict[str, int] = {}
    for r in rows:
        lik = r.get("classified_likelihood")
        conv = r.get("classified_convention")
        if lik is not None:
            likelihood_counts[lik] = likelihood_counts.get(lik, 0) + 1
        if conv is not None:
            convention_counts[conv] = convention_counts.get(conv, 0) + 1
    print(f"Classifications by likelihood: {likelihood_counts}")
    print(f"Classifications by convention: {convention_counts}")


if __name__ == "__main__":
    main()
