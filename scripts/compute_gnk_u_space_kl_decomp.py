#!/usr/bin/env python
"""Coordinate-aware GNK Gaussian-NPE KL decomposition.

This script reads existing GNK NUTS and Gaussian-NPE sample caches only.  It
does not train models and does not write into cache directories.

Definitions follow the weekend execution brief.  The native Gaussian-NPE
coordinate is

    u = (logit(theta / 10) - mu_eta) / sigma_eta.

Exact KLs and the analytic Gaussian-Gaussian decomposition are invariant to the
common diagonal affine standardisation by (mu_eta, sigma_eta).  To avoid
regenerating massive prior-simulation arrays just to recover those constants,
the implementation works in eta = logit(theta / 10).  The finite-sample
Perez-Cruz/kNN estimates of K_u^* are therefore eta-coordinate estimates of the
same exact u-space target; unlike exact KL, the estimator is not guaranteed to
be exactly invariant under anisotropic diagonal scaling.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle as pkl
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from npe_convergence.metrics import kullback_leibler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "res" / "gnk"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "plots"

PARAM_NAMES = ("A", "B", "g", "k")
D_S = 7
D_THETA = 4
D_TOTAL = D_S + D_THETA
FIT_SIZE = 5000
N_METRIC = 2000
THETA_EPS = 1e-6
JITTER = 1e-8

GAUSSIAN_DIR_RE = re.compile(
    r"^gaussian_npe_n_obs_(?P<n>\d+)_n_sims_(?P<N>\d+)_seed_(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class GaussianCache:
    n: int
    N: int
    seed: int
    directory: Path
    posterior_path: Path


@dataclass
class OracleMoments:
    K_theta_oracle: float
    K_u_oracle: float
    coord_offset: float
    theta_mean: np.ndarray
    theta_cov: np.ndarray
    u_mean: np.ndarray
    u_cov: np.ndarray
    nuts_path: Path
    nuts_sample_count: int
    nuts_u_clip_count: int
    theta_kl_jitter: float
    u_kl_jitter: float


def parse_int_list(spec: str | None) -> set[int] | None:
    if spec is None or spec.strip().lower() in {"", "all"}:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_pickle_array(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        return np.asarray(pkl.load(f), dtype=np.float64)


def find_nuts_cache(cache_dir: Path, n: int, seed: int) -> Path | None:
    for prefix in ("nuts_cache_v2_n_obs", "nuts_cache_v2_flow_n_obs"):
        path = cache_dir / f"{prefix}_{n}_seed_{seed}.pkl"
        if path.exists():
            return path
    return None


def discover_gaussian_caches(
    cache_dir: Path,
    n_filter: set[int] | None,
    N_filter: set[int] | None,
    seed_filter: set[int] | None,
) -> list[GaussianCache]:
    caches: list[GaussianCache] = []
    for directory in sorted(cache_dir.glob("gaussian_npe_n_obs_*_n_sims_*_seed_*")):
        if not directory.is_dir():
            continue
        match = GAUSSIAN_DIR_RE.match(directory.name)
        if match is None:
            continue
        n = int(match.group("n"))
        N = int(match.group("N"))
        seed = int(match.group("seed"))
        if n_filter is not None and n not in n_filter:
            continue
        if N_filter is not None and N not in N_filter:
            continue
        if seed_filter is not None and seed not in seed_filter:
            continue
        posterior_path = directory / "posterior_samples.pkl"
        if posterior_path.exists():
            caches.append(GaussianCache(n, N, seed, directory, posterior_path))
    return sorted(caches, key=lambda c: (c.n, c.N, c.seed))


def theta_to_u_affine_invariant(theta: np.ndarray) -> tuple[np.ndarray, int]:
    """Return eta = logit(theta / 10), the affine-invariant u coordinate.

    The returned coordinate differs from u only by a shared diagonal affine map,
    so exact KLs and analytic Gaussian-Gaussian decomposition terms are
    unchanged. Finite-sample kNN oracle estimates can differ slightly after
    diagonal standardisation.
    """
    clipped = np.clip(theta, THETA_EPS, 10.0 - THETA_EPS)
    n_clipped = int(np.count_nonzero(clipped != theta))
    x = clipped / 10.0
    eta = np.log(x) - np.log1p(-x)
    return eta, n_clipped


def stable_cov(samples: np.ndarray, jitter: float = JITTER) -> tuple[np.ndarray, float, float]:
    cov = np.cov(samples, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    eye = np.eye(cov.shape[0])
    used = 0.0
    for scale in (0.0, jitter, 1e-7, 1e-6, 1e-5, 1e-4):
        candidate = cov + scale * eye
        try:
            sign, logdet = np.linalg.slogdet(candidate)
            if sign > 0:
                np.linalg.cholesky(candidate)
                eig_min = float(np.linalg.eigvalsh(candidate).min())
                return candidate, float(logdet), eig_min
        except np.linalg.LinAlgError:
            pass
        used = scale
    eig_min = float(np.linalg.eigvalsh(cov).min())
    raise np.linalg.LinAlgError(f"Could not regularise covariance; min eig={eig_min}, last jitter={used}")


def gaussian_samples(mean: np.ndarray, cov: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(mean, cov, size=size, check_valid="raise")


def finite_sample_kl(
    true_samples: np.ndarray,
    sim_samples: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Pérez-Cruz KL with deterministic tiny-jitter retries for duplicate points."""
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(kullback_leibler(true_samples, sim_samples))
    if np.isfinite(value):
        return value, 0.0

    pooled = np.vstack([true_samples, sim_samples])
    scale = np.maximum(pooled.std(axis=0, ddof=1), 1.0)
    for eps in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        true_j = true_samples + rng.normal(0.0, eps * scale, size=true_samples.shape)
        sim_j = sim_samples + rng.normal(0.0, eps * scale, size=sim_samples.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = float(kullback_leibler(true_j, sim_j))
        if np.isfinite(value):
            return value, eps
    return value, math.nan


def oracle_kl(samples: np.ndarray, seed: int) -> tuple[float, np.ndarray, np.ndarray, float]:
    if len(samples) < 3:
        raise ValueError("Need at least three samples for oracle KL")
    fit_size = min(FIT_SIZE, max(2, len(samples) // 2))
    metric_size = min(N_METRIC, max(1, len(samples) - fit_size))
    fit = samples[:fit_size]
    held_out = samples[fit_size : fit_size + metric_size]
    if len(held_out) < metric_size:
        rng_idx = np.random.default_rng(seed + 17)
        held_out = samples[rng_idx.permutation(len(samples))[:metric_size]]
    mean = fit.mean(axis=0)
    cov, _, _ = stable_cov(fit)
    rng = np.random.default_rng(seed)
    oracle = gaussian_samples(mean, cov, len(held_out), rng)
    kl, jitter_used = finite_sample_kl(held_out, oracle, rng)
    return kl, mean, cov, jitter_used


def compute_oracle_moments(cache_dir: Path, n: int, seed: int) -> OracleMoments | None:
    nuts_path = find_nuts_cache(cache_dir, n, seed)
    if nuts_path is None:
        return None
    nuts = load_pickle_array(nuts_path)
    u_nuts, nuts_clip_count = theta_to_u_affine_invariant(nuts)
    k_theta, theta_mean, theta_cov, theta_jitter = oracle_kl(nuts, seed=seed)
    k_u, u_mean, u_cov, u_jitter = oracle_kl(u_nuts, seed=seed + 10_000)
    return OracleMoments(
        K_theta_oracle=k_theta,
        K_u_oracle=k_u,
        coord_offset=k_u - k_theta,
        theta_mean=theta_mean,
        theta_cov=theta_cov,
        u_mean=u_mean,
        u_cov=u_cov,
        nuts_path=nuts_path,
        nuts_sample_count=len(nuts),
        nuts_u_clip_count=nuts_clip_count,
        theta_kl_jitter=theta_jitter,
        u_kl_jitter=u_jitter,
    )


def stable_cov_matrix(cov: np.ndarray, jitter: float = JITTER) -> tuple[np.ndarray, float, float]:
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    eye = np.eye(cov.shape[0])
    for scale in (0.0, jitter, 1e-7, 1e-6, 1e-5, 1e-4):
        candidate = cov + scale * eye
        try:
            sign, logdet = np.linalg.slogdet(candidate)
            if sign > 0:
                np.linalg.cholesky(candidate)
                eig_min = float(np.linalg.eigvalsh(candidate).min())
                return candidate, float(logdet), eig_min
        except np.linalg.LinAlgError:
            pass
    eig_min = float(np.linalg.eigvalsh(cov).min())
    raise np.linalg.LinAlgError(f"Could not regularise covariance matrix; min eig={eig_min}")


def _gaussian_kl_from_covs(
    p_mean: np.ndarray,
    p_cov: np.ndarray,
    p_logdet: float,
    p_eig_min: float,
    q_mean: np.ndarray,
    q_cov: np.ndarray,
    q_logdet: float,
    q_eig_min: float,
) -> tuple[float, float, float, float, float]:
    d = p_mean.shape[0]
    chol_q = np.linalg.cholesky(q_cov)
    solved_cov = np.linalg.solve(chol_q, p_cov)
    solved_cov = np.linalg.solve(chol_q.T, solved_cov)
    trace_term = float(np.trace(solved_cov))
    diff = q_mean - p_mean
    y = np.linalg.solve(chol_q, diff)
    quad = float(y @ y)
    mean_term = 0.5 * quad
    cov_term = 0.5 * (trace_term - d + q_logdet - p_logdet)
    total = mean_term + cov_term
    return float(total), float(mean_term), float(cov_term), p_eig_min, q_eig_min


def gaussian_kl_decomp_from_moments(
    p_mean: np.ndarray,
    p_cov_raw: np.ndarray,
    q_mean: np.ndarray,
    q_cov_raw: np.ndarray,
) -> tuple[float, float, float, float, float]:
    p_cov, p_logdet, p_eig_min = stable_cov_matrix(p_cov_raw)
    q_cov, q_logdet, q_eig_min = stable_cov_matrix(q_cov_raw)
    return _gaussian_kl_from_covs(
        p_mean, p_cov, p_logdet, p_eig_min, q_mean, q_cov, q_logdet, q_eig_min
    )


def self_consistency_kl_u(
    u_samples: np.ndarray,
    q_mean: np.ndarray,
    q_cov: np.ndarray,
    seed: int,
    n_metric: int,
) -> float:
    metric_size = min(n_metric, len(u_samples))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(u_samples))[:metric_size]
    q_cov_stable, _, _ = stable_cov_matrix(q_cov)
    fresh = gaussian_samples(q_mean, q_cov_stable, metric_size, rng)
    kl, _ = finite_sample_kl(u_samples[idx], fresh, rng)
    return kl


def sigma_ratio_columns(prefix: str, numerator: np.ndarray, denominator: np.ndarray) -> dict[str, float]:
    num = numerator.std(axis=0, ddof=1)
    den = denominator.std(axis=0, ddof=1)
    return {f"sigma_ratio_{prefix}_{p}": float(num[j] / den[j]) for j, p in enumerate(PARAM_NAMES)}


def summarise(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"median": math.nan, "q25": math.nan, "q75": math.nan}
    return {
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default="gnk_u_space_kl_decomp_20260425")
    parser.add_argument("--n-values", type=str, default="all", help="Comma/range list, e.g. '500,1000' or '500-1000'; default all")
    parser.add_argument("--N-values", type=str, default="all", help="Comma/range list of simulation budgets; default all")
    parser.add_argument("--seeds", type=str, default="all", help="Comma/range list of seeds; default all")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for subset dry runs")
    parser.add_argument("--self-check-metric-size", type=int, default=2000)
    args = parser.parse_args()

    n_filter = parse_int_list(args.n_values)
    N_filter = parse_int_list(args.N_values)
    seed_filter = parse_int_list(args.seeds)

    caches = discover_gaussian_caches(args.cache_dir, n_filter, N_filter, seed_filter)
    if args.max_rows is not None:
        caches = caches[: args.max_rows]
    if not caches:
        raise SystemExit("No Gaussian-NPE posterior sample caches matched the requested filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    created_at = datetime.now(timezone.utc).isoformat()

    oracle_cache: dict[tuple[int, int], OracleMoments | None] = {}
    nuts_u_cache: dict[tuple[int, int], np.ndarray] = {}
    nuts_theta_cache: dict[tuple[int, int], np.ndarray] = {}
    rows: list[dict] = []
    skipped: list[dict] = []

    for i, cache in enumerate(caches, start=1):
        key = (cache.n, cache.seed)
        if key not in oracle_cache:
            oracle_cache[key] = compute_oracle_moments(args.cache_dir, cache.n, cache.seed)
            if oracle_cache[key] is not None:
                nuts_theta = load_pickle_array(oracle_cache[key].nuts_path)
                nuts_u, _ = theta_to_u_affine_invariant(nuts_theta)
                nuts_theta_cache[key] = nuts_theta
                nuts_u_cache[key] = nuts_u
        oracle = oracle_cache[key]
        if oracle is None:
            skipped.append({"n": cache.n, "N": cache.N, "seed": cache.seed, "reason": "missing_nuts_cache"})
            continue

        gnpe_theta = load_pickle_array(cache.posterior_path)
        gnpe_u, gnpe_clipped = theta_to_u_affine_invariant(gnpe_theta)
        q_mean = gnpe_u.mean(axis=0)
        q_cov, q_logdet, q_eig_min = stable_cov(gnpe_u)
        p_cov, p_logdet, p_eig_min = stable_cov_matrix(oracle.u_cov)
        delta_total, delta_mean, delta_cov, _, _ = _gaussian_kl_from_covs(
            oracle.u_mean, p_cov, p_logdet, p_eig_min, q_mean, q_cov, q_logdet, q_eig_min
        )
        self_kl = self_consistency_kl_u(
            gnpe_u,
            q_mean,
            q_cov,
            seed=cache.seed + cache.N + 20_000,
            n_metric=args.self_check_metric_size,
        )

        nuts_theta = nuts_theta_cache[key]
        nuts_u = nuts_u_cache[key]
        row = {
            "n": cache.n,
            "N": cache.N,
            "d_s": D_S,
            "d_theta": D_THETA,
            "d_total": D_TOTAL,
            "seed": cache.seed,
            "method": "Gaussian-NPE",
            "K_theta_oracle": oracle.K_theta_oracle,
            "K_u_oracle": oracle.K_u_oracle,
            "coord_offset": oracle.coord_offset,
            "Delta_u_total": delta_total,
            "Delta_u_mean": delta_mean,
            "Delta_u_cov": delta_cov,
            "Delta_theta_total": delta_total + oracle.coord_offset,
            "scaled_budget": cache.N / (D_TOTAL * D_TOTAL * cache.n),
            "nuts_cache_path": str(oracle.nuts_path.relative_to(REPO_ROOT)),
            "gaussian_npe_dir": str(cache.directory.relative_to(REPO_ROOT)),
            "posterior_samples_path": str(cache.posterior_path.relative_to(REPO_ROOT)),
            "nuts_sample_count": oracle.nuts_sample_count,
            "gnpe_sample_count": len(gnpe_theta),
            "oracle_fit_size": min(FIT_SIZE, max(2, oracle.nuts_sample_count // 2)),
            "oracle_metric_size": min(N_METRIC, max(1, oracle.nuts_sample_count - min(FIT_SIZE, max(2, oracle.nuts_sample_count // 2)))),
            "self_consistency_kl_u": self_kl,
            "self_consistency_metric_size": min(args.self_check_metric_size, len(gnpe_u)),
            "K_theta_oracle_kl_jitter": oracle.theta_kl_jitter,
            "K_u_oracle_kl_jitter": oracle.u_kl_jitter,
            "theta_clip_count_nuts": oracle.nuts_u_clip_count,
            "qhat_reconstruction": "sample_moments_of_saved_Gaussian_NPE_samples_in_logit_theta_over_10_coordinate",
            "u_affine_note": "exact KL and analytic Gaussian-Gaussian terms are invariant to common diagonal affine standardisation by mu_eta/sigma_eta; finite-sample kNN oracle estimates are eta-coordinate estimates and may differ slightly after diagonal standardisation",
            "existing_fresh_resample_diagnostic_source": "not_found_as_machine_readable_cache; recomputed_self_consistency_kl_u_here",
            "theta_clip_count_gnpe": gnpe_clipped,
            "qhat_logdet_u": q_logdet,
            "qhat_min_eig_u": q_eig_min,
            "oracle_min_eig_u": p_eig_min,
            "created_at_utc": created_at,
            "commit": commit,
        }
        row.update(sigma_ratio_columns("theta", gnpe_theta, nuts_theta))
        row.update(sigma_ratio_columns("u", gnpe_u, nuts_u))
        rows.append(row)

        if i == 1 or i % 100 == 0 or i == len(caches):
            print(f"[{i}/{len(caches)}] processed n={cache.n}, N={cache.N}, seed={cache.seed}")

    if not rows:
        raise SystemExit(f"No rows computed. Skipped: {skipped[:5]}")

    df = pd.DataFrame(rows).sort_values(["n", "N", "seed"]).reset_index(drop=True)
    csv_path = args.output_dir / f"{args.output_prefix}_per_seed.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    summary = {
        "task": "gnk-coordinate-aware-u-space-kl-decomp",
        "created_at_utc": created_at,
        "commit": commit,
        "cache_dir": str(args.cache_dir.relative_to(REPO_ROOT) if args.cache_dir.is_relative_to(REPO_ROOT) else args.cache_dir),
        "output_csv": str(csv_path.relative_to(REPO_ROOT)),
        "rows": int(len(df)),
        "skipped": skipped,
        "n_values": sorted(map(int, df["n"].unique())),
        "N_values_by_n": {
            str(n): sorted(map(int, df.loc[df["n"] == n, "N"].unique()))
            for n in sorted(df["n"].unique())
        },
        "seed_count_by_n_N": {
            f"n={int(n)},N={int(N)}": int(len(g))
            for (n, N), g in df.groupby(["n", "N"])
        },
        "summaries": {
            "K_theta_oracle": summarise(df.drop_duplicates(["n", "seed"])["K_theta_oracle"]),
            "K_u_oracle": summarise(df.drop_duplicates(["n", "seed"])["K_u_oracle"]),
            "coord_offset": summarise(df.drop_duplicates(["n", "seed"])["coord_offset"]),
            "Delta_u_total": summarise(df["Delta_u_total"]),
            "Delta_u_mean": summarise(df["Delta_u_mean"]),
            "Delta_u_cov": summarise(df["Delta_u_cov"]),
            "self_consistency_kl_u": summarise(df["self_consistency_kl_u"]),
        },
        "notes": [
            "K_theta_oracle is the theta-space BvM-premise diagnostic.",
            "coord_offset = K_u_oracle - K_theta_oracle is the coordinate-projection offset.",
            "Delta_u_total is native u-space Gaussian-NPE approximation error, not a pure BvM residual.",
            "Qhat_N,u is reconstructed from sample moments of saved Gaussian-NPE posterior samples.",
            "The affine standardisation constants mu_eta/sigma_eta are not regenerated for the full grid. Exact KL and analytic Gaussian-Gaussian decomposition terms are invariant to that common affine map, but finite-sample kNN K_u^* estimates may differ slightly after diagonal standardisation.",
            "Sample-based Perez-Cruz KL estimates are retried with deterministic tiny jitter eps * max(pooled_sd, 1.0) only when duplicate/zero nearest-neighbour distances produce non-finite values.",
        ],
    }
    json_path = args.output_dir / f"{args.output_prefix}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {json_path}")

    print("\nHeadline medians:")
    for key_name, value in summary["summaries"].items():
        print(
            f"  {key_name}: median={value['median']:.6g}, "
            f"IQR=[{value['q25']:.6g}, {value['q75']:.6g}]"
        )


if __name__ == "__main__":
    main()
