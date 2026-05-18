#!/usr/bin/env python
"""Evaluate the completed GNK HPC calibration with reviewed u-space KL logic.

This script is intentionally read-only with respect to GNK caches and
calibration outputs. It writes diagnostics into an ``evaluation/`` subdirectory
under the completed calibration run.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "res"
    / "gnk_hpc_calibration"
    / "gnk_gaussian_npe_n500_x50_seed88_20260425T065035Z"
)
DEFAULT_NUTS_CACHE = REPO_ROOT / "res" / "gnk" / "nuts_cache_v2_n_obs_500_seed_88.pkl"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    except Exception:
        return True


def load_reviewed_decomp_module() -> Any:
    path = REPO_ROOT / "scripts" / "compute_gnk_u_space_kl_decomp.py"
    spec = importlib.util.spec_from_file_location("gnk_u_space_decomp", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load reviewed decomposition helper from {rel(path)}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_shape(name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def gaussian_kl_between(decomp: Any, p_mean: np.ndarray, p_cov: np.ndarray, q_mean: np.ndarray, q_cov: np.ndarray) -> float:
    total, _, _, _, _ = decomp.gaussian_kl_decomp_from_moments(p_mean, p_cov, q_mean, q_cov)
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--nuts-cache", type=Path, default=DEFAULT_NUTS_CACHE)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--x", type=int, default=50)
    parser.add_argument("--N", type=int, default=3_025_000)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--self-check-metric-size", type=int, default=2000)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    gaussian_path = run_dir / "gaussian_npe_u_posterior.npz"
    samples_path = run_dir / "posterior_samples_10k.npz"
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    required_paths = {
        "gaussian_npe_u_posterior": gaussian_path,
        "posterior_samples_10k": samples_path,
        "nuts_cache": args.nuts_cache.resolve(),
    }
    missing = [f"{key}: {rel(path)}" for key, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + "; ".join(missing))

    decomp = load_reviewed_decomp_module()
    if decomp.D_THETA != 4 or decomp.D_TOTAL != 11:
        raise RuntimeError("Reviewed decomposition helper constants do not match GNK octile setup.")

    with np.load(gaussian_path) as posterior:
        mu_u = np.asarray(posterior["mu_u"], dtype=np.float64)
        cov_u = np.asarray(posterior["cov_u"], dtype=np.float64)
        theta_unbounded_mean = np.asarray(posterior["theta_unbounded_mean"], dtype=np.float64)
        theta_unbounded_std = np.asarray(posterior["theta_unbounded_std"], dtype=np.float64)

    require_shape("mu_u", mu_u, (4,))
    require_shape("cov_u", cov_u, (4, 4))
    require_shape("theta_unbounded_mean", theta_unbounded_mean, (4,))
    require_shape("theta_unbounded_std", theta_unbounded_std, (4,))
    if np.any(theta_unbounded_std <= 0):
        raise ValueError("theta_unbounded_std must be positive")

    with np.load(samples_path) as samples:
        theta = np.asarray(samples["theta"], dtype=np.float64)
        sample_u = np.asarray(samples["u"], dtype=np.float64)
        eta = np.asarray(samples["eta"], dtype=np.float64)

    require_shape("posterior_samples.theta", theta, (10_000, 4))
    require_shape("posterior_samples.u", sample_u, (10_000, 4))
    require_shape("posterior_samples.eta", eta, (10_000, 4))

    eta_from_theta, theta_clip_count = decomp.theta_to_u_affine_invariant(theta)
    eta_from_u = sample_u * theta_unbounded_std + theta_unbounded_mean
    theta_from_eta = 10.0 / (1.0 + np.exp(-eta))
    eta_theta_max_abs_diff = float(np.max(np.abs(eta - eta_from_theta)))
    eta_u_max_abs_diff = float(np.max(np.abs(eta - eta_from_u)))
    theta_eta_max_abs_diff = float(np.max(np.abs(theta - theta_from_eta)))

    scale = np.diag(theta_unbounded_std)
    direct_eta_mean = mu_u * theta_unbounded_std + theta_unbounded_mean
    direct_eta_cov = scale @ cov_u @ scale

    oracle = decomp.compute_oracle_moments(REPO_ROOT / "res" / "gnk", args.n, args.seed)
    if oracle is None:
        raise FileNotFoundError(f"Reviewed helper could not find NUTS cache for n={args.n}, seed={args.seed}")
    if oracle.nuts_path.resolve() != args.nuts_cache.resolve():
        raise RuntimeError(
            f"Reviewed helper selected {rel(oracle.nuts_path)}, expected {rel(args.nuts_cache.resolve())}"
        )

    sample_eta_mean = eta.mean(axis=0)
    sample_eta_cov = np.cov(eta, rowvar=False)
    sample_total, sample_mean, sample_cov, oracle_min_eig, sample_qhat_min_eig = (
        decomp.gaussian_kl_decomp_from_moments(
            oracle.u_mean,
            oracle.u_cov,
            sample_eta_mean,
            sample_eta_cov,
        )
    )
    direct_total, direct_mean, direct_cov, _, direct_qhat_min_eig = (
        decomp.gaussian_kl_decomp_from_moments(
            oracle.u_mean,
            oracle.u_cov,
            direct_eta_mean,
            direct_eta_cov,
        )
    )

    self_sample = decomp.self_consistency_kl_u(
        eta,
        sample_eta_mean,
        sample_eta_cov,
        seed=args.seed + args.N + 30_000,
        n_metric=args.self_check_metric_size,
    )
    self_direct = decomp.self_consistency_kl_u(
        eta,
        direct_eta_mean,
        direct_eta_cov,
        seed=args.seed + args.N + 31_000,
        n_metric=args.self_check_metric_size,
    )

    _, _, sample_cov_min_eig = decomp.stable_cov_matrix(sample_eta_cov)
    _, _, direct_cov_u_min_eig = decomp.stable_cov_matrix(cov_u)
    _, _, direct_cov_eta_min_eig = decomp.stable_cov_matrix(direct_eta_cov)

    sample_vs_direct_kl = gaussian_kl_between(
        decomp,
        sample_eta_mean,
        sample_eta_cov,
        direct_eta_mean,
        direct_eta_cov,
    )
    direct_vs_sample_kl = gaussian_kl_between(
        decomp,
        direct_eta_mean,
        direct_eta_cov,
        sample_eta_mean,
        sample_eta_cov,
    )

    finite_components = [
        oracle.K_theta_oracle,
        oracle.K_u_oracle,
        oracle.coord_offset,
        sample_total,
        sample_mean,
        sample_cov,
        direct_total,
        direct_mean,
        direct_cov,
        self_sample,
        self_direct,
    ]
    schema_checks = {
        "required_files_exist": True,
        "mu_u_shape": list(mu_u.shape),
        "cov_u_shape": list(cov_u.shape),
        "posterior_theta_shape": list(theta.shape),
        "posterior_u_shape": list(sample_u.shape),
        "posterior_eta_shape": list(eta.shape),
        "all_required_arrays_finite": True,
        "theta_clip_count_when_recomputing_eta": int(theta_clip_count),
        "eta_from_theta_max_abs_diff": eta_theta_max_abs_diff,
        "eta_from_saved_u_max_abs_diff": eta_u_max_abs_diff,
        "theta_from_eta_max_abs_diff": theta_eta_max_abs_diff,
        "direct_cov_u_min_eig": direct_cov_u_min_eig,
        "direct_cov_eta_min_eig": direct_cov_eta_min_eig,
        "sample_eta_cov_min_eig": sample_cov_min_eig,
        "finite_decomposition_components": bool(np.isfinite(finite_components).all()),
        "covariance_component_nonnegative_with_tolerance": bool(sample_cov >= -1e-8 and direct_cov >= -1e-8),
    }

    created_at = datetime.now(timezone.utc).isoformat()
    command = "python " + " ".join(sys.argv)
    result = {
        "task": "gnk-hpc-calibration-u-space-decomposition-evaluation",
        "created_at_utc": created_at,
        "git_commit": git(["rev-parse", "--short", "HEAD"]),
        "git_commit_full": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "git_dirty": git_dirty(),
        "command": command,
        "reviewed_decomposition_script": "scripts/compute_gnk_u_space_kl_decomp.py",
        "reviewed_convention_note": (
            "Primary reported Delta_N,u uses sample moments of saved posterior eta samples, matching "
            "the reviewed decomposition script's affine-equivalent logit coordinate. The direct saved "
            "Gaussian-NPE mu_u/cov_u are converted from standardized u to eta and reported as a "
            "reconstruction check."
        ),
        "n": args.n,
        "x": args.x,
        "N": args.N,
        "seed": args.seed,
        "d_s": decomp.D_S,
        "d_theta": decomp.D_THETA,
        "d_total": decomp.D_TOTAL,
        "scaled_budget": args.N / (decomp.D_TOTAL * decomp.D_TOTAL * args.n),
        "inputs": {key: rel(path) for key, path in required_paths.items()},
        "outputs": {
            "json": rel(output_dir / f"{args.output_prefix}.json"),
            "csv": rel(output_dir / f"{args.output_prefix}.csv"),
        },
        "schema_checks": schema_checks,
        "oracle": {
            "K_theta_star": float(oracle.K_theta_oracle),
            "K_u_star": float(oracle.K_u_oracle),
            "coord_offset": float(oracle.coord_offset),
            "nuts_sample_count": int(oracle.nuts_sample_count),
            "nuts_theta_clip_count": int(oracle.nuts_u_clip_count),
            "theta_kl_jitter": float(oracle.theta_kl_jitter),
            "u_kl_jitter": float(oracle.u_kl_jitter),
            "oracle_min_eig_eta": float(oracle_min_eig),
        },
        "reviewed_sample_moment_decomposition": {
            "coordinate": "eta = logit(theta / 10), affine-equivalent to standardized u",
            "qhat_source": "posterior_samples_10k.npz eta sample moments",
            "Delta_N_u": float(sample_total),
            "Delta_N_u_mean_component": float(sample_mean),
            "Delta_N_u_covariance_component": float(sample_cov),
            "qhat_min_eig_eta": float(sample_qhat_min_eig),
            "self_consistency_kl_reconstructed_Qhat_N_u": float(self_sample),
        },
        "direct_saved_gaussian_reconstruction": {
            "coordinate": "direct gaussian_npe_u_posterior.npz mu_u/cov_u converted to eta",
            "Delta_N_u": float(direct_total),
            "Delta_N_u_mean_component": float(direct_mean),
            "Delta_N_u_covariance_component": float(direct_cov),
            "qhat_min_eig_eta": float(direct_qhat_min_eig),
            "self_consistency_kl_saved_Gaussian_against_saved_eta_samples": float(self_direct),
            "sample_moment_Qhat_to_direct_Gaussian_KL": float(sample_vs_direct_kl),
            "direct_Gaussian_to_sample_moment_Qhat_KL": float(direct_vs_sample_kl),
            "max_abs_mean_diff_eta": float(np.max(np.abs(sample_eta_mean - direct_eta_mean))),
            "frobenius_cov_diff_eta": float(np.linalg.norm(sample_eta_cov - direct_eta_cov, ord="fro")),
            "relative_frobenius_cov_diff_eta": float(
                np.linalg.norm(sample_eta_cov - direct_eta_cov, ord="fro")
                / max(np.linalg.norm(direct_eta_cov, ord="fro"), np.finfo(float).eps)
            ),
        },
        "gate_status": {
            "schema_compatible": bool(
                schema_checks["required_files_exist"]
                and schema_checks["mu_u_shape"] == [4]
                and schema_checks["cov_u_shape"] == [4, 4]
                and schema_checks["posterior_theta_shape"] == [10_000, 4]
                and schema_checks["posterior_u_shape"] == [10_000, 4]
                and schema_checks["posterior_eta_shape"] == [10_000, 4]
            ),
            "finite_sane_decomposition": bool(
                schema_checks["finite_decomposition_components"]
                and schema_checks["covariance_component_nonnegative_with_tolerance"]
                and sample_qhat_min_eig > 0.0
                and direct_qhat_min_eig > 0.0
            ),
            "passes_operational_evaluation_gate": bool(
                schema_checks["finite_decomposition_components"]
                and schema_checks["covariance_component_nonnegative_with_tolerance"]
                and sample_qhat_min_eig > 0.0
                and direct_qhat_min_eig > 0.0
            ),
        },
        "interpretation": (
            "The high-budget calibration is schema-compatible and finite under the reviewed u-space "
            "decomposition convention if the gate flags are true. The magnitude of Delta_N,u is a "
            "native-coordinate Gaussian-NPE approximation diagnostic, not a pure BvM target-Gaussianity "
            "residual and not the raw metrics.json KL/MMD."
        ),
    }

    json_path = output_dir / f"{args.output_prefix}.json"
    csv_path = output_dir / f"{args.output_prefix}.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    row = {
        "created_at_utc": created_at,
        "git_commit": result["git_commit"],
        "n": args.n,
        "x": args.x,
        "N": args.N,
        "seed": args.seed,
        "K_theta_star": result["oracle"]["K_theta_star"],
        "K_u_star": result["oracle"]["K_u_star"],
        "coord_offset": result["oracle"]["coord_offset"],
        "Delta_N_u": result["reviewed_sample_moment_decomposition"]["Delta_N_u"],
        "Delta_N_u_mean_component": result["reviewed_sample_moment_decomposition"]["Delta_N_u_mean_component"],
        "Delta_N_u_covariance_component": result["reviewed_sample_moment_decomposition"]["Delta_N_u_covariance_component"],
        "self_consistency_kl_reconstructed_Qhat_N_u": result["reviewed_sample_moment_decomposition"][
            "self_consistency_kl_reconstructed_Qhat_N_u"
        ],
        "direct_saved_gaussian_Delta_N_u": result["direct_saved_gaussian_reconstruction"]["Delta_N_u"],
        "direct_saved_gaussian_Delta_N_u_mean_component": result["direct_saved_gaussian_reconstruction"][
            "Delta_N_u_mean_component"
        ],
        "direct_saved_gaussian_Delta_N_u_covariance_component": result["direct_saved_gaussian_reconstruction"][
            "Delta_N_u_covariance_component"
        ],
        "sample_moment_Qhat_to_direct_Gaussian_KL": result["direct_saved_gaussian_reconstruction"][
            "sample_moment_Qhat_to_direct_Gaussian_KL"
        ],
        "schema_compatible": result["gate_status"]["schema_compatible"],
        "finite_sane_decomposition": result["gate_status"]["finite_sane_decomposition"],
        "passes_operational_evaluation_gate": result["gate_status"]["passes_operational_evaluation_gate"],
        "json_output": rel(json_path),
    }
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {rel(json_path)}")
    print(f"Wrote {rel(csv_path)}")


if __name__ == "__main__":
    main()
