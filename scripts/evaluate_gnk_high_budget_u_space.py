#!/usr/bin/env python
"""Evaluate completed-only GNK high-budget Gaussian-NPE rows in u-space.

The script consumes the post-run audit for the bounded empirical-GNK
high-budget array, evaluates only rows classified ``complete`` plus the single
``reuse`` row, and writes summary tables/figures without modifying any cache or
run directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT = REPO_ROOT / "res" / "gnk_high_budget" / "post_run_audit_20260429T214347Z.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "res" / "gnk_high_budget" / "evaluation"
DEFAULT_EXISTING_DECOMP = (
    REPO_ROOT / "notebooks" / "plots" / "gnk_u_space_kl_decomp_20260425_per_seed.csv"
)
EXPECTED_SEEDS_PER_X = 101


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def repo_path(path: str | Path | None) -> Path | None:
    if path in {None, ""}:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


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


def parse_only(spec: str | None) -> set[tuple[int, int]] | None:
    if spec is None or spec.strip().lower() in {"", "all"}:
        return None
    out: set[tuple[int, int]] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("--only entries must have form x:seed, e.g. 25:0,50:88")
        x_s, seed_s = part.split(":", 1)
        out.add((int(x_s), int(seed_s)))
    return out


def require_shape(name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def shape_text(array: np.ndarray) -> str:
    return "x".join(str(v) for v in array.shape)


def gaussian_kl_between(decomp: Any, p_mean: np.ndarray, p_cov: np.ndarray, q_mean: np.ndarray, q_cov: np.ndarray) -> float:
    total, _, _, _, _ = decomp.gaussian_kl_decomp_from_moments(p_mean, p_cov, q_mean, q_cov)
    return float(total)


def finite_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"median": math.nan, "q25": math.nan, "q75": math.nan}
    return {
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def no_overwrite(paths: list[Path], allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [rel(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output paths. Use --allow-overwrite only for intentional reruns: "
            + ", ".join(existing)
        )


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_manifest(audit: dict[str, Any]) -> dict[int, dict[str, Any]]:
    manifest_path = repo_path(audit.get("manifest_path"))
    if manifest_path is None or not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    return {int(row["manifest_index"]): row for row in manifest.get("rows", [])}


def row_path(row: dict[str, Any], manifest_row: dict[str, Any] | None, audit_key: str, manifest_key: str) -> str:
    if row.get(audit_key):
        return str(row[audit_key])
    if manifest_row is not None:
        paths = manifest_row.get("paths", {})
        if paths.get(manifest_key):
            return str(paths[manifest_key])
    return ""


def evaluate_row(
    audit_row: dict[str, Any],
    manifest_row: dict[str, Any] | None,
    oracle_cache: dict[tuple[int, int], Any],
    decomp: Any,
    self_check_metric_size: int,
    created_at: str,
    commit: str,
    audit_path: Path,
) -> dict[str, Any]:
    n = int(audit_row["n"])
    x = int(audit_row["x"])
    N = int(audit_row["N"])
    seed = int(audit_row["seed"])

    gaussian_rel = row_path(audit_row, manifest_row, "gaussian_npe_u_posterior_npz_path", "predicted_u_mean_cov")
    samples_rel = row_path(audit_row, manifest_row, "posterior_samples_10k_npz_path", "samples_10k")
    timing_rel = row_path(audit_row, manifest_row, "timing_metadata_json_path", "timing_metadata")
    validation_rel = row_path(audit_row, manifest_row, "validation_curve_csv_path", "validation_curve")
    stdout_rel = row_path(audit_row, manifest_row, "stdout_log_path", "stdout_log")
    stderr_rel = row_path(audit_row, manifest_row, "stderr_log_path", "stderr_log")
    metrics_rel = row_path(audit_row, manifest_row, "metrics_json_path", "metrics")
    pkl_rel = row_path(audit_row, manifest_row, "posterior_samples_pkl_path", "posterior_samples_pkl")
    config_rel = ""
    reuse_marker_rel = ""
    if manifest_row is not None:
        config_rel = str(manifest_row.get("config_path") or manifest_row.get("paths", {}).get("config") or "")
        reuse_marker_rel = str(manifest_row.get("reuse_marker_path") or "")

    required_paths = {
        "gaussian_npe_u_posterior": repo_path(gaussian_rel),
        "posterior_samples_10k": repo_path(samples_rel),
    }
    missing = [f"{name}: {rel(path)}" for name, path in required_paths.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs for x={x}, seed={seed}: " + "; ".join(missing))

    gaussian_path = required_paths["gaussian_npe_u_posterior"]
    samples_path = required_paths["posterior_samples_10k"]
    assert gaussian_path is not None
    assert samples_path is not None

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
        raise ValueError(f"theta_unbounded_std must be positive for x={x}, seed={seed}")

    with np.load(samples_path) as samples:
        theta = np.asarray(samples["theta"], dtype=np.float64)
        sample_u = np.asarray(samples["u"], dtype=np.float64)
        eta = np.asarray(samples["eta"], dtype=np.float64)

    require_shape("posterior_samples.theta", theta, (10_000, 4))
    require_shape("posterior_samples.u", sample_u, (10_000, 4))
    require_shape("posterior_samples.eta", eta, (10_000, 4))

    key = (n, seed)
    if key not in oracle_cache:
        oracle = decomp.compute_oracle_moments(REPO_ROOT / "res" / "gnk", n, seed)
        if oracle is None:
            raise FileNotFoundError(f"Reviewed helper could not find NUTS cache for n={n}, seed={seed}")
        oracle_cache[key] = oracle
    oracle = oracle_cache[key]

    eta_from_theta, theta_clip_count = decomp.theta_to_u_affine_invariant(theta)
    eta_from_u = sample_u * theta_unbounded_std + theta_unbounded_mean
    theta_from_eta = 10.0 / (1.0 + np.exp(-eta))
    eta_theta_max_abs_diff = float(np.max(np.abs(eta - eta_from_theta)))
    eta_u_max_abs_diff = float(np.max(np.abs(eta - eta_from_u)))
    theta_eta_max_abs_diff = float(np.max(np.abs(theta - theta_from_eta)))

    scale = np.diag(theta_unbounded_std)
    direct_eta_mean = mu_u * theta_unbounded_std + theta_unbounded_mean
    direct_eta_cov = scale @ cov_u @ scale
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
        seed=seed + N + 40_000,
        n_metric=self_check_metric_size,
    )
    self_direct = decomp.self_consistency_kl_u(
        eta,
        direct_eta_mean,
        direct_eta_cov,
        seed=seed + N + 41_000,
        n_metric=self_check_metric_size,
    )
    _, _, sample_eta_cov_min_eig = decomp.stable_cov_matrix(sample_eta_cov)
    _, _, direct_cov_u_min_eig = decomp.stable_cov_matrix(cov_u)
    _, _, direct_cov_eta_min_eig = decomp.stable_cov_matrix(direct_eta_cov)
    sample_vs_direct_kl = gaussian_kl_between(
        decomp, sample_eta_mean, sample_eta_cov, direct_eta_mean, direct_eta_cov
    )
    direct_vs_sample_kl = gaussian_kl_between(
        decomp, direct_eta_mean, direct_eta_cov, sample_eta_mean, sample_eta_cov
    )

    finite_components = np.asarray(
        [
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
        ],
        dtype=float,
    )
    schema_compatible = bool(
        mu_u.shape == (4,)
        and cov_u.shape == (4, 4)
        and theta.shape == (10_000, 4)
        and sample_u.shape == (10_000, 4)
        and eta.shape == (10_000, 4)
        and np.isfinite(mu_u).all()
        and np.isfinite(cov_u).all()
        and np.isfinite(theta).all()
        and np.isfinite(sample_u).all()
        and np.isfinite(eta).all()
    )
    covariance_nonnegative = bool(sample_cov >= -1e-8 and direct_cov >= -1e-8)
    finite_sane = bool(
        np.isfinite(finite_components).all()
        and covariance_nonnegative
        and sample_qhat_min_eig > 0.0
        and direct_qhat_min_eig > 0.0
    )

    return {
        "n": n,
        "N": N,
        "x": x,
        "d_s": int(decomp.D_S),
        "d_theta": int(decomp.D_THETA),
        "d_total": int(decomp.D_TOTAL),
        "seed": seed,
        "method": str(audit_row.get("method", "Gaussian-NPE")),
        "simulator": str(audit_row.get("simulator", "")),
        "manifest_index": int(audit_row.get("manifest_index", -1)),
        "pbs_array_index": audit_row.get("pbs_array_index", ""),
        "action": str(audit_row.get("action", "")),
        "classification": str(audit_row.get("classification", "")),
        "reuse_existing": bool(audit_row.get("classification") == "reuse" or audit_row.get("action") == "reuse"),
        "run_id": str(audit_row.get("run_id", "")),
        "K_theta_oracle": float(oracle.K_theta_oracle),
        "K_u_oracle": float(oracle.K_u_oracle),
        "coord_offset": float(oracle.coord_offset),
        "Delta_u_total": float(sample_total),
        "Delta_u_mean": float(sample_mean),
        "Delta_u_cov": float(sample_cov),
        "Delta_theta_total": float(sample_total + oracle.coord_offset),
        "scaled_budget": float(N / (decomp.D_TOTAL * decomp.D_TOTAL * n)),
        "direct_Delta_u_total": float(direct_total),
        "direct_Delta_u_mean": float(direct_mean),
        "direct_Delta_u_cov": float(direct_cov),
        "direct_Delta_theta_total": float(direct_total + oracle.coord_offset),
        "self_consistency_kl_u": float(self_sample),
        "direct_self_consistency_kl_u": float(self_direct),
        "sample_moment_Qhat_to_direct_Gaussian_KL": float(sample_vs_direct_kl),
        "direct_Gaussian_to_sample_moment_Qhat_KL": float(direct_vs_sample_kl),
        "max_abs_mean_diff_eta_sample_vs_direct": float(np.max(np.abs(sample_eta_mean - direct_eta_mean))),
        "frobenius_cov_diff_eta_sample_vs_direct": float(np.linalg.norm(sample_eta_cov - direct_eta_cov, ord="fro")),
        "relative_frobenius_cov_diff_eta_sample_vs_direct": float(
            np.linalg.norm(sample_eta_cov - direct_eta_cov, ord="fro")
            / max(np.linalg.norm(direct_eta_cov, ord="fro"), np.finfo(float).eps)
        ),
        "nuts_cache_path": rel(Path(oracle.nuts_path)),
        "gaussian_npe_dir": str(audit_row.get("output_dir", "")),
        "config_path": config_rel,
        "gaussian_npe_u_posterior_path": gaussian_rel,
        "posterior_samples_10k_path": samples_rel,
        "posterior_samples_pkl_path": pkl_rel,
        "metrics_json_path": metrics_rel,
        "timing_metadata_path": timing_rel,
        "validation_curve_csv_path": validation_rel,
        "stdout_log_path": stdout_rel,
        "stderr_log_path": stderr_rel,
        "reuse_marker_path": reuse_marker_rel,
        "audit_path": rel(audit_path),
        "start_timestamp_utc": str(audit_row.get("start_timestamp_utc", "")),
        "end_timestamp_utc": str(audit_row.get("end_timestamp_utc", "")),
        "total_wall_time_seconds": audit_row.get("total_wall_time_seconds", math.nan),
        "simulation_time_seconds": audit_row.get("simulation_time_seconds", math.nan),
        "training_time_seconds": audit_row.get("training_time_seconds", math.nan),
        "posterior_sampling_time_seconds": audit_row.get("posterior_sampling_time_seconds", math.nan),
        "peak_memory_kb": audit_row.get("peak_memory_kb", math.nan),
        "hostname": str(audit_row.get("hostname", "")),
        "scheduler_job_id": str(audit_row.get("scheduler_job_id", "")),
        "stderr_log_size_bytes": audit_row.get("stderr_log_size_bytes", math.nan),
        "stdout_log_size_bytes": audit_row.get("stdout_log_size_bytes", math.nan),
        "nuts_sample_count": int(oracle.nuts_sample_count),
        "gnpe_sample_count": int(len(theta)),
        "oracle_fit_size": min(decomp.FIT_SIZE, max(2, int(oracle.nuts_sample_count) // 2)),
        "oracle_metric_size": min(
            decomp.N_METRIC,
            max(1, int(oracle.nuts_sample_count) - min(decomp.FIT_SIZE, max(2, int(oracle.nuts_sample_count) // 2))),
        ),
        "self_consistency_metric_size": int(min(self_check_metric_size, len(eta))),
        "K_theta_oracle_kl_jitter": float(oracle.theta_kl_jitter),
        "K_u_oracle_kl_jitter": float(oracle.u_kl_jitter),
        "theta_clip_count_nuts": int(oracle.nuts_u_clip_count),
        "theta_clip_count_samples": int(theta_clip_count),
        "mu_u_shape": shape_text(mu_u),
        "cov_u_shape": shape_text(cov_u),
        "posterior_theta_shape": shape_text(theta),
        "posterior_u_shape": shape_text(sample_u),
        "posterior_eta_shape": shape_text(eta),
        "all_required_arrays_finite": True,
        "eta_from_theta_max_abs_diff": eta_theta_max_abs_diff,
        "eta_from_saved_u_max_abs_diff": eta_u_max_abs_diff,
        "theta_from_eta_max_abs_diff": theta_eta_max_abs_diff,
        "oracle_min_eig_eta": float(oracle_min_eig),
        "sample_eta_cov_min_eig": float(sample_eta_cov_min_eig),
        "direct_cov_u_min_eig": float(direct_cov_u_min_eig),
        "direct_cov_eta_min_eig": float(direct_cov_eta_min_eig),
        "sample_qhat_min_eig_eta": float(sample_qhat_min_eig),
        "direct_qhat_min_eig_eta": float(direct_qhat_min_eig),
        "finite_decomposition_components": bool(np.isfinite(finite_components).all()),
        "covariance_component_nonnegative_with_tolerance": covariance_nonnegative,
        "schema_compatible": schema_compatible,
        "finite_sane_decomposition": finite_sane,
        "passes_operational_evaluation_gate": bool(schema_compatible and finite_sane),
        "qhat_reconstruction": "sample_moments_of_posterior_samples_10k_eta_primary; direct_saved_gaussian_reported_as_diagnostic",
        "u_affine_note": (
            "eta = logit(theta / 10) is affine-equivalent to saved standardized u; exact Gaussian-Gaussian "
            "KL terms are invariant to the common affine standardisation"
        ),
        "created_at_utc": created_at,
        "commit": commit,
    }


def build_group_summary(df: pd.DataFrame, failed_rows: list[dict[str, Any]]) -> pd.DataFrame:
    failed_by_x = {
        int(x): group.sort_values("seed")["seed"].astype(int).tolist()
        for x, group in pd.DataFrame(failed_rows).groupby("x")
    } if failed_rows else {}
    records: list[dict[str, Any]] = []
    summary_cols = [
        "K_theta_oracle",
        "K_u_oracle",
        "coord_offset",
        "Delta_u_total",
        "Delta_u_mean",
        "Delta_u_cov",
        "Delta_theta_total",
        "direct_Delta_u_total",
        "direct_Delta_u_mean",
        "direct_Delta_u_cov",
        "self_consistency_kl_u",
        "direct_self_consistency_kl_u",
        "sample_moment_Qhat_to_direct_Gaussian_KL",
        "total_wall_time_seconds",
        "training_time_seconds",
        "simulation_time_seconds",
        "peak_memory_kb",
    ]
    for (n, x, N, scaled_budget), group in df.groupby(["n", "x", "N", "scaled_budget"], sort=True):
        failed_seeds = failed_by_x.get(int(x), [])
        n_usable = int(group["seed"].nunique())
        n_reuse = int(group["reuse_existing"].sum())
        n_complete = int((group["classification"] == "complete").sum())
        complete_grid = n_usable == EXPECTED_SEEDS_PER_X and len(failed_seeds) == 0
        status = "complete theorem-facing high-budget evidence" if complete_grid else "incomplete diagnostic"
        rec: dict[str, Any] = {
            "n": int(n),
            "x": int(x),
            "N": int(N),
            "scaled_budget": float(scaled_budget),
            "n_usable_seeds": n_usable,
            "n_complete_new_rows": n_complete,
            "n_reuse_rows": n_reuse,
            "n_failed_excluded_rows": int(len(failed_seeds)),
            "failed_excluded_seeds": ",".join(str(s) for s in failed_seeds),
            "expected_seeds": EXPECTED_SEEDS_PER_X,
            "status_label": status,
        }
        for col in summary_cols:
            stats = finite_summary(group[col])
            rec[f"{col}_median"] = stats["median"]
            rec[f"{col}_q25"] = stats["q25"]
            rec[f"{col}_q75"] = stats["q75"]
        records.append(rec)
    return pd.DataFrame(records).sort_values(["x", "N"]).reset_index(drop=True)


def plot_delta_components(group_summary: pd.DataFrame, out_path: Path) -> None:
    labels = [
        f"x={int(row.x)}\n{int(row.n_usable_seeds)} seeds"
        + ("\nincomplete" if "incomplete" in row.status_label else "")
        for row in group_summary.itertuples(index=False)
    ]
    x_pos = np.arange(len(group_summary))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = {"mean": "#4C78A8", "cov": "#F58518"}
    for offset, key, label in [
        (-width / 2, "Delta_u_mean", r"$\Delta_{\mathrm{mean},u}$"),
        (width / 2, "Delta_u_cov", r"$\Delta_{\mathrm{cov},u}$"),
    ]:
        med = group_summary[f"{key}_median"].to_numpy(dtype=float)
        q25 = group_summary[f"{key}_q25"].to_numpy(dtype=float)
        q75 = group_summary[f"{key}_q75"].to_numpy(dtype=float)
        yerr = np.vstack([med - q25, q75 - med])
        ax.bar(
            x_pos + offset,
            med,
            width=width,
            yerr=yerr,
            capsize=4,
            color=colors["mean" if "mean" in key else "cov"],
            label=label,
        )
    totals = group_summary["Delta_u_total_median"].to_numpy(dtype=float)
    ax.scatter(x_pos, totals, marker="D", s=42, color="#222222", label=r"median $\Delta_{N,u}$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("u-space KL component (nats)")
    ax.set_title("High-budget empirical-GNK Gaussian-NPE u-space decomposition")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scaled_budget_with_existing(high_df: pd.DataFrame, existing_csv: Path, out_path: Path) -> dict[str, Any]:
    if not existing_csv.exists():
        raise FileNotFoundError(f"Missing existing reviewed decomposition CSV: {rel(existing_csv)}")
    existing = pd.read_csv(existing_csv)
    required = {"n", "N", "seed", "scaled_budget", "Delta_u_total"}
    missing = sorted(required - set(existing.columns))
    if missing:
        raise ValueError(f"Existing decomposition CSV missing required columns: {missing}")
    existing = existing[existing["N"] > existing["n"]].copy()
    counts = existing.groupby(["n", "N"])["seed"].nunique().reset_index(name="n_seeds")
    complete_keys = counts[counts["n_seeds"] >= EXPECTED_SEEDS_PER_X][["n", "N"]]
    existing = existing.merge(complete_keys, on=["n", "N"], how="inner")
    grouped = (
        existing.groupby(["n", "N", "scaled_budget"], as_index=False)
        .agg(
            Delta_u_total_median=("Delta_u_total", "median"),
            Delta_u_total_q25=("Delta_u_total", lambda s: s.quantile(0.25)),
            Delta_u_total_q75=("Delta_u_total", lambda s: s.quantile(0.75)),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["n", "scaled_budget", "N"])
        .reset_index(drop=True)
    )

    high_grouped = (
        high_df.groupby(["n", "x", "N", "scaled_budget"], as_index=False)
        .agg(
            Delta_u_total_median=("Delta_u_total", "median"),
            Delta_u_total_q25=("Delta_u_total", lambda s: s.quantile(0.25)),
            Delta_u_total_q75=("Delta_u_total", lambda s: s.quantile(0.75)),
            n_seeds=("seed", "nunique"),
            n_reuse=("reuse_existing", "sum"),
        )
        .sort_values(["scaled_budget", "x"])
        .reset_index(drop=True)
    )

    n_values = sorted(grouped["n"].unique())
    cmap = plt.colormaps["viridis"]
    colors = {n: cmap(i / max(1, len(n_values) - 1)) for i, n in enumerate(n_values)}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for n in n_values:
        sub = grouped[grouped["n"] == n].sort_values("scaled_budget")
        ax.plot(
            sub["scaled_budget"],
            sub["Delta_u_total_median"],
            marker="o",
            linewidth=1.3,
            color=colors[n],
            label=f"reviewed cached n={int(n)}",
        )
        ax.fill_between(
            sub["scaled_budget"].to_numpy(dtype=float),
            sub["Delta_u_total_q25"].to_numpy(dtype=float),
            sub["Delta_u_total_q75"].to_numpy(dtype=float),
            color=colors[n],
            alpha=0.13,
            linewidth=0.0,
        )

    for row in high_grouped.itertuples(index=False):
        yerr = np.array([[row.Delta_u_total_median - row.Delta_u_total_q25], [row.Delta_u_total_q75 - row.Delta_u_total_median]])
        marker = "s" if int(row.x) == 25 else "X"
        label = f"high-budget x={int(row.x)}"
        if int(row.x) == 50:
            label += " incomplete"
        ax.errorbar(
            [row.scaled_budget],
            [row.Delta_u_total_median],
            yerr=yerr,
            marker=marker,
            markersize=7.5,
            color="#222222",
            ecolor="#222222",
            capsize=4,
            linestyle="none",
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"scaled budget $N/(d_{\mathrm{total}}^2 n)$")
    ax.set_ylabel(r"median $\Delta_{N,u} = \mathrm{KL}(G_u^*\,\|\,\widehat Q_{N,u})$ (nats)")
    ax.set_title("GNK Gaussian-NPE native u-space error with high-budget rows")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "historical_complete_groups": int(grouped[["n", "N"]].drop_duplicates().shape[0]),
        "historical_rows_plotted": int(len(existing)),
        "high_budget_groups": int(len(high_grouped)),
        "high_budget_rows_plotted": int(len(high_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--existing-decomp-csv", type=Path, default=DEFAULT_EXISTING_DECOMP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--only", type=str, default=None, help="Optional comma list of x:seed rows, e.g. 25:0,50:88")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--self-check-metric-size", type=int, default=2000)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    audit_path = args.audit.resolve()
    audit = load_json(audit_path)
    manifest_rows = load_manifest(audit)
    only = parse_only(args.only)
    created_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = args.output_prefix or f"gnk_high_budget_u_space_decomp_{timestamp}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = args.output_dir / f"{prefix}_per_seed.csv"
    group_summary_path = args.output_dir / f"{prefix}_group_summary.csv"
    summary_path = args.output_dir / f"{prefix}_summary.json"
    delta_plot_path = args.output_dir / f"{prefix}_delta_u_components.pdf"
    scaled_plot_path = args.output_dir / f"{prefix}_scaled_budget_with_existing.pdf"
    plot_metadata_path = args.output_dir / f"{prefix}_plot_metadata.json"
    output_paths = [
        per_seed_path,
        group_summary_path,
        summary_path,
        delta_plot_path,
        scaled_plot_path,
        plot_metadata_path,
    ]
    no_overwrite(output_paths, args.allow_overwrite)

    rows = audit.get("rows", [])
    eligible = [row for row in rows if row.get("classification") in {"complete", "reuse"}]
    failed_rows = [row for row in rows if row.get("classification") == "failed_exit_status"]
    if only is not None:
        eligible = [row for row in eligible if (int(row["x"]), int(row["seed"])) in only]
    if args.max_rows is not None:
        eligible = eligible[: args.max_rows]
    if not eligible:
        raise SystemExit("No eligible rows selected.")

    decomp = load_reviewed_decomp_module()
    if decomp.D_THETA != 4 or decomp.D_TOTAL != 11:
        raise RuntimeError("Reviewed decomposition helper constants do not match GNK octile setup.")

    commit = git(["rev-parse", "--short", "HEAD"])
    oracle_cache: dict[tuple[int, int], Any] = {}
    evaluated_rows: list[dict[str, Any]] = []
    for index, audit_row in enumerate(eligible, start=1):
        manifest_row = manifest_rows.get(int(audit_row.get("manifest_index", -1)))
        evaluated_rows.append(
            evaluate_row(
                audit_row,
                manifest_row,
                oracle_cache,
                decomp,
                self_check_metric_size=args.self_check_metric_size,
                created_at=created_at,
                commit=commit,
                audit_path=audit_path,
            )
        )
        if index == 1 or index % 25 == 0 or index == len(eligible):
            print(
                f"[{index}/{len(eligible)}] evaluated x={audit_row['x']} "
                f"seed={audit_row['seed']} classification={audit_row['classification']}"
            )

    df = pd.DataFrame(evaluated_rows).sort_values(["x", "seed"]).reset_index(drop=True)
    group_summary = build_group_summary(df, failed_rows)
    df.to_csv(per_seed_path, index=False)
    group_summary.to_csv(group_summary_path, index=False)

    plot_delta_components(group_summary, delta_plot_path)
    scaled_plot_info = plot_scaled_budget_with_existing(df, args.existing_decomp_csv.resolve(), scaled_plot_path)

    failed_records = [
        {
            "manifest_index": int(row.get("manifest_index", -1)),
            "x": int(row.get("x", -1)),
            "seed": int(row.get("seed", -1)),
            "output_dir": str(row.get("output_dir", "")),
            "classification": str(row.get("classification", "")),
        }
        for row in failed_rows
    ]
    by_x_counts = (
        df.groupby("x", as_index=False)
        .agg(
            evaluated_rows=("seed", "size"),
            usable_seeds=("seed", "nunique"),
            complete_rows=("classification", lambda s: int((s == "complete").sum())),
            reuse_rows=("reuse_existing", "sum"),
        )
        .to_dict(orient="records")
    )
    all_gates_pass = bool(df["passes_operational_evaluation_gate"].all())
    all_finite = bool(df["finite_decomposition_components"].all())
    cov_nonnegative = bool(df["covariance_component_nonnegative_with_tolerance"].all())
    full_expected = only is None and args.max_rows is None
    expected_rows = int(audit.get("classification_counts", {}).get("complete", 0)) + int(
        audit.get("classification_counts", {}).get("reuse", 0)
    )
    acceptance = {
        "expected_full_evaluated_rows": expected_rows,
        "evaluated_rows": int(len(df)),
        "full_run_selected": bool(full_expected),
        "full_row_count_matches": bool(len(df) == expected_rows),
        "all_schema_compatible": bool(df["schema_compatible"].all()),
        "all_finite_decomposition_components": all_finite,
        "all_covariance_components_nonnegative_with_tolerance": cov_nonnegative,
        "all_operational_gates_pass": all_gates_pass,
        "x25_usable_seeds": int(df.loc[df["x"] == 25, "seed"].nunique()),
        "x50_usable_seeds": int(df.loc[df["x"] == 50, "seed"].nunique()),
        "failed_rows_excluded": int(len(failed_rows)),
        "failed_x50_seeds_excluded": [int(row["seed"]) for row in failed_rows if int(row.get("x", -1)) == 50],
    }

    summary = {
        "task": "gnk-high-budget-completed-only-u-space-decomposition",
        "created_at_utc": created_at,
        "command": "python " + " ".join(sys.argv),
        "git_commit": commit,
        "git_commit_full": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "git_dirty": git_dirty(),
        "reviewed_decomposition_script": "scripts/compute_gnk_u_space_kl_decomp.py",
        "audit_path": rel(audit_path),
        "manifest_path": str(audit.get("manifest_path", "")),
        "existing_decomposition_csv": rel(args.existing_decomp_csv.resolve()),
        "outputs": {
            "per_seed_csv": rel(per_seed_path),
            "group_summary_csv": rel(group_summary_path),
            "summary_json": rel(summary_path),
            "delta_u_components_pdf": rel(delta_plot_path),
            "scaled_budget_with_existing_pdf": rel(scaled_plot_path),
            "plot_metadata_json": rel(plot_metadata_path),
        },
        "audit_classification_counts": audit.get("classification_counts", {}),
        "audit_classification_counts_by_x": audit.get("classification_counts_by_x", {}),
        "evaluated_counts_by_x": by_x_counts,
        "acceptance_checks": acceptance,
        "group_summary": group_summary.to_dict(orient="records"),
        "excluded_failed_rows": failed_records,
        "non_empty_stderr_logs": audit.get("non_empty_stderr_logs", []),
        "non_empty_stderr_note": audit.get("non_empty_stderr_note", ""),
        "notes": [
            "Only audit rows classified complete plus the reuse row are evaluated.",
            "Failed x=50 rows are excluded and listed in metadata; no failed row is rerun.",
            "Primary Delta_N,u is KL(G_u^* || Qhat_N,u), native-coordinate Gaussian-NPE error.",
            "K_theta_oracle remains the theta-space BvM target-Gaussianity diagnostic.",
            "x=25 is complete over 101 seeds; x=50 is labelled incomplete diagnostic over usable completed/reused seeds.",
            "Direct saved Gaussian diagnostics convert gaussian_npe_u_posterior.npz mu_u/cov_u back to eta and compare against saved eta sample moments.",
            "The script writes only evaluation artifacts and does not modify cache/run directories.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    plot_metadata = {
        "created_at_utc": created_at,
        "command": "python " + " ".join(sys.argv),
        "git_commit": commit,
        "git_commit_full": summary["git_commit_full"],
        "git_branch": summary["git_branch"],
        "git_dirty": summary["git_dirty"],
        "script": "scripts/evaluate_gnk_high_budget_u_space.py",
        "input_paths": {
            "audit": rel(audit_path),
            "existing_decomposition_csv": rel(args.existing_decomp_csv.resolve()),
        },
        "output_paths": {
            "delta_u_components_pdf": rel(delta_plot_path),
            "scaled_budget_with_existing_pdf": rel(scaled_plot_path),
        },
        "seed_counts": acceptance,
        "scaled_budget_plot": scaled_plot_info,
        "group_summary": group_summary.to_dict(orient="records"),
        "plot_notes": [
            "Historical reviewed groups use N>n and at least 101 seeds per (n,N).",
            "High-budget x=50 is plotted as incomplete diagnostic because 12 planned rows failed and were excluded.",
        ],
    }
    plot_metadata_path.write_text(json.dumps(plot_metadata, indent=2, sort_keys=True))

    print(f"Wrote {rel(per_seed_path)}")
    print(f"Wrote {rel(group_summary_path)}")
    print(f"Wrote {rel(summary_path)}")
    print(f"Wrote {rel(delta_plot_path)}")
    print(f"Wrote {rel(scaled_plot_path)}")
    print(f"Wrote {rel(plot_metadata_path)}")
    print("\nGroup medians:")
    for row in group_summary.itertuples(index=False):
        print(
            f"  x={int(row.x)} seeds={int(row.n_usable_seeds)} status={row.status_label}: "
            f"Delta_u_total median={row.Delta_u_total_median:.6g}, "
            f"IQR=[{row.Delta_u_total_q25:.6g}, {row.Delta_u_total_q75:.6g}]"
        )


if __name__ == "__main__":
    main()
