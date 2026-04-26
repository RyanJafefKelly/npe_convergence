"""Prepare a guarded GNK Gaussian-NPE high-budget PBS array dry-run.

This creates manifests and per-job configs only. It never submits the PBS array.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "res/gnk_high_budget"
CONFIG_DIR = OUTPUT_ROOT / "configs"
RUN_ROOT = OUTPUT_ROOT / "runs"
REUSE_MARKER_DIR = OUTPUT_ROOT / "reuse_markers"
DRIVER_LOG_DIR = OUTPUT_ROOT / "pbs_driver_logs"

EXISTING_CALIBRATION_DIR = (
    REPO_ROOT
    / "res/gnk_hpc_calibration/gnk_gaussian_npe_n500_x50_seed88_20260425T065035Z"
)

N_OBS = 500
D_S = 7
D_THETA = 4
D_TOTAL = D_S + D_THETA
X_VALUES = (25, 50)
SEEDS = tuple(range(101))
REUSE_X = 50
REUSE_SEED = 88
METHOD = "Gaussian-NPE"
SIMULATOR = "empirical_gnk_prior_predictive_octile_summaries"

RESOURCE_REQUEST = {
    "scheduler": "PBS",
    "job_type": "array",
    "array_indices": "0-200",
    "concurrency_cap": 20,
    "walltime": "47:00:00",
    "mem": "64GB",
    "ncpus": 4,
    "ngpus": 0,
    "resource_rationale": (
        "Matches the completed n=500, x=50, seed=88 calibration request. "
        "That run used about 6.5h wall time and about 1.7GB peak RSS, but "
        "the dry-run keeps the conservative 47h/64GB/4CPU envelope."
    ),
}

TRAIN_CONFIG = {
    "lr": 5e-4,
    "batch_size": 256,
    "max_epochs": 2000,
    "patience": 200,
    "val_frac": 0.1,
    "hidden_dims": [128, 128],
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_stamp(created_at: str) -> str:
    return datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def run_git(args: list[str], default: str = "unknown") -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)
    except Exception:
        return default
    return out.strip() or default


def git_dirty() -> bool:
    return bool(run_git(["status", "--porcelain"], default="dirty"))


def n_sims_for_x(x: int) -> int:
    return x * D_TOTAL * D_TOTAL * N_OBS


def expected_paths(run_id: str, config_path: Path) -> dict[str, str]:
    out = RUN_ROOT / run_id
    return {
        "output_dir": rel(out),
        "config": rel(config_path),
        "validation_curve": rel(out / "validation_curve.csv"),
        "validation_curve_plot": rel(out / "validation_curve.pdf"),
        "predicted_u_mean_cov": rel(out / "gaussian_npe_u_posterior.npz"),
        "samples_10k": rel(out / "posterior_samples_10k.npz"),
        "posterior_samples_pkl": rel(out / "posterior_samples.pkl"),
        "metrics": rel(out / "metrics.json"),
        "timing_metadata": rel(out / "timing_metadata.json"),
        "stdout_log": rel(out / "logs/stdout.log"),
        "stderr_log": rel(out / "logs/stderr.log"),
    }


def existing_required_paths() -> dict[str, Path]:
    return {
        "config": EXISTING_CALIBRATION_DIR / "config.yaml",
        "validation_curve": EXISTING_CALIBRATION_DIR / "validation_curve.csv",
        "validation_curve_plot": EXISTING_CALIBRATION_DIR / "validation_curve.pdf",
        "predicted_u_mean_cov": EXISTING_CALIBRATION_DIR / "gaussian_npe_u_posterior.npz",
        "samples_10k": EXISTING_CALIBRATION_DIR / "posterior_samples_10k.npz",
        "posterior_samples_pkl": EXISTING_CALIBRATION_DIR / "posterior_samples.pkl",
        "metrics": EXISTING_CALIBRATION_DIR / "metrics.json",
        "timing_metadata": EXISTING_CALIBRATION_DIR / "timing_metadata.json",
        "stdout_log": EXISTING_CALIBRATION_DIR / "logs/stdout.log",
        "stderr_log": EXISTING_CALIBRATION_DIR / "logs/stderr.log",
    }


def check_reuse_compatibility() -> dict[str, Any]:
    required = existing_required_paths()
    missing = [rel(path) for path in required.values() if not path.exists()]
    checks: dict[str, Any] = {
        "compatible": False,
        "source_output_dir": rel(EXISTING_CALIBRATION_DIR),
        "missing_required_paths": missing,
        "config_checks": {},
    }
    config_path = required["config"]
    if missing or not config_path.exists():
        return checks

    with config_path.open() as f:
        config = yaml.safe_load(f)
    expected = {
        "method": METHOD,
        "simulator": SIMULATOR,
        "n": N_OBS,
        "d": D_TOTAL,
        "x": REUSE_X,
        "N": n_sims_for_x(REUSE_X),
        "observed_seed": REUSE_SEED,
        "simulation_seed": REUSE_SEED,
        "training_seed": REUSE_SEED,
        "posterior_sampling_seed": REUSE_SEED,
    }
    config_checks = {key: config.get(key) == value for key, value in expected.items()}
    checks["config_checks"] = config_checks
    checks["compatible"] = all(config_checks.values())
    checks["expected_values"] = expected
    return checks


def no_overwrite_write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def no_overwrite_write_json(path: Path, payload: Any) -> None:
    no_overwrite_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_job_config(row: dict[str, Any], created_at: str) -> dict[str, Any]:
    seed = int(row["seed"])
    return {
        "task": "gnk-high-budget-array",
        "run_id": row["run_id"],
        "created_at_utc": created_at,
        "git_commit": run_git(["rev-parse", "--short", "HEAD"]),
        "git_dirty": git_dirty(),
        "branch": run_git(["branch", "--show-current"]),
        "method": METHOD,
        "simulator": SIMULATOR,
        "n": N_OBS,
        "d_s": D_S,
        "d_theta": D_THETA,
        "d": D_TOTAL,
        "x": int(row["x"]),
        "N": int(row["N"]),
        "observed_seed": seed,
        "simulation_seed": seed,
        "training_seed": seed,
        "posterior_sampling_seed": seed,
        "output_dir": row["output_dir"],
        "gpu_requested": False,
        "scheduler_resource_request": RESOURCE_REQUEST,
        "train_config": TRAIN_CONFIG,
        "expected_outputs": row["paths"],
        "dry_run_only": True,
        "array_submitted": False,
        "notes": (
            "Prepared for review only. Submit with qsub only after explicit "
            "instruction from Ryan."
        ),
    }


def build_rows(stamp: str, reuse_checks: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pbs_array_index = 0
    manifest_index = 0
    for x in X_VALUES:
        for seed in SEEDS:
            n_sims = n_sims_for_x(x)
            is_reuse = x == REUSE_X and seed == REUSE_SEED
            run_id = f"gnk_gaussian_npe_n500_x{x}_seed{seed}_{stamp}"
            config_path = CONFIG_DIR / f"{run_id}.yaml"
            paths = expected_paths(run_id, config_path)
            reuse_marker_path = (
                REUSE_MARKER_DIR / f"gnk_high_budget_n500_x{x}_seed{seed}_reuse.json"
            )
            row = {
                "manifest_index": manifest_index,
                "pbs_array_index": None if is_reuse else pbs_array_index,
                "action": "reuse" if is_reuse else "run",
                "n": N_OBS,
                "d": D_TOTAL,
                "x": x,
                "N": n_sims,
                "seed": seed,
                "method": METHOD,
                "simulator": SIMULATOR,
                "run_id": run_id,
                "config_path": None if is_reuse else rel(config_path),
                "output_dir": (
                    rel(EXISTING_CALIBRATION_DIR) if is_reuse else paths["output_dir"]
                ),
                "paths": (
                    {name: rel(path) for name, path in existing_required_paths().items()}
                    if is_reuse
                    else paths
                ),
                "reuse_marker_path": rel(reuse_marker_path) if is_reuse else "",
                "reuse_source_output_dir": rel(EXISTING_CALIBRATION_DIR) if is_reuse else "",
                "reuse_compatible": bool(reuse_checks["compatible"]) if is_reuse else None,
                "collision_status": "checked-no-collisions",
                "notes": (
                    "Reused completed compatible calibration output; excluded from PBS array."
                    if is_reuse
                    else "Prepared dry-run row; not submitted."
                ),
            }
            rows.append(row)
            if not is_reuse:
                pbs_array_index += 1
            manifest_index += 1
    return rows


def check_new_output_collisions(rows: list[dict[str, Any]]) -> list[str]:
    collisions: list[str] = []
    for row in rows:
        if row["action"] != "run":
            continue
        output_dir = REPO_ROOT / row["paths"]["output_dir"]
        if output_dir.exists():
            collisions.append(rel(output_dir))
        config_path = REPO_ROOT / row["paths"]["config"]
        if config_path.exists():
            collisions.append(rel(config_path))
    return collisions


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "manifest_index",
        "pbs_array_index",
        "action",
        "n",
        "d",
        "x",
        "N",
        "seed",
        "method",
        "simulator",
        "run_id",
        "config_path",
        "output_dir",
        "validation_curve",
        "validation_curve_plot",
        "predicted_u_mean_cov",
        "samples_10k",
        "posterior_samples_pkl",
        "metrics",
        "timing_metadata",
        "stdout_log",
        "stderr_log",
        "reuse_source_output_dir",
        "reuse_marker_path",
        "reuse_compatible",
        "collision_status",
        "notes",
    ]
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {rel(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {name: row.get(name, "") for name in fieldnames}
            for key in (
                "validation_curve",
                "validation_curve_plot",
                "predicted_u_mean_cov",
                "samples_10k",
                "posterior_samples_pkl",
                "metrics",
                "timing_metadata",
                "stdout_log",
                "stderr_log",
            ):
                flat[key] = row["paths"].get(key, "")
            writer.writerow(flat)


def write_configs(rows: list[dict[str, Any]], created_at: str) -> None:
    for row in rows:
        if row["action"] != "run":
            continue
        config = build_job_config(row, created_at)
        text = yaml.safe_dump(config, sort_keys=False)
        no_overwrite_write_text(REPO_ROOT / row["config_path"], text)


def write_pbs_script(path: Path, manifest_json: str) -> None:
    script = f"""#!/bin/bash -l
#PBS -N gnk_high_budget_gnpe
#PBS -J 0-200%20
#PBS -l walltime=47:00:00
#PBS -l mem=64GB
#PBS -l ncpus=4
#PBS -o res/gnk_high_budget/pbs_driver_logs/
#PBS -e res/gnk_high_budget/pbs_driver_logs/

# Dry-run prepared PBS array. Do not submit unless Ryan explicitly instructs it.
# Grid: n=500, d=11, x in {{25,50}}, N=x*d^2*n, seeds 0:100.
# Runnable rows: 201. The x=50, seed=88 row is marked reuse and excluded.
# Resource request: 47h walltime, 64GB memory, 4 CPUs, no GPU.
# Concurrency cap: %20 in the PBS -J directive. Adjust only after review.

set -euo pipefail

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate

export MANIFEST_JSON="${{MANIFEST_JSON:-{manifest_json}}}"
export MPLCONFIGDIR="$PBS_O_WORKDIR/res/gnk_high_budget/mplconfig/${{PBS_ARRAY_INDEX}}"

python npe_convergence/scripts/run_gnk_high_budget_array_job.py \\
  --manifest "$MANIFEST_JSON" \\
  --array-index "$PBS_ARRAY_INDEX"

deactivate
"""
    no_overwrite_write_text(path, script)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--created-at", default=utc_now())
    parser.add_argument(
        "--pbs-script",
        default="npe_convergence/scripts/pbs_jobs/gnk_high_budget_gaussian_npe_array.sh",
    )
    args = parser.parse_args(argv)

    created_at = args.created_at
    stamp = compact_stamp(created_at)
    manifest_csv = OUTPUT_ROOT / f"dry_run_manifest_{stamp}.csv"
    manifest_json = OUTPUT_ROOT / f"dry_run_manifest_{stamp}.json"
    reuse_checks = check_reuse_compatibility()
    if not reuse_checks["compatible"]:
        print("Existing x=50 seed=88 calibration is not compatible for reuse:", file=sys.stderr)
        print(json.dumps(reuse_checks, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    rows = build_rows(stamp, reuse_checks)
    collisions = check_new_output_collisions(rows)
    if collisions:
        print("Refusing to prepare dry-run because output collisions were found:", file=sys.stderr)
        for path in collisions:
            print(f"  {path}", file=sys.stderr)
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    REUSE_MARKER_DIR.mkdir(parents=True, exist_ok=True)
    DRIVER_LOG_DIR.mkdir(parents=True, exist_ok=True)

    write_configs(rows, created_at)
    reuse_row = next(row for row in rows if row["action"] == "reuse")
    no_overwrite_write_json(
        REPO_ROOT / reuse_row["reuse_marker_path"],
        {
            "created_at_utc": created_at,
            "action": "reuse",
            "n": N_OBS,
            "d": D_TOTAL,
            "x": REUSE_X,
            "N": n_sims_for_x(REUSE_X),
            "seed": REUSE_SEED,
            "method": METHOD,
            "simulator": SIMULATOR,
            "source_output_dir": rel(EXISTING_CALIBRATION_DIR),
            "compatibility": reuse_checks,
            "excluded_from_pbs_array": True,
        },
    )

    manifest = {
        "created_at_utc": created_at,
        "dry_run_only": True,
        "array_submitted": False,
        "submit_command": f"qsub {args.pbs_script}",
        "output_namespace": rel(OUTPUT_ROOT),
        "grid": {
            "n": N_OBS,
            "d_s": D_S,
            "d_theta": D_THETA,
            "d": D_TOTAL,
            "x_values": list(X_VALUES),
            "seeds": "0:100 inclusive",
            "N_formula": "N = x * d^2 * n",
            "N_by_x": {str(x): n_sims_for_x(x) for x in X_VALUES},
            "max_x": max(X_VALUES),
            "x_greater_than_50_included": False,
        },
        "method": METHOD,
        "simulator": SIMULATOR,
        "excluded_methods": ["flow-NPE"],
        "resource_request": RESOURCE_REQUEST,
        "pbs_script": args.pbs_script,
        "manifest_csv": rel(manifest_csv),
        "manifest_json": rel(manifest_json),
        "row_counts": {
            "total_grid_rows": len(rows),
            "run_rows": sum(row["action"] == "run" for row in rows),
            "reuse_rows": sum(row["action"] == "reuse" for row in rows),
        },
        "reuse": reuse_checks,
        "rows": rows,
    }
    no_overwrite_write_json(manifest_json, manifest)
    write_manifest_csv(manifest_csv, rows)
    write_pbs_script(REPO_ROOT / args.pbs_script, rel(manifest_json))

    print(f"Wrote {rel(manifest_json)}")
    print(f"Wrote {rel(manifest_csv)}")
    print(f"Wrote {sum(row['action'] == 'run' for row in rows)} run configs in {rel(CONFIG_DIR)}")
    print(f"Wrote {args.pbs_script}")
    print("No qsub command was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
