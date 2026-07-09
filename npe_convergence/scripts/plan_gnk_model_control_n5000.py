"""Generate per-seed configs for the GNK model-control extension at n=5000, N=n^2.

The existing run_gnk_model_control_pilot.py is hard-coded to n=500, seed=88. Its
run_pilot() reads `n`, `N`, `observed_seed`, `nuts_reference_cache_path`, etc.
from a YAML config rather than from the module-level constants, so the simplest
way to extend it is to generate a config per seed under a fresh output namespace.

This script writes 30 YAML configs (seeds 0..29 by default) into
docs/meeting_2026_05_18/gnk_model_control_n5000_plan/configs/ plus a manifest CSV
and a PBS array template. Per Pro's section 8 #2, two additional configs use a
distinct --training-seed to decompose data-seed vs training-RNG variability.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_DIR = (
    REPO_ROOT
    / "docs"
    / "meeting_2026_05_18"
    / "gnk_model_control_n5000_plan"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "res" / "gnk_model_control_n5000"

PARAM_NAMES = ("A", "B", "g", "k")
N_OBS = 5000
D_S = 7
D_THETA = 4
N_SIMS = N_OBS * N_OBS  # N = n^2 = 25,000,000
NUTS_CACHE_TEMPLATE = "res/gnk_v3_refs/nuts_n_obs_{n_obs}_seed_{seed}_conv_gaussian.pkl"
DEFAULT_SEEDS = tuple(range(30))
# Two extra training-seed-repeat cells per Pro's section 8 #2.
DEFAULT_TRAINING_SEED_REPEATS = ((0, 17), (15, 17))  # (observed_seed, training_seed)
RESOURCE_REQUEST = {
    "scheduler": "PBS",
    "job_type": "array",
    "walltime": "47:00:00",
    "mem": "64GB",
    "ncpus": 4,
    "ngpus": 0,
}


@dataclass(frozen=True)
class CellRow:
    cell_id: str
    observed_seed: int
    training_seed: int
    n_obs: int
    n_sims: int
    config_path: str
    output_dir: str
    runtime_command: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_git(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return default


def build_paths(output_dir: Path) -> dict[str, str]:
    return {
        "output_dir": rel(output_dir),
        "config": rel(output_dir / "config.yaml"),
        "validation_curve": rel(output_dir / "validation_curve.csv"),
        "validation_curve_plot": rel(output_dir / "validation_curve.pdf"),
        "predicted_u_mean_cov": rel(output_dir / "gaussian_npe_u_posterior.npz"),
        "samples_10k": rel(output_dir / "posterior_samples_10k.npz"),
        "posterior_samples_pkl": rel(output_dir / "posterior_samples.pkl"),
        "metrics": rel(output_dir / "metrics.json"),
        "timing_metadata": rel(output_dir / "timing_metadata.json"),
        "simulator_diagnostics": rel(output_dir / "simulator_diagnostics.json"),
        "evaluation": rel(output_dir / "u_space_decomposition.json"),
        "stdout_log": rel(output_dir / "logs/stdout.log"),
        "stderr_log": rel(output_dir / "logs/stderr.log"),
    }


def build_one_config(
    *,
    observed_seed: int,
    training_seed: int,
    output_root: Path,
    created_at: str,
    code_commit: str,
) -> dict[str, Any]:
    stamp = (
        created_at.replace("-", "").replace(":", "").replace("+00:00", "Z").replace("Z", "Z")
    )
    cell_id = (
        f"gnk_asymptotic_mvn_gaussian_npe_n5000_seed{observed_seed}"
        f"_train{training_seed}_{stamp}"
    )
    output_dir = output_root / cell_id
    paths = build_paths(output_dir)
    nuts_cache_path = NUTS_CACHE_TEMPLATE.format(n_obs=N_OBS, seed=observed_seed)
    run_cmd = (
        f"python npe_convergence/scripts/run_gnk_model_control_pilot.py "
        f"--config {paths['config']}"
    )
    return {
        "task": "gnk-model-control-n5000",
        "run_id": cell_id,
        "created_at_utc": created_at,
        "git_commit": code_commit,
        "model_control_code_commit": code_commit,
        "method": "Gaussian-NPE",
        "simulator": "gnk_asymptotic_mvn",
        "empirical_simulator_name": "gnk_empirical_quantile",
        "n": N_OBS,
        "d_s": D_S,
        "d_theta": D_THETA,
        "d": D_S + D_THETA,
        "x": "n_squared",
        "N": N_SIMS,
        "observed_seed": observed_seed,
        "simulation_seed": observed_seed,
        "training_seed": training_seed,
        "posterior_sampling_seed": observed_seed,
        "prior": {
            "implementation": "numpyro.distributions.Uniform(1e-6, 10-1e-6)",
            "scientific_prior": "independent Uniform(0, 10) for A, B, g, k",
            "theta_names": list(PARAM_NAMES),
        },
        "observed_summary": {
            "source": (
                "regenerated from true theta=(3,1,2,0.5), observed seed "
                f"{observed_seed}, and "
                "npe_convergence.examples.gnk.ss_octile, matching the NUTS v3 "
                "canonical reference under the gaussian convention"
            ),
            "dimension": D_S,
            "coordinate_convention": (
                "octiles at probabilities 1/(d_s+1), ..., d_s/(d_s+1)"
            ),
        },
        "nuts_reference_cache_path": nuts_cache_path,
        "nuts_reference_kind": "v3_canonical_gaussian_convention",
        "training_pair_generator": {
            "theta_sampler": "prior",
            "summary_simulator": "gnk_asymptotic_mvn",
            "mvn_mean_function": "npe_convergence.examples.gnk.gnk at normal quantiles",
            "mvn_covariance_function": (
                "npe_convergence.examples.gnk.compute_covariance_matrix"
            ),
            "base_jitter": 1e-6,
            "covariance_handling": (
                "Use gnk_model's 1e-6 diagonal jitter. Resample rare prior "
                "draws whose asymptotic covariance is still non-SPD or "
                "non-finite; record invalid_covariance_* counts in "
                "simulator_diagnostics.json."
            ),
        },
        "output_dir": rel(output_dir),
        "scheduler_resource_request": RESOURCE_REQUEST,
        "train_config": {
            "lr": 5e-4,
            "batch_size": 256,
            "max_epochs": 2000,
            "patience": 200,
            "val_frac": 0.1,
            "hidden_dims": [128, 128],
        },
        "expected_outputs": paths,
        "post_run_evaluation_command": (
            f"python npe_convergence/scripts/run_gnk_model_control_pilot.py "
            f"--evaluate --config {paths['config']}"
        ),
        "run_command": run_cmd,
        "full_array_submitted": False,
        "selected_output_collisions": [],
        "selected_output_collisions_at_prepare_time": False,
        "selection_rationale": (
            "30-seed paired model-control extension at n=5000, N=n^2 as the "
            "decisive test of broad-prior amortisation vs finite-n target "
            "mismatch. Pairs with real-GNK Gaussian-NPE results at the same "
            "observed seed."
        ),
    }


def write_pbs_template(path: Path, manifest_csv: Path, row_count: int) -> None:
    manifest_rel = rel(manifest_csv)
    content = f"""#!/bin/bash -l
#PBS -N gnk_model_control_n5000
#PBS -J 1-{row_count}
#PBS -l walltime={RESOURCE_REQUEST['walltime']}
#PBS -l mem={RESOURCE_REQUEST['mem']}
#PBS -l ncpus={RESOURCE_REQUEST['ncpus']}

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate
export JAX_ENABLE_X64=1

MANIFEST="{manifest_rel}"
CMD=$(python - "$MANIFEST" "$PBS_ARRAY_INDEX" <<'PY'
import csv
import sys

manifest, pbs_idx = sys.argv[1], int(sys.argv[2])
idx = pbs_idx - 1
with open(manifest, newline="") as handle:
    row = next(row for ii, row in enumerate(csv.DictReader(handle)) if ii == idx)
print(row["runtime_command"])
PY
)

echo "$CMD"
eval "$CMD"

deactivate
"""
    path.write_text(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated observed seeds (paired with each its own training seed=observed).",
    )
    parser.add_argument(
        "--training-seed-repeats",
        type=str,
        default=";".join(f"{a},{b}" for a, b in DEFAULT_TRAINING_SEED_REPEATS),
        help=(
            "Semicolon-separated (observed_seed,training_seed) pairs to run "
            "with a non-default training seed."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument(
        "--write-configs",
        action="store_true",
        help="Write per-cell config YAMLs plus the manifest and PBS template.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
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
    return sorted(out)


def parse_training_seed_repeats(value: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for spec in value.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        a, b = spec.split(",")
        out.append((int(a), int(b)))
    return out


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    repeats = parse_training_seed_repeats(args.training_seed_repeats)

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    plan_dir = args.plan_dir
    if not plan_dir.is_absolute():
        plan_dir = REPO_ROOT / plan_dir
    configs_dir = plan_dir / "configs"

    created_at = utc_now()
    code_commit = run_git(["rev-parse", "--short", "HEAD"])

    cells: list[CellRow] = []

    # Primary cells: observed_seed == training_seed.
    for seed in seeds:
        config = build_one_config(
            observed_seed=seed,
            training_seed=seed,
            output_root=output_root,
            created_at=created_at,
            code_commit=code_commit,
        )
        config_path = configs_dir / f"config_seed_{seed}_train_{seed}.yaml"
        runtime_command = (
            f"python npe_convergence/scripts/run_gnk_model_control_pilot.py "
            f"--config {rel(config_path)}"
        )
        cells.append(
            CellRow(
                cell_id=config["run_id"],
                observed_seed=seed,
                training_seed=seed,
                n_obs=N_OBS,
                n_sims=N_SIMS,
                config_path=rel(config_path),
                output_dir=config["output_dir"],
                runtime_command=runtime_command,
            )
        )

    # Training-seed-repeat cells.
    for observed_seed, training_seed in repeats:
        config = build_one_config(
            observed_seed=observed_seed,
            training_seed=training_seed,
            output_root=output_root,
            created_at=created_at,
            code_commit=code_commit,
        )
        config_path = configs_dir / (
            f"config_seed_{observed_seed}_train_{training_seed}.yaml"
        )
        runtime_command = (
            f"python npe_convergence/scripts/run_gnk_model_control_pilot.py "
            f"--config {rel(config_path)}"
        )
        cells.append(
            CellRow(
                cell_id=config["run_id"],
                observed_seed=observed_seed,
                training_seed=training_seed,
                n_obs=N_OBS,
                n_sims=N_SIMS,
                config_path=rel(config_path),
                output_dir=config["output_dir"],
                runtime_command=runtime_command,
            )
        )

    print(f"GNK model-control n=5000 plan: {len(cells)} cells "
          f"({len(seeds)} primary + {len(repeats)} training-seed repeats)")
    print(f"  output root: {rel(output_root)}")
    print(f"  plan dir:    {rel(plan_dir)}")

    if not args.write_configs:
        print("  (dry run; pass --write-configs to write configs and manifest)")
        return

    configs_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = plan_dir / "gnk_model_control_n5000_manifest.csv"
    summary_json = plan_dir / "gnk_model_control_n5000_summary.json"
    pbs_path = plan_dir / "gnk_model_control_n5000_pbs_template.sh"

    if not args.allow_overwrite:
        for path in (manifest_csv, summary_json, pbs_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite: {path}")

    # Write per-cell configs.
    for cell in cells:
        config_path = REPO_ROOT / cell.config_path
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = build_one_config(
            observed_seed=cell.observed_seed,
            training_seed=cell.training_seed,
            output_root=output_root,
            created_at=created_at,
            code_commit=code_commit,
        )
        with config_path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=True)

    # Write manifest CSV.
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=[k for k in asdict(cells[0]).keys()]
        )
        writer.writeheader()
        for cell in cells:
            writer.writerow(asdict(cell))

    # Write summary JSON.
    summary = {
        "created_at": created_at,
        "git_commit": code_commit,
        "total_cells": len(cells),
        "primary_seeds": list(seeds),
        "training_seed_repeats": [list(t) for t in repeats],
        "n_obs": N_OBS,
        "n_sims": N_SIMS,
        "output_root": rel(output_root),
        "plan_dir": rel(plan_dir),
        "note": (
            "Day 2 submission. Submit by end of Wednesday 2026-05-27 so the "
            "30-cell array finishes before HPC access ends in early June. "
            "Per-cell training takes the same order as the n=500 retry1 run "
            "scaled by n_sims=25M; budget about 5 to 10 hours per cell."
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    write_pbs_template(pbs_path, manifest_csv, len(cells))

    print(f"  wrote {len(cells)} config YAMLs under {rel(configs_dir)}")
    print(f"  wrote {rel(manifest_csv)}")
    print(f"  wrote {rel(summary_json)}")
    print(f"  wrote {rel(pbs_path)}")


if __name__ == "__main__":
    main()
