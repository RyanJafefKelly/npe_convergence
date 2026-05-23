"""Dry-run manifest for exact MA(2)-b0 delta0=1.0 refresh.

This script does not train, sample, submit jobs, or write under result caches.
It enumerates the full 4x4 grid for both flow-NPE and Gaussian-NPE in a fresh
output namespace and records collision checks plus runnable commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("res/ma2_b0_delta1_refresh")
DEFAULT_MANIFEST_DIR = Path("docs/meeting_2026_05_18/ma2_delta1_refresh_plan")
DEFAULT_OUTPUT_PREFIX = "ma2_delta1_refresh_dry_run"
DEFAULT_N_OBS = (100, 500, 1000, 5000)
DEFAULT_SEEDS = tuple(range(101))
DEFAULT_DELTA0_VALUES = (1.0,)
METHODS = ("flow_npe", "gaussian_npe")


@dataclass(frozen=True)
class Budget:
    label: str
    slug: str
    n_sims: int


@dataclass(frozen=True)
class ManifestRow:
    method: str
    n_obs: int
    budget_label: str
    budget_slug: str
    n_sims: int
    seed: int
    delta0: str
    output_dir: str
    delta_dir: str
    metrics_json: str
    kl_txt: str
    mmd_txt: str
    reference_samples_npz: str
    posterior_samples_npz: str
    runtime_command: str
    output_dir_exists: bool
    expected_file_collisions: int
    colliding_expected_files: str
    legacy_delta1_kl_exists: bool
    legacy_delta1_mmd_exists: bool


def parse_ints(value: str) -> tuple[int, ...]:
    parsed: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            parsed.update(range(int(start), int(end) + 1))
        else:
            parsed.add(int(part))
    return tuple(sorted(parsed))


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def delta_label(value: float) -> str:
    return f"{value:.12g}"


def budgets_for_n(n_obs: int) -> tuple[Budget, ...]:
    return (
        Budget("N=n", "n", n_obs),
        Budget("N=n log(n)", "nlogn", int(n_obs * math.log(n_obs))),
        Budget("N=n^(3/2)", "n32", int(n_obs ** (3 / 2))),
        Budget("N=n^2", "n2", n_obs**2),
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def method_prefix(method: str) -> str:
    if method == "flow_npe":
        return "flow_npe_compat"
    if method == "gaussian_npe":
        return "gaussian_npe_compat"
    raise ValueError(f"unknown method: {method}")


def runner_script(method: str) -> str:
    if method == "flow_npe":
        return "npe_convergence/scripts/run_ma2_b0_flow_delta_refresh.py"
    if method == "gaussian_npe":
        return "npe_convergence/scripts/run_ma2_b0_gaussian_compat_refresh.py"
    raise ValueError(f"unknown method: {method}")


def ordinary_prefix(method: str) -> str:
    if method == "flow_npe":
        return "npe"
    if method == "gaussian_npe":
        return "gaussian_npe"
    raise ValueError(f"unknown method: {method}")


def posterior_filename(method: str) -> str:
    if method == "flow_npe":
        return "flow_npe_posterior_samples.npz"
    if method == "gaussian_npe":
        return "gaussian_npe_posterior_samples.npz"
    raise ValueError(f"unknown method: {method}")


def run_dir(output_root: Path, method: str, n_obs: int, n_sims: int, seed: int) -> Path:
    return output_root / f"{method_prefix(method)}_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}"


def expected_files(out: Path, ddir: Path, method: str) -> list[Path]:
    return [
        out / "config.json",
        out / "validation_curve.csv",
        out / "training_metadata.json",
        out / "standardisation.npz",
        ddir / "reference_samples.npz",
        ddir / posterior_filename(method),
        ddir / "metrics.json",
        ddir / "kl.txt",
        ddir / "mmd.txt",
    ]


def runtime_command(
    *,
    method: str,
    n_obs: int,
    n_sims: int,
    seed: int,
    delta0_values: tuple[float, ...],
    output_root: Path,
) -> str:
    delta0_spec = ",".join(delta_label(value) for value in delta0_values)
    return (
        f".venv/bin/python {runner_script(method)} "
        f"--seed {seed} --n-obs {n_obs} --n-sims {n_sims} "
        f"--delta0-values {delta0_spec} --output-root {rel(output_root)} "
        "--fail-if-output-exists --no-save-plots"
    )


def build_manifest(
    *,
    output_root: Path,
    n_obs_values: tuple[int, ...],
    seeds: tuple[int, ...],
    delta0_values: tuple[float, ...],
    methods: tuple[str, ...],
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    source_root = REPO_ROOT / "res" / "ma2_b0"
    for method in methods:
        for n_obs in n_obs_values:
            for budget in budgets_for_n(n_obs):
                for seed in seeds:
                    out = run_dir(output_root, method, n_obs, budget.n_sims, seed)
                    ordinary = (
                        source_root
                        / f"{ordinary_prefix(method)}_n_obs_{n_obs}_n_sims_{budget.n_sims}_seed_{seed}"
                    )
                    for delta0_value in delta0_values:
                        delta0 = delta_label(delta0_value)
                        ddir = out / f"delta_0_{delta0}"
                        files = expected_files(out, ddir, method)
                        colliding = [path for path in files if path.exists()]
                        rows.append(
                            ManifestRow(
                                method=method,
                                n_obs=n_obs,
                                budget_label=budget.label,
                                budget_slug=budget.slug,
                                n_sims=budget.n_sims,
                                seed=seed,
                                delta0=delta0,
                                output_dir=rel(out),
                                delta_dir=rel(ddir),
                                metrics_json=rel(ddir / "metrics.json"),
                                kl_txt=rel(ddir / "kl.txt"),
                                mmd_txt=rel(ddir / "mmd.txt"),
                                reference_samples_npz=rel(
                                    ddir / "reference_samples.npz"
                                ),
                                posterior_samples_npz=rel(
                                    ddir / posterior_filename(method)
                                ),
                                runtime_command=runtime_command(
                                    method=method,
                                    n_obs=n_obs,
                                    n_sims=budget.n_sims,
                                    seed=seed,
                                    delta0_values=delta0_values,
                                    output_root=output_root,
                                ),
                                output_dir_exists=out.exists(),
                                expected_file_collisions=len(colliding),
                                colliding_expected_files=";".join(
                                    rel(path) for path in colliding
                                ),
                                legacy_delta1_kl_exists=(
                                    (ordinary / f"kl_{delta0}.txt").is_file()
                                    or (ordinary / "kl_1.0.txt").is_file()
                                ),
                                legacy_delta1_mmd_exists=(
                                    (ordinary / f"mmd_{delta0}.txt").is_file()
                                    or (ordinary / "mmd_1.0.txt").is_file()
                                ),
                            )
                        )
    return rows


def summarise(rows: list[ManifestRow], output_root: Path) -> dict[str, Any]:
    methods = sorted({row.method for row in rows})
    n_obs_values = sorted({row.n_obs for row in rows})
    seeds = sorted({row.seed for row in rows})
    deltas = sorted({row.delta0 for row in rows}, key=float)
    output_dirs = sorted({row.output_dir for row in rows})
    colliding_files = sorted(
        {
            path
            for row in rows
            for path in row.colliding_expected_files.split(";")
            if path
        }
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "npe_convergence/scripts/plan_ma2_b0_delta1_refresh.py",
        "commit_hash": git_commit(),
        "repo_root": str(REPO_ROOT),
        "fresh_output_root": rel(output_root),
        "methods": methods,
        "n_obs": n_obs_values,
        "delta0_values": deltas,
        "seed_min": min(seeds) if seeds else None,
        "seed_max": max(seeds) if seeds else None,
        "seed_count": len(seeds),
        "runtime_job_count": len(output_dirs),
        "manifest_row_count": len(rows),
        "output_collision_count": len(colliding_files),
        "output_dir_collision_count": sum(row.output_dir_exists for row in rows),
        "output_collisions": colliding_files,
        "legacy_delta1_kl_rows": sum(row.legacy_delta1_kl_exists for row in rows),
        "legacy_delta1_mmd_rows": sum(row.legacy_delta1_mmd_exists for row in rows),
        "note": (
            "This is a dry-run manifest only. It does not launch the full grid. "
            "Run one pilot cell per method before submitting all rows."
        ),
    }


def write_csv(path: Path, rows: list[ManifestRow]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_table(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MA(2) Exact Delta0=1.0 Refresh Dry-Run",
        "",
        f"- Fresh namespace: `{summary['fresh_output_root']}`",
        f"- Methods: {', '.join(summary['methods'])}",
        f"- n_obs grid: {summary['n_obs']}",
        f"- Seeds: {summary['seed_min']} to {summary['seed_max']} ({summary['seed_count']})",
        f"- Runtime jobs: {summary['runtime_job_count']}",
        f"- Manifest rows: {summary['manifest_row_count']}",
        f"- Output collisions: {summary['output_collision_count']}",
        f"- Legacy delta0=1 KL rows under `res/ma2_b0`: {summary['legacy_delta1_kl_rows']}",
        "",
        "Submit from the repo root after one pilot cell per method passes.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_pbs_template(path: Path, manifest_csv: Path, row_count: int) -> None:
    manifest_rel = rel(manifest_csv)
    content = f"""#!/bin/bash -l
#PBS -N ma2_delta1_refresh
#PBS -J 1-{row_count}
#PBS -l walltime=47:00:00
#PBS -l mem=64GB
#PBS -l ncpus=1

# Template only. Run one local/runtime pilot cell per method before submission.

cd "$PBS_O_WORKDIR"
module load GCCcore/13.2.0
module load Python/3.11.5
source .venv/bin/activate

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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-obs", default=",".join(str(value) for value in DEFAULT_N_OBS))
    parser.add_argument("--seeds", default="0-100")
    parser.add_argument(
        "--delta0-values",
        default=",".join(str(value) for value in DEFAULT_DELTA0_VALUES),
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    rows = build_manifest(
        output_root=output_root,
        n_obs_values=parse_ints(args.n_obs),
        seeds=parse_ints(args.seeds),
        delta0_values=parse_floats(args.delta0_values),
        methods=methods,
    )
    summary = summarise(rows, output_root)

    print("MA2 exact delta0=1.0 refresh dry-run")
    print(f"  fresh namespace: {summary['fresh_output_root']}")
    print(f"  methods: {', '.join(summary['methods'])}")
    print(f"  n_obs: {summary['n_obs']}")
    print(f"  delta0 values: {', '.join(summary['delta0_values'])}")
    print(f"  runtime jobs: {summary['runtime_job_count']}")
    print(f"  manifest rows: {summary['manifest_row_count']}")
    print(f"  output collisions: {summary['output_collision_count']}")
    print(f"  legacy delta0=1 KL rows: {summary['legacy_delta1_kl_rows']}")

    if not args.write_manifest:
        return

    manifest_dir = args.manifest_dir
    if not manifest_dir.is_absolute():
        manifest_dir = REPO_ROOT / manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = manifest_dir / f"{args.output_prefix}_manifest.csv"
    json_path = manifest_dir / f"{args.output_prefix}_summary.json"
    table_path = manifest_dir / f"{args.output_prefix}_table.md"
    pbs_path = manifest_dir / f"{args.output_prefix}_pbs_template.sh"

    for path in (csv_path, json_path, table_path, pbs_path):
        if path.exists() and not args.allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    write_csv(csv_path, rows)
    summary.update(
        {
            "manifest_csv": rel(csv_path),
            "summary_json": rel(json_path),
            "table_md": rel(table_path),
            "pbs_template": rel(pbs_path),
        }
    )
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_table(table_path, summary)
    write_pbs_template(pbs_path, csv_path, len(rows))
    print(f"  wrote: {rel(csv_path)}")
    print(f"  wrote: {rel(json_path)}")
    print(f"  wrote: {rel(table_path)}")
    print(f"  wrote: {rel(pbs_path)}")


if __name__ == "__main__":
    main()
