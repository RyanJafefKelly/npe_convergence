"""Dry-run manifest for canonical GNK NUTS reference refresh.

This script does not run NUTS, train, sample, submit jobs, or write under
result caches. It enumerates the full STANDARD_N x seeds x conventions grid
in a fresh output namespace and writes a manifest, summary, table, and PBS
template. Mirrors plan_ma2_b0_delta1_refresh.py.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("res/gnk_v3_refs")
DEFAULT_MANIFEST_DIR = Path("docs/meeting_2026_05_18/gnk_nuts_refresh_plan")
DEFAULT_OUTPUT_PREFIX = "gnk_nuts_refresh"
DEFAULT_N_OBS = (100, 500, 1000, 5000)
DEFAULT_SEEDS = tuple(range(101))
CONVENTIONS = ("flow", "gaussian")


@dataclass(frozen=True)
class ManifestRow:
    n_obs: int
    seed: int
    convention: str
    output_path: str
    runtime_command: str
    output_exists: bool
    legacy_central_cache: str
    legacy_central_cache_exists: bool


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


def parse_strs(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def output_path_for(output_root: Path, n_obs: int, seed: int, convention: str) -> Path:
    return output_root / f"nuts_n_obs_{n_obs}_seed_{seed}_conv_{convention}.pkl"


def legacy_cache_path_for(n_obs: int, seed: int, convention: str) -> Path:
    if convention == "flow":
        return REPO_ROOT / "res" / "gnk" / f"nuts_cache_v2_flow_n_obs_{n_obs}_seed_{seed}.pkl"
    if convention == "gaussian":
        return REPO_ROOT / "res" / "gnk" / f"nuts_cache_v2_n_obs_{n_obs}_seed_{seed}.pkl"
    raise ValueError(f"unknown convention: {convention}")


def runtime_command(
    *, n_obs: int, seed: int, convention: str, output_root: Path
) -> str:
    return (
        "JAX_ENABLE_X64=1 .venv/bin/python "
        "npe_convergence/scripts/run_gnk_nuts_refresh.py "
        f"--n-obs {n_obs} --seed {seed} --convention {convention} "
        f"--output-root {rel(output_root)}"
    )


def build_manifest(
    *,
    output_root: Path,
    n_obs_values: tuple[int, ...],
    seeds: tuple[int, ...],
    conventions: tuple[str, ...],
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for n_obs in n_obs_values:
        for seed in seeds:
            for convention in conventions:
                out = output_path_for(output_root, n_obs, seed, convention)
                legacy = legacy_cache_path_for(n_obs, seed, convention)
                rows.append(
                    ManifestRow(
                        n_obs=n_obs,
                        seed=seed,
                        convention=convention,
                        output_path=rel(out),
                        runtime_command=runtime_command(
                            n_obs=n_obs,
                            seed=seed,
                            convention=convention,
                            output_root=output_root,
                        ),
                        output_exists=out.exists(),
                        legacy_central_cache=rel(legacy),
                        legacy_central_cache_exists=legacy.exists(),
                    )
                )
    return rows


def summarise(rows: list[ManifestRow], output_root: Path) -> dict[str, Any]:
    n_obs_values = sorted({row.n_obs for row in rows})
    seeds = sorted({row.seed for row in rows})
    conventions = sorted({row.convention for row in rows})
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "npe_convergence/scripts/plan_gnk_nuts_refresh.py",
        "commit_hash": git_commit(),
        "repo_root": str(REPO_ROOT),
        "fresh_output_root": rel(output_root),
        "n_obs": n_obs_values,
        "conventions": conventions,
        "seed_min": min(seeds) if seeds else None,
        "seed_max": max(seeds) if seeds else None,
        "seed_count": len(seeds),
        "runtime_job_count": len(rows),
        "manifest_row_count": len(rows),
        "output_collision_count": sum(row.output_exists for row in rows),
        "legacy_cache_present_count": sum(
            row.legacy_central_cache_exists for row in rows
        ),
        "note": (
            "Canonical x64 NUTS refresh. Run JAX_ENABLE_X64=1 wrapper so x64 is on "
            "before any JAX import. Each cell writes one fingerprinted dict to the "
            "output_path. Pass two-cell smoke tests before submitting the full grid."
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
        "# GNK canonical x64 NUTS reference refresh: dry-run",
        "",
        f"- Fresh namespace: `{summary['fresh_output_root']}`",
        f"- n_obs grid: {summary['n_obs']}",
        f"- Conventions: {', '.join(summary['conventions'])}",
        f"- Seeds: {summary['seed_min']} to {summary['seed_max']} "
        f"({summary['seed_count']})",
        f"- Runtime jobs: {summary['runtime_job_count']}",
        f"- Manifest rows: {summary['manifest_row_count']}",
        f"- Output collisions: {summary['output_collision_count']}",
        f"- Legacy central caches present: {summary['legacy_cache_present_count']}",
        "",
        "Run two-cell smoke test before submitting the full grid:",
        "",
        "    JAX_ENABLE_X64=1 .venv/bin/python "
        "npe_convergence/scripts/run_gnk_nuts_refresh.py "
        "--n-obs 1000 --seed 36 --convention flow --output-root res/gnk_v3_refs",
        "",
        "    JAX_ENABLE_X64=1 .venv/bin/python "
        "npe_convergence/scripts/run_gnk_nuts_refresh.py "
        "--n-obs 5000 --seed 50 --convention flow --output-root res/gnk_v3_refs",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_pbs_template(path: Path, manifest_csv: Path, row_count: int) -> None:
    manifest_rel = rel(manifest_csv)
    content = f"""#!/bin/bash -l
#PBS -N gnk_nuts_refresh
#PBS -J 1-{row_count}
#PBS -l walltime=04:00:00
#PBS -l mem=8GB
#PBS -l ncpus=4

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
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--n-obs", default=",".join(str(value) for value in DEFAULT_N_OBS)
    )
    parser.add_argument("--seeds", default="0-100")
    parser.add_argument("--conventions", default=",".join(CONVENTIONS))
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
    conventions = parse_strs(args.conventions)
    rows = build_manifest(
        output_root=output_root,
        n_obs_values=parse_ints(args.n_obs),
        seeds=parse_ints(args.seeds),
        conventions=conventions,
    )
    summary = summarise(rows, output_root)

    print("GNK canonical x64 NUTS reference refresh dry-run")
    print(f"  fresh namespace: {summary['fresh_output_root']}")
    print(f"  n_obs: {summary['n_obs']}")
    print(f"  conventions: {', '.join(summary['conventions'])}")
    print(f"  runtime jobs: {summary['runtime_job_count']}")
    print(f"  manifest rows: {summary['manifest_row_count']}")
    print(f"  output collisions: {summary['output_collision_count']}")
    print(f"  legacy caches present: {summary['legacy_cache_present_count']}")

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
