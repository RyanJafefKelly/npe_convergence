"""Dry-run plan for MA(2)-b0 Gaussian-NPE compatibility refresh.

This script does not train, sample, submit jobs, or write under result caches.
It enumerates a cache-safe output namespace and records collision checks for the
paper-facing delta_0(y) compatibility cases.
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


DEFAULT_DELTA0_VALUES = (0.01, 0.99)
DEFAULT_N_OBS = (1000,)
DEFAULT_SEEDS = tuple(range(101))
DEFAULT_OUTPUT_ROOT = Path("res/ma2_b0_compat_refresh")
DEFAULT_SOURCE_CACHE_ROOT = Path("res/ma2_b0")
DEFAULT_MANIFEST_DIR = Path("docs/weekend_2026_05_02/ma2_b0_task3_dry_run")
DEFAULT_OUTPUT_PREFIX = "ma2_b0_compat_refresh_dry_run_20260502"
DEFAULT_PBS_TEMPLATE = (
    DEFAULT_MANIFEST_DIR / "ma2_b0_compat_refresh_pbs_template.sh"
)


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
    config_json: str
    validation_curve_csv: str
    training_metadata_json: str
    gaussian_standardised_posterior_npz: str
    reference_samples_npz: str
    gaussian_npe_posterior_samples_npz: str
    metrics_json: str
    kl_txt: str
    mmd_txt: str
    t1_overlay_pdf: str
    t2_overlay_pdf: str
    output_dir_exists: bool
    expected_file_collisions: int
    colliding_expected_files: str
    ordinary_gaussian_cache_dir_exists: bool
    ordinary_flow_cache_dir_exists: bool
    legacy_flow_kl_exists: bool
    legacy_flow_exact_pdf_exists: bool
    legacy_flow_npe_pdf_exists: bool
    legacy_gaussian_delta_outputs_exist: bool


def parse_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer/range list, e.g. '0-3,8'."""
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
        Budget("N=nlog(n)", "nlogn", int(n_obs * math.log(n_obs))),
        Budget("N=n^3/2", "n32", int(n_obs ** (3 / 2))),
        Budget("N=n^2", "n2", n_obs**2),
    )


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def expected_files(output_dir: Path, delta_dir: Path) -> list[Path]:
    return [
        output_dir / "config.json",
        output_dir / "validation_curve.csv",
        output_dir / "training_metadata.json",
        delta_dir / "gaussian_npe_standardised_posterior.npz",
        delta_dir / "reference_samples.npz",
        delta_dir / "gaussian_npe_posterior_samples.npz",
        delta_dir / "metrics.json",
        delta_dir / "kl.txt",
        delta_dir / "mmd.txt",
        delta_dir / "t1_posterior_overlay.pdf",
        delta_dir / "t2_posterior_overlay.pdf",
    ]


def legacy_flow_outputs_exist(flow_dir: Path, delta0: str) -> tuple[bool, bool, bool]:
    return (
        (flow_dir / f"kl_{delta0}.txt").is_file(),
        (flow_dir / f"t1_posterior_exact_b_0_0_{delta0}.pdf").is_file()
        and (flow_dir / f"t2_posterior_exact_b_0_0_{delta0}.pdf").is_file(),
        (flow_dir / f"t1_posterior_flow_b_0_0_{delta0}.pdf").is_file()
        and (flow_dir / f"t2_posterior_flow_b_0_0_{delta0}.pdf").is_file(),
    )


def legacy_gaussian_delta_outputs_exist(gaussian_dir: Path, delta0: str) -> bool:
    patterns = (
        f"kl_{delta0}.txt",
        f"*b_0_0_{delta0}*",
        f"*delta*{delta0}*",
        f"*{delta0}*.npz",
        f"*{delta0}*.pkl",
    )
    return any(any(gaussian_dir.glob(pattern)) for pattern in patterns)


def build_manifest(
    *,
    repo_root: Path,
    source_cache_root: Path,
    output_root: Path,
    n_obs_values: tuple[int, ...],
    seeds: tuple[int, ...],
    delta0_values: tuple[float, ...],
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for n_obs in n_obs_values:
        for budget in budgets_for_n(n_obs):
            for seed in seeds:
                run_name = (
                    "gaussian_npe_compat"
                    f"_n_obs_{n_obs}_n_sims_{budget.n_sims}_seed_{seed}"
                )
                output_dir = output_root / run_name
                ordinary_gaussian_dir = (
                    source_cache_root
                    / f"gaussian_npe_n_obs_{n_obs}_n_sims_{budget.n_sims}_seed_{seed}"
                )
                ordinary_flow_dir = (
                    source_cache_root / f"npe_n_obs_{n_obs}_n_sims_{budget.n_sims}_seed_{seed}"
                )
                for delta0_float in delta0_values:
                    delta0 = delta_label(delta0_float)
                    delta_dir = output_dir / f"delta_0_{delta0}"
                    files = expected_files(output_dir, delta_dir)
                    colliding = [p for p in files if p.exists()]
                    flow_kl, flow_exact_pdf, flow_npe_pdf = legacy_flow_outputs_exist(
                        ordinary_flow_dir, delta0
                    )
                    rows.append(
                        ManifestRow(
                            method="gaussian_npe_compat_refresh",
                            n_obs=n_obs,
                            budget_label=budget.label,
                            budget_slug=budget.slug,
                            n_sims=budget.n_sims,
                            seed=seed,
                            delta0=delta0,
                            output_dir=rel(output_dir, repo_root),
                            delta_dir=rel(delta_dir, repo_root),
                            config_json=rel(output_dir / "config.json", repo_root),
                            validation_curve_csv=rel(
                                output_dir / "validation_curve.csv", repo_root
                            ),
                            training_metadata_json=rel(
                                output_dir / "training_metadata.json", repo_root
                            ),
                            gaussian_standardised_posterior_npz=rel(
                                delta_dir / "gaussian_npe_standardised_posterior.npz",
                                repo_root,
                            ),
                            reference_samples_npz=rel(
                                delta_dir / "reference_samples.npz", repo_root
                            ),
                            gaussian_npe_posterior_samples_npz=rel(
                                delta_dir / "gaussian_npe_posterior_samples.npz",
                                repo_root,
                            ),
                            metrics_json=rel(delta_dir / "metrics.json", repo_root),
                            kl_txt=rel(delta_dir / "kl.txt", repo_root),
                            mmd_txt=rel(delta_dir / "mmd.txt", repo_root),
                            t1_overlay_pdf=rel(
                                delta_dir / "t1_posterior_overlay.pdf", repo_root
                            ),
                            t2_overlay_pdf=rel(
                                delta_dir / "t2_posterior_overlay.pdf", repo_root
                            ),
                            output_dir_exists=output_dir.exists(),
                            expected_file_collisions=len(colliding),
                            colliding_expected_files=";".join(
                                rel(path, repo_root) for path in colliding
                            ),
                            ordinary_gaussian_cache_dir_exists=ordinary_gaussian_dir.is_dir(),
                            ordinary_flow_cache_dir_exists=ordinary_flow_dir.is_dir(),
                            legacy_flow_kl_exists=flow_kl,
                            legacy_flow_exact_pdf_exists=flow_exact_pdf,
                            legacy_flow_npe_pdf_exists=flow_npe_pdf,
                            legacy_gaussian_delta_outputs_exist=legacy_gaussian_delta_outputs_exist(
                                ordinary_gaussian_dir, delta0
                            ),
                        )
                    )
    return rows


def summarise(rows: list[ManifestRow], *, repo_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    n_obs_values = sorted({row.n_obs for row in rows})
    budgets = sorted({(row.budget_label, row.n_sims) for row in rows}, key=lambda x: x[1])
    seeds = sorted({row.seed for row in rows})
    deltas = sorted({row.delta0 for row in rows}, key=float)
    unique_output_dirs = sorted({row.output_dir for row in rows})
    colliding_files = sorted(
        {
            path
            for row in rows
            for path in row.colliding_expected_files.split(";")
            if path
        }
    )
    legacy_gaussian_rows = sum(row.legacy_gaussian_delta_outputs_exist for row in rows)
    legacy_flow_kl_rows = sum(row.legacy_flow_kl_exists for row in rows)
    legacy_flow_exact_pdf_rows = sum(row.legacy_flow_exact_pdf_exists for row in rows)
    legacy_flow_npe_pdf_rows = sum(row.legacy_flow_npe_pdf_exists for row in rows)
    ordinary_gaussian_dirs = sum(row.ordinary_gaussian_cache_dir_exists for row in rows)
    ordinary_flow_dirs = sum(row.ordinary_flow_cache_dir_exists for row in rows)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "npe_convergence/scripts/plan_ma2_b0_compat_refresh.py",
        "commit_hash": git_commit(repo_root),
        "repo_root": str(repo_root),
        "source_cache_root": rel(args.source_cache_root.resolve(), repo_root),
        "fresh_output_root": rel(args.output_root, repo_root),
        "paper_target": "fig:ma2_kld_compare",
        "paper_target_n_obs": n_obs_values,
        "delta0_values": deltas,
        "budget_grid": [
            {"label": label, "n_sims": n_sims} for label, n_sims in budgets
        ],
        "seed_min": min(seeds) if seeds else None,
        "seed_max": max(seeds) if seeds else None,
        "seed_count": len(seeds),
        "runtime_job_count": len(unique_output_dirs),
        "manifest_row_count": len(rows),
        "rows_per_runtime_job": len(deltas),
        "decision": "save per-b0 reference samples and Gaussian-NPE posterior samples, not scalar KL only",
        "compatibility_scope": (
            "MA2-b0 is compatibility-failure evidence only; do not pool with GNK "
            "or stereological BvM-rate evidence."
        ),
        "output_collision_count": len(colliding_files),
        "output_collisions": colliding_files,
        "output_dir_collision_count": sum(row.output_dir_exists for row in rows),
        "ordinary_gaussian_cache_dir_rows": ordinary_gaussian_dirs,
        "ordinary_flow_cache_dir_rows": ordinary_flow_dirs,
        "legacy_flow_kl_rows": legacy_flow_kl_rows,
        "legacy_flow_exact_pdf_rows": legacy_flow_exact_pdf_rows,
        "legacy_flow_npe_pdf_rows": legacy_flow_npe_pdf_rows,
        "legacy_gaussian_delta_output_rows": legacy_gaussian_rows,
        "collision_interpretation": (
            "Fresh output namespace has no selected-file collisions when "
            "output_collision_count is zero. Ordinary MA2-b0 cache dirs are "
            "reported as provenance only and must not be reused as compatibility outputs."
        ),
        "expected_per_run_outputs": [
            "config.json",
            "validation_curve.csv",
            "training_metadata.json",
        ],
        "expected_per_delta_outputs": [
            "gaussian_npe_standardised_posterior.npz",
            "reference_samples.npz",
            "gaussian_npe_posterior_samples.npz",
            "metrics.json",
            "kl.txt",
            "mmd.txt",
            "t1_posterior_overlay.pdf",
            "t2_posterior_overlay.pdf",
        ],
        "proposed_runtime_command_template": (
            "python npe_convergence/scripts/run_ma2_b0_gaussian_compat_refresh.py "
            "--seed {seed} --n-obs {n_obs} --n-sims {n_sims} "
            "--delta0-values 0.01,0.99 --output-root res/ma2_b0_compat_refresh "
            "--fail-if-output-exists"
        ),
        "proposed_pbs_array": {
            "array": "0-403",
            "mapping": "seed = PBS_ARRAY_INDEX / 4; n_sims_index = PBS_ARRAY_INDEX % 4",
            "n_sims_values_for_n_obs_1000": [1000, 6907, 31622, 1000000],
            "note": "Each array element trains one Gaussian-NPE cell and exports both delta0 values.",
        },
        "proposed_pbs_template": str(DEFAULT_PBS_TEMPLATE),
        "runtime_blocker": (
            "This dry-run records the design and collisions only. The runtime "
            "runner or patch must still implement training plus per-b0 SMC "
            "reference export before any qsub launch."
        ),
    }


def write_csv(path: Path, rows: list[ManifestRow]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_table(path: Path, rows: list[ManifestRow], summary: dict[str, Any]) -> None:
    grouped: dict[tuple[str, int, str], list[ManifestRow]] = {}
    for row in rows:
        grouped.setdefault((row.budget_label, row.n_sims, row.delta0), []).append(row)

    lines = [
        "# MA2-b0 Compatibility Refresh Dry-Run Table",
        "",
        f"- Fresh namespace: `{summary['fresh_output_root']}`",
        f"- Decision: {summary['decision']}.",
        f"- Output collisions: {summary['output_collision_count']}",
        f"- Runtime jobs: {summary['runtime_job_count']}; manifest rows: {summary['manifest_row_count']}",
        "",
        "| n_obs | N label | n_sims | delta0 | seeds | expected reference samples | expected Gaussian-NPE samples | output collisions | legacy flow scalar/PDF rows | legacy Gaussian delta rows |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for (label, n_sims, delta0), group in sorted(grouped.items(), key=lambda item: (item[0][1], float(item[0][2]))):
        sample = group[0]
        legacy_flow_rows = sum(
            row.legacy_flow_kl_exists
            and row.legacy_flow_exact_pdf_exists
            and row.legacy_flow_npe_pdf_exists
            for row in group
        )
        lines.append(
            f"| {sample.n_obs} "
            f"| {label} "
            f"| {n_sims} "
            f"| {delta0} "
            f"| {len(group)} "
            f"| `{sample.reference_samples_npz}` "
            f"| `{sample.gaussian_npe_posterior_samples_npz}` "
            f"| {sum(row.expected_file_collisions for row in group)} "
            f"| {legacy_flow_rows} "
            f"| {sum(row.legacy_gaussian_delta_outputs_exist for row in group)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-cache-root", type=Path, default=DEFAULT_SOURCE_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-obs", default=",".join(str(v) for v in DEFAULT_N_OBS))
    parser.add_argument("--seeds", default="0-100")
    parser.add_argument(
        "--delta0-values",
        default=",".join(str(v) for v in DEFAULT_DELTA0_VALUES),
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_cache_root = args.source_cache_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    rows = build_manifest(
        repo_root=repo_root,
        source_cache_root=source_cache_root,
        output_root=output_root,
        n_obs_values=parse_ints(args.n_obs),
        seeds=parse_ints(args.seeds),
        delta0_values=parse_floats(args.delta0_values),
    )
    summary = summarise(rows, repo_root=repo_root, args=args)

    print("MA2-b0 Gaussian-NPE compatibility dry-run")
    print(f"  fresh namespace: {summary['fresh_output_root']}")
    print(f"  n_obs: {summary['paper_target_n_obs']}")
    print(f"  delta0 values: {', '.join(summary['delta0_values'])}")
    print(f"  runtime jobs: {summary['runtime_job_count']}")
    print(f"  manifest rows: {summary['manifest_row_count']}")
    print(f"  output collisions: {summary['output_collision_count']}")
    print(f"  legacy flow KL rows: {summary['legacy_flow_kl_rows']}")
    print(f"  legacy Gaussian delta rows: {summary['legacy_gaussian_delta_output_rows']}")

    if not args.write_manifest:
        return

    manifest_dir = args.manifest_dir
    if not manifest_dir.is_absolute():
        manifest_dir = repo_root / manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = manifest_dir / f"{args.output_prefix}_manifest.csv"
    json_path = manifest_dir / f"{args.output_prefix}_summary.json"
    table_path = manifest_dir / f"{args.output_prefix}_table.md"

    for path in (csv_path, json_path, table_path):
        if path.exists() and not args.allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    write_csv(csv_path, rows)
    summary.update(
        {
            "manifest_csv": rel(csv_path, repo_root),
            "summary_json": rel(json_path, repo_root),
            "table_md": rel(table_path, repo_root),
        }
    )
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_table(table_path, rows, summary)
    print(f"  wrote: {rel(csv_path, repo_root)}")
    print(f"  wrote: {rel(json_path, repo_root)}")
    print(f"  wrote: {rel(table_path, repo_root)}")


if __name__ == "__main__":
    main()
