"""Manifest-driven targeted repair for stereological standard-grid gaps.

This wrapper avoids the broad stereological PBS scripts. It builds a repair
manifest from the Task 1 inventory, runs each selected row into a staging root,
verifies the staged outputs, and copies only missing required files into the
canonical cache. Existing canonical required files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import pickle as pkl
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpyro  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.scripts.run_stereological import run_stereological
from npe_convergence.scripts.run_stereological_gaussian import run_stereological_gaussian


DEFAULT_INVENTORY = (
    REPO_ROOT / "notebooks" / "plots" / "stereological_task1_20260502" / "seed_count_inventory.csv"
)
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "res" / "stereological"
DEFAULT_STAGING_ROOT = REPO_ROOT / "res" / "stereological_targeted_repair_20260504"
STANDARD_LABELS = {"N=n", "N=n log(n)", "N=n^(3/2)", "N=n^2"}
FLOW_REQUIRED = ("posterior_samples.pkl", "estimated_coverage.npy", "biases.npy")
GAUSSIAN_REQUIRED = ("estimated_coverage.npy", "biases.npy")


@dataclass(frozen=True)
class RepairRow:
    row_index: int
    method: str
    seed: int
    n_obs: int
    n_sims: int
    n_label: str

    @property
    def prefix(self) -> str:
        if self.method == "flow_npe":
            return "npe"
        if self.method == "gaussian_npe":
            return "gaussian_npe"
        raise ValueError(f"unsupported method: {self.method}")

    @property
    def required_files(self) -> tuple[str, ...]:
        if self.method == "flow_npe":
            return FLOW_REQUIRED
        if self.method == "gaussian_npe":
            return GAUSSIAN_REQUIRED
        raise ValueError(f"unsupported method: {self.method}")

    def output_dir(self, root: Path) -> Path:
        return root / f"{self.prefix}_n_obs_{self.n_obs}_n_sims_{self.n_sims}_seed_{self.seed}"


def resolve_root(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_seed_list(spec: str) -> set[int]:
    if not spec:
        return set()
    return {int(part) for part in spec.split(";") if part != ""}


def renumber(rows: list[RepairRow]) -> list[RepairRow]:
    return [replace(row, row_index=i) for i, row in enumerate(rows)]


def build_candidate_rows(inventory: Path) -> list[RepairRow]:
    rows: list[RepairRow] = []
    with inventory.open() as f:
        for inv in csv.DictReader(f):
            method = inv["method"]
            if method not in {"flow_npe", "gaussian_npe"}:
                continue
            if inv["N_label"] not in STANDARD_LABELS:
                continue

            complete = int(inv["complete_bias_coverage_seed_count"])
            posterior = int(inv["posterior_seed_count"])
            missing = (
                parse_seed_list(inv["missing_coverage_seeds"])
                | parse_seed_list(inv["missing_bias_seeds"])
            )
            if method == "flow_npe":
                missing |= parse_seed_list(inv["missing_posterior_seeds"])
                include = complete < 100 or posterior < 100
            else:
                include = complete < 100

            if not include:
                continue
            for seed in sorted(missing):
                rows.append(
                    RepairRow(
                        row_index=len(rows),
                        method=method,
                        seed=seed,
                        n_obs=int(inv["n"]),
                        n_sims=int(inv["N"]),
                        n_label=inv["N_label"],
                    )
                )
    return rows


def missing_required(row: RepairRow, root: Path) -> tuple[str, ...]:
    out = row.output_dir(root)
    return tuple(name for name in row.required_files if not (out / name).exists())


def _array_finite(path: Path) -> bool:
    return bool(np.isfinite(np.load(path)).all())


def invalid_required(
    row: RepairRow,
    root: Path,
    *,
    expected_posterior_samples: int,
    expected_coverage_samples: int,
) -> tuple[str, ...]:
    out = row.output_dir(root)
    invalid: list[str] = []
    coverage = out / "estimated_coverage.npy"
    if coverage.exists():
        try:
            arr = np.load(coverage)
            if arr.shape != (3, 3) or not np.isfinite(arr).all():
                invalid.append("estimated_coverage.npy")
        except Exception:
            invalid.append("estimated_coverage.npy")

    biases = out / "biases.npy"
    if biases.exists():
        try:
            arr = np.load(biases)
            if arr.shape != (3 * expected_coverage_samples,) or not np.isfinite(arr).all():
                invalid.append("biases.npy")
        except Exception:
            invalid.append("biases.npy")

    posterior = out / "posterior_samples.pkl"
    if row.method == "flow_npe" and posterior.exists():
        try:
            with posterior.open("rb") as f:
                arr = np.asarray(pkl.load(f))
            if arr.shape != (expected_posterior_samples, 3) or not np.isfinite(arr).all():
                invalid.append("posterior_samples.pkl")
        except Exception:
            invalid.append("posterior_samples.pkl")

    return tuple(invalid)


def is_complete(
    row: RepairRow,
    root: Path,
    *,
    expected_posterior_samples: int,
    expected_coverage_samples: int,
) -> bool:
    return not missing_required(row, root) and not invalid_required(
        row,
        root,
        expected_posterior_samples=expected_posterior_samples,
        expected_coverage_samples=expected_coverage_samples,
    )


def build_manifest_rows(
    inventory: Path,
    canonical_root: Path,
    *,
    expected_posterior_samples: int,
    expected_coverage_samples: int,
) -> list[RepairRow]:
    rows = [
        row
        for row in build_candidate_rows(inventory)
        if not is_complete(
            row,
            canonical_root,
            expected_posterior_samples=expected_posterior_samples,
            expected_coverage_samples=expected_coverage_samples,
        )
    ]
    return renumber(rows)


def write_manifest(path: Path, rows: list[RepairRow], canonical_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "method",
                "seed",
                "n_obs",
                "n_sims",
                "N_label",
                "output_dir",
                "required_files",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row_index": row.row_index,
                    "method": row.method,
                    "seed": row.seed,
                    "n_obs": row.n_obs,
                    "n_sims": row.n_sims,
                    "N_label": row.n_label,
                    "output_dir": rel(row.output_dir(canonical_root)),
                    "required_files": ";".join(row.required_files),
                }
            )


def load_manifest(path: Path) -> list[RepairRow]:
    rows: list[RepairRow] = []
    with path.open() as f:
        for item in csv.DictReader(f):
            rows.append(
                RepairRow(
                    row_index=int(item["row_index"]),
                    method=item["method"],
                    seed=int(item["seed"]),
                    n_obs=int(item["n_obs"]),
                    n_sims=int(item["n_sims"]),
                    n_label=item["N_label"],
                )
            )
    return rows


def print_dry_run(
    rows: list[RepairRow],
    canonical_root: Path,
    staging_root: Path,
    *,
    expected_posterior_samples: int,
    expected_coverage_samples: int,
) -> None:
    print("stereological targeted repair dry-run")
    print(
        "row_index,method,seed,n_obs,n_sims,N_label,canonical_dir,staging_dir,"
        "canonical_missing,canonical_invalid,complete,will_run"
    )
    for row in rows:
        missing = missing_required(row, canonical_root)
        invalid = invalid_required(
            row,
            canonical_root,
            expected_posterior_samples=expected_posterior_samples,
            expected_coverage_samples=expected_coverage_samples,
        )
        complete = not missing and not invalid
        print(
            f"{row.row_index},{row.method},{row.seed},{row.n_obs},{row.n_sims},"
            f"{row.n_label},{rel(row.output_dir(canonical_root))},"
            f"{rel(row.output_dir(staging_root))},{';'.join(missing)},"
            f"{';'.join(invalid)},{str(complete).lower()},{str(not complete).lower()}"
        )


def verify_outputs(
    row: RepairRow,
    root: Path,
    *,
    expected_posterior_samples: int,
    expected_coverage_samples: int,
) -> None:
    missing = missing_required(row, root)
    if missing:
        raise FileNotFoundError(f"missing required outputs: {row.output_dir(root)}: {missing}")

    invalid = invalid_required(
        row,
        root,
        expected_posterior_samples=expected_posterior_samples,
        expected_coverage_samples=expected_coverage_samples,
    )
    if invalid:
        raise ValueError(f"invalid required outputs: {row.output_dir(root)}: {invalid}")
    print(f"verified {rel(row.output_dir(root))}")


def merge_required_files(row: RepairRow, staging_root: Path, canonical_root: Path) -> None:
    staged = row.output_dir(staging_root)
    canonical = row.output_dir(canonical_root)
    canonical.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in row.required_files:
        src = staged / name
        dst = canonical / name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        copied.append(name)
    print(
        f"merged {rel(staged)} -> {rel(canonical)}; "
        f"copied={','.join(copied) if copied else 'none'}"
    )


def run_row(row: RepairRow, args: argparse.Namespace) -> None:
    canonical_root = resolve_root(args.canonical_root)
    staging_root = resolve_root(args.staging_root)
    canonical_missing = missing_required(row, canonical_root)
    canonical_invalid = invalid_required(
        row,
        canonical_root,
        expected_posterior_samples=args.expected_posterior_samples,
        expected_coverage_samples=args.expected_coverage_samples,
    )
    if not canonical_missing and not canonical_invalid:
        raise FileExistsError(f"refusing already-complete row: {rel(row.output_dir(canonical_root))}")
    if canonical_invalid:
        raise ValueError(
            f"refusing to overwrite invalid canonical required files in "
            f"{rel(row.output_dir(canonical_root))}: {canonical_invalid}"
        )

    staged = row.output_dir(staging_root)
    if staged.exists():
        verify_outputs(
            row,
            staging_root,
            expected_posterior_samples=args.expected_posterior_samples,
            expected_coverage_samples=args.expected_coverage_samples,
        )
    else:
        print(
            f"repairing row={row.row_index} method={row.method} seed={row.seed} "
            f"n_obs={row.n_obs} n_sims={row.n_sims}; "
            f"canonical_missing={','.join(canonical_missing)}"
        )
        if row.method == "flow_npe":
            run_stereological(
                row.seed,
                row.n_obs,
                row.n_sims,
                output_root=staging_root,
                num_posterior_samples=args.num_posterior_samples,
                num_coverage_samples=args.num_coverage_samples,
                max_epochs=args.max_epochs,
                max_patience=args.max_patience,
                train_batch_size=args.train_batch_size,
                learning_rate=args.learning_rate,
            )
        elif row.method == "gaussian_npe":
            run_stereological_gaussian(
                row.seed,
                row.n_obs,
                row.n_sims,
                output_root=staging_root,
                num_posterior_samples=args.num_posterior_samples,
                num_coverage_samples=args.num_coverage_samples,
                max_epochs=args.max_epochs,
                patience=args.max_patience,
                train_batch_size=args.train_batch_size,
                learning_rate=args.learning_rate,
            )
        else:
            raise ValueError(f"unsupported method: {row.method}")
        verify_outputs(
            row,
            staging_root,
            expected_posterior_samples=args.expected_posterior_samples,
            expected_coverage_samples=args.expected_coverage_samples,
        )

    merge_required_files(row, staging_root, canonical_root)
    verify_outputs(
        row,
        canonical_root,
        expected_posterior_samples=args.expected_posterior_samples,
        expected_coverage_samples=args.expected_coverage_samples,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run targeted stereological repair rows.")
    parser.add_argument("--build-manifest-from-inventory", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--num-coverage-samples", type=int, default=100)
    parser.add_argument("--expected-posterior-samples", type=int, default=10_000)
    parser.add_argument("--expected-coverage-samples", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--max-patience", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    canonical_root = resolve_root(args.canonical_root)
    staging_root = resolve_root(args.staging_root)

    if args.build_manifest_from_inventory:
        rows = build_manifest_rows(
            args.build_manifest_from_inventory,
            canonical_root,
            expected_posterior_samples=args.expected_posterior_samples,
            expected_coverage_samples=args.expected_coverage_samples,
        )
        if not args.write_manifest:
            raise ValueError("--write-manifest is required when building a manifest")
        write_manifest(args.write_manifest, rows, canonical_root)
        print(f"wrote manifest with {len(rows)} rows: {args.write_manifest}")
        return

    if not args.manifest:
        raise ValueError("--manifest is required unless building a manifest")
    rows = load_manifest(args.manifest)
    if args.dry_run:
        print_dry_run(
            rows,
            canonical_root,
            staging_root,
            expected_posterior_samples=args.expected_posterior_samples,
            expected_coverage_samples=args.expected_coverage_samples,
        )
        return
    if args.row_index is None:
        raise ValueError("--row-index is required for execution")
    lookup = {row.row_index: row for row in rows}
    if args.row_index not in lookup:
        raise IndexError(f"row index {args.row_index} not in manifest")
    run_row(lookup[args.row_index], args)


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    main()
