"""Targeted repair runner for the incomplete GNK octile flow cell.

This intentionally does not run the full GNK paper grid. It only permits the
reviewed shortfall cell:

    flow-NPE, n_obs=5000, n_sims=25000000, seeds 0-35.

The legacy output directories already exist for these seeds, but they are
partial. This wrapper refuses complete cells and only calls the legacy runner
when required artifacts are missing.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle as pkl
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpyro  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.scripts.run_gnk import run_gnk


TARGET_N_OBS = 5000
TARGET_N_SIMS = 25_000_000
TARGET_SEEDS = tuple(range(36))
DEFAULT_CANONICAL_ROOT = REPO_ROOT / "res" / "gnk"
DEFAULT_STAGING_ROOT = REPO_ROOT / "res" / "gnk_octile_flow_targeted_repair_20260513_canary"
REQUIRED_FILES = (
    "posterior_samples.pkl",
    "true_posterior_samples.pkl",
    "kl.txt",
    "mmd.txt",
    "estimated_coverage.npy",
    "biases.npy",
)


@dataclass(frozen=True)
class CellState:
    seed: int
    n_obs: int
    n_sims: int
    output_dir: Path
    dir_exists: bool
    existing_files: tuple[str, ...]
    missing_required: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.dir_exists and not self.missing_required

    @property
    def will_run(self) -> bool:
        return not self.complete


def parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return values


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


def output_dir(root: Path, seed: int, n_obs: int, n_sims: int) -> Path:
    return root / f"npe_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}"


def inspect_cell(root: Path, seed: int, n_obs: int, n_sims: int) -> CellState:
    out = output_dir(root, seed, n_obs, n_sims)
    existing_files = tuple(sorted(path.name for path in out.iterdir())) if out.exists() else ()
    missing_required = tuple(name for name in REQUIRED_FILES if not (out / name).exists())
    return CellState(
        seed=seed,
        n_obs=n_obs,
        n_sims=n_sims,
        output_dir=out,
        dir_exists=out.exists(),
        existing_files=existing_files,
        missing_required=missing_required,
    )


def validate_target(seed: int, n_obs: int, n_sims: int) -> None:
    if n_obs != TARGET_N_OBS or n_sims != TARGET_N_SIMS:
        raise ValueError(
            "This repair runner only permits "
            f"n_obs={TARGET_N_OBS}, n_sims={TARGET_N_SIMS}; "
            f"got n_obs={n_obs}, n_sims={n_sims}"
        )
    if seed not in TARGET_SEEDS:
        raise ValueError(
            "This repair runner only permits seeds 0-35 for the reviewed "
            f"shortfall; got seed={seed}"
        )


def state_row(state: CellState) -> dict[str, str | int | bool]:
    return {
        "seed": state.seed,
        "n_obs": state.n_obs,
        "n_sims": state.n_sims,
        "output_dir": rel(state.output_dir),
        "dir_exists": state.dir_exists,
        "existing_file_count": len(state.existing_files),
        "missing_required": ";".join(state.missing_required),
        "complete": state.complete,
        "will_run": state.will_run,
    }


def write_manifest(path: Path, states: list[CellState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [state_row(state) for state in states]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_dry_run(states: list[CellState], staging_root: Path) -> None:
    print("GNK octile flow targeted repair dry-run")
    print(f"target: n_obs={TARGET_N_OBS}, n_sims={TARGET_N_SIMS}, seeds=0-35")
    print(
        "seed,n_obs,n_sims,canonical_dir,staging_dir,dir_exists,"
        "existing_file_count,missing_required,complete,will_run"
    )
    for state in states:
        row = state_row(state)
        staged = output_dir(staging_root, state.seed, state.n_obs, state.n_sims)
        print(
            f"{row['seed']},{row['n_obs']},{row['n_sims']},{row['output_dir']},"
            f"{rel(staged)},"
            f"{str(row['dir_exists']).lower()},{row['existing_file_count']},"
            f"{row['missing_required']},{str(row['complete']).lower()},"
            f"{str(row['will_run']).lower()}"
        )


def parse_float_file(path: Path) -> float:
    return float(path.read_text().strip())


def verify_outputs(state: CellState) -> None:
    missing = [name for name in REQUIRED_FILES if not (state.output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Repair did not create required files in {state.output_dir}: {missing}"
        )

    with (state.output_dir / "posterior_samples.pkl").open("rb") as f:
        posterior = np.asarray(pkl.load(f))
    with (state.output_dir / "true_posterior_samples.pkl").open("rb") as f:
        true = np.asarray(pkl.load(f))
    coverage = np.load(state.output_dir / "estimated_coverage.npy")
    biases = np.load(state.output_dir / "biases.npy")
    kl = parse_float_file(state.output_dir / "kl.txt")
    mmd = parse_float_file(state.output_dir / "mmd.txt")

    checks = {
        "posterior_samples.pkl shape": posterior.shape == (10_000, 4),
        "true_posterior_samples.pkl shape": true.shape == (10_000, 4),
        "estimated_coverage.npy shape": coverage.shape == (4, 3),
        "biases.npy shape": biases.shape == (400,),
        "mmd finite": math.isfinite(mmd),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Repair output verification failed for {state.output_dir}: {failed}")
    print(
        f"verified {rel(state.output_dir)} "
        f"kl={kl}, kl_finite={math.isfinite(kl)}, mmd={mmd}"
    )


def merge_required_files(staged_state: CellState, canonical_root: Path) -> None:
    canonical = output_dir(
        canonical_root,
        staged_state.seed,
        staged_state.n_obs,
        staged_state.n_sims,
    )
    canonical.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in REQUIRED_FILES:
        src = staged_state.output_dir / name
        dst = canonical / name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        copied.append(name)
    print(
        f"merged {rel(staged_state.output_dir)} -> {rel(canonical)}; "
        f"copied={','.join(copied) if copied else 'none'}"
    )


def run_repair_for_state(
    state: CellState,
    *,
    staging_root: Path,
    canonical_root: Path,
    merge_to_canonical: bool,
) -> None:
    validate_target(state.seed, state.n_obs, state.n_sims)
    if state.complete:
        raise FileExistsError(
            f"Refusing to rerun complete cell: {rel(state.output_dir)}"
        )
    if state.dir_exists:
        print(
            "repairing existing partial directory "
            f"{rel(state.output_dir)}; "
            f"missing required files: {', '.join(state.missing_required)}"
        )
    else:
        print(f"repairing missing directory {rel(state.output_dir)}")

    staged_state = inspect_cell(staging_root, state.seed, state.n_obs, state.n_sims)
    if staged_state.complete:
        print(f"reusing complete staged output {rel(staged_state.output_dir)}")
    else:
        run_gnk(state.seed, state.n_obs, state.n_sims, output_root=staging_root)
        staged_state = inspect_cell(staging_root, state.seed, state.n_obs, state.n_sims)

    verify_outputs(staged_state)
    if merge_to_canonical:
        merge_required_files(staged_state, canonical_root)
        verify_outputs(inspect_cell(canonical_root, state.seed, state.n_obs, state.n_sims))
    else:
        print("merge_to_canonical=false; leaving verified outputs in staging only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run only the reviewed GNK octile flow n=5000, N=n^2 repair.",
    )
    parser.add_argument("--seed", type=str, default="0")
    parser.add_argument("--n-obs", type=int, default=TARGET_N_OBS)
    parser.add_argument("--n-sims", type=int, default=TARGET_N_SIMS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    parser.add_argument("--merge-to-canonical", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    canonical_root = resolve_root(args.canonical_root)
    staging_root = resolve_root(args.staging_root)
    seeds = parse_int_list(args.seed)
    for seed in seeds:
        validate_target(seed, args.n_obs, args.n_sims)
    states = [inspect_cell(canonical_root, seed, args.n_obs, args.n_sims) for seed in seeds]

    if args.dry_run:
        print_dry_run(states, staging_root)
        if args.write_manifest:
            write_manifest(args.write_manifest, states)
            print(f"wrote manifest: {args.write_manifest}")
        return

    for state in states:
        run_repair_for_state(
            state,
            staging_root=staging_root,
            canonical_root=canonical_root,
            merge_to_canonical=args.merge_to_canonical,
        )


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    main()
