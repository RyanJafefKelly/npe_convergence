"""Run or inspect one GNK high-budget array row from a dry-run manifest."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpyro  # type: ignore

from npe_convergence.scripts.run_gnk_gaussian_hpc_calibration import (
    REPO_ROOT,
    load_config,
    run_calibration,
)


def load_row(manifest_path: Path, array_index: int) -> dict[str, Any]:
    with manifest_path.open() as f:
        manifest = json.load(f)
    if not manifest.get("dry_run_only"):
        raise ValueError("manifest is not marked as a dry-run artifact")
    matches = [
        row
        for row in manifest["rows"]
        if row["action"] == "run" and row["pbs_array_index"] == array_index
    ]
    if len(matches) != 1:
        raise ValueError(f"array index {array_index} did not resolve to exactly one run row")
    return matches[0]


def runtime_collision_paths(row: dict[str, Any]) -> list[str]:
    collisions: list[str] = []
    output_dir = REPO_ROOT / row["paths"]["output_dir"]
    if output_dir.exists():
        collisions.append(row["paths"]["output_dir"])
    for key, path in row["paths"].items():
        if key in {"config", "output_dir"}:
            continue
        if (REPO_ROOT / path).exists():
            collisions.append(path)
    return collisions


def print_row_summary(row: dict[str, Any], array_index: int) -> None:
    print(f"pbs_array_index: {array_index}")
    print(f"manifest_index: {row['manifest_index']}")
    print(f"action: {row['action']}")
    print(f"run_id: {row['run_id']}")
    print(f"n: {row['n']}")
    print(f"d: {row['d']}")
    print(f"x: {row['x']}")
    print(f"N: {row['N']}")
    print(f"seed: {row['seed']}")
    print(f"config_path: {row['config_path']}")
    print(f"output_dir: {row['paths']['output_dir']}")
    print(f"stdout_log: {row['paths']['stdout_log']}")
    print(f"stderr_log: {row['paths']['stderr_log']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the row for this array index without training.",
    )
    args = parser.parse_args(argv)

    array_index = args.array_index
    if array_index is None:
        array_index = int(os.environ["PBS_ARRAY_INDEX"])

    manifest_path = REPO_ROOT / args.manifest
    row = load_row(manifest_path, array_index)
    if args.dry_run:
        print_row_summary(row, array_index)
        collisions = runtime_collision_paths(row)
        if collisions:
            print("runtime_output_collisions:")
            for path in collisions:
                print(f"  {path}")
            return 1
        print("runtime_output_collisions: none")
        return 0

    collisions = runtime_collision_paths(row)
    if collisions:
        print("Refusing to run because output paths already exist:", file=sys.stderr)
        for path in collisions:
            print(f"  {path}", file=sys.stderr)
        return 1

    config = load_config(row["config_path"])
    stdout_path = REPO_ROOT / row["paths"]["stdout_log"]
    stderr_path = REPO_ROOT / row["paths"]["stderr_log"]
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    numpyro.set_host_device_count(4)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"manifest: {args.manifest}")
            print(f"pbs_array_index: {array_index}")
            print(f"run_id: {row['run_id']}")
            print(f"config: {row['config_path']}")
            try:
                return run_calibration(config)
            except Exception:
                traceback.print_exc()
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
