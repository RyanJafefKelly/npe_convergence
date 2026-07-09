#!/usr/bin/env python
"""Drive the GNK canonical NUTS reference rerun locally.

Reads the manifest CSV produced by plan_gnk_nuts_refresh.py and runs each
cell sequentially, skipping cells whose output already exists. Writes a
running progress log to stdout and a final summary to a JSON file.

Intended for local-CPU execution. For HPC, submit the PBS template instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "docs"
    / "meeting_2026_05_18"
    / "gnk_nuts_refresh_plan"
    / "gnk_nuts_refresh_manifest.csv"
)
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "docs"
    / "meeting_2026_05_18"
    / "gnk_nuts_refresh_plan"
    / "gnk_nuts_refresh_run_log.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--filter-n-obs",
        type=str,
        default="",
        help="If non-empty, only run rows with n_obs in this comma list.",
    )
    parser.add_argument(
        "--filter-convention",
        type=str,
        default="",
        help="If non-empty, only run rows with this convention.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after this many cells (0 = all)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to runner, overwriting existing output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands but do not execute.",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open()))
    n_total = len(rows)

    filt_n_obs = (
        {int(x) for x in args.filter_n_obs.split(",") if x.strip()}
        if args.filter_n_obs.strip()
        else None
    )
    filt_conv = args.filter_convention or None

    runs: list[dict[str, object]] = []
    skipped = 0
    failed = 0
    succeeded = 0
    start_wall = time.perf_counter()

    for i, row in enumerate(rows, 1):
        if filt_n_obs is not None and int(row["n_obs"]) not in filt_n_obs:
            continue
        if filt_conv is not None and row["convention"] != filt_conv:
            continue
        output_path = REPO_ROOT / row["output_path"]
        if output_path.exists() and not args.force:
            skipped += 1
            continue
        cmd = row["runtime_command"]
        if args.force and "--force" not in cmd:
            cmd = cmd + " --force"
        print(
            f"[{i}/{n_total}] n_obs={row['n_obs']} seed={row['seed']} "
            f"convention={row['convention']}",
            flush=True,
        )
        if args.dry_run:
            print(f"  DRY: {cmd}", flush=True)
            continue
        t0 = time.perf_counter()
        result = subprocess.run(
            cmd, shell=True, cwd=REPO_ROOT, capture_output=True, text=True
        )
        elapsed = time.perf_counter() - t0
        ok = result.returncode == 0
        if ok:
            succeeded += 1
        else:
            failed += 1
            print(f"  FAIL ({result.returncode}) in {elapsed:.1f}s", flush=True)
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
        runs.append(
            {
                "n_obs": int(row["n_obs"]),
                "seed": int(row["seed"]),
                "convention": row["convention"],
                "output_path": row["output_path"],
                "elapsed_seconds": elapsed,
                "return_code": result.returncode,
                "stdout_tail": result.stdout[-500:],
                "stderr_tail": result.stderr[-500:],
            }
        )
        if args.limit and (succeeded + failed) >= args.limit:
            print(
                f"  hit --limit {args.limit}, stopping",
                flush=True,
            )
            break

    wall = time.perf_counter() - start_wall
    summary = {
        "manifest": str(args.manifest),
        "total_rows": n_total,
        "executed": len(runs),
        "succeeded": succeeded,
        "failed": failed,
        "skipped_existing": skipped,
        "wall_seconds": wall,
        "completed_at": utc_now(),
        "runs": runs,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Done. {succeeded} ok, {failed} failed, {skipped} skipped, "
        f"wall {wall:.0f}s. Wrote {args.summary}",
        flush=True,
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
