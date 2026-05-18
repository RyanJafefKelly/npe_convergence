"""Submit the May 2026 GNK high-budget staged runs from manifests.

The script is intentionally a thin driver around
``launch_high_budget_staged_pipeline.sh`` so the PBS dependency-chain logic
stays in one place. Dry-run is the default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "docs" / "paper_empirical_push_2026_05_13" / "hpc_canary_planning_20260513"
GNK_MANIFEST = MANIFEST_DIR / "gnk_octile_flow_high_budget_manifest_20260513.csv"
STEREO_MANIFEST = MANIFEST_DIR / "stereological_high_budget_manifest_20260513.csv"
LAUNCHER = REPO_ROOT / "npe_convergence" / "scripts" / "launch_high_budget_staged_pipeline.sh"


@dataclass(frozen=True)
class Cell:
    source: str
    row_index: int
    model: str
    method: str
    seed: int
    n_obs: int
    n_sims: int


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="") as f:
        yield from csv.DictReader(f)


def load_gnk_cells(path: Path) -> list[Cell]:
    cells: list[Cell] = []
    for row in iter_csv(path):
        if not truthy(row.get("will_run")) or truthy(row.get("complete")):
            continue
        method = row["method"]
        if method != "gnk_octile_flow_npe":
            raise ValueError(f"unexpected GNK method in {path}: {method}")
        cells.append(
            Cell(
                source="gnk",
                row_index=int(row["row_index"]),
                model="gnk",
                method="flow_npe",
                seed=int(row["seed"]),
                n_obs=int(row["n_obs"]),
                n_sims=int(row["n_sims"]),
            )
        )
    return cells


def load_stereo_cells(path: Path) -> list[Cell]:
    cells: list[Cell] = []
    for row in iter_csv(path):
        if not truthy(row.get("will_run")) or truthy(row.get("complete")):
            continue
        method = row["method"]
        if method not in {"flow_npe", "gaussian_npe"}:
            raise ValueError(f"unexpected stereological method in {path}: {method}")
        cells.append(
            Cell(
                source="stereological",
                row_index=int(row["row_index"]),
                model="stereological",
                method=method,
                seed=int(row["seed"]),
                n_obs=int(row["n_obs"]),
                n_sims=int(row["n_sims"]),
            )
        )
    return cells


def selected_cells(args: argparse.Namespace) -> list[Cell]:
    cells: list[Cell] = []
    if args.only in {"all", "gnk"}:
        cells.extend(load_gnk_cells(args.gnk_manifest))
    if args.only in {"all", "stereological"}:
        cells.extend(load_stereo_cells(args.stereological_manifest))
    cells = cells[args.start_at :]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    return cells


def cell_env(args: argparse.Namespace, cell: Cell) -> dict[str, str]:
    env = {
        "MODEL": cell.model,
        "METHOD": cell.method,
        "SEED": str(cell.seed),
        "N_OBS": str(cell.n_obs),
        "N_SIMS": str(cell.n_sims),
        "STAGING_ROOT": args.staging_root,
        "SHARD_SIZE": str(args.shard_size),
        "SIM_BATCH_SIZE": str(args.sim_batch_size),
        "COVERAGE_SAMPLES": str(args.coverage_samples),
        "COVERAGE_REPS": str(args.coverage_reps),
        "TRAIN_REPEATS": str(args.train_repeats),
        "EPOCHS_THIS_RUN": str(args.epochs_this_run),
        "MAX_EPOCHS": str(args.max_epochs),
        "PATIENCE": str(args.patience),
        "TRAIN_WALLTIME": args.train_walltime,
        "SIM_WALLTIME": args.sim_walltime,
        "EVAL_WALLTIME": args.eval_walltime,
        "AGGREGATE_SIMS_WALLTIME": args.aggregate_sims_walltime,
        "FINAL_WALLTIME": args.final_walltime,
        "SIM_MEM": args.sim_mem,
        "AGGREGATE_SIMS_MEM": args.aggregate_sims_mem,
        "TRAIN_MEM": args.train_mem,
        "EVAL_MEM": args.eval_mem,
        "FINAL_MEM": args.final_mem,
    }
    if args.queue:
        env["QUEUE"] = args.queue
    return env


def append_log(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Call qsub instead of printing dry-run commands.")
    parser.add_argument("--only", choices=("all", "gnk", "stereological"), default="gnk")
    parser.add_argument("--start-at", type=int, default=0, help="Skip this many selected cells.")
    parser.add_argument("--max-cells", type=int, default=None, help="Limit selected cells, useful for smoke checks.")
    parser.add_argument("--staging-root", default="res/staged_high_budget_20260517_all_remaining")
    parser.add_argument("--gnk-manifest", type=Path, default=GNK_MANIFEST)
    parser.add_argument("--stereological-manifest", type=Path, default=STEREO_MANIFEST)
    parser.add_argument(
        "--submission-log",
        type=Path,
        default=REPO_ROOT
        / "docs"
        / "paper_empirical_push_2026_05_13"
        / "hpc_submission_20260517"
        / "staged_all_remaining_submission.jsonl",
    )
    parser.add_argument("--shard-size", type=int, default=100000)
    parser.add_argument("--sim-batch-size", type=int, default=1000)
    parser.add_argument("--coverage-samples", type=int, default=100)
    parser.add_argument("--coverage-reps", type=int, default=10)
    parser.add_argument("--train-repeats", type=int, default=10)
    parser.add_argument("--epochs-this-run", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--queue", default="")
    parser.add_argument("--sim-walltime", default="08:00:00")
    parser.add_argument("--aggregate-sims-walltime", default="04:00:00")
    parser.add_argument("--train-walltime", default="47:00:00")
    parser.add_argument("--eval-walltime", default="08:00:00")
    parser.add_argument("--final-walltime", default="04:00:00")
    parser.add_argument("--sim-mem", default="32gb")
    parser.add_argument("--aggregate-sims-mem", default="64gb")
    parser.add_argument("--train-mem", default="64gb")
    parser.add_argument("--eval-mem", default="32gb")
    parser.add_argument("--final-mem", default="64gb")
    args = parser.parse_args()

    if args.max_cells is not None and args.max_cells < 0:
        raise SystemExit("--max-cells must be non-negative")
    cells = selected_cells(args)
    mode = "submit" if args.submit else "dry-run"
    print(f"mode={mode} selected_cells={len(cells)} staging_root={args.staging_root}", flush=True)
    if not cells:
        return

    command = ["bash", str(LAUNCHER), "--submit" if args.submit else "--dry-run"]
    for ordinal, cell in enumerate(cells, start=args.start_at):
        env_overrides = cell_env(args, cell)
        run_env = os.environ.copy()
        run_env.update(env_overrides)
        print(
            f"[{ordinal}] {cell.source} {cell.model}/{cell.method} "
            f"seed={cell.seed} n_obs={cell.n_obs} n_sims={cell.n_sims}",
            flush=True,
        )
        started = time.time()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=run_env,
            text=True,
            capture_output=True,
            check=False,
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        payload = {
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": mode,
            "ordinal": ordinal,
            "source": cell.source,
            "row_index": cell.row_index,
            "model": cell.model,
            "method": cell.method,
            "seed": cell.seed,
            "n_obs": cell.n_obs,
            "n_sims": cell.n_sims,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
            "command": command,
            "env": env_overrides,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        append_log(args.submission_log, payload)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
