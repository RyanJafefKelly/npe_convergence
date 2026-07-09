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
import shutil
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
PBS_DIR = REPO_ROOT / "npe_convergence" / "scripts" / "pbs_jobs"


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
        "LEARNING_RATE": str(args.learning_rate),
        "TRAIN_BATCH_SIZE": str(args.train_batch_size),
        "EPOCHS_THIS_RUN": str(args.epochs_this_run),
        "MAX_EPOCHS": str(args.max_epochs),
        "PATIENCE": str(args.patience),
        "VAL_FRAC": str(args.val_frac),
        "CHECKPOINT_EVERY_EPOCHS": str(args.checkpoint_every_epochs),
        "LOG_EVERY_EPOCHS": str(args.log_every_epochs),
        "MAX_RUNTIME_SECONDS": str(args.max_runtime_seconds),
        "WALLTIME_BUFFER_SECONDS": str(args.walltime_buffer_seconds),
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


def qsub_command() -> str:
    return shutil.which("qsub") or "/opt/pbs/bin/qsub"


def run_or_print_qsub(args: argparse.Namespace, command: list[str]) -> subprocess.CompletedProcess[str] | None:
    if not args.submit:
        print("+" + "".join(f" {part!r}" for part in command), flush=True)
        return None
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def qsub_base_args(args: argparse.Namespace) -> list[str]:
    qsub = qsub_command()
    command = [qsub]
    if args.queue:
        command.extend(["-q", args.queue])
    return command


def comma_vars(env: dict[str, str], keys: Iterable[str]) -> str:
    return ",".join(f"{key}={env[key]}" for key in keys)


COMMON_KEYS = ("MODEL", "METHOD", "SEED", "N_OBS", "N_SIMS", "STAGING_ROOT")
TRAIN_KEYS = (
    *COMMON_KEYS,
    "LEARNING_RATE",
    "TRAIN_BATCH_SIZE",
    "MAX_EPOCHS",
    "EPOCHS_THIS_RUN",
    "PATIENCE",
    "VAL_FRAC",
    "CHECKPOINT_EVERY_EPOCHS",
    "LOG_EVERY_EPOCHS",
    "MAX_RUNTIME_SECONDS",
    "WALLTIME_BUFFER_SECONDS",
)


def run_train_only(args: argparse.Namespace, cell: Cell, env: dict[str, str]) -> tuple[int, str, str]:
    train_template = PBS_DIR / "high_budget_staged_train.sh"
    prev_job = ""
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    for index in range(1, args.train_repeats + 1):
        require_complete = "1" if args.require_complete_on_last_train and index == args.train_repeats else "0"
        command = qsub_base_args(args)
        if prev_job:
            command.extend(["-W", f"depend=afterok:{prev_job}"])
        command.extend(
            [
                "-l",
                f"walltime={args.train_walltime}",
                "-l",
                f"mem={args.train_mem}",
                "-v",
                f"{comma_vars(env, TRAIN_KEYS)},REQUIRE_TRAINING_COMPLETE={require_complete}",
                str(train_template),
            ]
        )
        result = run_or_print_qsub(args, command)
        if result is None:
            prev_job = f"<TRAIN_JOB_{index}>"
            continue
        stdout_parts.append(result.stdout)
        stderr_parts.append(result.stderr)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            return result.returncode, "".join(stdout_parts), "".join(stderr_parts)
        prev_job = result.stdout.strip()
        print(f"train job {index}: {prev_job}", flush=True)
    return 0, "".join(stdout_parts), "".join(stderr_parts)


def cell_run_dir(args: argparse.Namespace, cell: Cell) -> Path:
    return (
        REPO_ROOT
        / args.staging_root
        / f"{cell.model}_{cell.method}_n_obs_{cell.n_obs}_n_sims_{cell.n_sims}_seed_{cell.seed}"
    )


def training_complete(args: argparse.Namespace, cell: Cell) -> bool:
    path = cell_run_dir(args, cell) / "training" / f"{cell.method}_complete.json"
    if not path.exists():
        return False
    with path.open() as f:
        payload = json.load(f)
    return bool(payload.get("complete"))


def run_eval_complete(args: argparse.Namespace, cell: Cell, env: dict[str, str]) -> tuple[int, str, str]:
    if not training_complete(args, cell):
        message = f"skip incomplete training: seed={cell.seed}\n"
        print(message, end="", flush=True)
        return 0, message, ""

    eval_template = PBS_DIR / "high_budget_staged_evaluate_array.sh"
    final_template = PBS_DIR / "high_budget_staged_aggregate_verify.sh"
    eval_last = (args.coverage_samples + args.coverage_reps - 1) // args.coverage_reps - 1
    eval_vars = (
        f"{comma_vars(env, COMMON_KEYS)},COVERAGE_SAMPLES={args.coverage_samples},"
        f"COVERAGE_REPS={args.coverage_reps},NUM_POSTERIOR_SAMPLES={args.num_posterior_samples},"
        "INCLUDE_POSTERIOR_ON_SHARD0=1"
    )
    eval_command = qsub_base_args(args)
    eval_command.extend(
        [
            "-J",
            f"0-{eval_last}",
            "-l",
            f"walltime={args.eval_walltime}",
            "-l",
            f"mem={args.eval_mem}",
            "-v",
            eval_vars,
            str(eval_template),
        ]
    )
    eval_result = run_or_print_qsub(args, eval_command)
    if eval_result is None:
        eval_job = "<EVAL_JOB>"
        stdout = ""
        stderr = ""
    else:
        sys.stdout.write(eval_result.stdout)
        sys.stderr.write(eval_result.stderr)
        stdout = eval_result.stdout
        stderr = eval_result.stderr
        if eval_result.returncode != 0:
            return eval_result.returncode, stdout, stderr
        eval_job = eval_result.stdout.strip()
        print(f"evaluation job: {eval_job}", flush=True)

    final_vars = (
        f"{comma_vars(env, COMMON_KEYS)},METRIC_SAMPLES={args.metric_samples},"
        f"EXPECTED_POSTERIOR_SAMPLES={args.expected_posterior_samples},"
        f"EXPECTED_COVERAGE_SAMPLES={args.coverage_samples}"
    )
    final_command = qsub_base_args(args)
    final_command.extend(
        [
            "-W",
            f"depend=afterok:{eval_job}",
            "-l",
            f"walltime={args.final_walltime}",
            "-l",
            f"mem={args.final_mem}",
            "-v",
            final_vars,
            str(final_template),
        ]
    )
    final_result = run_or_print_qsub(args, final_command)
    if final_result is None:
        return 0, stdout, stderr
    sys.stdout.write(final_result.stdout)
    sys.stderr.write(final_result.stderr)
    stdout += final_result.stdout
    stderr += final_result.stderr
    if final_result.returncode == 0:
        print(f"final aggregate+verify job: {final_result.stdout.strip()}", flush=True)
    return final_result.returncode, stdout, stderr


def append_log(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Call qsub instead of printing dry-run commands.")
    parser.add_argument(
        "--mode",
        choices=("full", "train-only", "eval-complete"),
        default="full",
        help="Submit the full pipeline, only additional train chunks, or eval/final for completed training.",
    )
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
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--metric-samples", type=int, default=2000)
    parser.add_argument("--expected-posterior-samples", type=int, default=10_000)
    parser.add_argument("--train-repeats", type=int, default=10)
    parser.add_argument("--learning-rate", default="5e-4")
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--epochs-this-run", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--log-every-epochs", type=int, default=1)
    parser.add_argument("--max-runtime-seconds", type=int, default=162000)
    parser.add_argument("--walltime-buffer-seconds", type=int, default=1800)
    parser.add_argument(
        "--require-complete-on-last-train",
        action="store_true",
        help="Make the last train-only job fail if training is not complete.",
    )
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
    print(
        f"mode={mode} run_mode={args.mode} selected_cells={len(cells)} "
        f"staging_root={args.staging_root}",
        flush=True,
    )
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
        if args.mode == "full":
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
            returncode = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        elif args.mode == "train-only":
            returncode, stdout, stderr = run_train_only(args, cell, env_overrides)
        else:
            returncode, stdout, stderr = run_eval_complete(args, cell, env_overrides)
        payload = {
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": mode,
            "run_mode": args.mode,
            "ordinal": ordinal,
            "source": cell.source,
            "row_index": cell.row_index,
            "model": cell.model,
            "method": cell.method,
            "seed": cell.seed,
            "n_obs": cell.n_obs,
            "n_sims": cell.n_sims,
            "returncode": returncode,
            "elapsed_seconds": time.time() - started,
            "command": command,
            "env": env_overrides,
            "stdout": stdout,
            "stderr": stderr,
        }
        append_log(args.submission_log, payload)
        if returncode != 0:
            raise SystemExit(returncode)


if __name__ == "__main__":
    main()
