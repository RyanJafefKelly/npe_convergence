from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "npe_convergence"
    / "scripts"
    / "submit_remaining_high_budget_staged.py"
)


def write_gnk_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "method",
        "seed",
        "n_obs",
        "n_sims",
        "complete",
        "will_run",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "row_index": "0",
                "method": "gnk_octile_flow_npe",
                "seed": "0",
                "n_obs": "5000",
                "n_sims": "25000000",
                "complete": "False",
                "will_run": "True",
            }
        )


def test_train_only_dry_run_starts_at_train_template(tmp_path: Path) -> None:
    manifest = tmp_path / "gnk_manifest.csv"
    write_gnk_manifest(manifest)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode=train-only",
            "--gnk-manifest",
            str(manifest),
            "--submission-log",
            str(tmp_path / "submission.jsonl"),
            "--train-repeats=2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "run_mode=train-only selected_cells=1" in result.stdout
    assert "high_budget_staged_train.sh" in result.stdout
    assert "high_budget_staged_simulate_array.sh" not in result.stdout
    assert "REQUIRE_TRAINING_COMPLETE=0" in result.stdout


def test_eval_complete_dry_run_skips_incomplete_training(tmp_path: Path) -> None:
    manifest = tmp_path / "gnk_manifest.csv"
    write_gnk_manifest(manifest)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode=eval-complete",
            "--gnk-manifest",
            str(manifest),
            "--staging-root",
            str(tmp_path / "missing_stage"),
            "--submission-log",
            str(tmp_path / "submission.jsonl"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "skip incomplete training: seed=0" in result.stdout
    assert "high_budget_staged_evaluate_array.sh" not in result.stdout
