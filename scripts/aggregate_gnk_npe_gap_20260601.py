#!/usr/bin/env python
"""Aggregate the 2026-06-01 GNK NPE-gap diagnostic outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


GAUSSIAN_SEEDS = range(25)
FLOW_SEEDS = range(15)
GAUSSIAN_BUDGETS = (1000, 6907, 31623, 1_000_000, 3_000_000)
FLOW_BUDGETS = (1_000_000, 3_000_000)
N_OBS = 1000


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {"_read_error": str(exc)}


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def metric_row(
    *,
    cells_root: Path,
    method: str,
    standardisation: str,
    seed: int,
    n_obs: int,
    n_sims: int,
    dirname: str,
) -> dict[str, Any]:
    metrics_path = cells_root / dirname / "metrics.json"
    metrics = read_json(metrics_path)
    theta_kl = None
    status = "missing"
    if metrics is not None:
        if "_read_error" in metrics:
            status = "metrics_read_error"
        else:
            theta_kl = finite_float(metrics.get("kl_value"))
            status = "finite_metric" if theta_kl is not None else "nonfinite_metric"
    return {
        "method": method,
        "standardisation": standardisation,
        "seed": seed,
        "n_obs": n_obs,
        "N_sims": n_sims,
        "theta_kl": "" if theta_kl is None else f"{theta_kl:.12g}",
        "status": status,
    }


def baseline_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            try:
                n = int(rec["n"])
                n_sims = int(rec["N"])
                seed = int(rec["seed"])
            except (KeyError, TypeError, ValueError):
                continue
            if n != N_OBS or n_sims != 1_000_000:
                continue
            rows.append(
                {
                    "method": "gaussian_npe",
                    "standardisation": "standard_zscore_baseline",
                    "seed": seed,
                    "n_obs": n,
                    "N_sims": n_sims,
                    "theta_kl": rec.get("gaussian_theta_kl", ""),
                    "status": "baseline_from_v3_csv",
                }
            )
            rows.append(
                {
                    "method": "flow_npe",
                    "standardisation": "standard_zscore_baseline",
                    "seed": seed,
                    "n_obs": n,
                    "N_sims": n_sims,
                    "theta_kl": rec.get("flow_theta_kl", ""),
                    "status": "baseline_from_v3_csv",
                }
            )
    return rows


def expected_rows(cells_root: Path, include_flow_1e7: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in GAUSSIAN_SEEDS:
        for n_sims in GAUSSIAN_BUDGETS:
            rows.append(
                metric_row(
                    cells_root=cells_root,
                    method="gaussian_npe",
                    standardisation="robust_asinh_median_iqr",
                    seed=seed,
                    n_obs=N_OBS,
                    n_sims=n_sims,
                    dirname=(
                        f"gaussian_npe_n_obs_{N_OBS}_n_sims_{n_sims}_"
                        f"seed_{seed}_transform_asinh"
                    ),
                )
            )
        rows.append(
            metric_row(
                cells_root=cells_root,
                method="gaussian_npe",
                standardisation="identity_median_iqr",
                seed=seed,
                n_obs=N_OBS,
                n_sims=1_000_000,
                dirname=(
                    f"gaussian_npe_n_obs_{N_OBS}_n_sims_1000000_"
                    f"seed_{seed}_transform_identity"
                ),
            )
        )

    flow_budgets = list(FLOW_BUDGETS)
    if include_flow_1e7:
        flow_budgets.append(10_000_000)
    for seed in FLOW_SEEDS:
        for n_sims in flow_budgets:
            rows.append(
                metric_row(
                    cells_root=cells_root,
                    method="flow_npe",
                    standardisation="robust_asinh_median_iqr",
                    seed=seed,
                    n_obs=N_OBS,
                    n_sims=n_sims,
                    dirname=(
                        f"flow_npe_n_obs_{N_OBS}_n_sims_{n_sims}_"
                        f"seed_{seed}_transform_asinh"
                    ),
                )
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "standardisation", "seed", "n_obs", "N_sims", "theta_kl", "status"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def values_by_group(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[float]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = finite_float(row.get("theta_kl"))
        if value is None:
            continue
        key = (str(row["method"]), str(row["standardisation"]), int(row["N_sims"]))
        grouped[key].append(value)
    return grouped


def summary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = values_by_group(rows)
    out = []
    for key in sorted(grouped):
        values = grouped[key]
        out.append(
            {
                "method": key[0],
                "standardisation": key[1],
                "N_sims": key[2],
                "count": len(values),
                "median": median(values),
            }
        )
    return out


def oracle_floor(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("n") == str(N_OBS):
                return finite_float(row.get("K_theta_star_median"))
    return None


def write_outcome(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    aggregate_path: Path,
    staged_path: Path,
    oracle_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = summary_table(rows)
    floor = oracle_floor(oracle_path)
    missing = [row for row in rows if row["status"] in {"missing", "metrics_read_error", "nonfinite_metric"}]
    finite_count = sum(1 for row in rows if finite_float(row.get("theta_kl")) is not None)

    lines = [
        "# GNK NPE-gap diagnostic outcome",
        "",
        f"Aggregated CSV: `{aggregate_path}`",
        f"Report-staged CSV: `{staged_path}`",
        "",
        "One-line report-builder pointer: read `notebooks/coauthor_report_2026_05_31/data/gnk_gaussian_robust_n1000.csv` for the Section 3.2 diagnostic table.",
        "",
        "## KL summary",
        "",
        "| method | standardisation | N_sims | finite cells | median KL |",
        "|---|---|---:|---:|---:|",
    ]
    for rec in table:
        lines.append(
            "| {method} | {standardisation} | {N_sims} | {count} | {median:.4g} |".format(
                **rec
            )
        )
    if floor is not None:
        lines.extend(["", f"Moment-matched Gaussian oracle floor at n=1000: {floor:.4g}."])
    else:
        lines.extend(["", "Moment-matched Gaussian oracle floor at n=1000: not found."])

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if finite_count == 0:
        lines.append(
            "No diagnostic cells have finite metrics yet. The interpretation is pending PBS completion."
        )
    else:
        lines.append(
            "Interpretation should be updated after all submitted cells finish. Current finite rows are partial and should not be treated as the final verdict."
        )

    lines.extend(
        [
            "",
            "## Failure summary",
            "",
            f"Rows without finite metrics: {len(missing)}.",
        ]
    )
    if missing:
        by_status: dict[str, int] = defaultdict(int)
        for row in missing:
            by_status[str(row["status"])] += 1
        for status, count in sorted(by_status.items()):
            lines.append(f"- {status}: {count}")
    else:
        lines.append("- None.")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("res") / "overnight_20260601" / "npe_gap",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("notebooks")
        / "plots"
        / "gnk_task2_20260526_v3"
        / "raw_theta_kl_paired_per_seed.csv",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("notebooks")
        / "coauthor_report_2026_05_31"
        / "data"
        / "gnk_theta_oracle_by_n.csv",
    )
    parser.add_argument(
        "--staged",
        type=Path,
        default=Path("notebooks")
        / "coauthor_report_2026_05_31"
        / "data"
        / "gnk_gaussian_robust_n1000.csv",
    )
    parser.add_argument(
        "--outcome",
        type=Path,
        default=Path("docs")
        / "coauthor_report_2026_05_31"
        / "codex_outcome_3p2_npe_gap_2026_06_01.md",
    )
    parser.add_argument("--include-flow-1e7", action="store_true")
    args = parser.parse_args()

    aggregate_path = args.root / "gnk_gaussian_robust_n1000" / "aggregated.csv"
    cells_root = args.root / "gnk_gaussian_robust_n1000" / "cells"

    rows = expected_rows(cells_root, args.include_flow_1e7)
    rows.extend(baseline_rows(args.baseline))
    rows.sort(key=lambda row: (row["method"], row["standardisation"], int(row["N_sims"]), int(row["seed"])))

    write_csv(aggregate_path, rows)
    write_csv(args.staged, rows)
    write_outcome(
        args.outcome,
        rows=rows,
        aggregate_path=aggregate_path,
        staged_path=args.staged,
        oracle_path=args.oracle,
    )
    print(f"wrote {aggregate_path}")
    print(f"wrote {args.staged}")
    print(f"wrote {args.outcome}")


if __name__ == "__main__":
    main()
