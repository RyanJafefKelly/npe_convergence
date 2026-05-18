"""Audit GNK hexadecile Gaussian-NPE post-run outputs.

Writes a cell inventory, NUTS divergence diagnostics parsed from PBS stdout
logs, and a compact JSON summary for the May 2026 empirical refresh.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "res" / "gnk_hexadeciles_gaussian"
DEFAULT_AUDIT_DIR = (
    REPO_ROOT
    / "docs"
    / "weekend_2026_05_02"
    / "gnk_hexadecile_postrun_audit_20260504"
)
REQUIRED_FILES = (
    "config.json",
    "metadata.json",
    "x_obs.npy",
    "standardization.npz",
    "gaussian_npe_native_u_posterior.npz",
    "posterior_samples.pkl",
    "true_posterior_samples.pkl",
    "kl.txt",
    "mmd.txt",
    "estimated_coverage.npy",
    "biases.npy",
    "losses.csv",
)
CELL_RE = re.compile(r"gaussian_npe_n_obs_(?P<n_obs>\d+)_n_sims_(?P<n_sims>\d+)_seed_(?P<seed>\d+)$")
NUTS_RE = re.compile(r"nuts_cache_v1_n_obs_(?P<n_obs>\d+)_seed_(?P<seed>\d+)\.pkl")
DIVERGENCE_RE = re.compile(r"Number of divergences:\s+(?P<divergences>\d+)")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_float(path: Path) -> float:
    try:
        return float(path.read_text().strip())
    except Exception:
        return math.nan


def shape_text(path: Path) -> str:
    try:
        return "x".join(str(part) for part in np.load(path).shape)
    except Exception:
        return ""


def parse_cell(path: Path) -> dict[str, object] | None:
    match = CELL_RE.match(path.name)
    if not match:
        return None

    n_obs = int(match.group("n_obs"))
    n_sims = int(match.group("n_sims"))
    seed = int(match.group("seed"))
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    kl = read_float(path / "kl.txt") if (path / "kl.txt").exists() else math.nan
    mmd = read_float(path / "mmd.txt") if (path / "mmd.txt").exists() else math.nan
    coverage_shape = shape_text(path / "estimated_coverage.npy")
    biases_shape = shape_text(path / "biases.npy")
    coverage_shape_ok = coverage_shape == "4x3"
    biases_shape_ok = biases_shape == "400"
    return {
        "seed": seed,
        "n_obs": n_obs,
        "n_sims": n_sims,
        "output_dir": rel(path),
        "missing_required": ";".join(missing),
        "complete": not missing,
        "kl": kl,
        "kl_finite": math.isfinite(kl),
        "mmd": mmd,
        "mmd_finite": math.isfinite(mmd),
        "coverage_shape": coverage_shape,
        "coverage_shape_ok": coverage_shape_ok,
        "biases_shape": biases_shape,
        "biases_shape_ok": biases_shape_ok,
        "flagged": bool(
            missing
            or not math.isfinite(kl)
            or not math.isfinite(mmd)
            or not coverage_shape_ok
            or not biases_shape_ok
        ),
    }


def audit_cells(output_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(output_root.iterdir()):
        if not path.is_dir():
            continue
        row = parse_cell(path)
        if row is not None:
            rows.append(row)
    return sorted(rows, key=lambda r: (int(r["n_obs"]), int(r["n_sims"]), int(r["seed"])))


def parse_nuts_logs(log_dir: Path, pattern: str) -> list[dict[str, int | str]]:
    rows = []
    for path in sorted(log_dir.glob(pattern)):
        current: tuple[int, int] | None = None
        for line in path.read_text(errors="replace").splitlines():
            nuts = NUTS_RE.search(line)
            if nuts:
                current = (int(nuts.group("seed")), int(nuts.group("n_obs")))
                continue
            div = DIVERGENCE_RE.search(line)
            if div and current is not None:
                seed, n_obs = current
                rows.append(
                    {
                        "seed": seed,
                        "n_obs": n_obs,
                        "divergences": int(div.group("divergences")),
                        "log_file": rel(path),
                    }
                )
                current = None
    return sorted(rows, key=lambda r: (int(r["n_obs"]), int(r["seed"])))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(cells: list[dict[str, object]], nuts: list[dict[str, int | str]]) -> dict[str, object]:
    flagged_cells = [row for row in cells if row["flagged"]]
    nonfinite_kl = [row for row in cells if not row["kl_finite"]]
    nonfinite_mmd = [row for row in cells if not row["mmd_finite"]]
    divergent_nuts = [row for row in nuts if int(row["divergences"]) > 0]
    return {
        "total_cells": len(cells),
        "complete_cells": sum(1 for row in cells if row["complete"]),
        "finite_kl_cells": sum(1 for row in cells if row["kl_finite"]),
        "nonfinite_kl_cells": len(nonfinite_kl),
        "finite_mmd_cells": sum(1 for row in cells if row["mmd_finite"]),
        "nonfinite_mmd_cells": len(nonfinite_mmd),
        "coverage_shape_ok_cells": sum(1 for row in cells if row["coverage_shape_ok"]),
        "biases_shape_ok_cells": sum(1 for row in cells if row["biases_shape_ok"]),
        "flagged_cell_count": len(flagged_cells),
        "nonfinite_kl_cells_detail": [
            {
                "seed": row["seed"],
                "n_obs": row["n_obs"],
                "n_sims": row["n_sims"],
                "output_dir": row["output_dir"],
            }
            for row in nonfinite_kl
        ],
        "nonfinite_mmd_cells_detail": [
            {
                "seed": row["seed"],
                "n_obs": row["n_obs"],
                "n_sims": row["n_sims"],
                "output_dir": row["output_dir"],
            }
            for row in nonfinite_mmd
        ],
        "nuts_diagnostic_rows": len(nuts),
        "nuts_rows_with_divergences": len(divergent_nuts),
        "nuts_divergence_detail": divergent_nuts,
        "max_nuts_divergences": max((int(row["divergences"]) for row in nuts), default=0),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit GNK hexadecile Gaussian-NPE outputs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--log-pattern", type=str, default="gnk_hexadecile_gaussian.o20724351.*")
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    cells = audit_cells(args.output_root)
    nuts = parse_nuts_logs(args.log_dir, args.log_pattern)
    summary = build_summary(cells, nuts)

    write_csv(
        args.audit_dir / "cell_inventory.csv",
        cells,
        [
            "seed",
            "n_obs",
            "n_sims",
            "output_dir",
            "missing_required",
            "complete",
            "kl",
            "kl_finite",
            "mmd",
            "mmd_finite",
            "coverage_shape",
            "coverage_shape_ok",
            "biases_shape",
            "biases_shape_ok",
            "flagged",
        ],
    )
    write_csv(
        args.audit_dir / "nuts_diagnostics.csv",
        nuts,
        ["seed", "n_obs", "divergences", "log_file"],
    )
    with (args.audit_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
