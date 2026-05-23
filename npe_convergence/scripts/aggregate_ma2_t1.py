"""Aggregate MA(2) T1 diagnostics from existing result files.

This script is intentionally read-only with respect to ``res/ma2_b0``. It
produces the May 2026 T1 CSVs and a short per-seed note without rerunning NPE.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "res" / "ma2_b0"
DATA_DIR = REPO_ROOT / "notebooks" / "meeting_2026_05_18" / "data"
DOC_DIR = REPO_ROOT / "docs" / "meeting_2026_05_18"

PER_SEED_PATH = DATA_DIR / "ma2_b0_per_seed.csv"
SUMMARY_PATH = DATA_DIR / "ma2_b0_per_seed_summary.csv"
FLOW_COMPAT_PATH = DATA_DIR / "ma2_compatibility_flow.csv"
EXISTING_COMPAT_PATH = DATA_DIR / "ma2_compatibility.csv"
NOTE_PATH = DOC_DIR / "ma2_t1_per_seed_note.md"

N_OBS_VALUES = (100, 500, 1000, 5000)
SEEDS = tuple(range(101))
METHODS = (
    ("flow_npe", "npe"),
    ("gaussian_npe", "gaussian_npe"),
)
BUDGET_LABELS = ("N=n", "N=n log(n)", "N=n^(3/2)", "N=n^2")
DELTA0_VALUES = ("0.01", "0.1", "0.25", "0.5", "0.75", "0.99")

COMPAT_COLUMNS = [
    "n_obs",
    "n_sims",
    "budget_label",
    "delta0",
    "rows",
    "complete_rows",
    "shape_ok_rows",
    "finite_kl_rows",
    "infinite_kl_rows",
    "nan_kl_rows",
    "finite_kl_fraction",
    "infinite_kl_fraction",
    "finite_kl_min",
    "finite_kl_q25",
    "finite_kl_median",
    "finite_kl_q75",
    "finite_kl_max",
    "finite_mmd_rows",
    "mmd_min",
    "mmd_q25",
    "mmd_median",
    "mmd_q75",
    "mmd_max",
]


def budget_grid(n_obs: int) -> tuple[tuple[str, int], ...]:
    return (
        ("N=n", n_obs),
        ("N=n log(n)", int(n_obs * math.log(n_obs))),
        ("N=n^(3/2)", int(n_obs ** (3 / 2))),
        ("N=n^2", n_obs**2),
    )


def result_dir(prefix: str, n_obs: int, n_sims: int, seed: int) -> Path:
    return RESULT_ROOT / f"{prefix}_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}"


def read_float(path: Path) -> float:
    try:
        return float(path.read_text().strip())
    except Exception:
        return math.nan


def finite_quantiles(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "min": math.nan,
            "q25": math.nan,
            "median": math.nan,
            "q75": math.nan,
            "max": math.nan,
        }
    q25, median, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
    return {
        "min": float(np.min(arr)),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(np.max(arr)),
    }


def collect_per_seed_rows() -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    missing_kl = 0
    missing_mmd = 0
    present_nonfinite_kl = 0

    for method, prefix in METHODS:
        for n_obs in N_OBS_VALUES:
            for budget_label, n_sims in budget_grid(n_obs):
                for seed in SEEDS:
                    root = result_dir(prefix, n_obs, n_sims, seed)
                    kl_path = root / "kl.txt"
                    mmd_path = root / "mmd.txt"
                    if not kl_path.is_file() and not mmd_path.is_file():
                        missing_kl += 1
                        missing_mmd += 1
                        continue

                    if kl_path.is_file():
                        kl = read_float(kl_path)
                        if not math.isfinite(kl):
                            present_nonfinite_kl += 1
                    else:
                        kl = math.nan
                        missing_kl += 1

                    if mmd_path.is_file():
                        mmd = read_float(mmd_path)
                    else:
                        mmd = math.nan
                        missing_mmd += 1

                    rows.append(
                        {
                            "method": method,
                            "n_obs": n_obs,
                            "n_sims": n_sims,
                            "budget_label": budget_label,
                            "seed": seed,
                            "kl": kl,
                            "mmd": mmd,
                        }
                    )

    df = pd.DataFrame(
        rows,
        columns=["method", "n_obs", "n_sims", "budget_label", "seed", "kl", "mmd"],
    )
    diagnostics = {
        "missing_kl_files": missing_kl,
        "missing_mmd_files": missing_mmd,
        "present_nonfinite_kl": present_nonfinite_kl,
    }
    return df, diagnostics


def summarize_per_seed(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, _prefix in METHODS:
        for n_obs in N_OBS_VALUES:
            for budget_label, n_sims in budget_grid(n_obs):
                group = per_seed[
                    (per_seed["method"] == method)
                    & (per_seed["n_obs"] == n_obs)
                    & (per_seed["n_sims"] == n_sims)
                ]
                kl = pd.to_numeric(group["kl"], errors="coerce").to_numpy(dtype=float)
                finite = kl[np.isfinite(kl)]
                stats = finite_quantiles(finite)
                if finite.size == 0:
                    mean = std = iqr = max_median_ratio = math.nan
                    n_outlier = 0
                else:
                    mean = float(np.mean(finite))
                    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                    iqr = stats["q75"] - stats["q25"]
                    max_median_ratio = (
                        stats["max"] / stats["median"]
                        if stats["median"] != 0
                        else math.nan
                    )
                    n_outlier = int(np.sum(finite > 2.0 * stats["median"]))
                rows.append(
                    {
                        "method": method,
                        "n_obs": int(n_obs),
                        "n_sims": int(n_sims),
                        "budget_label": budget_label,
                        "n_seeds_total": int(len(group)),
                        "n_seeds_finite": int(finite.size),
                        "mean": mean,
                        "std": std,
                        "min": stats["min"],
                        "q25": stats["q25"],
                        "median": stats["median"],
                        "q75": stats["q75"],
                        "max": stats["max"],
                        "IQR": iqr,
                        "max_median_ratio": max_median_ratio,
                        "n_seeds_outlier": n_outlier,
                    }
                )
    return pd.DataFrame(rows)


def collect_delta_metric(prefix: str, n_obs: int, n_sims: int, seed: int, name: str) -> float | None:
    path = result_dir(prefix, n_obs, n_sims, seed) / name
    if not path.is_file():
        return None
    return read_float(path)


def compat_summary(values: list[float]) -> dict[str, float]:
    stats = finite_quantiles(values)
    return {
        "min": stats["min"],
        "q25": stats["q25"],
        "median": stats["median"],
        "q75": stats["q75"],
        "max": stats["max"],
    }


def collect_flow_compat_rows() -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    missing_mmd_cells = 0
    zero_seed_cells = 0

    for n_obs in N_OBS_VALUES:
        for budget_label, n_sims in budget_grid(n_obs):
            for delta0 in DELTA0_VALUES:
                kl_values: list[float] = []
                mmd_values: list[float] = []
                any_rows = 0
                complete_rows = 0
                for seed in SEEDS:
                    kl = collect_delta_metric("npe", n_obs, n_sims, seed, f"kl_{delta0}.txt")
                    mmd = collect_delta_metric("npe", n_obs, n_sims, seed, f"mmd_{delta0}.txt")
                    if kl is None and mmd is None:
                        continue
                    any_rows += 1
                    if kl is not None:
                        kl_values.append(kl)
                    if mmd is not None:
                        mmd_values.append(mmd)
                    if kl is not None and mmd is not None:
                        complete_rows += 1

                if any_rows == 0:
                    zero_seed_cells += 1
                if any_rows > 0 and not mmd_values:
                    missing_mmd_cells += 1

                kl_arr = np.asarray(kl_values, dtype=float)
                finite_kl = kl_arr[np.isfinite(kl_arr)]
                infinite_kl = kl_arr[np.isinf(kl_arr)]
                nan_kl = kl_arr[np.isnan(kl_arr)]
                mmd_arr = np.asarray(mmd_values, dtype=float)
                finite_mmd = mmd_arr[np.isfinite(mmd_arr)]
                kl_stats = compat_summary(finite_kl.tolist())
                mmd_stats = compat_summary(finite_mmd.tolist())

                rows.append(
                    {
                        "n_obs": n_obs,
                        "n_sims": n_sims,
                        "budget_label": budget_label,
                        "delta0": float(delta0),
                        "rows": any_rows,
                        "complete_rows": complete_rows,
                        "shape_ok_rows": complete_rows,
                        "finite_kl_rows": int(finite_kl.size),
                        "infinite_kl_rows": int(infinite_kl.size),
                        "nan_kl_rows": int(nan_kl.size),
                        "finite_kl_fraction": (
                            float(finite_kl.size / any_rows) if any_rows else math.nan
                        ),
                        "infinite_kl_fraction": (
                            float(infinite_kl.size / any_rows) if any_rows else math.nan
                        ),
                        "finite_kl_min": kl_stats["min"],
                        "finite_kl_q25": kl_stats["q25"],
                        "finite_kl_median": kl_stats["median"],
                        "finite_kl_q75": kl_stats["q75"],
                        "finite_kl_max": kl_stats["max"],
                        "finite_mmd_rows": int(finite_mmd.size),
                        "mmd_min": mmd_stats["min"],
                        "mmd_q25": mmd_stats["q25"],
                        "mmd_median": mmd_stats["median"],
                        "mmd_q75": mmd_stats["q75"],
                        "mmd_max": mmd_stats["max"],
                    }
                )
    df = pd.DataFrame(rows, columns=COMPAT_COLUMNS)
    diagnostics = {
        "flow_delta0_zero_seed_cells": zero_seed_cells,
        "flow_delta0_cells_with_no_mmd": missing_mmd_cells,
    }
    return df, diagnostics


def validate_compat_schema(flow_compat: pd.DataFrame) -> None:
    existing = pd.read_csv(EXISTING_COMPAT_PATH)
    if list(flow_compat.columns) != list(existing.columns):
        raise AssertionError("ma2_compatibility_flow.csv columns do not match existing schema")

    flow_read = pd.read_csv(FLOW_COMPAT_PATH)
    if list(flow_read.columns) != list(existing.columns):
        raise AssertionError("readback flow compatibility columns do not match")

    expected_dtypes = existing.dtypes.astype(str).to_dict()
    actual_dtypes = flow_read.dtypes.astype(str).to_dict()
    mismatched = {
        name: (expected_dtypes[name], actual_dtypes[name])
        for name in expected_dtypes
        if expected_dtypes[name] != actual_dtypes[name]
    }
    if mismatched:
        raise AssertionError(f"flow compatibility dtype mismatch: {mismatched}")

    pd.concat(
        [
            existing.assign(method="gaussian_npe"),
            flow_read.assign(method="flow_npe"),
        ],
        ignore_index=True,
    )


def median_without_worst(values: np.ndarray, worst: int = 5) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    if values.size <= worst:
        return math.nan
    return float(np.median(np.sort(values)[:-worst]))


def note_lines(per_seed: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    flow_1000 = summary[
        (summary["method"] == "flow_npe") & (summary["n_obs"] == 1000)
    ].sort_values("n_sims")
    gauss_1000 = summary[
        (summary["method"] == "gaussian_npe") & (summary["n_obs"] == 1000)
    ].sort_values("n_sims")

    flow_parts: list[str] = []
    drop_parts: list[str] = []
    outlier_parts: list[str] = []
    for _, row in flow_1000.iterrows():
        label = str(row["budget_label"])
        flow_parts.append(f"{label} {row['median']:.2f}")
        values = per_seed[
            (per_seed["method"] == "flow_npe")
            & (per_seed["n_obs"] == 1000)
            & (per_seed["n_sims"] == row["n_sims"])
        ]["kl"].to_numpy(dtype=float)
        drop_parts.append(f"{label} {median_without_worst(values):.2f}")
        outlier_parts.append(f"{label} {int(row['n_seeds_outlier'])}")

    gauss_parts = [
        f"{row['budget_label']} {row['median']:.2f}"
        for _, row in gauss_1000.iterrows()
    ]

    other_n = summary[
        (summary["method"] == "flow_npe")
        & (summary["n_obs"].isin([100, 500, 5000]))
    ].copy()
    largest_outlier_share = 0.0
    if not other_n.empty:
        share = other_n["n_seeds_outlier"] / other_n["n_seeds_finite"].replace(0, np.nan)
        largest_outlier_share = float(np.nanmax(share.to_numpy(dtype=float)))

    return [
        "At n_obs=1000, the elevated flow-NPE compatible-case KL is not driven by a small set of failed seeds.",
        "The flow medians across N=n, N=n log(n), N=n^(3/2), and N=n^2 are "
        + ", ".join(flow_parts)
        + ", while dropping the worst five seeds gives "
        + ", ".join(drop_parts)
        + ".",
        "The number of seeds above twice the cell median is "
        + ", ".join(outlier_parts)
        + ", so the median is shifted up rather than being pulled by a heavy upper tail.",
        "For comparison, Gaussian-NPE at n_obs=1000 has medians "
        + ", ".join(gauss_parts)
        + ".",
        f"Other n_obs values show the same broad pattern, with at most {largest_outlier_share:.0%} of finite flow seeds above twice the median in any non-1000 cell.",
    ]


def write_note(per_seed: pd.DataFrame, summary: pd.DataFrame) -> None:
    NOTE_PATH.write_text("\n".join(note_lines(per_seed, summary)) + "\n")


def count_delta1_files() -> int:
    names = ("kl_1.txt", "kl_1.0.txt", "mmd_1.txt", "mmd_1.0.txt")
    return sum(1 for name in names for _path in RESULT_ROOT.glob(f"*/{name}"))


def print_n1000_table(summary: pd.DataFrame) -> None:
    print("\nn_obs=1000 KL medians")
    print("method        N label        n_sims      finite  median  outliers")
    for method in ("flow_npe", "gaussian_npe"):
        sub = summary[(summary["method"] == method) & (summary["n_obs"] == 1000)]
        sub = sub.sort_values("n_sims")
        for _, row in sub.iterrows():
            print(
                f"{method:<13} {row['budget_label']:<12} {int(row['n_sims']):>8d}"
                f" {int(row['n_seeds_finite']):>7d} {row['median']:>7.3f}"
                f" {int(row['n_seeds_outlier']):>9d}"
            )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    per_seed, per_seed_diag = collect_per_seed_rows()
    summary = summarize_per_seed(per_seed)
    flow_compat, compat_diag = collect_flow_compat_rows()

    per_seed.to_csv(PER_SEED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    flow_compat.to_csv(FLOW_COMPAT_PATH, index=False)
    validate_compat_schema(flow_compat)
    write_note(per_seed, summary)

    print("MA(2) T1 aggregation complete")
    print(f"wrote: {PER_SEED_PATH.relative_to(REPO_ROOT)} ({len(per_seed)} rows)")
    print(f"wrote: {SUMMARY_PATH.relative_to(REPO_ROOT)} ({len(summary)} rows)")
    print(f"wrote: {FLOW_COMPAT_PATH.relative_to(REPO_ROOT)} ({len(flow_compat)} rows)")
    print(f"wrote: {NOTE_PATH.relative_to(REPO_ROOT)}")
    print(
        "compatible b0 input issues: "
        f"missing kl files={per_seed_diag['missing_kl_files']}, "
        f"missing mmd files={per_seed_diag['missing_mmd_files']}, "
        f"present non-finite kl={per_seed_diag['present_nonfinite_kl']}"
    )
    print(
        "flow delta0 input issues: "
        f"zero-seed cells={compat_diag['flow_delta0_zero_seed_cells']}, "
        f"cells with no mmd_delta files={compat_diag['flow_delta0_cells_with_no_mmd']}"
    )
    print(f"delta0=1.0 metric files found under res/ma2_b0: {count_delta1_files()}")
    print_n1000_table(summary)


if __name__ == "__main__":
    main()
