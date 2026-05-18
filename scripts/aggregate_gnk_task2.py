"""Aggregate GNK octile flow/Gaussian/oracle caches for Task 2.

The script is intentionally cache read-only. It writes co-author/debug CSVs and
short notes into a fresh output directory and refuses to overwrite that
directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PARAMS = ("A", "B", "g", "k")
COVERAGE_LEVELS = (0.8, 0.9, 0.95)
STANDARD_N = (100, 500, 1000, 5000)
EXPECTED_SEEDS = tuple(range(101))
DIR_RE = re.compile(
    r"^(?P<prefix>gaussian_npe|npe)_n_obs_(?P<n>\d+)_n_sims_"
    r"(?P<N>\d+)_seed_(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class CacheRecord:
    method: str
    prefix: str
    n: int
    N: int
    seed: int
    path: Path
    has_kl: bool
    has_coverage: bool
    has_biases: bool
    has_posterior_samples: bool

    @property
    def kl_path(self) -> Path:
        return self.path / "kl.txt"

    @property
    def coverage_path(self) -> Path:
        return self.path / "estimated_coverage.npy"

    @property
    def biases_path(self) -> Path:
        return self.path / "biases.npy"


def standard_n_sims(n: int) -> dict[str, int]:
    return {
        "N=n": n,
        "N=n log(n)": int(n * math.log(n)),
        "N=n^(3/2)": int(n ** 1.5),
        "N=n^2": n**2,
    }


def n_sims_label(n: int, N: int) -> str:
    for label, value in standard_n_sims(n).items():
        if N == value:
            return label
    return "other"


def is_standard_paper_grid(n: int, N: int) -> bool:
    return n in STANDARD_N and n_sims_label(n, N) != "other"


def is_legacy_gaussian_one_row(method: str, n: int, N: int) -> bool:
    return method == "gaussian_npe" and n == 1000 and N == 31623


def method_from_prefix(prefix: str) -> str:
    if prefix == "npe":
        return "flow_npe"
    if prefix == "gaussian_npe":
        return "gaussian_npe"
    raise ValueError(f"unknown prefix: {prefix}")


def fmt_seed_list(seeds: Iterable[int]) -> str:
    values = sorted(set(seeds))
    if not values:
        return ""
    ranges: list[str] = []
    start = prev = values[0]
    for seed in values[1:]:
        if seed == prev + 1:
            prev = seed
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = seed
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def finite_or_nan(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def quantiles(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    keys = ("q05", "q25", "median", "q75", "q95")
    if values.size == 0:
        return {f"{prefix}{key}": math.nan for key in keys}
    qs = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {f"{prefix}{key}": float(value) for key, value in zip(keys, qs)}


def summary_stats(values: Iterable[float], prefix: str = "") -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    base = {
        f"{prefix}count": int(arr.size),
        f"{prefix}mean": math.nan,
        f"{prefix}sd": math.nan,
        f"{prefix}min": math.nan,
        f"{prefix}max": math.nan,
    }
    base.update(quantiles(arr, prefix=prefix))
    if arr.size == 0:
        return base
    base[f"{prefix}mean"] = float(np.mean(arr))
    base[f"{prefix}sd"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    base[f"{prefix}min"] = float(np.min(arr))
    base[f"{prefix}max"] = float(np.max(arr))
    return base


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_cache_records(cache_root: Path) -> list[CacheRecord]:
    records: list[CacheRecord] = []
    for path in cache_root.iterdir():
        if not path.is_dir():
            continue
        match = DIR_RE.match(path.name)
        if match is None:
            continue
        prefix = match.group("prefix")
        records.append(
            CacheRecord(
                method=method_from_prefix(prefix),
                prefix=prefix,
                n=int(match.group("n")),
                N=int(match.group("N")),
                seed=int(match.group("seed")),
                path=path,
                has_kl=(path / "kl.txt").is_file(),
                has_coverage=(path / "estimated_coverage.npy").is_file(),
                has_biases=(path / "biases.npy").is_file(),
                has_posterior_samples=(path / "posterior_samples.pkl").is_file(),
            )
        )
    return sorted(records, key=lambda r: (r.method, r.n, r.N, r.seed))


def group_records(records: Iterable[CacheRecord]) -> dict[tuple[str, int, int], list[CacheRecord]]:
    grouped: dict[tuple[str, int, int], list[CacheRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.method, record.n, record.N)].append(record)
    return dict(grouped)


def load_kl(record: CacheRecord) -> float:
    if not record.has_kl:
        return math.nan
    return finite_or_nan(record.kl_path.read_text().strip())


def load_coverage(record: CacheRecord) -> np.ndarray:
    cov = np.asarray(np.load(record.coverage_path), dtype=float)
    if cov.shape != (len(PARAMS), len(COVERAGE_LEVELS)):
        raise ValueError(f"unexpected coverage shape {cov.shape}: {record.coverage_path}")
    return cov


def load_biases(record: CacheRecord) -> np.ndarray:
    biases = np.asarray(np.load(record.biases_path), dtype=float).reshape(-1, len(PARAMS))
    if biases.shape[1] != len(PARAMS):
        raise ValueError(f"unexpected bias shape {biases.shape}: {record.biases_path}")
    return biases


def inventory_rows(
    grouped: dict[tuple[str, int, int], list[CacheRecord]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected = set(EXPECTED_SEEDS)
    for (method, n, N), records in sorted(grouped.items()):
        seeds = {record.seed for record in records}
        kl_seeds = {record.seed for record in records if record.has_kl}
        coverage_seeds = {record.seed for record in records if record.has_coverage}
        bias_seeds = {record.seed for record in records if record.has_biases}
        posterior_seeds = {record.seed for record in records if record.has_posterior_samples}
        complete_kl_cov_bias = kl_seeds & coverage_seeds & bias_seeds
        legacy_excluded = is_legacy_gaussian_one_row(method, n, N)
        rows.append(
            {
                "method": method,
                "n": n,
                "N": N,
                "N_label": n_sims_label(n, N),
                "standard_paper_grid": int(is_standard_paper_grid(n, N)),
                "legacy_one_row_excluded": int(legacy_excluded),
                "dir_count": len(records),
                "seed_count": len(seeds),
                "min_seed": min(seeds),
                "max_seed": max(seeds),
                "seeds": fmt_seed_list(seeds),
                "missing_dir_seeds_0_100": fmt_seed_list(expected - seeds),
                "kl_seed_count": len(kl_seeds),
                "coverage_seed_count": len(coverage_seeds),
                "bias_seed_count": len(bias_seeds),
                "posterior_seed_count": len(posterior_seeds),
                "complete_kl_coverage_bias_seed_count": len(complete_kl_cov_bias),
                "complete_101_kl_coverage_bias": int(
                    len(complete_kl_cov_bias) == len(EXPECTED_SEEDS)
                ),
                "include_in_complete_group_summaries": int(
                    not legacy_excluded and len(complete_kl_cov_bias) == len(EXPECTED_SEEDS)
                ),
                "missing_kl_seeds_observed_dirs": fmt_seed_list(seeds - kl_seeds),
                "missing_coverage_seeds_observed_dirs": fmt_seed_list(seeds - coverage_seeds),
                "missing_bias_seeds_observed_dirs": fmt_seed_list(seeds - bias_seeds),
                "missing_posterior_seeds_observed_dirs": fmt_seed_list(
                    seeds - posterior_seeds
                ),
            }
        )
    return rows


def raw_theta_kl_tables(
    records: list[CacheRecord],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    kl_by_key: dict[tuple[str, int, int, int], float] = {}
    for record in records:
        if is_legacy_gaussian_one_row(record.method, record.n, record.N):
            continue
        kl = load_kl(record)
        if math.isfinite(kl):
            kl_by_key[(record.method, record.n, record.N, record.seed)] = kl

    pair_rows: list[dict[str, object]] = []
    pair_groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    method_rows: list[dict[str, object]] = []
    method_groups: dict[tuple[str, int, int], list[float]] = defaultdict(list)

    for (method, n, N, seed), kl in kl_by_key.items():
        method_groups[(method, n, N)].append(kl)

    comparable_seed_keys = {
        (n, N, seed)
        for method, n, N, seed in kl_by_key
        if ("flow_npe", n, N, seed) in kl_by_key
        and ("gaussian_npe", n, N, seed) in kl_by_key
    }
    for n, N, seed in sorted(comparable_seed_keys):
        row = {
            "n": n,
            "N": N,
            "N_label": n_sims_label(n, N),
            "standard_paper_grid": int(is_standard_paper_grid(n, N)),
            "seed": seed,
            "flow_theta_kl": kl_by_key[("flow_npe", n, N, seed)],
            "gaussian_theta_kl": kl_by_key[("gaussian_npe", n, N, seed)],
        }
        pair_rows.append(row)
        pair_groups[(n, N)].append(row)

    pair_summary_rows: list[dict[str, object]] = []
    for (n, N), rows in sorted(pair_groups.items()):
        flow_values = [float(row["flow_theta_kl"]) for row in rows]
        gaussian_values = [float(row["gaussian_theta_kl"]) for row in rows]
        pair_summary_rows.append(
            {
                "n": n,
                "N": N,
                "N_label": n_sims_label(n, N),
                "standard_paper_grid": int(is_standard_paper_grid(n, N)),
                "paired_seed_count": len(rows),
                "complete_101_paired": int(len(rows) == len(EXPECTED_SEEDS)),
                **summary_stats(flow_values, prefix="flow_theta_kl_"),
                **summary_stats(gaussian_values, prefix="gaussian_theta_kl_"),
            }
        )

    for (method, n, N), values in sorted(method_groups.items()):
        method_rows.append(
            {
                "method": method,
                "n": n,
                "N": N,
                "N_label": n_sims_label(n, N),
                "standard_paper_grid": int(is_standard_paper_grid(n, N)),
                "kl_seed_count": len(values),
                "complete_101_kl": int(len(values) == len(EXPECTED_SEEDS)),
                **summary_stats(values, prefix="theta_kl_"),
            }
        )

    return pair_rows, pair_summary_rows, method_rows


def theta_oracle_tables(
    u_space_csv: Path,
    n500_gate_csv: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    df = pd.read_csv(u_space_csv)
    needed = {"n", "seed", "K_theta_oracle"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{u_space_csv} missing columns: {sorted(missing)}")

    distinct = df.sort_values(["n", "seed", "N"]).drop_duplicates(["n", "seed"])
    oracle_rows: list[dict[str, object]] = []
    for n, group in distinct.groupby("n", sort=True):
        values = group["K_theta_oracle"].to_numpy(dtype=float)
        oracle_rows.append(
            {
                "n": int(n),
                "source": str(u_space_csv),
                "seed_count": int(group["seed"].nunique()),
                "K_theta_star_count": int(np.isfinite(values).sum()),
                **summary_stats(values, prefix="K_theta_star_"),
            }
        )

    gate_rows: list[dict[str, object]] = []
    if n500_gate_csv.is_file():
        gate = pd.read_csv(n500_gate_csv)
        merged = (
            distinct[distinct["n"] == 500][["seed", "K_theta_oracle"]]
            .merge(gate[["seed", "k_theta_star"]], on="seed", how="inner")
        )
        gate_rows.append(
            {
                "n": 500,
                "u_space_source": str(u_space_csv),
                "gate_source": str(n500_gate_csv),
                "matched_seed_count": int(len(merged)),
                "max_abs_K_theta_star_difference": float(
                    np.max(
                        np.abs(
                            merged["K_theta_oracle"].to_numpy(dtype=float)
                            - merged["k_theta_star"].to_numpy(dtype=float)
                        )
                    )
                )
                if len(merged)
                else math.nan,
                "u_space_K_theta_star_median": float(
                    np.median(merged["K_theta_oracle"])
                )
                if len(merged)
                else math.nan,
                "gate_K_theta_star_median": float(np.median(merged["k_theta_star"]))
                if len(merged)
                else math.nan,
            }
        )
    return oracle_rows, gate_rows


def u_space_delta_tables(
    u_space_csv: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    df = pd.read_csv(u_space_csv)
    required = {
        "n",
        "N",
        "seed",
        "scaled_budget",
        "Delta_u_total",
        "Delta_u_mean",
        "Delta_u_cov",
        "self_consistency_kl_u",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{u_space_csv} missing columns: {sorted(missing)}")

    legacy_mask = (
        (df["method"] == "Gaussian-NPE")
        & (df["n"] == 1000)
        & (df["N"] == 31623)
    )
    rows: list[dict[str, object]] = []
    per_seed_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []

    for (n, N), group in df.groupby(["n", "N"], sort=True):
        n_int = int(n)
        N_int = int(N)
        seed_count = int(group["seed"].nunique())
        excluded = bool(legacy_mask.loc[group.index].any() or seed_count < len(EXPECTED_SEEDS))
        row = {
            "n": n_int,
            "N": N_int,
            "N_label": n_sims_label(n_int, N_int),
            "standard_paper_grid": int(is_standard_paper_grid(n_int, N_int)),
            "seed_count": seed_count,
            "excluded_from_complete_delta_summary": int(excluded),
            "exclusion_reason": "legacy_one_row_or_incomplete" if excluded else "",
            "scaled_budget_median": float(np.median(group["scaled_budget"])),
            **summary_stats(group["Delta_u_total"], prefix="Delta_N_u_total_"),
            **summary_stats(group["Delta_u_mean"], prefix="Delta_N_u_mean_term_"),
            **summary_stats(group["Delta_u_cov"], prefix="Delta_N_u_cov_term_"),
            **summary_stats(group["self_consistency_kl_u"], prefix="self_consistency_kl_u_"),
        }
        if excluded:
            excluded_rows.append(row)
        else:
            rows.append(row)
            for _, item in group.sort_values("seed").iterrows():
                per_seed_rows.append(
                    {
                        "n": n_int,
                        "N": N_int,
                        "N_label": n_sims_label(n_int, N_int),
                        "standard_paper_grid": int(is_standard_paper_grid(n_int, N_int)),
                        "seed": int(item["seed"]),
                        "scaled_budget": float(item["scaled_budget"]),
                        "Delta_N_u_total": float(item["Delta_u_total"]),
                        "Delta_N_u_mean_term": float(item["Delta_u_mean"]),
                        "Delta_N_u_cov_term": float(item["Delta_u_cov"]),
                        "self_consistency_kl_u": float(item["self_consistency_kl_u"]),
                    }
                )
    return rows, per_seed_rows, excluded_rows


def coverage_rows(
    records: list[CacheRecord],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []
    grouped = group_records(
        record
        for record in records
        if is_standard_paper_grid(record.n, record.N)
        and not is_legacy_gaussian_one_row(record.method, record.n, record.N)
    )
    by_method_param: dict[tuple[str, str, int], dict[str, str]] = defaultdict(dict)

    for (method, n, N), group_records_ in sorted(grouped.items()):
        coverage_records = [record for record in group_records_ if record.has_coverage]
        if not coverage_records:
            continue
        covs = np.stack([load_coverage(record) for record in coverage_records])
        for param_index, param in enumerate(PARAMS):
            row = {
                "method": method,
                "n": n,
                "N": N,
                "N_label": n_sims_label(n, N),
                "param": param,
                "seed_count": len(coverage_records),
            }
            cell_values: list[float] = []
            for level_index, level in enumerate(COVERAGE_LEVELS):
                level_values = covs[:, param_index, level_index]
                suffix = str(int(level * 100))
                row[f"coverage_{suffix}_mean"] = float(np.mean(level_values))
                row[f"coverage_{suffix}_sd"] = (
                    float(np.std(level_values, ddof=1))
                    if level_values.size > 1
                    else 0.0
                )
                row.update(
                    {
                        f"coverage_{suffix}_{key}": value
                        for key, value in quantiles(level_values).items()
                    }
                )
                cell_values.append(float(np.mean(level_values)))
            rows.append(row)
            by_method_param[(method, param, n)][n_sims_label(n, N)] = (
                f"{cell_values[0]:.2f}/{cell_values[1]:.2f}/{cell_values[2]:.2f}"
            )

    labels = list(standard_n_sims(100).keys())
    for (method, param, n), cells in sorted(by_method_param.items()):
        table_rows.append(
            {
                "method": method,
                "param": param,
                "n": n,
                **{label: cells.get(label, "") for label in labels},
            }
        )
    return rows, table_rows


def bias_rows(
    records: list[CacheRecord],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    seed_rows: list[dict[str, object]] = []
    grouped = group_records(
        record
        for record in records
        if is_standard_paper_grid(record.n, record.N)
        and not is_legacy_gaussian_one_row(record.method, record.n, record.N)
    )

    for (method, n, N), group_records_ in sorted(grouped.items()):
        for record in group_records_:
            if not record.has_biases:
                continue
            biases = load_biases(record)
            for param_index, param in enumerate(PARAMS):
                values = biases[:, param_index]
                row = {
                    "method": method,
                    "n": n,
                    "N": N,
                    "N_label": n_sims_label(n, N),
                    "seed": record.seed,
                    "param": param,
                    "replicate_count": int(values.size),
                    "seed_mean_bias": float(np.mean(values)),
                    "seed_mean_abs_bias": float(np.mean(np.abs(values))),
                    "seed_rmse_bias": float(np.sqrt(np.mean(values**2))),
                }
                row.update(quantiles(values, prefix="replicate_bias_"))
                seed_rows.append(row)

    summary_groups: dict[tuple[str, int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        summary_groups[
            (str(row["method"]), int(row["n"]), int(row["N"]), str(row["param"]))
        ].append(row)

    summary_rows: list[dict[str, object]] = []
    for (method, n, N, param), group in sorted(summary_groups.items()):
        seed_mean_bias = np.asarray([float(row["seed_mean_bias"]) for row in group])
        seed_mean_abs_bias = np.asarray([float(row["seed_mean_abs_bias"]) for row in group])
        row = {
            "method": method,
            "n": n,
            "N": N,
            "N_label": n_sims_label(n, N),
            "param": param,
            "seed_count": len(group),
            "replicate_count_total": int(sum(int(row["replicate_count"]) for row in group)),
            "seed_mean_bias_mean": float(np.mean(seed_mean_bias)),
            "seed_mean_bias_sd": float(np.std(seed_mean_bias, ddof=1))
            if seed_mean_bias.size > 1
            else 0.0,
            "seed_mean_abs_bias_mean": float(np.mean(np.abs(seed_mean_bias))),
            "seed_mean_bias_rmse": float(np.sqrt(np.mean(seed_mean_bias**2))),
            "replicate_mean_abs_bias_mean": float(np.mean(seed_mean_abs_bias)),
        }
        row.update(quantiles(seed_mean_bias, prefix="seed_mean_bias_"))
        summary_rows.append(row)

    g_paper_rows = [
        row
        for row in seed_rows
        if row["param"] == "g" and int(row["n"]) in {1000, 5000}
    ]
    return seed_rows, summary_rows, g_paper_rows


def write_rerun_note(path: Path, inventory: list[dict[str, object]]) -> None:
    def complete_count(method: str) -> int:
        return sum(
            1
            for row in inventory
            if row["method"] == method
            and int(row["standard_paper_grid"]) == 1
            and int(row["complete_101_kl_coverage_bias"]) == 1
            and int(row["legacy_one_row_excluded"]) == 0
        )

    flow_complete = complete_count("flow_npe")
    gaussian_complete = complete_count("gaussian_npe")
    legacy_rows = [
        row
        for row in inventory
        if int(row["legacy_one_row_excluded"]) == 1
    ]
    standard_gaps = [
        row
        for row in inventory
        if int(row["standard_paper_grid"]) == 1
        and int(row["legacy_one_row_excluded"]) == 0
        and int(row["complete_101_kl_coverage_bias"]) == 0
    ]
    nonstandard_gaps = [
        row
        for row in inventory
        if int(row["standard_paper_grid"]) == 0
        and int(row["legacy_one_row_excluded"]) == 0
        and int(row["complete_101_kl_coverage_bias"]) == 0
    ]

    lines = [
        "# GNK Task 2 Octile HPC Rerun Note",
        "",
        "- Standard paper-grid status: "
        f"flow_npe {flow_complete}/16 groups and gaussian_npe {gaussian_complete}/16 "
        "groups have 101 seeds with `kl.txt`, `estimated_coverage.npy`, and "
        "`biases.npy`.",
        "- Conclusion: no broad GNK octile HPC rerun is needed for the Task 2 "
        "co-author/debug aggregation. A targeted flow-NPE recovery/rerun would be "
        "needed only if the paper-style flow cell with a shortfall must be refreshed "
        "at 101 complete seeds.",
        "- Do not run `qsub` from this task output.",
        "- `K_theta^*` is reported separately as the theta-space BvM target-Gaussianity "
        "diagnostic. `Delta_N,u` is reported separately as native Gaussian-NPE "
        "u-space approximation error.",
        "- The Gaussian-NPE rows are framed as BvM/Gaussian-family diagnostics, not as a "
        "flow-vs-Gaussian horse race.",
    ]

    if standard_gaps:
        lines.extend(
            [
                "",
                "Standard paper-grid shortfalls:",
            ]
        )
        for row in standard_gaps:
            lines.append(
                "- "
                f"{row['method']} n={row['n']} N={row['N']}: "
                f"seed_count={row['seed_count']}, kl={row['kl_seed_count']}, "
                f"coverage={row['coverage_seed_count']}, bias={row['bias_seed_count']}, "
                f"missing artifact seeds={row['missing_kl_seeds_observed_dirs']}"
            )

    lines.extend(
        [
            "",
            "Raw theta-space KL caveat:",
            "- The n=100 groups have 101 `kl.txt` files per method but include "
            "non-finite KL values. The comparable raw theta-KL tables use finite "
            "paired values only: 72 paired seeds for each n=100 budget.",
            "",
            "Bias caveat:",
            "- Flow seed `41` has non-finite replicate-level bias diagnostics in "
            "`n=1000, N=1000000` and `n=5000, N=353553`. If these bias summaries "
            "become paper-facing, prefer finite-value filtering or targeted "
            "evaluation repair for seed `41`.",
        ]
    )

    if legacy_rows:
        lines.extend(
            [
                "",
                "Legacy diagnostic excluded from complete summaries:",
            ]
        )
        for row in legacy_rows:
            lines.append(
                "- "
                f"{row['method']} n={row['n']} N={row['N']}: "
                f"seed_count={row['seed_count']}, "
                f"complete_kl_coverage_bias_seed_count="
                f"{row['complete_kl_coverage_bias_seed_count']}"
            )

    if nonstandard_gaps:
        lines.extend(
            [
                "",
                "Nonstandard/high-n groups with cache gaps, outside the Task 2 standard "
                "paper-grid refresh:",
            ]
        )
        for row in nonstandard_gaps:
            lines.append(
                "- "
                f"{row['method']} n={row['n']} N={row['N']}: "
                f"seed_count={row['seed_count']}, kl={row['kl_seed_count']}, "
                f"coverage={row['coverage_seed_count']}, bias={row['bias_seed_count']}"
            )

    path.write_text("\n".join(lines) + "\n")


def write_readme(path: Path, output_files: list[str]) -> None:
    lines = [
        "# GNK Task 2 Outputs",
        "",
        "All artifacts in this directory were generated from existing GNK octile caches "
        "and reviewed diagnostic CSV/JSON inputs. No cache files were modified.",
        "",
        "Key separation:",
        "",
        "- `theta_oracle_by_n.csv` contains theta-space `K_theta^*` only.",
        "- `gaussian_u_space_delta_N_u_by_n_N.csv` contains native Gaussian-NPE "
        "`Delta_N,u` diagnostics only.",
        "- The one-row `gaussian_npe_n_obs_1000_n_sims_31623` cache group is excluded "
        "from complete-group summaries.",
        "",
        "Generated files:",
    ]
    lines.extend(f"- `{name}`" for name in output_files)
    path.write_text("\n".join(lines) + "\n")


def write_outputs(
    records: list[CacheRecord],
    u_space_csv: Path,
    n500_gate_csv: Path,
    n500_gate_summary: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    grouped = group_records(records)

    inventory = inventory_rows(grouped)
    complete_inventory = [
        row for row in inventory if int(row["include_in_complete_group_summaries"]) == 1
    ]
    paper_inventory = [
        row
        for row in inventory
        if int(row["standard_paper_grid"]) == 1
        and int(row["legacy_one_row_excluded"]) == 0
    ]
    paired_kl, paired_kl_summary, method_kl_summary = raw_theta_kl_tables(records)
    complete_paired_kl_summary = [
        row for row in paired_kl_summary if int(row["complete_101_paired"]) == 1
    ]
    oracle_by_n, oracle_gate_check = theta_oracle_tables(u_space_csv, n500_gate_csv)
    u_delta, u_delta_per_seed, u_delta_excluded = u_space_delta_tables(u_space_csv)
    coverage, coverage_table = coverage_rows(records)
    bias_seed, bias_summary, bias_g_paper = bias_rows(records)

    inventory_fields = [
        "method",
        "n",
        "N",
        "N_label",
        "standard_paper_grid",
        "legacy_one_row_excluded",
        "dir_count",
        "seed_count",
        "min_seed",
        "max_seed",
        "seeds",
        "missing_dir_seeds_0_100",
        "kl_seed_count",
        "coverage_seed_count",
        "bias_seed_count",
        "posterior_seed_count",
        "complete_kl_coverage_bias_seed_count",
        "complete_101_kl_coverage_bias",
        "include_in_complete_group_summaries",
        "missing_kl_seeds_observed_dirs",
        "missing_coverage_seeds_observed_dirs",
        "missing_bias_seeds_observed_dirs",
        "missing_posterior_seeds_observed_dirs",
    ]
    paired_kl_fields = [
        "n",
        "N",
        "N_label",
        "standard_paper_grid",
        "seed",
        "flow_theta_kl",
        "gaussian_theta_kl",
    ]
    paired_kl_summary_fields = [
        "n",
        "N",
        "N_label",
        "standard_paper_grid",
        "paired_seed_count",
        "complete_101_paired",
        *[f"flow_theta_kl_{key}" for key in ("count", "mean", "sd", "min", "max", "q05", "q25", "median", "q75", "q95")],
        *[f"gaussian_theta_kl_{key}" for key in ("count", "mean", "sd", "min", "max", "q05", "q25", "median", "q75", "q95")],
    ]
    method_kl_summary_fields = [
        "method",
        "n",
        "N",
        "N_label",
        "standard_paper_grid",
        "kl_seed_count",
        "complete_101_kl",
        *[f"theta_kl_{key}" for key in ("count", "mean", "sd", "min", "max", "q05", "q25", "median", "q75", "q95")],
    ]
    oracle_fields = [
        "n",
        "source",
        "seed_count",
        *[f"K_theta_star_{key}" for key in ("count", "mean", "sd", "min", "max", "q05", "q25", "median", "q75", "q95")],
    ]
    oracle_gate_fields = [
        "n",
        "u_space_source",
        "gate_source",
        "matched_seed_count",
        "max_abs_K_theta_star_difference",
        "u_space_K_theta_star_median",
        "gate_K_theta_star_median",
    ]
    u_delta_fields = [
        "n",
        "N",
        "N_label",
        "standard_paper_grid",
        "seed_count",
        "excluded_from_complete_delta_summary",
        "exclusion_reason",
        "scaled_budget_median",
        *[f"Delta_N_u_total_{key}" for key in ("count", "mean", "sd", "min", "max", "q05", "q25", "median", "q75", "q95")],
        *[f"Delta_N_u_mean_term_{key}" for key in ("count", "mean", "sd", "min", "max", "q05", "q25", "median", "q75", "q95")],
        *[f"Delta_N_u_cov_term_{key}" for key in ("count", "mean", "sd", "min", "max", "q05", "q25", "median", "q75", "q95")],
        *[f"self_consistency_kl_u_{key}" for key in ("count", "mean", "sd", "min", "max", "q05", "q25", "median", "q75", "q95")],
    ]
    u_delta_seed_fields = [
        "n",
        "N",
        "N_label",
        "standard_paper_grid",
        "seed",
        "scaled_budget",
        "Delta_N_u_total",
        "Delta_N_u_mean_term",
        "Delta_N_u_cov_term",
        "self_consistency_kl_u",
    ]
    coverage_fields = [
        "method",
        "n",
        "N",
        "N_label",
        "param",
        "seed_count",
        "coverage_80_mean",
        "coverage_80_sd",
        "coverage_80_q05",
        "coverage_80_q25",
        "coverage_80_median",
        "coverage_80_q75",
        "coverage_80_q95",
        "coverage_90_mean",
        "coverage_90_sd",
        "coverage_90_q05",
        "coverage_90_q25",
        "coverage_90_median",
        "coverage_90_q75",
        "coverage_90_q95",
        "coverage_95_mean",
        "coverage_95_sd",
        "coverage_95_q05",
        "coverage_95_q25",
        "coverage_95_median",
        "coverage_95_q75",
        "coverage_95_q95",
    ]
    coverage_table_fields = [
        "method",
        "param",
        "n",
        "N=n",
        "N=n log(n)",
        "N=n^(3/2)",
        "N=n^2",
    ]
    bias_seed_fields = [
        "method",
        "n",
        "N",
        "N_label",
        "seed",
        "param",
        "replicate_count",
        "seed_mean_bias",
        "seed_mean_abs_bias",
        "seed_rmse_bias",
        "replicate_bias_q05",
        "replicate_bias_q25",
        "replicate_bias_median",
        "replicate_bias_q75",
        "replicate_bias_q95",
    ]
    bias_summary_fields = [
        "method",
        "n",
        "N",
        "N_label",
        "param",
        "seed_count",
        "replicate_count_total",
        "seed_mean_bias_mean",
        "seed_mean_bias_sd",
        "seed_mean_abs_bias_mean",
        "seed_mean_bias_rmse",
        "replicate_mean_abs_bias_mean",
        "seed_mean_bias_q05",
        "seed_mean_bias_q25",
        "seed_mean_bias_median",
        "seed_mean_bias_q75",
        "seed_mean_bias_q95",
    ]

    output_map = {
        "seed_count_inventory_all.csv": (inventory_fields, inventory),
        "complete_group_seed_count_inventory.csv": (inventory_fields, complete_inventory),
        "paper_grid_seed_count_inventory.csv": (inventory_fields, paper_inventory),
        "raw_theta_kl_paired_per_seed.csv": (paired_kl_fields, paired_kl),
        "raw_theta_kl_summary_comparable.csv": (
            paired_kl_summary_fields,
            paired_kl_summary,
        ),
        "raw_theta_kl_summary_complete_groups.csv": (
            paired_kl_summary_fields,
            complete_paired_kl_summary,
        ),
        "raw_theta_kl_summary_by_method.csv": (
            method_kl_summary_fields,
            method_kl_summary,
        ),
        "theta_oracle_by_n.csv": (oracle_fields, oracle_by_n),
        "theta_oracle_n500_gate_crosscheck.csv": (
            oracle_gate_fields,
            oracle_gate_check,
        ),
        "gaussian_u_space_delta_N_u_by_n_N.csv": (u_delta_fields, u_delta),
        "gaussian_u_space_delta_N_u_per_seed.csv": (
            u_delta_seed_fields,
            u_delta_per_seed,
        ),
        "gaussian_u_space_delta_N_u_excluded_groups.csv": (
            u_delta_fields,
            u_delta_excluded,
        ),
        "coverage_paper_grid_all_params.csv": (coverage_fields, coverage),
        "coverage_paper_grid_table_wide.csv": (
            coverage_table_fields,
            coverage_table,
        ),
        "bias_boxplot_values_paper_grid.csv": (bias_seed_fields, bias_seed),
        "bias_summary_paper_grid.csv": (bias_summary_fields, bias_summary),
        "bias_g_paper_figure_values.csv": (bias_seed_fields, bias_g_paper),
    }

    for filename, (fieldnames, rows) in output_map.items():
        write_csv(output_dir / filename, fieldnames, rows)

    write_rerun_note(output_dir / "gnk_task2_octile_hpc_rerun_note.md", inventory)

    summary = {
        "task": "gnk_task2_octile_flow_gaussian_oracle_aggregation",
        "cache_record_count": len(records),
        "cache_group_count": len(grouped),
        "inventory_rows": len(inventory),
        "complete_inventory_rows": len(complete_inventory),
        "paper_grid_inventory_rows": len(paper_inventory),
        "paired_theta_kl_rows": len(paired_kl),
        "paired_theta_kl_summary_rows": len(paired_kl_summary),
        "theta_oracle_rows": len(oracle_by_n),
        "u_space_delta_rows": len(u_delta),
        "coverage_rows": len(coverage),
        "bias_seed_rows": len(bias_seed),
        "bias_summary_rows": len(bias_summary),
        "inputs": {
            "u_space_csv": str(u_space_csv),
            "n500_gate_csv": str(n500_gate_csv),
            "n500_gate_summary": str(n500_gate_summary),
        },
    }
    if n500_gate_summary.is_file():
        summary["n500_gate_summary_json"] = json.loads(n500_gate_summary.read_text())

    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_readme(output_dir / "README.md", sorted([*output_map.keys(), "gnk_task2_octile_hpc_rerun_note.md", "run_summary.json"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate GNK octile Task 2 caches into debug outputs."
    )
    parser.add_argument("--cache-root", type=Path, default=Path("res/gnk"))
    parser.add_argument(
        "--u-space-csv",
        type=Path,
        default=Path("notebooks/plots/gnk_u_space_kl_decomp_20260425_per_seed.csv"),
    )
    parser.add_argument(
        "--n500-gate-summary",
        type=Path,
        default=Path("notebooks/plots/gnk_n500_oracle_gate_20260425_summary.json"),
    )
    parser.add_argument(
        "--n500-gate-csv",
        type=Path,
        default=Path("notebooks/plots/gnk_n500_oracle_gate_20260425_per_seed.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/plots/gnk_task2_20260502"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = parse_cache_records(args.cache_root)
    if not records:
        raise SystemExit(f"no GNK cache records found in {args.cache_root}")
    write_outputs(
        records=records,
        u_space_csv=args.u_space_csv,
        n500_gate_csv=args.n500_gate_csv,
        n500_gate_summary=args.n500_gate_summary,
        output_dir=args.output_dir,
    )
    print(f"wrote GNK Task 2 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
