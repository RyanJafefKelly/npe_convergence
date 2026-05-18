"""Aggregate stereological flow/Gaussian cache artifacts for Task 1.

This is intentionally cache read-only. It writes co-author/debug CSVs and a
short rerun note into a fresh output directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import gaussian_kde


PARAMS = ("lambda", "sigma", "xi")
TRUE_VALUES = {"lambda": 100.0, "sigma": 2.0, "xi": -0.1}
COVERAGE_LEVELS = (0.8, 0.9, 0.95)
STANDARD_N = (100, 500, 1000, 5000)
DIR_RE = re.compile(
    r"^(?P<prefix>gaussian_npe|npe)_n_obs_(?P<n_obs>\d+)_n_sims_"
    r"(?P<n_sims>\d+)_seed_(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class CacheRecord:
    method: str
    prefix: str
    n_obs: int
    n_sims: int
    seed: int
    path: Path
    has_coverage: bool
    has_biases: bool
    has_posterior_samples: bool

    @property
    def coverage_path(self) -> Path:
        return self.path / "estimated_coverage.npy"

    @property
    def biases_path(self) -> Path:
        return self.path / "biases.npy"

    @property
    def posterior_path(self) -> Path:
        return self.path / "posterior_samples.pkl"


def standard_n_sims(n_obs: int) -> dict[str, int]:
    return {
        "N=n": n_obs,
        "N=n log(n)": int(n_obs * math.log(n_obs)),
        "N=n^(3/2)": int(n_obs ** (3 / 2)),
        "N=n^2": n_obs**2,
    }


def n_sims_label(n_obs: int, n_sims: int) -> str:
    for label, value in standard_n_sims(n_obs).items():
        if n_sims == value:
            return label
    return "other"


def method_from_prefix(prefix: str) -> str:
    if prefix == "gaussian_npe":
        return "gaussian_npe"
    if prefix == "npe":
        return "flow_npe"
    raise ValueError(f"unknown prefix: {prefix}")


def parse_caches(cache_root: Path) -> list[CacheRecord]:
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
                n_obs=int(match.group("n_obs")),
                n_sims=int(match.group("n_sims")),
                seed=int(match.group("seed")),
                path=path,
                has_coverage=(path / "estimated_coverage.npy").is_file(),
                has_biases=(path / "biases.npy").is_file(),
                has_posterior_samples=(path / "posterior_samples.pkl").is_file(),
            )
        )
    return sorted(records, key=lambda r: (r.method, r.n_obs, r.n_sims, r.seed))


def group_records(records: Iterable[CacheRecord]) -> dict[tuple[str, int, int], list[CacheRecord]]:
    grouped: dict[tuple[str, int, int], list[CacheRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.method, record.n_obs, record.n_sims)].append(record)
    return dict(grouped)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_seed_list(seeds: Iterable[int]) -> str:
    return ";".join(str(seed) for seed in sorted(seeds))


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "q05": math.nan,
            "q25": math.nan,
            "q50": math.nan,
            "q75": math.nan,
            "q95": math.nan,
        }
    qs = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "q05": float(qs[0]),
        "q25": float(qs[1]),
        "q50": float(qs[2]),
        "q75": float(qs[3]),
        "q95": float(qs[4]),
    }


def summarize_values(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            f"{prefix}mean": math.nan,
            f"{prefix}sd": math.nan,
            f"{prefix}min": math.nan,
            f"{prefix}max": math.nan,
            **{f"{prefix}{key}": value for key, value in quantiles(values).items()},
        }
    return {
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        f"{prefix}min": float(np.min(values)),
        f"{prefix}max": float(np.max(values)),
        **{f"{prefix}{key}": value for key, value in quantiles(values).items()},
    }


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


def load_posterior_samples(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        samples = pickle.load(handle)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(PARAMS):
        raise ValueError(f"unexpected posterior sample shape {samples.shape}: {path}")
    return samples


def inventory_rows(grouped: dict[tuple[str, int, int], list[CacheRecord]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (method, n_obs, n_sims), records in sorted(grouped.items()):
        seeds = {record.seed for record in records}
        coverage_seeds = {record.seed for record in records if record.has_coverage}
        bias_seeds = {record.seed for record in records if record.has_biases}
        posterior_seeds = {record.seed for record in records if record.has_posterior_samples}
        complete_bias_coverage = coverage_seeds & bias_seeds
        rows.append(
            {
                "method": method,
                "n": n_obs,
                "N": n_sims,
                "N_label": n_sims_label(n_obs, n_sims),
                "dir_count": len(records),
                "seed_count": len(seeds),
                "min_seed": min(seeds),
                "max_seed": max(seeds),
                "seeds": fmt_seed_list(seeds),
                "coverage_seed_count": len(coverage_seeds),
                "bias_seed_count": len(bias_seeds),
                "posterior_seed_count": len(posterior_seeds),
                "complete_bias_coverage_seed_count": len(complete_bias_coverage),
                "missing_coverage_seeds": fmt_seed_list(seeds - coverage_seeds),
                "missing_bias_seeds": fmt_seed_list(seeds - bias_seeds),
                "missing_posterior_seeds": fmt_seed_list(seeds - posterior_seeds),
            }
        )
    return rows


def coverage_rows(grouped: dict[tuple[str, int, int], list[CacheRecord]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (method, n_obs, n_sims), records in sorted(grouped.items()):
        coverage_records = [record for record in records if record.has_coverage]
        if not coverage_records:
            continue
        covs = np.stack([load_coverage(record) for record in coverage_records])
        for param_index, param in enumerate(PARAMS):
            values = covs[:, param_index, :]
            row = {
                "method": method,
                "n": n_obs,
                "N": n_sims,
                "N_label": n_sims_label(n_obs, n_sims),
                "param": param,
                "seed_count": len(coverage_records),
            }
            for level_index, level in enumerate(COVERAGE_LEVELS):
                level_values = values[:, level_index]
                suffix = str(int(level * 100))
                row[f"coverage_{suffix}_mean"] = float(np.mean(level_values))
                row[f"coverage_{suffix}_sd"] = (
                    float(np.std(level_values, ddof=1)) if len(level_values) > 1 else 0.0
                )
            rows.append(row)
    return rows


def bias_by_seed_rows(grouped: dict[tuple[str, int, int], list[CacheRecord]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (method, n_obs, n_sims), records in sorted(grouped.items()):
        for record in records:
            if not record.has_biases:
                continue
            biases = load_biases(record)
            for param_index, param in enumerate(PARAMS):
                values = biases[:, param_index]
                row = {
                    "method": method,
                    "n": n_obs,
                    "N": n_sims,
                    "N_label": n_sims_label(n_obs, n_sims),
                    "seed": record.seed,
                    "param": param,
                    "source": "biases.npy",
                    "replicate_count": values.size,
                    "mean_bias": float(np.mean(values)),
                    "mean_abs_bias": float(np.mean(np.abs(values))),
                    "rmse_bias": float(np.sqrt(np.mean(values**2))),
                }
                row.update(quantiles(values))
                rows.append(row)
    return rows


def bias_summary_rows(bias_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped_values: dict[tuple[str, int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in bias_rows:
        grouped_values[(str(row["method"]), int(row["n"]), int(row["N"]), str(row["param"]))].append(row)

    rows: list[dict[str, object]] = []
    for (method, n_obs, n_sims, param), group in sorted(grouped_values.items()):
        seed_mean_bias = np.asarray([float(row["mean_bias"]) for row in group], dtype=float)
        mean_abs_bias = np.asarray([float(row["mean_abs_bias"]) for row in group], dtype=float)
        replicate_count_total = int(sum(int(row["replicate_count"]) for row in group))
        row = {
            "method": method,
            "n": n_obs,
            "N": n_sims,
            "N_label": n_sims_label(n_obs, n_sims),
            "param": param,
            "seed_count": len(group),
            "replicate_count_total": replicate_count_total,
            "seed_mean_bias_mean": float(np.mean(seed_mean_bias)),
            "seed_mean_bias_sd": (
                float(np.std(seed_mean_bias, ddof=1)) if seed_mean_bias.size > 1 else 0.0
            ),
            "seed_mean_abs_bias_mean": float(np.mean(np.abs(seed_mean_bias))),
            "seed_mean_bias_rmse": float(np.sqrt(np.mean(seed_mean_bias**2))),
            "replicate_mean_abs_bias_mean": float(np.mean(mean_abs_bias)),
        }
        row.update({f"seed_mean_bias_{key}": value for key, value in quantiles(seed_mean_bias).items()})
        rows.append(row)
    return rows


def posterior_mean_bias_rows(
    grouped: dict[tuple[str, int, int], list[CacheRecord]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (method, n_obs, n_sims), records in sorted(grouped.items()):
        for record in records:
            if not record.has_posterior_samples:
                continue
            samples = load_posterior_samples(record.posterior_path)
            means = np.mean(samples, axis=0)
            for param_index, param in enumerate(PARAMS):
                rows.append(
                    {
                        "method": method,
                        "n": n_obs,
                        "N": n_sims,
                        "N_label": n_sims_label(n_obs, n_sims),
                        "seed": record.seed,
                        "param": param,
                        "source": "posterior_samples.pkl",
                        "sample_count": samples.shape[0],
                        "posterior_mean": float(means[param_index]),
                        "bias": float(means[param_index] - TRUE_VALUES[param]),
                    }
                )
    return rows


def build_standard_keys() -> list[tuple[int, int, str]]:
    keys: list[tuple[int, int, str]] = []
    for n_obs in STANDARD_N:
        for label, n_sims in standard_n_sims(n_obs).items():
            keys.append((n_obs, n_sims, label))
    return keys


def record_lookup(records: Iterable[CacheRecord]) -> dict[tuple[str, int, int, int], CacheRecord]:
    return {
        (record.method, record.n_obs, record.n_sims, record.seed): record
        for record in records
    }


def overlay_manifest_and_samples(
    records: list[CacheRecord],
    smc_root: Path,
    overlay_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    lookup = record_lookup(records)
    manifest: list[dict[str, object]] = []
    sample_blocks: list[tuple[dict[str, object], np.ndarray]] = []

    n_obs = 1000
    for method in ("flow_npe", "gaussian_npe"):
        for label, n_sims in standard_n_sims(n_obs).items():
            record = lookup.get((method, n_obs, n_sims, overlay_seed))
            has_samples = bool(record and record.has_posterior_samples)
            sample_count = 0
            path = ""
            if record is not None:
                path = str(record.posterior_path)
            if has_samples and record is not None:
                samples = load_posterior_samples(record.posterior_path)
                sample_count = samples.shape[0]
                sample_blocks.append(
                    (
                        {
                            "method": method,
                            "source": "posterior_samples.pkl",
                            "n": n_obs,
                            "N": n_sims,
                            "N_label": label,
                            "seed": overlay_seed,
                        },
                        samples,
                    )
                )
            manifest.append(
                {
                    "method": method,
                    "source": "posterior_samples.pkl",
                    "n": n_obs,
                    "N": n_sims,
                    "N_label": label,
                    "seed": overlay_seed,
                    "has_samples": int(has_samples),
                    "sample_count": sample_count,
                    "path": path,
                }
            )

    smc_path = (
        smc_root
        / "npe_n_obs_1000_n_sims_None_seed_1_max_iter_9"
        / "adaptive_smc_samples.pkl"
    )
    if smc_path.is_file():
        with smc_path.open("rb") as handle:
            smc_samples_raw = np.asarray(pickle.load(handle), dtype=float)
        # Saved ELFI sample order follows the existing notebook for lambda:
        # col2=lambda. The corresponding parameter order is xi, sigma, lambda.
        smc_samples = np.column_stack(
            [smc_samples_raw[:, 2], smc_samples_raw[:, 1], smc_samples_raw[:, 0]]
        )
        sample_blocks.append(
            (
                {
                    "method": "abc_smc",
                    "source": "adaptive_smc_samples.pkl",
                    "n": n_obs,
                    "N": "",
                    "N_label": "ABC-SMC",
                    "seed": 1,
                },
                smc_samples,
            )
        )
        manifest.append(
            {
                "method": "abc_smc",
                "source": "adaptive_smc_samples.pkl",
                "n": n_obs,
                "N": "",
                "N_label": "ABC-SMC",
                "seed": 1,
                "has_samples": 1,
                "sample_count": smc_samples.shape[0],
                "path": str(smc_path),
            }
        )
    else:
        manifest.append(
            {
                "method": "abc_smc",
                "source": "adaptive_smc_samples.pkl",
                "n": n_obs,
                "N": "",
                "N_label": "ABC-SMC",
                "seed": 1,
                "has_samples": 0,
                "sample_count": 0,
                "path": str(smc_path),
            }
        )

    sample_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    grids = {
        "lambda": np.linspace(90.0, 110.0, 400),
        "sigma": np.linspace(0.0, 4.0, 400),
        "xi": np.linspace(-1.0, 1.0, 400),
    }

    for metadata, samples in sample_blocks:
        for param_index, param in enumerate(PARAMS):
            values = samples[:, param_index]
            for sample_index, value in enumerate(values):
                sample_rows.append(
                    {
                        **metadata,
                        "param": param,
                        "sample_index": sample_index,
                        "value": float(value),
                    }
                )
            try:
                kde = gaussian_kde(values)
                for x, density in zip(grids[param], kde(grids[param])):
                    density_rows.append(
                        {
                            **metadata,
                            "param": param,
                            "x": float(x),
                            "density": float(density),
                        }
                    )
            except Exception as exc:  # pragma: no cover - diagnostic path
                density_rows.append(
                    {
                        **metadata,
                        "param": param,
                        "x": "",
                        "density": "",
                        "error": repr(exc),
                    }
                )

    return manifest, sample_rows, density_rows


def count_complete_standard_groups(
    inventory: list[dict[str, object]], method: str, artifact_column: str, threshold: int
) -> tuple[int, list[str]]:
    by_key = {
        (str(row["method"]), int(row["n"]), int(row["N"])): int(row[artifact_column])
        for row in inventory
    }
    complete = 0
    missing: list[str] = []
    for n_obs, n_sims, label in build_standard_keys():
        count = by_key.get((method, n_obs, n_sims), 0)
        if count >= threshold:
            complete += 1
        else:
            missing.append(f"{method} n={n_obs} {label} N={n_sims}: {count}")
    return complete, missing


def write_rerun_note(
    path: Path,
    inventory: list[dict[str, object]],
    manifest: list[dict[str, object]],
) -> None:
    flow_cov_complete, flow_cov_missing = count_complete_standard_groups(
        inventory, "flow_npe", "complete_bias_coverage_seed_count", 100
    )
    gaussian_cov_complete, gaussian_cov_missing = count_complete_standard_groups(
        inventory, "gaussian_npe", "complete_bias_coverage_seed_count", 100
    )
    flow_posterior_complete, flow_posterior_missing = count_complete_standard_groups(
        inventory, "flow_npe", "posterior_seed_count", 100
    )
    gaussian_posterior_complete, gaussian_posterior_missing = count_complete_standard_groups(
        inventory, "gaussian_npe", "posterior_seed_count", 1
    )
    gaussian_overlay_present = any(
        row["method"] == "gaussian_npe" and int(row["has_samples"]) == 1
        for row in manifest
    )
    abc_present = any(row["method"] == "abc_smc" and int(row["has_samples"]) == 1 for row in manifest)

    nonstandard_shortfalls = [
        row
        for row in inventory
        if row["N_label"] == "other"
        and (
            int(row["complete_bias_coverage_seed_count"]) < 100
            or int(row["posterior_seed_count"]) < 100
        )
    ]

    has_standard_bias_gaps = bool(flow_cov_missing or gaussian_cov_missing)
    has_gaussian_overlay_gap = not gaussian_overlay_present
    if has_standard_bias_gaps:
        recommendation = (
            "Recommendation: a targeted stereological rerun or recovery is needed if the "
            "paper-style standard grid must be complete at 100 seeds for every `(n, N)` "
            "cell. Do not launch broad stereological scripts from this task output alone; "
            "prepare a collision-checked manifest containing only the shortfall groups "
            "listed below."
        )
    else:
        recommendation = (
            "Recommendation: no stereological HPC rerun is needed for the Task 1 seed "
            "inventory, coverage tables, or bias summaries. Do not launch the broad "
            "stereological flow or Gaussian job scripts on this evidence."
        )

    lines = [
        "# Stereological Task 1 Rerun Note",
        "",
        "- Standard coverage/bias cache status: "
        f"flow_npe {flow_cov_complete}/16 groups have at least 100 complete seeds; "
        f"gaussian_npe {gaussian_cov_complete}/16 groups have at least 100 complete seeds.",
        "- Standard flow posterior-sample status: "
        f"{flow_posterior_complete}/16 groups have at least 100 saved posterior sample files.",
        "- Standard Gaussian posterior-sample status: "
        f"{gaussian_posterior_complete}/16 groups have any saved posterior sample file.",
        f"- ABC-SMC n=1000 benchmark sample file present: {'yes' if abc_present else 'no'}.",
        "",
        recommendation,
    ]

    if has_gaussian_overlay_gap:
        lines.extend(
            [
                "",
                "Caveat: Gaussian-NPE cache directories in this snapshot contain coverage "
                "and bias arrays but no `posterior_samples.pkl` files for the n=1000 "
                "overlay inputs, so a Gaussian posterior overlay cannot be regenerated "
                "from saved posterior draws. A fresh Gaussian run or a cache-safe "
                "posterior export would only be needed if a Gaussian posterior overlay "
                "is required for the paper-facing figure.",
                "",
                "Gaussian overlay status: the n=1000 Gaussian coverage/bias caches exist for "
                "`N in {n, n log(n), n^(3/2)}`, but there are no saved Gaussian posterior "
                "sample files for the overlay inputs.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Gaussian overlay status: saved Gaussian-NPE posterior samples are present "
                "for the n=1000 overlay inputs listed in "
                "`posterior_overlay_manifest_n1000_seed1.csv`.",
            ]
        )

    if flow_cov_missing or gaussian_cov_missing:
        lines.extend(["", "Coverage/bias standard-grid shortfalls:"])
        lines.extend(f"- {item}" for item in flow_cov_missing + gaussian_cov_missing)
    if flow_posterior_missing:
        lines.extend(["", "Flow posterior standard-grid shortfalls:"])
        lines.extend(f"- {item}" for item in flow_posterior_missing)
    if gaussian_posterior_missing and not gaussian_overlay_present:
        lines.extend(["", "Gaussian posterior standard-grid shortfalls:"])
        lines.extend(f"- {item}" for item in gaussian_posterior_missing)
    if nonstandard_shortfalls:
        lines.extend(["", "Nonstandard/incomplete diagnostic groups:"])
        for row in nonstandard_shortfalls:
            lines.append(
                "- "
                f"{row['method']} n={row['n']} N={row['N']}: "
                f"complete coverage/bias seeds={row['complete_bias_coverage_seed_count']}, "
                f"posterior seeds={row['posterior_seed_count']}"
            )
    path.write_text("\n".join(lines) + "\n")


def write_outputs(
    records: list[CacheRecord],
    smc_root: Path,
    output_dir: Path,
    overlay_seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    grouped = group_records(records)

    inventory = inventory_rows(grouped)
    coverage = coverage_rows(grouped)
    bias_seed = bias_by_seed_rows(grouped)
    bias_summary = bias_summary_rows(bias_seed)
    posterior_bias = posterior_mean_bias_rows(grouped)
    overlay_manifest, overlay_samples, overlay_density = overlay_manifest_and_samples(
        records, smc_root, overlay_seed
    )

    inventory_fields = [
        "method",
        "n",
        "N",
        "N_label",
        "dir_count",
        "seed_count",
        "min_seed",
        "max_seed",
        "seeds",
        "coverage_seed_count",
        "bias_seed_count",
        "posterior_seed_count",
        "complete_bias_coverage_seed_count",
        "missing_coverage_seeds",
        "missing_bias_seeds",
        "missing_posterior_seeds",
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
        "coverage_90_mean",
        "coverage_90_sd",
        "coverage_95_mean",
        "coverage_95_sd",
    ]
    bias_seed_fields = [
        "method",
        "n",
        "N",
        "N_label",
        "seed",
        "param",
        "source",
        "replicate_count",
        "mean_bias",
        "mean_abs_bias",
        "rmse_bias",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
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
        "seed_mean_bias_q50",
        "seed_mean_bias_q75",
        "seed_mean_bias_q95",
    ]
    posterior_bias_fields = [
        "method",
        "n",
        "N",
        "N_label",
        "seed",
        "param",
        "source",
        "sample_count",
        "posterior_mean",
        "bias",
    ]
    overlay_manifest_fields = [
        "method",
        "source",
        "n",
        "N",
        "N_label",
        "seed",
        "has_samples",
        "sample_count",
        "path",
    ]
    overlay_sample_fields = [
        "method",
        "source",
        "n",
        "N",
        "N_label",
        "seed",
        "param",
        "sample_index",
        "value",
    ]
    overlay_density_fields = [
        "method",
        "source",
        "n",
        "N",
        "N_label",
        "seed",
        "param",
        "x",
        "density",
    ]

    write_csv(output_dir / "seed_count_inventory.csv", inventory_fields, inventory)
    write_csv(output_dir / "coverage_all_params.csv", coverage_fields, coverage)
    for param in PARAMS:
        write_csv(
            output_dir / f"coverage_{param}.csv",
            coverage_fields,
            [row for row in coverage if row["param"] == param],
        )
    write_csv(output_dir / "bias_boxplot_by_seed.csv", bias_seed_fields, bias_seed)
    write_csv(output_dir / "bias_summary.csv", bias_summary_fields, bias_summary)
    write_csv(
        output_dir / "posterior_mean_bias_by_seed_if_available.csv",
        posterior_bias_fields,
        posterior_bias,
    )
    write_csv(
        output_dir / "posterior_overlay_manifest_n1000_seed1.csv",
        overlay_manifest_fields,
        overlay_manifest,
    )
    write_csv(
        output_dir / "posterior_overlay_samples_n1000_seed1.csv",
        overlay_sample_fields,
        overlay_samples,
    )
    write_csv(
        output_dir / "posterior_overlay_density_n1000_seed1.csv",
        overlay_density_fields,
        overlay_density,
    )
    write_rerun_note(output_dir / "stereological_task1_rerun_note.md", inventory, overlay_manifest)

    summary = {
        "cache_record_count": len(records),
        "group_count": len(grouped),
        "inventory_rows": len(inventory),
        "coverage_rows": len(coverage),
        "bias_boxplot_rows": len(bias_seed),
        "bias_summary_rows": len(bias_summary),
        "posterior_mean_bias_rows": len(posterior_bias),
        "overlay_manifest_rows": len(overlay_manifest),
        "overlay_sample_rows": len(overlay_samples),
        "overlay_density_rows": len(overlay_density),
    }
    (output_dir / "run_summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate stereological Task 1 cache artifacts into debug outputs."
    )
    parser.add_argument("--cache-root", type=Path, default=Path("res/stereological"))
    parser.add_argument(
        "--smc-root", type=Path, default=Path("res/stereological_smc_abc")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/plots/stereological_task1_20260502"),
    )
    parser.add_argument("--overlay-seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = parse_caches(args.cache_root)
    if not records:
        raise SystemExit(f"no stereological cache records found in {args.cache_root}")
    write_outputs(records, args.smc_root, args.output_dir, args.overlay_seed)
    print(f"wrote stereological Task 1 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
