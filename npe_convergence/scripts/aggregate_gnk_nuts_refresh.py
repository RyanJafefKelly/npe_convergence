"""Aggregate the canonical x64 NUTS reference cells under res/gnk_v3_refs/.

Reads every nuts_n_obs_*_seed_*_conv_*.pkl, extracts diagnostics, writes a
summary CSV and a short markdown note. Surfaces cells that breach R-hat,
ESS, or divergence thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle as pkl
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "res" / "gnk_v3_refs"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "docs" / "meeting_2026_05_18" / "gnk_nuts_refresh_plan"
)
PARAM_NAMES = ("A", "B", "g", "k")

FILENAME_RE = re.compile(
    r"^nuts_n_obs_(?P<n_obs>\d+)_seed_(?P<seed>\d+)_conv_(?P<convention>[a-z]+)\.pkl$"
)

RHAT_THRESHOLD = 1.01
RHAT_FAIL = 1.05
ESS_BULK_THRESHOLD = 1000
ESS_TAIL_THRESHOLD = 1000  # tracked but we currently only have n_eff; reported.
DIVERGENCE_FAIL = 1
MCSE_PER_SD_THRESHOLD = 0.03


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_filename(name: str) -> dict[str, Any] | None:
    match = FILENAME_RE.match(name)
    if not match:
        return None
    return {
        "n_obs": int(match.group("n_obs")),
        "seed": int(match.group("seed")),
        "convention": match.group("convention"),
    }


def load_fingerprint(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            return pkl.load(f)
    except Exception as exc:
        print(f"  skip {path.name}: {exc}")
        return None


def collect_row(path: Path) -> dict[str, Any]:
    parsed = parse_filename(path.name)
    if parsed is None:
        return {"file": path.name, "error": "unparseable_filename"}
    row: dict[str, Any] = {**parsed, "file": path.name}
    fingerprint = load_fingerprint(path)
    if fingerprint is None:
        row["error"] = "load_failure"
        return row

    diagnostics = fingerprint.get("diagnostics", {})
    per_param = diagnostics.get("per_parameter", {})
    for name in PARAM_NAMES:
        params = per_param.get(name, {})
        row[f"{name}_r_hat"] = params.get("r_hat")
        row[f"{name}_n_eff"] = params.get("n_eff")
        row[f"{name}_median"] = params.get("median")
        row[f"{name}_std"] = params.get("std")
        row[f"{name}_mcse_mean_per_sd"] = params.get("mcse_mean_per_sd")
        row[f"{name}_mcse_median_per_sd"] = params.get("mcse_median_per_sd")
    row["divergence_count"] = diagnostics.get("divergence_count")
    row["runtime_seconds"] = fingerprint.get("runtime_seconds")
    row["x_obs_sha256"] = fingerprint.get("x_obs_summary_unstandardised_sha256")
    row["data_epoch"] = fingerprint.get("data_epoch")
    row["density_version"] = fingerprint.get("density_version")
    row["num_chains"] = fingerprint.get("num_chains")
    row["num_warmup"] = fingerprint.get("num_warmup")
    row["num_samples_per_chain"] = fingerprint.get("num_samples_per_chain")
    row["thinning"] = fingerprint.get("thinning")
    row["sampler_seed"] = fingerprint.get("sampler_seed")
    env = fingerprint.get("environment", {})
    row["jax_x64_enabled"] = env.get("jax_x64_enabled")
    row["jax_version"] = env.get("jax_version")
    row["numpyro_version"] = env.get("numpyro_version")
    row["git_commit"] = env.get("git_commit")
    row["git_dirty"] = env.get("git_dirty")
    row["utc_timestamp"] = fingerprint.get("utc_timestamp")

    # Pass / fail flags.
    rhat_vals = [row[f"{n}_r_hat"] for n in PARAM_NAMES if row[f"{n}_r_hat"] is not None]
    ess_vals = [row[f"{n}_n_eff"] for n in PARAM_NAMES if row[f"{n}_n_eff"] is not None]
    mcse_vals = [
        row[f"{n}_mcse_mean_per_sd"]
        for n in PARAM_NAMES
        if row[f"{n}_mcse_mean_per_sd"] is not None
    ]

    if rhat_vals:
        row["max_r_hat"] = max(rhat_vals)
        row["rhat_pass"] = row["max_r_hat"] < RHAT_THRESHOLD
        row["rhat_borderline"] = (
            RHAT_THRESHOLD <= row["max_r_hat"] < RHAT_FAIL
        )
        row["rhat_fail"] = row["max_r_hat"] >= RHAT_FAIL
    if ess_vals:
        row["min_n_eff"] = min(ess_vals)
        row["ess_pass"] = row["min_n_eff"] >= ESS_BULK_THRESHOLD
    if mcse_vals:
        row["max_mcse_per_sd"] = max(mcse_vals)
        row["mcse_pass"] = row["max_mcse_per_sd"] < MCSE_PER_SD_THRESHOLD
    if row.get("divergence_count") is not None:
        row["divergence_pass"] = row["divergence_count"] < DIVERGENCE_FAIL

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list({k for row in rows for k in row.keys()})
    fieldnames.sort()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    rhat_fail = sum(1 for r in rows if r.get("rhat_fail"))
    rhat_borderline = sum(1 for r in rows if r.get("rhat_borderline"))
    ess_fail = sum(1 for r in rows if r.get("ess_pass") is False)
    divergence_fail = sum(1 for r in rows if r.get("divergence_pass") is False)
    mcse_fail = sum(1 for r in rows if r.get("mcse_pass") is False)
    x64_off = sum(1 for r in rows if r.get("jax_x64_enabled") is False)
    convention_counts: dict[str, int] = {}
    n_obs_counts: dict[int, int] = {}
    for r in rows:
        if "convention" in r:
            convention_counts[r["convention"]] = convention_counts.get(r["convention"], 0) + 1
        if "n_obs" in r:
            n_obs_counts[r["n_obs"]] = n_obs_counts.get(r["n_obs"], 0) + 1
    runtimes = [r["runtime_seconds"] for r in rows if r.get("runtime_seconds") is not None]
    return {
        "created_at": utc_now(),
        "total_rows": total,
        "convention_counts": convention_counts,
        "n_obs_counts": {str(k): v for k, v in n_obs_counts.items()},
        "rhat_fail_count": rhat_fail,
        "rhat_borderline_count": rhat_borderline,
        "ess_fail_count": ess_fail,
        "divergence_fail_count": divergence_fail,
        "mcse_fail_count": mcse_fail,
        "x64_off_count": x64_off,
        "runtime_seconds_min": min(runtimes) if runtimes else None,
        "runtime_seconds_max": max(runtimes) if runtimes else None,
        "runtime_seconds_total": sum(runtimes) if runtimes else None,
        "thresholds": {
            "rhat": RHAT_THRESHOLD,
            "rhat_fail": RHAT_FAIL,
            "n_eff_bulk": ESS_BULK_THRESHOLD,
            "n_eff_tail": ESS_TAIL_THRESHOLD,
            "mcse_per_sd": MCSE_PER_SD_THRESHOLD,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-prefix", type=str, default="gnk_nuts_refresh_audit"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    if not input_root.exists():
        print(f"input root does not exist: {input_root}")
        return
    files = sorted(input_root.glob("nuts_n_obs_*_seed_*_conv_*.pkl"))
    rows = [collect_row(f) for f in files]
    summary = summarise(rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}.csv"
    json_path = output_dir / f"{args.output_prefix}.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Aggregated {summary['total_rows']} canonical references.")
    print(f"  R-hat fail (>= {RHAT_FAIL}): {summary['rhat_fail_count']}")
    print(
        f"  R-hat borderline ({RHAT_THRESHOLD} to {RHAT_FAIL}): "
        f"{summary['rhat_borderline_count']}"
    )
    print(f"  ESS fail (n_eff < {ESS_BULK_THRESHOLD}): {summary['ess_fail_count']}")
    print(f"  divergence fail: {summary['divergence_fail_count']}")
    print(f"  MCSE fail: {summary['mcse_fail_count']}")
    print(f"  x64 off: {summary['x64_off_count']}")
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")


if __name__ == "__main__":
    main()
