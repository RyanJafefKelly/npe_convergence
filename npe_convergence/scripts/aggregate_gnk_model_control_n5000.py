"""Aggregate the GNK n=5000 asymptotic-MVN model-control array."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle as pkl
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)
assert jax.config.read("jax_enable_x64")

import jax.numpy as jnp
import jax.random as random
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.metrics import kullback_leibler, median_heuristic, unbiased_mmd


N_OBS = 5000
N_SIMS = 25_000_000
N_METRIC = 2000
PARAM_NAMES = ("A", "B", "g", "k")
DEFAULT_CONTROL_ROOT = REPO_ROOT / "res" / "gnk_model_control_n5000"
DEFAULT_REAL_ROOT = REPO_ROOT / "res" / "gnk"
DEFAULT_V3_ROOT = REPO_ROOT / "res" / "gnk_v3_refs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "plots" / "gnk_model_control_n5000"
DEFAULT_DOC_PATH = (
    REPO_ROOT
    / "docs"
    / "meeting_2026_05_18"
    / "gnk_model_control_n5000_outcome.md"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_git(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return default


def git_dirty() -> bool:
    return bool(run_git(["status", "--porcelain"], default="dirty"))


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_float(path: Path) -> float:
    return float(path.read_text().strip())


def deduplicate(samples: np.ndarray) -> tuple[np.ndarray, int]:
    unique = np.unique(np.asarray(samples, dtype=np.float64), axis=0)
    return unique, int(samples.shape[0] - unique.shape[0])


def deterministic_subsample(samples: np.ndarray, n: int, seed: int) -> np.ndarray:
    if samples.shape[0] <= n:
        return samples
    key = random.key(seed)
    idx = np.asarray(random.permutation(key, samples.shape[0])[:n])
    return samples[idx]


def load_reference(v3_root: Path, seed: int) -> dict[str, Any]:
    path = v3_root / f"nuts_n_obs_{N_OBS}_seed_{seed}_conv_gaussian.pkl"
    if not path.exists():
        raise FileNotFoundError(f"missing v3 reference: {path}")
    with path.open("rb") as f:
        ref = pkl.load(f)
    ref["_path"] = str(path)
    return ref


def reference_diagnostics(ref: dict[str, Any]) -> tuple[int | None, float | None, float | None]:
    diag = ref.get("diagnostics", {})
    divergences = diag.get("divergence_count")
    per_param = diag.get("per_parameter", {})
    rhat_vals = [
        float(values["r_hat"])
        for values in per_param.values()
        if values.get("r_hat") is not None
    ]
    ess_vals = [
        float(values["n_eff"])
        for values in per_param.values()
        if values.get("n_eff") is not None
    ]
    return (
        int(divergences) if divergences is not None else None,
        max(rhat_vals) if rhat_vals else None,
        min(ess_vals) if ess_vals else None,
    )


def noise_floor_kl(ref: dict[str, Any], seed: int) -> float:
    grouped = np.asarray(ref["samples"], dtype=np.float64)
    if grouped.ndim != 3:
        raise ValueError("v3 reference samples must have shape (chains, draws, theta)")
    unique, _ = deduplicate(grouped.reshape(-1, grouped.shape[-1]))
    rng = np.random.default_rng(stable_int("nuts_self_split", N_OBS, seed, "gaussian"))
    perm = rng.permutation(unique.shape[0])
    half = unique.shape[0] // 2
    left = unique[perm[:half]]
    right = unique[perm[half : 2 * half]]
    left = deterministic_subsample(left, N_METRIC, stable_int("noise_left", seed))
    right = deterministic_subsample(right, N_METRIC, stable_int("noise_right", seed))
    return max(0.0, float(kullback_leibler(jnp.asarray(left), jnp.asarray(right))))


def load_control_rows(control_root: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(control_root.glob("*/metrics.json")):
        cell_dir = metrics_path.parent
        config_path = cell_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"missing copied config for completed cell: {config_path}")
        metrics = read_json(metrics_path)
        config = yaml.safe_load(config_path.read_text())
        rows.append(
            {
                "run_id": config["run_id"],
                "observed_seed": int(config["observed_seed"]),
                "training_seed": int(config.get("training_seed", config["observed_seed"])),
                "control_kl": float(metrics["kl_theta_knn_2000"]),
                "control_mmd": float(metrics["mmd_theta_2000"]),
                "cell_dir": cell_dir,
                "metrics_path": metrics_path,
                "config_path": config_path,
            }
        )
    return rows


def real_cell_paths(real_root: Path, seed: int) -> dict[str, Path]:
    cell = real_root / (
        f"gaussian_npe_n_obs_{N_OBS}_n_sims_{N_SIMS}_seed_{seed}"
    )
    return {
        "cell": cell,
        "kl": cell / "kl.txt",
        "mmd": cell / "mmd.txt",
        "metrics_v3": cell / "metrics_v3.json",
    }


def add_real_and_reference_metrics(
    rows: list[dict[str, Any]], real_root: Path, v3_root: Path
) -> list[dict[str, Any]]:
    noise_by_seed: dict[int, float] = {}
    diag_by_seed: dict[int, tuple[int | None, float | None, float | None]] = {}
    for row in rows:
        seed = int(row["observed_seed"])
        paths = real_cell_paths(real_root, seed)
        if not paths["kl"].exists():
            raise FileNotFoundError(f"missing real-GNK KL: {paths['kl']}")
        if not paths["mmd"].exists():
            raise FileNotFoundError(f"missing real-GNK MMD: {paths['mmd']}")
        if seed not in noise_by_seed:
            ref = load_reference(v3_root, seed)
            noise_by_seed[seed] = noise_floor_kl(ref, seed)
            diag_by_seed[seed] = reference_diagnostics(ref)
        divergences, rhat_max, ess_min = diag_by_seed[seed]
        timing_path = row["cell_dir"] / "timing_metadata.json"
        runtime_seconds = None
        if timing_path.exists():
            runtime_seconds = read_json(timing_path).get("total_wall_time_seconds")
        real_kl = read_float(paths["kl"])
        real_mmd = read_float(paths["mmd"])
        row.update(
            {
                "real_kl": real_kl,
                "delta_s": real_kl - float(row["control_kl"]),
                "noise_floor_kl": noise_by_seed[seed],
                "real_mmd": real_mmd,
                "runtime_seconds": runtime_seconds,
                "divergences": divergences,
                "rhat_max": rhat_max,
                "ess_min": ess_min,
            }
        )
    return rows


def bootstrap_median_ci(values: np.ndarray, draws: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    medians = np.empty(draws, dtype=np.float64)
    n = values.shape[0]
    for i in range(draws):
        sample = values[rng.integers(0, n, size=n)]
        medians[i] = np.median(sample)
    return {
        "median": float(np.median(values)),
        "ci_low": float(np.quantile(medians, 0.025)),
        "ci_high": float(np.quantile(medians, 0.975)),
        "bootstrap_draws": draws,
    }


def training_seed_variance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {(r["observed_seed"], r["training_seed"]): r for r in rows}
    out = []
    for row in rows:
        seed = row["observed_seed"]
        train = row["training_seed"]
        if seed == train:
            continue
        primary = by_seed.get((seed, seed))
        if primary is None:
            raise RuntimeError(f"missing primary cell for observed seed {seed}")
        diff = float(row["control_kl"] - primary["control_kl"])
        out.append(
            {
                "observed_seed": seed,
                "primary_training_seed": seed,
                "repeat_training_seed": train,
                "primary_run_id": primary["run_id"],
                "repeat_run_id": row["run_id"],
                "primary_control_kl": primary["control_kl"],
                "repeat_control_kl": row["control_kl"],
                "repeat_minus_primary_control_kl": diff,
                "abs_repeat_minus_primary_control_kl": abs(diff),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def make_overlay(path: Path, primary_rows: list[dict[str, Any]]) -> None:
    ordered = sorted(primary_rows, key=lambda r: r["control_kl"])
    x = np.arange(len(ordered))
    control = np.asarray([r["control_kl"] for r in ordered])
    real = np.asarray([r["real_kl"] for r in ordered])
    seeds = [str(r["observed_seed"]) for r in ordered]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, control, marker="o", linewidth=1.5, label="MVN control")
    ax.plot(x, real, marker="o", linewidth=1.5, label="real GNK")
    for xi, c, rr in zip(x, control, real):
        ax.plot([xi, xi], [c, rr], color="0.75", linewidth=0.8, zorder=0)
    ax.set_xlabel("observed seed, sorted by control KL")
    ax.set_ylabel("kNN KL")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=90)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def classify_outcome(summary: dict[str, Any]) -> str:
    control = summary["median_control_kl"]
    real = summary["median_real_kl"]
    delta = summary["delta_s_bootstrap_ci"]["median"]
    noise = summary["mean_noise_floor_kl"]
    if control <= noise + 0.5 and real <= noise + 0.5:
        return "Both KLs are small, so the original problem was stale references."
    if control <= noise + 0.5 and real > noise + 0.5:
        return "MVN-control KL is near the noise floor while real-GNK KL is large."
    if abs(delta) <= 0.5:
        return "MVN-control KL median is close to real-GNK KL median."
    return "MVN-control KL remains meaningfully separated from the real-GNK KL."


def write_outcome(path: Path, summary: dict[str, Any], variance_rows: list[dict[str, Any]]) -> None:
    ci = summary["delta_s_bootstrap_ci"]
    variance_text = "not available"
    if variance_rows:
        diffs = np.asarray(
            [row["abs_repeat_minus_primary_control_kl"] for row in variance_rows],
            dtype=np.float64,
        )
        variance_text = f"median absolute repeat difference {np.median(diffs):.3f}"
    snippet = (
        f"At n=5000 and N=n^2, the asymptotic-MVN model-control median KL was "
        f"{summary['median_control_kl']:.3f}, versus real-GNK Gaussian-NPE median "
        f"{summary['median_real_kl']:.3f}. The paired median delta "
        f"(real minus control) was {ci['median']:.3f} with bootstrap 95% CI "
        f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}], and the mean NUTS self-split "
        f"noise floor was {summary['mean_noise_floor_kl']:.3f}."
    )
    text = f"""# GNK n=5000 model-control outcome

Run on {summary['created_at_utc']}. Aggregated {summary['n_cells']} asymptotic-MVN model-control cells at n=5000 and N=n^2 against the v3 gaussian-convention references.

Headline: model-control median KL = {summary['median_control_kl']:.3f}, real-GNK Gaussian-NPE median KL = {summary['median_real_kl']:.3f}, paired median delta_s = {ci['median']:.3f} with bootstrap 95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]. Mean NUTS self-split noise floor = {summary['mean_noise_floor_kl']:.3f}. Training-seed variability: {variance_text}.

Interpretation: {summary['outcome_classification']} The residual excess over the noise floor is {summary['median_control_excess_over_noise']:.3f} for the model control and {summary['median_real_excess_over_noise']:.3f} for the real-GNK cell. This diagnostic isolates the conditional estimator under the asymptotic-MVN simulator, but it does not by itself prove which part of training, architecture, or optimisation is responsible.

For the audit doc:

> {snippet}
"""
    path.write_text(text)


def check_outputs(targets: list[Path], force: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing aggregate outputs without --force:\n"
            + "\n".join(str(path) for path in existing)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--real-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--expected-cells", type=int, default=32)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_control_rows(args.control_root)
    if len(rows) != args.expected_cells:
        raise RuntimeError(
            f"expected {args.expected_cells} completed cells, found {len(rows)} under "
            f"{args.control_root}"
        )

    rows = add_real_and_reference_metrics(rows, args.real_root, args.v3_root)
    primary_rows = [
        row for row in rows if row["observed_seed"] == row["training_seed"]
    ]
    if len(primary_rows) != 30:
        raise RuntimeError(f"expected 30 primary cells, found {len(primary_rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        args.output_dir / "per_seed.csv",
        args.output_dir / "summary.json",
        args.output_dir / "training_seed_variance.csv",
        args.output_dir / "overlay.png",
        args.doc_path,
    ]
    check_outputs(targets, args.force)

    per_seed_fields = [
        "observed_seed",
        "training_seed",
        "run_id",
        "control_kl",
        "real_kl",
        "delta_s",
        "noise_floor_kl",
        "control_mmd",
        "real_mmd",
        "runtime_seconds",
        "divergences",
        "rhat_max",
        "ess_min",
    ]
    write_csv(args.output_dir / "per_seed.csv", rows, per_seed_fields)

    variance_rows = training_seed_variance(rows)
    variance_fields = [
        "observed_seed",
        "primary_training_seed",
        "repeat_training_seed",
        "primary_run_id",
        "repeat_run_id",
        "primary_control_kl",
        "repeat_control_kl",
        "repeat_minus_primary_control_kl",
        "abs_repeat_minus_primary_control_kl",
    ]
    write_csv(args.output_dir / "training_seed_variance.csv", variance_rows, variance_fields)

    delta_values = np.asarray([row["delta_s"] for row in primary_rows], dtype=np.float64)
    noise_values = np.asarray(
        [row["noise_floor_kl"] for row in primary_rows], dtype=np.float64
    )
    summary = {
        "created_at_utc": utc_now(),
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "n_cells": len(rows),
        "n_primary_cells": len(primary_rows),
        "median_control_kl": float(np.median([r["control_kl"] for r in primary_rows])),
        "median_real_kl": float(np.median([r["real_kl"] for r in primary_rows])),
        "mean_noise_floor_kl": float(np.mean(noise_values)),
        "median_noise_floor_kl": float(np.median(noise_values)),
        "delta_s_bootstrap_ci": bootstrap_median_ci(
            delta_values, args.bootstrap_draws, stable_int("model_control_delta_bootstrap")
        ),
        "input_paths": {
            "control_root": str(args.control_root),
            "real_root": str(args.real_root),
            "v3_root": str(args.v3_root),
        },
    }
    summary["median_control_excess_over_noise"] = (
        summary["median_control_kl"] - summary["mean_noise_floor_kl"]
    )
    summary["median_real_excess_over_noise"] = (
        summary["median_real_kl"] - summary["mean_noise_floor_kl"]
    )
    summary["outcome_classification"] = classify_outcome(summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    make_overlay(args.output_dir / "overlay.png", primary_rows)
    write_outcome(args.doc_path, summary, variance_rows)

    print(f"wrote {args.output_dir / 'per_seed.csv'}")
    print(f"wrote {args.output_dir / 'summary.json'}")
    print(f"wrote {args.output_dir / 'training_seed_variance.csv'}")
    print(f"wrote {args.output_dir / 'overlay.png'}")
    print(f"wrote {args.doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
