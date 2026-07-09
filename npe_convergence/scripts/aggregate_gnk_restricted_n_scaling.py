"""Aggregate the GNK restricted-prior N-scaling diagnostic."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.metrics import kullback_leibler


PARAM_NAMES = ("A", "B", "g", "k")
DEFAULT_ROOT = REPO_ROOT / "res" / "gnk_restricted_prior"
DEFAULT_V3_ROOT = REPO_ROOT / "res" / "gnk_v3_refs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "plots" / "gnk_restricted_n_scaling"
DEFAULT_DOC_PATH = (
    REPO_ROOT
    / "docs"
    / "meeting_2026_05_18"
    / "gnk_restricted_n_scaling_outcome.md"
)
EXPECTED_CELLS = {
    (250_000, 50),
    (1_000_000, 50),
    (4_000_000, 50),
    (1_000_000, 7),
    (1_000_000, 23),
}
DEFAULT_BOX = {
    "A": (2.5, 3.5),
    "B": (0.6, 1.4),
    "g": (1.4, 2.6),
    "k": (0.2, 0.8),
}


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


def read_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    payload["_summary_path"] = str(path)
    payload["_run_dir"] = str(path.parent)
    payload["training_seed"] = int(payload.get("training_seed", payload["seed"]))
    payload["created_at_utc"] = payload.get("created_at_utc", payload.get("created_at"))
    return payload


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("restricted_n_obs_5000_seed_50_conv_gaussian_n_sims_*/summary.json")):
        summary = read_summary(path)
        if int(summary["n_obs"]) != 5000:
            continue
        if int(summary["seed"]) != 50 or summary["convention"] != "gaussian":
            continue
        shifts = summary["median_shifts_in_nuts_sds"]
        ratios = summary["sd_ratios_npe_over_nuts"]
        rows.append(
            {
                "N": int(summary["n_sims"]),
                "training_seed": int(summary["training_seed"]),
                "kl_vs_truncated_nuts": float(summary["kl_npe_vs_truncated_nuts"]),
                "median_shift_max": float(max(abs(float(v)) for v in shifts.values())),
                "sd_ratio_min": float(min(float(v) for v in ratios.values())),
                "sd_ratio_max": float(max(float(v) for v in ratios.values())),
                "created_at_utc": summary["created_at_utc"],
                "summary_path": summary["_summary_path"],
            }
        )
    return rows


def choose_latest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["N"], row["training_seed"])
        if key not in latest or str(row["created_at_utc"]) > str(latest[key]["created_at_utc"]):
            latest[key] = row
    return [latest[key] for key in sorted(latest)]


def deduplicate(samples: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(samples, dtype=np.float64), axis=0)


def deterministic_subsample(samples: np.ndarray, n: int, seed: int) -> np.ndarray:
    if samples.shape[0] <= n:
        return samples
    key = random.key(seed)
    idx = np.asarray(random.permutation(key, samples.shape[0])[:n])
    return samples[idx]


def compute_noise_floor(v3_root: Path) -> float:
    path = v3_root / "nuts_n_obs_5000_seed_50_conv_gaussian.pkl"
    with path.open("rb") as f:
        ref = pkl.load(f)
    grouped = np.asarray(ref["samples"], dtype=np.float64)
    lows = np.asarray([DEFAULT_BOX[name][0] for name in PARAM_NAMES])
    highs = np.asarray([DEFAULT_BOX[name][1] for name in PARAM_NAMES])
    flat = grouped.reshape(-1, grouped.shape[-1])
    mask = np.all((flat >= lows) & (flat <= highs), axis=1)
    unique = deduplicate(flat[mask])
    rng = np.random.default_rng(stable_int("restricted_noise_floor", 50))
    perm = rng.permutation(unique.shape[0])
    half = unique.shape[0] // 2
    left = unique[perm[:half]]
    right = unique[perm[half : 2 * half]]
    left = deterministic_subsample(left, 2000, stable_int("left", 50))
    right = deterministic_subsample(right, 2000, stable_int("right", 50))
    return float(kullback_leibler(jnp.asarray(left), jnp.asarray(right)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "N",
        "training_seed",
        "kl_vs_truncated_nuts",
        "median_shift_max",
        "sd_ratio_min",
        "sd_ratio_max",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def make_plot(path: Path, rows: list[dict[str, Any]], noise_floor: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    primary = sorted([r for r in rows if r["training_seed"] == 50], key=lambda r: r["N"])
    if primary:
        ax.plot(
            [r["N"] for r in primary],
            [r["kl_vs_truncated_nuts"] for r in primary],
            marker="o",
            linewidth=1.8,
            label="training seed 50",
        )
    repeats = [r for r in rows if r["N"] == 1_000_000]
    if repeats:
        xs = np.asarray([r["N"] for r in repeats], dtype=np.float64)
        ys = np.asarray([r["kl_vs_truncated_nuts"] for r in repeats], dtype=np.float64)
        ax.errorbar(
            [1_000_000],
            [float(np.median(ys))],
            yerr=[[float(np.median(ys) - np.min(ys))], [float(np.max(ys) - np.median(ys))]],
            marker="s",
            capsize=4,
            color="black",
            label="1M training-seed spread",
        )
        ax.scatter(xs, ys, color="black", s=28)
    ax.axhline(max(noise_floor, 1e-6), color="0.4", linestyle=":", label="NUTS self-split")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("kNN KL versus truncated NUTS")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def interpretation(rows: list[dict[str, Any]], noise_floor: float) -> str:
    primary = {row["N"]: row for row in rows if row["training_seed"] == 50}
    if 4_000_000 not in primary or 1_000_000 not in primary:
        return "The expected curve is incomplete, so the decision rule is not applied."
    kl_1m = primary[1_000_000]["kl_vs_truncated_nuts"]
    kl_4m = primary[4_000_000]["kl_vs_truncated_nuts"]
    if kl_4m <= noise_floor + 0.25:
        return "KL plateaus at or close to the noise floor, so broad-prior amortisation is supported as the main mechanism."
    if kl_4m > 0.5 and abs(kl_4m - kl_1m) / max(kl_1m, 1e-12) < 0.1:
        return "KL plateaus well above the noise floor, so architecture or optimisation contributes to the gap."
    if kl_4m < kl_1m:
        return "KL is still falling at N=4M, so this budget does not close the mechanism question."
    return "KL remains above the noise floor without a clear monotone pattern, so the mechanism claim should stay cautious."


def write_outcome(
    path: Path, rows: list[dict[str, Any]], noise_floor: float, decision: str
) -> None:
    primary = {row["N"]: row for row in rows if row["training_seed"] == 50}
    values = ", ".join(
        f"N={n}: KL={primary[n]['kl_vs_truncated_nuts']:.3f}"
        for n in sorted(primary)
    )
    repeats = [r["kl_vs_truncated_nuts"] for r in rows if r["N"] == 1_000_000]
    repeat_text = "not available"
    if repeats:
        repeat_text = f"range {min(repeats):.3f} to {max(repeats):.3f}"
    snippet = (
        f"Restricted-prior N-scaling at n=5000, seed=50 gave {values}; "
        f"the NUTS self-split noise floor was {noise_floor:.3f}. "
        f"At N=1M, training-seed KL spread was {repeat_text}. {decision}"
    )
    text = f"""# GNK restricted-prior N-scaling outcome

Run on {utc_now()}. Aggregated restricted-prior Gaussian-NPE cells at n=5000, seed=50, gaussian convention.

Headline: {values if values else "no primary-seed curve values found"}. The NUTS self-split noise floor is {noise_floor:.3f}. Training-seed variability at N=1M is {repeat_text}.

Interpretation: {decision} This diagnostic uses the fixed local box around the truth and compares against the v3 canonical NUTS reference truncated to that box. It does not rule out larger-N changes beyond the current budget.

For the audit doc:

> {snippet}
"""
    path.write_text(text)


def check_outputs(targets: list[Path], force: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --force:\n"
            + "\n".join(str(path) for path in existing)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = choose_latest(collect_rows(args.root))
    present = {(row["N"], row["training_seed"]) for row in rows}
    missing = sorted(EXPECTED_CELLS - present)
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"missing restricted-prior cells: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        args.output_dir / "curve.csv",
        args.output_dir / "curve.png",
        args.doc_path,
    ]
    check_outputs(targets, args.force)

    raw_noise_floor = compute_noise_floor(args.v3_root)
    noise_floor = max(0.0, raw_noise_floor)
    write_csv(args.output_dir / "curve.csv", rows)
    make_plot(args.output_dir / "curve.png", rows, noise_floor)
    decision = interpretation(rows, noise_floor)
    write_outcome(args.doc_path, rows, noise_floor, decision)

    metadata = {
        "created_at_utc": utc_now(),
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "noise_floor_kl": noise_floor,
        "noise_floor_kl_raw": raw_noise_floor,
        "missing_expected_cells": [list(cell) for cell in missing],
        "input_root": str(args.root),
        "v3_root": str(args.v3_root),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote {args.output_dir / 'curve.csv'}")
    print(f"wrote {args.output_dir / 'curve.png'}")
    print(f"wrote {args.output_dir / 'summary.json'}")
    print(f"wrote {args.doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
