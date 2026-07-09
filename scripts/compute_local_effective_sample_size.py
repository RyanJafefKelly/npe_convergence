"""Compute local effective sample size in GNK summary space."""

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
import numpy as np
import numpyro.distributions as dist  # type: ignore
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.gnk import gnk, get_summaries_batches, ss_octile


PARAM_NAMES = ("A", "B", "g", "k")
TRUE_THETA = jnp.asarray([3.0, 1.0, 2.0, 0.5])
DEFAULT_BOX = {
    "A": (2.5, 3.5),
    "B": (0.6, 1.4),
    "g": (1.4, 2.6),
    "k": (0.2, 0.8),
}
DEFAULT_RESTRICTED_ROOT = REPO_ROOT / "res" / "gnk_restricted_prior"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks" / "plots" / "gnk_restricted_n_scaling"
DEFAULT_DOC_PATH = (
    REPO_ROOT
    / "docs"
    / "meeting_2026_05_18"
    / "gnk_local_effective_sample_size_outcome.md"
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


def reconstruct_x_obs(n_obs: int, seed: int, convention: str) -> np.ndarray:
    if convention == "flow":
        z_key = random.key(seed)
    elif convention == "gaussian":
        _, z_key = random.split(random.key(seed))
    else:
        raise ValueError(f"unknown convention: {convention}")
    z = random.normal(z_key, shape=(n_obs,))
    x_raw = gnk(z, *TRUE_THETA)
    return np.asarray(jnp.squeeze(ss_octile(jnp.atleast_2d(x_raw))), dtype=np.float64)


def parse_box(box: dict[str, list[float]] | dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    lows = np.asarray([box[name][0] for name in PARAM_NAMES], dtype=np.float64)
    highs = np.asarray([box[name][1] for name in PARAM_NAMES], dtype=np.float64)
    return lows, highs


def read_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    payload["_summary_path"] = str(path)
    payload["_run_dir"] = str(path.parent)
    payload["training_seed"] = int(payload.get("training_seed", payload["seed"]))
    payload["created_at_utc"] = payload.get("created_at_utc", payload.get("created_at"))
    return payload


def collect_restricted(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("restricted_n_obs_5000_seed_50_conv_gaussian_n_sims_*/summary.json")):
        summary = read_summary(path)
        if int(summary["n_obs"]) == 5000 and int(summary["seed"]) == 50:
            rows.append(summary)
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["n_sims"]), int(row["training_seed"]))
        if key not in latest or str(row["created_at_utc"]) > str(latest[key]["created_at_utc"]):
            latest[key] = row
    return [latest[key] for key in sorted(latest)]


def load_saved_training_summaries(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    with np.load(path) as data:
        summaries = np.asarray(data["summaries"], dtype=np.float64)
        if summaries.shape[0] == 7 and summaries.ndim == 2:
            summaries = summaries.T
        mean = np.asarray(data["summary_mean"], dtype=np.float64) if "summary_mean" in data else None
        sd_name = "summary_sd" if "summary_sd" in data else "summary_std"
        sd = np.asarray(data[sd_name], dtype=np.float64) if sd_name in data else None
        x_obs = np.asarray(data["x_obs"], dtype=np.float64) if "x_obs" in data else None
    return summaries, mean, sd, x_obs


def regenerate_restricted_summaries(summary: dict[str, Any], batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_obs = int(summary["n_obs"])
    seed = int(summary["seed"])
    n_sims = int(summary["n_sims"])
    convention = summary["convention"]
    box = summary.get("box", DEFAULT_BOX)
    lows, highs = parse_box(box)

    rng = random.key(seed)
    rng, sim_key = random.split(rng)
    u_samples = random.uniform(
        sim_key, shape=(n_sims, 4), minval=jnp.zeros(4), maxval=jnp.ones(4)
    )
    thetas = jnp.asarray(lows) + u_samples * jnp.asarray(highs - lows)
    A_s, B_s, g_s, k_s = thetas.T
    rng, summary_key = random.split(rng)
    x_sims = get_summaries_batches(
        summary_key,
        A_s,
        B_s,
        g_s,
        k_s,
        n_obs=n_obs,
        n_sims=n_sims,
        batch_size=min(batch_size, n_sims),
    )
    summaries = np.asarray(x_sims.T, dtype=np.float64)
    mean = summaries.mean(axis=0)
    sd = summaries.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    x_obs = reconstruct_x_obs(n_obs, seed, convention)
    return summaries, mean, sd, x_obs


def regenerate_full_prior_summaries(
    n_obs: int, seed: int, n_sims: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = random.key(seed)
    key, _ = random.split(key)
    key, theta_key = random.split(key)
    thetas = dist.Uniform(1e-6, 10 - 1e-6).sample(theta_key, (n_sims, 4))
    A_s, B_s, g_s, k_s = thetas.T
    key, summary_key = random.split(key)
    x_sims = get_summaries_batches(
        summary_key,
        A_s,
        B_s,
        g_s,
        k_s,
        n_obs=n_obs,
        n_sims=n_sims,
        batch_size=min(batch_size, n_sims),
    )
    summaries = np.asarray(x_sims.T, dtype=np.float64)
    mean = summaries.mean(axis=0)
    sd = summaries.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    x_obs = reconstruct_x_obs(n_obs, seed, "gaussian")
    return summaries, mean, sd, x_obs


def density_stats(
    *,
    label: str,
    n_obs: int,
    seed: int,
    training_seed: int | None,
    n_sims: int,
    source: str,
    status: str,
    summaries: np.ndarray | None,
    summary_mean: np.ndarray | None,
    summary_sd: np.ndarray | None,
    x_obs: np.ndarray | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": label,
        "n_obs": n_obs,
        "seed": seed,
        "training_seed": training_seed,
        "n_sims": n_sims,
        "source": source,
        "status": status,
        "count_chi2_7_0p5": None,
        "count_chi2_7_0p9": None,
        "knn_distance_100": None,
        "knn_distance_1000": None,
        "knn_distance_10000": None,
    }
    if summaries is None:
        return row
    if summary_mean is None:
        summary_mean = summaries.mean(axis=0)
    if summary_sd is None:
        summary_sd = summaries.std(axis=0)
    summary_sd = np.where(summary_sd == 0, 1.0, summary_sd)
    if x_obs is None:
        x_obs = reconstruct_x_obs(n_obs, seed, "gaussian")
    diffs = (summaries - summary_mean) / summary_sd - (x_obs - summary_mean) / summary_sd
    d2 = np.sum(diffs * diffs, axis=1)
    distances = np.sqrt(d2)
    row["count_chi2_7_0p5"] = int(np.count_nonzero(d2 < chi2.ppf(0.5, 7)))
    row["count_chi2_7_0p9"] = int(np.count_nonzero(d2 < chi2.ppf(0.9, 7)))
    for k in (100, 1000, 10000):
        if distances.shape[0] >= k:
            row[f"knn_distance_{k}"] = float(np.partition(distances, k - 1)[k - 1])
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "n_obs",
        "seed",
        "training_seed",
        "n_sims",
        "source",
        "status",
        "count_chi2_7_0p5",
        "count_chi2_7_0p9",
        "knn_distance_100",
        "knn_distance_1000",
        "knn_distance_10000",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [f"# {title}", ""]
    lines.append(f"Generated {utc_now()}. Distances use the summary standardisation from each training set.")
    lines.append("")
    lines.append("| label | N | train seed | status | count D2<chi2_0.5 | count D2<chi2_0.9 | d100 | d1000 | d10000 |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in rows:
        def fmt(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["n_sims"]),
                    fmt(row["training_seed"]),
                    str(row["status"]),
                    fmt(row["count_chi2_7_0p5"]),
                    fmt(row["count_chi2_7_0p9"]),
                    fmt(row["knn_distance_100"]),
                    fmt(row["knn_distance_1000"]),
                    fmt(row["knn_distance_10000"]),
                ]
            )
            + " |"
        )
    lines.append("")
    unavailable = [r for r in rows if r["status"] != "ok"]
    if unavailable:
        lines.append(
            "Rows marked unavailable did not have saved training summaries and were not regenerated in this run."
        )
    path.write_text("\n".join(lines) + "\n")


def write_outcome(path: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    full = next((r for r in rows if r["label"] == "full_prior"), None)
    restricted = [r for r in ok_rows if r["label"] == "restricted_prior"]
    best = ""
    if restricted:
        latest = max(restricted, key=lambda r: int(r["n_sims"]))
        best = (
            f"At the largest restricted-prior N available ({latest['n_sims']}), "
            f"count D2<chi2_7(0.5) was {latest['count_chi2_7_0p5']} and "
            f"the 1000th-nearest distance was {latest['knn_distance_1000']:.4f}."
        )
    full_text = "Full-prior local density was unavailable because no saved 25M training summaries were found."
    if full and full["status"] == "ok":
        full_text = (
            f"Full-prior count D2<chi2_7(0.5) was {full['count_chi2_7_0p5']} "
            f"and count D2<chi2_7(0.9) was {full['count_chi2_7_0p9']}."
        )
    snippet = f"{full_text} {best}".strip()
    text = f"""# GNK local effective sample-size outcome

Run on {utc_now()}. Computed local summary-space density diagnostics for the GNK n=5000, seed=50 training sets that were available or explicitly regenerated.

Headline: {full_text} {best}

Interpretation: The restricted-prior rows quantify local information near the canonical observed summary under the same diagonal summary standardisation used by Gaussian-NPE. If the full-prior row is unavailable, this note supports the restricted-prior side of the comparison but cannot by itself quantify the full-prior density deficit.

For the audit doc:

> {snippet}
"""
    path.write_text(text)


def check_outputs(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --force:\n"
            + "\n".join(str(path) for path in existing)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restricted-root", type=Path, default=DEFAULT_RESTRICTED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--full-training-summaries", type=Path, default=None)
    parser.add_argument("--regenerate-full-prior", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "local_density.csv"
    md_path = args.output_dir / "local_density.md"
    check_outputs([csv_path, md_path, args.doc_path], args.force)

    rows: list[dict[str, Any]] = []
    if args.full_training_summaries and args.full_training_summaries.exists():
        summaries, mean, sd, x_obs = load_saved_training_summaries(args.full_training_summaries)
        rows.append(
            density_stats(
                label="full_prior",
                n_obs=5000,
                seed=50,
                training_seed=50,
                n_sims=summaries.shape[0],
                source=str(args.full_training_summaries),
                status="ok",
                summaries=summaries,
                summary_mean=mean,
                summary_sd=sd,
                x_obs=x_obs,
            )
        )
    elif args.regenerate_full_prior:
        summaries, mean, sd, x_obs = regenerate_full_prior_summaries(
            5000, 50, 25_000_000, args.batch_size
        )
        rows.append(
            density_stats(
                label="full_prior",
                n_obs=5000,
                seed=50,
                training_seed=50,
                n_sims=25_000_000,
                source="regenerated_full_prior",
                status="ok",
                summaries=summaries,
                summary_mean=mean,
                summary_sd=sd,
                x_obs=x_obs,
            )
        )
    else:
        rows.append(
            density_stats(
                label="full_prior",
                n_obs=5000,
                seed=50,
                training_seed=50,
                n_sims=25_000_000,
                source="missing_saved_training_summaries",
                status="unavailable",
                summaries=None,
                summary_mean=None,
                summary_sd=None,
                x_obs=None,
            )
        )

    for summary in collect_restricted(args.restricted_root):
        run_dir = Path(summary["_run_dir"])
        saved = run_dir / "training_summaries.npz"
        if saved.exists():
            summaries, mean, sd, x_obs = load_saved_training_summaries(saved)
            source = str(saved)
        else:
            summaries, mean, sd, x_obs = regenerate_restricted_summaries(
                summary, args.batch_size
            )
            source = "regenerated_restricted_prior"
        rows.append(
            density_stats(
                label="restricted_prior",
                n_obs=int(summary["n_obs"]),
                seed=int(summary["seed"]),
                training_seed=int(summary["training_seed"]),
                n_sims=int(summary["n_sims"]),
                source=source,
                status="ok",
                summaries=summaries,
                summary_mean=mean,
                summary_sd=sd,
                x_obs=x_obs,
            )
        )

    write_csv(csv_path, rows)
    write_markdown(md_path, rows, "GNK Local Density")
    write_outcome(args.doc_path, rows)
    metadata = {
        "created_at_utc": utc_now(),
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
    }
    (args.output_dir / "local_density_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {args.doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
