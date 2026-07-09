"""Refresh MA(2) b0 flow KLs against the current joint-covariance reference.

This script is intentionally read-only with respect to the stored
``res/ma2_b0/npe_n_obs_*`` experiment directories. It writes regenerated NUTS
references to seed-specific ``current_reference_refresh_*`` directories and
emits the per-seed/summary artifacts requested by T27.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro  # type: ignore
from numpyro.infer import MCMC, NUTS  # type: ignore

from npe_convergence.examples.ma2 import get_summaries_batches, numpyro_model_b0
from npe_convergence.metrics import kullback_leibler


TRUE_PARAMS = jnp.array([0.6, 0.2])
DEFAULT_N_OBS = 1_000
DEFAULT_N_SIMS = 1_000_000
DEFAULT_ROOT = Path("res/ma2_b0")
DEFAULT_PER_SEED_CSV = Path(
    "notebooks/meeting_2026_05_18/data/ma2_b0_flow_current_reference_per_seed.csv"
)
DEFAULT_SUMMARY_CSV = Path(
    "notebooks/meeting_2026_05_18/data/ma2_b0_flow_current_reference_summary.csv"
)
DEFAULT_OUTCOME_MD = Path(
    "docs/meeting_2026_05_18/codex_outcome_T27_ma2_b0_current_reference_refresh.md"
)


SUMMARY_METRICS = [
    "stored_flow_kl",
    "current_flow_kl",
    "current_minus_stored_flow_kl",
    "reference_mean_shift_l2",
    "reference_mean_shift_t1",
    "reference_mean_shift_t2",
]


@dataclass(frozen=True)
class SeedPaths:
    seed: int
    flow_dir: Path
    refresh_dir: Path

    @property
    def reference_path(self) -> Path:
        return self.refresh_dir / "nuts_current_reference.pkl"


def current_x_obs(seed: int, n_obs: int) -> jnp.ndarray:
    """Generate observed summaries using the current MA(2) code path."""
    key = random.key(seed)
    _, subkey = random.split(key)
    x_obs = get_summaries_batches(
        subkey,
        jnp.atleast_1d(TRUE_PARAMS[0]),
        jnp.atleast_1d(TRUE_PARAMS[1]),
        n_obs,
        1,
        1,
    )
    return jnp.squeeze(x_obs)


def load_pickle(path: Path) -> jnp.ndarray:
    with path.open("rb") as f:
        return pkl.load(f)


def load_float(path: Path) -> float:
    text = path.read_text().strip()
    return float(text)


def finite_float(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        value = load_float(path)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def discover_flow_seeds(root: Path, n_obs: int, n_sims: int) -> list[int]:
    pattern = f"npe_n_obs_{n_obs}_n_sims_{n_sims}_seed_*"
    seeds: list[int] = []
    for flow_dir in root.glob(pattern):
        try:
            seed = int(flow_dir.name.rsplit("_seed_", 1)[1])
        except (IndexError, ValueError):
            continue
        if finite_float(flow_dir / "kl.txt") is None:
            continue
        if not (flow_dir / "posterior_samples.pkl").exists():
            continue
        if not (flow_dir / "true_posterior_samples.pkl").exists():
            continue
        seeds.append(seed)
    return sorted(seeds)


def run_or_load_reference(
    x_obs: jnp.ndarray,
    n_obs: int,
    output_path: Path,
    num_warmup: int,
    num_samples: int,
    thinning: int,
    num_chains: int,
    reuse: bool,
    progress_bar: bool,
) -> tuple[jnp.ndarray, bool]:
    if reuse and output_path.exists():
        return load_pickle(output_path), True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nuts_kernel = NUTS(numpyro_model_b0)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples * thinning,
        thinning=thinning,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(
        random.key(1),
        obs=x_obs,
        init_params={"t1": 0.0, "t2": 0.0},
        n_obs=n_obs,
    )
    samples = mcmc.get_samples()
    reference = jnp.column_stack([samples["t1"], samples["t2"]])
    with output_path.open("wb") as f:
        pkl.dump(reference, f)
    return reference, False


def seed_paths(root: Path, n_obs: int, n_sims: int, seed: int) -> SeedPaths:
    return SeedPaths(
        seed=seed,
        flow_dir=root / f"npe_n_obs_{n_obs}_n_sims_{n_sims}_seed_{seed}",
        refresh_dir=root / f"current_reference_refresh_n_obs_{n_obs}_seed_{seed}",
    )


def json_array(values: np.ndarray) -> str:
    return json.dumps([float(x) for x in values])


def row_for_seed(args: argparse.Namespace, seed: int) -> dict[str, object]:
    paths = seed_paths(args.root, args.n_obs, args.n_sims, seed)
    x_obs = current_x_obs(seed, args.n_obs)
    paths.refresh_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.refresh_dir / "x_obs_current.npy", np.asarray(x_obs))

    current_ref, reused = run_or_load_reference(
        x_obs=x_obs,
        n_obs=args.n_obs,
        output_path=paths.reference_path,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        thinning=args.thinning,
        num_chains=args.num_chains,
        reuse=args.reuse,
        progress_bar=args.progress,
    )
    flow_samples = load_pickle(paths.flow_dir / "posterior_samples.pkl")
    stored_ref = load_pickle(paths.flow_dir / "true_posterior_samples.pkl")

    stored_flow_kl = load_float(paths.flow_dir / "kl.txt")
    current_flow_kl = float(kullback_leibler(current_ref, flow_samples))

    x_obs_np = np.asarray(x_obs, dtype=float)
    stored_ref_np = np.asarray(stored_ref, dtype=float)
    current_ref_np = np.asarray(current_ref, dtype=float)
    flow_np = np.asarray(flow_samples, dtype=float)

    stored_ref_mean = stored_ref_np.mean(axis=0)
    current_ref_mean = current_ref_np.mean(axis=0)
    flow_mean = flow_np.mean(axis=0)
    ref_mean_shift = current_ref_mean - stored_ref_mean

    row: dict[str, object] = {
        "seed": seed,
        "x_obs_current": json_array(x_obs_np),
        "x_obs_current_gamma0": float(x_obs_np[0]),
        "x_obs_current_gamma1": float(x_obs_np[1]),
        "x_obs_current_gamma2": float(x_obs_np[2]),
        "stored_reference_mean": json_array(stored_ref_mean),
        "stored_reference_mean_t1": float(stored_ref_mean[0]),
        "stored_reference_mean_t2": float(stored_ref_mean[1]),
        "current_reference_mean": json_array(current_ref_mean),
        "current_reference_mean_t1": float(current_ref_mean[0]),
        "current_reference_mean_t2": float(current_ref_mean[1]),
        "flow_mean": json_array(flow_mean),
        "flow_mean_t1": float(flow_mean[0]),
        "flow_mean_t2": float(flow_mean[1]),
        "reference_mean_shift": json_array(ref_mean_shift),
        "reference_mean_shift_t1": float(ref_mean_shift[0]),
        "reference_mean_shift_t2": float(ref_mean_shift[1]),
        "reference_mean_shift_l2": float(np.linalg.norm(ref_mean_shift)),
        "stored_flow_kl": float(stored_flow_kl),
        "current_flow_kl": current_flow_kl,
        "current_minus_stored_flow_kl": current_flow_kl - float(stored_flow_kl),
        "flow_posterior_sample_count": int(flow_np.shape[0]),
        "stored_reference_sample_count": int(stored_ref_np.shape[0]),
        "current_reference_sample_count": int(current_ref_np.shape[0]),
        "flow_dir": str(paths.flow_dir),
        "current_reference_path": str(paths.reference_path),
        "reference_reused": bool(reused),
        "nuts_num_warmup": int(args.num_warmup),
        "nuts_num_samples": int(args.num_samples),
        "nuts_thinning": int(args.thinning),
        "nuts_num_chains": int(args.num_chains),
        "status": "complete",
    }
    with (paths.refresh_dir / "refresh_metrics.json").open("w") as f:
        json.dump(row, f, indent=2)
        f.write("\n")
    return row


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for metric in SUMMARY_METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        summary.append(
            {
                "metric": metric,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
            }
        )
    return summary


def write_summary_csv(path: Path, summary: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "n", "mean", "std", "min", "q25", "median", "q75", "max"],
        )
        writer.writeheader()
        writer.writerows(summary)


def value_from_summary(summary: list[dict[str, object]], metric: str, stat: str) -> float:
    for row in summary:
        if row["metric"] == metric:
            return float(row[stat])
    raise KeyError((metric, stat))


def write_outcome_markdown(
    path: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    stored_median = value_from_summary(summary, "stored_flow_kl", "median")
    current_median = value_from_summary(summary, "current_flow_kl", "median")
    current_q75 = value_from_summary(summary, "current_flow_kl", "q75")
    current_max = value_from_summary(summary, "current_flow_kl", "max")
    shift_median = value_from_summary(summary, "reference_mean_shift_l2", "median")

    if current_median < 0.1 and current_median < stored_median / 5:
        conclusion = "The current-reference flow median collapses relative to the stale stored-reference result."
    elif current_median >= 0.5:
        conclusion = "The current-reference flow median remains large."
    else:
        conclusion = "The current-reference flow median is reduced but not negligible."

    lines = [
        "# T27 MA(2) b0 Current-Reference Refresh Outcome",
        "",
        "## Summary",
        "",
        (
            f"Regenerated current joint-covariance NUTS references for {len(rows)} "
            f"finite stored flow seeds at `n_obs={args.n_obs}`, `n_sims={args.n_sims}`."
        ),
        "",
        conclusion,
        "",
        (
            f"Stored flow KL median: `{stored_median:.6g}`. Current-reference flow KL "
            f"median: `{current_median:.6g}`. Current-reference q75: "
            f"`{current_q75:.6g}`; max: `{current_max:.6g}`."
        ),
        "",
        (
            f"Median stored-to-current reference mean shift, measured as L2 distance, "
            f"is `{shift_median:.6g}`."
        ),
        "",
        "## NUTS Convention",
        "",
        (
            f"Used one current `numpyro_model_b0` joint-covariance reference per seed "
            f"with `{args.num_warmup}` warmup draws, `{args.num_samples * args.thinning}` "
            f"post-warmup draws, thinning `{args.thinning}`, `{args.num_samples}` saved "
            f"samples, and `{args.num_chains}` chain(s)."
        ),
        "",
        "## Output Files",
        "",
        f"- `{args.per_seed_csv}`",
        f"- `{args.summary_csv}`",
        (
            f"- `res/ma2_b0/current_reference_refresh_n_obs_{args.n_obs}_seed_*/"
            "nuts_current_reference.pkl`"
        ),
        "",
        "## Summary Table",
        "",
        "| metric | n | mean | std | min | q25 | median | q75 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {metric} | {n} | {mean:.6g} | {std:.6g} | {min:.6g} | "
            "{q25:.6g} | {median:.6g} | {q75:.6g} | {max:.6g} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Integrity Notes",
            "",
            (
                "The stored `res/ma2_b0/npe_n_obs_*` directories were used only as "
                "inputs for `posterior_samples.pkl`, `true_posterior_samples.pkl`, "
                "and `kl.txt`; refreshed references and metrics were written to "
                "separate seed-specific directories."
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--n_obs", type=int, default=DEFAULT_N_OBS)
    parser.add_argument("--n_sims", type=int, default=DEFAULT_N_SIMS)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--num_warmup", type=int, default=2_000)
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--thinning", type=int, default=10)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--per_seed_csv", type=Path, default=DEFAULT_PER_SEED_CSV)
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--outcome_md", type=Path, default=DEFAULT_OUTCOME_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numpyro.set_host_device_count(args.num_chains)

    seeds = args.seeds if args.seeds else discover_flow_seeds(args.root, args.n_obs, args.n_sims)
    if not seeds:
        raise RuntimeError("No finite stored flow seeds found")

    rows: list[dict[str, object]] = []
    for index, seed in enumerate(seeds, start=1):
        print(f"[{index}/{len(seeds)}] refreshing seed {seed}", flush=True)
        row = row_for_seed(args, seed)
        rows.append(row)
        print(
            (
                f"seed {seed}: stored KL={row['stored_flow_kl']:.6g}, "
                f"current KL={row['current_flow_kl']:.6g}, "
                f"ref shift={row['reference_mean_shift_l2']:.6g}"
            ),
            flush=True,
        )

    summary = make_summary(rows)
    write_csv(args.per_seed_csv, rows)
    write_summary_csv(args.summary_csv, summary)
    write_outcome_markdown(args.outcome_md, rows, summary, args)

    print(f"Wrote {args.per_seed_csv}", flush=True)
    print(f"Wrote {args.summary_csv}", flush=True)
    print(f"Wrote {args.outcome_md}", flush=True)


if __name__ == "__main__":
    main()
