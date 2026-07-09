#!/usr/bin/env python
"""In-process driver for the GNK canonical NUTS reference rerun.

This is the same workload as drive_gnk_nuts_refresh.py but runs every
cell in one python process. Subprocess startup was dominating the
per-cell wall time when each cell was a separate invocation; running
all cells in-process keeps the JAX cache warm and amortises JIT
compilation.

The script imports run_gnk_nuts_refresh.py once and calls
run_nuts_canonical / reconstruct_x_obs_float32 / etc. directly. It
also reuses environment_record once and writes the same fingerprint
dicts to res/gnk_v3_refs/.

Run with JAX_ENABLE_X64=1 set in the environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle as pkl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
_DEFAULT_NUM_CHAINS = 5
os.environ.setdefault(
    "XLA_FLAGS", f"--xla_force_host_platform_device_count={_DEFAULT_NUM_CHAINS}"
)

import jax

jax.config.update("jax_enable_x64", True)
assert jax.config.jax_enable_x64

import jax.numpy as jnp
import numpy as np
import numpyro  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

numpyro.set_host_device_count(_DEFAULT_NUM_CHAINS)

from npe_convergence.scripts.run_gnk_nuts_refresh import (
    DENSITY_VERSION,
    PARAM_NAMES,
    diagnostics_from_samples,
    environment_record,
    output_path_for,
    reconstruct_x_obs_float32,
    run_nuts_canonical,
    stable_int,
    utc_now,
)


def write_fingerprint(
    out_path: Path,
    *,
    n_obs: int,
    seed: int,
    convention: str,
    sampler_seed: int,
    samples_grouped: np.ndarray,
    diagnostics: dict,
    extra_fields: dict,
    runtime_seconds: float,
    x_obs_f32: jnp.ndarray,
    x_obs_f64: jnp.ndarray,
    sha: str,
    num_chains: int,
    num_warmup: int,
    num_samples: int,
    target_accept_prob: float,
    env: dict,
) -> None:
    fingerprint: dict = {
        "samples": np.asarray(samples_grouped, dtype=np.float64),
        "param_order": list(PARAM_NAMES),
        "x_obs_summary_unstandardised_float32": np.asarray(x_obs_f32),
        "x_obs_summary_unstandardised_float64": np.asarray(x_obs_f64),
        "x_obs_summary_unstandardised_sha256": sha,
        "n_obs": n_obs,
        "data_seed": seed,
        "convention": convention,
        "sampler_seed": sampler_seed,
        "density_version": DENSITY_VERSION,
        "data_epoch": "post_random_key_2024_11",
        "jitter": 1e-6,
        "c": 0.8,
        "quantile_method": "type-7",
        "num_chains": num_chains,
        "num_warmup": num_warmup,
        "num_samples_per_chain": num_samples,
        "thinning": 1,
        "target_accept_prob": target_accept_prob,
        "mass_matrix": "dense",
        "runtime_seconds": runtime_seconds,
        "utc_timestamp": utc_now(),
        "environment": env,
        "diagnostics": diagnostics,
    }
    for name, arr in extra_fields.items():
        fingerprint[f"extra_{name}"] = np.asarray(arr)
    with out_path.open("wb") as f:
        pkl.dump(fingerprint, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "meeting_2026_05_18"
            / "gnk_nuts_refresh_plan"
            / "gnk_nuts_refresh_manifest.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "meeting_2026_05_18"
            / "gnk_nuts_refresh_plan"
            / "gnk_nuts_refresh_run_log.json"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "res" / "gnk_v3_refs"
    )
    parser.add_argument("--num-chains", type=int, default=5)
    parser.add_argument("--num-warmup", type=int, default=10_000)
    parser.add_argument("--num-samples", type=int, default=2_000)
    parser.add_argument("--target-accept-prob", type=float, default=0.9)
    parser.add_argument(
        "--filter-n-obs",
        type=str,
        default="",
        help="Comma-separated n_obs to keep (default: all).",
    )
    parser.add_argument(
        "--filter-convention",
        type=str,
        default="",
        help="Convention filter (default: both).",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells whose output already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open()))
    n_total = len(rows)

    filt_n_obs = (
        {int(x) for x in args.filter_n_obs.split(",") if x.strip()}
        if args.filter_n_obs.strip()
        else None
    )
    filt_conv = args.filter_convention.strip() or None

    args.output_root.mkdir(parents=True, exist_ok=True)
    env = environment_record()
    print(
        f"In-process rerun: {n_total} rows in manifest, "
        f"x64={env['jax_x64_enabled']}, devices={env['jax_devices']}",
        flush=True,
    )

    start_wall = time.perf_counter()
    runs: list[dict] = []
    skipped = 0
    failed = 0
    succeeded = 0

    for i, row in enumerate(rows, 1):
        n_obs = int(row["n_obs"])
        seed = int(row["seed"])
        convention = row["convention"]
        if filt_n_obs is not None and n_obs not in filt_n_obs:
            continue
        if filt_conv is not None and convention != filt_conv:
            continue
        out_path = output_path_for(args.output_root, n_obs, seed, convention)
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        print(
            f"[{i}/{n_total}] n_obs={n_obs} seed={seed} convention={convention}",
            flush=True,
        )
        cell_start = time.perf_counter()
        try:
            x_obs_f32 = reconstruct_x_obs_float32(seed, n_obs, convention)
            x_obs_f64 = jnp.asarray(x_obs_f32, dtype=jnp.float64)
            sha = hashlib.sha256(np.asarray(x_obs_f32).tobytes()).hexdigest()
            sampler_seed = stable_int("nuts_v3", n_obs, seed, convention)
            result = run_nuts_canonical(
                x_obs_f64=x_obs_f64,
                n_obs=n_obs,
                sampler_seed=sampler_seed,
                num_chains=args.num_chains,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                target_accept_prob=args.target_accept_prob,
            )
            diagnostics = diagnostics_from_samples(
                result.samples_grouped, result.extra_fields
            )
            write_fingerprint(
                out_path,
                n_obs=n_obs,
                seed=seed,
                convention=convention,
                sampler_seed=sampler_seed,
                samples_grouped=result.samples_grouped,
                diagnostics=diagnostics,
                extra_fields=result.extra_fields,
                runtime_seconds=result.runtime_seconds,
                x_obs_f32=x_obs_f32,
                x_obs_f64=x_obs_f64,
                sha=sha,
                num_chains=args.num_chains,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                target_accept_prob=args.target_accept_prob,
                env=env,
            )
            wall = time.perf_counter() - cell_start
            rhat_max = max(
                diagnostics["per_parameter"][n]["r_hat"] for n in PARAM_NAMES
            )
            print(
                f"  ok in {wall:.1f}s, max r_hat={rhat_max:.4f}, "
                f"divergences={diagnostics.get('divergence_count', 'na')}",
                flush=True,
            )
            succeeded += 1
            runs.append(
                {
                    "n_obs": n_obs,
                    "seed": seed,
                    "convention": convention,
                    "wall_seconds": wall,
                    "max_r_hat": rhat_max,
                    "divergence_count": diagnostics.get("divergence_count"),
                    "ok": True,
                }
            )
        except Exception as exc:
            wall = time.perf_counter() - cell_start
            print(f"  FAIL in {wall:.1f}s: {exc}", flush=True)
            failed += 1
            runs.append(
                {
                    "n_obs": n_obs,
                    "seed": seed,
                    "convention": convention,
                    "wall_seconds": wall,
                    "error": str(exc),
                    "ok": False,
                }
            )

        if args.limit and (succeeded + failed) >= args.limit:
            print(f"  hit --limit {args.limit}, stopping", flush=True)
            break

    total_wall = time.perf_counter() - start_wall
    summary = {
        "manifest": str(args.manifest),
        "total_rows": n_total,
        "executed": succeeded + failed,
        "succeeded": succeeded,
        "failed": failed,
        "skipped_existing": skipped,
        "wall_seconds": total_wall,
        "completed_at": utc_now(),
        "filter_n_obs": list(filt_n_obs) if filt_n_obs else None,
        "filter_convention": filt_conv,
        "env": env,
        "runs": runs,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Done. {succeeded} ok, {failed} failed, {skipped} skipped, "
        f"wall {total_wall:.0f}s. Wrote {args.summary}",
        flush=True,
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
