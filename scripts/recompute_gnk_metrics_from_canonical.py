#!/usr/bin/env python
"""Recompute per-cell GNK KL and MMD against the v3 canonical NUTS references.

For each existing paper-grid cell directory under res/gnk/:
  - res/gnk/npe_n_obs_X_n_sims_Y_seed_Z/        (flow-NPE)
  - res/gnk/gaussian_npe_n_obs_X_n_sims_Y_seed_Z/ (gaussian-NPE)

we:
  1. Load posterior_samples.pkl (existing NPE samples, unchanged).
  2. Load the matching convention canonical reference from res/gnk_v3_refs/.
  3. Recompute KL via npe_convergence.metrics.kullback_leibler and MMD via
     unbiased_mmd + median_heuristic, matching the convention in run_gnk.py
     (2000 samples per side, kNN k=1).
  4. Back up the existing kl.txt and mmd.txt as kl_legacy.txt and
     mmd_legacy.txt (idempotent: skip backup if already done).
  5. Write the new kl.txt, mmd.txt, and metrics_v3.json with full
     provenance.

Idempotent: cells whose metrics_v3.json already records a matching
ref_x_obs_sha256 are skipped unless --force is passed.
"""
from __future__ import annotations

import argparse
import json
import pickle as pkl
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as random
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.metrics import kullback_leibler, unbiased_mmd
from scipy.spatial.distance import pdist


def median_heuristic_fast(x: np.ndarray) -> float:
    """Numpy median-heuristic lengthscale.

    Matches the math in npe_convergence.metrics.median_heuristic
    (median pairwise euclidean distance, divided by 2, square-rooted),
    but uses scipy.spatial.distance.pdist instead of a JAX double-loop
    so it is roughly 50x faster.
    """
    dists = pdist(np.asarray(x, dtype=np.float64), metric="euclidean")
    return float(np.sqrt(np.median(dists) / 2.0))


PARAM_NAMES = ("A", "B", "g", "k")
N_METRIC = 2000
DEFAULT_V3_ROOT = REPO_ROOT / "res" / "gnk_v3_refs"
DEFAULT_GNK_ROOT = REPO_ROOT / "res" / "gnk"
PER_CELL_RE = re.compile(
    r"^(?P<prefix>npe|gaussian_npe)_n_obs_(?P<n_obs>\d+)_n_sims_(?P<n_sims>\d+)_seed_(?P<seed>\d+)$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_v3_reference(v3_root: Path, n_obs: int, seed: int, convention: str) -> dict | None:
    path = v3_root / f"nuts_n_obs_{n_obs}_seed_{seed}_conv_{convention}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pkl.load(f)


def deterministic_subsample(samples: np.ndarray, n: int, rng_seed: int) -> np.ndarray:
    if samples.shape[0] <= n:
        return samples
    key = random.key(rng_seed)
    idx = np.asarray(random.permutation(key, samples.shape[0])[:n])
    return samples[idx]


def deduplicate(samples: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove exact duplicate rows from samples.

    NUTS chains occasionally produce exact-duplicate rows (zero-energy
    transitions). The kNN KL estimator returns inf when a query sample
    has zero distance to its nearest neighbour, so we dedupe before
    subsampling. Removing < 0.5% of NUTS rows does not materially change
    the empirical distribution. Returns (unique_samples, n_duplicates).
    """
    unique = np.unique(samples, axis=0)
    return unique, samples.shape[0] - unique.shape[0]


def already_processed(metrics_v3_path: Path, expected_sha: str) -> bool:
    if not metrics_v3_path.exists():
        return False
    try:
        existing = json.loads(metrics_v3_path.read_text())
    except Exception:
        return False
    return existing.get("ref_x_obs_sha256") == expected_sha


def process_cell(
    cell_dir: Path,
    v3_root: Path,
    *,
    force: bool,
    rng_seed: int = 0,
) -> dict[str, Any]:
    name = cell_dir.name
    match = PER_CELL_RE.match(name)
    if match is None:
        return {"cell": str(cell_dir), "skipped": True, "reason": "unparsed_name"}
    prefix = match.group("prefix")
    n_obs = int(match.group("n_obs"))
    n_sims = int(match.group("n_sims"))
    seed = int(match.group("seed"))
    convention = "flow" if prefix == "npe" else "gaussian"
    method = "flow_npe" if prefix == "npe" else "gaussian_npe"

    posterior_path = cell_dir / "posterior_samples.pkl"
    if not posterior_path.exists():
        return {"cell": str(cell_dir), "skipped": True, "reason": "no_posterior_samples"}

    ref = load_v3_reference(v3_root, n_obs, seed, convention)
    if ref is None:
        return {
            "cell": str(cell_dir),
            "skipped": True,
            "reason": f"v3_reference_missing_{convention}",
        }

    expected_sha = ref["x_obs_summary_unstandardised_sha256"]
    metrics_v3_path = cell_dir / "metrics_v3.json"
    if not force and already_processed(metrics_v3_path, expected_sha):
        return {
            "cell": str(cell_dir),
            "skipped": True,
            "reason": "already_processed",
            "ref_x_obs_sha256": expected_sha,
        }

    with posterior_path.open("rb") as f:
        posterior_samples = np.asarray(pkl.load(f), dtype=np.float64)
    if posterior_samples.ndim != 2 or posterior_samples.shape[1] != 4:
        return {"cell": str(cell_dir), "skipped": True, "reason": "bad_posterior_shape"}

    # Flatten v3 reference (chain-grouped) to (n_total, 4).
    ref_grouped = np.asarray(ref["samples"])
    ref_flat = ref_grouped.reshape(-1, ref_grouped.shape[-1]).astype(np.float64)
    ref_unique, ref_n_dup = deduplicate(ref_flat)
    ps_unique, ps_n_dup = deduplicate(posterior_samples)

    # Subsample deterministically.
    ps_thin = deterministic_subsample(ps_unique, N_METRIC, rng_seed=rng_seed)
    ref_thin = deterministic_subsample(ref_unique, N_METRIC, rng_seed=rng_seed + 1)

    if not (np.all(np.isfinite(ps_thin)) and np.all(np.isfinite(ref_thin))):
        return {"cell": str(cell_dir), "skipped": True, "reason": "non_finite_samples"}

    t0 = time.perf_counter()
    kl_value = float(kullback_leibler(jnp.asarray(ref_thin), jnp.asarray(ps_thin)))
    lengthscale = median_heuristic_fast(np.vstack([ref_thin, ps_thin]))
    mmd_value = float(
        unbiased_mmd(jnp.asarray(ref_thin), jnp.asarray(ps_thin), lengthscale)
    )
    runtime = time.perf_counter() - t0

    # Backup existing metrics if not already backed up.
    for legacy_name, new_name in (("kl.txt", "kl_legacy.txt"), ("mmd.txt", "mmd_legacy.txt")):
        cur = cell_dir / legacy_name
        legacy = cell_dir / new_name
        if cur.exists() and not legacy.exists():
            legacy.write_text(cur.read_text())

    # Write new kl.txt and mmd.txt and metrics_v3.json.
    (cell_dir / "kl.txt").write_text(str(kl_value))
    (cell_dir / "mmd.txt").write_text(str(mmd_value))
    payload = {
        "method": method,
        "n_obs": n_obs,
        "n_sims": n_sims,
        "seed": seed,
        "convention": convention,
        "ref_path": str(v3_root / f"nuts_n_obs_{n_obs}_seed_{seed}_conv_{convention}.pkl"),
        "ref_x_obs_sha256": expected_sha,
        "ref_density_version": ref.get("density_version"),
        "kl_value": kl_value,
        "mmd_value": mmd_value,
        "mmd_lengthscale": lengthscale,
        "n_metric": N_METRIC,
        "rng_seed": rng_seed,
        "n_posterior_samples": int(posterior_samples.shape[0]),
        "n_posterior_duplicates_removed": int(ps_n_dup),
        "n_reference_samples": int(ref_flat.shape[0]),
        "n_reference_duplicates_removed": int(ref_n_dup),
        "runtime_seconds": runtime,
        "created_at_utc": utc_now(),
    }
    metrics_v3_path.write_text(json.dumps(payload, indent=2) + "\n")

    return {
        "cell": str(cell_dir.relative_to(REPO_ROOT)),
        "method": method,
        "n_obs": n_obs,
        "n_sims": n_sims,
        "seed": seed,
        "convention": convention,
        "kl": kl_value,
        "mmd": mmd_value,
        "ref_x_obs_sha256": expected_sha,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--gnk-root", type=Path, default=DEFAULT_GNK_ROOT)
    parser.add_argument(
        "--filter-n-obs",
        type=str,
        default="",
        help="Comma-separated n_obs filter (default: all).",
    )
    parser.add_argument(
        "--filter-method",
        choices=("flow", "gaussian", "both"),
        default="both",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when metrics_v3.json already matches the ref hash.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many processed cells (0 = all).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "meeting_2026_05_18"
            / "gnk_nuts_refresh_plan"
            / "gnk_recompute_log.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filt_n = (
        {int(x) for x in args.filter_n_obs.split(",") if x.strip()}
        if args.filter_n_obs.strip()
        else None
    )
    method_filter = args.filter_method

    cells = []
    for d in sorted(args.gnk_root.iterdir()):
        if not d.is_dir():
            continue
        m = PER_CELL_RE.match(d.name)
        if m is None:
            continue
        prefix = m.group("prefix")
        if method_filter == "flow" and prefix != "npe":
            continue
        if method_filter == "gaussian" and prefix != "gaussian_npe":
            continue
        n_obs = int(m.group("n_obs"))
        if filt_n is not None and n_obs not in filt_n:
            continue
        cells.append(d)

    print(f"Recomputing metrics on {len(cells)} cells (force={args.force}).")
    results = []
    processed = 0
    for i, cell_dir in enumerate(cells, 1):
        result = process_cell(cell_dir, args.v3_root, force=args.force)
        results.append(result)
        if not result.get("skipped"):
            processed += 1
            if processed % 25 == 0:
                print(
                    f"  [{i}/{len(cells)}] processed {processed} cells",
                    flush=True,
                )
        if args.limit and processed >= args.limit:
            print(f"  hit --limit {args.limit}, stopping", flush=True)
            break

    summary = {
        "total_cells": len(cells),
        "processed": processed,
        "skipped": sum(1 for r in results if r.get("skipped")),
        "skipped_reasons": {},
        "completed_at": utc_now(),
        "results": results,
    }
    for r in results:
        if r.get("skipped"):
            reason = r.get("reason", "unknown")
            summary["skipped_reasons"][reason] = summary["skipped_reasons"].get(reason, 0) + 1

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  done. processed={processed}, skipped={summary['skipped']}")
    print(f"  skip reasons: {summary['skipped_reasons']}")
    print(f"  wrote {args.log}")


if __name__ == "__main__":
    main()
