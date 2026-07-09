"""Aggregate the split-replicate stereological blackjax SMC-ABC reference.

The corrected stereological reference is produced as THREE separate 1-replicate
HPC jobs (replicate offsets 0, 1, 2), each writing its own output directory. The
runner's built-in between-replicate KL/MMD only compares replicates WITHIN one
run, so when the run is split it is empty. This script does that cross-replicate
check post hoc: it loads the smallest-tolerance particle snapshot from each of the
three output directories, reports per-replicate posterior summaries, the pairwise
KL/MMD between replicates, and a pooled reference.

It reuses the exact KL/MMD functions from the runner so the numbers are
consistent with the in-run diagnostics.

Usage (point at the three output dirs, in any order):

    python scripts/aggregate_stereological_blackjax_replicates.py \
        res/stereological_blackjax_smc_abc/<dir_rep0> \
        res/stereological_blackjax_smc_abc/<dir_rep1> \
        res/stereological_blackjax_smc_abc/<dir_rep2>

or let it auto-discover the three most recent dirs:

    python scripts/aggregate_stereological_blackjax_replicates.py --auto

Writes a JSON + Markdown verdict next to the first directory's parent, at
res/stereological_blackjax_smc_abc/cross_replicate_summary_<stamp>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the runner's own metrics so cross-replicate numbers match in-run ones.
from npe_convergence.scripts.run_stereological_blackjax_smc_abc import (  # noqa: E402
    rbf_mmd2,
)
from npe_convergence.metrics import kullback_leibler  # noqa: E402

PARAM_NAMES = ("lambda", "sigma", "xi")
TRUTH = {"lambda": 100.0, "sigma": 2.0, "xi": -0.1}
DEFAULT_PARENT = REPO_ROOT / "res" / "stereological_blackjax_smc_abc"


def smallest_tau_snapshot(run_dir: Path) -> tuple[float, Path]:
    """Return (tau, path) of the smallest-tolerance particle snapshot in run_dir."""
    snaps = list(run_dir.glob("particles_tau_*_rep_*.npz"))
    if not snaps:
        raise FileNotFoundError(f"no particle snapshots in {run_dir}")
    # filename: particles_tau_{tau:g}_rep_{rep}.npz
    def tau_of(p: Path) -> float:
        token = p.name.split("particles_tau_")[1].split("_rep_")[0]
        return float(token)
    best = min(snaps, key=tau_of)
    return tau_of(best), best


def load_theta(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["theta"], dtype=np.float64)


def summarise(theta: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(PARAM_NAMES):
        col = theta[:, i]
        out[name] = {
            "mean": float(np.mean(col)),
            "sd": float(np.std(col, ddof=1)),
            "q05": float(np.quantile(col, 0.05)),
            "q50": float(np.quantile(col, 0.50)),
            "q95": float(np.quantile(col, 0.95)),
        }
    return out


def subsample(theta: np.ndarray, n: int, seed: int) -> np.ndarray:
    if theta.shape[0] <= n:
        return theta
    rng = np.random.default_rng(seed)
    return theta[rng.choice(theta.shape[0], size=n, replace=False)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path,
                        help="The per-replicate output directories.")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover the 3 most recent run dirs under "
                             "res/stereological_blackjax_smc_abc/.")
    parser.add_argument("--max-kl-points", type=int, default=3000)
    args = parser.parse_args()

    if args.auto:
        candidates = sorted(
            (p for p in DEFAULT_PARENT.glob("n_obs_*_seed_*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        run_dirs = candidates[-3:]
        print(f"auto-discovered {len(run_dirs)} dirs:")
        for d in run_dirs:
            print("  ", d)
    else:
        run_dirs = args.run_dirs
    if len(run_dirs) < 2:
        raise SystemExit("need at least 2 run directories (got "
                         f"{len(run_dirs)}); pass them or use --auto")

    # Load each replicate's smallest-tolerance posterior.
    loaded: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        tau, snap = smallest_tau_snapshot(Path(run_dir))
        theta = load_theta(snap)
        loaded.append({
            "run_dir": str(run_dir),
            "snapshot": str(snap),
            "tau": tau,
            "n": int(theta.shape[0]),
            "theta": theta,
            "summary": summarise(theta),
        })
        s = loaded[-1]["summary"]
        print(f"{Path(run_dir).name}: tau={tau:g} n={theta.shape[0]} "
              f"lambda={s['lambda']['mean']:.2f} sigma={s['sigma']['mean']:.3f} "
              f"xi={s['xi']['mean']:.3f}")

    # Pairwise between-replicate KL / MMD at the (shared) smallest tau.
    pair_rows: list[dict[str, Any]] = []
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            li, lj = loaded[i], loaded[j]
            left = subsample(li["theta"], args.max_kl_points, seed=1000 + i)
            right = subsample(lj["theta"], args.max_kl_points, seed=2000 + j)
            pair_rows.append({
                "i": Path(li["run_dir"]).name,
                "j": Path(lj["run_dir"]).name,
                "tau_i": li["tau"], "tau_j": lj["tau"],
                "kl_i_to_j": float(kullback_leibler(left, right)),
                "kl_j_to_i": float(kullback_leibler(right, left)),
                "mmd2": float(rbf_mmd2(left, right, seed=7)),
                "points": int(min(left.shape[0], right.shape[0])),
            })

    # Pooled reference across replicates.
    pooled = np.concatenate([d["theta"] for d in loaded], axis=0)
    pooled_summary = summarise(pooled)

    # Verdict heuristic: between-replicate KL should be small in absolute terms
    # and small relative to the NPE-vs-reference gap (the stereological NPE-vs-ref
    # KL is order 1+; between-replicate KL well under ~0.1 is reassuring).
    all_kls = [r[k] for r in pair_rows for k in ("kl_i_to_j", "kl_j_to_i")]
    finite_kls = [v for v in all_kls if np.isfinite(v)]
    nonfinite_kls = len(all_kls) - len(finite_kls)
    max_pair_kl = max(finite_kls) if finite_kls else float("nan")
    sigma_means = [d["summary"]["sigma"]["mean"] for d in loaded]
    sigma_spread = float(max(sigma_means) - min(sigma_means))
    sigma_near_truth = all(abs(m - TRUTH["sigma"]) < 0.2 for m in sigma_means)

    verdict_lines = []
    if sigma_near_truth:
        verdict_lines.append(
            f"PASS sigma: all replicates within 0.2 of truth 2.0 "
            f"(means {', '.join(f'{m:.3f}' for m in sigma_means)}), "
            f"vs OLD BUGGY reference 0.39.")
    else:
        verdict_lines.append(
            f"CHECK sigma: replicate means {sigma_means} not all near truth 2.0. "
            "If any is ~0.4 the buggy simulator path may have returned.")
    verdict_lines.append(
        f"between-replicate sigma-mean spread {sigma_spread:.3f}; "
        f"max finite pairwise KL {max_pair_kl:.4f} "
        "(small relative to the order-1 NPE-vs-reference gap is reassuring).")
    if nonfinite_kls:
        verdict_lines.append(
            f"WARNING: {nonfinite_kls} non-finite pairwise KL value(s). This is "
            "expected only if two replicate directories are identical (the kNN KL "
            "estimator returns -inf on duplicate point sets). If the three jobs "
            "are genuinely distinct runs, a non-finite KL indicates a problem "
            "(e.g. a duplicated/empty snapshot) and must be investigated, not "
            "treated as 'small'.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = DEFAULT_PARENT / f"cross_replicate_summary_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": stamp,
        "run_dirs": [d["run_dir"] for d in loaded],
        "per_replicate": [
            {k: d[k] for k in ("run_dir", "snapshot", "tau", "n", "summary")}
            for d in loaded
        ],
        "pairwise": pair_rows,
        "pooled_summary": pooled_summary,
        "pooled_n": int(pooled.shape[0]),
        "truth": TRUTH,
        "verdict": verdict_lines,
    }
    (out_dir / "cross_replicate_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")

    md = ["# Stereological reference: cross-replicate check", "",
          f"Created `{stamp}`", "", "## Verdict", ""]
    md += [f"- {line}" for line in verdict_lines]
    md += ["", "## Per-replicate (smallest tolerance)", "",
           "| run | tau | n | lambda | sigma | xi |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for d in loaded:
        s = d["summary"]
        md.append(f"| {Path(d['run_dir']).name} | {d['tau']:g} | {d['n']} | "
                  f"{s['lambda']['mean']:.2f} | {s['sigma']['mean']:.3f} | "
                  f"{s['xi']['mean']:.3f} |")
    md += ["", "## Pairwise between-replicate", "",
           "| i | j | KL i->j | KL j->i | MMD^2 |",
           "| --- | --- | ---: | ---: | ---: |"]
    for r in pair_rows:
        md.append(f"| {r['i']} | {r['j']} | {r['kl_i_to_j']:.4f} | "
                  f"{r['kl_j_to_i']:.4f} | {r['mmd2']:.4e} |")
    md += ["", "## Pooled reference",
           f"- n = {pooled.shape[0]}",
           f"- lambda {pooled_summary['lambda']['mean']:.2f}, "
           f"sigma {pooled_summary['sigma']['mean']:.3f}, "
           f"xi {pooled_summary['xi']['mean']:.3f}"]
    (out_dir / "cross_replicate_summary.md").write_text("\n".join(md) + "\n")

    print()
    for line in verdict_lines:
        print(line)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
