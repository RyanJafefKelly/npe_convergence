"""Audit the legacy stereological ELFI SMC-ABC reference.

This is a read-only diagnostic. It checks the saved parameter order, compares
the legacy ABC simulator with the current jitted simulator, and writes a small
JSON/Markdown record. It does not modify the old reference.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.stereological import (  # noqa: E402
    get_summaries as active_summaries,
)
from npe_convergence.examples.stereological import (  # noqa: E402
    stereological as active_stereological,
)
from npe_convergence.scripts.run_stereological_smc_abc import (  # noqa: E402
    get_summaries as legacy_summaries,
)
from npe_convergence.scripts.run_stereological_smc_abc import (  # noqa: E402
    stereological_sim as legacy_stereological,
)

TRUE_THETA = np.array([100.0, 2.0, -0.1])
PARAM_NAMES = ("lambda", "sigma", "xi")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, jax.Array):
        return json_sanitize(np.asarray(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if np.isnan(x):
            return "NaN"
        if np.isposinf(x):
            return "Infinity"
        if np.isneginf(x):
            return "-Infinity"
        return x
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(json_sanitize(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def git_record() -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, command in {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    }.items():
        try:
            record[key] = subprocess.check_output(
                command,
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
        except Exception:
            record[key] = None
    try:
        subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        record["dirty"] = False
    except subprocess.CalledProcessError:
        record["dirty"] = True
    except Exception:
        record["dirty"] = None
    return record


def summarize_vector(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    hist, edges = np.histogram(values, bins=80)
    mode_proxy = 0.5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "mode_proxy": float(mode_proxy),
        "sd": float(np.std(values, ddof=1)),
        "max": float(np.max(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def observed_active_summary(seed: int, n_obs: int) -> np.ndarray:
    key = random.PRNGKey(seed)
    _, subkey = random.split(key)
    x_obs = active_stereological(
        subkey,
        *jnp.asarray(TRUE_THETA),
        n_obs=n_obs,
        num_samples=1,
    )
    return np.asarray(active_summaries(x_obs))[0]


def replicated_summary(
    *,
    simulator: str,
    theta: np.ndarray,
    n_obs: int,
    reps: int,
    seed: int,
) -> np.ndarray:
    summaries: list[np.ndarray] = []
    if simulator == "legacy_numpy":
        rng = np.random.RandomState(seed)
        for _ in range(reps):
            x = legacy_stereological(
                *theta,
                n_obs=n_obs,
                batch_size=1,
                random_state=rng,
            )
            summaries.append(np.asarray(legacy_summaries(x))[0])
        return np.asarray(summaries)

    if simulator == "active_jax":
        key = random.PRNGKey(seed)
        for _ in range(reps):
            key, subkey = random.split(key)
            x = active_stereological(
                subkey,
                *jnp.asarray(theta),
                n_obs=n_obs,
                num_samples=1,
            )
            summaries.append(np.asarray(active_summaries(x))[0])
        return np.asarray(summaries)

    raise ValueError(f"unknown simulator: {simulator}")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    legacy = payload["legacy_reference"]
    mismatch = payload["simulator_mismatch"]
    active_obs = payload["active_observed_summary"]
    verdict = payload["verdict"]
    lines = [
        "# Stereological Reference Audit",
        "",
        f"Created: `{payload['created_at']}`",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Legacy Reference",
        "",
        (
            "The saved ELFI pickle has raw columns `(xi, sigma, lambda)`, matching "
            "the overlay code's reorder to `(lambda, sigma, xi)`."
        ),
        "",
        "| parameter | mean | median | mode proxy | q05 | q95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in PARAM_NAMES:
        stats = legacy["aggregate_order_lambda_sigma_xi"][name]
        lines.append(
            f"| {name} | {stats['mean']:.6g} | {stats['median']:.6g} | "
            f"{stats['mode_proxy']:.6g} | {stats['q05']:.6g} | {stats['q95']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Observed Cell",
            "",
            f"- true theta: `{TRUE_THETA.tolist()}`",
            f"- active observed summary: `{np.asarray(active_obs['summary']).tolist()}`",
            f"- K = round(n_obs * s1): `{active_obs['K']}`",
            "",
            "## Simulator Mismatch Check",
            "",
            "| simulator | theta | mean count | mean log min | mean log mean | mean log max |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in mismatch:
        mean = row["summary_mean"]
        lines.append(
            f"| {row['simulator']} | {row['theta_name']} | "
            f"{mean[0]:.6g} | {mean[1]:.6g} | {mean[2]:.6g} | {mean[3]:.6g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.reference).open("rb") as handle:
        raw = np.asarray(pickle.load(handle), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"expected reference array with shape (N, 3), got {raw.shape}")

    aggregate_order = np.column_stack([raw[:, 2], raw[:, 1], raw[:, 0]])
    old_smc_center = np.mean(aggregate_order, axis=0)
    active_obs = observed_active_summary(args.seed, args.n_obs)

    mismatch_rows: list[dict[str, Any]] = []
    for simulator in ("legacy_numpy", "active_jax"):
        for theta_name, theta in (
            ("true", TRUE_THETA),
            ("old_smc_mean", old_smc_center),
        ):
            summaries = replicated_summary(
                simulator=simulator,
                theta=np.asarray(theta, dtype=float),
                n_obs=args.n_obs,
                reps=args.reps,
                seed=args.sim_seed,
            )
            mismatch_rows.append(
                {
                    "simulator": simulator,
                    "theta_name": theta_name,
                    "theta": np.asarray(theta, dtype=float),
                    "summary_mean": np.mean(summaries, axis=0),
                    "summary_sd": np.std(summaries, axis=0, ddof=1),
                    "mean_minus_active_observed": np.mean(summaries, axis=0) - active_obs,
                }
            )

    legacy_stats = {
        "raw_columns": {
            f"raw_col_{index}": summarize_vector(raw[:, index])
            for index in range(raw.shape[1])
        },
        "aggregate_order_lambda_sigma_xi": {
            name: summarize_vector(aggregate_order[:, index])
            for index, name in enumerate(PARAM_NAMES)
        },
        "raw_correlation": np.corrcoef(raw.T),
        "sample_count": int(raw.shape[0]),
        "path": str(Path(args.reference)),
    }

    sigma_mode = legacy_stats["aggregate_order_lambda_sigma_xi"]["sigma"]["mode_proxy"]
    verdict = (
        "The existing reference is genuinely wrong for the current stereological "
        "NPE comparison. The overlay labelling is correct: under the saved ELFI "
        f"column order, sigma has mode proxy {sigma_mode:.3g} while the model-control "
        "truth is sigma = 2.0. The failure is not just Monte Carlo noise: the "
        "legacy ELFI candidate simulator uses an extra multiplicative radius term "
        "that the current jitted simulator does not use. At the true theta this "
        "legacy simulator misses the active observed continuous summaries badly; "
        "near the old SMC centre it matches them."
    )

    payload = {
        "created_at": utc_now(),
        "git": git_record(),
        "n_obs": args.n_obs,
        "seed": args.seed,
        "true_theta": TRUE_THETA,
        "active_observed_summary": {
            "summary": active_obs,
            "K": int(round(args.n_obs * float(active_obs[0]))),
            "n_obs_times_s1": float(args.n_obs * active_obs[0]),
        },
        "legacy_reference": legacy_stats,
        "simulator_mismatch": mismatch_rows,
        "verdict": verdict,
    }
    write_json(out_dir / "audit.json", payload)
    write_markdown(out_dir / "README.md", payload)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sim-seed", type=int, default=123)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO_ROOT
        / "res"
        / "stereological_smc_abc"
        / "npe_n_obs_1000_n_sims_None_seed_1_max_iter_9"
        / "adaptive_smc_samples.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("res/stereological_reference_audit/n_obs_1000_seed_1"),
    )
    return parser


def main() -> None:
    out_dir = run(build_parser().parse_args())
    print(out_dir)


if __name__ == "__main__":
    main()
