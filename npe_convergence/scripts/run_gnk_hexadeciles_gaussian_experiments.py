"""Run the reviewed GNK hexadecile Gaussian-NPE paper-facing grid."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpyro  # type: ignore

from npe_convergence.scripts.run_gnk_hexadeciles_gaussian import (
    DEFAULT_OUTPUT_ROOT,
    output_dir,
    parse_hidden_dims,
    resolve_output_root,
    run_gnk_hexadeciles_gaussian,
)


def parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    return values


def standard_budget_values(n_obs: int) -> list[tuple[str, int]]:
    return [
        ("n", n_obs),
        ("n_log_n", int(n_obs * math.log(n_obs))),
        ("n_3_over_2", int(n_obs ** (3 / 2))),
        ("n_2", n_obs**2),
    ]


def build_grid(n_obs_values: list[int], budget_family: str) -> list[dict[str, int | str]]:
    if budget_family != "standard":
        raise ValueError(f"Unsupported budget family: {budget_family}")
    rows: list[dict[str, int | str]] = []
    for n_obs in n_obs_values:
        for label, n_sims in standard_budget_values(n_obs):
            rows.append({"n_obs": n_obs, "n_sims": n_sims, "budget": label})
    return rows


def preflight_output_dirs(
    *,
    output_root: Path,
    seed: int,
    rows: list[dict[str, int | str]],
    fail_on_collision: bool,
) -> list[Path]:
    collisions = []
    for row in rows:
        out = output_dir(output_root, int(row["n_obs"]), int(row["n_sims"]), seed)
        if out.exists():
            collisions.append(out)
    if collisions and fail_on_collision:
        joined = "\n".join(str(path) for path in collisions)
        raise FileExistsError(
            "Refusing to run because selected output directories already exist:\n"
            + joined
        )
    return collisions


def print_dry_run(
    *,
    output_root: Path,
    seed: int,
    rows: list[dict[str, int | str]],
    collisions: list[Path],
) -> None:
    print("GNK hexadecile Gaussian-NPE experiment dry-run")
    print(f"seed: {seed}")
    print(f"output_root: {output_root}")
    print("")
    print("n_obs,n_sims,budget,output_dir,exists")
    collision_set = {path.resolve() for path in collisions}
    for row in rows:
        out = output_dir(output_root, int(row["n_obs"]), int(row["n_sims"]), seed)
        exists = out.resolve() in collision_set or out.exists()
        print(
            f"{row['n_obs']},{row['n_sims']},{row['budget']},"
            f"{out},{str(exists).lower()}"
        )


def run_experiments(args: argparse.Namespace) -> None:
    output_root = resolve_output_root(args.output_root)
    n_obs_values = parse_int_list(args.n_obs)
    rows = build_grid(n_obs_values, args.budget_family)
    collisions = preflight_output_dirs(
        output_root=output_root,
        seed=args.seed,
        rows=rows,
        fail_on_collision=args.fail_on_collision,
    )
    if args.dry_run:
        print_dry_run(
            output_root=output_root,
            seed=args.seed,
            rows=rows,
            collisions=collisions,
        )
        return

    for row in rows:
        run_gnk_hexadeciles_gaussian(
            seed=args.seed,
            n_obs=int(row["n_obs"]),
            n_sims=int(row["n_sims"]),
            output_root=output_root,
            fail_on_collision=args.fail_on_collision,
            nuts_seed=args.nuts_seed,
            num_posterior_samples=args.num_posterior_samples,
            num_nuts_samples=args.num_nuts_samples,
            num_nuts_warmup=args.num_nuts_warmup,
            num_coverage_samples=args.num_coverage_samples,
            num_metric_samples=args.num_metric_samples,
            hidden_dims=parse_hidden_dims(args.hidden_dims),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            val_frac=args.val_frac,
            prior_batch_size=args.prior_batch_size,
            smoke_reference_samples=args.smoke_reference_samples,
            save_plots=args.save_plots,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_gnk_hexadeciles_gaussian_experiments.py",
        description="Run paper-facing GNK hexadecile Gaussian-NPE cells.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-obs", dest="n_obs", type=str, default="100,1000")
    parser.add_argument(
        "--budget-family",
        choices=["standard"],
        default="standard",
        help="standard means N in {n, int(n log n), int(n^(3/2)), n^2}.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-collision", action="store_true", default=True)
    parser.add_argument("--nuts-seed", type=int, default=1)
    parser.add_argument("--num-posterior-samples", type=int, default=10_000)
    parser.add_argument("--num-nuts-samples", type=int, default=10_000)
    parser.add_argument("--num-nuts-warmup", type=int, default=10_000)
    parser.add_argument("--num-coverage-samples", type=int, default=100)
    parser.add_argument("--num-metric-samples", type=int, default=2000)
    parser.add_argument("--hidden-dims", type=str, default="128,128")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--prior-batch-size", type=int, default=None)
    parser.add_argument("--smoke-reference-samples", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    return parser


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    try:
        run_experiments(build_parser().parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
