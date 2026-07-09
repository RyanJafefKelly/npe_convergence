"""Audit MA(2) b0 stored NPE samples against current reference posteriors."""

from __future__ import annotations

import argparse
import json
import math
import pickle as pkl
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro  # type: ignore
from numpyro.infer import MCMC, NUTS  # type: ignore

from npe_convergence.examples.ma2 import get_summaries_batches, numpyro_model_b0
from npe_convergence.metrics import kullback_leibler


TRUE_PARAMS = jnp.array([0.6, 0.2])
N_OBS = 1_000
N_SIMS = 1_000_000
ROOT = Path("res/ma2_b0")


def current_x_obs(seed: int, n_obs: int) -> jnp.ndarray:
    """Generate the current observed MA(2) summaries for a driver seed."""
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


def run_current_reference(
    seed: int,
    x_obs: jnp.ndarray,
    n_obs: int,
    output_path: Path,
    num_warmup: int,
    num_samples: int,
    thinning: int,
    num_chains: int,
    reuse: bool,
) -> jnp.ndarray:
    """Run or load the current NUTS reference posterior."""
    if reuse and output_path.exists():
        with output_path.open("rb") as f:
            return pkl.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nuts_kernel = NUTS(numpyro_model_b0)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples * thinning,
        thinning=thinning,
        num_chains=num_chains,
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
    return reference


def load_pickle(path: Path) -> jnp.ndarray:
    with path.open("rb") as f:
        return pkl.load(f)


def load_float(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    return float(text)


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isfinite(value):
        return value
    return None


def row_for_seed(args: argparse.Namespace, seed: int) -> dict[str, object]:
    audit_dir = ROOT / f"audit_n_obs_{args.n_obs}_seed_{seed}"
    reference_path = audit_dir / "nuts_current_reference.pkl"
    x_obs = current_x_obs(seed, args.n_obs)
    audit_dir.mkdir(parents=True, exist_ok=True)
    np.save(audit_dir / "x_obs_current.npy", np.asarray(x_obs))

    current_ref = run_current_reference(
        seed=seed,
        x_obs=x_obs,
        n_obs=args.n_obs,
        output_path=reference_path,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        thinning=args.thinning,
        num_chains=args.num_chains,
        reuse=args.reuse,
    )

    flow_dir = ROOT / f"npe_n_obs_{args.n_obs}_n_sims_{args.n_sims}_seed_{seed}"
    flow_samples = load_pickle(flow_dir / "posterior_samples.pkl")
    stored_ref = load_pickle(flow_dir / "true_posterior_samples.pkl")
    flow_kl_current = float(kullback_leibler(current_ref, flow_samples))

    gauss_dir = ROOT / f"gaussian_npe_n_obs_{args.n_obs}_n_sims_{args.n_sims}_seed_{seed}"
    gauss_kl_current = None
    gauss_samples_path = gauss_dir / "posterior_samples.pkl"
    if not gauss_samples_path.exists() and seed == 22:
        overlay_path = (
            ROOT
            / f"gaussian_npe_n_obs_{args.n_obs}_n_sims_{args.n_sims}_seed_{seed}_overlay"
            / "posterior_samples.pkl"
        )
        if overlay_path.exists():
            gauss_samples_path = overlay_path
    if gauss_samples_path.exists():
        gauss_samples = load_pickle(gauss_samples_path)
        gauss_kl_current = float(kullback_leibler(current_ref, gauss_samples))

    stored_ref_mean = np.asarray(stored_ref).mean(axis=0)
    current_ref_mean = np.asarray(current_ref).mean(axis=0)
    flow_mean = np.asarray(flow_samples).mean(axis=0)
    ref_mean_shift = float(np.linalg.norm(stored_ref_mean - current_ref_mean))

    row = {
        "seed": seed,
        "x_obs_current": np.asarray(x_obs).tolist(),
        "stored_ref_mean": stored_ref_mean.tolist(),
        "current_ref_mean": current_ref_mean.tolist(),
        "flow_mean": flow_mean.tolist(),
        "ref_mean_shift_l2": ref_mean_shift,
        "flow_kl_stored": finite_or_none(load_float(flow_dir / "kl.txt")),
        "flow_kl_current": flow_kl_current,
        "gauss_kl_stored": finite_or_none(load_float(gauss_dir / "kl.txt")),
        "gauss_kl_current": gauss_kl_current,
        "current_reference_path": str(reference_path),
        "gauss_samples_path": str(gauss_samples_path) if gauss_samples_path.exists() else None,
        "nuts_num_warmup": args.num_warmup,
        "nuts_num_samples": args.num_samples,
        "nuts_thinning": args.thinning,
        "nuts_num_chains": args.num_chains,
    }

    with (audit_dir / "audit_metrics.json").open("w") as f:
        json.dump(row, f, indent=2)
        f.write("\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--n_obs", type=int, default=N_OBS)
    parser.add_argument("--n_sims", type=int, default=N_SIMS)
    parser.add_argument("--num_warmup", type=int, default=2_000)
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--thinning", type=int, default=10)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("res/ma2_b0/audit_n_obs_1000_pilot_metrics.json"),
    )
    args = parser.parse_args()

    numpyro.set_host_device_count(args.num_chains)
    rows = []
    for seed in args.seeds:
        print(f"Auditing seed {seed}")
        rows.append(row_for_seed(args, seed))
        print(json.dumps(rows[-1], indent=2))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
