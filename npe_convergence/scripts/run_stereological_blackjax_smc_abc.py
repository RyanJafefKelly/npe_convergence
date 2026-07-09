"""Count-factorised blackjax SMC-ABC for the stereological example.

This script targets the summary posterior used by the current NPE experiments.
It conditions on the observed total count K exactly, runs SMC-ABC only for
``(sigma, xi)`` using the three log summaries, and draws ``lambda`` exactly from
the truncated Gamma conditional posterior.

The required SMC-ABC kernel is Ryan's local blackjax implementation, currently
available in ``~/python_projects/blackjax`` or through ``--blackjax-root``. The
PyPI blackjax 1.2.x package does not include ``blackjax.smc.abc``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.scipy.special import expit, logit
from scipy.spatial.distance import pdist
from scipy.stats import gamma as scipy_gamma
from scipy.stats import chi2, norm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from npe_convergence.examples.stereological import (  # noqa: E402
    get_summaries as active_summaries,
)
from npe_convergence.examples.stereological import (  # noqa: E402
    stereological as active_stereological,
)
from npe_convergence.metrics import kullback_leibler  # noqa: E402

PARAM_NAMES = ("lambda", "sigma", "xi")
PHI_NAMES = ("sigma", "xi")
TRUE_THETA = jnp.array([100.0, 2.0, -0.1])
V0 = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_stamp(created_at: str) -> str:
    return created_at.replace("-", "").replace(":", "").replace("+00:00", "Z")


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def rng_for(seed: int, *parts: object) -> jax.Array:
    key = random.PRNGKey(seed)
    for part in parts:
        key = random.fold_in(key, stable_int(part))
    return key


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if isinstance(value, jax.Array):
        return json_sanitize(np.asarray(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        if np.isnan(scalar):
            return "NaN"
        if np.isposinf(scalar):
            return "Infinity"
        if np.isneginf(scalar):
            return "-Infinity"
        return scalar
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(json_sanitize(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_sanitize(row))


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


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def import_blackjax_abc(blackjax_root: Path | None) -> Any:
    candidates: list[Path] = []
    if blackjax_root is not None:
        candidates.append(blackjax_root.expanduser())
    env_root = os.environ.get("BLACKJAX_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            Path.home() / "python_projects" / "blackjax",
            Path.home() / "blackjax",
            REPO_ROOT.parent / "blackjax",
        ]
    )

    def clear_blackjax_modules() -> None:
        for name in list(sys.modules):
            if name == "blackjax" or name.startswith("blackjax."):
                del sys.modules[name]

    try:
        return importlib.import_module("blackjax.smc.abc")
    except Exception:
        clear_blackjax_modules()

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if not (candidate / "blackjax" / "smc" / "abc.py").is_file():
            continue
        sys.path.insert(0, str(candidate))
        clear_blackjax_modules()
        try:
            return importlib.import_module("blackjax.smc.abc")
        except Exception:
            clear_blackjax_modules()
            try:
                sys.path.remove(str(candidate))
            except ValueError:
                pass

    raise RuntimeError(
        "Could not import blackjax.smc.abc. Pass --blackjax-root pointing at "
        "Ryan's local blackjax checkout. Checked: " + ", ".join(checked)
    )


def eta_to_phi(eta: jax.Array) -> jax.Array:
    sigma = expit(eta[..., 0]) * 15.0
    xi = expit(eta[..., 1]) * 6.0 - 3.0
    return jnp.stack([sigma, xi], axis=-1)


def phi_to_eta(phi: jax.Array) -> jax.Array:
    sigma = phi[..., 0]
    xi = phi[..., 1]
    return jnp.stack([logit(sigma / 15.0), logit((xi + 3.0) / 6.0)], axis=-1)


def log_prior_eta(eta: jax.Array) -> jax.Array:
    # Uniform priors on sigma in [0, 15] and xi in [-3, 3], transformed to R^2.
    return jnp.sum(-jax.nn.softplus(-eta) - jax.nn.softplus(eta))


def prior_sample_eta(key: jax.Array) -> jax.Array:
    key_sigma, key_xi = random.split(key)
    sigma = random.uniform(key_sigma, ()) * 15.0
    xi = random.uniform(key_xi, ()) * 6.0 - 3.0
    return phi_to_eta(jnp.array([sigma, xi]))


def observed_summary(seed: int, n_obs: int) -> np.ndarray:
    key = random.PRNGKey(seed)
    _, subkey = random.split(key)
    x_obs = active_stereological(
        subkey,
        *TRUE_THETA,
        n_obs=n_obs,
        num_samples=1,
    )
    return np.asarray(active_summaries(x_obs))[0]


def sample_lambda_given_count(
    *,
    count: int,
    n_obs: int,
    size: int,
    seed: int,
) -> np.ndarray:
    shape = count + 1.0
    rate = float(n_obs)
    dist = scipy_gamma(a=shape, scale=1.0 / rate)
    lower = float(dist.cdf(30.0))
    upper = float(dist.cdf(200.0))
    rng = np.random.default_rng(seed)
    u = rng.uniform(lower, upper, size=size)
    return np.asarray(dist.ppf(u), dtype=np.float64)


def _gpd_from_uniform(u: jax.Array, sigma: jax.Array, xi: jax.Array) -> jax.Array:
    u = jnp.clip(u, 1e-7, 1.0 - 1e-7)
    return jnp.where(
        jnp.abs(xi) > 1e-6,
        sigma * ((1.0 - u) ** (-xi) - 1.0) / xi,
        sigma * (-jnp.log1p(-u)),
    )


def _sample_multinomial_counts(key: jax.Array, total_count: jax.Array, n_obs: int) -> jax.Array:
    keys = random.split(key, max(n_obs - 1, 1))

    def body(remaining: jax.Array, xs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        index, subkey = xs
        cells_left = n_obs - index
        draw = random.binomial(
            subkey,
            n=remaining,
            p=1.0 / cells_left,
            dtype=jnp.float32,
        ).astype(jnp.int32)
        draw = jnp.minimum(draw, remaining)
        return remaining - draw, draw

    if n_obs == 1:
        return jnp.asarray([total_count], dtype=jnp.int32)

    remaining, first_counts = jax.lax.scan(
        body,
        jnp.asarray(total_count, dtype=jnp.int32),
        (jnp.arange(n_obs - 1), keys),
    )
    return jnp.concatenate([first_counts, remaining[None]], axis=0)


@partial(jax.jit, static_argnames=("n_obs", "max_locs"))
def conditional_log_summaries(
    key: jax.Array,
    eta: jax.Array,
    total_count: jax.Array,
    *,
    n_obs: int,
    max_locs: int,
) -> jax.Array:
    """Simulate the three log summaries conditional on the observed count K."""
    sigma, xi = eta_to_phi(eta)
    key_counts, key_cells = random.split(key)
    counts = _sample_multinomial_counts(key_counts, total_count, n_obs)
    cell_keys = random.split(key_cells, n_obs)
    loc_index = jnp.arange(max_locs)

    def cell_summary(cell_key: jax.Array, count: jax.Array) -> jax.Array:
        count_clipped = jnp.minimum(count, max_locs)
        mask = loc_index < count_clipped
        key_gpd, key_unif = random.split(cell_key)
        u_gpd = random.uniform(key_gpd, shape=(max_locs, 3))
        gpd = _gpd_from_uniform(u_gpd, sigma, xi)
        radius = jnp.max(gpd, axis=-1) + V0
        u_cross = random.uniform(key_unif, shape=(max_locs, 2))
        values = (radius - V0) * jnp.max(u_cross, axis=-1) + V0
        values = jnp.where(mask, values, jnp.nan)
        finite_count = jnp.maximum(jnp.sum(mask), 1)
        min_value = jnp.nanmin(values)
        mean_value = jnp.nansum(values) / finite_count
        max_value = jnp.nanmax(values)
        fallback = jnp.array([V0, V0, V0])
        logs = jnp.log(jnp.array([min_value, mean_value, max_value]))
        return jnp.where(count_clipped > 0, logs, jnp.log(fallback))

    per_cell = jax.vmap(cell_summary)(cell_keys, counts)
    return jnp.mean(per_cell, axis=0)


def simulate_summary_batch(
    *,
    key: jax.Array,
    eta: np.ndarray,
    count: int,
    n_obs: int,
    max_locs: int,
) -> np.ndarray:
    eta_jax = jnp.asarray(eta)
    keys = random.split(key, eta_jax.shape[0])
    summaries = jax.vmap(
        lambda subkey, particle: conditional_log_summaries(
            subkey,
            particle,
            jnp.asarray(count, dtype=jnp.int32),
            n_obs=n_obs,
            max_locs=max_locs,
        )
    )(keys, eta_jax)
    return np.asarray(summaries)


def estimate_summary_covariance(
    *,
    seed: int,
    phi_center: np.ndarray,
    count: int,
    n_obs: int,
    max_locs: int,
    num_sims: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    eta_center = np.asarray(phi_to_eta(jnp.asarray(phi_center)))
    rows: list[np.ndarray] = []
    done = 0
    while done < num_sims:
        size = min(batch_size, num_sims - done)
        key = rng_for(seed, "pilot", done)
        eta = np.repeat(eta_center[None, :], size, axis=0)
        rows.append(
            simulate_summary_batch(
                key=key,
                eta=eta,
                count=count,
                n_obs=n_obs,
                max_locs=max_locs,
            )
        )
        done += size
    summaries = np.concatenate(rows, axis=0)
    cov = np.cov(summaries, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    scale = float(np.trace(cov) / cov.shape[0]) if np.isfinite(cov).all() else 1.0
    ridge = max(1e-10, 1e-6 * scale)
    for multiplier in (1.0, 10.0, 100.0, 1000.0):
        candidate = cov + ridge * multiplier * np.eye(cov.shape[0])
        try:
            np.linalg.cholesky(candidate)
            return candidate, summaries
        except np.linalg.LinAlgError:
            continue
    return np.diag(np.maximum(np.diag(cov), ridge)) + ridge * np.eye(cov.shape[0]), summaries


def posterior_stats(samples: np.ndarray, names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        values = np.asarray(samples[:, index], dtype=np.float64)
        stats[name] = {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return stats


def rbf_mmd2(x: np.ndarray, y: np.ndarray, max_points: int = 3000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_points:
        x = x[rng.choice(x.shape[0], size=max_points, replace=False)]
    if y.shape[0] > max_points:
        y = y[rng.choice(y.shape[0], size=max_points, replace=False)]
    combined = np.vstack([x, y])
    dists = pdist(combined, metric="sqeuclidean")
    bandwidth2 = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0
    bandwidth2 = max(bandwidth2, 1e-12)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aa = np.sum(a * a, axis=1)[:, None]
        bb = np.sum(b * b, axis=1)[None, :]
        sq = aa + bb - 2.0 * a @ b.T
        return np.exp(-0.5 * sq / bandwidth2)

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(
        kxx.sum() / (x.shape[0] * (x.shape[0] - 1))
        + kyy.sum() / (y.shape[0] * (y.shape[0] - 1))
        - 2.0 * kxy.mean()
    )


def bvm_diagnostics(samples: np.ndarray) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=np.float64)
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    ridge = max(1e-10, 1e-8 * float(np.trace(cov) / cov.shape[0]))
    chol = np.linalg.cholesky(cov + ridge * np.eye(cov.shape[0]))
    whitened = np.linalg.solve(chol, (samples - mean).T).T
    probs = (np.arange(samples.shape[0]) + 0.5) / samples.shape[0]
    normal_q = norm.ppf(probs)
    marginal = {}
    for index, name in enumerate(PARAM_NAMES):
        marginal[name] = {
            "qq_corr": float(np.corrcoef(np.sort(whitened[:, index]), normal_q)[0, 1]),
            "mean": float(np.mean(whitened[:, index])),
            "sd": float(np.std(whitened[:, index], ddof=1)),
        }
    radius2 = np.sum(whitened * whitened, axis=1)
    chi_q = chi2.ppf(probs, df=samples.shape[1])
    return {
        "mean": mean,
        "covariance": cov,
        "marginal": marginal,
        "radius2_qq_corr": float(np.corrcoef(np.sort(radius2), chi_q)[0, 1]),
        "ellipsoid_90_coverage": float(np.mean(radius2 <= chi2.ppf(0.90, df=samples.shape[1]))),
        "ellipsoid_95_coverage": float(np.mean(radius2 <= chi2.ppf(0.95, df=samples.shape[1]))),
    }


def save_population(
    *,
    path: Path,
    eta: np.ndarray,
    summaries: np.ndarray,
    distances: np.ndarray,
    count: int,
    n_obs: int,
    lambda_seed: int,
    tau_target: float,
    epsilon_actual: float,
    replicate: int,
) -> dict[str, Any]:
    phi = np.asarray(eta_to_phi(jnp.asarray(eta)))
    lambda_samples = sample_lambda_given_count(
        count=count,
        n_obs=n_obs,
        size=phi.shape[0],
        seed=lambda_seed,
    )
    theta = np.column_stack([lambda_samples, phi])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        theta=theta,
        eta=eta,
        phi=phi,
        summaries=summaries,
        distances=distances,
        weights=np.ones(theta.shape[0]) / theta.shape[0],
        tau_target=np.asarray(tau_target),
        epsilon_actual=np.asarray(epsilon_actual),
        replicate=np.asarray(replicate),
    )
    return {
        "path": path,
        "replicate": replicate,
        "tau_target": tau_target,
        "epsilon_actual": epsilon_actual,
        "sample_count": int(theta.shape[0]),
        "theta_stats": posterior_stats(theta, PARAM_NAMES),
        "phi_stats": posterior_stats(phi, PHI_NAMES),
        "bvm": bvm_diagnostics(theta),
    }


def run_smc_replicate(
    *,
    abc_module: Any,
    args: argparse.Namespace,
    out_dir: Path,
    replicate: int,
    count: int,
    ell_obs: np.ndarray,
    cov: np.ndarray,
) -> dict[str, Any]:
    key = rng_for(args.seed, "smc", replicate)
    key, key_particles, key_init = random.split(key, 3)
    particle_keys = random.split(key_particles, args.particles)
    particles0 = jax.vmap(prior_sample_eta)(particle_keys)
    inv_cov = np.linalg.inv(cov)
    chol_inv = jnp.asarray(np.linalg.cholesky(inv_cov))
    ell_obs_jax = jnp.asarray(ell_obs)
    count_jax = jnp.asarray(count, dtype=jnp.int32)

    def distance_fn(summary: jax.Array) -> jax.Array:
        delta = jnp.nan_to_num(summary - ell_obs_jax, nan=1e6, posinf=1e6, neginf=-1e6)
        whitened = chol_inv.T @ delta
        return jnp.linalg.norm(whitened)

    def simulate_fn(subkey: jax.Array, eta: jax.Array) -> jax.Array:
        return conditional_log_summaries(
            subkey,
            eta,
            count_jax,
            n_obs=args.n_obs,
            max_locs=args.max_locs,
        )

    def summary_fn(x: jax.Array) -> jax.Array:
        return x

    state = abc_module.init(
        key_init,
        particles0,
        args.initial_epsilon,
        distance_fn=distance_fn,
        simulate_fn=simulate_fn,
        summary_fn=summary_fn,
        initial_R=args.initial_R,
        sim_batch=args.sim_batch,
    )
    targets = sorted(args.tolerances, reverse=True)
    snapshots: dict[float, dict[str, Any]] = {}
    history: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "epsilon": float(state.epsilon),
            "max_distance": float(np.max(np.asarray(state.distances))),
            "median_distance": float(np.median(np.asarray(state.distances))),
            "acceptance_rate": None,
            "R": int(state.R),
            "num_simulations": int(args.particles),
        }
    ]

    step = jax.jit(
        lambda step_key, step_state, r_cur: abc_module.abc_step(
            step_key,
            step_state,
            r_cur,
            simulate_fn=simulate_fn,
            summary_fn=summary_fn,
            distance_fn=distance_fn,
            prior_logpdf=log_prior_eta,
            alpha=args.alpha,
            c=args.c_tuning,
            B_sim=1,
            sim_batch=args.sim_batch,
        ),
        static_argnums=(2,),
    )

    started = time.time()
    for iteration in range(1, args.max_iters + 1):
        key, step_key = random.split(key)
        state, info = step(step_key, state, int(state.R))
        jax.block_until_ready(state.particles)
        epsilon = float(state.epsilon)
        acceptance = float(info.acceptance_rate)
        print(
            f"[rep {replicate}] iter={iteration} epsilon={epsilon:.6g} "
            f"acc={acceptance:.3f} R={int(state.R)}"
        )
        history.append(
            {
                "iteration": iteration,
                "epsilon": epsilon,
                "max_distance": float(np.max(np.asarray(state.distances))),
                "median_distance": float(np.median(np.asarray(state.distances))),
                "acceptance_rate": acceptance,
                "R": int(state.R),
                "num_simulations": int(info.num_simulations),
            }
        )
        for tau in targets:
            if tau not in snapshots and epsilon <= tau:
                filename = f"particles_tau_{tau:g}_rep_{replicate}.npz"
                snapshots[tau] = save_population(
                    path=out_dir / filename,
                    eta=np.asarray(state.particles),
                    summaries=np.asarray(state.summaries),
                    distances=np.asarray(state.distances),
                    count=count,
                    n_obs=args.n_obs,
                    lambda_seed=stable_int(args.seed, "lambda", replicate, tau),
                    tau_target=tau,
                    epsilon_actual=epsilon,
                    replicate=replicate,
                )
        if epsilon <= min(targets):
            break
        if acceptance < args.acc_min and iteration >= args.min_iters:
            break

    return {
        "replicate": replicate,
        "history": history,
        "snapshots": snapshots,
        "elapsed_seconds": time.time() - started,
        "final_epsilon": float(state.epsilon),
        "final_R": int(state.R),
        "completed_smallest_tolerance": min(targets) in snapshots,
    }


def tolerance_stability(snapshot_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rep: dict[int, list[dict[str, Any]]] = {}
    for record in snapshot_records:
        by_rep.setdefault(int(record["replicate"]), []).append(record)
    rows: list[dict[str, Any]] = []
    for rep, records in by_rep.items():
        records = sorted(records, key=lambda row: float(row["tau_target"]), reverse=True)
        for left, right in zip(records[:-1], records[1:]):
            row: dict[str, Any] = {
                "replicate": rep,
                "tau_from": left["tau_target"],
                "tau_to": right["tau_target"],
            }
            for name in PARAM_NAMES:
                row[f"{name}_mean_delta"] = (
                    right["theta_stats"][name]["mean"] - left["theta_stats"][name]["mean"]
                )
                row[f"{name}_sd_delta"] = (
                    right["theta_stats"][name]["sd"] - left["theta_stats"][name]["sd"]
                )
            rows.append(row)
    return rows


def replicate_metrics(snapshot_records: list[dict[str, Any]], final_tau: float) -> list[dict[str, Any]]:
    final_records = [
        record for record in snapshot_records if float(record["tau_target"]) == float(final_tau)
    ]
    rows: list[dict[str, Any]] = []
    loaded = []
    for record in final_records:
        with np.load(record["path"]) as data:
            loaded.append((record, np.asarray(data["theta"], dtype=np.float64)))
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            left_record, left = loaded[i]
            right_record, right = loaded[j]
            max_kl_points = min(argsafe_int("KL_MAX_POINTS", 3000), left.shape[0], right.shape[0])
            rng = np.random.default_rng(stable_int("replicate-kl", i, j))
            left_eval = left
            right_eval = right
            if left.shape[0] > max_kl_points:
                left_eval = left[rng.choice(left.shape[0], size=max_kl_points, replace=False)]
            if right.shape[0] > max_kl_points:
                right_eval = right[rng.choice(right.shape[0], size=max_kl_points, replace=False)]
            rows.append(
                {
                    "tau": final_tau,
                    "replicate_i": left_record["replicate"],
                    "replicate_j": right_record["replicate"],
                    "mmd2": rbf_mmd2(left_eval, right_eval, seed=stable_int(i, j, "mmd")),
                    "kl_i_to_j": float(kullback_leibler(left_eval, right_eval)),
                    "kl_j_to_i": float(kullback_leibler(right_eval, left_eval)),
                    "points": int(max_kl_points),
                }
            )
    return rows


def argsafe_int(env_name: str, default: int) -> int:
    try:
        return int(os.environ.get(env_name, default))
    except ValueError:
        return default


def write_readme(path: Path, payload: dict[str, Any]) -> None:
    diagnostics = payload["diagnostics"]
    lines = [
        "# Stereological Count-Factorised Blackjax SMC-ABC",
        "",
        f"Created: `{payload['created_at']}`",
        "",
        "## Design",
        "",
        "- Conditioned on the observed total inclusion count K.",
        "- Ran SMC-ABC only for `(sigma, xi)` on the three log summaries.",
        "- Drew `lambda | K` from the truncated Gamma posterior on `[30, 200]`.",
        "- Used Ryan's local `blackjax.smc.abc` implementation, not ELFI.",
        "",
        "## Cell",
        "",
        f"- n_obs: `{payload['n_obs']}`",
        f"- seed: `{payload['seed']}`",
        f"- true theta: `{payload['true_theta']}`",
        f"- observed summary: `{payload['observed_summary']}`",
        f"- K: `{payload['count']}`",
        "",
        "## Diagnostics",
        "",
        f"- tolerance stability rows: `{len(diagnostics['tolerance_stability'])}`",
        f"- final replicate metric rows: `{len(diagnostics['replicate_metrics'])}`",
        f"- BSL cross-check: `not run by this script`",
        "",
        "See `metadata.json`, `diagnostics.json`, and `summary.csv` for numeric details.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> Path:
    created_at = args.created_at or utc_now()
    if args.output_dir is None:
        output_label = ""
        if args.output_label:
            output_label = "_" + "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in args.output_label
            )
        out_dir = (
            REPO_ROOT
            / "res"
            / "stereological_blackjax_smc_abc"
            / f"n_obs_{args.n_obs}_seed_{args.seed}_{compact_stamp(created_at)}{output_label}"
        )
    else:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    abc_module = import_blackjax_abc(args.blackjax_root)
    s_obs = observed_summary(args.seed, args.n_obs)
    count = int(round(args.n_obs * float(s_obs[0])))
    ell_obs = np.asarray(s_obs[1:4], dtype=np.float64)
    print(f"observed summary={s_obs} K={count}")

    pilot_phi = np.asarray(args.pilot_phi, dtype=np.float64)
    cov, pilot_summaries = estimate_summary_covariance(
        seed=args.seed,
        phi_center=pilot_phi,
        count=count,
        n_obs=args.n_obs,
        max_locs=args.max_locs,
        num_sims=args.pilot_sims,
        batch_size=args.pilot_batch_size,
    )
    np.save(out_dir / "observed_summary.npy", s_obs)
    np.save(out_dir / "summary_covariance.npy", cov)
    np.save(out_dir / "pilot_summaries.npy", pilot_summaries)

    replicate_records = []
    snapshot_records: list[dict[str, Any]] = []
    for replicate in range(args.replicate_offset, args.replicate_offset + args.replicates):
        result = run_smc_replicate(
            abc_module=abc_module,
            args=args,
            out_dir=out_dir,
            replicate=replicate,
            count=count,
            ell_obs=ell_obs,
            cov=cov,
        )
        replicate_records.append(result)
        snapshot_records.extend(result["snapshots"].values())

    stability = tolerance_stability(snapshot_records)
    final_tau = min(args.tolerances)
    rep_metrics = replicate_metrics(snapshot_records, final_tau)
    summary_rows = []
    for record in snapshot_records:
        base = {
            "replicate": record["replicate"],
            "tau_target": record["tau_target"],
            "epsilon_actual": record["epsilon_actual"],
            "sample_count": record["sample_count"],
            "path": record["path"],
        }
        for name in PARAM_NAMES:
            for key, value in record["theta_stats"][name].items():
                base[f"{name}_{key}"] = value
        summary_rows.append(base)
    write_csv(
        out_dir / "summary.csv",
        [
            "replicate",
            "tau_target",
            "epsilon_actual",
            "sample_count",
            "path",
            *[
                f"{name}_{stat}"
                for name in PARAM_NAMES
                for stat in ("mean", "sd", "q05", "q50", "q95")
            ],
        ],
        summary_rows,
    )

    metadata = {
        "created_at": created_at,
        "git": git_record(),
        "environment": environment_record(),
        "blackjax_module": getattr(abc_module, "__file__", None),
        "n_obs": args.n_obs,
        "seed": args.seed,
        "true_theta": np.asarray(TRUE_THETA),
        "observed_summary": s_obs,
        "count": count,
        "ell_obs": ell_obs,
        "pilot_phi": pilot_phi,
        "summary_covariance": cov,
        "args": vars(args),
    }
    diagnostics = {
        "replicates": replicate_records,
        "snapshots": snapshot_records,
        "tolerance_stability": stability,
        "replicate_metrics": rep_metrics,
    }
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "diagnostics.json", diagnostics)
    write_readme(
        out_dir / "README.md",
        {
            **metadata,
            "diagnostics": diagnostics,
        },
    )
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--particles", type=int, default=50_000)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--replicate-offset",
        type=int,
        default=0,
        help="First replicate index; use this to split replicates across jobs.",
    )
    parser.add_argument(
        "--tolerances",
        type=lambda text: [float(part) for part in text.split(",") if part],
        default=[20.0, 10.0, 5.0, 2.0, 1.0],
    )
    parser.add_argument("--pilot-sims", type=int, default=20_000)
    parser.add_argument("--pilot-batch-size", type=int, default=64)
    parser.add_argument(
        "--pilot-phi",
        type=lambda text: np.asarray([float(part) for part in text.split(",")], dtype=float),
        default=np.asarray([2.0, -0.1], dtype=float),
        help="Pilot centre as sigma,xi.",
    )
    parser.add_argument("--max-locs", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--c-tuning", type=float, default=0.01)
    parser.add_argument("--initial-R", type=int, default=1)
    parser.add_argument("--initial-epsilon", type=float, default=1.0e9)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--min-iters", type=int, default=2)
    parser.add_argument("--acc-min", type=float, default=0.05)
    parser.add_argument(
        "--sim-batch",
        type=int,
        default=2000,
        help="Particle batch size for chunked BlackJAX simulator vmaps.",
    )
    parser.add_argument("--blackjax-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-label")
    parser.add_argument("--created-at")
    return parser


def main() -> None:
    out_dir = run(build_parser().parse_args())
    print(out_dir)


if __name__ == "__main__":
    main()
