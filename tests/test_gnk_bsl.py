from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import numpy as np

from npe_convergence.examples.gnk import gnk, ss_octile


SCRIPT = Path(__file__).resolve().parents[1] / "npe_convergence" / "scripts" / "run_gnk_bsl.py"


def load_bsl_module():
    spec = importlib.util.spec_from_file_location("run_gnk_bsl_for_tests", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_observed_summary_matches_gnk_gaussian_cache_convention() -> None:
    bsl = load_bsl_module()
    n_obs = 1000
    seed = 0
    true_params = jnp.array([3.0, 1.0, 2.0, 0.5])

    key = random.key(seed)
    _, subkey = random.split(key)
    z = random.normal(subkey, shape=(n_obs,))
    x_obs = gnk(z, *true_params)
    x_obs = jnp.atleast_2d(x_obs)
    expected = jnp.squeeze(ss_octile(x_obs))

    np.testing.assert_allclose(bsl.observed_summary(seed, n_obs), np.asarray(expected))


def test_smoke_bsl_run(tmp_path: Path) -> None:
    bsl = load_bsl_module()
    args = bsl.build_parser().parse_args(
        [
            "--n-obs=30",
            "--seed=0",
            "--M=20",
            "--num-chains=2",
            "--num-warmup=50",
            "--num-samples=50",
            f"--output-root={tmp_path}",
            "--initial-scale=0.05",
            "--progress-every=0",
            "--created-at=2026-05-26T000000Z",
        ]
    )

    output_dir = bsl.run(args)

    expected_files = [
        "posterior_samples.npz",
        "posterior_samples_unbounded.npz",
        "observed_summary.npy",
        "proposal_covariance.npy",
        "diagnostics.json",
        "metadata.json",
        "traceplots.png",
        "MISSING_NUTS_REFERENCE.txt",
    ]
    for name in expected_files:
        assert (output_dir / name).exists(), name

    theta = np.load(output_dir / "posterior_samples.npz")["theta"]
    eta = np.load(output_dir / "posterior_samples_unbounded.npz")["eta"]
    assert theta.shape == (100, 4)
    assert eta.shape == (100, 4)
    assert np.isfinite(theta).all()
    assert np.isfinite(eta).all()
