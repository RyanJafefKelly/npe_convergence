from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "npe_convergence" / "scripts" / "run_gnk_dim_scaling.py"


def load_dim_module():
    spec = importlib.util.spec_from_file_location("run_gnk_dim_scaling_for_tests", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_fake_nuts_cache(dim, root: Path, *, d_s: int, seed: int, n_obs: int, n_samples: int = 32) -> None:
    cache = dim.nuts_cache_path(root, d_s, n_obs, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + d_s)
    samples = rng.normal(
        loc=np.array([3.0, 1.0, 2.0, 0.5]),
        scale=np.array([0.08, 0.05, 0.08, 0.03]),
        size=(n_samples, 4),
    )
    with cache.open("wb") as f:
        pickle.dump(
            {
                "samples": samples,
                "chains": samples.reshape(1, n_samples, 4),
                "param_names": list(dim.PARAM_NAMES),
            },
            f,
        )
    x_obs = dim.observed_summary(seed, n_obs, d_s)
    np.save(dim.nuts_observed_path(cache), x_obs)
    dim.write_json(
        dim.nuts_metadata_path(cache),
        {
            "status": "complete",
            "d_s": d_s,
            "d_theta": 4,
            "d": d_s + 4,
            "n_obs": n_obs,
            "seed": seed,
            "obs_hash": dim.hash_array(x_obs),
            "quantile_probabilities": dim.quantile_probabilities(d_s).tolist(),
            "param_names": list(dim.PARAM_NAMES),
            "diagnostics": {
                name: {"r_hat": 1.0, "ess_bulk": 2000.0, "ess_tail": 2000.0}
                for name in dim.PARAM_NAMES
            },
            "diagnostic_pass": True,
        },
    )


def test_summary_dims() -> None:
    dim = load_dim_module()
    assert [dim.d_total(d_s) for d_s in dim.D_S_GRID] == [9, 11, 15, 19, 23]
    assert [dim.scaled_n_sims(d_s, 500) for d_s in dim.D_S_GRID] == [
        202500,
        302500,
        562500,
        902500,
        1322500,
    ]


def test_tiny_nuts_cache_metadata(tmp_path: Path) -> None:
    dim = load_dim_module()
    cache = dim.precompute_nuts(
        output_root=tmp_path,
        d_s=5,
        seed=0,
        n_obs=20,
        num_samples=8,
        num_warmup=8,
        num_chains=1,
        thinning=1,
        max_rhat=10.0,
        min_ess=0.0,
        skip_diagnostic_gate=True,
        force=False,
        progress_bar=False,
    )
    metadata = dim.load_json(dim.nuts_metadata_path(cache))
    assert metadata["obs_hash"] == dim.hash_array(dim.observed_summary(0, 20, 5))
    assert len(metadata["quantile_probabilities"]) == 5
    for name in dim.PARAM_NAMES:
        assert {"r_hat", "ess_bulk", "ess_tail"} <= set(metadata["diagnostics"][name])
    with cache.open("rb") as f:
        payload = pickle.load(f)
    assert payload["samples"].shape == (8, 4)


def test_runner_smoke(tmp_path: Path) -> None:
    dim = load_dim_module()
    for d_s in (5, 7):
        write_fake_nuts_cache(dim, tmp_path, d_s=d_s, seed=0, n_obs=30)
        out = dim.run_cell(
            output_root=tmp_path,
            d_s=d_s,
            method="gaussian_npe",
            seed=0,
            n_obs=30,
            n_sims=40,
            sim_batch_size=10,
            num_posterior_samples=24,
            metric_samples=12,
            learning_rate=5e-4,
            train_batch_size=16,
            max_epochs=1,
            patience=1,
            val_frac=0.2,
            max_rhat=1.05,
            min_ess=1000.0,
            skip_nuts_diagnostic_gate=False,
            force=False,
        )
        for name in ["kl.txt", "mmd.txt", "diagnostics.json", "posterior_samples.npz"]:
            assert (out / name).exists(), name
        assert np.isfinite(float((out / "kl.txt").read_text()))
        assert np.isfinite(float((out / "mmd.txt").read_text()))


def test_aggregator_smoke(tmp_path: Path) -> None:
    dim = load_dim_module()
    complete = tmp_path / "gnk_gaussian_npe_d_s_5_n_obs_30_n_sims_40_seed_0"
    failed = tmp_path / "gnk_flow_npe_d_s_5_n_obs_30_n_sims_40_seed_0"
    complete.mkdir()
    failed.mkdir()
    (complete / "kl.txt").write_text("1.25\n")
    (complete / "mmd.txt").write_text("0.05\n")
    dim.write_json(complete / "diagnostics.json", {"status": "complete", "obs_hash": "abc"})
    (failed / "kl.txt").write_text("nan\n")
    (failed / "mmd.txt").write_text("0.1\n")
    dim.write_json(failed / "diagnostics.json", {"status": "failed", "failure_reason": "synthetic"})

    outputs = dim.aggregate_results(
        output_root=tmp_path,
        plot_dir=tmp_path / "plots",
        timestamp="2026-05-26T00:00:00Z",
    )
    per_seed = outputs["per_seed_csv"].read_text()
    summary = outputs["summary_csv"].read_text()
    assert "finite_metric" in per_seed
    assert "kl_median" in summary
    assert outputs["plot_metadata"].exists()
