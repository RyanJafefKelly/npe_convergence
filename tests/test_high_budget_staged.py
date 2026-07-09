from __future__ import annotations

import json
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "npe_convergence" / "scripts" / "run_high_budget_staged.py"


def load_staged_module():
    spec = importlib.util.spec_from_file_location("run_high_budget_staged_for_tests", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_cmd(*args: str) -> None:
    subprocess.run([sys.executable, str(SCRIPT), *args], check=True)


def shard_array(root: Path, shard: int = 0) -> dict[str, np.ndarray]:
    path = (
        root
        / "gnk_flow_npe_n_obs_8_n_sims_6_seed_0"
        / "sim_shards"
        / f"sim_shard_{shard:05d}.npz"
    )
    data = np.load(path)
    return {key: data[key] for key in ("theta_bounded", "theta_unbounded", "summaries")}


def test_gnk_simulation_shard_is_deterministic(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    common = [
        "simulate-shard",
        "--model=gnk",
        "--method=flow_npe",
        "--seed=0",
        "--n-obs=8",
        "--n-sims=6",
        "--shard-index=0",
        "--shard-size=4",
        "--sim-batch-size=2",
    ]
    run_cmd(f"--staging-root={root_a}", *common)
    run_cmd(f"--staging-root={root_b}", *common)

    first = shard_array(root_a)
    second = shard_array(root_b)
    for key in first:
        np.testing.assert_allclose(first[key], second[key])


def test_aggregate_simulation_shards_writes_training_data(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    base_args = [
        "--model=gnk",
        "--method=flow_npe",
        "--seed=0",
        "--n-obs=8",
        "--n-sims=6",
    ]
    for shard in (0, 1):
        run_cmd(
            f"--staging-root={root}",
            "simulate-shard",
            *base_args,
            f"--shard-index={shard}",
            "--shard-size=3",
            "--sim-batch-size=3",
        )
    run_cmd(f"--staging-root={root}", "aggregate-sims", *base_args, "--expected-shards=2")

    run_root = root / "gnk_flow_npe_n_obs_8_n_sims_6_seed_0"
    data = np.load(run_root / "training_data.npz")
    assert data["theta_train"].shape == (6, 4)
    assert data["summary_train"].shape == (6, 7)
    assert data["x_obs_std"].shape == (7,)

    metadata = json.loads((run_root / "training_data.json").read_text())
    assert metadata["finite_rows"] == 6
    assert metadata["shard_count"] == 2


def test_standardised_gnk_samples_are_destandardised_before_bounding() -> None:
    staged = load_staged_module()
    theta_bounded = np.array(
        [
            [3.0, 1.0, 2.0, 0.5],
            [3.1, 1.2, 1.8, 0.7],
        ],
        dtype=np.float32,
    )
    theta_unbounded = np.asarray(staged.gnk_to_unbounded(theta_bounded))
    theta_mean = np.array([0.5, -0.25, 0.1, 0.2], dtype=np.float32)
    theta_std = np.array([2.0, 0.5, 1.5, 0.75], dtype=np.float32)
    theta_standardised = (theta_unbounded - theta_mean) / theta_std

    recovered = np.asarray(
        staged.transform_standardised_to_bounded("gnk", theta_standardised, theta_mean, theta_std)
    )

    np.testing.assert_allclose(recovered, theta_bounded, rtol=1e-6, atol=1e-6)


def test_flow_training_state_skips_unserializable_opt_state(tmp_path: Path) -> None:
    staged = load_staged_module()
    original_save_equinox_tree = staged.save_equinox_tree

    def fake_save_equinox_tree(path: Path, payload: object) -> None:
        if Path(path).name.endswith("_opt_state.eqx"):
            raise TypeError("synthetic opt_state serialization failure")

    state = {
        "model": object(),
        "best_model": object(),
        "opt_state": object(),
        "epoch": 1,
        "best_epoch": 0,
        "best_val_loss": 1.25,
        "wait": 0,
        "losses": {"train": [1.0], "val": [1.25], "epoch_seconds": [0.5]},
    }

    staged.save_equinox_tree = fake_save_equinox_tree
    try:
        staged._save_training_state(tmp_path, "flow_npe", state)
    finally:
        staged.save_equinox_tree = original_save_equinox_tree

    with (tmp_path / "training" / "flow_npe_state.pkl").open("rb") as f:
        saved_state = pickle.load(f)
    assert "opt_state" not in saved_state
    assert "opt_state" not in saved_state["tree_checkpoints"]

    diagnostics = [
        json.loads(line)
        for line in (tmp_path / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert diagnostics[-1]["event"] == "opt-state-save-skipped"
    assert diagnostics[-1]["format"] == "equinox"
