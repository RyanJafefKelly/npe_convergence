import argparse

import numpyro  # type: ignore

from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_gnk import run_gnk


def run_gnk_experiments(args):
    seed = args.seed
    _ = run_experiment(run_gnk, seed)
    return None


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    parser = argparse.ArgumentParser(
        prog="run_gnk_experiments.py",
        description="Run experiments for gnk model.",
        epilog="Example usage: python run_gnk_experiments.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_gnk_experiments(args)
