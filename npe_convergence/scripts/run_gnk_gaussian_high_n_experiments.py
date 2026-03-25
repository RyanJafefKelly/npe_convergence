import argparse

import numpyro  # type: ignore

from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_gnk_gaussian import run_gnk_gaussian


def run_gnk_gaussian_high_n_experiments(args):
    seed = args.seed
    _ = run_experiment(run_gnk_gaussian, seed, n_obs=[10000, 20000])
    return None


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    parser = argparse.ArgumentParser(
        prog="run_gnk_gaussian_high_n_experiments.py",
        description="Run high-n experiments for gnk model (Gaussian NPE).",
        epilog="Example usage: python run_gnk_gaussian_high_n_experiments.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_gnk_gaussian_high_n_experiments(args)
