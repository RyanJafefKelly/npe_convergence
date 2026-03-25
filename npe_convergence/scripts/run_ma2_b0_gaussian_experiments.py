import argparse

import numpyro  # type: ignore

from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_ma2_b0_gaussian import run_ma2_b0_gaussian


def run_ma2_b0_gaussian_experiments(args):
    seed = args.seed
    _ = run_experiment(run_ma2_b0_gaussian, seed)
    return None


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    parser = argparse.ArgumentParser(
        prog="run_ma2_b0_gaussian_experiments.py",
        description="Run experiments for MA(2) b0 model with Gaussian NPE.",
        epilog="Example usage: python run_ma2_b0_gaussian_experiments.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_ma2_b0_gaussian_experiments(args)
