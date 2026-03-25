import argparse

from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_stereological_gaussian import run_stereological_gaussian


def run_stereological_gaussian_experiments(args):
    seed = args.seed
    _ = run_experiment(run_stereological_gaussian, seed)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_stereological_gaussian_experiments.py",
        description="Run experiments for stereological model with Gaussian NPE.",
        epilog="Example usage: python run_stereological_gaussian_experiments.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_stereological_gaussian_experiments(args)
