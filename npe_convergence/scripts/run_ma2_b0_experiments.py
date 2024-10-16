import argparse

from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_ma2_b0 import run_ma2_b0


def run_ma2_experiments(args):
    seed = args.seed
    _ = run_experiment(run_ma2_b0, seed)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_ma2_experiments.py",
        description="Run experiments for MA(2) model.",
        epilog="Example usage: python run_ma2_experiments.py"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    run_ma2_experiments(args)
