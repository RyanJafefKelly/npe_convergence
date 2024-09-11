import numpyro  # type: ignore
import argparse
from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_stereological import run_stereological


def run_stereological_experiments(args):
    seed = args.seed
    _ = run_experiment(run_stereological, seed)
    return None


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    parser = argparse.ArgumentParser(
        prog="run_stereological_experiments.py",
        description="Run experiments for stereological model.",
        epilog="Example usage: python run_stereological_experiments.py"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_stereological_experiments(args)
