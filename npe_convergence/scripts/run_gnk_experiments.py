from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_gnk import run_gnk


def run_gnk_experiments():
    kl_mat = run_experiment(run_gnk)
    return None


if __name__ == "__main__":
    run_gnk_experiments()
