from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_mak import run_mak


def run_mak_experiments():
    kl_mat = run_experiment(run_mak)
    return None


if __name__ == "__main__":
    run_mak_experiments()
