from npe_convergence.scripts.run_experiment import run_experiment
from npe_convergence.scripts.run_svar import run_svar


def run_svar_experiments():
    kl_mat = run_experiment(run_svar)
    return None


if __name__ == "__main__":
    run_svar_experiments()